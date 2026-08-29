"""Deepgram Flux: the ears of the cascade, and its turn-taking.

Flux is a conversational speech-to-text model that does two jobs in one socket:
it transcribes, and it decides when the user has finished talking. That matters
here, because a cascaded stack has to answer both questions before it can reply,
and doing them separately (a voice-activity detector plus a transcription pass)
costs a round trip that the user hears as dead air.

    audio ──> wss://api.deepgram.com/v2/listen ──> TurnInfo events

The events, and what this demo does with each:

    StartOfTurn      the user began speaking. Barge-in fires from this.
    Update           interim transcript. Drives the "You" caption only.
    EagerEndOfTurn   Flux thinks the turn is probably over. The reply is
                     speculated on here, before the user has actually stopped.
    TurnResumed      it was wrong, the user kept going. Throw the speculation
                     away.
    EndOfTurn        the turn is really over. If the speculation survived, the
                     reply is already in hand and goes straight to the voice.

Deepgram guarantees that the transcript in ``EagerEndOfTurn`` exactly matches
the one in the following ``EndOfTurn`` when no ``TurnResumed`` intervenes, which
is what makes speculating safe rather than merely fast: the agent never speaks a
reply to something the user did not say. See ``cascade.py`` for that machinery.

Layer 2 (Tuned) lands here as well. Flux takes its thresholds at connect time
*and* over a ``Configure`` control message, so the eager and patient profiles in
``decision.py`` are swapped live as the room changes, with no reconnect:

    eot_threshold        confidence needed to end a turn. Low is snappy and
                         sometimes cuts people off; high is patient.
    eager_eot_threshold  when to start speculating. Must be <= eot_threshold.
                         ``None`` in the patient profile: in a noisy room the
                         speculation would mostly be wrong.
    eot_timeout_ms       hard stop on a turn that never resolves.

``eager_eot_threshold: None`` is enforced on this side rather than on the wire,
and that is deliberate. Flux's valid range is 0.3 to 0.9, so there is no value
that means "off", and a Configure it rejects is not an error you can see: the
connection simply continues on the previous thresholds. Trying to disable
speculation remotely would therefore fail silently and leave the eager profile's
value in force in exactly the noisy room it was meant to be turned off for. So
the socket keeps emitting EagerEndOfTurn and the patient profile drops the
events here, where it cannot fail.

The connection lives on its own asyncio loop on a background thread and hands
events out through callbacks, the same shape as ``DeepgramTTS``.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
import urllib.parse
from typing import Callable

import numpy as np

LISTEN_URL = "wss://api.deepgram.com/v2/listen"
MODEL = "flux-general-en"
# Flux takes raw linear16. 16 kHz is also Tyto's native rate, so the whole
# capture chain is 16 kHz and nothing resamples.
#
# The audio sent here is the raw microphone signal, and it is the same signal
# Tyto scores. Nothing enhances it on the way past, deliberately: cleaning the
# audio first would leave Tyto measuring the enhancer's output rather than the
# room the user is actually in, and the room is the entire subject of the demo.
SAMPLE_RATE = 16000

# Words this demo hears constantly that a general model mishears. Without the
# keyterm, "Tyto" comes back as "Taito".
KEYTERMS = ("Tyto", "ai-coustics")

# Reconnect budget for the listen socket. See FluxSTT._run for why this exists.
RECONNECT_ATTEMPTS = 5
RECONNECT_BACKOFF_SECONDS = 1.0


class FluxSTT:
    """One Flux socket. Commands are safe to call from any thread.

        send_audio(mono)     forward one block of mic audio
        configure(profile)   Layer 2 - Tuned, applied live
        discard_turn()       void the turn in progress, drop its events
        close()

    Callbacks fire on the socket thread. All take ``(turn_index, transcript)``:

        on_start_of_turn        the user started talking
        on_interim              a partial transcript, for the caption
        on_eager_end_of_turn    speculate on this now
        on_turn_resumed         the speculation was wrong, drop it
        on_end_of_turn          the turn is final, answer it
    """

    def __init__(
        self,
        api_key: str,
        *,
        profile: dict,
        sample_rate: int = SAMPLE_RATE,
        keyterms: tuple[str, ...] = KEYTERMS,
        on_start_of_turn: Callable[[int, str], None] | None = None,
        on_interim: Callable[[int, str], None] | None = None,
        on_eager_end_of_turn: Callable[[int, str], None] | None = None,
        on_turn_resumed: Callable[[int, str], None] | None = None,
        on_end_of_turn: Callable[[int, str], None] | None = None,
        on_log: Callable[[str, str], None] | None = None,
    ):
        self._api_key = api_key
        self._profile = dict(profile)
        self._sample_rate = sample_rate
        self._keyterms = keyterms
        self._on_start_of_turn = on_start_of_turn
        self._on_interim = on_interim
        self._on_eager_end_of_turn = on_eager_end_of_turn
        self._on_turn_resumed = on_turn_resumed
        self._on_end_of_turn = on_end_of_turn
        self._on_log = on_log

        self._loop: asyncio.AbstractEventLoop | None = None
        self._ws = None
        self._thread: threading.Thread | None = None
        self.ready = threading.Event()
        self.closed = threading.Event()

        self._lock = threading.Lock()
        self._eager = bool(self._profile.get("eager_eot_threshold"))
        self._turn_index = -1
        # Turn indices whose events are to be ignored. A turn is voided when the
        # Reactive layer cuts the user off mid-sentence: Flux has not heard the
        # end of it and will still report one, and answering half a sentence the
        # user was talked over would be worse than answering nothing.
        self._void: set[int] = set()

    # -- lifecycle ----------------------------------------------------------- #

    def connect(self) -> None:
        self._thread = threading.Thread(target=self._run, name="deepgram-flux", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._send({"type": "CloseStream"})
        if self._loop and self._ws:
            asyncio.run_coroutine_threadsafe(self._ws.close(), self._loop)
        self.closed.set()

    def _run(self) -> None:
        """Hold the socket open, reconnecting if it drops.

        This retries rather than giving up, because losing this socket is the
        one failure in the demo with no symptom. Audio keeps being handed to
        :meth:`send_audio`, which quietly drops it while ``_ws`` is None, so the
        agent simply never hears anything again: no error in the room, nothing
        on screen but a line in the log, and the greeting has already played so
        it looks like a working demo that has stopped listening. A single
        "timed out during opening handshake" on connect used to end the session
        that way.
        """
        for attempt in range(RECONNECT_ATTEMPTS):
            if self.closed.is_set():
                break
            try:
                asyncio.run(self._main())
            except Exception as err:  # noqa: BLE001 - surfaced to the UI
                self._log("error", f"flux: {err}")
            self._ws = None
            if self.closed.is_set():
                break
            if attempt < RECONNECT_ATTEMPTS - 1:
                self._log("stt.reconnect", f"attempt {attempt + 2}")
                time.sleep(RECONNECT_BACKOFF_SECONDS * (attempt + 1))
        self.closed.set()
        self.ready.set()  # never leave a waiter blocked on a dead socket

    async def _main(self) -> None:
        from websockets.asyncio.client import connect

        self._loop = asyncio.get_running_loop()
        query = [
            ("model", MODEL),
            ("encoding", "linear16"),
            ("sample_rate", str(self._sample_rate)),
        ]
        query += _threshold_query(self._profile)
        query += [("keyterm", term) for term in self._keyterms]
        headers = [("Authorization", f"Token {self._api_key}")]
        url = f"{LISTEN_URL}?{urllib.parse.urlencode(query)}"
        async with connect(url, additional_headers=headers, max_size=None) as ws:
            self._ws = ws
            self.ready.set()
            self._log("stt.open", MODEL)
            async for raw in ws:
                if isinstance(raw, (bytes, bytearray)):
                    continue
                self._receive(json.loads(raw))

    # -- commands ------------------------------------------------------------ #

    def send_audio(self, mono: np.ndarray) -> None:
        """Forward one block of mono float32 audio at :attr:`sample_rate`."""
        if self._ws is None or self.closed.is_set():
            return
        pcm16 = (np.clip(mono, -1.0, 1.0) * 32767).astype("<i2").tobytes()
        self._send_bytes(pcm16)

    def configure(self, profile: dict) -> None:
        """Layer 2 - Tuned. Swap the end-of-turn thresholds without reconnecting.

        Only the two thresholds Flux can express are sent. Whether we act on
        eager events is a local switch, for the reason in the module docstring.
        """
        with self._lock:
            self._profile = dict(profile)
            self._eager = bool(profile.get("eager_eot_threshold"))
        thresholds = {
            "eot_threshold": profile["eot_threshold"],
            "eot_timeout_ms": profile["eot_timeout_ms"],
        }
        eager = profile.get("eager_eot_threshold")
        if eager:
            thresholds["eager_eot_threshold"] = eager
        self._send({"type": "Configure", "thresholds": thresholds})

    def discard_turn(self) -> None:
        """Void the turn in progress, so its transcript is never answered."""
        with self._lock:
            if self._turn_index >= 0:
                self._void.add(self._turn_index)

    @property
    def turn_index(self) -> int:
        return self._turn_index

    # -- socket events -------------------------------------------------------- #

    def _receive(self, message: dict) -> None:
        kind = message.get("type")
        if kind == "TurnInfo":
            self._turn_info(message)
        elif kind == "Connected":
            self._log("stt.ready", message.get("request_id", ""))
        elif kind == "ConfigureSuccess":
            self._log("stt.tuned", json.dumps(message.get("thresholds", {})))
        elif kind in ("ConfigureFailure", "FatalError", "Error"):
            self._log("error", f"flux: {message.get('description', kind)}")

    def _turn_info(self, message: dict) -> None:
        event = message.get("event")
        index = int(message.get("turn_index", 0))
        transcript = (message.get("transcript") or "").strip()

        with self._lock:
            if index != self._turn_index:
                # A new turn started, so nothing older can still be voided.
                self._turn_index = index
                self._void = {i for i in self._void if i >= index}
            voided = index in self._void
            if voided and event == "EndOfTurn":
                self._void.discard(index)  # the voided turn is now over
        if voided:
            return

        # Everything except the interim Update stream, which is once a word and
        # would drown the panel. Without this the only visible symptom of Flux
        # hearing nothing is the absence of a reply, which is indistinguishable
        # from every other failure in the stack.
        if event != "Update":
            self._log("stt.turn", f"{event} turn={index} conf={message.get('end_of_turn_confidence')}")

        if event == "StartOfTurn" and self._on_start_of_turn:
            self._on_start_of_turn(index, transcript)
        elif event == "Update" and self._on_interim:
            self._on_interim(index, transcript)
        elif event == "EagerEndOfTurn" and self._on_eager_end_of_turn:
            if self._eager:  # the patient profile does not speculate
                self._on_eager_end_of_turn(index, transcript)
        elif event == "TurnResumed" and self._on_turn_resumed:
            self._on_turn_resumed(index, transcript)
        elif event == "EndOfTurn" and self._on_end_of_turn:
            self._on_end_of_turn(index, transcript)

    # -- internals ------------------------------------------------------------ #

    def _send(self, obj: dict) -> None:
        self._send_raw(json.dumps(obj))

    def _send_bytes(self, data: bytes) -> None:
        self._send_raw(data)

    def _send_raw(self, payload) -> None:
        if not self._loop or not self._ws or self._loop.is_closed():
            return
        try:
            asyncio.run_coroutine_threadsafe(self._ws.send(payload), self._loop)
        except RuntimeError:
            self.closed.set()

    def _log(self, kind: str, text: str) -> None:
        if self._on_log:
            self._on_log(kind, text)


def _threshold_query(profile: dict) -> list[tuple[str, str]]:
    """The turn-detection half of the connect query string.

    ``eager_eot_threshold`` is omitted when the profile does not set one, which
    at connect time really does mean Flux never emits EagerEndOfTurn. Mid-stream
    it cannot be taken back, which is why the events are also gated locally.
    """
    query = [
        ("eot_threshold", str(profile["eot_threshold"])),
        ("eot_timeout_ms", str(profile["eot_timeout_ms"])),
    ]
    eager = profile.get("eager_eot_threshold")
    if eager:
        query.append(("eager_eot_threshold", str(eager)))
    return query
