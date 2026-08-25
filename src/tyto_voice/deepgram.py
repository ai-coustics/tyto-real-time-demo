"""Deepgram: the voice of the cascade, and the caption under the user's turn.

``DeepgramTTS`` holds one persistent ``/v1/speak`` websocket. Keeping the socket
open matters: time-to-first-audio measured 0.24 s on a warm socket against
0.60 s for a fresh REST request, and that difference is a third of the demo's
whole response latency. The connection lives on its own asyncio loop on a
background thread, the same shape the OpenAI provider used, and hands audio out
through callbacks so it can drive a local speaker or a browser.

Note that ``/v2/speak`` is rejected with HTTP 400; the aura-2 voices are served
from ``/v1/speak``.

``transcribe`` is a caption, and nothing more. Inkling hears the user's audio
directly, so this is not how the agent understands anything: it runs on its own
thread, never gates a reply, and never reaches the prompt. It exists so the UI
can show roughly what was said. Because it stays out of history, a slow or wrong
transcript can only make the panel wrong, never the conversation.
"""

from __future__ import annotations

import asyncio
import json
import threading
import urllib.parse
import urllib.request
from typing import Callable

SPEAK_URL = "wss://api.deepgram.com/v1/speak"
LISTEN_URL = "https://api.deepgram.com/v1/listen"
# Deepgram returns raw linear16 at whatever rate we ask for. 24 kHz is the
# playback rate for both frontends; capture runs at 16 kHz, which is separate.
PLAYBACK_RATE = 24000
# Male aura-2 voice. Other male options: zeus, apollo, atlas, draco.
DEFAULT_VOICE = "aura-2-orion-en"

# Words the demo hears constantly that a general model mishears. Without the
# keyterm, "Tyto" comes back as "Taito".
KEYTERMS = ("Tyto", "ai-coustics")
LISTEN_TIMEOUT = 15.0
# The endpoint can reject urllib's default agent outright; see inkling.py.
USER_AGENT = "tyto-voice/0.1 (+https://github.com/ai-coustics)"


class DeepgramTTS:
    """A persistent speak socket.

    Commands are safe to call from any thread; they are marshalled onto the
    socket's own event loop.

        speak(text)   synthesize one line, then flush
        clear()       abandon whatever is queued or playing
        close()

    Callbacks fire on the socket thread:
        audio_out(pcm16)   one chunk of agent audio, mono PCM16 at PLAYBACK_RATE
        on_started()       first audio of a line is on its way
        on_finished()      the line is fully generated (Deepgram sent Flushed)

    ``clear`` is a pure command and never fires ``on_finished``; the caller that
    interrupted knows it did so. Deepgram numbers each Flush it answers, and a
    line abandoned by ``clear`` may still have a ``Flushed`` in flight, so those
    are matched by sequence id and ignored. Without that, the tail of a cancelled
    line ends the line that replaced it.
    """

    def __init__(
        self,
        api_key: str,
        *,
        audio_out: Callable[[bytes], None],
        on_started: Callable[[], None] | None = None,
        on_finished: Callable[[], None] | None = None,
        voice: str = DEFAULT_VOICE,
        sample_rate: int = PLAYBACK_RATE,
        on_log: Callable[[str, str], None] | None = None,
    ):
        self._api_key = api_key
        self._audio_out = audio_out
        self._on_started = on_started
        self._on_finished = on_finished
        self._voice = voice
        self._sample_rate = sample_rate
        self._on_log = on_log

        self._loop: asyncio.AbstractEventLoop | None = None
        self._ws = None
        self._thread: threading.Thread | None = None
        self.ready = threading.Event()
        self.closed = threading.Event()

        # Guards the bookkeeping below, written by the socket thread and read by
        # whichever thread is driving a turn.
        self._lock = threading.Lock()
        self._speaking = False
        self._announced = False  # has this line reported its first audio chunk
        self._sent = 0           # Flush messages sent, so far
        self._expect = 0         # sequence id of the first Flushed we still want

    # -- lifecycle ----------------------------------------------------------- #

    def connect(self) -> None:
        self._thread = threading.Thread(target=self._run, name="deepgram-tts", daemon=True)
        self._thread.start()

    def close(self) -> None:
        if self._loop and self._ws:
            asyncio.run_coroutine_threadsafe(self._ws.close(), self._loop)
        self.closed.set()

    def _run(self) -> None:
        try:
            asyncio.run(self._main())
        except Exception as err:  # noqa: BLE001 - surfaced to the UI
            self._log("error", f"deepgram tts: {err}")
        finally:
            self.closed.set()
            self.ready.set()  # never leave a waiter blocked on a dead socket
            # The socket died, so the Flushed for a line in progress will never
            # arrive. Report it finished or the caller stays muted for good.
            with self._lock:
                orphaned = self._speaking
                self._speaking = False
                self._announced = False
            if orphaned and self._on_finished:
                self._on_finished()

    async def _main(self) -> None:
        from websockets.asyncio.client import connect

        self._loop = asyncio.get_running_loop()
        query = urllib.parse.urlencode({
            "model": self._voice,
            "encoding": "linear16",
            "sample_rate": self._sample_rate,
        })
        headers = [("Authorization", f"Token {self._api_key}")]
        async with connect(f"{SPEAK_URL}?{query}", additional_headers=headers, max_size=None) as ws:
            self._ws = ws
            self.ready.set()
            self._log("tts.open", self._voice)
            async for raw in ws:
                self._receive(raw)

    # -- commands ------------------------------------------------------------ #

    def speak(self, text: str) -> bool:
        """Queue one line and flush it.

        False means nothing was sent and no ``on_finished`` will follow, so the
        caller must release whatever it was holding. Returning True on a dead
        socket would leave the agent permanently mid-sentence.
        """
        text = (text or "").strip()
        if not text or self._ws is None or self.closed.is_set():
            return False
        with self._lock:
            self._speaking = True
            self._announced = False
            self._sent += 1
        self._send({"type": "Speak", "text": text})
        self._send({"type": "Flush"})
        return True

    def clear(self) -> None:
        """Drop queued and in-flight audio. Used by the Reactive layer."""
        with self._lock:
            self._speaking = False
            self._announced = False
            # Everything flushed up to now is abandoned; ignore its Flushed.
            self._expect = self._sent
        self._send({"type": "Clear"})

    @property
    def speaking(self) -> bool:
        return self._speaking

    # -- socket events ------------------------------------------------------- #

    def _receive(self, raw) -> None:
        if isinstance(raw, (bytes, bytearray)):
            with self._lock:
                if not self._speaking:
                    return  # tail of a cleared line, drop it
                started = not self._announced
                self._announced = True
            if started and self._on_started:
                self._on_started()
            self._audio_out(bytes(raw))
            return

        message = json.loads(raw)
        kind = message.get("type")
        if kind == "Flushed":
            with self._lock:
                stale = message.get("sequence_id", 0) < self._expect
                if stale or not self._speaking:
                    return
                self._speaking = False
                self._announced = False
            if self._on_finished:
                self._on_finished()
        elif kind == "Warning":
            self._log("error", f"deepgram: {message.get('description', '')}")

    # -- internals ----------------------------------------------------------- #

    def _send(self, obj: dict) -> None:
        if not self._loop or not self._ws or self._loop.is_closed():
            return
        try:
            asyncio.run_coroutine_threadsafe(self._ws.send(json.dumps(obj)), self._loop)
        except RuntimeError:
            self.closed.set()

    def _log(self, kind: str, text: str) -> None:
        if self._on_log:
            self._on_log(kind, text)


def transcribe(api_key: str, wav_bytes: bytes, *, model: str = "nova-3") -> str:
    """Blocking speech-to-text for the UI caption. Returns "" on failure.

    This never gates a reply and never reaches the model's prompt, so every
    error is swallowed: a missing caption should leave a panel line unwritten,
    not interrupt the conversation.
    """
    query = [("model", model), ("smart_format", "true")]
    query += [("keyterm", term) for term in KEYTERMS]
    request = urllib.request.Request(
        f"{LISTEN_URL}?{urllib.parse.urlencode(query)}",
        data=wav_bytes,
        headers={
            "Authorization": f"Token {api_key}",
            "Content-Type": "audio/wav",
            "User-Agent": USER_AGENT,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=LISTEN_TIMEOUT) as response:
            payload = json.loads(response.read())
        return payload["results"]["channels"][0]["alternatives"][0]["transcript"].strip()
    except Exception:  # noqa: BLE001 - best effort by design
        return ""
