"""OpenAI GPT-Live 1 provider over WebSocket (server-side).

GPT-Live (``gpt-live-1``, the Live API at ``wss://api.openai.com/v1/live/sessions``)
is a full-duplex voice model: it listens while it speaks, decides on its own when
to answer, stops when the user talks over it, and streams one continuous 24 kHz
PCM16 track back, silence included. That is a different animal from the Realtime
API, and the mapping onto the provider seam is:

    set_instructions    Layer 1  session.instructions.append. Instructions are immutable
                                 after start, so only the changed room note is appended.
    set_turn_detection  Layer 2  GPT-Live has no VAD knobs. The eager/patient profile
                                 becomes one appended turn-taking instruction.
    nudge               Layer 3  session.commentary.append: content the model says aloud.
    interrupt                    There is no cancel event, and measured live the model
                                 finishes its sentence before it speaks a commentary
                                 (3 to 4 s). So the local playback buffer is flushed and
                                 the model's output is held (dropped) until it pauses
                                 and starts the nudge: the listener hears the agent stop
                                 at once and the nudge from its first word.
    request_response             session.thinking.append asking it to open the call.
    check_audio_quality          client delegation: session.delegation.created ->
                                 on_tool_call -> commentary with that delegation_id.

Because the audio track never stops, "the agent is speaking" is read off the
audio itself: a per-chunk RMS gate with a short hangover marks speech segments,
and only those chunks reach ``audio_out``. Measured live, silence between the
model's utterances is RMS < 0.001 and speech is > 0.01.

Verified against the API on 2026-09-22: session.start with ``audio.format``,
``delegation.type = client``, ``*.append`` with ``delegation_id: null``, and the
server events below. Output audio deltas carry no timing fields on OpenAI's
endpoint; transcript deltas do (``start_ms`` / ``end_ms``).
"""

from __future__ import annotations

import asyncio
import base64
import json
import threading
import time
from typing import Callable

import numpy as np

from .jev import keywords
from .provider import Handlers, VoiceProvider

LIVE_URL = "wss://api.openai.com/v1/live/sessions"
SAMPLE_RATE = 24000
DEFAULT_MODEL = "gpt-live-1"
DEFAULT_VOICE = "marin"

SPEECH_RMS = 0.01  # RMS (full scale 1.0) above which a chunk counts as speech
SPEECH_HANGOVER_S = 0.5  # silence this long ends a speech segment
HOLD_RELEASE_GAP_S = 0.45  # a pause this long, then speech, is the nudge starting
HOLD_MAX_S = 6.0  # never hold the model's audio back longer than this
NUDGE_TIMEOUT_S = 10.0  # the model never spoke the nudge: release the state machine
TRANSCRIPT_IDLE_S = 1.2  # no transcript fragment for this long = the line is final
TICK_S = 0.1

# Appended to the caller's instructions at session start: how this transport works.
# Measured live: the model will happily talk *about* an appended note ("I'll make
# sure to pause for the background noise") unless told that notes are silent.
LIVE_ADDENDUM = (
    "\n\nHow this call works. Two kinds of messages reach you besides the user's voice. "
    "Audio notes appended to your instructions come from Tyto, the audio monitor. They are "
    "silent updates about the user's room: never read them out, mention them, thank anyone for "
    "them, or change the subject because of one. Let them shape how you listen and reply, and "
    "carry on with whatever the user was doing. Commentary lines also come from Tyto: say that "
    "line to the user in your own words right away, then wait for the user's reply. "
    "When the user asks how they sound, whether you can hear them, or about their audio or "
    "connection, do not guess: delegate to the backend, which answers with the live Tyto "
    "reading, and read that back to them."
)

# Every appended instruction ends with this so the model applies it without a word.
SILENT = " Silent update: do not say anything about this, carry on."

TURN_NOTES = {
    "patient": (
        "Audio note: the user's background is noisy right now. Wait for a clear pause before "
        "you answer, do not react to faint voices or sounds in the background, and be tolerant "
        "of misheard words." + SILENT
    ),
    "eager": "Audio note: the user's background is quiet again. Normal, snappy turn-taking is fine." + SILENT,
}
CLEAN_NOTE = "Audio note: the user's audio is clean again, so the previous audio note no longer applies." + SILENT
OPEN_NOTE = "The user just connected and is listening. Open the conversation now, as your instructions say."


def _rms(pcm16: bytes) -> float:
    if len(pcm16) < 2:
        return 0.0
    x = np.frombuffer(pcm16[: len(pcm16) // 2 * 2], dtype="<i2").astype(np.float32) / 32768.0
    return float(np.sqrt(np.mean(x * x)))


class OpenAILiveProvider(VoiceProvider):
    def __init__(
        self,
        handlers: Handlers,
        *,
        api_key: str,
        instructions: str,
        audio_out: Callable[[bytes], None],
        audio_done: Callable[[], None] | None = None,
        audio_flush: Callable[[], None] | None = None,
        model: str = DEFAULT_MODEL,
        voice: str = DEFAULT_VOICE,
        on_log=None,
        clock: Callable[[], float] | None = None,
    ):
        super().__init__(handlers)
        self._api_key = api_key
        self._base_instructions = instructions
        self._audio_out = audio_out
        self._audio_done = audio_done
        self._audio_flush = audio_flush
        self.model = model
        self._voice = voice
        self._on_log = on_log
        self._clock = clock or time.monotonic

        self._loop: asyncio.AbstractEventLoop | None = None
        self._ws = None
        self._thread: threading.Thread | None = None
        self.closed = threading.Event()

        self._mic_enabled = True
        self._turn_profile: str | None = None

        # Speech segmentation of the continuous output track.
        self._speaking = False
        self._segment_nudge = False
        self._last_voiced = 0.0

        # Interrupt: hold the model's output until it pauses and starts the nudge.
        self._hold = False
        self._hold_since = 0.0
        self._hold_gap_seen = False
        self._hold_silence_since: float | None = None

        self._pending_nudge = False
        self._nudge_sent_at = 0.0
        self._nudge_words: set[str] = set()

        # Transcript fragments -> lines (GPT-Live has no "final" marker).
        self._agent_line, self._agent_tx_at = "", 0.0
        self._user_line, self._user_tx_at = "", 0.0

    # -- lifecycle ---------------------------------------------------------- #

    def connect(self) -> None:
        self._thread = threading.Thread(target=self._run, name="openai-live", daemon=True)
        self._thread.start()

    def disconnect(self) -> None:
        if self._loop and self._ws and not self._loop.is_closed():
            asyncio.run_coroutine_threadsafe(self._close(), self._loop)
        self.closed.set()

    def _run(self) -> None:
        error = None
        try:
            asyncio.run(self._main())
        except Exception as err:  # noqa: BLE001
            error = str(err)
            self._log("error", error)
        finally:
            expected = self.closed.is_set()  # disconnect() sets it first
            self.closed.set()
            if not expected:
                self._closed_unexpectedly(error)

    async def _main(self) -> None:
        from websockets.asyncio.client import connect

        self._loop = asyncio.get_running_loop()
        headers = [("Authorization", f"Bearer {self._api_key}")]
        async with connect(LIVE_URL, additional_headers=headers, max_size=None) as ws:
            self._ws = ws
            await ws.send(json.dumps(self.session_start()))
            self._log("session.start", self.model)
            watchdog = asyncio.create_task(self._watchdog())
            try:
                async for raw in ws:
                    self._receive(json.loads(raw))
            finally:
                watchdog.cancel()

    async def _watchdog(self) -> None:
        while True:
            await asyncio.sleep(TICK_S)
            self._tick(self._clock())

    async def _close(self) -> None:
        try:
            await self._ws.send(json.dumps({"type": "session.close"}))
            await asyncio.sleep(0.5)  # let session.closed arrive (final usage)
        finally:
            await self._ws.close()

    def session_start(self) -> dict:
        """The strict startup config. Unknown fields are rejected by the API."""
        return {
            "type": "session.start",
            "session": {
                "model": self.model,
                "instructions": self._base_instructions + LIVE_ADDENDUM,
                "audio": {"format": {"type": "audio/pcm", "rate": SAMPLE_RATE}, "output": {"voice": self._voice}},
                "delegation": {"type": "client"},
            },
        }

    # -- commands (app -> provider) ----------------------------------------- #

    def set_instructions(self, text: str) -> None:
        # The controller sends BASE + room note; instructions are immutable here,
        # so append only the note (or say the room is clean again).
        note = text[len(self._base_instructions):] if text.startswith(self._base_instructions) else text
        note = note.strip()
        self._append("instructions", note + SILENT if note else CLEAN_NOTE)

    def set_turn_detection(self, turn_detection: dict | None) -> None:
        if turn_detection is None:
            return  # the listen gate: nothing to gate on a full-duplex model
        profile = "patient" if turn_detection.get("type") == "server_vad" else "eager"
        if profile == self._turn_profile:
            return
        self._turn_profile = profile
        self._append("instructions", TURN_NOTES[profile])
        self._log("live.turn_note", profile)

    def set_mic_enabled(self, on: bool) -> None:
        self._mic_enabled = on

    def interrupt(self, clear_input: bool = False) -> None:
        now = self._clock()
        if self._audio_flush:
            self._audio_flush()
        self._pending_nudge = False
        was_speaking = self._speaking
        if was_speaking:
            self._end_segment(cancelled=True)
        # Hold the model's output. If it is silent right now the pause condition
        # is already met and the next speech is let through from its first chunk.
        self._hold = True
        self._hold_since = now
        self._hold_gap_seen = not was_speaking
        self._hold_silence_since = None if was_speaking else now
        self._log("live.hold", "on" if was_speaking else "armed")

    def nudge(self, text: str) -> None:
        self._pending_nudge = True
        self._nudge_sent_at = self._clock()
        self._nudge_words = keywords(text)
        self._append("commentary", text)

    def request_response(self) -> None:
        self._append("thinking", OPEN_NOTE)

    def send_tool_result(self, call_id: str, output: dict) -> None:
        content = output.get("summary") or output.get("status") or json.dumps(output)
        if "tyto_score" in output:
            content += f" The Tyto risk score is {output['tyto_score']:.2f}."
        self._append("commentary", content, delegation_id=call_id)

    def send_audio(self, mono: np.ndarray) -> None:
        """Forward one block of mono float32 mic audio. Muted = silence, so the track stays continuous."""
        if not self._ws:
            return
        if self._mic_enabled:
            pcm16 = (np.clip(mono, -1.0, 1.0) * 32767).astype("<i2").tobytes()
        else:
            pcm16 = bytes(len(mono) * 2)
        self._send({"type": "session.input_audio.append", "audio": base64.b64encode(pcm16).decode("ascii")})

    # -- events (provider -> app) ------------------------------------------- #

    def _receive(self, msg: dict) -> None:
        t = msg.get("type", "")
        if t == "session.output_audio.delta":
            self._on_audio(base64.b64decode(msg.get("delta", "")))
        elif t == "session.output_transcript.delta":
            self._on_agent_tx(msg.get("delta", ""))
        elif t == "session.input_transcript.delta":
            self._on_user_tx(msg.get("delta", ""))
        elif t == "session.started":
            self._log("session.started", (msg.get("session") or {}).get("id", ""))
            if self.h.on_ready:
                self.h.on_ready()
        elif t == "session.delegation.created":
            d = msg.get("delegation") or {}
            if d.get("target", "client") == "client" and self.h.on_tool_call:
                self.h.on_tool_call("check_audio_quality", d.get("id", ""))
        elif t == "error":
            err = msg.get("error") or {}
            self._log("error", f"{err.get('code', '')} {err.get('message', '')}".strip())
        elif t == "session.closed":
            reason = msg.get("reason", "")
            self._log("session.closed", f"{reason} {json.dumps(msg.get('usage') or {})}")
            if not self.closed.is_set():  # the server ended it, not disconnect()
                self._closed_unexpectedly(reason or None)
            self.closed.set()  # so _run's finally does not report it a second time

    def _on_audio(self, pcm16: bytes) -> None:
        now = self._clock()
        voiced = _rms(pcm16) >= SPEECH_RMS
        if self._hold:
            self._track_hold(voiced, now)
            if self._hold:
                return  # dropped: the tail of what the agent was saying
        if voiced:
            self._last_voiced = now
            if not self._speaking:
                self._start_segment()
            self._audio_out(pcm16)
        elif self._speaking:
            self._audio_out(pcm16)  # a short pause inside a sentence: keep playback continuous

    def _track_hold(self, voiced: bool, now: float) -> None:
        if not voiced:
            if self._hold_silence_since is None:
                self._hold_silence_since = now
            if now - self._hold_silence_since >= HOLD_RELEASE_GAP_S:
                self._hold_gap_seen = True
            return
        self._hold_silence_since = None
        if self._hold_gap_seen:
            self._release_hold("pause")
        elif now - self._hold_since >= HOLD_MAX_S:
            self._release_hold("timeout")

    def _release_hold(self, why: str) -> None:
        self._hold = False
        self._log("live.hold", f"released ({why})")

    def _start_segment(self) -> None:
        self._speaking = True
        self._segment_nudge = self._pending_nudge
        self._pending_nudge = False
        if self.h.on_agent_speaking:
            self.h.on_agent_speaking(True, nudge=self._segment_nudge)

    def _end_segment(self, cancelled: bool = False) -> None:
        self._speaking = False
        nudge, self._segment_nudge = self._segment_nudge, False
        if not cancelled and self._audio_done:
            self._audio_done()
        if self.h.on_agent_speaking:
            self.h.on_agent_speaking(False, nudge=nudge, cancelled=cancelled)

    def _tick(self, now: float) -> None:
        """Timers: segment end, hold cap, missing nudge, transcript line ends."""
        if self._speaking and now - self._last_voiced >= SPEECH_HANGOVER_S:
            self._end_segment()
        if self._hold and now - self._hold_since >= HOLD_MAX_S:
            self._release_hold("timeout")
        if self._pending_nudge and now - self._nudge_sent_at >= NUDGE_TIMEOUT_S:
            # The model never spoke it. Run the lifecycle so the controller resumes.
            self._pending_nudge = False
            self._log("live.nudge", "no audio for the nudge, releasing")
            if self.h.on_agent_speaking:
                self.h.on_agent_speaking(True, nudge=True)
                self.h.on_agent_speaking(False, nudge=True, cancelled=True)
        if self._agent_line and now - self._agent_tx_at >= TRANSCRIPT_IDLE_S:
            self._flush_agent_tx()
        if self._user_line and now - self._user_tx_at >= TRANSCRIPT_IDLE_S:
            self._flush_user_tx()

    # -- transcripts -------------------------------------------------------- #

    def _on_agent_tx(self, delta: str) -> None:
        if self._hold:
            # Words the listener never hears are not shown. Unless they are the
            # nudge itself and the pause detector missed the switch.
            if self._nudge_words & keywords(delta):
                self._release_hold("transcript")
            else:
                return
        self._agent_line += delta
        self._agent_tx_at = self._clock()
        if self.h.on_agent_transcript:
            self.h.on_agent_transcript(delta, False)
        if delta.rstrip().endswith((".", "?", "!")):
            self._flush_agent_tx()

    def _flush_agent_tx(self) -> None:
        line, self._agent_line = self._agent_line.strip(), ""
        if line and self.h.on_agent_transcript:
            self.h.on_agent_transcript(line, True)

    def _on_user_tx(self, delta: str) -> None:
        self._user_line += delta
        self._user_tx_at = self._clock()
        if self.h.on_user_transcript:
            self.h.on_user_transcript(delta, False)
        if delta.rstrip().endswith((".", "?", "!")):
            self._flush_user_tx()

    def _flush_user_tx(self) -> None:
        line, self._user_line = self._user_line.strip(), ""
        if line and self.h.on_user_transcript:
            self.h.on_user_transcript(line, True)

    # -- helpers ------------------------------------------------------------ #

    def _append(self, kind: str, content: str, delegation_id: str | None = None) -> None:
        self._send({"type": f"session.{kind}.append", "delegation_id": delegation_id, "content": content})

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
