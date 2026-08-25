"""The cascaded voice provider: ai-coustics VAD -> Inkling-Small -> Deepgram TTS.

This is the backend behind the ``VoiceProvider`` seam, replacing the single
speech-to-speech session the OpenAI Realtime provider opened. Everything the
control layers need is still expressed through the same nine methods, so
``TytoController`` and ``decision.py`` did not have to learn anything new.

    mic ──> LiveVad.feed() ──(utterance)──┬─> Inkling-Small ──> Deepgram TTS ──> audio_out
                                          └─> Deepgram STT (caption for the UI)

The user's audio goes to Inkling whole. Inkling is multimodal, so it hears the
utterance directly and that is the only path that decides anything.

The speech-to-text branch is a caption and nothing more. It runs on its own
thread, never gates a reply, and its text never reaches the prompt, so the UI can
show roughly what was said without a wrong or slow transcript being able to
affect the conversation. Keeping it out of history also means there is no
question of which turn a late transcript belongs to.

Threads:
    caller's audio thread   send_audio -> VAD, and dispatches a turn
    "cascade-turn"          one per turn: Inkling, then hands text to TTS
    "cascade-stt"           one per turn: the caption, off the critical path
    "deepgram-tts"          the speak socket's asyncio loop, emits agent audio

Where the layers land:
    Layer 1 Aware      set_instructions -> the system message of the next turn,
                       plus a live Tyto reading attached to every turn so the
                       agent can say how the user sounds without a tool call
    Layer 2 Tuned      set_turn_detection -> VAD sensitivity and hold, and the
                       listen gate (None stops listening)
    Layer 3 Reactive   interrupt -> Deepgram Clear plus a playback flush,
                       nudge -> the line goes straight to TTS

One deliberate difference from the Realtime backend: a nudge costs no model round
trip. The text is a fixed string from ``decision.py``, so it is spoken directly
and then recorded in history so the agent knows it said it. That makes the
Reactive layer the fastest part of the demo rather than the slowest.

Barge-in is optional and off everywhere by default, including the browser.

It is not a small switch. Leaving the microphone open while the agent talks
closes an acoustic loop unless echo cancellation is genuinely removing our own
voice, and browser ``getUserMedia`` cancellation does not reliably cover Web
Audio playback. What survives is enough to trip the VAD, and then: the agent's
own voice is heard as speech, barge-in cuts the reply off, the echo finishes as
an "utterance", Inkling answers the agent's own words, and it goes round again.
The symptom is the agent repeating itself over and over.

Turn it on only where you know the microphone cannot hear the speaker, which in
practice means headphones.
"""

from __future__ import annotations

import threading
from typing import Callable

import numpy as np

from .decision import NO_READING, READING_MAX_AGE_SECONDS, VAD_PROFILES, live_reading
from .deepgram import DEFAULT_VOICE, DeepgramTTS, transcribe
from .inkling import InklingClient, encode_wav
from .provider import Handlers, VoiceProvider
from .vad import LiveVad

# The whole capture chain: the VAD model, Tyto and Inkling are all 16 kHz
# native, so nothing resamples between the microphone and the model.
SAMPLE_RATE = 16000
# Agent audio comes back at Deepgram's rate, which is separate on purpose.
PLAYBACK_RATE = 24000

# How long to wait for the controller to answer a tool call. It answers
# synchronously today; the timeout only stops a different controller from
# wedging a turn forever.
TOOL_TIMEOUT = 5.0

# Continuous speech required before barge-in believes the user really is talking
# over the agent. The VAD's own minimum_speech_duration is 60 ms, which is right
# for starting a turn in silence but far too twitchy against residual echo of
# our own voice. A third of a second is about one syllable.
BARGE_IN_MIN_SPEECH = 0.35


class CascadeProvider(VoiceProvider):
    def __init__(
        self,
        handlers: Handlers,
        *,
        license_key: str,
        inkling_key: str,
        deepgram_key: str,
        instructions: str,
        audio_out,
        audio_done=None,
        audio_flush=None,
        greeting: str | None = None,
        voice: str = DEFAULT_VOICE,
        turn_detection: dict | None = None,
        tools: list | None = None,
        models_dir: str = "./models",
        transcribe_user: bool = True,
        scores: Callable[[], object] | None = None,
        allow_barge_in: bool = False,
        on_log=None,
    ):
        super().__init__(handlers)
        self._deepgram_key = deepgram_key
        self._transcribe_user = transcribe_user
        self._audio_done = audio_done
        self._audio_flush = audio_flush
        self._greeting = greeting
        # Returns (Scores, age_in_seconds), or None while Tyto is warming up.
        # Read fresh at the top of every turn, never cached.
        self._scores = scores
        self._allow_barge_in = allow_barge_in
        self._on_log = on_log

        self.vad = LiveVad(
            license_key,
            sample_rate=SAMPLE_RATE,
            profile=turn_detection or VAD_PROFILES["eager"],
            models_dir=models_dir,
        )
        self.llm = InklingClient(
            inkling_key,
            instructions=instructions,
            tools=self._as_openai_tools(tools),
            on_log=on_log,
        )
        self.tts = DeepgramTTS(
            deepgram_key,
            audio_out=audio_out,
            on_started=self._on_tts_started,
            on_finished=self._on_tts_finished,
            voice=voice,
            sample_rate=PLAYBACK_RATE,
            on_log=on_log,
        )

        self.closed = self.tts.closed

        self._lock = threading.RLock()
        self._mic_enabled = True
        self._busy = False       # a turn or a nudge owns the agent right now
        self._nudge = False      # ...and it is a nudge
        self._turn_id = 0        # bumped to abandon whatever is in flight

        self._tool_result: dict | None = None
        self._tool_call_id: str | None = None
        self._tool_ready = threading.Event()

    # -- lifecycle ----------------------------------------------------------- #

    def connect(self) -> None:
        self.vad.start()  # downloads the VAD model (cached) and checks the licence
        self.tts.connect()
        self.tts.ready.wait(timeout=10.0)
        if self.h.on_ready:
            self.h.on_ready()

    def disconnect(self) -> None:
        with self._lock:
            self._turn_id += 1
            self._busy = False
        self.tts.close()
        self.vad.stop()

    @property
    def sample_rate(self) -> int:
        """Capture rate. Feed :meth:`send_audio` audio at this rate."""
        return self.vad.sample_rate or SAMPLE_RATE

    @property
    def block_size(self) -> int:
        return self.vad.block_size

    # -- commands (app -> provider) ------------------------------------------ #

    def set_instructions(self, text: str) -> None:
        self.llm.set_instructions(text)

    def set_turn_detection(self, turn_detection: dict | None) -> None:
        self.vad.set_profile(turn_detection)

    def set_mic_enabled(self, on: bool) -> None:
        was_muted = not self._mic_enabled
        with self._lock:
            self._mic_enabled = on
        if not on and not self._allow_barge_in:
            self.vad.reset()  # do not carry half an utterance across the gap
        elif on and was_muted:
            # Unmuting after the agent spoke. Whatever the VAD was holding was
            # collected while our own voice was in the room, so it is echo, not
            # the user. Start the next turn from silence.
            self.vad.reset()

    def interrupt(self, clear_input: bool = False) -> None:
        with self._lock:
            # Invalidate the turn in flight. Without this a worker still inside
            # llm.respond would come back and speak its reply with no turn
            # owning it, and its Flushed would end whatever replaced it.
            self._turn_id += 1
        self.tts.clear()
        if self._audio_flush:
            self._audio_flush()
        if clear_input:
            self.vad.reset()
        self._end_turn(cancelled=True)

    def nudge(self, text: str) -> None:
        """Layer 3. Spoken directly, with no model round trip."""
        turn_id = self._begin_turn(nudge=True)
        self.llm.add_agent_line(text)
        if self.h.on_agent_transcript:
            self.h.on_agent_transcript(text, True)
        if not self._speak_if_current(turn_id, text):
            self._end_turn()

    def request_response(self) -> None:
        """Open the conversation. A fixed greeting, so the demo makes a sound
        the moment it is ready instead of after a model round trip."""
        if not self._greeting:
            return
        turn_id = self._begin_turn(nudge=False)
        if turn_id is None:
            return
        self.llm.add_agent_line(self._greeting)
        if self.h.on_agent_transcript:
            self.h.on_agent_transcript(self._greeting, True)
        if not self._speak_if_current(turn_id, self._greeting):
            self._end_turn()

    def send_tool_result(self, call_id: str, output: dict) -> None:
        with self._lock:
            if call_id != self._tool_call_id:
                return  # a late answer to a call this turn no longer cares about
            self._tool_result = output
        self._tool_ready.set()

    # -- audio in (mic thread) ----------------------------------------------- #

    def send_audio(self, mono: np.ndarray) -> None:
        """Feed one block of mono float32 mic audio at :attr:`sample_rate`."""
        with self._lock:
            busy = self._busy
            enabled = self._mic_enabled
            barge_in = self._allow_barge_in
        if not enabled and not barge_in:
            return
        if busy and not barge_in:
            return

        utterance = self.vad.feed(mono)

        if busy:
            # The agent is mid-turn. Cut it off only once there is a real run of
            # speech over the top, not on the first frame the VAD likes.
            if self.vad.speech_samples >= BARGE_IN_MIN_SPEECH * self.sample_rate:
                self._log("turn.barge_in", "")
                self.interrupt()
            return  # whatever the VAD returned belongs to the turn we just cut

        if utterance is not None:
            self._dispatch(utterance)

    # -- a turn --------------------------------------------------------------- #

    def _dispatch(self, utterance: np.ndarray) -> None:
        turn_id = self._begin_turn(nudge=False)
        if turn_id is None:
            return
        if self._transcribe_user:
            threading.Thread(
                target=self._run_caption,
                args=(encode_wav(utterance, self.sample_rate),),
                name="cascade-stt",
                daemon=True,
            ).start()
        threading.Thread(
            target=self._run_turn, args=(utterance, turn_id), name="cascade-turn", daemon=True
        ).start()

    def _run_turn(self, utterance: np.ndarray, turn_id: int) -> None:
        reply = None
        try:
            reply = self.llm.respond(
                utterance,
                sample_rate=self.sample_rate,
                context=self._reading(),
                tool_handler=self._call_tool,
                cancelled=lambda: self._stale(turn_id),
            )
        except Exception as err:  # noqa: BLE001 - a turn must never kill the thread
            self._log("error", f"turn: {err}")

        if self._stale(turn_id):
            return  # a nudge took over; it owns the lifecycle now
        if reply is None:
            # No reply and nothing to speak. Release the agent explicitly or the
            # mic stays muted and the demo goes silent for good.
            self._end_turn()
            return
        if self.h.on_agent_transcript:
            self.h.on_agent_transcript(reply.text, True)
        if not self._speak_if_current(turn_id, reply.text):
            # Either the turn was taken over between the check above and here,
            # or the voice socket is gone. Only the second case is ours to end.
            if not self._stale(turn_id):
                self._end_turn()

    def _reading(self) -> str | None:
        """The Tyto reading for this turn's prompt, or an honest admission that
        there is not one.

        Carrying it inline is what lets "how do I sound?" be answered in one
        round trip. Asking the model to call a tool for the same numbers cost a
        second call to Inkling, which measured about 1.2 s on top of the reply.

        The age check is the important part. Tyto is reset every time the agent
        speaks and needs a fresh 5 s window, so a run of short turns produces no
        new reading at all and the last one can be minutes old. Passing that off
        as current is what makes the agent insist the room is still noisy long
        after the noise has stopped, so a stale reading is withheld. The age
        never reaches the prompt text: it decides whether the agent may talk
        about the room at all, and the user never hears about it.
        """
        if self._scores is None:
            return None
        current = self._scores()
        if current is None:
            return NO_READING
        scores, age = current
        if age is None or age > READING_MAX_AGE_SECONDS:
            return NO_READING
        return live_reading(scores)

    def reset_conversation(self) -> None:
        """Forget the conversation so far. The next turn starts clean."""
        self.llm.reset()
        self._log("turn.reset", "conversation cleared")

    def _run_caption(self, wav: bytes) -> None:
        """Fill in the UI's "You" panel. Display only, never the prompt.

        If the transcript does not come back, the line is simply not written.
        There is nothing useful to say in its place.
        """
        text = transcribe(self._deepgram_key, wav)
        if text and self.h.on_user_transcript:
            self.h.on_user_transcript(text, True)

    def _call_tool(self, name: str, call_id: str) -> dict:
        """Bridge the seam's async-shaped tool contract into this worker."""
        with self._lock:
            self._tool_result = None
            self._tool_call_id = call_id
        self._tool_ready.clear()
        if self.h.on_tool_call:
            self.h.on_tool_call(name, call_id)
        self._tool_ready.wait(timeout=TOOL_TIMEOUT)
        with self._lock:
            result = self._tool_result
            self._tool_call_id = None
        return result or {"status": "unavailable"}

    # -- agent speech bookkeeping -------------------------------------------- #

    def _begin_turn(self, *, nudge: bool) -> int | None:
        """Claim the agent. Returns the new turn id, or None if already busy.

        A nudge always wins: the controller has already interrupted whatever was
        running before it calls us.
        """
        with self._lock:
            if self._busy and not nudge:
                return None
            self._turn_id += 1
            self._busy = True
            self._nudge = nudge
            turn_id = self._turn_id
        if self.h.on_agent_speaking:
            # Reported at the start of the turn, not when audio starts, so the
            # mic is muted and Tyto scoring is paused while the model thinks.
            self.h.on_agent_speaking(True, nudge=nudge)
        return turn_id

    def _end_turn(self, cancelled: bool = False) -> None:
        with self._lock:
            if not self._busy:
                return
            self._busy = False
            nudge = self._nudge
            self._nudge = False
        if self._audio_done:
            self._audio_done()
        if self.h.on_agent_speaking:
            self.h.on_agent_speaking(False, nudge=nudge, cancelled=cancelled)

    def _stale(self, turn_id: int) -> bool:
        with self._lock:
            return turn_id != self._turn_id

    def _speak_if_current(self, turn_id: int, text: str) -> bool:
        """Speak only if this turn still owns the agent.

        Checking and speaking has to be atomic: an interrupt landing between the
        two would put the abandoned reply on air over the nudge that replaced it.
        The TTS lock is always taken inside this one, never the other way round.
        """
        with self._lock:
            if turn_id != self._turn_id:
                return False
            return self.tts.speak(text)

    def _on_tts_started(self) -> None:
        self._log("tts.speaking", "")

    def _on_tts_finished(self) -> None:
        self._end_turn()

    # -- helpers -------------------------------------------------------------- #

    @staticmethod
    def _as_openai_tools(tools: list | None) -> list:
        """Accept the repo's flat tool dicts and emit chat-completions shape.

        The Realtime API takes ``{"type": "function", "name": ..., "parameters":
        ...}``; chat completions nests that under a ``function`` key. Converting
        here keeps ``CHECK_AUDIO_QUALITY_TOOL`` as the single definition.
        """
        wrapped = []
        for tool in tools or []:
            if "function" in tool:
                wrapped.append(tool)
                continue
            wrapped.append({
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
                },
            })
        return wrapped

    def _log(self, kind: str, text: str) -> None:
        if self._on_log:
            self._on_log(kind, text)
