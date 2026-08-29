"""The cascaded voice provider: Deepgram Flux -> PhoneLLM on Modal -> Aura-2.

This is the backend behind the ``VoiceProvider`` seam, replacing the single
speech-to-speech session the OpenAI Realtime provider opened. Everything the
control layers need is still expressed through the same nine methods, so
``TytoController`` and ``decision.py`` did not have to learn anything new.

    mic ──> FluxSTT ──(turn + transcript)──> PhoneLLM ──> DeepgramTTS ──> audio_out

Three services, one per job, which is the trade a cascade makes: more moving
parts than speech-to-speech, in exchange for being able to see and cancel every
stage. The Reactive layer is the reason that matters here. When Tyto trips, this
provider can throw away the user's half-finished sentence *and* stop a reply
mid-word, because both are things it owns rather than state inside somebody
else's session.

Speculation is what keeps it fast. Flux emits ``EagerEndOfTurn`` when it thinks
the user is probably done, before it is sure, and guarantees the transcript will
match the eventual ``EndOfTurn`` if the user really has stopped. So the PhoneLLM
request is fired on the guess: when it holds, the reply is already in hand at
``EndOfTurn`` and goes straight to the voice, and the model's latency disappears
into the turn gap entirely. When the user carries on talking, ``TurnResumed``
throws the speculation away and it cost nothing anybody heard.

Threads:
    caller's audio thread   send_audio -> the Flux socket
    "deepgram-flux"         the listen socket's asyncio loop, emits turn events
    "cascade-spec"          one per speculation: an early PhoneLLM request
    "cascade-turn"          one per turn: waits on the reply, hands it to TTS
    "deepgram-tts"          the speak socket's asyncio loop, emits agent audio

Where the layers land:
    Layer 1 Aware      set_instructions -> the system message of the next turn
    Layer 2 Tuned      set_turn_detection -> Flux end-of-turn thresholds, live
                       over a Configure message, and the listen gate (None stops
                       forwarding audio)
    Layer 3 Reactive   interrupt -> cancel the speculation, Clear the voice,
                       flush playback, void the user's turn;
                       nudge -> the line goes straight to TTS

One deliberate difference from the Realtime backend: a nudge costs no model round
trip. The text is a fixed string from ``decision.py``, so it is spoken directly
and then recorded in history so the agent knows it said it. That makes the
Reactive layer the fastest part of the demo rather than the slowest.

Barge-in is on wherever the microphone is captured with echo cancellation, and
off otherwise.

It is not a small switch. Leaving the microphone open while the agent talks
closes an acoustic loop unless echo cancellation is genuinely removing our own
voice. What survives is enough for Flux to hear as a turn, and then: the agent's
own voice starts a turn, barge-in cuts the reply off, the echo is transcribed,
PhoneLLM answers the agent's own words, and it goes round again. The symptom is
the agent talking to itself. Turn it on only where the microphone cannot hear
the speaker.
"""

from __future__ import annotations

import threading

import numpy as np

from .decision import VAD_PROFILES
from .deepgram import DEFAULT_VOICE, PLAYBACK_RATE, DeepgramTTS
from .flux import SAMPLE_RATE, FluxSTT
from .phonellm import PhoneLLMClient
from .provider import Handlers, VoiceProvider

# How long to wait for the controller to answer a tool call. It answers
# synchronously today; the timeout only stops a different controller from
# wedging a turn forever.
TOOL_TIMEOUT = 5.0

# How long a committed turn will wait for a speculation that is still in flight
# before giving up and asking again from scratch. Generous, because waiting is
# almost always faster than a second round trip: this is a guard against a hung
# request, not a latency budget.
SPECULATION_TIMEOUT = 20.0


class _Speculation:
    """One PhoneLLM request fired on Flux's guess that the turn is over."""

    def __init__(self, turn_index: int, transcript: str):
        self.turn_index = turn_index
        self.transcript = transcript
        self.done = threading.Event()
        self.reply = None
        self.cancelled = False


class CascadeProvider(VoiceProvider):
    def __init__(
        self,
        handlers: Handlers,
        *,
        endpoint_url: str,
        modal_key: str,
        deepgram_key: str,
        instructions: str,
        audio_out,
        audio_done=None,
        audio_flush=None,
        greeting: str | None = None,
        voice: str = DEFAULT_VOICE,
        turn_detection: dict | None = None,
        tools: list | None = None,
        allow_barge_in: bool = False,
        on_log=None,
    ):
        super().__init__(handlers)
        self._audio_done = audio_done
        self._audio_flush = audio_flush
        self._greeting = greeting
        self._allow_barge_in = allow_barge_in
        self._on_log = on_log

        self.stt = FluxSTT(
            deepgram_key,
            profile=turn_detection or VAD_PROFILES["eager"],
            sample_rate=SAMPLE_RATE,
            on_start_of_turn=self._on_start_of_turn,
            on_interim=self._on_interim,
            on_eager_end_of_turn=self._on_eager_end_of_turn,
            on_turn_resumed=self._on_turn_resumed,
            on_end_of_turn=self._on_end_of_turn,
            on_log=on_log,
        )
        self.llm = PhoneLLMClient(
            endpoint_url,
            modal_key,
            instructions=instructions,
            tools=tools,
            on_log=on_log,
        )
        self.tts = DeepgramTTS(
            deepgram_key,
            audio_out=audio_out,
            on_started=lambda: self._log("tts.speaking", ""),
            on_finished=self._end_turn,
            voice=voice,
            sample_rate=PLAYBACK_RATE,
            on_log=on_log,
        )

        self.closed = self.tts.closed

        self._lock = threading.RLock()
        self._mic_enabled = True
        self._listening = True   # the Layer 2 gate: is the user's audio wanted
        self._busy = False       # a turn or a nudge owns the agent right now
        self._nudge = False      # ...and it is a nudge
        self._turn_id = 0        # bumped to abandon whatever is in flight
        self._spec: _Speculation | None = None

        # One waiter per outstanding tool call, so a speculation and a committed
        # turn can never take each other's answer.
        self._tool_waiters: dict[str, tuple[threading.Event, list]] = {}

    # -- lifecycle ----------------------------------------------------------- #

    def connect(self) -> None:
        self.stt.connect()
        self.tts.connect()
        self.stt.ready.wait(timeout=10.0)
        self.tts.ready.wait(timeout=10.0)
        # Modal endpoints scale to zero, and waking a 30B model takes minutes.
        # Warm it in the background: the greeting is spoken by the voice alone,
        # so the demo is audible immediately either way.
        threading.Thread(target=self._warm_up, name="cascade-warmup", daemon=True).start()
        if self.h.on_ready:
            self.h.on_ready()

    def _warm_up(self) -> None:
        if self.llm.wait_until_ready():
            self._log("llm.ready", "PhoneLLM endpoint is live")
        else:
            self._log("error", "PhoneLLM endpoint did not come up")

    def disconnect(self) -> None:
        with self._lock:
            self._turn_id += 1
            self._busy = False
        self._cancel_speculation()
        self.stt.close()
        self.tts.close()

    @property
    def sample_rate(self) -> int:
        """Capture rate. Feed :meth:`send_audio` audio at this rate."""
        return SAMPLE_RATE

    # -- commands (app -> provider) ------------------------------------------ #

    def set_instructions(self, text: str) -> None:
        self.llm.set_instructions(text)

    def set_turn_detection(self, turn_detection: dict | None) -> None:
        """Layer 2 - Tuned, and the listen gate.

        ``None`` means stop listening. Flux has no way to be told that, so we
        stop forwarding audio and void whatever turn it had open, which is the
        same thing from the user's side.
        """
        if turn_detection is None:
            with self._lock:
                self._listening = False
            self.stt.discard_turn()
            return
        with self._lock:
            self._listening = True
        self.stt.configure(turn_detection)

    def set_mic_enabled(self, on: bool) -> None:
        with self._lock:
            was_muted = not self._mic_enabled
            self._mic_enabled = on
        if not on or was_muted:
            # Either we are closing the gate, or we are reopening it after the
            # agent spoke. In both cases anything Flux is holding was collected
            # across a boundary the user did not speak over; start clean.
            self.stt.discard_turn()

    def interrupt(self, clear_input: bool = False) -> None:
        with self._lock:
            # Invalidate the turn in flight. Without this a worker still waiting
            # on PhoneLLM would come back and speak its reply with no turn owning
            # it, and its Flushed would end whatever replaced it.
            self._turn_id += 1
        self._cancel_speculation()
        self.tts.clear()
        if self._audio_flush:
            self._audio_flush()
        if clear_input:
            self.stt.discard_turn()
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
        the moment it is ready instead of after a model round trip (and while
        the Modal endpoint may still be waking up)."""
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
            waiter = self._tool_waiters.get(call_id)
        if waiter is None:
            return  # a late answer to a call nobody is waiting on any more
        event, box = waiter
        box.append(output)
        event.set()

    # -- audio in (mic thread) ----------------------------------------------- #

    def send_audio(self, mono: np.ndarray) -> None:
        """Feed one block of mono float32 mic audio at :attr:`sample_rate`."""
        with self._lock:
            if not self._listening:
                return
            if not self._mic_enabled and not self._allow_barge_in:
                return
            if self._busy and not self._allow_barge_in:
                return
        self.stt.send_audio(mono)

    # -- Flux turn events (socket thread) ------------------------------------ #

    def _on_start_of_turn(self, _index: int, _transcript: str) -> None:
        with self._lock:
            busy = self._busy
        if busy and self._allow_barge_in:
            # Flux only reports a turn once it is confident this is speech, so
            # unlike a raw voice-activity detector there is no extra hold needed
            # before believing the user really is talking over the agent.
            self._log("turn.barge_in", "")
            self.interrupt()

    def _on_interim(self, _index: int, transcript: str) -> None:
        if transcript and self.h.on_user_transcript:
            self.h.on_user_transcript(transcript, False)

    def _on_eager_end_of_turn(self, index: int, transcript: str) -> None:
        """Flux thinks the user is done. Ask PhoneLLM now, on the guess."""
        if not transcript:
            return
        with self._lock:
            if self._busy:
                return
            self._cancel_speculation_locked()
            spec = _Speculation(index, transcript)
            self._spec = spec
        self._log("turn.speculate", transcript)
        threading.Thread(
            target=self._run_speculation, args=(spec,), name="cascade-spec", daemon=True
        ).start()

    def _on_turn_resumed(self, _index: int, _transcript: str) -> None:
        """The user was not done after all. The speculation is now wrong."""
        self._cancel_speculation()
        self._log("turn.resumed", "")

    def _on_end_of_turn(self, index: int, transcript: str) -> None:
        if not transcript:
            return
        if self.h.on_user_transcript:
            self.h.on_user_transcript(transcript, True)

        with self._lock:
            spec = self._spec
            self._spec = None
            # Deepgram guarantees the eager transcript matches this one when no
            # TurnResumed intervened, so an exact match is the whole test. On a
            # mismatch the speculation answers something the user did not
            # finish saying, and is thrown away.
            usable = (
                spec is not None
                and not spec.cancelled
                and spec.turn_index == index
                and spec.transcript == transcript
            )
        if spec is not None and not usable:
            spec.cancelled = True
            spec = None

        turn_id = self._begin_turn(nudge=False)
        if turn_id is None:
            if spec is not None:
                spec.cancelled = True
            return
        threading.Thread(
            target=self._run_turn,
            args=(transcript, spec, turn_id),
            name="cascade-turn",
            daemon=True,
        ).start()

    # -- a turn --------------------------------------------------------------- #

    def _run_speculation(self, spec: _Speculation) -> None:
        try:
            spec.reply = self.llm.respond(
                spec.transcript,
                tool_handler=self._call_tool,
                cancelled=lambda: spec.cancelled,
            )
        except Exception as err:  # noqa: BLE001 - never kill the thread
            self._log("error", f"speculation: {err}")
        finally:
            spec.done.set()

    def _run_turn(self, transcript: str, spec: _Speculation | None, turn_id: int) -> None:
        reply = None
        if spec is not None:
            spec.done.wait(timeout=SPECULATION_TIMEOUT)
            reply = spec.reply
            self._log("turn.speculation_hit" if reply else "turn.speculation_empty", "")
        if reply is None and not self._stale(turn_id):
            # No speculation, or it came back empty. Ask properly.
            try:
                reply = self.llm.respond(
                    transcript,
                    tool_handler=self._call_tool,
                    cancelled=lambda: self._stale(turn_id),
                )
            except Exception as err:  # noqa: BLE001 - a turn must never kill the thread
                self._log("error", f"turn: {err}")

        if self._stale(turn_id):
            return  # a nudge or a barge-in took over; it owns the lifecycle now
        if reply is None:
            # Nothing to speak. Release the agent explicitly or the mic stays
            # muted and the demo goes silent for good.
            self._end_turn()
            return

        self.llm.commit(transcript, reply.text)
        if self.h.on_agent_transcript:
            self.h.on_agent_transcript(reply.text, True)
        if not self._speak_if_current(turn_id, reply.text):
            # Either the turn was taken over between the check above and here,
            # or the voice socket is gone. Only the second case is ours to end.
            if not self._stale(turn_id):
                self._end_turn()

    def reset_conversation(self) -> None:
        """Forget the conversation so far. The next turn starts clean."""
        self.llm.reset()
        self._log("turn.reset", "conversation cleared")

    def _cancel_speculation(self) -> None:
        with self._lock:
            self._cancel_speculation_locked()

    def _cancel_speculation_locked(self) -> None:
        if self._spec is not None:
            self._spec.cancelled = True
            self._spec = None

    def _call_tool(self, name: str, call_id: str) -> dict:
        """Bridge the seam's async-shaped tool contract into this worker."""
        event, box = threading.Event(), []
        with self._lock:
            self._tool_waiters[call_id] = (event, box)
        try:
            if self.h.on_tool_call:
                self.h.on_tool_call(name, call_id)
            event.wait(timeout=TOOL_TIMEOUT)
        finally:
            with self._lock:
                self._tool_waiters.pop(call_id, None)
        return box[0] if box else {"status": "unavailable"}

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
            # controller knows the agent has the floor while the model thinks.
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

    def _log(self, kind: str, text: str) -> None:
        if self._on_log:
            self._on_log(kind, text)
