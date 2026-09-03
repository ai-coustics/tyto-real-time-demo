"""The cascade backend: Deepgram Flux, gpt-5-mini, Deepgram Aura-2, on Pipecat.

One ``VoiceProvider`` (see [provider.py](provider.py)) driving a Pipecat
pipeline. The control layers do not change: ``TytoController`` and
``decision.py`` drive this through the same seam they drove the old
speech-to-speech backend through, and only the frames behind it are different.

The whole demo is this pipeline::

    transport.input()      browser mic, PCM16 16 kHz over WebRTC
    TytoAudioTap           <- Tyto listens here, and this is the mic gate
    VoiceFocusProcessor    optional Quail enhancement, agent-only
    stt                    Deepgram Flux: transcription AND turn detection
    aggregators.user()
    llm                    gpt-5-mini
    tts                    Deepgram Aura-2
    transport.output()     agent audio back to the browser
    aggregators.assistant()

Three services instead of one speech-to-speech session, which is the trade a
cascade makes: more moving parts, in exchange for being able to see and cancel
every stage. The Reactive layer is why that matters. When Tyto trips, this
provider can stop a reply mid-word and put its own line in the agent's mouth,
because both are frames it owns rather than state inside somebody else's session.

Where the three layers land, each one frame:

    Layer 1 Aware      LLMMessagesTransformFrame rewrites the system message
    Layer 2 Tuned      STTUpdateSettingsFrame retunes Flux's end-of-turn
                       thresholds mid-stream (Deepgram calls this a Configure)
    Layer 3 Reactive   InterruptionFrame cuts the reply, then TTSSpeakFrame
                       speaks the fixed line straight out of decision.py

Layer 3 costs no model round trip. The nudge text is a constant, so it goes
directly to the voice, and ``TTSSpeakFrame(append_to_context=True)`` records it
in the context so the agent knows it said it. That makes the Reactive layer the
fastest part of the demo rather than the slowest.

Voice Focus sits *after* the Tyto tap, and the order is the whole point: Tyto
scores the microphone as it actually is, while Flux hears whatever the switch
says. Enhance what the agent hears, measure what the microphone heard. Putting
the enhancer first would have Tyto scoring Quail's output, the meters would go
green and the Reactive layer would fall silent in a room that had not changed.

Threads. Scores arrive on the scorer's thread, and the nudge watchdog fires on a
``threading.Timer`` thread; both call provider methods. The pipeline lives on the
server's asyncio loop. Every command crosses over through ``_call_soon`` or
``_queue``, and every event comes back on the loop thread through
``TytoFrameObserver``. ``TytoController`` holds a re-entrant lock that makes the
crossing safe, and it stays fully synchronous so it can be tested without an
event loop.
"""

from __future__ import annotations

import asyncio
from typing import Callable

import numpy as np
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TranscriptionFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from .decision import VAD_PROFILES
from .provider import Handlers, VoiceProvider

# Capture rate. 16 kHz is Tyto 1.1's optimal rate and Flux's native rate, so the
# same frames feed the scorer and the transcriber with no resampling anywhere.
SAMPLE_RATE = 16000
# Playback rate for Aura-2. Independent of capture: only the browser hears it.
PLAYBACK_RATE = 24000

# Male aura-2 voice. Other options: thalia, helena, zeus, apollo, atlas, draco.
DEFAULT_VOICE = "aura-2-orion-en"
DEFAULT_MODEL = "gpt-5-mini"

# gpt-5-mini is a reasoning model, so the settings that matter are the ones that
# stop it thinking. Measured on this stack: 1.23 s mean with these against
# 2.84 s on the defaults, and on the defaults one reply in three spent its whole
# token budget reasoning and came back EMPTY, which a voice agent cannot use.
# The budget is generous for the same reason: reasoning tokens are drawn from it.
REASONING = {"reasoning_effort": "minimal", "verbosity": "low"}
MAX_COMPLETION_TOKENS = 400


class CascadeProvider(VoiceProvider):
    """A Pipecat cascade behind the nine ``VoiceProvider`` methods."""

    def __init__(
        self,
        handlers: Handlers,
        *,
        deepgram_key: str,
        openai_key: str,
        instructions: str,
        greeting: str,
        scorer,
        webrtc_connection,
        voice_focus=None,
        model: str = DEFAULT_MODEL,
        voice: str = DEFAULT_VOICE,
        turn_detection: dict | None = None,
        audio_quality_fn: Callable[[], dict] | None = None,
        on_client_message: Callable[[dict], None] | None = None,
        on_connected: Callable[[], None] | None = None,
        on_log: Callable[[str, str], None] | None = None,
    ):
        super().__init__(handlers)
        self._deepgram_key = deepgram_key
        self._openai_key = openai_key
        self._instructions = instructions
        self._greeting = greeting
        self._scorer = scorer
        self._connection = webrtc_connection
        self._voice_focus = voice_focus
        self._model = model
        self._voice = voice
        self._turn_detection = turn_detection or VAD_PROFILES["eager"]
        # Settable after construction so the session can wire the controller's
        # snapshot in once the controller exists.
        self.audio_quality_fn = audio_quality_fn
        self._on_client_message = on_client_message
        self._on_connected = on_connected
        self._on_log = on_log

        self._loop: asyncio.AbstractEventLoop | None = None
        self._tap = None
        self._vf = None
        self._stt = None
        self._llm = None
        self._tts = None
        self._context = None
        self._worker = None
        self._runner = None
        self._run_handle = None

        # Nudge bookkeeping. A nudge is a TTSSpeakFrame, so it never produces the
        # LLMFullResponse frames a normal reply does; the observer uses these to
        # report it to the controller as agent speech anyway.
        self._pending_nudge = False
        self._nudge_in_flight = False

    # -- lifecycle (called on the server event loop) ------------------------ #

    def connect(self) -> None:
        """Build the pipeline and start it on the current event loop.

        Must be called from inside the asyncio loop that owns the WebRTC
        connection, because aiortc objects are loop-bound.
        """
        self._loop = asyncio.get_event_loop()
        self._build_pipeline()
        self._run_handle = self._loop.create_task(self._runner.run(self._worker))

    def disconnect(self) -> None:
        if self._loop and self._worker:
            asyncio.run_coroutine_threadsafe(self._worker.cancel(), self._loop)

    def _build_pipeline(self) -> None:
        from pipecat.adapters.schemas.function_schema import FunctionSchema
        from pipecat.pipeline.pipeline import Pipeline
        from pipecat.pipeline.worker import PipelineParams, PipelineWorker
        from pipecat.processors.aggregators.llm_context import LLMContext
        from pipecat.processors.aggregators.llm_response_universal import (
            LLMContextAggregatorPair,
            LLMUserAggregatorParams,
        )
        from pipecat.services.deepgram.flux.stt import DeepgramFluxSTTService
        from pipecat.services.deepgram.tts import DeepgramTTSService
        from pipecat.services.openai.llm import OpenAILLMService
        from pipecat.transports.base_transport import TransportParams
        from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
        from pipecat.turns.user_turn_strategies import ExternalUserTurnStrategies
        from pipecat.workers.runner import WorkerRunner

        transport = SmallWebRTCTransport(
            webrtc_connection=self._connection,
            params=TransportParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                audio_in_sample_rate=SAMPLE_RATE,
                audio_out_sample_rate=PLAYBACK_RATE,
            ),
        )

        # Flux does its own turn detection, so the transport needs no VAD
        # analyzer: end of turn is decided by the same service that transcribes,
        # which is what Layer 2 retunes.
        self._stt = DeepgramFluxSTTService(
            api_key=self._deepgram_key,
            sample_rate=SAMPLE_RATE,
            settings=DeepgramFluxSTTService.Settings(**_flux_settings(self._turn_detection)),
        )

        tool = FunctionSchema(
            name="check_audio_quality",
            description=(
                "Get the current real-time audio quality of the user's mic input. Returns a "
                "summary, verdict, the Tyto Score, and the top current issue. Call this "
                "whenever the user asks if you can hear them, how their audio sounds, or "
                "about their connection/environment."
            ),
            properties={},
            required=[],
        )
        self._llm = OpenAILLMService(
            api_key=self._openai_key,
            settings=OpenAILLMService.Settings(
                model=self._model,
                max_completion_tokens=MAX_COMPLETION_TOKENS,
                extra=dict(REASONING),
            ),
        )

        async def _check_audio_quality(params):
            result = self.audio_quality_fn() if self.audio_quality_fn else {"status": "unavailable"}
            self._log("tool.check_audio_quality", result.get("summary", ""))
            await params.result_callback(result)

        self._llm.register_function("check_audio_quality", _check_audio_quality)

        self._tts = DeepgramTTSService(
            api_key=self._deepgram_key,
            sample_rate=PLAYBACK_RATE,
            settings=DeepgramTTSService.Settings(voice=self._voice),
        )

        self._context = LLMContext(
            messages=[{"role": "system", "content": self._instructions}],
            tools=[tool],
        )
        aggregators = LLMContextAggregatorPair(
            self._context,
            user_params=LLMUserAggregatorParams(
                # Flux decides turns, so nothing else should. Left at the
                # default, the aggregator builds a LocalSmartTurnAnalyzerV3
                # and loads an ONNX model to do the same job, badly, from
                # audio Flux has already ruled on.
                user_turn_strategies=ExternalUserTurnStrategies(enable_interruptions=True),
            ),
        )

        self._tap = TytoAudioTap(self._scorer)
        self._vf = VoiceFocusProcessor(self._voice_focus)

        # This list is the demo. Tyto sits second, one hop after the microphone,
        # so it scores exactly what the microphone heard; Voice Focus sits after
        # it, so only the agent hears the cleaned version.
        pipeline = Pipeline(
            [
                transport.input(),
                self._tap,
                self._vf,
                self._stt,
                aggregators.user(),
                self._llm,
                self._tts,
                transport.output(),
                aggregators.assistant(),
            ]
        )

        @transport.event_handler("on_client_connected")
        async def _on_client_connected(_transport, _client):
            self._log("pipecat.client", "connected")
            # The data channel is up now, so early UI state can be flushed.
            if self._on_connected:
                self._on_connected()
            if self.h.on_ready:
                self.h.on_ready()

        @transport.event_handler("on_client_disconnected")
        async def _on_client_disconnected(_transport, _client):
            self._log("pipecat.client", "disconnected")

        @transport.event_handler("on_app_message")
        async def _on_app_message(_transport, message, _sender):
            if isinstance(message, dict) and self._on_client_message:
                self._on_client_message(message)

        self._worker = PipelineWorker(
            pipeline,
            params=PipelineParams(
                audio_in_sample_rate=SAMPLE_RATE, audio_out_sample_rate=PLAYBACK_RATE
            ),
            observers=[TytoFrameObserver(self)],
            idle_timeout_secs=None,
            # The browser's data channel carries our own JSON, not RTVI. Left on,
            # RTVIProcessor consumes those messages and warns on every one.
            enable_rtvi=False,
        )
        self._runner = WorkerRunner(handle_sigint=False)

    # -- commands (controller -> provider, from the scorer thread) ---------- #

    def set_instructions(self, text: str) -> None:
        """Layer 1, Aware. Rewrite the system message, keep the history."""
        from pipecat.frames.frames import LLMMessagesTransformFrame

        def _swap(messages):
            out = list(messages)
            for i, m in enumerate(out):
                if _role_of(m) == "system":
                    out[i] = {"role": "system", "content": text}
                    return out
            return [{"role": "system", "content": text}] + out

        self._queue(LLMMessagesTransformFrame(transform=_swap, run_llm=False))

    def set_turn_detection(self, turn_detection: dict | None) -> None:
        """Layer 2, Tuned, and the listen gate.

        ``None`` is the gate: the controller uses it to stop the agent taking a
        turn while a nudge is in flight. The microphone is already closed by
        ``set_mic_enabled`` at that point, so Flux simply hears nothing, and
        there is no Configure worth sending.

        The typed ``delta`` is the supported path. Passing a plain mapping as
        ``settings`` still works but is deprecated and warns.
        """
        from pipecat.frames.frames import STTUpdateSettingsFrame
        from pipecat.services.deepgram.flux.stt import DeepgramFluxSTTService

        if not turn_detection:
            return
        delta = DeepgramFluxSTTService.Settings(**_flux_settings(turn_detection))
        self._queue(STTUpdateSettingsFrame(delta=delta))

    def set_mic_enabled(self, on: bool) -> None:
        """Open or close the microphone, for Tyto and for Flux alike.

        Done in the tap rather than at the transport so there is exactly one
        gate: when it is shut, neither the scorer nor the transcriber sees a
        sample, and the agent cannot be triggered by its own nudge.
        """
        self._call_soon(lambda: self._tap.set_enabled(on))

    def interrupt(self, clear_input: bool = False) -> None:
        """Stop whatever the agent is saying, right now.

        ``clear_input`` is part of the seam and is a no-op here: the controller
        has already shut the mic gate before it calls this, so there is no
        half-spoken user turn left inside Flux to discard.
        """
        from pipecat.frames.frames import InterruptionFrame

        self._pending_nudge = False
        self._queue(InterruptionFrame())

    def nudge(self, text: str) -> None:
        """Layer 3, Reactive. Put one fixed line in the agent's mouth.

        Straight to the voice: no LLM call, so this is the lowest-latency thing
        the demo does. ``append_to_context`` records the line as an assistant
        turn, so the agent knows it said it and does not repeat itself.
        """
        from pipecat.frames.frames import TTSSpeakFrame

        self._pending_nudge = True
        self._queue(TTSSpeakFrame(text=text, append_to_context=True))

    def request_response(self) -> None:
        """Open the conversation. Spoken directly, so it lands immediately."""
        from pipecat.frames.frames import TTSSpeakFrame

        self._queue(TTSSpeakFrame(text=self._greeting, append_to_context=True))

    def set_voice_focus(self, on: bool) -> bool:
        """Flip Quail enhancement on the agent's input. Returns the real state.

        Synchronous and safe from any thread: ``VoiceFocus`` guards itself with
        its own lock, and the processor only ever reads the flag.
        """
        if self._voice_focus is None:
            return False
        return self._voice_focus.set_enabled(on)

    # -- outbound to the browser UI ----------------------------------------- #

    def send_ui(self, message: dict) -> None:
        """Push one JSON message to the browser over the WebRTC data channel."""
        if self._loop:
            self._loop.call_soon_threadsafe(self._connection.send_app_message, message)

    # -- internals ---------------------------------------------------------- #

    def _queue(self, frame) -> None:
        """Put a frame into the running pipeline from any thread."""
        if self._loop and self._worker:
            asyncio.run_coroutine_threadsafe(self._worker.queue_frame(frame), self._loop)

    def _call_soon(self, fn: Callable[[], None]) -> None:
        if self._loop:
            self._loop.call_soon_threadsafe(fn)

    def _log(self, kind: str, text: str) -> None:
        if self._on_log:
            self._on_log(kind, text)


def _role_of(message) -> str:
    """Read a role off either a plain dict message or a Pipecat message object."""
    if isinstance(message, dict):
        return message.get("role", "")
    return getattr(message, "role", "")


def _flux_settings(profile: dict) -> dict:
    """Turn a decision-layer VAD profile into Deepgram Flux settings.

    One translation, and it is load-bearing. ``patient`` sets
    ``eager_eot_threshold`` to None, meaning "stop speculating on the reply".
    Neither obvious way of saying that to Flux works:

    - Sending an explicit null is rejected by Deepgram, and a rejected Configure
      fails silently, so speculation would quietly stay on.
    - Omitting the key entirely leaves the *previous* value in place, because
      Pipecat treats an absent field as NOT_GIVEN, meaning "do not change". So
      after one swap into patient, the eager 0.3 would still be live.

    Instead, pin it to this profile's own ``eot_threshold``. Flux may then only
    speculate once it is already as confident as it needs to be to end the turn,
    which is the same thing as not speculating ahead, expressed in a value
    Deepgram accepts. It also honors Flux's rule that the eager threshold must
    never exceed the end-of-turn threshold.
    """
    out = {k: v for k, v in profile.items() if v is not None}
    if profile.get("eager_eot_threshold") is None and "eot_threshold" in out:
        out["eager_eot_threshold"] = out["eot_threshold"]
    return out


def _to_float32(pcm: bytes) -> np.ndarray:
    """Transport PCM16 to the mono float32 both Tyto and Quail expect."""
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0


def _to_pcm16(mono: np.ndarray) -> bytes:
    """Back the other way, for a frame continuing down the pipeline."""
    return (np.clip(mono, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


class TytoAudioTap(FrameProcessor):
    """Feeds the user's microphone into Tyto, and gates it.

    Sits one hop after the transport input, so it sees every
    ``InputAudioRawFrame`` before the transcriber does. It normally passes every
    frame through untouched. When the gate is shut it drops the audio instead,
    which is how the Reactive layer stops the agent hearing itself say a nudge.
    """

    def __init__(self, scorer):
        super().__init__()
        self._scorer = scorer
        self._enabled = True

    def set_enabled(self, on: bool) -> None:
        self._enabled = on

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, InputAudioRawFrame):
            if not self._enabled:
                return  # gate shut: neither Tyto nor Flux sees this audio
            self._scorer.feed(_to_float32(frame.audio))
        await self.push_frame(frame, direction)


class VoiceFocusProcessor(FrameProcessor):
    """Optionally cleans the audio continuing to the agent. Never Tyto's copy.

    A pass-through when the switch is off or the model did not load, which is
    why it is always in the pipeline rather than being wired in conditionally:
    one shape to read, on camera and in a stack trace.

    Enhancement is block-aligned, so a frame in is not a frame out. The enhancer
    carries a residual and returns only the audio that is ready, so a frame is
    rewritten to whatever came back, and dropped when nothing did. The samples
    are not lost, they arrive on a later frame.
    """

    def __init__(self, voice_focus):
        super().__init__()
        self._vf = voice_focus

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if self._vf is not None and self._vf.enabled and isinstance(frame, InputAudioRawFrame):
            out = self._vf.process(_to_float32(frame.audio))
            if len(out) == 0:
                return  # nothing ready yet; it will arrive on a later frame
            frame.audio = _to_pcm16(out)
            # num_frames is derived from len(audio) and is not settable via the
            # constructor, so it has to be corrected by hand.
            frame.num_frames = len(out)
        await self.push_frame(frame, direction)


# The only frames the observer acts on. Everything else, and in particular the
# flood of InputAudioRawFrames, is rejected before it can touch the dedupe set.
# See the note in on_push_frame.
HANDLED_FRAMES = (
    LLMFullResponseStartFrame,
    LLMFullResponseEndFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    LLMTextFrame,
    TranscriptionFrame,
    InterimTranscriptionFrame,
)


class TytoFrameObserver(BaseObserver):
    """Turns pipeline frames into ``Handlers`` calls for the controller.

    Every frame push is observed, so react exactly once per frame.
    """

    def __init__(self, provider: CascadeProvider):
        super().__init__()
        self._p = provider
        self._seen: set[int] = set()
        self._agent_text = ""

    async def on_push_frame(self, data: FramePushed):
        frame = data.frame

        # Filter BEFORE deduping. One frame is pushed once per processor it
        # crosses, so dedupe is needed, but it must key on ``frame.id``, which is
        # monotonic, and never on ``id(frame)``, which is a memory address
        # CPython recycles the moment a frame is freed. Audio frames arrive about
        # every 10 ms and are freed immediately, so an address-keyed set fills
        # with exactly the addresses the next transcript will be allocated at,
        # and transcripts silently vanish.
        if not isinstance(frame, HANDLED_FRAMES):
            return
        if frame.id in self._seen:
            return
        self._seen.add(frame.id)
        if len(self._seen) > 4096:
            # Safe to forget: ids only increase, so a cleared id cannot come back
            # around and be mistaken for a new frame.
            self._seen.clear()
            self._seen.add(frame.id)

        p, h = self._p, self._p.h

        # A normal reply: the LLM response frames bracket the agent's turn.
        if isinstance(frame, LLMFullResponseStartFrame):
            self._agent_text = ""
            if h.on_agent_speaking:
                h.on_agent_speaking(True)
        elif isinstance(frame, LLMFullResponseEndFrame):
            if self._agent_text and h.on_agent_transcript:
                h.on_agent_transcript(self._agent_text, True)
            self._agent_text = ""
            if h.on_agent_speaking:
                h.on_agent_speaking(False)

        # Agent audio. A nudge is a TTSSpeakFrame and produces no LLM response
        # frames, so it is reported as agent speech from here.
        elif isinstance(frame, BotStartedSpeakingFrame):
            if p._pending_nudge:
                p._pending_nudge = False
                p._nudge_in_flight = True
                if h.on_agent_speaking:
                    h.on_agent_speaking(True, nudge=True)
            if h.on_agent_audio:
                h.on_agent_audio(True)
        elif isinstance(frame, BotStoppedSpeakingFrame):
            if p._nudge_in_flight:
                p._nudge_in_flight = False
                if h.on_agent_speaking:
                    h.on_agent_speaking(False, nudge=True)
            if h.on_agent_audio:
                h.on_agent_audio(False)

        elif isinstance(frame, LLMTextFrame):
            self._agent_text += frame.text
            if h.on_agent_transcript:
                h.on_agent_transcript(self._agent_text, False)
        elif isinstance(frame, TranscriptionFrame):
            if h.on_user_transcript:
                h.on_user_transcript(frame.text, True)
        elif isinstance(frame, InterimTranscriptionFrame):
            if h.on_user_transcript:
                h.on_user_transcript(frame.text, False)
