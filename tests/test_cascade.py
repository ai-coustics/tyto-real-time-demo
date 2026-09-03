"""The three Tyto layers, checked at the frame the cascade actually emits.

These are the only tests that tie ``decision.py``'s intent to Pipecat. Everything
above this line is provider-agnostic and tested without pipecat installed; this
file is where "Layer 2 retunes turn-taking" becomes "an STTUpdateSettingsFrame
carrying these keys".

The pipeline is never started. ``_queue`` is replaced with a list append, so each
test reads the frame the provider would have put into a running pipeline.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pipecat", reason="cascade tests need the voice stack installed")

from pipecat.frames.frames import (  # noqa: E402
    InterruptionFrame,
    LLMMessagesTransformFrame,
    STTUpdateSettingsFrame,
    TTSSpeakFrame,
)

from tyto_voice.cascade import (  # noqa: E402
    CascadeProvider,
    TytoAudioTap,
    TytoFrameObserver,
    VoiceFocusProcessor,
    _flux_settings,
)
from tyto_voice.decision import VAD_PROFILES  # noqa: E402
from tyto_voice.provider import Handlers  # noqa: E402


class FakeScorer:
    scoring = True

    def feed(self, mono):
        pass


def make_provider():
    """A provider with its pipeline replaced by a frame recorder."""
    p = CascadeProvider(
        Handlers(),
        deepgram_key="dg",
        openai_key="oa",
        instructions="SYSTEM",
        greeting="Hello there.",
        scorer=FakeScorer(),
        webrtc_connection=object(),
    )
    p.sent = []
    p._queue = p.sent.append
    return p


# -- Layer 1, Aware --------------------------------------------------------- #


def test_aware_rewrites_the_system_message_and_keeps_history():
    p = make_provider()
    p.set_instructions("SYSTEM\n\nAudio note: degraded input, loud background noise.")

    (frame,) = p.sent
    assert isinstance(frame, LLMMessagesTransformFrame)

    history = [
        {"role": "system", "content": "SYSTEM"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    out = frame.transform(history)
    assert out[0]["content"].endswith("loud background noise.")
    assert out[1:] == history[1:], "the conversation must survive an Aware update"


def test_aware_does_not_trigger_a_reply():
    """The room note changes how the agent talks, it does not make it talk."""
    p = make_provider()
    p.set_instructions("SYSTEM\n\nAudio note: marginal input.")
    assert p.sent[0].run_llm is False


def test_aware_inserts_a_system_message_when_there_is_none():
    p = make_provider()
    p.set_instructions("SYSTEM")
    out = p.sent[0].transform([{"role": "user", "content": "hi"}])
    assert out[0] == {"role": "system", "content": "SYSTEM"}


# -- Layer 2, Tuned --------------------------------------------------------- #


def test_tuned_sends_flux_thresholds_as_a_typed_delta():
    """The mapping form of STTUpdateSettingsFrame is deprecated and warns."""
    p = make_provider()
    p.set_turn_detection(VAD_PROFILES["patient"])

    (frame,) = p.sent
    assert isinstance(frame, STTUpdateSettingsFrame)
    assert frame.delta is not None, "must use the typed delta, not settings={...}"
    assert frame.delta.eot_threshold == 0.7
    assert frame.delta.eot_timeout_ms == 4000


def test_stopping_speculation_is_a_real_threshold_not_a_null():
    """"Stop speculating" has to survive the trip to Deepgram.

    An explicit null is rejected and fails silently; omitting the key leaves the
    previous eager value live, because Pipecat reads an absent field as
    NOT_GIVEN. Pinning it to this profile's own eot_threshold says the same
    thing in a value Flux accepts.
    """
    assert VAD_PROFILES["patient"]["eager_eot_threshold"] is None

    patient = _flux_settings(VAD_PROFILES["patient"])
    assert patient["eager_eot_threshold"] == patient["eot_threshold"]
    assert patient["eager_eot_threshold"] is not None

    # Eager still speculates well ahead of the turn ending.
    eager = _flux_settings(VAD_PROFILES["eager"])
    assert eager["eager_eot_threshold"] == 0.3
    assert eager["eager_eot_threshold"] < eager["eot_threshold"]


def test_eager_threshold_never_exceeds_the_end_of_turn_threshold():
    """Flux's own rule, for both profiles."""
    for name in ("eager", "patient"):
        cfg = _flux_settings(VAD_PROFILES[name])
        assert cfg["eager_eot_threshold"] <= cfg["eot_threshold"], name


def test_patient_is_actually_more_patient_than_eager():
    eager, patient = _flux_settings(VAD_PROFILES["eager"]), _flux_settings(VAD_PROFILES["patient"])
    assert patient["eot_threshold"] > eager["eot_threshold"]
    assert patient["eot_timeout_ms"] > eager["eot_timeout_ms"]


def test_listen_gate_sends_no_configure():
    """None is the gate, not a profile. The mic is already shut by then."""
    p = make_provider()
    p.set_turn_detection(None)
    assert p.sent == []


# -- Layer 3, Reactive ------------------------------------------------------ #


def test_nudge_is_spoken_directly_and_recorded_in_history():
    p = make_provider()
    p.nudge("Could you move somewhere quieter?")

    (frame,) = p.sent
    assert isinstance(frame, TTSSpeakFrame), "a nudge must not cost an LLM round trip"
    assert frame.text == "Could you move somewhere quieter?"
    assert frame.append_to_context is True, "the agent must know it said the line"


def test_interrupt_cuts_the_reply_in_flight():
    p = make_provider()
    p.interrupt(clear_input=True)
    assert isinstance(p.sent[0], InterruptionFrame)


def test_interrupt_abandons_a_nudge_that_has_not_started():
    p = make_provider()
    p.nudge("Could you move somewhere quieter?")
    assert p._pending_nudge is True
    p.interrupt()
    assert p._pending_nudge is False


def test_greeting_is_spoken_without_a_model_round_trip():
    p = make_provider()
    p.request_response()

    (frame,) = p.sent
    assert isinstance(frame, TTSSpeakFrame)
    assert frame.text == "Hello there."
    assert frame.append_to_context is True


# -- the mic gate ----------------------------------------------------------- #


def test_mic_gate_stops_tyto_and_flux_together():
    """One gate, so the agent can never be triggered by its own nudge.

    Behavioural on purpose. Asserting that set_enabled flips a boolean proves
    nothing: deleting the drop in TytoAudioTap.process_frame leaves that green
    while the agent starts answering its own nudge.
    """
    import asyncio

    fed = []
    scorer = type("S", (), {"feed": lambda self, m: fed.append(len(m))})()
    tap = TytoAudioTap(scorer)

    out = asyncio.run(_run(tap, _audio_frame([0.1, 0.2])))
    assert fed == [2], "Tyto should be fed while the gate is open"
    assert len(out) == 1, "Flux should see the audio while the gate is open"

    tap.set_enabled(False)
    out = asyncio.run(_run(tap, _audio_frame([0.3, 0.4])))
    assert fed == [2], "gate shut: Tyto must not be fed"
    assert out == [], "gate shut: Flux must not see the audio either"

    tap.set_enabled(True)
    asyncio.run(_run(tap, _audio_frame([0.5, 0.6])))
    assert fed == [2, 2], "the gate must reopen"


def test_set_mic_enabled_reaches_the_tap():
    """The controller's only route to the gate. Untested until now."""
    p = make_provider()
    p._tap = TytoAudioTap(FakeScorer())
    p._call_soon = lambda fn: fn()  # run the loop hop inline

    p.set_mic_enabled(False)
    assert p._tap._enabled is False
    p.set_mic_enabled(True)
    assert p._tap._enabled is True


# -- Voice Focus ------------------------------------------------------------ #


class FakeVoiceFocus:
    """Marks everything it touches, so "did this reach Tyto?" is answerable."""

    def __init__(self, enabled=True):
        self.enabled = enabled
        self.available = True
        self.seen = []

    def process(self, mono):
        import numpy as np

        self.seen.append(len(mono))
        return np.full(len(mono), 0.5, dtype=np.float32)


def _pcm(values):
    import numpy as np

    return (np.array(values, dtype=np.float32) * 32767).astype("<i2").tobytes()


def _audio_frame(values):
    from pipecat.frames.frames import InputAudioRawFrame

    return InputAudioRawFrame(audio=_pcm(values), sample_rate=16000, num_channels=1)


async def _run(processor, frame):
    """Push one frame through a processor, collecting what comes out."""
    from pipecat.processors.frame_processor import FrameDirection

    out = []
    processor.push_frame = lambda f, d=None: out.append(f) or _noop()
    await processor.process_frame(frame, FrameDirection.DOWNSTREAM)
    return out


async def _noop():
    return None


def test_voice_focus_is_a_passthrough_when_off():
    import asyncio

    vf = FakeVoiceFocus(enabled=False)
    proc = VoiceFocusProcessor(vf)

    frame = _audio_frame([0.1, 0.2, 0.3, 0.4])
    original = frame.audio
    out = asyncio.run(_run(proc, frame))

    assert vf.seen == [], "the enhancer must not be called while it is off"
    assert out[0].audio == original


def test_voice_focus_rewrites_the_audio_when_on():
    import asyncio

    vf = FakeVoiceFocus(enabled=True)
    proc = VoiceFocusProcessor(vf)

    frame = _audio_frame([0.1, 0.2, 0.3, 0.4])
    original = frame.audio
    out = asyncio.run(_run(proc, frame))

    assert vf.seen == [4]
    assert out[0].audio != original
    assert out[0].num_frames == 4, "num_frames is derived from len(audio), fix it by hand"


def test_voice_focus_drops_a_frame_when_nothing_is_ready_yet():
    """Enhancement is block-aligned, so a frame in is not always a frame out."""
    import asyncio

    import numpy as np

    vf = FakeVoiceFocus(enabled=True)
    vf.process = lambda mono: np.empty(0, dtype=np.float32)
    proc = VoiceFocusProcessor(vf)

    out = asyncio.run(_run(proc, _audio_frame([0.1, 0.2])))
    assert out == [], "an empty result must not be pushed as a silent frame"


def test_tyto_is_upstream_of_voice_focus_in_the_real_pipeline():
    """The one wiring mistake that would silently ruin the demo.

    If the enhancer ran before the tap, Tyto would score Quail's output: the
    meters would go green, the room note would go quiet and the Reactive layer
    would stop firing, in a room that had not changed at all. It would still
    look like it worked, which is why this is asserted rather than commented.
    """
    import asyncio

    from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection

    async def build():
        p = CascadeProvider(
            Handlers(),
            deepgram_key="dg",
            openai_key="oa",
            instructions="SYSTEM",
            greeting="Hi.",
            scorer=FakeScorer(),
            voice_focus=FakeVoiceFocus(),
            webrtc_connection=SmallWebRTCConnection(),
        )
        p._loop = asyncio.get_event_loop()
        p._build_pipeline()
        # The worker wraps the real pipeline in a source/sink pair, so the
        # processors we built are one level down.
        for stage in p._worker.pipeline._processors:
            inner = getattr(stage, "_processors", None)
            if inner and any(type(x).__name__ == "TytoAudioTap" for x in inner):
                return [type(x).__name__ for x in inner]
        raise AssertionError("could not find the built pipeline")

    names = asyncio.run(build())
    tap = names.index("TytoAudioTap")
    vf = names.index("VoiceFocusProcessor")
    stt = names.index("DeepgramFluxSTTService")

    assert tap < vf, "Tyto must score the microphone, never the enhanced signal"
    assert vf < stt, "the agent must hear the enhanced signal"


# -- the observer ----------------------------------------------------------- #


def _observer_with_capture():
    """A real observer whose Handlers calls are recorded."""
    got = {"user": [], "agent": [], "speaking": [], "audio": []}
    h = Handlers(
        on_user_transcript=lambda t, f: got["user"].append((t, f)),
        on_agent_transcript=lambda t, f: got["agent"].append((t, f)),
        on_agent_speaking=lambda a, **kw: got["speaking"].append((a, kw)),
        on_agent_audio=lambda p: got["audio"].append(p),
    )
    p = make_provider()
    p.h = h
    return TytoFrameObserver(p), got


class _Pushed:
    def __init__(self, frame):
        self.frame = frame


def test_dedupe_keys_on_frame_id_not_on_the_memory_address():
    """The regression: two different frames can share a memory address.

    CPython hands the address of a freed object straight back to the next
    allocation, so a set keyed on id(frame) reports a brand new transcript as
    "already seen" and silently drops it. frame.id is a monotonic counter and is
    the only safe key. Asserted on the set itself rather than by racing the
    allocator, so this cannot skip.
    """
    import asyncio

    from pipecat.frames.frames import TranscriptionFrame

    obs, got = _observer_with_capture()
    frame = TranscriptionFrame("hello there", "user", "t", None)
    asyncio.run(obs.on_push_frame(_Pushed(frame)))

    assert got["user"] == [("hello there", True)]
    assert obs._seen == {frame.id}, "dedupe must key on the monotonic frame.id"
    assert frame.id != id(frame), "frame.id is not the memory address"


def test_audio_frames_never_enter_the_dedupe_set():
    """The churn must not be tracked: it is the source of the recycling above."""
    import asyncio

    from pipecat.frames.frames import InputAudioRawFrame

    obs, _ = _observer_with_capture()

    async def go():
        for _ in range(500):
            f = InputAudioRawFrame(audio=b"\x00\x00" * 160, sample_rate=16000, num_channels=1)
            await obs.on_push_frame(_Pushed(f))
            del f

    asyncio.run(go())
    assert obs._seen == set(), "audio frames must be filtered out before deduping"


def test_a_frame_pushed_by_several_processors_is_handled_once():
    """Dedupe still has to work: one frame crosses many processors."""
    import asyncio

    from pipecat.frames.frames import TranscriptionFrame

    obs, got = _observer_with_capture()
    frame = TranscriptionFrame("only once", "user", "t", None)

    async def go():
        for _ in range(5):  # as if crossing five processors
            await obs.on_push_frame(_Pushed(frame))

    asyncio.run(go())
    assert got["user"] == [("only once", True)]


def test_interim_is_reported_as_not_final():
    import asyncio

    from pipecat.frames.frames import InterimTranscriptionFrame

    obs, got = _observer_with_capture()
    asyncio.run(
        obs.on_push_frame(_Pushed(InterimTranscriptionFrame("half a sen", "user", "t", None)))
    )
    assert got["user"] == [("half a sen", False)]


def test_nudge_is_reported_as_agent_speech_without_llm_frames():
    """A nudge is a TTSSpeakFrame, so only the Bot*SpeakingFrames bracket it."""
    import asyncio

    from pipecat.frames.frames import BotStartedSpeakingFrame, BotStoppedSpeakingFrame

    obs, got = _observer_with_capture()
    obs._p._pending_nudge = True

    async def go():
        await obs.on_push_frame(_Pushed(BotStartedSpeakingFrame()))
        await obs.on_push_frame(_Pushed(BotStoppedSpeakingFrame()))

    asyncio.run(go())
    assert got["speaking"] == [(True, {"nudge": True}), (False, {"nudge": True})]
    assert got["audio"] == [True, False]
