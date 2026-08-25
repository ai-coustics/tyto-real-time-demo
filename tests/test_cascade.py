"""Cascade tests with fakes: no SDK, no keys, no network, no hardware.

Covers the three pieces that would be expensive to debug live: how the VAD turns
a stream into utterances, how Inkling history is compacted, and how the Deepgram
speak socket decides a line is finished.
"""

import numpy as np
import pytest

from tyto_voice.decision import VAD_PROFILES
from tyto_voice.deepgram import DeepgramTTS
from tyto_voice.inkling import InklingClient, encode_wav
from tyto_voice.vad import (
    MIN_UTTERANCE_SECONDS,
    PREROLL_SECONDS,
    TAIL_SECONDS,
    LiveVad,
)

RATE = 16000
BLOCK = 160  # 10 ms, the VAD model's window order of magnitude


class FakeParams:
    Sensitivity = "sensitivity"
    MinimumSpeechDuration = "minimum_speech_duration"
    SpeechHoldDuration = "speech_hold_duration"


class FakeCtx:
    """Speech detection is driven by the test, not by the audio."""

    def __init__(self):
        self.speech = False
        self.params = {}
        self.resets = 0

    def is_speech_detected(self):
        return self.speech

    def raw_vad_probability(self):
        return 1.0 if self.speech else 0.0

    def set_parameter(self, param, value):
        self.params[param] = value

    def reset(self):
        self.resets += 1


class FakeVad:
    def __init__(self, ctx):
        self.ctx = ctx
        self.blocks = 0

    def process(self, audio):
        assert len(audio) == BLOCK, "the SDK requires exactly block_size samples"
        self.blocks += 1

    def get_context(self):
        return self.ctx


def build_vad(profile=None):
    live = LiveVad("fake-key", sample_rate=RATE, profile=profile)
    ctx = FakeCtx()
    live._configure(FakeVad(ctx), ctx, FakeParams, RATE, BLOCK)
    return live, ctx


def blocks(n, value=0.1):
    """n blocks worth of audio as one array."""
    return np.full(n * BLOCK, value, dtype=np.float32)


def end_silence_blocks(profile="eager"):
    """Enough blocks of silence to end a turn under `profile`, plus a margin.

    Derived rather than hardcoded so tuning end_silence does not silently turn
    these tests into no-ops.
    """
    return int(VAD_PROFILES[profile]["end_silence"] * RATE / BLOCK) + 5


# -- LiveVad segmentation --------------------------------------------------- #


def test_utterance_is_emitted_on_the_falling_edge():
    live, ctx = build_vad()
    assert live.feed(blocks(10)) is None  # silence
    ctx.speech = True
    assert live.feed(blocks(100)) is None  # one second of speech, still talking
    ctx.speech = False
    assert live.feed(blocks(1)) is None    # a single silent block is not the end
    utterance = live.feed(blocks(end_silence_blocks()))  # a full run of silence is
    assert utterance is not None
    assert len(utterance) > RATE  # at least the second of speech


def test_utterance_includes_preroll_so_the_first_syllable_survives():
    live, ctx = build_vad()
    live.feed(blocks(100))  # a second of silence fills the pre-roll
    ctx.speech = True
    live.feed(blocks(50))
    ctx.speech = False
    utterance = live.feed(blocks(end_silence_blocks()))
    # 50 blocks of speech is 0.5 s; the pre-roll must have added roughly its own
    # window on top, otherwise the onset is being clipped.
    assert len(utterance) > round(0.5 * RATE)
    assert len(utterance) <= round((0.5 + PREROLL_SECONDS + TAIL_SECONDS + 0.05) * RATE)


def test_preroll_is_bounded():
    live, _ = build_vad()
    live.feed(blocks(1000))  # ten seconds of silence
    assert live._preroll_samples <= round(PREROLL_SECONDS * RATE) + BLOCK


def test_short_dropouts_do_not_split_an_utterance():
    """Measured against real speech, is_speech_detected() drops out for 45 to
    285 ms at ordinary pauses inside one sentence. Ending the turn on the
    falling edge split a single 4.8 s question into three utterances."""
    live, ctx = build_vad(VAD_PROFILES["eager"])
    ctx.speech = True
    assert live.feed(blocks(100)) is None      # 1.0 s of speech
    ctx.speech = False
    assert live.feed(blocks(30)) is None       # 300 ms gap, longer than measured
    ctx.speech = True
    assert live.feed(blocks(100)) is None      # speech resumes, still one turn
    ctx.speech = False
    utterance = live.feed(blocks(end_silence_blocks()))  # a real run of silence
    assert utterance is not None
    assert len(utterance) > round(2.0 * RATE)  # both halves are in there


def test_the_turn_ends_after_a_continuous_run_of_silence():
    live, ctx = build_vad(VAD_PROFILES["eager"])
    end_silence = VAD_PROFILES["eager"]["end_silence"]
    ctx.speech = True
    live.feed(blocks(100))
    ctx.speech = False
    short = int(end_silence * RATE / BLOCK) - 2
    assert live.feed(blocks(short)) is None    # not yet
    assert live.feed(blocks(3)) is not None    # now


def test_trailing_silence_is_trimmed_off_the_utterance():
    """The silence that ended the turn is dead weight in the prompt."""
    from tyto_voice.vad import TAIL_SECONDS

    live, ctx = build_vad(VAD_PROFILES["eager"])
    ctx.speech = True
    live.feed(blocks(100))  # 1.0 s of speech
    ctx.speech = False
    utterance = live.feed(blocks(end_silence_blocks() * 2))  # well past end_silence
    assert utterance is not None
    # Speech plus pre-roll plus only the short tail, nowhere near 3 s.
    assert len(utterance) <= round((1.0 + PREROLL_SECONDS + TAIL_SECONDS + 0.1) * RATE)


def test_patient_holds_the_turn_open_longer_than_eager():
    assert (VAD_PROFILES["patient"]["end_silence"]
            > VAD_PROFILES["eager"]["end_silence"])
    assert (VAD_PROFILES["patient"]["sensitivity"]
            > VAD_PROFILES["eager"]["sensitivity"])


def test_blips_shorter_than_the_minimum_are_dropped():
    live, ctx = build_vad()
    ctx.speech = True
    live.feed(blocks(5))  # 50 ms, a click
    ctx.speech = False
    assert live.feed(blocks(end_silence_blocks())) is None


def test_long_monologue_is_cut_and_the_turn_stays_open():
    live, ctx = build_vad()
    live._max_samples = round(1.0 * RATE)  # shorten the cap for the test
    ctx.speech = True
    forced = live.feed(blocks(120))  # 1.2 s, past the cap
    assert forced is not None
    assert live.speaking is True  # still mid-sentence, keep collecting


def test_odd_sized_feeds_are_buffered_not_dropped():
    """Mic callbacks do not arrive in model-sized blocks, so the remainder of
    each feed has to survive until the next one."""
    live, ctx = build_vad()
    ctx.speech = True
    chunk = BLOCK + 37  # deliberately not a whole block
    feeds = 60          # 60 * 197 = 11820 samples, comfortably past the minimum
    for _ in range(feeds):
        live.feed(np.full(chunk, 0.1, dtype=np.float32))
    ctx.speech = False
    utterance = live.feed(blocks(end_silence_blocks()))
    assert utterance is not None
    assert len(utterance) >= round(MIN_UTTERANCE_SECONDS * RATE)
    # Everything fed should be accounted for, minus at most one partial block
    # still held as the residual.
    assert len(utterance) >= feeds * chunk - BLOCK


def test_profile_switch_reaches_the_sdk():
    live, ctx = build_vad(profile=VAD_PROFILES["eager"])
    assert ctx.params["sensitivity"] == VAD_PROFILES["eager"]["sensitivity"]
    live.set_profile(VAD_PROFILES["patient"])
    assert ctx.params["sensitivity"] == VAD_PROFILES["patient"]["sensitivity"]
    assert ctx.params["speech_hold_duration"] == VAD_PROFILES["patient"]["speech_hold_duration"]


def test_none_profile_is_the_listen_gate():
    live, ctx = build_vad()
    ctx.speech = True
    live.feed(blocks(50))
    live.set_profile(None)  # controller stopped listening
    assert live.enabled is False
    assert ctx.resets == 1  # the half-finished utterance was dropped
    ctx.speech = False
    assert live.feed(blocks(10)) is None  # no audio accepted while gated
    live.set_profile(VAD_PROFILES["eager"])
    assert live.enabled is True


# -- Inkling history -------------------------------------------------------- #


def build_llm():
    return InklingClient("fake-key", instructions="SYSTEM")


def test_wav_header_is_well_formed():
    wav = encode_wav(np.zeros(RATE, dtype=np.float32), RATE)
    import io
    import wave

    with wave.open(io.BytesIO(wav)) as w:
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getframerate() == RATE
        # The bug this guards: Deepgram's own container writes a placeholder
        # length that reads back as roughly nineteen hours.
        assert w.getnframes() == RATE


def test_only_the_current_utterance_is_ever_sent_as_audio():
    """Pinned deliberately. The model listens to a retained past utterance and
    describes the room it hears there as though it were the room now: with two
    turns retained, a recording made while a TV was on produced "you sound like
    you're in a quiet room with people talking in the background" in answer to
    an unrelated question, long after the TV was off."""
    from tyto_voice.inkling import AUDIO_HISTORY_TURNS

    assert AUDIO_HISTORY_TURNS == 1


def test_the_audio_window_carries_the_newest_turns_and_drops_the_rest():
    from tyto_voice.inkling import AUDIO_HISTORY_TURNS, _AgentTurn, _UserTurn

    llm = build_llm()
    llm._turns = [
        _UserTurn("AAA"), _AgentTurn("reply one"),
        _UserTurn("BBB"), _AgentTurn("reply two"),
        _UserTurn("CCC"),
    ]
    messages = llm._build_messages()
    audio = [m for m in messages if isinstance(m["content"], list)]
    assert len(audio) == AUDIO_HISTORY_TURNS
    carried = [m["content"][0]["input_audio"]["data"] for m in audio]
    assert carried == ["BBB", "CCC"][-AUDIO_HISTORY_TURNS:]
    # Older user turns fall out entirely, but every agent reply stays as text,
    # so the thread of the conversation survives for almost no tokens.
    assert [m["content"] for m in messages if m["role"] == "assistant"] == [
        "reply one", "reply two"
    ]


def test_no_user_turn_is_ever_sent_as_text():
    """Nothing transcribes the user, so audio is the only representation."""
    from tyto_voice.inkling import _UserTurn

    llm = build_llm()
    llm._turns = [_UserTurn("AAA"), _UserTurn("BBB"), _UserTurn("CCC")]
    for m in llm._build_messages():
        if m["role"] == "user":
            assert isinstance(m["content"], list)


def test_history_is_bounded():
    from tyto_voice.inkling import MAX_HISTORY_TURNS

    llm = build_llm()
    for i in range(MAX_HISTORY_TURNS * 3):
        llm.add_agent_line(f"line {i}")
    assert len(llm._turns) == MAX_HISTORY_TURNS


def test_instructions_swap_is_the_system_message():
    llm = build_llm()
    llm.set_instructions("SYSTEM\n\nAudio note: loud background noise.")
    assert "Audio note" in llm._build_messages()[0]["content"]


# -- Deepgram speak socket -------------------------------------------------- #


def build_tts():
    events = {"audio": [], "started": 0, "finished": 0}

    def audio_out(chunk):
        events["audio"].append(chunk)

    tts = DeepgramTTS(
        "fake-key",
        audio_out=audio_out,
        on_started=lambda: events.__setitem__("started", events["started"] + 1),
        on_finished=lambda: events.__setitem__("finished", events["finished"] + 1),
    )
    tts._send = lambda obj: None  # nothing is actually written in tests
    tts._ws = object()            # stand in for a live socket
    return tts, events


def test_flushed_finishes_a_line_once():
    tts, events = build_tts()
    tts.speak("hello")
    tts._receive(b"\x00\x01")
    tts._receive(b"\x02\x03")
    assert events["started"] == 1  # only the first chunk announces
    assert len(events["audio"]) == 2
    tts._receive('{"type":"Flushed","sequence_id":0}')
    assert events["finished"] == 1
    tts._receive('{"type":"Flushed","sequence_id":0}')  # duplicate
    assert events["finished"] == 1


def test_clear_drops_audio_and_never_reports_finished():
    tts, events = build_tts()
    tts.speak("hello")
    tts.clear()
    tts._receive(b"\x00\x01")  # tail of the abandoned line
    assert events["audio"] == []
    assert events["finished"] == 0


def test_a_cleared_lines_flushed_cannot_end_the_next_line():
    """The race this guards: interrupt, speak again, and the old Flushed lands."""
    tts, events = build_tts()
    tts.speak("first")
    tts.clear()
    tts.speak("second")
    tts._receive('{"type":"Flushed","sequence_id":0}')  # belongs to "first"
    assert events["finished"] == 0
    assert tts.speaking is True
    tts._receive('{"type":"Flushed","sequence_id":1}')  # belongs to "second"
    assert events["finished"] == 1


def test_empty_text_is_not_spoken():
    tts, _ = build_tts()
    assert tts.speak("   ") is False
    assert tts.speaking is False


def test_speak_fails_honestly_when_the_socket_is_gone():
    """A True here would leave the caller waiting for a Flushed that can never
    arrive, which means the mic stays muted and the demo goes silent."""
    tts, _ = build_tts()
    tts._ws = None
    assert tts.speak("hello") is False

    tts, _ = build_tts()
    tts.closed.set()
    assert tts.speak("hello") is False


def test_socket_death_closes_out_a_line_in_progress():
    tts, events = build_tts()

    async def dead_socket():
        raise ConnectionError("socket dropped")

    tts._main = dead_socket  # no network in unit tests
    tts.speak("half a sentence")
    tts._run()  # the reader loop exits with no Flushed for the line in flight
    assert events["finished"] == 1
    assert tts.speaking is False
    assert tts.closed.is_set()


# -- provider glue ----------------------------------------------------------- #


def test_tool_definition_is_converted_to_chat_completions_shape():
    from tyto_voice.cascade import CascadeProvider
    from tyto_voice.controller import CHECK_AUDIO_QUALITY_TOOL

    wrapped = CascadeProvider._as_openai_tools([CHECK_AUDIO_QUALITY_TOOL])
    assert wrapped[0]["type"] == "function"
    assert wrapped[0]["function"]["name"] == "check_audio_quality"
    assert "parameters" in wrapped[0]["function"]
    # Already-nested definitions pass through untouched.
    assert CascadeProvider._as_openai_tools(wrapped) == wrapped


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))


# -- the whole loop: real controller, real provider, fakes at the edges ------ #
#
# These are the tests that matter most. Every failure mode here ends the same
# way for the user: the mic never unmutes and the demo goes silent with no
# error on screen.


class FakeTTS:
    def __init__(self):
        self.lines = []
        self.clears = 0
        self.alive = True
        self.speaking = False
        self.on_finished = None

    def speak(self, text):
        if not text.strip() or not self.alive:
            return False
        self.lines.append(text)
        self.speaking = True
        return True

    def clear(self):
        self.clears += 1
        self.speaking = False

    def finish(self):
        """Deepgram sent Flushed."""
        if self.speaking:
            self.speaking = False
            self.on_finished()


class FakeLLM:
    def __init__(self, reply="sure thing"):
        self.reply = reply
        self.lines = []
        self.contexts = []
        self.instructions = ""
        self.calls = 0

    def respond(self, audio, sample_rate=16000, context=None,
                tool_handler=None, cancelled=None):
        self.calls += 1
        self.contexts.append(context)
        if callable(self.reply):
            return self.reply(tool_handler, cancelled)
        if self.reply is None:
            return None
        return type("R", (), {"text": self.reply, "tool_calls": 0})()

    def set_instructions(self, text):
        self.instructions = text

    def add_agent_line(self, text):
        self.lines.append(text)


def wait_until(predicate, timeout=3.0):
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def build_loop(reply="sure thing"):
    from tyto_voice.cascade import CascadeProvider
    from tyto_voice.controller import TytoController
    from tyto_voice.provider import Handlers

    handlers = Handlers()
    provider = CascadeProvider(
        handlers,
        license_key="x", inkling_key="x", deepgram_key="x",
        instructions="SYSTEM", greeting="hello there",
        audio_out=lambda b: None,
        transcribe_user=False,  # the caption is a network call; not in unit tests
    )
    tts, llm = FakeTTS(), FakeLLM(reply)
    tts.on_finished = provider._on_tts_finished
    provider.tts, provider.llm = tts, llm

    ctx = FakeCtx()
    provider.vad._configure(FakeVad(ctx), ctx, FakeParams, RATE, BLOCK)

    scorer = type("S", (), {
        "scoring": True,
        "pause": lambda self: setattr(self, "scoring", False),
        "resume": lambda self: setattr(self, "scoring", True),
    })()
    controller = TytoController(provider, scorer)
    handlers.on_ready = controller.on_ready
    handlers.on_agent_speaking = controller.on_agent_speaking
    handlers.on_agent_audio = controller.on_agent_audio
    handlers.on_user_transcript = controller.on_user_transcript
    handlers.on_agent_transcript = controller.on_agent_transcript
    handlers.on_tool_call = controller.on_tool_call
    controller.set_connected(True)
    return provider, controller, tts, llm, ctx, scorer


def speak_an_utterance(provider, ctx):
    ctx.speech = True
    provider.send_audio(blocks(100))
    ctx.speech = False
    # Use the longest profile: a noisy score can have flipped Tuned to patient,
    # which needs more silence than eager before the turn is considered over.
    provider.send_audio(blocks(end_silence_blocks("patient")))


def test_greeting_opens_and_releases_the_agent():
    provider, controller, tts, llm, _, scorer = build_loop()
    controller.on_ready()
    assert tts.lines == ["hello there"]
    assert llm.lines == ["hello there"]  # recorded, so it knows what it said
    assert controller.agent_speaking is True
    assert scorer.scoring is False       # Tyto pauses while the agent talks
    tts.finish()
    assert controller.agent_speaking is False
    assert scorer.scoring is True


def test_a_full_turn_returns_the_mic():
    provider, controller, tts, llm, ctx, scorer = build_loop("sure thing")
    controller.on_ready()
    tts.finish()
    speak_an_utterance(provider, ctx)
    assert wait_until(lambda: "sure thing" in tts.lines)
    assert controller.agent_speaking is True  # muted while thinking and speaking
    tts.finish()
    assert wait_until(lambda: controller.agent_speaking is False)
    assert scorer.scoring is True


def test_a_failed_model_call_still_returns_the_mic():
    """The silent-forever bug: no reply means nothing will ever say Flushed."""
    provider, controller, tts, llm, ctx, scorer = build_loop(None)
    controller.on_ready()
    tts.finish()
    speak_an_utterance(provider, ctx)
    assert wait_until(lambda: llm.calls == 1)
    assert wait_until(lambda: controller.agent_speaking is False)
    assert scorer.scoring is True
    assert tts.lines == ["hello there"]  # nothing was spoken for the failed turn


def test_a_dead_voice_socket_still_returns_the_mic():
    provider, controller, tts, llm, ctx, scorer = build_loop("sure thing")
    controller.on_ready()
    tts.finish()
    tts.alive = False  # the speak socket died
    speak_an_utterance(provider, ctx)
    assert wait_until(lambda: llm.calls == 1)
    assert wait_until(lambda: controller.agent_speaking is False)
    assert scorer.scoring is True


def test_an_empty_reply_still_returns_the_mic():
    provider, controller, tts, llm, ctx, scorer = build_loop("   ")
    controller.on_ready()
    tts.finish()
    speak_an_utterance(provider, ctx)
    assert wait_until(lambda: llm.calls == 1)
    assert wait_until(lambda: controller.agent_speaking is False)


def test_nudge_interrupts_speaks_and_resumes():
    from tyto_voice.decision import NUDGE_MIN_PERSIST, Scores

    provider, controller, tts, llm, ctx, scorer = build_loop()
    controller.on_ready()
    tts.finish()

    bad = Scores(risk_score=0.7, noise=0.0, speaker_reverb=0.0, speaker_loudness=0.0,
                 interfering_speech=0.8, packet_loss=0.0, codec_degradation=0.0)
    for _ in range(NUDGE_MIN_PERSIST):
        controller.on_scores(bad)

    assert tts.clears == 1                      # the interrupt reached the voice
    assert controller.nudge_active is True
    assert controller.listening is False
    assert "other voices" in tts.lines[-1]      # the nudge itself was spoken
    assert tts.lines[-1] == llm.lines[-1]       # and recorded in history

    tts.finish()
    assert controller.listening is True         # back to listening
    assert controller.nudge_active is False
    assert scorer.scoring is True


def test_no_new_turn_starts_while_the_agent_is_busy():
    provider, controller, tts, llm, ctx, scorer = build_loop()
    controller.on_ready()  # greeting is speaking, mic is muted
    speak_an_utterance(provider, ctx)
    assert llm.calls == 0, "audio during agent speech must not start a turn"


def test_interrupt_when_nothing_is_speaking_is_harmless():
    provider, controller, tts, llm, ctx, scorer = build_loop()
    controller.on_ready()
    tts.finish()
    provider.interrupt()
    provider.interrupt()
    speak_an_utterance(provider, ctx)
    assert wait_until(lambda: "sure thing" in tts.lines)
    tts.finish()
    assert wait_until(lambda: controller.agent_speaking is False)


def test_check_audio_quality_round_trips_through_the_controller():
    from tyto_voice.decision import Scores

    seen = {}

    def reply(tool_handler, cancelled):
        seen["result"] = tool_handler("check_audio_quality", "call-1")
        return type("R", (), {"text": "you sound noisy", "tool_calls": 1})()

    provider, controller, tts, llm, ctx, scorer = build_loop(reply)
    controller.on_ready()
    tts.finish()
    # Noisy enough to name a cause, but under the nudge gate, so the snapshot is
    # populated without the Reactive layer taking over the turn.
    controller.on_scores(Scores(risk_score=0.35, noise=0.8, speaker_reverb=0.0,
                                speaker_loudness=0.0, interfering_speech=0.0,
                                packet_loss=0.0, codec_degradation=0.0))
    assert controller.awaiting_nudge is False
    speak_an_utterance(provider, ctx)
    assert wait_until(lambda: "result" in seen)
    assert seen["result"]["verdict"] == "marginal"
    assert seen["result"]["top_issue"]["key"] == "noise"
    assert wait_until(lambda: "you sound noisy" in tts.lines)
    tts.finish()
    assert wait_until(lambda: controller.agent_speaking is False)


# -- the live Tyto reading attached to every turn ---------------------------- #


def test_live_reading_names_the_cause_and_stays_private():
    from tyto_voice.decision import Scores, live_reading

    noisy = Scores(risk_score=0.62, noise=0.71, speaker_reverb=0.1,
                   speaker_loudness=0.8, interfering_speech=0.05,
                   packet_loss=0.0, codec_degradation=0.0)
    text = live_reading(noisy)
    assert "degraded" in text
    assert "noise high" in text
    assert "interfering speech none" in text
    assert "private" in text.lower()

    clean = Scores(risk_score=0.05, noise=0.05, speaker_reverb=0.1,
                   speaker_loudness=0.8, interfering_speech=0.02,
                   packet_loss=0.0, codec_degradation=0.0)
    assert "clean" in live_reading(clean)


def test_the_reading_rides_along_but_never_enters_history():
    """It describes the room right now. Replaying a stale copy on a later turn
    would be worse than not having it at all."""
    llm = build_llm()
    messages = llm._build_messages("Microphone reading, private: overall clean (0.05).")
    # Exactly one system message: several models on this endpoint reject a
    # second one with "System message must be at the beginning".
    assert [m["role"] for m in messages] == ["system"]
    assert "Microphone reading" in messages[0]["content"]
    assert messages[0]["content"].startswith("SYSTEM")  # the instructions survive
    # Same client, no context: the previous reading is gone.
    assert "Microphone reading" not in llm._build_messages()[0]["content"]


def test_the_provider_reads_scores_fresh_each_turn():
    from tyto_voice.decision import Scores

    provider, controller, tts, llm, ctx, scorer = build_loop()
    current = {"reading": None}
    provider._scores = lambda: current["reading"]

    # Warming up: say so rather than describe a room we have not measured.
    assert "No microphone reading" in provider._reading()

    degraded = Scores(risk_score=0.62, noise=0.71, speaker_reverb=0.1,
                      speaker_loudness=0.8, interfering_speech=0.05,
                      packet_loss=0.0, codec_degradation=0.0)
    current["reading"] = (degraded, 1.0)
    assert "degraded" in provider._reading()

    controller.on_ready()
    tts.finish()
    speak_an_utterance(provider, ctx)
    assert wait_until(lambda: llm.calls == 1)
    assert "degraded" in llm.contexts[-1]


def test_a_stale_reading_is_withheld_rather_than_passed_off_as_now():
    """The bug this guards: Tyto is reset on every agent turn and needs a fresh
    5 s window, so a run of short turns produces no new reading at all. Handing
    over the last one regardless made the agent insist the room was still noisy
    long after the noise had stopped."""
    from tyto_voice.decision import READING_MAX_AGE_SECONDS, Scores

    provider = build_loop()[0]
    degraded = Scores(risk_score=0.62, noise=0.71, speaker_reverb=0.1,
                      speaker_loudness=0.8, interfering_speech=0.05,
                      packet_loss=0.0, codec_degradation=0.0)

    provider._scores = lambda: (degraded, 2.0)
    assert "degraded" in provider._reading()

    provider._scores = lambda: (degraded, READING_MAX_AGE_SECONDS + 1)
    stale = provider._reading()
    assert "No microphone reading" in stale
    assert "degraded" not in stale
    assert "out of date" in stale
    assert "not sure right now" in stale


def test_the_mechanics_never_reach_the_prompt_text():
    """Age gates whether the agent may speak about the room. The user must never
    hear about measuring, windows or seconds."""
    from tyto_voice.decision import NO_READING, Scores, live_reading

    sc = Scores(risk_score=0.62, noise=0.71, speaker_reverb=0.1, speaker_loudness=0.8,
                interfering_speech=0.05, packet_loss=0.0, codec_degradation=0.0)
    for text in (live_reading(sc), NO_READING):
        low = text.lower()
        for leak in ("seconds ago", "measured", "5 s", "window", "warm", "keep talking"):
            assert leak not in low, f"{leak!r} would end up spoken"
    # It must still override what the agent said before.
    assert "replaces anything" in live_reading(sc)
    assert "out of date" in NO_READING


def test_reset_conversation_forgets_history():
    provider, controller, tts, llm, ctx, scorer = build_loop()
    provider.llm = InklingClient("fake-key", instructions="SYSTEM")
    provider.llm.add_agent_line("your mic sounds noisy")
    assert provider.llm._turns
    provider.reset_conversation()
    assert provider.llm._turns == []


# -- barge-in ---------------------------------------------------------------- #


def test_barge_in_is_off_by_default_and_audio_is_dropped_mid_turn():
    provider, controller, tts, llm, ctx, scorer = build_loop()
    controller.on_ready()          # the greeting is speaking
    ctx.speech = True
    provider.send_audio(blocks(200))
    assert tts.clears == 0         # the agent was not cut off
    assert llm.calls == 0


def test_barge_in_cuts_the_agent_off_when_enabled():
    provider, controller, tts, llm, ctx, scorer = build_loop()
    provider._allow_barge_in = True
    controller.on_ready()          # the greeting is speaking
    assert controller.agent_speaking is True

    ctx.speech = True
    provider.send_audio(blocks(int(0.1 * RATE / BLOCK)))   # 100 ms, too short
    assert tts.clears == 0, "must not cut off on a syllable of echo"

    provider.send_audio(blocks(int(0.4 * RATE / BLOCK)))   # now past the floor
    assert tts.clears == 1
    assert controller.agent_speaking is False              # the turn was released

    # The interrupted speech still becomes the next turn.
    ctx.speech = True
    provider.send_audio(blocks(100))
    ctx.speech = False
    provider.send_audio(blocks(end_silence_blocks("patient")))
    assert wait_until(lambda: llm.calls == 1)


def test_barge_in_keeps_the_vad_listening_while_the_mic_is_muted():
    """set_mic_enabled(False) normally drops the utterance in progress. With
    barge-in on it must not, or there is nothing to detect."""
    provider, controller, tts, llm, ctx, scorer = build_loop()
    provider._allow_barge_in = True
    ctx.speech = True
    provider.send_audio(blocks(50))
    before = provider.vad.speech_samples
    provider.set_mic_enabled(False)
    assert provider.vad.speech_samples == before

    provider._allow_barge_in = False
    provider.set_mic_enabled(False)
    assert provider.vad.speech_samples == 0


def test_unmuting_after_the_agent_spoke_discards_what_the_vad_heard():
    """Belt and braces against the echo loop, independent of any setting.

    Whatever the VAD collected while the agent was audible is our own voice
    coming back, not the user. Carrying it into the next turn is how the agent
    ends up answering itself and repeating the same thing over and over.
    """
    provider, controller, tts, llm, ctx, scorer = build_loop()
    provider._allow_barge_in = True          # worst case: the VAD kept listening
    provider.set_mic_enabled(False)
    ctx.speech = True
    provider.send_audio(blocks(50))          # the agent's own voice, echoed back
    assert provider.vad.speech_samples > 0

    provider.set_mic_enabled(True)
    assert provider.vad.speech_samples == 0, "echo must not become the next turn"
    assert provider.vad.speaking is False
