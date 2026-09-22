"""Controller wiring tests with a fake provider and scorer (no hardware).

These check the provider-agnostic glue: that scores drive the three layers and
that the mute/nudge/resume state machine gates scoring correctly.
"""

from tyto_voice.controller import TytoController
from tyto_voice.provider import VoiceProvider


class FakeProvider(VoiceProvider):
    def __init__(self):
        super().__init__(handlers=None)
        self.calls = []

    def connect(self): ...
    def disconnect(self): ...
    def set_instructions(self, text): self.calls.append(("instructions", text))
    def set_turn_detection(self, td): self.calls.append(("turn_detection", td))
    def set_mic_enabled(self, on): self.calls.append(("mic", on))
    def interrupt(self, clear_input=False): self.calls.append(("interrupt", clear_input))
    def nudge(self, text): self.calls.append(("nudge", text))
    def request_response(self): self.calls.append(("request_response", None))
    def send_tool_result(self, call_id, output): self.calls.append(("tool_result", call_id))

    def kinds(self):
        return [c[0] for c in self.calls]


class FakeScorer:
    def __init__(self):
        self.scoring = True
        self.events = []

    def pause(self):
        self.scoring = False
        self.events.append("pause")

    def resume(self):
        self.scoring = True
        self.events.append("resume")


def make(**overrides):
    from tyto_voice.decision import Scores

    base = dict(
        risk_score=0.0, noise=0.0, speaker_reverb=0.0, speaker_loudness=0.0,
        interfering_speech=0.0, packet_loss=0.0, codec_degradation=0.0,
    )
    base.update(overrides)
    return Scores(**base)


def build():
    provider = FakeProvider()
    scorer = FakeScorer()
    controller = TytoController(provider, scorer)
    controller.set_connected(True)
    return provider, scorer, controller


def test_aware_pushes_room_note_then_clears():
    provider, _, controller = build()
    controller.on_scores(make(risk_score=0.7, interfering_speech=0.8))
    assert any(k == "instructions" and "other voices" in v for k, v in provider.calls)
    provider.calls.clear()
    controller.on_scores(make(risk_score=0.1))  # clean again
    assert any(k == "instructions" for k in provider.kinds())  # instructions reset


def test_tuned_swaps_turn_detection_on_noise():
    provider, _, controller = build()
    # Noisy room but risk below the clear band, so Tuned acts without a nudge.
    controller.on_scores(make(risk_score=0.2, noise=0.6))  # noisy -> patient
    tds = [v for k, v in provider.calls if k == "turn_detection"]
    assert tds and tds[-1]["type"] == "server_vad"


def test_reactive_nudge_mutes_interrupts_and_dispatches():
    provider, scorer, controller = build()
    controller.on_scores(make(risk_score=0.7, interfering_speech=0.8))
    kinds = provider.kinds()
    assert "interrupt" in kinds and "nudge" in kinds
    assert controller.awaiting_nudge is True
    assert ("mic", False) in provider.calls  # mic muted for the nudge


def test_nudge_lifecycle_resumes_listening():
    provider, scorer, controller = build()
    controller.on_scores(make(risk_score=0.7, interfering_speech=0.8))
    assert controller.awaiting_nudge
    # Agent starts speaking the nudge, then finishes with no audio left to play.
    controller.on_agent_speaking(True, nudge=True)
    assert controller.nudge_active
    controller.on_agent_speaking(False, nudge=True, cancelled=False)
    # Back to listening and scoring gated back on.
    assert controller.listening is True
    assert scorer.scoring is True


def test_scoring_pauses_while_agent_speaks():
    _, scorer, controller = build()
    controller.on_agent_speaking(True)  # ordinary reply, not a nudge
    assert scorer.scoring is False
    controller.on_agent_speaking(False)
    controller.on_agent_audio(False)
    assert scorer.scoring is True


def test_audio_quality_snapshot_summarizes_top_issue():
    _, _, controller = build()
    controller.on_scores(make(risk_score=0.7, interfering_speech=0.8))
    snap = controller.audio_quality_snapshot()
    assert snap["verdict"] == "degraded"
    assert snap["top_issue"]["key"] == "interfering_speech"


# -- the judge (Jev) ---------------------------------------------------------- #

from tyto_voice.jev import ASK_AFTER_SENTENCE, ASK_NOW, STAY_SILENT, Decision  # noqa: E402


class FakeJudge:
    model = "fake-jev"

    def __init__(self):
        self.asked = []
        self.busy = False

    def ask(self, situation, callback):
        self.asked.append((situation, callback))
        return True

    def answer(self, action, **kw):
        situation, callback = self.asked[-1]
        callback(
            Decision(action=action, confidence=kw.get("confidence", 0.9), reason=action, latency_ms=300, source=kw.get("source", "jev")),
            situation,
        )


class Clock:
    t = 1000.0

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


def build_judged():
    provider, scorer, judge, clock, updates = FakeProvider(), FakeScorer(), FakeJudge(), Clock(), []
    controller = TytoController(provider, scorer, judge=judge, on_update=updates.append, clock=clock)
    controller.set_connected(True)
    return provider, judge, controller, clock, updates


def test_judge_is_consulted_and_ask_now_fires_the_nudge():
    provider, judge, controller, _, _ = build_judged()
    controller.on_scores(make(risk_score=0.7, interfering_speech=0.8))
    assert judge.asked and "nudge" not in provider.kinds()  # gate tripped, verdict pending
    judge.answer(ASK_NOW)
    assert "interrupt" in provider.kinds() and "nudge" in provider.kinds()
    assert controller.awaiting_nudge


def test_judge_wait_for_sentence_defers_until_the_agent_is_idle():
    provider, judge, controller, _, _ = build_judged()
    controller.on_agent_speaking(True)
    controller.on_scores(make(risk_score=0.7, noise=0.8))
    assert judge.asked[-1][0].agent_speaking is True
    judge.answer(ASK_AFTER_SENTENCE)
    assert "nudge" not in provider.kinds()
    controller.on_agent_speaking(False)  # sentence over, nothing left to play
    assert "nudge" in provider.kinds()


def test_judge_stay_silent_suppresses_the_nudge_and_reports_the_verdict():
    provider, judge, controller, _, updates = build_judged()
    controller.on_scores(make(risk_score=0.7, noise=0.8))
    judge.answer(STAY_SILENT)
    assert "nudge" not in provider.kinds()
    assert any("jev" in u and u["jev"]["action"] == STAY_SILENT and u["jev"]["cause"] == "Noise" for u in updates)


def test_judge_fallback_fires_the_rule():
    provider, judge, controller, _, _ = build_judged()
    controller.on_scores(make(risk_score=0.7, noise=0.8))
    judge.answer(ASK_NOW, source="fallback")
    assert "nudge" in provider.kinds()


def test_fresh_verdict_makes_the_trip_instant():
    provider, judge, controller, clock, _ = build_judged()
    controller.on_scores(make(risk_score=0.35, noise=0.6))  # warn band: consulted, gate not tripped
    assert judge.asked and "nudge" not in provider.kinds()
    judge.answer(ASK_NOW)
    assert "nudge" not in provider.kinds()
    clock.advance(1.0)
    controller.on_scores(make(risk_score=0.7, noise=0.8))  # trips: acts on the cached verdict at once
    assert "nudge" in provider.kinds()


def test_situation_carries_transcript_context_and_ask_history():
    provider, judge, controller, clock, _ = build_judged()
    controller.on_agent_transcript("your order number is four seven two nine", True)
    controller.on_user_transcript("hang on", True)
    controller.on_scores(make(risk_score=0.7, noise=0.8))
    s = judge.asked[-1][0]
    assert s.cause == "noise" and s.severity == "severe" and s.since_ask_s is None and s.times_asked == 0
    assert "four seven two nine" in s.agent_last_words and s.caller_last_words == "hang on" and s.caller_speaking
    judge.answer(ASK_NOW)
    controller.on_agent_speaking(True, nudge=True)
    controller.on_agent_speaking(False, nudge=True)
    clock.advance(12.0)
    controller.on_scores(make(risk_score=0.7, noise=0.8))
    s2 = judge.asked[-1][0]
    assert s2.times_asked == 1 and 11 < s2.since_ask_s < 13 and not s2.caller_speaking
