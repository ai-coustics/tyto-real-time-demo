"""Tests for the scoring contract and decision layer.

Pure Python, no SDK or audio hardware required. These pin the behavior that must
stay identical across every branch of the demo.
"""

from tyto_voice.decision import (
    COMPOSITE_CLEAR,
    NUDGE_COOLDOWN_WINDOWS,
    NUDGE_THRESHOLD_DEFAULT,
    SCORE_EMA_ALPHA,
    VAD_PROFILES,
    EnvMonitor,
    Scores,
    pick_vad_profile,
    room_state_summary,
    strongest_cause,
)


def make(**overrides) -> Scores:
    base = dict(
        risk_score=0.0,
        noise=0.0,
        speaker_reverb=0.0,
        speaker_loudness=0.0,
        interfering_speech=0.0,
        packet_loss=0.0,
        codec_degradation=0.0,
    )
    base.update(overrides)
    return Scores(**base)


# -- scoring contract ------------------------------------------------------- #


def test_ema_no_history_returns_self():
    s = make(risk_score=0.8)
    assert s.ema(None) is s


def test_ema_blends_with_alpha_half():
    prev = make(risk_score=0.2, noise=0.2)
    cur = make(risk_score=0.8, noise=0.4)
    blended = cur.ema(prev, alpha=0.5)
    assert abs(blended.risk_score - 0.5) < 1e-9
    assert abs(blended.noise - 0.3) < 1e-9


def test_ema_default_alpha_matches_docs_recommendation():
    prev = make(risk_score=0.0)
    cur = make(risk_score=1.0)
    assert SCORE_EMA_ALPHA == 0.3
    assert abs(cur.ema(prev).risk_score - 0.3) < 1e-9


def test_scores_carry_the_six_tyto_dimensions():
    keys = set(make().as_dict())
    assert keys == {
        "risk_score",
        "noise",
        "speaker_reverb",
        "speaker_loudness",
        "interfering_speech",
        "packet_loss",
        "codec_degradation",
    }


# -- strongest_cause -------------------------------------------------------- #


def test_no_cause_when_clean():
    assert strongest_cause(make(risk_score=0.1, noise=0.1)) is None


def test_loudness_and_reverb_never_named():
    # Both are informational only, so even when high they are never the cause.
    assert strongest_cause(make(speaker_loudness=0.9, speaker_reverb=0.9)) is None


def test_most_exceeded_dimension_wins():
    # interfering_speech is further past its red cutoff (0.35) than noise is past 0.45.
    cause = strongest_cause(make(noise=0.5, interfering_speech=0.9))
    assert cause is not None and cause["key"] == "interfering_speech"


def test_below_min_explanation_value_is_ignored():
    # packet_loss past its 0.15 cutoff but under MIN_EXPLANATION_VALUE (0.30).
    assert strongest_cause(make(packet_loss=0.25)) is None


def test_codec_degradation_is_a_cause_but_never_actionable():
    bad = make(codec_degradation=0.9)
    cause = strongest_cause(bad)
    assert cause is not None and cause["key"] == "codec_degradation"
    assert cause["text"] is None
    assert strongest_cause(bad, actionable_only=True) is None


# -- room note (Aware) ------------------------------------------------------ #


def test_room_note_empty_when_clean():
    assert room_state_summary(make(risk_score=0.2)) == ""


def test_room_note_mentions_cause_and_severity():
    note = room_state_summary(make(risk_score=0.7, interfering_speech=0.8))
    assert "Audio note:" in note and "degraded" in note and "other voices" in note


def test_room_note_covers_codec_degradation_with_transport_advice():
    note = room_state_summary(make(risk_score=0.7, codec_degradation=0.9))
    assert "codec compression" in note
    assert "never ask the user to change their surroundings" in note


# -- turn-taking (Tuned) ---------------------------------------------------- #


def test_vad_eager_when_quiet():
    assert pick_vad_profile(make(noise=0.1)) == "eager"


def test_vad_patient_when_noisy():
    assert pick_vad_profile(make(noise=0.6)) == "patient"


def test_vad_patient_on_interfering_speech():
    assert pick_vad_profile(make(interfering_speech=0.6)) == "patient"


def test_vad_eager_on_transport_problems():
    # Dropouts and codec artifacts are not background activity: turn-taking
    # profiles do not change for them.
    assert pick_vad_profile(make(packet_loss=0.9, codec_degradation=0.9)) == "eager"


# -- nudge (Reactive) ------------------------------------------------------- #


def test_no_nudge_below_clear_band():
    m = EnvMonitor(min_persist=1, threshold=NUDGE_THRESHOLD_DEFAULT)
    assert m.evaluate(make(risk_score=COMPOSITE_CLEAR - 0.01, interfering_speech=0.9)) is None


def test_no_nudge_without_a_cause():
    # High risk but no dominant cause: a red score alone never fires.
    m = EnvMonitor(min_persist=1)
    assert m.evaluate(make(risk_score=0.9)) is None


def test_no_nudge_for_codec_degradation_alone():
    # The user cannot fix the codec, so it informs the agent but never nudges.
    m = EnvMonitor(min_persist=1, threshold=0.40)
    assert m.evaluate(make(risk_score=0.9, codec_degradation=0.9)) is None


def test_nudge_fires_with_risk_and_cause():
    m = EnvMonitor(min_persist=1, threshold=0.40)
    nudge = m.evaluate(make(risk_score=0.7, interfering_speech=0.8))
    assert nudge is not None and nudge.key == "interfering_speech"


def test_min_persist_one_fires_on_the_first_bad_window():
    # A streak of one is already a full streak, so it fires immediately. This is
    # the tuning this branch ships: the Reactive layer is meant to be quick.
    m = EnvMonitor(min_persist=1, threshold=0.40, cooldown=0)
    bad = make(risk_score=0.7, interfering_speech=0.8)
    assert m.evaluate(bad) is not None
    assert m.evaluate(bad) is not None


def test_persistence_and_rearm_with_min_persist_two():
    m = EnvMonitor(min_persist=2, threshold=0.40, cooldown=0)
    bad = make(risk_score=0.7, interfering_speech=0.8)
    assert m.evaluate(bad) is None  # window 1: not yet persistent
    assert m.evaluate(bad) is not None  # window 2: fires, then re-arms
    assert m.evaluate(bad) is None  # window 3: streak rebuilding
    assert m.evaluate(bad) is not None  # window 4: fires again


def test_cooldown_silences_the_windows_after_a_nudge():
    # Without this, min_persist=1 plus continuous scoring is a nudge every hop.
    m = EnvMonitor(min_persist=1, threshold=0.40, cooldown=3)
    bad = make(risk_score=0.7, interfering_speech=0.8)
    assert m.evaluate(bad) is not None
    assert [m.evaluate(bad) for _ in range(3)] == [None, None, None]
    assert m.evaluate(bad) is not None


def test_default_cooldown_is_long_enough_to_not_machine_gun():
    m = EnvMonitor()
    bad = make(risk_score=0.7, interfering_speech=0.8)
    assert m.evaluate(bad) is not None
    assert [m.evaluate(bad) for _ in range(NUDGE_COOLDOWN_WINDOWS)] == [None] * NUDGE_COOLDOWN_WINDOWS
    assert m.evaluate(bad) is not None  # the window after the cooldown re-arms


def test_flux_profiles_are_within_deepgram_ranges():
    eager, patient = VAD_PROFILES["eager"], VAD_PROFILES["patient"]
    for profile in (eager, patient):
        assert 0.5 <= profile["eot_threshold"] <= 1.0
        assert 500 <= profile["eot_timeout_ms"] <= 60000
    # Eager speculates, patient does not, and Flux rejects an eager threshold
    # above the committed one.
    assert 0.3 <= eager["eager_eot_threshold"] <= eager["eot_threshold"]
    assert patient["eager_eot_threshold"] is None
    # Patient must actually be more patient, or Layer 2 is doing nothing.
    assert patient["eot_threshold"] > eager["eot_threshold"]
    assert patient["eot_timeout_ms"] > eager["eot_timeout_ms"]


def test_nudge_gated_by_threshold():
    m = EnvMonitor(min_persist=1, threshold=0.50)
    # Cause present and over clear band, but risk under the (raised) gate.
    assert m.evaluate(make(risk_score=0.45, interfering_speech=0.8)) is None
