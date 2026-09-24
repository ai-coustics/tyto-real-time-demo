"""Test the scorer's block accumulation without the SDK or a microphone.

feed() must emit exactly block_size-sized mono buffers regardless of the caller's
block size, and must drop audio while paused.
"""

import numpy as np

from tyto_voice.scorer import LiveTytoScorer


class FakeCollector:
    def __init__(self):
        self.sizes = []

    def buffer(self, block):
        # aic-sdk 3.x buffers mono: one 1D float32 array of exactly block_size.
        assert block.ndim == 1
        assert block.dtype == np.float32
        self.sizes.append(block.shape[0])


def make_scorer(block_size=160):
    scorer = LiveTytoScorer("dummy")
    scorer.block_size = block_size
    scorer._collector = FakeCollector()
    return scorer


def test_feed_emits_fixed_size_blocks():
    scorer = make_scorer(block_size=160)
    # Feed 500 samples in odd-sized chunks; expect 3 full 160-sample blocks.
    scorer.feed(np.zeros(100, dtype=np.float32))
    scorer.feed(np.zeros(100, dtype=np.float32))
    scorer.feed(np.zeros(100, dtype=np.float32))
    scorer.feed(np.zeros(200, dtype=np.float32))
    sizes = scorer._collector.sizes
    assert sizes == [160, 160, 160]  # 500 buffered, 480 emitted, 20 residual
    assert scorer._buffered == 480


def test_feed_dropped_while_paused():
    scorer = make_scorer()
    scorer.pause()
    scorer.feed(np.zeros(320, dtype=np.float32))
    assert scorer._collector.sizes == []
    assert scorer._buffered == 0


class FakeResult:
    def __init__(self, v):
        for k in ("risk_score", "noise", "speaker_reverb", "speaker_loudness",
                  "interfering_speech", "packet_loss", "codec_degradation"):
            setattr(self, k, v)


class FakeAnalyzer:
    def __init__(self, values, on_analyze=None):
        self.values, self.on_analyze = list(values), on_analyze
        self.resets = self.terminated = 0

    def analyze_buffered(self):
        if self.on_analyze:
            self.on_analyze()
        return FakeResult(self.values.pop(0))

    def reset(self):
        self.resets += 1

    def terminate_session(self):
        self.terminated += 1


def run_loop_once(scorer):
    """Drive one iteration of _loop without the timer thread."""
    stop_after = iter([False, True])
    scorer._stop.wait = lambda _t: next(stop_after)
    scorer._loop()


def scored(values, on_analyze=None):
    scorer = make_scorer()
    scorer._analyzer = FakeAnalyzer(values, on_analyze)
    scorer._window_samples = 0
    out = []
    scorer.on_scores = out.append
    return scorer, out


def test_resume_clears_smoothing_so_a_fixed_problem_does_not_renudge():
    scorer, out = scored([0.8, 0.1])
    run_loop_once(scorer)
    scorer.resume()
    scorer._window_samples = 0  # pretend the fresh window is full
    run_loop_once(scorer)
    # Without the reset the EMA would read 0.3*0.1 + 0.7*0.8 = 0.59.
    assert [round(s.noise, 2) for s in out] == [0.8, 0.1]


def test_window_reset_mid_analysis_is_dropped():
    scorer, out = scored([0.8], on_analyze=lambda: scorer.resume())
    run_loop_once(scorer)
    assert out == []


def test_stop_terminates_the_session():
    scorer, _ = scored([])
    analyzer = scorer._analyzer
    scorer.stop()
    assert analyzer.terminated == 1 and scorer._analyzer is None
