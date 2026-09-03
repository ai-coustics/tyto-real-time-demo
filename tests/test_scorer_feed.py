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
