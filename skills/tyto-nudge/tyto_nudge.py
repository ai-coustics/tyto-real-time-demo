"""Tyto nudge policy: turn live Tyto readings into one spoken line for the user.

Drop this file into any voice stack. It has no dependencies and does not touch
audio or the agent: you call ``on_result`` with each Tyto reading, and when it
returns a string, your integration interrupts the agent and has it say that
string. Everything framework-specific stays in your code.

The constants match the reference demo (github.com/ai-coustics/tyto-real-time-demo,
``src/tyto_voice/decision.py``). They are demo defaults: calibrate on your traffic.
"""

from __future__ import annotations

import time
from typing import Callable

# Only dimensions the user can act on get a line. Order breaks ties.
# key: (red cutoff, what the agent says)
CAUSES = {
    "noise": (
        0.45,
        "Sorry, there is a lot of background noise. Could you move somewhere quieter?",
    ),
    "packet_loss": (
        0.15,
        "Sorry, your connection seems unstable. Could you check it and try again?",
    ),
    "interfering_speech": (
        0.35,
        "Sorry, I am hearing other voices in the background. Could you move somewhere "
        "quieter, or turn down anything playing nearby?",
    ),
}
# Never add speaker_loudness or speaker_reverb (informational) or
# codec_degradation (transport, the user cannot fix it).

FIELDS = ("risk_score", *CAUSES)

RISK_GATE = 0.40  # smoothed risk_score at/above which a nudge may fire
RISK_CLEAR = 0.30  # below this the episode is over and every cause re-arms
MIN_CAUSE_VALUE = 0.30  # a dimension must be at least this high to be named
EMA_ALPHA = 0.3  # the Tyto docs' starting point
COOLDOWN_S = 30.0  # never repeat the same line sooner than this


class TytoNudger:
    def __init__(
        self,
        *,
        gate: float = RISK_GATE,
        alpha: float = EMA_ALPHA,
        cooldown_s: float = COOLDOWN_S,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.gate, self.alpha, self.cooldown_s, self._clock = gate, alpha, cooldown_s, clock
        self._last_said: dict[str, float] = {}
        self.reset()

    def reset(self) -> None:
        """Call on every analyzer.reset() (stream discontinuity, new call)."""
        self.smoothed: dict[str, float] | None = None
        self._fired: set[str] = set()

    def on_result(self, result) -> str | None:
        """Feed one Tyto reading (an aic_sdk AnalysisResult or a dict). Returns a line or None."""
        raw = {k: float(result[k] if isinstance(result, dict) else getattr(result, k)) for k in FIELDS}
        prev = self.smoothed
        self.smoothed = raw if prev is None else {
            k: self.alpha * raw[k] + (1 - self.alpha) * prev[k] for k in FIELDS
        }
        s = self.smoothed

        if s["risk_score"] < RISK_CLEAR:
            self._fired.clear()
            return None
        if s["risk_score"] < self.gate:
            return None

        cause = self.strongest_cause(s)
        if cause is None or cause in self._fired:
            return None
        now = self._clock()
        if now - self._last_said.get(cause, float("-inf")) < self.cooldown_s:
            return None
        self._fired.add(cause)
        self._last_said[cause] = now
        return CAUSES[cause][1]

    @staticmethod
    def strongest_cause(s: dict[str, float]) -> str | None:
        """The dimension furthest past its cutoff, or None if none clearly dominates."""
        best, best_sev = None, 0.0
        for key, (cutoff, _line) in CAUSES.items():
            value = s[key]
            if value <= cutoff or value < MIN_CAUSE_VALUE:
                continue
            severity = (value - cutoff) / (1 - cutoff)
            if severity > best_sev:
                best, best_sev = key, severity
        return best
