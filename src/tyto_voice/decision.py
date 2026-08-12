"""Tyto scoring contract and decision layer.

This module is the heart of the demo and the one part that is identical across
every branch of this repo (OpenAI WebRTC, ElevenLabs, LiveKit, this Python
reference). It is pure Python with no dependencies, so it is trivial to read,
test, and reuse.

It does two things:

1. Defines the *scoring contract*: the ``Scores`` value object and the tuned
   constants (5 s window, ~2 s hop, EMA alpha 0.3). These match the browser
   reference (``index.html``) byte for byte so behavior is comparable.

2. Defines the *decision layer*: given a smoothed ``Scores`` reading it answers
   three questions, one per adaptation layer:
     - Aware:    what one-sentence room note should the agent know about?
     - Tuned:    eager or patient turn-taking?
     - Reactive: should we nudge the user right now, and about what?

Nothing here talks to a voice provider or to the SDK. The controller wires this
into a live agent; the scorer feeds it live numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

# --------------------------------------------------------------------------- #
# Scoring contract                                                            #
# --------------------------------------------------------------------------- #

# The six explanatory dimensions Tyto 1.1 returns, in the docs' display order.
# Tyto 1.1 merged the old ``media_speech`` into ``interfering_speech`` (any
# competing speech, live or from a TV / radio / phone) and added
# ``codec_degradation``.
ENV_KEYS = (
    "noise",
    "speaker_reverb",
    "speaker_loudness",
    "interfering_speech",
    "packet_loss",
    "codec_degradation",
)

LABELS = {
    "noise": "Noise",
    "speaker_reverb": "Speaker Reverb",
    "speaker_loudness": "Speaker Loudness",
    "interfering_speech": "Interfering Speech",
    "packet_loss": "Packet Loss",
    "codec_degradation": "Codec Degradation",
}

# Tuned constants. Keep these identical across branches for comparability.
WINDOW_SECONDS = 5.0  # Tyto's analysis window is fixed at 5 s by the model.
# How often we slide that window and read a new score (UI + decision cadence).
# Tyto 1.1 analyses a window in about 100 ms, roughly a quarter of what 1.0
# needed, so a 1 s hop costs ~10% of one core and the UI moves visibly.
HOP_SECONDS = 1.0
# Smoothing of successive analyze() reads; the docs recommend 0.3. Note that 0.3
# at a 1 s hop takes about as long to cross the nudge gate as the old 0.5 at a
# 2 s hop, so the extra points do not make the agent quicker to nudge.
SCORE_EMA_ALPHA = 0.3


@dataclass(frozen=True)
class Scores:
    """One Tyto reading: the headline risk score plus six dimensions, all 0..1."""

    risk_score: float
    noise: float
    speaker_reverb: float
    speaker_loudness: float
    interfering_speech: float
    packet_loss: float
    codec_degradation: float

    @classmethod
    def from_result(cls, result) -> "Scores":
        """Build from an ``aic_sdk`` AnalysisResult (or anything with the same attrs)."""
        return cls(
            risk_score=result.risk_score,
            noise=result.noise,
            speaker_reverb=result.speaker_reverb,
            speaker_loudness=result.speaker_loudness,
            interfering_speech=result.interfering_speech,
            packet_loss=result.packet_loss,
            codec_degradation=result.codec_degradation,
        )

    def ema(self, previous: "Scores | None", alpha: float = SCORE_EMA_ALPHA) -> "Scores":
        """Exponential moving average against the previous smoothed reading.

        Returns ``self`` (no smoothing) when there is no history yet.
        """
        if previous is None:
            return self
        a = min(1.0, max(0.0, alpha))
        return Scores(
            **{
                f.name: a * getattr(self, f.name) + (1 - a) * getattr(previous, f.name)
                for f in fields(self)
            }
        )

    def as_dict(self) -> dict[str, float]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


# --------------------------------------------------------------------------- #
# Thresholds and bands                                                        #
# --------------------------------------------------------------------------- #

# Per-dimension [green, yellow] cutoffs for coloring (lower is better). The
# yellow cutoff doubles as each dimension's "red" line for the decision rules.
# The docs publish bands for the risk score only and recommend calibrating the
# rest against your own traffic, so these are demo defaults: codec_degradation
# uses the documented risk bands, the others are the demo's tuned values.
THRESHOLDS = {
    "noise": (0.20, 0.45),
    "interfering_speech": (0.15, 0.35),
    "packet_loss": (0.05, 0.15),
    "codec_degradation": (0.30, 0.50),
    "speaker_reverb": (0.25, 0.55),
    "speaker_loudness": (0.12, 0.25),
}

# Loudness and reverb are informational only: never colored as a problem,
# never named as a cause, never the reason for a nudge.
NO_POLARITY = frozenset({"speaker_loudness", "speaker_reverb"})

# A dimension must be at least this elevated before it can be named as the cause.
MIN_EXPLANATION_VALUE = 0.30

# Tyto Risk Score bands from the docs: <0.30 good, 0.30-0.50 warn, >0.50 bad.
COMPOSITE_TH = (0.30, 0.50)
COMPOSITE_NUDGE = 0.50  # boundary of the "bad" band, used for wording
COMPOSITE_CLEAR = 0.30  # below this the episode is considered over (hysteresis)

# Risk Score at/above which a nudge may fire (when a cause also dominates).
# Decoupled from the bands so it can be tuned without shifting them.
NUDGE_THRESHOLD_DEFAULT = 0.40
NUDGE_THRESHOLD_MIN = COMPOSITE_CLEAR
NUDGE_THRESHOLD_MAX = COMPOSITE_NUDGE

# Turn-detection profiles handed to the voice provider (Layer 2).
# Eager: snappy semantic VAD. Patient: longer end-of-speech in a noisy room so
# the agent stops triggering on background sound.
VAD_PROFILES = {
    "eager": {
        "type": "semantic_vad",
        "eagerness": "auto",
        "create_response": True,
        "interrupt_response": True,
    },
    "patient": {
        "type": "server_vad",
        "threshold": 0.55,
        "prefix_padding_ms": 400,
        "silence_duration_ms": 900,
        "create_response": True,
        "interrupt_response": True,
    },
}

# Per-cause spoken nudge text (``text``), the short phrase for the Aware room
# note (``room``), and the standing instruction that goes with it (``advice``),
# following the per-dimension real-time guidance in the Tyto docs. Order is
# priority order for ties.
#
# ``text`` is None for causes the user cannot act on: transport problems are
# still worth telling the agent about (Aware), but asking the speaker to fix
# them would be nonsense, so they never trigger a spoken nudge (Reactive).
EXPLANATIONS = (
    {
        "key": "noise",
        "thr": THRESHOLDS["noise"][1],
        "text": "Sorry, there is a lot of background noise. Could you move somewhere quieter?",
        "room": "loud background noise",
        "advice": (
            "Be patient with possible misunderstandings and confirm any critical "
            "details by repeating them back to the user."
        ),
    },
    {
        "key": "packet_loss",
        "thr": THRESHOLDS["packet_loss"][1],
        "text": "Sorry, your connection seems unstable. Could you check it and try again?",
        "room": "an unstable connection with audio dropouts",
        "advice": (
            "Whole words may be missing, so confirm names, numbers and addresses by "
            "repeating them back, and allow longer pauses before assuming the user "
            "has finished."
        ),
    },
    {
        # Tyto 1.1 reports one interfering-speech dimension for live speakers and
        # media alike, and the docs advise not naming a source we cannot identify.
        "key": "interfering_speech",
        "thr": THRESHOLDS["interfering_speech"][1],
        "text": (
            "Sorry, I am hearing other voices in the background. Could you move somewhere "
            "quieter, or turn down anything playing nearby?"
        ),
        "room": "other voices in the background, either people nearby or something playing",
        "advice": (
            "Those other voices can end up transcribed as the user, so confirm anything "
            "unexpected before acting on it."
        ),
    },
    {
        "key": "codec_degradation",
        "thr": THRESHOLDS["codec_degradation"][1],
        "text": None,  # transport problem: nothing the speaker can do about it
        "room": "heavy codec compression on the connection",
        "advice": (
            "This one is in the transport, not the room, so never ask the user to change "
            "their surroundings; just confirm names, numbers and addresses by repeating "
            "them back."
        ),
    },
)

# Any of these over the trip level -> patient turn-taking (Layer 2).
NOISY_KEYS = ("noise", "interfering_speech")
NOISY_TRIP = 0.45


# --------------------------------------------------------------------------- #
# Decision functions                                                          #
# --------------------------------------------------------------------------- #


def strongest_cause(scores: Scores, actionable_only: bool = False) -> dict | None:
    """The single dominant problem, or None if nothing is clearly elevated.

    A dimension qualifies only when it is past its red cutoff *and* above
    MIN_EXPLANATION_VALUE. Severity is how far past the cutoff it is, so the
    most-exceeded dimension wins.

    ``actionable_only`` restricts the search to causes the user can do something
    about, which is what the Reactive layer needs.
    """
    best, best_severity = None, 0.0
    for rule in EXPLANATIONS:
        if actionable_only and not rule["text"]:
            continue
        value = getattr(scores, rule["key"])
        if value <= rule["thr"] or value < MIN_EXPLANATION_VALUE:
            continue
        severity = (value - rule["thr"]) / max(1e-6, 1 - rule["thr"])
        if severity > best_severity:
            best = {**rule, "value": value, "severity": severity}
            best_severity = severity
    return best


def room_state_summary(scores: Scores) -> str:
    """The one-sentence Aware room note, or "" when the room sounds clean.

    Says nothing at all unless one cause clearly dominates, so the agent is not
    fed vague acoustic chatter.
    """
    cause = strongest_cause(scores)
    if not cause:
        return ""
    risk = scores.risk_score
    if risk >= COMPOSITE_NUDGE:
        severity = "degraded"
    elif risk >= COMPOSITE_TH[0]:
        severity = "marginal"
    else:
        severity = "borderline"
    return f"Audio note: {severity} input, {cause['room']}. {cause['advice']}"


def pick_vad_profile(scores: Scores) -> str:
    """"eager" in a quiet room, "patient" when background activity is high."""
    if any(getattr(scores, k) >= NOISY_TRIP for k in NOISY_KEYS):
        return "patient"
    return "eager"


@dataclass
class Nudge:
    """A Reactive directive: speak ``text`` because ``label`` is at ``value``."""

    key: str
    label: str
    value: float
    text: str


class EnvMonitor:
    """Fires a nudge when the smoothed risk is high AND one cause persists.

    A red risk score alone never fires; there must always be a dominant cause,
    and it must be one the user can act on (so codec degradation, a transport
    problem, informs the agent but is never spoken as a nudge). After firing, the
    cause's streak resets, so it takes another full run of bad windows to
    re-fire. No timers: ``min_persist`` is counted in scored windows.
    """

    def __init__(self, min_persist: int = 1, threshold: float = NUDGE_THRESHOLD_DEFAULT):
        self.min_persist = min_persist
        self.threshold = threshold  # live-adjustable risk gate
        self._streak: dict[str, int] = {}

    def evaluate(self, scores: Scores) -> Nudge | None:
        if scores.risk_score < COMPOSITE_CLEAR:
            self._streak = {}
            return None
        cause = strongest_cause(scores, actionable_only=True)
        if not cause:
            self._streak = {}
            return None
        key = cause["key"]
        self._streak[key] = self._streak.get(key, 0) + 1
        for other in self._streak:
            if other != key:
                self._streak[other] = 0
        if self._streak[key] < self.min_persist:
            return None
        if scores.risk_score < self.threshold:
            return None
        self._streak[key] = 0  # re-arm
        return Nudge(key=key, label=LABELS[key], value=cause["value"], text=cause["text"])
