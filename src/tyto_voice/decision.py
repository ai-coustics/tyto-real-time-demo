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
# The window cannot be shortened, but it can be slid faster, and the hop is what
# the demo actually feels: it sets how quickly the meters move and how quickly
# Aware and Tuned respond.
#
# Measured on this machine, analyze_buffered() takes 116 ms, so:
#     1.00 s hop -> 12% of one core, 1 reading per second
#     0.50 s hop -> 23% of one core, 2 readings per second
#     0.25 s hop -> 46% of one core, 4 readings per second
# 0.5 s is the point where it starts to feel live without the analyzer becoming
# a meaningful share of the machine. This is a deliberate divergence from the
# browser reference, which uses 1 s; see NUDGE_MIN_PERSIST for what it costs.
HOP_SECONDS = 0.5
# Smoothing of successive analyze() reads; the docs recommend 0.3. The EMA time
# constant is roughly hop / alpha, so halving the hop also halves the smoothing
# lag: about 1.7 s here against 3.3 s at a 1 s hop.
SCORE_EMA_ALPHA = 0.3
# Scored windows a cause must dominate before the Reactive layer will fire.
# This exists to keep the nudge gate at the same wall-clock sensitivity as the
# slower hop: 2 windows at 0.5 s is the same one second of sustained evidence
# that 1 window at 1 s used to be. Without it, doubling the hop rate would
# silently make the agent twice as quick to interrupt people.
NUDGE_MIN_PERSIST = 2


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

# Older than this and a reading describes a room that may no longer exist, so it
# is withheld entirely. Tyto resets on every agent turn and needs a fresh 5 s
# window, so short turns can leave the newest reading minutes behind the room.
# The age is never surfaced to the user: it decides whether the agent may speak
# about the room at all, and nothing else. Keep this tight, because a reading
# that survives the gate is stated flatly as current, with no hedge attached.
READING_MAX_AGE_SECONDS = 20.0

# Risk Score at/above which a nudge may fire (when a cause also dominates).
# Decoupled from the bands so it can be tuned without shifting them.
NUDGE_THRESHOLD_DEFAULT = 0.40
NUDGE_THRESHOLD_MIN = COMPOSITE_CLEAR
NUDGE_THRESHOLD_MAX = COMPOSITE_NUDGE

# Turn-detection profiles handed to the voice provider (Layer 2).
# Eager: snappy turns. Patient: harder to trigger and slower to end the turn, so
# a noisy room stops ending the user's sentences for them.
#
# These are the one place this file is NOT identical to the browser reference.
# The browser drives OpenAI Realtime, whose turn detection lives on the server
# and is configured with semantic_vad / server_vad dicts. This branch does its
# own turn-taking with the ai-coustics VAD, so the profiles are that model's
# parameters instead. The layer, the two profile names, and when they swap are
# unchanged, which is what keeps the demos comparable.
#
# Keys map to aic_sdk.VadParameter members:
#   sensitivity             speech-probability threshold, 0..1. Higher = more
#                           confidence needed, so background sound stops counting
#                           as speech.
#   minimum_speech_duration seconds of speech before a turn starts. Guards the
#                           silence -> speech edge against clicks and coughs.
#   speech_hold_duration    seconds of speech reported after the audio goes
#                           quiet, as a rolling majority over the last
#                           (hold * 2) seconds. Kept short here: it smooths the
#                           signal, it does not decide the end of the turn.
#
# ``end_silence`` is ours, not the SDK's, and it is what actually ends a turn.
#
# It has to exist. Measured against real speech, is_speech_detected() drops out
# for 45 to 285 ms at ordinary pauses inside a single sentence, and raising
# speech_hold_duration does not close those gaps (it is a rolling majority, so a
# longer window can make them longer). Ending the turn on the falling edge split
# one 4.8 s question into three utterances. So we require a continuous run of
# silence instead, comfortably longer than the worst gap observed.
#
# The values carry half a second of grace on top of that worst gap, so thinking
# mid-sentence does not end your turn. It is the direct trade against how quickly
# the agent comes back: every 0.1 s here is 0.1 s of dead air on every turn.
VAD_PROFILES = {
    "eager": {
        "sensitivity": 0.50,
        "minimum_speech_duration": 0.06,
        "speech_hold_duration": 0.10,
        "end_silence": 1.10,
    },
    "patient": {
        "sensitivity": 0.70,
        "minimum_speech_duration": 0.15,
        "speech_hold_duration": 0.20,
        "end_silence": 1.50,
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


def room_state_summary(scores: Scores, include_advice: bool = True) -> str:
    """The one-sentence Aware room note, or "" when the room sounds clean.

    Says nothing at all unless one cause clearly dominates, so the agent is not
    fed vague acoustic chatter.

    ``include_advice`` appends the cause's standing instruction. Leave it on for
    a conversational agent. Turn it off for a terse one: the advice is phrased as
    an instruction ("confirm anything unexpected before acting on it"), and a
    concise model tends to carry it out loud, opening replies with "just to
    confirm, you asked..." and volunteering that it can hear background voices.
    Without it the note is pure state, which is all the Aware layer needs to
    shape tone.
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
    note = f"Audio note: {severity} input, {cause['room']}."
    return f"{note} {cause['advice']}" if include_advice else note


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


def live_reading(scores: Scores) -> str:
    """The current Tyto reading, phrased for a model rather than a dashboard.

    Handed to the agent every turn so it can answer "how do I sound?" straight
    away instead of paying a tool round trip for it. Values are named as well as
    numbered because the agent is told to describe them in words: the numbers are
    there to rank the causes, not to be read out.

    Loudness and reverb are included as neutral facts, never as problems, which
    is the same rule the rest of the layer follows.

    Only call this for a reading young enough to still be true, because what it
    returns is stated as current with no hedge. Tyto needs a full 5 s window and
    the analyzer is reset every time the agent speaks, so a run of short turns
    can produce no new reading at all, and handing over a minutes-old one is how
    the agent ends up insisting the room is still noisy after the noise stopped.
    The caller does that gate; see READING_MAX_AGE_SECONDS.
    """
    verdict = (
        "degraded" if scores.risk_score >= COMPOSITE_TH[1]
        else "marginal" if scores.risk_score >= COMPOSITE_TH[0]
        else "clean"
    )
    parts = []
    for key in ("noise", "interfering_speech", "packet_loss", "codec_degradation"):
        value = getattr(scores, key)
        band = "high" if value > THRESHOLDS[key][1] else "some" if value > THRESHOLDS[key][0] else "none"
        parts.append(f"{LABELS[key].lower()} {band} ({value:.2f})")
    parts.append(f"speaker level {scores.speaker_loudness:.2f}")
    parts.append(f"room reverb {scores.speaker_reverb:.2f}")
    return (
        f"Microphone reading, private: overall {verdict} ({scores.risk_score:.2f}); "
        + ", ".join(parts)
        + ". This is the only true description of how the user sounds. It replaces anything"
        " said earlier in this conversation about their audio, however confident that was."
        " Use it only if they ask, and answer in plain words."
    )


# What goes in the reading's place when there is no current one. It must not
# leak the mechanics: the user should never hear about measurement, windows,
# seconds, or how long anything took. They asked a question, and the honest
# answer is a short "not right now".
NO_READING = (
    "No microphone reading is available, private. Anything said earlier in this conversation"
    " about how the user sounds is out of date and must not be repeated. If they ask how they"
    " sound, say in your own words, in one short sentence, that you are not sure right now, and"
    " leave it there. Do not explain why, do not say how you would know, and do not ask them for"
    " anything so you can check."
)
