"""Tyto scoring contract and decision layer.

This module is the heart of the demo and the one part that is identical across
every branch of this repo (OpenAI WebRTC, ElevenLabs, LiveKit, this Python
reference). It is pure Python with no dependencies, so it is trivial to read,
test, and reuse.

It does two things:

1. Defines the *scoring contract*: the ``Scores`` value object and the tuned
   constants (5 s window, 0.5 s hop, EMA alpha 0.3). These match the browser
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
# The window is fixed at 5 s by the model and cannot be shortened, but it can be
# slid faster, and the hop is what the demo actually feels: it sets how quickly
# the meters move and how quickly all three layers respond.
#
# Tyto 1.1 analyses a window in about 100 ms, so:
#     1.00 s hop -> 10% of one core, 1 reading per second
#     0.50 s hop -> 20% of one core, 2 readings per second
#     0.25 s hop -> 40% of one core, 4 readings per second
# This branch runs at 0.5 s. It is a deliberate divergence from the browser
# reference's 1 s: the Reactive layer here interrupts the agent mid-sentence, so
# the delay between a room going bad and the agent saying so is the whole point.
HOP_SECONDS = 0.5
# Smoothing of successive analyze() reads; the docs recommend 0.3. The EMA time
# constant is roughly hop / alpha, so halving the hop also halves the smoothing
# lag: about 1.7 s here against 3.3 s at a 1 s hop.
SCORE_EMA_ALPHA = 0.3
# Scored windows a cause must dominate before the Reactive layer will fire.
#
# One, on purpose. At a 0.5 s hop that is half a second of evidence, which is
# the most reactive this can be made without dropping the "a cause must
# dominate" rule itself. The EMA is what stops it being twitchy: a single bad
# window only moves the smoothed score 30% of the way, so a genuine transient
# still cannot cross the gate on its own.
NUDGE_MIN_PERSIST = 1
# Quiet period after a nudge, counted in scored windows so this file stays
# timer-free and testable.
#
# It has to exist here, and it did not before. The browser reference gets a
# cooldown for free: it stops scoring while the agent talks, so after a nudge the
# analyzer needs a fresh 5 s window before it can say anything at all. This
# branch keeps Tyto measuring straight through the agent's own voice (the
# browser cancels the echo, so there is nothing to protect against), which is
# exactly what lets a nudge interrupt a reply already in progress. Without a
# cooldown that same change turns one bad room into a nudge every half second.
NUDGE_COOLDOWN_SECONDS = 10.0
NUDGE_COOLDOWN_WINDOWS = round(NUDGE_COOLDOWN_SECONDS / HOP_SECONDS)

# Hard ceiling on how long a nudge may hold the microphone shut.
#
# The nudge is the only thing in this demo that closes the input gate, and it is
# meant to close it for the couple of seconds it takes to say one line. Normally
# it reopens when whoever is playing the audio reports it has finished. That
# report travels from the browser, over the socket, and it only fires once the
# generation is done AND every scheduled buffer has drained, so there are
# several ways for it never to arrive: a dropped message, a playback node that
# never fires onended, a flush racing the last buffer.
#
# When it did not arrive, the gate stayed shut for the rest of the session and
# the agent never heard another word. That failure is silent, permanent, and
# looks exactly like a broken microphone, which makes it far worse than the
# thing it is guarding. So the gate is on a timer as well: whatever else
# happens, the user gets their microphone back.
NUDGE_MAX_SECONDS = 8.0


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
#
# This branch sits at the bottom of the range, one point above the hysteresis
# floor, so the agent speaks up as soon as the room leaves the "good" band
# rather than waiting for it to get bad. The dominant-cause rule is what keeps
# that honest: a risk score alone, however high, still never nudges.
NUDGE_THRESHOLD_DEFAULT = 0.31
NUDGE_THRESHOLD_MIN = COMPOSITE_CLEAR
NUDGE_THRESHOLD_MAX = COMPOSITE_NUDGE

# Turn-detection profiles handed to the voice provider (Layer 2).
# Eager: snappy turns, and speculate on the reply before the user has finished.
# Patient: harder to end a turn in a noisy room, so background sound stops
# ending the user's sentences for them.
#
# These are the one place this file is NOT identical to the browser reference.
# The browser drives OpenAI Realtime, whose turn detection lives on the server
# and is configured with semantic_vad / server_vad dicts. This branch hears the
# user through Deepgram Flux, which does turn detection itself, so the profiles
# are its parameters instead. The layer, the two profile names, and when they
# swap are unchanged, which is what keeps the demos comparable.
#
# Keys map to Flux's end-of-turn parameters, which Pipecat's
# DeepgramFluxSTTService takes as DeepgramFluxSTTSettings and can change
# mid-stream via an STTUpdateSettingsFrame (see [cascade.py](cascade.py)):
#   eot_threshold        0.5..1.0, confidence needed to call the turn over.
#                        Lower is faster and cuts people off more often.
#   eager_eot_threshold  0.3..0.9, when to start speculating on the reply.
#                        Must be <= eot_threshold. Omitted (None) turns
#                        speculation off entirely.
#   eot_timeout_ms       500..60000, hard stop on a turn that never resolves.
#
# Eager sits at the floor of both ranges on purpose: this demo is tuned to be
# reactive, and a false turn end costs one abandoned speculation, which is
# invisible. Patient gives that up because in a noisy room a speculation is
# usually wrong and an early turn end is usually the room, not the user.
#
# Patient's eot_threshold is 0.7, Deepgram's own default, and it must NOT be
# pushed up towards the top of the range however patient you want to be.
# Measured against real speech on this stack, the confidence Flux reports at a
# genuine end of turn sits around 0.5: the EndOfTurn that ended a complete
# spoken question came in at 0.523, and the eager events before it at 0.31 and
# 0.52. A threshold of 0.85 was therefore never reached at all, and the only
# thing left that could end a turn was eot_timeout_ms. The agent went silent for
# the entire time the room was noisy, which is the exact opposite of the
# intent, and it looked like a broken microphone rather than a patient agent.
# If you raise this, measure end_of_turn_confidence on real speech first.
VAD_PROFILES = {
    "eager": {
        "eot_threshold": 0.5,
        "eager_eot_threshold": 0.3,
        "eot_timeout_ms": 2000,
    },
    "patient": {
        "eot_threshold": 0.7,
        "eager_eot_threshold": None,
        "eot_timeout_ms": 4000,
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
    cause's streak resets and ``cooldown`` windows must pass before anything can
    fire again. No timers: both counts are in scored windows.
    """

    def __init__(
        self,
        min_persist: int = NUDGE_MIN_PERSIST,
        threshold: float = NUDGE_THRESHOLD_DEFAULT,
        cooldown: int = NUDGE_COOLDOWN_WINDOWS,
    ):
        self.min_persist = min_persist
        self.threshold = threshold  # live-adjustable risk gate
        self.cooldown = cooldown
        self._streak: dict[str, int] = {}
        self._quiet = 0  # windows still owed before another nudge may fire

    def evaluate(self, scores: Scores) -> Nudge | None:
        if self._quiet > 0:
            self._quiet -= 1
            # Keep counting the streak down as well, so the window after a
            # cooldown is judged on the room now and not on the room during it.
            self._streak = {}
            return None
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
        self._quiet = self.cooldown
        return Nudge(key=key, label=LABELS[key], value=cause["value"], text=cause["text"])
