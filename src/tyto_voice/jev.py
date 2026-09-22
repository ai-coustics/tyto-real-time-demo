"""Jev as the judge of the Reactive layer (TypeSafe AI System One, via Vercel AI Gateway).

Tyto plus the tuned thresholds in ``decision.py`` still decide *that* the user's
audio has a problem the user can fix. That gate never leaves Python. Jev decides
*how the agent should act on it right now*:

    ask_now             cut in at once (mid-sentence if the agent is talking) and ask
    ask_after_sentence  let the agent finish the sentence it is saying, then ask
    adapt_quietly       say nothing, keep going and confirm details more carefully
    stay_silent         nothing new: the user was just asked, or is already fixing it

Jev is a System One model: it answers typed questions against a state with
calibrated probabilities in one round trip (about 300 ms through the gateway once
the connection is warm, 800 ms cold) and generates no text. So the state is
written for it as short named buckets ("severe", "about 10 seconds", "never")
rather than raw floats, which its model card says it reads poorly, and the
spoken line is still chosen in Python from ``decision.EXPLANATIONS``.

Three atomic questions go out in one request (fan-out) and the code combines
them (composite scoring):

    action            Choice over the options above (masked to what makes sense now)
    caller_fixing     Noul: the user's last words show they are already dealing with it
    detail_in_flight  Noul: the agent is mid-number / name / address right now

``JevJudge.ask`` never blocks the caller: one request is in flight at a time and
the answer arrives on a worker thread. A timeout or error yields a ``Decision``
with ``source="fallback"`` and the pre-Jev behaviour (ask now), so the demo
degrades to the deterministic rule instead of stalling.
"""

from __future__ import annotations

import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

# Vercel AI Gateway speaks the TypeSafe API under this base; billed to the gateway key.
GATEWAY_BASE_URL = "https://ai-gateway.vercel.sh/typesafe"
GATEWAY_MODEL = "typesafe-ai/jev"
# TypeSafe direct, if you have a TypeSafe key instead.
TYPESAFE_BASE_URL = "https://api.typesafe.ai"
TYPESAFE_MODEL = "jev-latest"

ASK_NOW = "ask_now"
ASK_AFTER_SENTENCE = "ask_after_sentence"
ADAPT_QUIETLY = "adapt_quietly"
STAY_SILENT = "stay_silent"
ACTIONS = (ASK_NOW, ASK_AFTER_SENTENCE, ADAPT_QUIETLY, STAY_SILENT)

REASONS = {
    ASK_NOW: "cut in now and ask the user to fix it",
    ASK_AFTER_SENTENCE: "let the agent finish its sentence, then ask",
    ADAPT_QUIETLY: "not worth interrupting, adapt quietly",
    STAY_SILENT: "nothing new to say right now",
}

# Below this confidence (about a coin flip across the offered options) the
# answer is not trusted and the deterministic rule (ask now) applies.
MIN_CONFIDENCE = 0.30
# Noul probability at/above which a companion question overrides the action.
NOUL_TRIP = 0.60
# A decision older than this no longer describes the situation.
FRESH_SECONDS = 2.5
# Per-request budget. Measured p50 through the gateway is ~300 ms warm.
TIMEOUT_SECONDS = 1.5


@dataclass(frozen=True)
class Situation:
    """What Jev is told, all of it already bucketed by the controller."""

    cause: str  # decision.py key, e.g. "noise"
    label: str  # human label, e.g. "Noise"
    problem: str  # the "room" phrase from EXPLANATIONS
    severity: str  # "slight" | "noticeable" | "severe"
    lasting_s: float
    trend: str  # "getting worse" | "steady" | "improving"
    agent_speaking: bool
    caller_speaking: bool
    agent_last_words: str
    caller_last_words: str
    since_ask_s: float | None  # None = never asked about this cause
    times_asked: int = 0


@dataclass
class Decision:
    action: str
    confidence: float = 0.0
    probabilities: dict = field(default_factory=dict)
    caller_fixing: float = 0.0
    detail_in_flight: float = 0.0
    latency_ms: int = 0
    source: str = "jev"  # "jev" | "fallback"
    reason: str = ""
    at: float = field(default_factory=time.monotonic)

    def as_dict(self) -> dict:
        return {
            "action": self.action,
            "confidence": round(self.confidence, 2),
            "probabilities": {k: round(v, 2) for k, v in self.probabilities.items()},
            "caller_fixing": round(self.caller_fixing, 2),
            "detail_in_flight": round(self.detail_in_flight, 2),
            "latency_ms": self.latency_ms,
            "source": self.source,
            "reason": self.reason,
        }


# --------------------------------------------------------------------------- #
# State and questions                                                         #
# --------------------------------------------------------------------------- #


def bucket_seconds(seconds: float | None, *, never: str = "never", now_word: str = "just now") -> str:
    """Durations as words. Jev's model card: it does not compare numbers reliably."""
    if seconds is None:
        return never
    if seconds < 3:
        return now_word
    if seconds < 8:
        return "a few seconds"
    if seconds < 15:
        return "about 10 seconds"
    if seconds < 25:
        return "about 20 seconds"
    if seconds < 45:
        return "about half a minute"
    if seconds < 90:
        return "about a minute"
    return "several minutes"


def severity_band(risk: float) -> str:
    """The Tyto risk bands from the docs, as words."""
    if risk >= 0.50:
        return "severe"
    if risk >= 0.30:
        return "noticeable"
    return "slight"


def _times(n: int) -> str:
    return "never" if n <= 0 else "once" if n == 1 else "twice" if n == 2 else "several times"


def build_state(s: Situation) -> dict:
    """The state Jev evaluates: named buckets only, no raw acoustics."""
    return {
        "call": {
            "agent_is_speaking": s.agent_speaking,
            "user_is_speaking": s.caller_speaking,
            "agent_last_words": s.agent_last_words or "(nothing yet)",
            "user_last_words": s.caller_last_words or "(nothing yet)",
        },
        "audio_problem": {
            "problem": s.problem,
            "severity": s.severity,
            "lasting": bucket_seconds(s.lasting_s, now_word="just started"),
            "trend": s.trend,
            "user_can_fix_it": True,
        },
        "history": {
            "user_last_asked_to_fix_this": bucket_seconds(s.since_ask_s),
            "times_asked_so_far": _times(s.times_asked),
        },
    }


def build_questions(s: Situation) -> dict:
    """Three atomic questions in one request. Options that make no sense now are masked."""
    if s.agent_speaking:
        criteria = {
            ASK_NOW: "Stop talking at once, even mid-sentence, and ask the user to fix the problem now.",
            ASK_AFTER_SENTENCE: "Finish the sentence being said, then ask the user to fix the problem.",
        }
    else:
        criteria = {ASK_NOW: "Ask the user to fix the problem right now."}
    criteria[ADAPT_QUIETLY] = (
        "Do not mention the problem. Keep the conversation going and confirm details more carefully."
    )
    criteria[STAY_SILENT] = (
        "Do nothing new right now, because the user was asked recently and may be dealing with it, "
        "or the problem is fading."
    )
    return {
        "action": {
            "type": "choice",
            "instructions": (
                "The user's audio has a problem the user could fix. "
                "What should the voice agent do right now?"
            ),
            "criteria": criteria,
        },
        "caller_fixing": {
            "type": "noul",
            "instructions": (
                "The user's most recent words show they are already dealing with the audio problem, "
                "for example moving, turning something down, or asking for a moment."
            ),
        },
        "detail_in_flight": {
            "type": "noul",
            "instructions": (
                "The agent's most recent words are in the middle of giving or confirming a specific "
                "detail such as a number, name, address, time or code."
            ),
        },
    }


def parse_answer(body: dict, latency_ms: int = 0) -> Decision:
    answers = body.get("answers") or {}
    action = answers.get("action") or {}
    choice = action.get("choice")
    if choice not in ACTIONS:
        raise ValueError(f"unexpected action {choice!r}")
    return Decision(
        action=choice,
        confidence=float(action.get("confidence") or 0.0),
        probabilities={k: float(v) for k, v in (action.get("probabilities") or {}).items()},
        caller_fixing=float((answers.get("caller_fixing") or {}).get("noul") or 0.0),
        detail_in_flight=float((answers.get("detail_in_flight") or {}).get("noul") or 0.0),
        latency_ms=latency_ms,
    )


def fallback(reason: str, latency_ms: int = 0) -> Decision:
    """The pre-Jev behaviour: a tripped gate asks the user right away."""
    return Decision(action=ASK_NOW, source="fallback", reason=reason, latency_ms=latency_ms)


def apply_policy(d: Decision, s: Situation) -> Decision:
    """Combine the atomic answers in code. This is where the policy lives, not in a prompt."""
    if d.source != "jev":
        return d
    if d.confidence < MIN_CONFIDENCE:
        d.action, d.reason = ASK_NOW, f"low confidence ({d.confidence:.2f}), rule default"
    elif d.caller_fixing >= NOUL_TRIP:
        d.action, d.reason = STAY_SILENT, "the user is already dealing with it"
    elif d.action == ASK_NOW and s.agent_speaking and d.detail_in_flight >= NOUL_TRIP:
        d.action, d.reason = ASK_AFTER_SENTENCE, "the agent is mid-detail, finish it first"
    else:
        d.reason = REASONS[d.action]
    return d


# --------------------------------------------------------------------------- #
# Client                                                                      #
# --------------------------------------------------------------------------- #


class JevJudge:
    """Non-blocking Jev client: one request in flight, answers on a worker thread."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = GATEWAY_BASE_URL,
        model: str | None = None,
        timeout_s: float = TIMEOUT_SECONDS,
        on_log: Callable[[str, str], None] | None = None,
        transport=None,  # httpx transport override (tests)
    ):
        import httpx  # only the judge needs it; the mic scorer does not

        self.model = model or (TYPESAFE_MODEL if base_url.startswith(TYPESAFE_BASE_URL) else GATEWAY_MODEL)
        self.base_url = base_url
        self._client = httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(timeout_s, connect=4.0),
            transport=transport,
        )
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="jev")
        self._lock = threading.Lock()
        self._busy = False
        self._on_log = on_log

    @property
    def busy(self) -> bool:
        return self._busy

    def warm_up(self) -> None:
        """Open the TLS connection now so the first decision does not pay for it."""
        self._pool.submit(self._warm)

    def ask(self, situation: Situation, callback: Callable[[Decision, Situation], None]) -> bool:
        """Evaluate in the background; ``callback(decision, situation)`` runs on the worker.

        Returns False (and does nothing) if a request is already in flight.
        """
        with self._lock:
            if self._busy:
                return False
            self._busy = True
        self._pool.submit(self._run, situation, callback)
        return True

    def evaluate(self, situation: Situation) -> Decision:
        """Blocking evaluation with the policy applied. Never raises."""
        t0 = time.perf_counter()
        try:
            r = self._client.post(
                "/v1/systemone",
                json={"model": self.model, "state": build_state(situation), "questions": build_questions(situation)},
            )
            r.raise_for_status()
            decision = parse_answer(r.json(), self._ms(t0))
        except Exception as err:  # noqa: BLE001 - any failure degrades to the rule
            decision = fallback(f"{type(err).__name__}: {err}"[:140], self._ms(t0))
            self._log("jev.fallback", decision.reason)
        return apply_policy(decision, situation)

    def close(self) -> None:
        self._pool.shutdown(wait=False)
        self._client.close()

    # -- internals ---------------------------------------------------------- #

    def _run(self, situation: Situation, callback) -> None:
        try:
            callback(self.evaluate(situation), situation)
        except Exception as err:  # noqa: BLE001
            self._log("jev.error", str(err))
        finally:
            with self._lock:
                self._busy = False

    def _warm(self) -> None:
        t0 = time.perf_counter()
        try:
            self._client.get("/v1/models")
            self._log("jev.warm", f"{self.model} ready in {self._ms(t0)} ms")
        except Exception as err:  # noqa: BLE001
            self._log("jev.warm", f"failed: {err}"[:140])

    @staticmethod
    def _ms(t0: float) -> int:
        return int((time.perf_counter() - t0) * 1000)

    def _log(self, kind: str, text: str) -> None:
        if self._on_log:
            self._on_log(kind, text)


def keywords(text: str, min_len: int = 6) -> set[str]:
    """Distinctive words of a nudge line, used to spot it in a transcript."""
    return {w for w in re.findall(r"[a-z]+", text.lower()) if len(w) >= min_len}
