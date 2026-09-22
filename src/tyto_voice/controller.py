"""The Tyto control layer, provider-agnostic.

This is the server-side port of the ``onScores`` handler and the mute/nudge
state machine in the browser reference. It takes a live ``Scores`` stream from
the scorer and a ``VoiceProvider``, and drives the three adaptation layers:

    Layer 1 - Aware:    swap a one-sentence room note into the instructions.
    Layer 2 - Tuned:    eager vs patient turn-taking based on background noise.
    Layer 3 - Reactive: interrupt the agent with one spoken nudge, then resume.

With a ``judge`` (``jev.JevJudge``) the Reactive layer gets a second stage. Tyto
and the tuned thresholds still decide *that* there is a fixable problem (the
``EnvMonitor`` gate). Jev decides *how to act on it right now*: cut in, let the
agent finish its sentence, adapt quietly, or stay silent because the user was
just asked. Jev is consulted speculatively from the moment a cause shows up in
the warn band, so by the time the gate trips a fresh verdict is usually already
here and the nudge fires with no added wait. Without a judge, or when Jev times
out, the gate fires the nudge directly as before.

It also answers the ``check_audio_quality`` tool, and gates scoring so Tyto only
reads the user's audio while the user is actually speaking (mic muted, scoring
paused while the agent talks).

All state is guarded by a single re-entrant lock because scores arrive on the
scorer thread, provider events on the transport thread, and Jev verdicts on the
judge's worker thread.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Callable

from .decision import (
    COMPOSITE_CLEAR,
    COMPOSITE_TH,
    ENV_KEYS,
    LABELS,
    MIN_EXPLANATION_VALUE,
    NO_POLARITY,
    NUDGE_THRESHOLD_DEFAULT,
    NUDGE_THRESHOLD_MAX,
    NUDGE_THRESHOLD_MIN,
    THRESHOLDS,
    VAD_PROFILES,
    EnvMonitor,
    Nudge,
    Scores,
    pick_vad_profile,
    room_state_summary,
    strongest_cause,
)
from .jev import ADAPT_QUIETLY, ASK_AFTER_SENTENCE, ASK_NOW, FRESH_SECONDS, Decision, Situation, severity_band
from .prompts import BASE_INSTRUCTIONS
from .provider import VoiceProvider

# The tool the agent calls when the user asks "how do I sound?".
CHECK_AUDIO_QUALITY_TOOL = {
    "type": "function",
    "name": "check_audio_quality",
    "description": (
        "Get the current real-time audio quality of the user's mic input. Returns a "
        "summary, verdict, the Tyto Score, and the top current issue. Call this whenever "
        "the user asks if you can hear them, how their audio sounds, or about their "
        "connection/environment."
    ),
    "parameters": {"type": "object", "properties": {}, "required": []},
}

CONSULT_INTERVAL_S = 1.0  # at most one speculative Jev question per second
DEFER_MAX_S = 6.0  # "finish the sentence first" waits at most this long
QUIET_HOLD_S = 4.0  # after "stay silent" / "adapt quietly", do not re-ask for this long
CALLER_SPEAKING_S = 1.5  # a user transcript fragment this recent = the user is talking
TREND_STEP = 0.06  # risk change across the history window that counts as a trend


class TytoController:
    def __init__(
        self,
        provider: VoiceProvider,
        scorer,
        *,
        judge=None,
        nudge_threshold: float = NUDGE_THRESHOLD_DEFAULT,
        on_update: Callable[[dict], None] | None = None,
        on_log: Callable[[str, str], None] | None = None,
        clock: Callable[[], float] | None = None,
    ):
        self.provider = provider
        self.scorer = scorer
        self.judge = judge
        self.on_update = on_update
        self.on_log = on_log
        self._clock = clock or time.monotonic

        self._lock = threading.RLock()
        self._monitor = EnvMonitor(min_persist=1, threshold=nudge_threshold)

        self.connected = False
        self.listening = True
        self.agent_speaking = False
        self.agent_audio_playing = False
        self.awaiting_nudge = False
        self.nudge_active = False
        self.nudge_playback_pending = False

        self._last_room = ""
        self._last_vad = "eager"
        self._last_scores: Scores | None = None
        self._last_risk: float | None = None

        # Context for the judge.
        self._risk_history: deque[float] = deque(maxlen=4)
        self._episode_cause: str | None = None
        self._episode_since = 0.0
        self._last_ask: dict[str, float] = {}
        self._times_asked: dict[str, int] = {}
        self._agent_words, self._agent_interim = "", ""
        self._caller_words, self._caller_interim = "", ""
        self._caller_tx_at = 0.0

        # Judge state.
        self._decision: Decision | None = None
        self._decision_cause: str | None = None
        self._pending_directive: Nudge | None = None
        self._deferred: tuple[Nudge, float] | None = None
        self._last_consult = 0.0
        self._quiet_until = 0.0

    # -- session ------------------------------------------------------------ #

    def set_connected(self, value: bool) -> None:
        self.connected = value
        self._sync_scoring_gate()

    @property
    def nudge_threshold(self) -> float:
        return self._monitor.threshold

    @nudge_threshold.setter
    def nudge_threshold(self, value: float) -> None:
        self._monitor.threshold = min(NUDGE_THRESHOLD_MAX, max(NUDGE_THRESHOLD_MIN, value))

    # -- Tyto scores in (scorer thread) ------------------------------------- #

    def on_scores(self, scores: Scores) -> None:
        with self._lock:
            self._last_scores = scores
            self._last_risk = scores.risk_score
            self._risk_history.append(scores.risk_score)

            room = room_state_summary(scores)
            vad = pick_vad_profile(scores)

            if room != self._last_room:  # Layer 1 - Aware
                self._last_room = room
                self.provider.set_instructions(
                    BASE_INSTRUCTIONS + (f"\n\n{room}" if room else "")
                )
                self._log("tyto.aware.update", room or "(clean)")

            if vad != self._last_vad:  # Layer 2 - Tuned
                self._last_vad = vad
                if self.listening:
                    self.provider.set_turn_detection(VAD_PROFILES[vad])
                self._log("tyto.vad.swap", vad)

            # Layer 3 - Reactive. The gate is Tyto's; the judgement is Jev's.
            candidate = strongest_cause(scores, actionable_only=True) if scores.risk_score >= COMPOSITE_CLEAR else None
            self._track_episode(candidate)
            directive = self._monitor.evaluate(scores)
            if self.judge is None:
                if directive:
                    self._fire_nudge(directive)
            else:
                if candidate and self._can_nudge():
                    self._consult(candidate, scores)
                if directive:
                    self._judge_directive(directive)

            self._push_update(room, vad, scores)

    # -- provider events (transport thread) --------------------------------- #

    def on_ready(self) -> None:
        # Once the session is configured, have the agent open the conversation.
        self.provider.request_response()

    def on_agent_speaking(self, active: bool, nudge: bool = False, cancelled: bool = False) -> None:
        with self._lock:
            self.agent_speaking = active
            if active:
                if nudge and self.awaiting_nudge:
                    self._mute_mic_for_agent()
                    self.nudge_active = True
                    self.awaiting_nudge = False
            elif nudge or self.nudge_active:
                # Generation done but audio may still be playing: stay muted
                # until playback stops.
                if cancelled or not self.agent_audio_playing:
                    self._resume_after_nudge()
                else:
                    self.nudge_playback_pending = True
            elif not self.agent_audio_playing:
                self._maybe_unmute_mic()
                self._maybe_fire_deferred()
            self._sync_scoring_gate()

    def on_agent_audio(self, playing: bool) -> None:
        with self._lock:
            self.agent_audio_playing = playing
            if not playing:
                if self.nudge_playback_pending:
                    self._resume_after_nudge()
                else:
                    self._maybe_unmute_mic()
                    self._maybe_fire_deferred()
            self._sync_scoring_gate()

    def on_tool_call(self, name: str, call_id: str) -> None:
        if name != "check_audio_quality":
            return
        result = self.audio_quality_snapshot()
        self._log("tool.check_audio_quality", result.get("summary", ""))
        self.provider.send_tool_result(call_id, result)

    def on_user_transcript(self, text: str, final: bool) -> None:
        with self._lock:
            self._caller_tx_at = self._clock()
            if final:
                self._caller_words, self._caller_interim = text, ""
            else:
                self._caller_interim += text
        self._push_transcript("user", text, final)

    def on_agent_transcript(self, text: str, final: bool) -> None:
        with self._lock:
            if final:
                self._agent_words, self._agent_interim = text, ""
            else:
                self._agent_interim += text
        self._push_transcript("agent", text, final)

    # -- Layer 3 internals (mirror the browser state machine) --------------- #

    def _can_nudge(self) -> bool:
        return not (self.awaiting_nudge or self.nudge_active or not self.listening)

    def _fire_nudge(self, directive: Nudge) -> None:
        if not self._can_nudge():
            return
        now = self._clock()
        self._last_ask[directive.key] = now
        self._times_asked[directive.key] = self._times_asked.get(directive.key, 0) + 1
        self._deferred = None
        self._log("tyto.nudge.trip", f"{directive.label}={directive.value:.2f}")
        # Cut the user's audio and the agent's output, then nudge immediately.
        self._set_mic_enabled(False)
        self._set_listening(False)
        self._sync_scoring_gate()
        self._interrupt_agent_speech(clear_input=True)
        self.awaiting_nudge = True
        self.provider.nudge(directive.text)
        self._log("tyto.nudge.dispatch", directive.text)
        if self.on_update:
            self.on_update({"nudge": {"label": directive.label, "value": directive.value, "text": directive.text}})

    def _resume_after_nudge(self) -> None:
        self.nudge_active = False
        self.awaiting_nudge = False
        self.nudge_playback_pending = False
        self._set_listening(True)
        self._maybe_unmute_mic()
        self._sync_scoring_gate()
        self._log("tyto.input.resumed", "")

    def _interrupt_agent_speech(self, clear_input: bool = False) -> None:
        self.awaiting_nudge = False
        self.nudge_active = False
        self.nudge_playback_pending = False
        self.provider.interrupt(clear_input=clear_input)

    def _mute_mic_for_agent(self) -> None:
        self._set_mic_enabled(False)
        self._sync_scoring_gate()

    def _maybe_unmute_mic(self) -> None:
        if self.agent_speaking or self.agent_audio_playing:
            return
        if self.awaiting_nudge or self.nudge_active or self.nudge_playback_pending or not self.listening:
            return
        self._set_mic_enabled(True)

    def _set_mic_enabled(self, on: bool) -> None:
        self.provider.set_mic_enabled(on)

    def _set_listening(self, on: bool) -> None:
        if self.listening == on:
            return
        self.listening = on
        self.provider.set_turn_detection(VAD_PROFILES.get(self._last_vad) if on else None)

    def _sync_scoring_gate(self) -> None:
        should_score = (
            self.connected and self.listening and not self.agent_speaking and not self.agent_audio_playing
        )
        if should_score and not self.scorer.scoring:
            self.scorer.resume()
        elif not should_score and self.scorer.scoring:
            self.scorer.pause()

    # -- the judge (Jev) ---------------------------------------------------- #

    def _track_episode(self, candidate: dict | None) -> None:
        key = candidate["key"] if candidate else None
        if key != self._episode_cause:
            self._episode_cause = key
            self._episode_since = self._clock()

    def _trend(self) -> str:
        if len(self._risk_history) < 2:
            return "steady"
        delta = self._risk_history[-1] - self._risk_history[0]
        return "getting worse" if delta > TREND_STEP else "improving" if delta < -TREND_STEP else "steady"

    def situation(self, cause: dict, scores: Scores) -> Situation:
        """Everything Jev is told about this moment, already bucketed."""
        now = self._clock()
        since = self._last_ask.get(cause["key"])
        return Situation(
            cause=cause["key"],
            label=LABELS[cause["key"]],
            problem=cause["room"],
            severity=severity_band(scores.risk_score),
            lasting_s=now - self._episode_since,
            trend=self._trend(),
            agent_speaking=self.agent_speaking or self.agent_audio_playing,
            caller_speaking=(now - self._caller_tx_at) < CALLER_SPEAKING_S,
            agent_last_words=_tail(self._agent_words, self._agent_interim),
            caller_last_words=_tail(self._caller_words, self._caller_interim),
            since_ask_s=None if since is None else now - since,
            times_asked=self._times_asked.get(cause["key"], 0),
        )

    def _consult(self, cause: dict, scores: Scores, force: bool = False) -> None:
        now = self._clock()
        if self.judge.busy or (not force and now - self._last_consult < CONSULT_INTERVAL_S):
            return
        self._last_consult = now
        self.judge.ask(self.situation(cause, scores), self._on_decision)

    def _judge_directive(self, directive: Nudge) -> None:
        """The gate tripped: act on a fresh verdict, or wait for the one in flight."""
        if not self._can_nudge() or self._clock() < self._quiet_until:
            return
        d = self._decision
        if d and self._decision_cause == directive.key and self._clock() - d.at < FRESH_SECONDS:
            self._apply(d, directive)
            return
        self._pending_directive = directive
        if not self.judge.busy and self._last_scores is not None:
            cause = strongest_cause(self._last_scores, actionable_only=True)
            if cause:
                self._consult(cause, self._last_scores, force=True)

    def _on_decision(self, decision: Decision, situation: Situation) -> None:
        """Jev answered (judge worker thread)."""
        with self._lock:
            self._decision, self._decision_cause = decision, situation.cause
            self._log(
                "jev.decision",
                f"{decision.action} p={decision.confidence:.2f} {decision.latency_ms}ms: {decision.reason}"
                + (" [fallback]" if decision.source != "jev" else ""),
            )
            if self.on_update:
                self.on_update({"jev": {**decision.as_dict(), "cause": situation.label}})
            directive, self._pending_directive = self._pending_directive, None
            if directive and directive.key == situation.cause and self._can_nudge():
                self._apply(decision, directive)

    def _apply(self, decision: Decision, directive: Nudge) -> None:
        self._decision = None  # consumed; speculation refreshes it
        now = self._clock()
        if decision.action == ASK_NOW:
            self._fire_nudge(directive)
        elif decision.action == ASK_AFTER_SENTENCE:
            if self.agent_speaking or self.agent_audio_playing:
                self._deferred = (directive, now + DEFER_MAX_S)
                self._log("jev.defer", "nudge waits for the end of the sentence")
            else:
                self._fire_nudge(directive)
        else:  # ADAPT_QUIETLY / STAY_SILENT: the Aware note already covers the room
            self._quiet_until = now + QUIET_HOLD_S
            self._log("jev.hold", f"{decision.action}: {decision.reason}")

    def _maybe_fire_deferred(self) -> None:
        if not self._deferred:
            return
        directive, deadline = self._deferred
        if self._clock() > deadline:
            self._deferred = None
            self._log("jev.defer", "expired")
            return
        if not (self.agent_speaking or self.agent_audio_playing) and self._can_nudge():
            self._deferred = None
            self._fire_nudge(directive)

    # -- check_audio_quality tool ------------------------------------------- #

    def _strongest_env(self, scores: Scores):
        best = None
        for key in ENV_KEYS:
            if key in NO_POLARITY:
                continue
            value = getattr(scores, key)
            if value < MIN_EXPLANATION_VALUE:
                continue
            low = THRESHOLDS.get(key, (0.30, 0.50))[0]
            severity = max(0.0, value - low) / max(1 - low, 1e-6)
            if best is None or severity > best["severity"]:
                best = {"key": key, "value": value, "severity": severity}
        return best

    def audio_quality_snapshot(self) -> dict:
        scores = self._last_scores
        if scores is None:
            return {"status": "warming up, ask again in a few seconds"}
        risk = scores.risk_score
        verdict = "degraded" if risk >= COMPOSITE_TH[1] else "marginal" if risk >= COMPOSITE_TH[0] else "good"
        result: dict = {"tyto_score": round(risk, 2), "verdict": verdict}
        for key in ENV_KEYS:
            result[LABELS[key].lower().replace(" ", "_")] = round(getattr(scores, key), 2)
        top = self._strongest_env(scores)
        if top and top["severity"] > 0:
            result["top_issue"] = {
                "key": top["key"],
                "label": LABELS[top["key"]],
                "value": round(top["value"], 2),
                "direction": "high",
            }
            result["summary"] = (
                f"Audio is {verdict}. Biggest issue: {LABELS[top['key']]} is high at {top['value']:.2f}."
            )
        else:
            result["summary"] = f"Audio is {verdict}."
        return result

    # -- UI plumbing -------------------------------------------------------- #

    def _push_update(self, room: str, vad: str, scores: Scores) -> None:
        if self.on_update:
            self.on_update({"room": room, "vad": vad, "scores": scores})

    def _push_transcript(self, who: str, text: str, final: bool) -> None:
        if self.on_update:
            self.on_update({"transcript": {"who": who, "text": text, "final": final}})

    def _log(self, kind: str, text: str) -> None:
        if self.on_log:
            self.on_log(kind, text if isinstance(text, str) else str(text))


def _tail(final: str, interim: str, words: int = 30) -> str:
    """The last few words someone said: the last final line plus what is still interim."""
    return " ".join(f"{final} {interim}".split()[-words:])
