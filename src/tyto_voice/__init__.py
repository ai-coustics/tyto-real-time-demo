"""Tyto voice-agent demo, Python reference.

A live voice agent that adapts to your acoustics on three layers (Aware, Tuned,
Reactive), driven by the ai-coustics Tyto audio-insight model scoring your mic
in real time. This package mirrors the browser reference (``index.html``) and
keeps the scoring contract and tuned constants identical for comparability.

Public surface:
    decision  - the scoring contract and decision layer (pure Python)
    scorer    - LiveTytoScorer: real-time Tyto scoring over the aic-sdk
    provider  - VoiceProvider seam (swap the voice backend behind one interface)
    controller- TytoController: wires scores into the three adaptation layers
    openai_live - OpenAILiveProvider over the GPT-Live 1 WebSocket API (default)
    openai_realtime - OpenAIRealtimeProvider over the Realtime WebSocket API
    jev       - JevJudge: TypeSafe Jev as the judge of the Reactive layer
    backends  - make_provider / make_judge from environment variables
"""

from .audio import SounddeviceSink
from .controller import CHECK_AUDIO_QUALITY_TOOL, TytoController
from .decision import (
    EnvMonitor,
    Nudge,
    Scores,
    pick_vad_profile,
    room_state_summary,
    strongest_cause,
)
from .jev import Decision, JevJudge, Situation
from .provider import Handlers, VoiceProvider
from .scorer import LiveTytoScorer

__all__ = [
    "CHECK_AUDIO_QUALITY_TOOL",
    "Decision",
    "EnvMonitor",
    "Handlers",
    "JevJudge",
    "LiveTytoScorer",
    "Nudge",
    "Scores",
    "Situation",
    "SounddeviceSink",
    "TytoController",
    "VoiceProvider",
    "pick_vad_profile",
    "room_state_summary",
    "strongest_cause",
]
