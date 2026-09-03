"""Tyto voice-agent demo, Python reference.

A live voice agent that adapts to your acoustics on three layers (Aware, Tuned,
Reactive), driven by the ai-coustics Tyto audio-insight model scoring your mic in
real time.

The agent is a cascade: Deepgram Flux hears the user and decides when the turn is
over, gpt-5-mini answers in text, and Deepgram Aura-2 speaks it. Tyto sits one
hop after the microphone and scores exactly what that stack is about to hear.

Public surface:
    decision  - the scoring contract and decision layer (pure Python, no deps)
    scorer    - LiveTytoScorer: real-time Tyto scoring over the aic-sdk
    provider  - VoiceProvider seam (swap the voice backend behind one interface)
    controller- TytoController: wires scores into the three adaptation layers
    voicefocus- VoiceFocus: optional Quail enhancement, on the agent's path only
    cascade   - CascadeProvider: the Pipecat cascade behind that seam

``cascade`` is deliberately NOT imported here: importing it pulls in pipecat,
aiortc and a websocket stack, and ``examples/score_mic.py`` and the decision
tests must keep working on a bare install. Import it directly where it is used.
"""

from .controller import TytoController
from .decision import (
    EnvMonitor,
    Nudge,
    Scores,
    pick_vad_profile,
    room_state_summary,
    strongest_cause,
)
from .provider import Handlers, VoiceProvider
from .scorer import LiveTytoScorer
from .voicefocus import VoiceFocus

__all__ = [
    "EnvMonitor",
    "Handlers",
    "LiveTytoScorer",
    "Nudge",
    "Scores",
    "TytoController",
    "VoiceFocus",
    "VoiceProvider",
    "pick_vad_profile",
    "room_state_summary",
    "strongest_cause",
]
