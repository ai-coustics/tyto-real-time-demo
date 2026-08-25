"""Tyto voice-agent demo, Python reference.

A live voice agent that adapts to your acoustics on three layers (Aware, Tuned,
Reactive), driven by the ai-coustics Tyto audio-insight model scoring your mic
in real time. The scoring contract and tuned constants match the browser
reference (``index.html``) so behavior stays comparable.

The agent itself is a cascade: the ai-coustics VAD decides when you have
finished talking, Inkling-Small hears the utterance and replies in text, and
Deepgram speaks it. ``OpenAIRealtimeProvider`` is still here as the second
implementation of the same seam.

Public surface:
    decision  - the scoring contract and decision layer (pure Python)
    scorer    - LiveTytoScorer: real-time Tyto scoring over the aic-sdk
    vad       - LiveVad: turn-taking over the ai-coustics VAD
    inkling   - InklingClient: audio-in / text-out replies
    deepgram  - DeepgramTTS: the agent's voice, transcribe: the UI caption
    provider  - VoiceProvider seam (swap the voice backend behind one interface)
    controller- TytoController: wires scores into the three adaptation layers
    cascade   - CascadeProvider: the VAD -> Inkling -> Deepgram backend
    openai_realtime - OpenAIRealtimeProvider over the Realtime WebSocket API
"""

from .audio import SounddeviceSink
from .cascade import CascadeProvider
from .controller import CHECK_AUDIO_QUALITY_TOOL, TytoController
from .decision import (
    EnvMonitor,
    Nudge,
    Scores,
    pick_vad_profile,
    room_state_summary,
    strongest_cause,
)
from .deepgram import DeepgramTTS, transcribe
from .inkling import InklingClient
from .provider import Handlers, VoiceProvider
from .scorer import LiveTytoScorer
from .vad import LiveVad

__all__ = [
    "CHECK_AUDIO_QUALITY_TOOL",
    "CascadeProvider",
    "DeepgramTTS",
    "EnvMonitor",
    "Handlers",
    "InklingClient",
    "LiveTytoScorer",
    "LiveVad",
    "Nudge",
    "Scores",
    "SounddeviceSink",
    "TytoController",
    "VoiceProvider",
    "pick_vad_profile",
    "room_state_summary",
    "strongest_cause",
    "transcribe",
]
