"""Tyto voice agent, Python reference.

A live voice agent that adapts to your acoustics on three layers (Aware, Tuned,
Reactive), driven by the ai-coustics Tyto audio-insight model scoring your mic
in real time. The scoring contract and tuned constants match the browser
reference (``index.html``) so behavior stays comparable.

The agent is a cascade: Deepgram Flux hears the user and decides when the turn
is over, Pipecat PhoneLLM on Modal answers in text, and Deepgram Aura-2 speaks
it. ``OpenAIRealtimeProvider`` is still here as the second implementation of the
same seam.

Public surface:
    decision  - the scoring contract and decision layer (pure Python)
    scorer    - LiveTytoScorer: real-time Tyto scoring over the aic-sdk
    flux      - FluxSTT: transcription and turn-taking in one socket
    phonellm  - PhoneLLMClient: the agent's replies, served from Modal
    deepgram  - DeepgramTTS: the agent's voice
    provider  - VoiceProvider seam (swap the voice backend behind one interface)
    controller- TytoController: wires scores into the three adaptation layers
    cascade   - CascadeProvider: the Flux -> PhoneLLM -> Aura-2 backend
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
from .deepgram import DeepgramTTS
from .flux import FluxSTT
from .phonellm import PhoneLLMClient
from .provider import Handlers, VoiceProvider
from .scorer import LiveTytoScorer

__all__ = [
    "CHECK_AUDIO_QUALITY_TOOL",
    "CascadeProvider",
    "DeepgramTTS",
    "EnvMonitor",
    "FluxSTT",
    "Handlers",
    "LiveTytoScorer",
    "Nudge",
    "PhoneLLMClient",
    "Scores",
    "SounddeviceSink",
    "TytoController",
    "VoiceProvider",
    "pick_vad_profile",
    "room_state_summary",
    "strongest_cause",
]
