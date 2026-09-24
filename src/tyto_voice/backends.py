"""Pick the voice backend and the judge from the environment.

Shared by the entry points so the web demo and the terminal agent read the same
variables:

    VOICE_BACKEND        live (default, GPT-Live 1) or realtime (gpt-realtime-2.1)
    OPENAI_API_KEY       both backends
    LIVE_MODEL           default gpt-live-1
    LIVE_VOICE           default marin
    AI_GATEWAY_API_KEY   Jev through Vercel AI Gateway (model typesafe-ai/jev)
    TYPESAFE_API_KEY     Jev direct from TypeSafe instead (model jev-latest)
    JEV_BASE_URL / JEV_MODEL   overrides for either
"""

from __future__ import annotations

import os

from .controller import CHECK_AUDIO_QUALITY_TOOL
from .decision import VAD_PROFILES
from .jev import GATEWAY_BASE_URL, TYPESAFE_BASE_URL, JevJudge
from .prompts import BASE_INSTRUCTIONS

BACKENDS = {"live": "gpt-live-1", "realtime": "gpt-realtime-2.1"}


def backend_name() -> str:
    name = os.environ.get("VOICE_BACKEND", "live").strip().lower()
    if name in ("live", "gpt-live", "gpt-live-1"):
        return "live"
    if name in ("realtime", "gpt-realtime"):
        return "realtime"
    raise SystemExit(f"Unknown VOICE_BACKEND={name!r}. Use 'live' (GPT-Live 1) or 'realtime'.")


def make_provider(handlers, *, api_key: str, audio_out, audio_done=None, audio_flush=None, on_log=None):
    """Build the configured voice backend behind the ``VoiceProvider`` seam."""
    common = dict(
        api_key=api_key,
        instructions=BASE_INSTRUCTIONS,
        audio_out=audio_out,
        audio_done=audio_done,
        audio_flush=audio_flush,
        on_log=on_log,
    )
    if backend_name() == "live":
        from .openai_live import OpenAILiveProvider

        return OpenAILiveProvider(
            handlers,
            model=os.environ.get("LIVE_MODEL", BACKENDS["live"]),
            voice=os.environ.get("LIVE_VOICE", "marin"),
            **common,
        )
    from .openai_realtime import OpenAIRealtimeProvider

    return OpenAIRealtimeProvider(
        handlers,
        model=os.environ.get("REALTIME_MODEL", BACKENDS["realtime"]),
        turn_detection=VAD_PROFILES["eager"],
        tools=[CHECK_AUDIO_QUALITY_TOOL],
        **common,
    )


def make_judge(on_log=None) -> JevJudge | None:
    """Jev via the gateway key (preferred), a TypeSafe key, or None: the rule decides alone."""
    model = os.environ.get("JEV_MODEL") or None
    if os.environ.get("AI_GATEWAY_API_KEY"):
        judge = JevJudge(
            os.environ["AI_GATEWAY_API_KEY"],
            base_url=os.environ.get("JEV_BASE_URL", GATEWAY_BASE_URL),
            model=model,
            on_log=on_log,
        )
    elif os.environ.get("TYPESAFE_API_KEY"):
        judge = JevJudge(
            os.environ["TYPESAFE_API_KEY"],
            base_url=os.environ.get("JEV_BASE_URL", TYPESAFE_BASE_URL),
            model=model,
            on_log=on_log,
        )
    else:
        return None
    judge.warm_up()
    return judge
