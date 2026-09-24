"""Web demo backend: the full Tyto voice agent, served to a browser.

This is the same demo as the browser reference, but the Python backend is the
whole brain. It runs Tyto scoring and the three adaptation layers, asks Jev how
to act when a problem trips the gate, and holds the agent session and your keys
(from env vars). The browser is a thin client: it captures the mic, plays the
agent, and renders the UI.

Per browser tab, one session:

    browser mic (PCM16, 24 kHz)  ── websocket ─>  scorer.feed + provider.send_audio
    agent audio (PCM16)          <─ websocket ──  provider audio_out
    scores / room / vad / nudge / jev  <─ websocket ──  controller (the three layers)

Keys live only here, never in the browser:
    AIC_SDK_LICENSE      runs Tyto locally on this backend
    OPENAI_API_KEY       opens the GPT-Live 1 session (VOICE_BACKEND=realtime for Realtime)
    AI_GATEWAY_API_KEY   Jev through Vercel AI Gateway, the judge (optional)

Run:
    uv pip install -e ".[web]"
    # put the keys in .env (see .env.example)
    uv run examples/web/server.py        # then open http://localhost:8080
"""

from __future__ import annotations

import asyncio
import ipaddress
import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
from aiohttp import WSMsgType, web

from tyto_voice.backends import make_judge, make_provider
from tyto_voice.controller import TytoController
from tyto_voice.decision import NUDGE_THRESHOLD_DEFAULT
from tyto_voice.env import load_env
from tyto_voice.openai_live import SAMPLE_RATE  # both backends stream PCM16 at 24 kHz
from tyto_voice.provider import Handlers
from tyto_voice.scorer import LiveTytoScorer

HERE = Path(__file__).parent
INDEX = HERE / "index.html"
APP_JS = HERE / "app.js"
DS_DIR = HERE / "ds"  # ai-coustics design-system tokens (fonts, colors, typography, spacing, base)
ASSETS_DIR = HERE / "assets"  # logo mark; the licensed Milling webfont goes in assets/fonts, gitignored

# Abuse guard for a public deploy. Every session spends the server's OpenAI and
# gateway credits, so cap how many run, how long, and how many start per hour.
# Per process: on Modal that is per container. Per-visitor limits need a real
# client address; Modal's web_server proxy sends none (every request arrives from
# its internal 172.x hop), so there only the global limits apply.
#
# Read in main(), after load_env(), so values from .env apply too.


def _env_list(name: str) -> list[str]:
    return [v.strip() for v in os.environ.get(name, "").split(",") if v.strip()]


@dataclass(frozen=True)
class Settings:
    max_sessions: int = 8
    max_session_seconds: float = 300.0
    max_starts_per_hour: int = 60
    max_per_ip: int = 2  # at the same time
    max_starts_per_ip_hour: int = 12
    # X-Forwarded-For is client-controlled, so it is only read when the direct
    # peer is one of these proxies (IPs or CIDRs). Empty = never trusted.
    trusted_proxies: tuple = ()
    # Browsers always send Origin on a websocket upgrade. Empty = only this
    # page's own host may open /ws, so another site cannot spend our credits
    # through its visitors. Non-browser clients can forge Origin; the caps cover those.
    allowed_origins: frozenset = frozenset()

    @classmethod
    def from_env(cls) -> "Settings":
        env = os.environ.get
        return cls(
            max_sessions=int(env("MAX_SESSIONS", cls.max_sessions)),
            max_session_seconds=float(env("MAX_SESSION_SECONDS", cls.max_session_seconds)),
            max_starts_per_hour=int(env("MAX_STARTS_PER_HOUR", cls.max_starts_per_hour)),
            max_per_ip=int(env("MAX_SESSIONS_PER_IP", cls.max_per_ip)),
            max_starts_per_ip_hour=int(env("MAX_STARTS_PER_IP_HOUR", cls.max_starts_per_ip_hour)),
            trusted_proxies=tuple(ipaddress.ip_network(p, strict=False) for p in _env_list("TRUSTED_PROXIES")),
            allowed_origins=frozenset(o.rstrip("/") for o in _env_list("ALLOWED_ORIGINS")),
        )


class SessionGate:
    """Admits a session or says why not. Pure bookkeeping, no I/O, so it is unit tested."""

    def __init__(
        self,
        max_sessions=Settings.max_sessions,
        max_starts_per_hour=Settings.max_starts_per_hour,
        max_per_ip=Settings.max_per_ip,
        max_starts_per_ip_hour=Settings.max_starts_per_ip_hour,
        clock=time.monotonic,
    ):
        self.max_sessions, self.max_starts = max_sessions, max_starts_per_hour
        self.max_per_ip, self.max_ip_starts = max_per_ip, max_starts_per_ip_hour
        self._clock = clock
        self._active: dict[str | None, int] = {}
        self._starts: dict[str | None, deque] = {}

    def _recent(self, key: str | None, now: float) -> deque:
        starts = self._starts.setdefault(key, deque())
        while starts and now - starts[0] > 3600:
            starts.popleft()
        return starts

    def _prune(self, now: float) -> None:
        """Forget visitors with no start in the last hour, so the table stays bounded."""
        for key in [k for k, d in self._starts.items() if k != "*" and (not d or now - d[-1] > 3600)]:
            del self._starts[key]

    def admit(self, ip: str | None) -> str | None:
        """Take a slot and return None, or return why not. ``ip`` None = unknown visitor."""
        now = self._clock()
        self._prune(now)
        everyone = self._recent("*", now)
        if sum(self._active.values()) >= self.max_sessions:
            return "The demo is busy right now. Try again in a minute."
        if len(everyone) >= self.max_starts:
            return "The demo has hit its hourly call limit. Try again later."
        if ip is not None:
            if self._active.get(ip, 0) >= self.max_per_ip:
                return "You already have a call open. Hang up there first."
            if len(self._recent(ip, now)) >= self.max_ip_starts:
                return "Call limit reached for this hour. Try again later."
            self._starts[ip].append(now)
        everyone.append(now)
        self._active[ip] = self._active.get(ip, 0) + 1
        return None

    def release(self, ip: str | None) -> None:
        if self._active.get(ip, 0) > 1:
            self._active[ip] -= 1
        else:
            self._active.pop(ip, None)


def origin_allowed(request: web.Request, allowed: frozenset = frozenset()) -> bool:
    origin = request.headers.get("Origin", "").rstrip("/")
    if allowed:
        return origin in allowed
    return bool(origin) and urlsplit(origin).netloc == request.host


def _public_ip(value: str | None) -> str | None:
    try:
        ip = ipaddress.ip_address((value or "").strip())
    except ValueError:
        return None
    return None if ip.is_private or ip.is_loopback else str(ip)


def client_ip(request: web.Request, trusted: tuple = ()) -> str | None:
    """The visitor's public address, or None when it cannot be known.

    X-Forwarded-For is honored only from a trusted proxy, and then its last hop is
    used: that one was appended by our proxy, the earlier ones by the client. A
    private peer with no trusted header (Modal: 172.x for everyone) gives None,
    so per-visitor limits are skipped instead of locking every visitor out.
    """
    remote = request.remote
    try:
        peer = ipaddress.ip_address(remote or "")
    except ValueError:
        peer = None
    if peer is not None and any(peer in net for net in trusted):
        hops = request.headers.get("X-Forwarded-For", "").split(",")
        return _public_ip(hops[-1])
    return _public_ip(remote)


class Session:
    """One browser connection wired to a scorer, provider, and controller.

    Outbound messages are funneled through one asyncio queue so the websocket is
    written from a single task, even though scores and audio originate on the
    scorer and provider background threads.
    """

    def __init__(self, ws: web.WebSocketResponse, loop: asyncio.AbstractEventLoop, keys: dict):
        self.ws = ws
        self.loop = loop
        self.keys = keys
        self.out: asyncio.Queue = asyncio.Queue()
        self.scorer: LiveTytoScorer | None = None
        self.provider = None
        self.judge = None
        self.controller: TytoController | None = None
        self._started = False

    # -- thread-safe outbound (called from any thread) ---------------------- #

    def send_json(self, obj: dict) -> None:
        self.loop.call_soon_threadsafe(self.out.put_nowait, ("json", obj))

    def send_bytes(self, data: bytes) -> None:
        self.loop.call_soon_threadsafe(self.out.put_nowait, ("bytes", data))

    async def writer(self) -> None:
        while True:
            kind, payload = await self.out.get()
            if kind == "json":
                await self.ws.send_json(payload)
            else:
                await self.ws.send_bytes(payload)

    # -- start / stop ------------------------------------------------------- #

    def start(self) -> None:
        if self._started:
            return
        self._started = True

        log = lambda k, t: self.send_json({"type": "log", "kind": k, "text": t})  # noqa: E731
        handlers = Handlers()
        provider = make_provider(
            handlers,
            api_key=self.keys["openai"],
            audio_out=self.send_bytes,  # agent audio -> browser plays it
            audio_done=lambda: self.send_json({"type": "agent_done"}),
            audio_flush=lambda: self.send_json({"type": "flush"}),
            on_log=log,
        )
        judge = make_judge(on_log=log)  # None without a gateway key: the rule decides alone
        scorer = LiveTytoScorer(
            self.keys["license"],
            sample_rate=SAMPLE_RATE,
            models_dir=os.environ.get("AIC_MODELS_DIR", "./models"),  # baked into the image on Modal
            on_state=lambda state, text: self.send_json({"type": "tyto_state", "state": state, "text": text}),
        )
        controller = TytoController(provider, scorer, judge=judge, on_update=self._on_update, on_log=log)
        scorer.on_scores = controller.on_scores

        def on_ready() -> None:
            self.send_json({"type": "status", "state": "live", "label": f"Live · {provider.model}"})
            controller.on_ready()

        def on_closed(error: str | None) -> None:
            text = f"The voice agent disconnected: {error}" if error else "The voice agent hung up."
            self.send_json({"type": "ended", "text": text})
            self.send_json({"type": "status", "state": "error", "label": "Disconnected"})
            self.loop.call_soon_threadsafe(self.close_soon)

        handlers.on_ready = on_ready
        handlers.on_closed = on_closed
        handlers.on_agent_speaking = controller.on_agent_speaking
        # on_agent_audio is driven by the browser, which plays the audio and
        # reports when the agent becomes audible / falls silent.
        handlers.on_user_transcript = controller.on_user_transcript
        handlers.on_agent_transcript = controller.on_agent_transcript
        handlers.on_tool_call = controller.on_tool_call

        self.provider, self.judge, self.scorer, self.controller = provider, judge, scorer, controller
        self.send_json({"type": "config", "backend": provider.model, "judge": judge.model if judge else None})
        try:
            scorer.start()  # downloads the model (cached) and checks the license
            provider.connect()  # "live" is reported from on_ready, once the session is up
            controller.set_connected(True)
        except Exception as err:  # noqa: BLE001 - surface to the browser
            self.send_json({"type": "tyto_state", "state": "error", "text": str(err)})
            self.send_json({"type": "status", "state": "error", "label": "Error"})

    def close_soon(self) -> None:
        """Close the browser socket after queued messages are written (event loop only)."""
        async def close() -> None:
            await asyncio.sleep(0.2)
            await self.ws.close()

        asyncio.ensure_future(close())

    def stop(self) -> None:
        if self.controller:
            self.controller.set_connected(False)
        if self.scorer:
            self.scorer.stop()
        if self.provider:
            self.provider.disconnect()
        if self.judge:
            self.judge.close()

    # -- inbound from the browser ------------------------------------------- #

    def on_mic(self, pcm16: bytes) -> None:
        if not self.scorer or not self.provider:
            return
        mono = np.frombuffer(pcm16, dtype="<i2").astype(np.float32) / 32768.0
        self.scorer.feed(mono)
        self.provider.send_audio(mono)

    def on_message(self, data: dict) -> None:
        t = data.get("type")
        if t == "start":
            self.start()
        elif t == "stop":
            self.stop()
        elif t == "agent_playing" and self.controller:
            self.controller.on_agent_audio(bool(data.get("value")))
        elif t == "nudge_threshold" and self.controller:
            self.controller.nudge_threshold = float(data.get("value", NUDGE_THRESHOLD_DEFAULT))

    # -- controller UI updates -> browser ----------------------------------- #

    def _on_update(self, state: dict) -> None:
        if "scores" in state:
            scores = state["scores"]
            self.send_json(
                {"type": "scores", "scores": scores.as_dict(), "room": state.get("room", ""), "vad": state.get("vad", "eager")}
            )
        elif "transcript" in state:
            tx = state["transcript"]
            self.send_json({"type": "transcript", "who": tx["who"], "text": tx["text"], "final": tx["final"]})
        elif "nudge" in state:
            self.send_json({"type": "nudge", **state["nudge"]})
        elif "jev" in state:
            self.send_json({"type": "jev", **state["jev"]})


async def ws_handler(request: web.Request) -> web.WebSocketResponse:
    settings: Settings = request.app["settings"]
    if not origin_allowed(request, settings.allowed_origins):
        raise web.HTTPForbidden(text="cross-origin websocket refused")
    ws = web.WebSocketResponse(max_msg_size=1 << 20)  # mic chunks are a few KB
    await ws.prepare(request)
    gate: SessionGate = request.app["gate"]
    ip = client_ip(request, settings.trusted_proxies)
    refused = gate.admit(ip)
    print(f"session ({'per-visitor' if ip else 'global'} limits): {refused or 'admitted'}", flush=True)
    if refused:
        await ws.send_json({"type": "ended", "text": refused})
        await ws.close()
        return ws

    session = Session(ws, asyncio.get_running_loop(), request.app["keys"])
    writer_task = asyncio.create_task(session.writer())

    async def time_cap() -> None:
        await asyncio.sleep(settings.max_session_seconds)
        cap = settings.max_session_seconds
        length = f"{cap / 60:g} minutes" if cap >= 60 else f"{cap:g} seconds"
        session.send_json({"type": "ended", "text": f"Demo calls end after {length}."})
        await asyncio.sleep(0.2)  # let the writer flush it
        await ws.close()

    cap_task = asyncio.create_task(time_cap())
    try:
        async for msg in ws:
            if msg.type == WSMsgType.BINARY:
                session.on_mic(msg.data)
            elif msg.type == WSMsgType.TEXT:
                session.on_message(msg.json())
    finally:
        cap_task.cancel()
        session.stop()
        writer_task.cancel()
        gate.release(ip)
    return ws


async def index_handler(_request: web.Request) -> web.Response:
    return web.Response(text=INDEX.read_text(), content_type="text/html")


async def app_js_handler(_request: web.Request) -> web.Response:
    return web.Response(text=APP_JS.read_text(), content_type="text/javascript")


def main() -> None:
    load_env()
    keys = {
        "license": os.environ.get("AIC_SDK_LICENSE", ""),
        "openai": os.environ.get("OPENAI_API_KEY", ""),
    }
    if not keys["license"] or not keys["openai"]:
        raise SystemExit("Set AIC_SDK_LICENSE and OPENAI_API_KEY (see .env.example).")
    if not (os.environ.get("AI_GATEWAY_API_KEY") or os.environ.get("TYPESAFE_API_KEY")):
        print("No AI_GATEWAY_API_KEY: Jev is off, the tuned rule nudges on its own.")

    app = web.Application()
    app["keys"] = keys
    settings = Settings.from_env()  # after load_env(), so .env values count
    app["settings"] = settings
    app["gate"] = SessionGate(
        settings.max_sessions, settings.max_starts_per_hour, settings.max_per_ip, settings.max_starts_per_ip_hour
    )
    app.add_routes(
        [
            web.get("/", index_handler),
            web.get("/app.js", app_js_handler),
            web.get("/ws", ws_handler),
            web.static("/ds", DS_DIR),
            web.static("/assets", ASSETS_DIR),
        ]
    )
    # Loopback by default; a container platform sets HOST=0.0.0.0 so its proxy can reach us.
    host, port = os.environ.get("HOST", "127.0.0.1"), int(os.environ.get("PORT", "8080"))
    print(f"Tyto web demo on http://{host}:{port}  (Ctrl-C to stop)")
    web.run_app(app, host=host, port=port, print=None)


if __name__ == "__main__":
    main()
