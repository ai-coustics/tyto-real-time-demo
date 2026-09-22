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
import os
from pathlib import Path

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

        handlers.on_ready = controller.on_ready
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
            provider.connect()
            controller.set_connected(True)
            self.send_json({"type": "status", "state": "live", "label": f"Live · {provider.model}"})
        except Exception as err:  # noqa: BLE001 - surface to the browser
            self.send_json({"type": "tyto_state", "state": "error", "text": str(err)})
            self.send_json({"type": "status", "state": "error", "label": "Error"})

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
    ws = web.WebSocketResponse(max_msg_size=0)
    await ws.prepare(request)
    session = Session(ws, asyncio.get_running_loop(), request.app["keys"])
    writer_task = asyncio.create_task(session.writer())
    try:
        async for msg in ws:
            if msg.type == WSMsgType.BINARY:
                session.on_mic(msg.data)
            elif msg.type == WSMsgType.TEXT:
                session.on_message(msg.json())
    finally:
        session.stop()
        writer_task.cancel()
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
