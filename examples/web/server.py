"""Web demo backend: the full Tyto voice agent, served to a browser.

This is the same demo as the browser reference, but the Python backend is the
whole brain. It runs Tyto scoring, the three adaptation layers, and the whole
voice cascade, and it holds your keys (from env vars). The browser is a thin
client: it captures the mic, plays the agent, and renders the UI.

Per browser tab, one session:

    browser mic (PCM16, 16 kHz)  ── websocket ─>  scorer.feed + provider.send_audio
    agent audio (PCM16, 24 kHz)  <─ websocket ──  provider audio_out
    scores / room / vad / nudge  <─ websocket ──  controller (the three layers)

Capture and playback run at different rates on purpose: 16 kHz is native for the
VAD, Tyto and Inkling, and Deepgram returns 24 kHz. The browser keeps one
AudioContext per direction, so neither side resamples.

Keys live only here, never in the browser:
    AIC_SDK_LICENSE   runs Tyto and the VAD locally on this backend
    INKLING_API_KEY   Inkling-Small, the agent's brain
    DEEPGRAM_API_KEY  the agent's voice (Aura-2), and the UI's caption (nova-3)

Run:
    uv pip install -e ".[web]"
    # put the three keys in .env
    uv run examples/web/server.py        # then open http://localhost:8080
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import numpy as np
from aiohttp import WSMsgType, web

from tyto_voice.cascade import PLAYBACK_RATE, SAMPLE_RATE, CascadeProvider
from tyto_voice.controller import TytoController
from tyto_voice.decision import NUDGE_THRESHOLD_DEFAULT, VAD_PROFILES
from tyto_voice.env import load_env
from tyto_voice.prompts import BASE_INSTRUCTIONS, GREETING
from tyto_voice.provider import Handlers
from tyto_voice.scorer import LiveTytoScorer

HERE = Path(__file__).parent
INDEX = HERE / "index.html"
APP_JS = HERE / "app.js"



def _reading_of(controller):
    """(scores, age) for the provider, or None if Tyto has nothing to say yet.

    Age travels with the scores because Tyto is reset on every agent turn and
    needs a fresh 5 s window, so the newest reading is often not recent.
    """
    if controller is None or controller.scores is None:
        return None
    return controller.scores, controller.scores_age

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
        self.provider: CascadeProvider | None = None
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

        handlers = Handlers()
        controller: TytoController | None = None
        provider = CascadeProvider(
            handlers,
            license_key=self.keys["license"],
            inkling_key=self.keys["inkling"],
            deepgram_key=self.keys["deepgram"],
            instructions=BASE_INSTRUCTIONS,
            greeting=GREETING,
            audio_out=self.send_bytes,  # agent audio -> browser plays it
            audio_done=lambda: self.send_json({"type": "agent_done"}),
            audio_flush=lambda: self.send_json({"type": "flush"}),
            turn_detection=VAD_PROFILES["eager"],
            # The reading rides along with every turn, so "how do I sound?" is
            # answered in one round trip instead of two.
            scores=lambda: _reading_of(controller),
            # The browser captures with echo cancellation on, so the agent will
            # not hear itself and cut itself off.
            allow_barge_in=True,
            on_log=lambda k, t: self.send_json({"type": "log", "kind": k, "text": t}),
        )
        scorer = LiveTytoScorer(
            self.keys["license"],
            sample_rate=SAMPLE_RATE,
            on_state=lambda state, text: self.send_json({"type": "tyto_state", "state": state, "text": text}),
        )
        controller = TytoController(
            provider,
            scorer,
            room_advice=False,  # this agent is terse; it would speak the advice
            # Also back on, for the same reason: whatever echo survives would be
            # measured as the user's room, inflating interfering speech and
            # nudging about voices that are our own. The cost is that Tyto only
            # gets a reading when you talk for five continuous seconds.
            pause_scoring_while_speaking=True,
            on_update=self._on_update,
            on_log=lambda k, t: self.send_json({"type": "log", "kind": k, "text": t}),
        )
        scorer.on_scores = controller.on_scores

        handlers.on_ready = controller.on_ready
        handlers.on_agent_speaking = controller.on_agent_speaking
        # on_agent_audio is driven by the browser, which plays the audio and
        # reports when the agent becomes audible / falls silent.
        handlers.on_user_transcript = controller.on_user_transcript
        handlers.on_agent_transcript = controller.on_agent_transcript
        handlers.on_tool_call = controller.on_tool_call

        self.provider, self.scorer, self.controller = provider, scorer, controller
        try:
            scorer.start()  # downloads the model (cached) and checks the license
            provider.connect()  # downloads the VAD model, opens the speak socket
            controller.set_connected(True)
            self.send_json({"type": "status", "state": "live", "label": "Live"})
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
                data = msg.json()
                if data.get("type") in ("start", "stop"):
                    # Both download models and wait on threads. Running them
                    # inline would park the event loop, so the writer task could
                    # not deliver the "loading Tyto model..." progress it is
                    # emitting, and no other tab would be served meanwhile.
                    asyncio.create_task(asyncio.to_thread(session.on_message, data))
                else:
                    session.on_message(data)
    finally:
        await asyncio.to_thread(session.stop)
        writer_task.cancel()
    return ws


async def index_handler(_request: web.Request) -> web.Response:
    return web.Response(text=INDEX.read_text(), content_type="text/html")


async def app_js_handler(_request: web.Request) -> web.Response:
    return web.Response(text=APP_JS.read_text(), content_type="text/javascript")


async def config_handler(_request: web.Request) -> web.Response:
    """Audio rates, so the browser never has to guess them."""
    return web.json_response({"captureRate": SAMPLE_RATE, "playbackRate": PLAYBACK_RATE})


def main() -> None:
    load_env()
    keys = {
        "license": os.environ.get("AIC_SDK_LICENSE", ""),
        "inkling": os.environ.get("INKLING_API_KEY", ""),
        "deepgram": os.environ.get("DEEPGRAM_API_KEY", ""),
    }
    missing = [name for name, value in keys.items() if not value]
    if missing:
        raise SystemExit(f"Missing keys: {', '.join(missing)} (see .env.example).")

    app = web.Application()
    app["keys"] = keys
    app.add_routes(
        [
            web.get("/", index_handler),
            web.get("/app.js", app_js_handler),
            web.get("/config", config_handler),
            web.get("/ws", ws_handler),
        ]
    )
    host, port = "127.0.0.1", int(os.environ.get("PORT", "8080"))
    print(f"Tyto web demo on http://{host}:{port}  (Ctrl-C to stop)")
    web.run_app(app, host=host, port=port, print=None)


if __name__ == "__main__":
    main()
