"""Web demo backend: the full Tyto voice agent, served to a browser.

This is the same demo as the browser reference, but the Python backend is the
whole brain. It runs Tyto scoring, the three adaptation layers, and the whole
voice cascade, and it holds your keys (from env vars). The browser is a thin
client: it captures the mic, plays the agent, and renders the UI.

Per browser tab, one session:

    browser mic (PCM16, 16 kHz)  ── websocket ─>  scorer.feed + provider.send_audio
    agent audio (PCM16, 24 kHz)  <─ websocket ──  provider audio_out
    scores / room / vad / nudge  <─ websocket ──  controller (the three layers)

Capture and playback run at different rates on purpose: 16 kHz is native for
both Tyto and Deepgram Flux, and Aura-2 returns 24 kHz. The browser keeps one
AudioContext per direction, so neither side resamples.

This is the frontend where the Reactive layer is at its most aggressive. The
browser captures with echo cancellation, so the agent cannot hear itself, which
lets two things be switched on that the terminal demo cannot have: barge-in, and
Tyto measuring straight through the agent's own replies. The second is what lets
a nudge interrupt a reply that is already being spoken.

Keys live only here, never in the browser:
    AIC_SDK_LICENSE     runs Tyto locally on this backend
    MODAL_ENDPOINT_URL  the PhoneLLM Auto Endpoint
    MODAL_API_KEY       its proxy token, <token-id>.<token-secret>
    DEEPGRAM_API_KEY    Flux on the way in, Aura-2 on the way out

Run:
    uv pip install -e ".[web]"
    # put the four values in .env
    uv run examples/web/server.py        # then open http://localhost:8080
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import numpy as np
from aiohttp import WSMsgType, web

from tyto_voice.cascade import SAMPLE_RATE, CascadeProvider
from tyto_voice.controller import CHECK_AUDIO_QUALITY_TOOL, TytoController
from tyto_voice.decision import NUDGE_THRESHOLD_DEFAULT, VAD_PROFILES
from tyto_voice.env import load_env
from tyto_voice.prompts import BASE_INSTRUCTIONS, GREETING
from tyto_voice.provider import Handlers
from tyto_voice.scorer import LiveTytoScorer
from tyto_voice.voicefocus import VoiceFocus

HERE = Path(__file__).parent
INDEX = HERE / "index.html"
APP_JS = HERE / "app.js"


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
        self.voice_focus: VoiceFocus | None = None
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
        provider = CascadeProvider(
            handlers,
            endpoint_url=self.keys["endpoint"],
            modal_key=self.keys["modal"],
            deepgram_key=self.keys["deepgram"],
            instructions=BASE_INSTRUCTIONS,
            greeting=GREETING,
            audio_out=self.send_bytes,  # agent audio -> browser plays it
            audio_done=lambda: self.send_json({"type": "agent_done"}),
            audio_flush=lambda: self.send_json({"type": "flush"}),
            turn_detection=VAD_PROFILES["eager"],
            tools=[CHECK_AUDIO_QUALITY_TOOL],
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
        voice_focus = VoiceFocus(
            self.keys["license"],
            sample_rate=SAMPLE_RATE,
            on_log=lambda k, t: self.send_json({"type": "log", "kind": k, "text": t}),
        )
        controller = TytoController(
            provider,
            scorer,
            room_advice=False,  # this agent is terse; it would speak the advice
            # On, for the same reason barge-in is: the browser cancels the echo,
            # so the agent's voice is not in the signal and Tyto can keep
            # measuring the room throughout. That is what lets the Reactive
            # layer cut into a reply already in progress.
            pause_scoring_while_speaking=False,
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
        self.voice_focus = voice_focus
        try:
            scorer.start()  # downloads the model (cached) and checks the license
            voice_focus.start()  # optional; a failure here only disables the switch
            self.send_json({"type": "voice_focus", "available": voice_focus.available, "on": False})
            provider.connect()
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
        if self.voice_focus:
            self.voice_focus.stop()
        if self.provider:
            self.provider.disconnect()

    # -- inbound from the browser ------------------------------------------- #

    def on_mic(self, pcm16: bytes) -> None:
        if not self.scorer or not self.provider:
            return
        mono = np.frombuffer(pcm16, dtype="<i2").astype(np.float32) / 32768.0
        # Tyto always gets the raw microphone. Voice Focus, when the visitor
        # switches it on, cleans only what the agent hears. Reversing this would
        # have Tyto scoring the enhancer instead of the room, which is the one
        # wiring mistake that would quietly invalidate the whole demo.
        self.scorer.feed(mono)
        to_agent = self.voice_focus.process(mono) if self.voice_focus else mono
        if len(to_agent):
            self.provider.send_audio(to_agent)
        self._mic_telemetry(mono)

    def _mic_telemetry(self, mono: np.ndarray) -> None:
        """Report, once a second, that audio is arriving and where it is going.

        Worth its keep. Every way this demo fails quietly looks identical from
        the browser: the greeting plays and then nothing ever happens again. The
        microphone may not be captured, the audio may be silence, the provider
        may be dropping it behind a gate, or the listen socket may be down. This
        line says which, and it lands in the same log panel as everything else.
        """
        self._mic_level = max(getattr(self, "_mic_level", 0.0), float(np.abs(mono).max()))
        self._mic_samples = getattr(self, "_mic_samples", 0) + len(mono)
        if self._mic_samples < SAMPLE_RATE:
            return
        provider, level = self.provider, self._mic_level
        self._mic_samples, self._mic_level = 0, 0.0
        self.send_json({
            "type": "log",
            "kind": "mic.rx",
            "text": (
                f"peak={level:.3f} listening={provider._listening} "
                f"mic_enabled={provider._mic_enabled} agent_busy={provider._busy} "
                f"stt_socket={'up' if provider.stt._ws is not None else 'DOWN'} "
                f"vf={'on' if (self.voice_focus and self.voice_focus.enabled) else 'off'} "
                f"turn={provider.stt.turn_index}"
            ),
        })

    def on_message(self, data: dict) -> None:
        t = data.get("type")
        if t == "start":
            self.start()
        elif t == "stop":
            self.stop()
        elif t == "agent_playing" and self.controller:
            self.controller.on_agent_audio(bool(data.get("value")))
        elif t == "voice_focus" and self.voice_focus:
            on = self.voice_focus.set_enabled(bool(data.get("value")))
            self.send_json({"type": "voice_focus", "available": self.voice_focus.available, "on": on})
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
        "endpoint": os.environ.get("MODAL_ENDPOINT_URL", ""),
        "modal": os.environ.get("MODAL_API_KEY", ""),
        "deepgram": os.environ.get("DEEPGRAM_API_KEY", ""),
    }
    missing = [name for name, value in keys.items() if not value]
    if missing:
        raise SystemExit(f"Missing: {', '.join(missing)} (see .env.example).")

    app = web.Application()
    app["keys"] = keys
    app.add_routes(
        [
            web.get("/", index_handler),
            web.get("/app.js", app_js_handler),
            web.get("/ws", ws_handler),
        ]
    )
    host, port = "127.0.0.1", int(os.environ.get("PORT", "8080"))
    print(f"Tyto web demo on http://{host}:{port}  (Ctrl-C to stop)")
    web.run_app(app, host=host, port=port, print=None)


if __name__ == "__main__":
    main()
