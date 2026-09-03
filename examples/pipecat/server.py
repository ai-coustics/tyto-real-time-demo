"""The Tyto demo backend: a cascaded voice agent that adapts to your acoustics.

One process. The browser captures the microphone, plays the agent, and draws the
UI; everything else, the scoring, the three adaptation layers, and all three API
keys, lives here.

Per browser connection, one session::

    browser mic  -- WebRTC audio -->  Tyto scores it, then the agent hears it
    agent audio  <-- WebRTC audio --  Deepgram Aura-2
    scores / room / vad / nudge  <-- WebRTC data channel --  controller

The voice stack is a Pipecat cascade, Deepgram Flux to gpt-5-mini to Deepgram
Aura-2, built in [cascade.py](../../src/tyto_voice/cascade.py). The decision
layer, scorer, and controller are shared with every other Tyto frontend and know
nothing about it.

Run::

    uv pip install -e .
    # put AIC_SDK_LICENSE, DEEPGRAM_API_KEY and OPENAI_API_KEY in .env
    uv run examples/pipecat/server.py        # then open http://localhost:8080
"""

# NOTE: no ``from __future__ import annotations`` here. FastAPI resolves route
# parameter annotations as real objects, and the request types are imported
# locally inside build_app(); stringized annotations would not resolve and the
# ``request: Request`` param would be misread as a query parameter (HTTP 422).

import os
from pathlib import Path

from tyto_voice.cascade import SAMPLE_RATE, CascadeProvider
from tyto_voice.controller import TytoController
from tyto_voice.decision import VAD_PROFILES
from tyto_voice.env import load_env
from tyto_voice.prompts import BASE_INSTRUCTIONS, GREETING
from tyto_voice.provider import Handlers
from tyto_voice.scorer import LiveTytoScorer
from tyto_voice.voicefocus import VoiceFocus

HERE = Path(__file__).parent
INDEX = HERE / "index.html"
APP_JS = HERE / "app.js"


class Session:
    """One WebRTC connection wired to a scorer, provider, and controller."""

    def __init__(self, connection, keys: dict):
        self.connection = connection
        self.keys = keys
        self.scorer: LiveTytoScorer | None = None
        self.voice_focus: VoiceFocus | None = None
        self.provider: CascadeProvider | None = None
        self.controller: TytoController | None = None
        self._last_tyto_state: dict | None = None
        self._load_task = None
        self._stopped = False

    async def start(self) -> None:
        import asyncio

        handlers = Handlers()
        scorer = LiveTytoScorer(
            self.keys["license"],
            sample_rate=SAMPLE_RATE,
            on_state=self._on_tyto_state,
        )
        # Optional Quail enhancement on the agent's input only. Off by default:
        # a visitor should meet their room as it is, and the switch is what shows
        # the difference.
        voice_focus = VoiceFocus(
            self.keys["license"],
            sample_rate=SAMPLE_RATE,
            on_log=self._log,
        )

        provider = CascadeProvider(
            handlers,
            deepgram_key=self.keys["deepgram"],
            openai_key=self.keys["openai"],
            instructions=BASE_INSTRUCTIONS,
            greeting=GREETING,
            scorer=scorer,
            voice_focus=voice_focus,
            webrtc_connection=self.connection,
            turn_detection=VAD_PROFILES["eager"],
            on_client_message=self._on_client_message,
            on_connected=self._on_connected,
            on_log=self._log,
        )
        controller = TytoController(
            provider,
            scorer,
            # The agent is terse, and the room note's advice is phrased as an
            # instruction, so a terse agent reads it out loud. Give it the state
            # only and let it shape the tone.
            room_advice=False,
            # The browser captures with echo cancellation, so the agent's own
            # voice is not in the signal and Tyto can keep measuring straight
            # through a reply. That is what lets Layer 3 interrupt one.
            pause_scoring_while_speaking=False,
            on_update=self._on_update,
            on_log=self._log,
        )
        scorer.on_scores = controller.on_scores
        provider.audio_quality_fn = controller.audio_quality_snapshot

        handlers.on_ready = controller.on_ready
        handlers.on_agent_speaking = controller.on_agent_speaking
        handlers.on_agent_audio = controller.on_agent_audio
        handlers.on_user_transcript = controller.on_user_transcript
        handlers.on_agent_transcript = controller.on_agent_transcript
        # on_tool_call is intentionally unwired: check_audio_quality is answered
        # by a registered Pipecat function handler instead.

        self.scorer, self.provider, self.controller = scorer, provider, controller
        self.voice_focus = voice_focus

        # Start the voice stack FIRST, before the models load.
        #
        # Both loads hit the network for a manifest and then read a model off
        # disk, which is a second or two, and the browser starts sending audio
        # the moment the connection is up. Loading first meant that audio had
        # nowhere to go: the pipeline did not exist yet, so a visitor who spoke
        # straight after clicking the mic lost their first utterance and saw no
        # transcript for it. Neither component minds being called early: the
        # scorer drops audio until its collector exists, and Voice Focus is a
        # passthrough until its processor does.
        provider.connect()  # builds and runs the pipeline on this loop
        controller.set_connected(True)

        loop = asyncio.get_event_loop()

        async def _load_models() -> None:
            # A scorer failure is non-fatal: the agent still works, and the
            # error is shown in the browser. Same for Voice Focus, whose switch
            # is simply shown disabled. State emitted here is remembered and
            # re-sent once the data channel is up.
            #
            # Cancelling this task does not stop an executor thread that is
            # already inside a model load, so each step checks whether the
            # session died while it was waiting and tidies up after itself.
            # Otherwise a licensed SDK handle is installed after stop() ran and
            # is never released.
            try:
                await loop.run_in_executor(None, scorer.start)
                if self._stopped:
                    scorer.stop()
                    return
            except Exception as err:  # noqa: BLE001 - surface to the browser
                self._on_tyto_state("error", str(err))
            try:
                await loop.run_in_executor(None, voice_focus.start)
                if self._stopped:
                    voice_focus.stop()
                    return
            except Exception as err:  # noqa: BLE001 - optional feature
                self._log("error", str(err))
            self._send_voice_focus()

        self._load_task = asyncio.ensure_future(_load_models())

    def stop(self) -> None:
        # Order matters: mark the session dead first, so a model load still
        # running in an executor thread cannot install a live processor behind
        # the teardown, then release the watchdog before anything it touches.
        self._stopped = True
        if self._load_task and not self._load_task.done():
            self._load_task.cancel()
        if self.controller:
            self.controller.close()
        if self.scorer:
            self.scorer.stop()
        if self.voice_focus:
            self.voice_focus.stop()
        if self.provider:
            self.provider.disconnect()

    # -- UI plumbing (controller -> browser over the data channel) ---------- #

    def _on_update(self, state: dict) -> None:
        if "scores" in state:
            scores = state["scores"]
            self._send(
                {
                    "type": "scores",
                    "scores": scores.as_dict(),
                    "room": state.get("room", ""),
                    "vad": state.get("vad", "eager"),
                }
            )
        elif "transcript" in state:
            tx = state["transcript"]
            self._send(
                {"type": "transcript", "who": tx["who"], "text": tx["text"], "final": tx["final"]}
            )
        elif "nudge" in state:
            self._send({"type": "nudge", **state["nudge"]})

    def _on_tyto_state(self, state: str, text: str) -> None:
        self._last_tyto_state = {"type": "tyto_state", "state": state, "text": text}
        self._send(self._last_tyto_state)

    def _on_connected(self) -> None:
        # The data channel is up now, so this reliably reaches the browser even
        # if the early, pre-connection sends were dropped.
        self._send({"type": "status", "state": "live", "label": "Live"})
        if self._last_tyto_state:
            self._send(self._last_tyto_state)
        self._send_voice_focus()

    def _send_voice_focus(self) -> None:
        """The server owns this state: only it knows whether the model loaded."""
        vf = self.voice_focus
        self._send(
            {
                "type": "voice_focus",
                "available": bool(vf and vf.available),
                "on": bool(vf and vf.enabled),
            }
        )

    def _on_client_message(self, message: dict) -> None:
        kind = message.get("type")
        if kind == "nudge_threshold" and self.controller:
            self.controller.nudge_threshold = float(message.get("value", 0.31))
        elif kind == "voice_focus" and self.provider:
            # Answer with the state actually reached, not the one requested: it
            # stays off if the model is unavailable.
            self.provider.set_voice_focus(bool(message.get("value")))
            self._send_voice_focus()

    def _send(self, message: dict) -> None:
        if self.provider:
            self.provider.send_ui(message)

    def _log(self, kind: str, text: str) -> None:
        self._send({"type": "log", "kind": kind, "text": text})
        if kind == "error" and self.voice_focus is not None:
            # A process() failure switches Voice Focus off internally, so resend
            # the real state or the checkbox stays on and the note keeps
            # claiming it is cleaning the input.
            self._send_voice_focus()


# --------------------------------------------------------------------------- #
# FastAPI app: static files + the WebRTC offer endpoint                       #
# --------------------------------------------------------------------------- #


def build_app(keys: dict):
    from fastapi import FastAPI, Request
    from fastapi.responses import FileResponse, JSONResponse, Response
    from pipecat.transports.smallwebrtc.request_handler import (
        SmallWebRTCRequest,
        SmallWebRTCRequestHandler,
    )

    app = FastAPI()
    handler = SmallWebRTCRequestHandler(esp32_mode=False, host="127.0.0.1")
    sessions: dict[str, Session] = {}

    @app.get("/")
    async def index():
        return FileResponse(INDEX)

    @app.get("/app.js")
    async def app_js():
        return FileResponse(APP_JS, media_type="text/javascript")

    @app.get("/favicon.ico")
    async def favicon():
        return Response(status_code=204)

    @app.post("/api/offer")
    async def offer(request: Request):
        body = await request.json()
        webrtc_request = SmallWebRTCRequest.from_dict(body)

        # Fires only for a brand-new connection, not for the renegotiations the
        # handler manages internally, so it is the right place to spin up
        # exactly one Tyto session per visitor.
        async def on_new_connection(connection):
            session = Session(connection, keys)

            @connection.event_handler("closed")
            async def _on_closed(_conn):
                sessions.pop(connection.pc_id, None)
                session.stop()

            await session.start()
            sessions[connection.pc_id] = session

        answer = await handler.handle_web_request(
            request=webrtc_request, webrtc_connection_callback=on_new_connection
        )
        return JSONResponse(answer)

    return app


def main() -> None:
    import uvicorn

    load_env()
    keys = {
        "license": os.environ.get("AIC_SDK_LICENSE", ""),
        "deepgram": os.environ.get("DEEPGRAM_API_KEY", ""),
        "openai": os.environ.get("OPENAI_API_KEY", ""),
    }
    if not all(keys.values()):
        raise SystemExit(
            "Set AIC_SDK_LICENSE, DEEPGRAM_API_KEY and OPENAI_API_KEY (see .env.example)."
        )

    host, port = "127.0.0.1", int(os.environ.get("PORT", "8080"))
    print(f"Tyto demo on http://{host}:{port}  (Ctrl-C to stop)")
    uvicorn.run(build_app(keys), host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
