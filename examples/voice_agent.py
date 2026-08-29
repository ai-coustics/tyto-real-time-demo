"""The full demo: a live voice agent that adapts to your acoustics.

Talk to the agent while Tyto scores your mic in real time. The agent adapts on
three layers, exactly like the browser reference:

    1 Aware    - a room note is injected into its instructions
    2 Tuned    - turn-taking goes patient when the room is noisy
    3 Reactive - it interrupts itself to nudge you when one issue dominates

The agent is a cascade: Deepgram Flux hears you and decides when your turn is
over, Pipecat PhoneLLM on Modal answers in text, and Deepgram Aura-2 speaks it.
This wires the scorer (Tyto over aic-sdk), the controller (the provider-agnostic
decision logic), and that cascade behind the provider seam. The microphone is
owned here so the same frames feed both Tyto and Flux.

Capture runs at 16 kHz, native for both Tyto and Flux, so nothing resamples on
the way in. The agent's own voice comes back at 24 kHz.

Run:
    uv pip install -e ".[agent]"
    export AIC_SDK_LICENSE=...        # https://developers.ai-coustics.com
    export MODAL_ENDPOINT_URL=...     # modal endpoint create --model pipecat-ai/phonellm-alpha-1
    export MODAL_API_KEY=...          # <token-id>.<token-secret>
    export DEEPGRAM_API_KEY=...       # https://console.deepgram.com
    uv run examples/voice_agent.py

Use headphones. There is no echo cancellation on a raw output device, so on
speakers the agent would hear itself, and barge-in is off here for that reason.
"""

from __future__ import annotations

import os
import sys

from tyto_voice.audio import SounddeviceSink
from tyto_voice.cascade import PLAYBACK_RATE, SAMPLE_RATE, CascadeProvider
from tyto_voice.controller import CHECK_AUDIO_QUALITY_TOOL, TytoController
from tyto_voice.decision import VAD_PROFILES
from tyto_voice.env import load_env
from tyto_voice.prompts import BASE_INSTRUCTIONS, GREETING
from tyto_voice.provider import Handlers
from tyto_voice.scorer import LiveTytoScorer


def make_logger():
    def log(kind: str, text: str) -> None:
        print(f"  [{kind}] {text}"[:110])

    return log


def make_updater():
    def update(state: dict) -> None:
        if "transcript" in state:
            tx = state["transcript"]
            if tx["final"] and tx["text"]:
                print(f"  {tx['who']}: {tx['text']}")
        elif "nudge" in state:
            n = state["nudge"]
            print(f"  >> NUDGE ({n['label']} {n['value']:0.2f}): {n['text']}")

    return update


def main() -> None:
    load_env()
    keys = {
        "license": os.environ.get("AIC_SDK_LICENSE"),
        "endpoint": os.environ.get("MODAL_ENDPOINT_URL"),
        "modal": os.environ.get("MODAL_API_KEY"),
        "deepgram": os.environ.get("DEEPGRAM_API_KEY"),
    }
    missing = [name for name, value in keys.items() if not value]
    if missing:
        sys.exit(f"Missing: {', '.join(missing)}. See .env.example.")

    log = make_logger()
    handlers = Handlers()  # filled in once the controller exists

    # The sink plays the agent locally and reports when it is audible.
    sink = SounddeviceSink(
        on_playing=lambda playing: controller.on_agent_audio(playing),
        sample_rate=PLAYBACK_RATE,
    )
    provider = CascadeProvider(
        handlers,
        endpoint_url=keys["endpoint"],
        modal_key=keys["modal"],
        deepgram_key=keys["deepgram"],
        instructions=BASE_INSTRUCTIONS,
        greeting=GREETING,
        audio_out=sink.write,
        audio_done=sink.notify_done,
        audio_flush=sink.flush,
        turn_detection=VAD_PROFILES["eager"],
        tools=[CHECK_AUDIO_QUALITY_TOOL],
        # No echo cancellation on a raw output device: with barge-in on, Flux
        # would hear the agent as a turn and cut it off. Headphones do not help,
        # because the problem is the microphone, not the speaker.
        allow_barge_in=False,
        on_log=log,
    )

    # Score at the capture rate so one mic stream feeds Tyto and Flux.
    scorer = LiveTytoScorer(
        keys["license"],
        sample_rate=SAMPLE_RATE,
        on_state=lambda state, text: log(f"tyto.{state}", text),
    )
    controller = TytoController(
        provider,
        scorer,
        room_advice=False,  # this agent is terse; it would speak the advice
        # Same reason as barge-in: with no echo cancellation Tyto would score
        # the agent's own voice. The cost is that the Reactive layer cannot
        # interrupt a reply here, and that short turns leave Tyto few readings.
        # The web demo, which has cancellation, keeps it measuring throughout.
        pause_scoring_while_speaking=True,
        on_update=make_updater(),
        on_log=log,
    )
    scorer.on_scores = controller.on_scores  # route smoothed scores into the layers

    handlers.on_ready = controller.on_ready
    handlers.on_agent_speaking = controller.on_agent_speaking
    handlers.on_agent_audio = controller.on_agent_audio
    handlers.on_user_transcript = controller.on_user_transcript
    handlers.on_agent_transcript = controller.on_agent_transcript
    handlers.on_tool_call = controller.on_tool_call

    scorer.start()
    sink.start()
    provider.connect()
    controller.set_connected(True)

    import sounddevice as sd

    def mic_cb(indata, _frames, _time, _status):
        mono = indata[:, 0]
        scorer.feed(mono)
        provider.send_audio(mono)

    print("Listening. Put your headphones on and say hello...\n")
    try:
        with sd.InputStream(
            samplerate=scorer.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=scorer.block_size,
            callback=mic_cb,
        ):
            provider.closed.wait()
    except KeyboardInterrupt:
        pass
    finally:
        controller.set_connected(False)
        scorer.stop()
        sink.stop()
        provider.disconnect()
        print("\nStopped.")


if __name__ == "__main__":
    main()
