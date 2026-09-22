"""The full demo: a live voice agent that adapts to your acoustics.

Talk to a GPT-Live 1 agent (or gpt-realtime with VOICE_BACKEND=realtime) while
Tyto scores your mic in real time. The agent adapts on three layers, exactly
like the browser reference:

    1 Aware    - a room note is injected into its instructions
    2 Tuned    - turn-taking goes patient when the room is noisy
    3 Reactive - it interrupts itself to nudge you when one issue dominates,
                 with Jev judging whether to cut in, wait, adapt or stay silent

This wires four pieces from the package: the scorer (Tyto over aic-sdk), the
controller (the provider-agnostic decision logic), the judge (Jev), and the
voice provider (the swappable backend). The microphone is owned here so the
same frames feed both Tyto and the agent.

Run:
    uv pip install -e ".[agent]"
    export AIC_SDK_LICENSE=...        # https://developers.ai-coustics.com
    export OPENAI_API_KEY=...         # https://platform.openai.com/api-keys
    export AI_GATEWAY_API_KEY=...     # optional: Jev via Vercel AI Gateway
    uv run examples/voice_agent.py

Use headphones. There is no echo cancellation server-side, so on speakers the
agent would hear itself.
"""

from __future__ import annotations

import os
import sys

from tyto_voice.env import load_env
from tyto_voice.audio import SounddeviceSink
from tyto_voice.backends import make_judge, make_provider
from tyto_voice.controller import TytoController
from tyto_voice.openai_live import SAMPLE_RATE  # both backends stream PCM16 at 24 kHz
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
        elif "jev" in state:
            j = state["jev"]
            print(f"  >> JEV {j['action']} p={j['confidence']:0.2f} {j['latency_ms']} ms: {j['reason']}")

    return update


def main() -> None:
    load_env()
    license_key = os.environ.get("AIC_SDK_LICENSE")
    openai_key = os.environ.get("OPENAI_API_KEY")
    if not license_key or not openai_key:
        sys.exit("Set AIC_SDK_LICENSE and OPENAI_API_KEY first.")

    log = make_logger()
    handlers = Handlers()  # filled in once the controller exists

    # The sink plays the agent locally and reports when it is audible.
    sink = SounddeviceSink(on_playing=lambda playing: controller.on_agent_audio(playing))
    provider = make_provider(
        handlers,
        api_key=openai_key,
        audio_out=sink.write,
        audio_done=sink.notify_done,
        audio_flush=sink.flush,
        on_log=log,
    )
    judge = make_judge(on_log=log)
    if judge is None:
        print("No AI_GATEWAY_API_KEY: Jev is off, the tuned rule nudges on its own.")

    # Score at the agent's sample rate so one mic stream feeds both.
    scorer = LiveTytoScorer(
        license_key,
        sample_rate=SAMPLE_RATE,
        on_state=lambda state, text: log(f"tyto.{state}", text),
    )
    controller = TytoController(provider, scorer, judge=judge, on_update=make_updater(), on_log=log)
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

    print("Connecting to the agent. Put your headphones on and say hello...\n")
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
        if judge:
            judge.close()
        print("\nStopped.")


if __name__ == "__main__":
    main()
