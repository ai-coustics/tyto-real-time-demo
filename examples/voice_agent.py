"""The full demo: a live voice agent that adapts to your acoustics.

Talk to the agent while Tyto scores your mic in real time. The agent adapts on
three layers, exactly like the browser reference:

    1 Aware    - a room note is injected into its instructions
    2 Tuned    - turn-taking goes patient when the room is noisy
    3 Reactive - it interrupts itself to nudge you when one issue dominates

The agent is a cascade: the ai-coustics VAD decides when you have stopped
talking, Inkling-Small hears the utterance and answers in text, and Deepgram
speaks it. This wires the scorer (Tyto over aic-sdk), the controller (the
provider-agnostic decision logic), and that cascade behind the provider seam.
The microphone is owned here so the same frames feed both Tyto and the VAD.

Capture runs at 16 kHz, which is native for the VAD, Tyto and Inkling alike, so
nothing resamples on the way in. The agent's own voice comes back at 24 kHz.

Run:
    uv pip install -e ".[agent]"
    export AIC_SDK_LICENSE=...        # https://developers.ai-coustics.com
    export INKLING_API_KEY=...        # Thinking Machines
    export DEEPGRAM_API_KEY=...       # https://console.deepgram.com
    uv run examples/voice_agent.py

Use headphones. There is no echo cancellation server-side, so on speakers the
agent would hear itself.
"""

from __future__ import annotations

import os
import sys

from tyto_voice.audio import SounddeviceSink
from tyto_voice.cascade import PLAYBACK_RATE, SAMPLE_RATE, CascadeProvider
from tyto_voice.controller import TytoController
from tyto_voice.decision import VAD_PROFILES
from tyto_voice.env import load_env
from tyto_voice.prompts import BASE_INSTRUCTIONS, GREETING
from tyto_voice.provider import Handlers
from tyto_voice.scorer import LiveTytoScorer



def _reading_of(controller):
    """(scores, age) for the provider, or None if Tyto has nothing to say yet.

    Age travels with the scores because Tyto is reset on every agent turn and
    needs a fresh 5 s window, so the newest reading is often not recent.
    """
    if controller is None or controller.scores is None:
        return None
    return controller.scores, controller.scores_age

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
        "inkling": os.environ.get("INKLING_API_KEY"),
        "deepgram": os.environ.get("DEEPGRAM_API_KEY"),
    }
    missing = [name for name, value in keys.items() if not value]
    if missing:
        sys.exit(f"Missing keys: {', '.join(missing)}. See .env.example.")

    log = make_logger()
    handlers = Handlers()  # filled in once the controller exists

    # The sink plays the agent locally and reports when it is audible.
    sink = SounddeviceSink(
        on_playing=lambda playing: controller.on_agent_audio(playing),
        sample_rate=PLAYBACK_RATE,
    )
    provider = CascadeProvider(
        handlers,
        license_key=keys["license"],
        inkling_key=keys["inkling"],
        deepgram_key=keys["deepgram"],
        instructions=BASE_INSTRUCTIONS,
        greeting=GREETING,
        audio_out=sink.write,
        audio_done=sink.notify_done,
        audio_flush=sink.flush,
        turn_detection=VAD_PROFILES["eager"],
        # The reading rides along with every turn, so "how do I sound?" is
        # answered in one round trip instead of two.
        scores=lambda: _reading_of(controller),
        # No echo cancellation on a raw output device: with barge-in on, the
        # agent would hear itself and cut itself off. Headphones do not help,
        # because the problem is the microphone, not the speaker.
        allow_barge_in=False,
        on_log=log,
    )

    # Score at the capture rate so one mic stream feeds Tyto and the VAD.
    scorer = LiveTytoScorer(
        keys["license"],
        sample_rate=SAMPLE_RATE,
        on_state=lambda state, text: log(f"tyto.{state}", text),
    )
    controller = TytoController(
        provider, scorer,
        room_advice=False,  # this agent is terse; it would speak the advice
        # No echo cancellation on a raw output device, so Tyto must stop while
        # the agent talks or it would score the agent's voice. The cost is that
        # short turns leave it with few readings; talk for longer to see it move.
        pause_scoring_while_speaking=True,
        on_update=make_updater(), on_log=log,
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
            blocksize=provider.block_size or scorer.block_size,
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
