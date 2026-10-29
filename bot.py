import os
import time

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.aic_quail_vad import AICQuailVADAnalyzer
from pipecat.frames.frames import LLMRunFrame, TTSSpeakFrame
from pipecat.observers.aic_tyto_observer import AICTytoObserver
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openai.stt import OpenAIRealtimeSTTService
from pipecat.services.openai.tts import OpenAITTSService
from pipecat.transports.base_transport import TransportParams
from pipecat.workers.runner import WorkerRunner

load_dotenv(".env.local")

# The Tyto risk score alone decides when to nudge. 0.5 is the start of Tyto's
# "higher risk" band; a quiet room stays around 0.0-0.25.
RISK_THRESHOLD = 0.5
CONSECUTIVE_WINDOWS = 2  # 2 x 5s analysis windows = ~10s of sustained risky audio
NUDGE_COOLDOWN = 60.0  # seconds between nudges

# After the call: flag it when 20% or more of its windows had a risk of 0.3 or more.
DEGRADED_THRESHOLD = 0.3
FLAG_FRACTION = 0.2

# The highest of these scores only picks the wording of the nudge, it never triggers one.
CAUSES = {
    "interfering_speech": "Sorry, I can hear other voices in the background. "
    "Could you pause that or turn it down?",
    "noise": "Sorry, it's quite noisy on your end. Could you move somewhere quieter?",
    "speaker_reverb": "You sound a bit far away. Could you move closer to your microphone?",
    "packet_loss": "Your audio is breaking up. Could you check your connection?",
    "codec_degradation": "The line sounds poor on my end. Could you check your connection?",
}


def nudge_frame(scores) -> TTSSpeakFrame:
    cause = max(CAUSES, key=lambda name: getattr(scores, name))
    frame = TTSSpeakFrame(CAUSES[cause])
    # Background audio keeps the VAD in "speaking", so it would cut the nudge off.
    frame.interruptible = False
    return frame


async def bot(runner_args: RunnerArguments):
    key = os.environ.get("AIC_SDK_LICENSE", "").strip()
    if not key:
        raise RuntimeError("Set AIC_SDK_LICENSE in .env.local and restart the bot")

    transport = await create_transport(
        runner_args,
        {"webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True)},
    )

    stt = OpenAIRealtimeSTTService(api_key=os.environ["OPENAI_API_KEY"])
    llm = OpenAILLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAILLMService.Settings(
            system_instruction="You are a helpful voice assistant. Keep replies brief.",
        ),
    )
    tts = OpenAITTSService(api_key=os.environ["OPENAI_API_KEY"])

    tyto = AICTytoObserver(license_key=key, degraded_threshold=DEGRADED_THRESHOLD)

    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=AICQuailVADAnalyzer(license_key=key)),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )
    worker = PipelineWorker(
        pipeline,
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
        observers=[tyto],
    )
    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(worker)

    risky_windows = 0
    last_nudge = 0.0
    replying = False
    pending_nudge: TTSSpeakFrame | None = None

    @tyto.event_handler("on_analysis_result")
    async def on_analysis_result(observer, scores, audio_secs):
        nonlocal risky_windows, last_nudge, pending_nudge
        logger.info(
            f"Tyto risk={scores.risk_score:.2f} noise={scores.noise:.2f} "
            f"interfering_speech={scores.interfering_speech:.2f} "
            f"reverb={scores.speaker_reverb:.2f} packet_loss={scores.packet_loss:.2f} "
            f"codec={scores.codec_degradation:.2f}"
        )

        risky = scores.risk_score > RISK_THRESHOLD
        risky_windows = risky_windows + 1 if risky else 0
        if risky_windows < CONSECUTIVE_WINDOWS or pending_nudge:
            return
        if time.monotonic() - last_nudge < NUDGE_COOLDOWN:
            return

        risky_windows = 0
        last_nudge = time.monotonic()
        if replying:
            # The bot is mid-reply: speak the nudge once it finishes.
            pending_nudge = nudge_frame(scores)
        else:
            await worker.queue_frames([nudge_frame(scores)])

    @tyto.event_handler("on_call_summary")
    async def on_call_summary(observer, summary):
        flagged = summary.degraded_fraction >= FLAG_FRACTION
        logger.info(
            f"Tyto call summary: {summary.windows} windows, "
            f"mean risk {summary.mean.risk_score:.2f}, "
            f"{summary.degraded_fraction:.0%} degraded, "
            f"largest degradation: {summary.dominant_dimension}, flagged={flagged}"
        )

    @assistant_aggregator.event_handler("on_assistant_turn_started")
    async def on_assistant_turn_started(aggregator):
        nonlocal replying
        replying = True

    @assistant_aggregator.event_handler("on_assistant_turn_stopped")
    async def on_assistant_turn_stopped(aggregator, message):
        nonlocal replying, pending_nudge
        replying = False
        if pending_nudge:
            await worker.queue_frames([pending_nudge])
            pending_nudge = None

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        context.add_message({"role": "developer", "content": "Greet the user briefly."})
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        await runner.cancel()

    await runner.run()


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
