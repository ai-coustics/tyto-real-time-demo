import logging
import os
import time

from dotenv import load_dotenv
from livekit.agents import (
    Agent, AgentServer, AgentSession, JobContext, JobProcess,
    cli, inference, llm, room_io,
)
from livekit.plugins import ai_coustics

load_dotenv(".env.local")
logger = logging.getLogger("aic-example")

# A background podcast is speech, so it shows up mainly as interfering_speech.
# noise catches intros, music beds and non-speech background audio.
NOISE_THRESHOLD = 0.5
INTERFERING_SPEECH_THRESHOLD = 0.5
CONSECUTIVE_WINDOWS = 2  # 2 x 5s analysis windows = ~10s of sustained background audio
NUDGE_COOLDOWN = 60.0  # seconds between nudges

NUDGE_INSTRUCTIONS = (
    "The user's microphone is picking up other people talking in the background, "
    "probably a podcast or video playing nearby. Briefly and politely ask them to pause "
    "it or turn it down so you can hear them clearly, then continue where you left off. "
    "Ignore any recent user messages that sound like podcast content rather than the user."
)


class AssistantAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You are a helpful voice assistant. Keep replies brief.")
        self.nudge_pending = False

    async def on_user_turn_completed(
        self, turn_ctx: llm.ChatContext, new_message: llm.ChatMessage
    ) -> None:
        # The user spoke before an idle moment came up: fold the nudge into this reply.
        if self.nudge_pending:
            turn_ctx.add_message(role="system", content=NUDGE_INSTRUCTIONS)
            self.nudge_pending = False


def setup(proc: JobProcess):
    proc.userdata["vad_model"] = ai_coustics.Model.from_file(
        "models/vad.aicmodel"
    )
    proc.userdata["analysis_model"] = ai_coustics.Model.from_file(
        "models/tyto.aicmodel"
    )


server = AgentServer(setup_fnc=setup)


@server.rtc_session(agent_name="aic-vad-demo")
async def entrypoint(ctx: JobContext):
    key = os.environ.get("AIC_SDK_LICENSE", "").strip()
    if not key:
        raise RuntimeError("Set AIC_SDK_LICENSE in .env.local and restart the agent")

    vad = ai_coustics.VAD(model=ctx.proc.userdata["vad_model"], license_key=key)

    @vad.on("metrics_collected")
    def on_vad_metrics(metrics):
        logger.info("AIC VAD inference count: %s", metrics.inference_count)

    analyzer = ai_coustics.Analyzer(
        model=ctx.proc.userdata["analysis_model"],
        analysis_interval=5.0,
    )

    frame_processor = ai_coustics.FrameProcessorChain(vad.processor, analyzer.collector)

    session = AgentSession(
        vad=vad,
        stt=inference.STT("deepgram/nova-3", language="multi"),
        llm=inference.LLM("openai/gpt-4.1-mini"),
        tts=inference.TTS("cartesia/sonic-3"),
    )
    agent = AssistantAgent()

    noisy_windows = 0
    last_nudge = 0.0

    @analyzer.on("analysis_result")
    def on_analysis(event):
        nonlocal noisy_windows, last_nudge
        result = event.result
        logger.info(
            "Tyto risk=%.2f noise=%.2f interfering_speech=%.2f",
            result.risk_score, result.noise, result.interfering_speech,
        )

        noisy = (
            result.noise > NOISE_THRESHOLD
            or result.interfering_speech > INTERFERING_SPEECH_THRESHOLD
        )
        noisy_windows = noisy_windows + 1 if noisy else 0
        if noisy_windows < CONSECUTIVE_WINDOWS or agent.nudge_pending:
            return
        if time.monotonic() - last_nudge < NUDGE_COOLDOWN:
            return

        noisy_windows = 0
        last_nudge = time.monotonic()
        if session.agent_state == "listening":
            # The podcast keeps VAD (and user_state) in "speaking", so don't wait for
            # silence. Disable interruptions or the podcast itself would cut the nudge off.
            session.generate_reply(instructions=NUDGE_INSTRUCTIONS, allow_interruptions=False)
        else:
            # The agent is mid-reply: fold the nudge into the next one instead.
            agent.nudge_pending = True

    await session.start(
        agent=agent,
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                sample_rate=16000,
                num_channels=1,
                noise_cancellation=frame_processor,
            ),
        ),
    )
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(server)
