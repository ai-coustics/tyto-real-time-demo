import logging
import os
import time

from dotenv import load_dotenv
from livekit.agents import (
    Agent, AgentServer, AgentSession, JobContext, JobProcess,
    cli, inference, llm, room_io,
)
from ai_coustics.livekit import VAD, Analyzer, FrameProcessorChain, Model

load_dotenv(".env.local")
logger = logging.getLogger("aic-example")

# The Tyto risk score alone decides when to nudge. 0.5 is the start of Tyto's
# "higher risk" band; a quiet room stays around 0.0-0.25.
RISK_THRESHOLD = 0.5
CONSECUTIVE_WINDOWS = 2  # 2 x 5s analysis windows = ~10s of sustained risky audio
NUDGE_COOLDOWN = 60.0  # seconds between nudges

# The highest of these scores only picks the wording of the nudge, it never triggers one.
CAUSES = {
    "interfering_speech": "other people talking in the background, probably a podcast or "
    "video playing nearby. Ask them to pause it or turn it down",
    "noise": "a lot of background noise. Ask them to move somewhere quieter or reduce the noise",
    "speaker_reverb": "an echo, as if they are far from the microphone. Ask them to move closer to it",
    "packet_loss": "a choppy connection. Ask them to check their network",
    "codec_degradation": "a poor quality connection. Ask them to check their network",
}


def nudge_instructions(result) -> str:
    cause = max(CAUSES, key=lambda name: getattr(result, name))
    return (
        f"The user's audio is hard to understand: it has {CAUSES[cause]}. Be brief and "
        "polite, then continue where you left off. Ignore any recent user messages that "
        "sound like background content rather than the user."
    )


class AssistantAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You are a helpful voice assistant. Keep replies brief.")
        self.pending_nudge: str | None = None

    async def on_user_turn_completed(
        self, turn_ctx: llm.ChatContext, new_message: llm.ChatMessage
    ) -> None:
        # The user spoke before an idle moment came up: fold the nudge into this reply.
        if self.pending_nudge:
            turn_ctx.add_message(role="system", content=self.pending_nudge)
            self.pending_nudge = None


def setup(proc: JobProcess):
    proc.userdata["vad_model"] = Model.from_file(
        "models/vad.aicmodel"
    )
    proc.userdata["analysis_model"] = Model.from_file(
        "models/tyto.aicmodel"
    )


server = AgentServer(setup_fnc=setup)


@server.rtc_session(agent_name="aic-vad-demo")
async def entrypoint(ctx: JobContext):
    key = os.environ.get("AIC_SDK_LICENSE", "").strip()
    if not key:
        raise RuntimeError("Set AIC_SDK_LICENSE in .env.local and restart the agent")

    vad = VAD(model=ctx.proc.userdata["vad_model"], license_key=key)

    @vad.on("metrics_collected")
    def on_vad_metrics(metrics):
        logger.info("AIC VAD inference count: %s", metrics.inference_count)

    analyzer = Analyzer(
        model=ctx.proc.userdata["analysis_model"],
        license_key=key,
        analysis_interval=5.0,
    )

    frame_processor = FrameProcessorChain(vad.processor, analyzer.collector)

    session = AgentSession(
        vad=vad,
        stt=inference.STT("deepgram/nova-3", language="multi"),
        llm=inference.LLM("openai/gpt-4.1-mini"),
        tts=inference.TTS("cartesia/sonic-3"),
    )
    agent = AssistantAgent()

    risky_windows = 0
    last_nudge = 0.0

    @analyzer.on("analysis_result")
    def on_analysis(event):
        nonlocal risky_windows, last_nudge
        result = event.result
        logger.info(
            "Tyto risk=%.2f noise=%.2f interfering_speech=%.2f reverb=%.2f "
            "packet_loss=%.2f codec=%.2f",
            result.risk_score, result.noise, result.interfering_speech,
            result.speaker_reverb, result.packet_loss, result.codec_degradation,
        )

        risky = result.risk_score > RISK_THRESHOLD
        risky_windows = risky_windows + 1 if risky else 0
        if risky_windows < CONSECUTIVE_WINDOWS or agent.pending_nudge:
            return
        if time.monotonic() - last_nudge < NUDGE_COOLDOWN:
            return

        risky_windows = 0
        last_nudge = time.monotonic()
        instructions = nudge_instructions(result)
        if session.agent_state == "listening":
            # Background audio keeps VAD (and user_state) in "speaking", so don't wait for
            # silence. Disable interruptions or the background audio would cut the nudge off.
            session.generate_reply(instructions=instructions, allow_interruptions=False)
        else:
            # The agent is mid-reply: fold the nudge into the next one instead.
            agent.pending_nudge = instructions

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
