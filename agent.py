import logging
import os
import shutil
import time
from datetime import datetime

from dotenv import load_dotenv
from livekit.agents import (
    Agent, AgentServer, AgentSession, JobContext, JobProcess,
    cli, inference, room_io,
)
from ai_coustics.livekit import VAD, Analyzer, FrameProcessorChain, Model

load_dotenv(".env.local")
logger = logging.getLogger("aic-example")

# The Tyto risk score alone decides when to nudge. 0.5 is the start of Tyto's
# "higher risk" band; a quiet room stays around 0.0-0.25.
RISK_THRESHOLD = 0.4  # compared against the smoothed risk
SMOOTHING = 0.7  # weight of the newest score in the moving average; lower is smoother
ANALYSIS_INTERVAL = 1.0  # seconds between scores of the rolling 5 s window
NUDGE_COOLDOWN = 60.0  # seconds between nudges

RECORDINGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recordings")

# The highest of these scores only picks the wording of the nudge, it never triggers one.
CAUSES = {
    "interfering_speech": "Sorry, I can hear other voices in the background. "
    "Could you pause that or turn it down?",
    "noise": "Sorry, it's quite noisy on your end. Could you move somewhere quieter?",
    "speaker_reverb": "You sound a bit far away. Could you move closer to your microphone?",
    "packet_loss": "Your audio is breaking up. Could you check your connection?",
    "codec_degradation": "The line sounds poor on my end. Could you check your connection?",
}


def top_cause(result) -> str:
    return max(CAUSES, key=lambda name: getattr(result, name))


def setup(proc: JobProcess):
    proc.userdata["vad_model"] = Model.from_file(
        "models/vad.aicmodel"
    )
    proc.userdata["analysis_model"] = Model.from_file(
        "models/tyto.aicmodel"
    )


async def save_recording(ctx: JobContext):
    # LiveKit records every session in stereo to a temp folder that is deleted after
    # the job: the user's raw mic (what Tyto scores) left, the agent right.
    source = ctx.session_directory / "audio.ogg"
    if not source.exists():
        logger.warning("No recording found at %s", source)
        return
    os.makedirs(RECORDINGS_DIR, exist_ok=True)
    path = os.path.join(RECORDINGS_DIR, f"{datetime.now():%Y%m%d-%H%M%S}.ogg")
    shutil.copyfile(source, path)
    logger.info("Saved recording: %s", path)


server = AgentServer(setup_fnc=setup)


@server.rtc_session(agent_name="aic-tyto-demo", on_session_end=save_recording)
async def entrypoint(ctx: JobContext):
    key = os.environ.get("AIC_SDK_LICENSE", "").strip()
    if not key:
        raise RuntimeError("Set AIC_SDK_LICENSE in .env.local and restart the agent")

    vad = VAD(model=ctx.proc.userdata["vad_model"], license_key=key)

    @vad.on("metrics_collected")
    def on_vad_metrics(metrics):
        logger.info("AIC VAD inference count: %s", metrics.inference_count)

    # Tyto always scores the latest 5 s of audio; this only sets how often.
    analyzer = Analyzer(
        model=ctx.proc.userdata["analysis_model"],
        license_key=key,
        analysis_interval=ANALYSIS_INTERVAL,
    )

    frame_processor = FrameProcessorChain(vad.processor, analyzer.collector)

    session = AgentSession(
        vad=vad,
        stt=inference.STT("deepgram/nova-3", language="multi"),
        llm=inference.LLM("openai/gpt-4.1-mini"),
        tts=inference.TTS("cartesia/sonic-3"),
    )

    smoothed_risk = 0.0
    last_nudge = 0.0

    @analyzer.on("analysis_result")
    def on_analysis(event):
        nonlocal smoothed_risk, last_nudge
        result = event.result
        # A moving average, so one risky second does not trigger a nudge on its own.
        smoothed_risk = SMOOTHING * result.risk_score + (1 - SMOOTHING) * smoothed_risk
        cause = top_cause(result)
        cooldown_left = NUDGE_COOLDOWN - (time.monotonic() - last_nudge)
        if smoothed_risk <= RISK_THRESHOLD:
            decision = "ok"
        elif cooldown_left > 0:
            decision = f"cooldown ({cooldown_left:.0f}s left)"
        else:
            decision = f"nudge ({cause})"

        logger.info(
            "Tyto #%d risk=%.2f smoothed=%.2f -> %s noise=%.2f interfering_speech=%.2f "
            "reverb=%.2f packet_loss=%.2f codec=%.2f",
            event.sequence, result.risk_score, smoothed_risk, decision, result.noise,
            result.interfering_speech, result.speaker_reverb, result.packet_loss,
            result.codec_degradation,
        )
        if not decision.startswith("nudge"):
            return

        last_nudge = time.monotonic()
        # Barge in right away: cut off any reply in progress (likely an answer to the
        # background audio) and speak the fixed line, skipping the LLM. Background speech
        # keeps the user "speaking", so the line must not be interruptible.
        session.interrupt(force=True)
        session.say(CAUSES[cause], allow_interruptions=False)

    await session.start(
        agent=Agent(instructions="You are a helpful voice assistant. Keep replies brief."),
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
