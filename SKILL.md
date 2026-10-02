---
name: tyto-livekit
description: Add ai-coustics Tyto audio insight to an existing LiveKit Agents project, log the scores and make the agent ask the caller to pause background audio (podcast, TV, music) during the call. Use when the user wants in-call audio quality insight or reactions in LiveKit.
---

# Tyto audio insight for LiveKit Agents

Goal: diagnose the caller's audio during the call. Log Tyto scores every 5 s and, when background audio is sustained, have the agent politely ask the caller to pause or turn it down. Prompting the user only. Do not add model routing or toggle enhancement.

Tyto is an experimental model in LiveKit terms, so it is not in the official LiveKit ai-coustics plugin. It ships in the ai-coustics-maintained plugin `ai-coustics-livekit-plugin` (Node: `@ai-coustics/livekit-plugin`).

Source of truth: https://docs.ai-coustics.com/models/get-started/livekit-quickstart and https://docs.ai-coustics.com/models/audio-insight/tyto. If the ai-coustics docs MCP is available, use it to confirm APIs before writing code.

## 1. Check the project first

- Find the LiveKit entrypoint (`AgentSession`, `session.start`, RoomIO options) and the `Agent` subclass, if any. Note the `livekit-agents` version.
- Python: if `livekit-plugins-ai-coustics` (the official plugin) is installed, STOP and tell the user. It shares the `livekit.plugins.ai_coustics` namespace with `ai-coustics-livekit-plugin`. They need a separate environment.
- Keep the user's existing STT, LLM and TTS providers and agent instructions.
- If the project already has a VAD the user wants to keep, use `analyzer.collector` alone as the RoomIO frame processor and skip the ai-coustics VAD.

## 2. Install and provision

Pinned versions known to work together:

- Python 3.11: `ai-coustics-livekit-plugin==0.2.0`, `aic-sdk==3.2.0`, `livekit-agents==1.8.2`
- Node 24: `@ai-coustics/livekit-plugin@0.2.0`, `@ai-coustics/aic-sdk@0.24.0`, `@livekit/agents@1.9.0`

Add `AIC_SDK_LICENSE` to the env file the project already loads (key from https://developers.ai-coustics.com/). Never hardcode it or put it in client code.

Download models into `models/` (add `models/` to `.gitignore`):

```sh
python -c 'from aic_sdk import Model; from shutil import copyfile; copyfile(Model.download("tyto-1.1-l-16khz", "models"), "models/tyto.aicmodel")'
python -c 'from aic_sdk import Model; from shutil import copyfile; copyfile(Model.download("vad-ms-2.1-xxs-16khz", "models"), "models/vad.aicmodel")'
```

## 3. Wire Tyto into RoomIO (the two Tyto parts)

1. Load the model once per worker in `setup`: `proc.userdata["analysis_model"] = ai_coustics.Model.from_file("models/tyto.aicmodel")`.
2. Per room, create `ai_coustics.Analyzer(model=..., analysis_interval=5.0)` and chain its collector with the VAD: `ai_coustics.FrameProcessorChain(vad.processor, analyzer.collector)`, passed as `noise_cancellation=` in `room_io.AudioInputOptions(sample_rate=16000, num_channels=1, ...)`.

Rules:

- The collector MUST be in RoomIO. An Analyzer whose collector is not in the audio path receives no audio.
- Tyto only buffers the original audio and analyzes on its own interval, so it runs in parallel to the pipeline. Keep 5 s unless the user measured CPU at their room count.
- Create a new Analyzer per room. Do not close components manually; RoomIO closes the chain.

## 4. Log the scores

Subscribe with `@analyzer.on("analysis_result")`. `event.result` carries `risk_score` plus the supporting metrics: `noise`, `interfering_speech`, `speaker_reverb`, `packet_loss`, `codec_degradation`, `speaker_loudness`. Log at least `risk_score`, `noise` and `interfering_speech` at info level so the user can tune thresholds.

## 5. React: nudge the caller

Use this logic (match the user's code style):

- Constants: `NOISE_THRESHOLD = 0.5`, `INTERFERING_SPEECH_THRESHOLD = 0.5`, `CONSECUTIVE_WINDOWS = 2` (about 10 s), `NUDGE_COOLDOWN = 60.0` seconds.
- A window is noisy when `noise` or `interfering_speech` is above its threshold. A podcast or TV is speech, so it shows up mainly as `interfering_speech`.
- Count consecutive noisy windows; reset the count on a clean window. Act only at `CONSECUTIVE_WINDOWS`, outside the cooldown, and when no nudge is already pending.
- `NUDGE_INSTRUCTIONS`: ask briefly and politely to pause or turn down the background audio, continue where it left off, and ignore recent user messages that sound like background content.
- If `session.agent_state == "listening"`: call `session.generate_reply(instructions=NUDGE_INSTRUCTIONS, allow_interruptions=False)`. Do not wait for user silence; background speech keeps the VAD in "speaking". Interruptions must be off or the background audio cuts the nudge off.
- Otherwise the agent is mid-reply: set `agent.nudge_pending = True`. In the `Agent` subclass, override `on_user_turn_completed(self, turn_ctx, new_message)` and, when pending, `turn_ctx.add_message(role="system", content=NUDGE_INSTRUCTIONS)` and clear the flag.
- Keep state (counter, last nudge time) inside the per-room entrypoint, never at module level.

Packet loss and codec degradation are network problems the caller cannot fix by moving. Do not nudge on them; mention to the user they can be logged or used to confirm details.

## 6. Verify

- Run `python agent.py dev` and join from the LiveKit agent playground. Python `console` mode skips RoomIO, so Tyto gets no audio there.
- About 5 s after you start talking, `Tyto risk=...` lines appear every 5 s. No lines means the collector is not in RoomIO.
- Play a podcast next to the mic: `interfering_speech` climbs and after about 10 s the agent asks to pause it, once, then not again for 60 s.
- Disconnect and start a new room: scores resume without errors.

Report to the user: files changed, thresholds used and how to tune them. Do not claim detection quality you did not observe.