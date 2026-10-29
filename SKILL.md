---
name: tyto-livekit
description: Add ai-coustics Tyto audio insight to an existing LiveKit Agents project, log the scores and make the agent ask the caller to pause background audio (podcast, TV, music) during the call. Use when the user wants in-call audio quality insight or reactions in LiveKit.
---

# Tyto audio insight for LiveKit Agents

Goal: diagnose the caller's audio during the call. Log Tyto scores every second and, when the smoothed risk stays high, have the agent politely ask the caller to pause or turn it down. Prompting the user only. Do not add model routing or toggle enhancement.

Tyto is not in the official LiveKit ai-coustics plugin. It ships in the ai-coustics LiveKit extras plugin `ai-coustics-livekit-extras` (Node: `@ai-coustics/livekit-extras`), imported from `ai_coustics.livekit`. It replaces the old `ai-coustics-livekit-plugin` 0.2.0; if the project still uses that, migrate per https://docs.ai-coustics.com/reference/livekit/migrate-to-livekit-extras.

Source of truth: https://docs.ai-coustics.com/models/get-started/livekit-quickstart and https://docs.ai-coustics.com/models/audio-insight/tyto. If the ai-coustics docs MCP is available, use it to confirm APIs before writing code.

## 1. Check the project first

- Find the LiveKit entrypoint (`AgentSession`, `session.start`, RoomIO options) and the `Agent` subclass, if any. Note the `livekit-agents` version.
- The official `livekit-plugins-ai-coustics` can share the environment (different namespace), but its VAD and VAD metadata are not interchangeable with the extras ones. Do not mix them in the chain. If the user wants official enhancement too, it goes after `vad.processor`; follow the migration guide.
- Keep the user's existing STT, LLM and TTS providers and agent instructions.
- If the project already has a VAD the user wants to keep, use `analyzer.collector` alone as the RoomIO frame processor and skip the ai-coustics VAD.

## 2. Install and provision

Pinned versions known to work together:

- Python 3.11: `ai-coustics-livekit-extras==0.25.2`, `aic-sdk==3.3.2`, `livekit-agents==1.8.2`
- Node 24: `@ai-coustics/livekit-extras@0.25.2`, `@ai-coustics/aic-sdk@0.25.2`, `@livekit/agents@1.9.0`

Add `AIC_SDK_LICENSE` to the env file the project already loads (key from https://developers.ai-coustics.com/). Never hardcode it or put it in client code.

Download models into `models/` (add `models/` to `.gitignore`):

```sh
python -c 'from aic_sdk import Model; from shutil import copyfile; copyfile(Model.download("tyto-1.1-l-16khz", "models"), "models/tyto.aicmodel")'
python -c 'from aic_sdk import Model; from shutil import copyfile; copyfile(Model.download("vad-ms-2.1-xxs-16khz", "models"), "models/vad.aicmodel")'
```

## 3. Wire Tyto into RoomIO (the two Tyto parts)

Import: `from ai_coustics.livekit import VAD, Analyzer, FrameProcessorChain, Model` (Node: `@ai-coustics/livekit-extras`).

1. Load the model once per worker in `setup`: `proc.userdata["analysis_model"] = Model.from_file("models/tyto.aicmodel")`.
2. Per room, create `Analyzer(model=..., license_key=key, analysis_interval=1.0)` and chain its collector with the VAD: `FrameProcessorChain(vad.processor, analyzer.collector)`, passed as `noise_cancellation=` in `room_io.AudioInputOptions(sample_rate=16000, num_channels=1, ...)`.

Rules:

- The collector MUST be in RoomIO. An Analyzer whose collector is not in the audio path receives no audio.
- Tyto only buffers the original audio and analyzes on its own interval, so it runs in parallel to the pipeline. The collector keeps a rolling 5 s window, so `analysis_interval=1.0` scores the latest 5 s every second (about 0.1 s of CPU per score). Raise it to 5.0 if CPU is tight at the user's room count.
- Create a new Analyzer per room. Do not close components manually; RoomIO closes the chain.

## 4. Log the scores

Subscribe with `@analyzer.on("analysis_result")`. `event.result` carries `risk_score` plus the supporting metrics: `noise`, `interfering_speech`, `speaker_reverb`, `packet_loss`, `codec_degradation`, `speaker_loudness`. Log `risk_score` and the dimensions used for wording at info level so the user can tune the threshold.

## 5. React: nudge the caller

Use this logic (match the user's code style):

- The Tyto `risk_score` is the ONLY trigger. Do not trigger on `noise`, `interfering_speech` or any other single dimension.
- Constants: `RISK_THRESHOLD = 0.4`, `SMOOTHING = 0.7`, `ANALYSIS_INTERVAL = 1.0`, `NUDGE_COOLDOWN = 60.0` seconds. 0.5 is the start of Tyto's "higher risk" band; a quiet room stays around 0.0 to 0.25.
- Smooth with a moving average on every result: `smoothed = SMOOTHING * risk_score + (1 - SMOOTHING) * smoothed`. Nudge when `smoothed > RISK_THRESHOLD` and the cooldown has passed. One risky second does not trigger on its own; sustained risk triggers within about 2 s. Lower `SMOOTHING` for a calmer trigger.
- Log one line per result with the raw risk, the smoothed risk and the decision (`ok`, `cooldown (Ns left)` or `nudge (cause)`), so the user can see why it did or did not nudge.
- Wording only: pick the highest of `interfering_speech`, `noise`, `speaker_reverb`, `packet_loss`, `codec_degradation` and speak a fixed line for it (pause the podcast, move somewhere quieter, move closer to the mic, check the connection). Skip `speaker_loudness`; higher is not worse. This avoids "I hear people talking" when the cause is noise.
- Barge in, whatever the agent or user is doing: `session.interrupt(force=True)` then `session.say(line, allow_interruptions=False)`. `say` skips the LLM and adds the line to the chat context. Interruptions must be off or the background audio cuts the nudge off. Do not wait for user silence; background audio keeps the VAD in "speaking".
- Keep state (smoothed risk, last nudge time) inside the per-room entrypoint, never at module level.

## 6. Record the call (optional)

`session.start` records by default (`record=True`) to `ctx.session_directory / "audio.ogg"`: stereo, the user's raw mic (what Tyto scored) left, the agent right. With LiveKit Cloud observability on, it is also uploaded to the session in the dashboard. The local folder is deleted after the job, so to keep a copy pass `on_session_end=save_recording` to `@server.rtc_session(...)` and copy the file there; the session is already closed and the file written at that point.

## 7. Verify

- Run `python agent.py dev` (or `lk agent dev` with the LiveKit CLI), then open your project's Agents dashboard, pick `aic-tyto-demo` and click Launch Console. Agent Console joins a real room, so audio goes through RoomIO and the collector. Needs `livekit-agents` 1.5.2 or later.
- Do not use CLI `console` mode (`python agent.py console`): it skips RoomIO, so Tyto gets no audio there. Agent Console (browser) and console mode (terminal) are different tools.
- Use the Console's Audio pane to confirm your mic reaches the agent and the Events pane to see the nudge reply. Tyto scores are in the worker logs, not in the Console.
- About a second after you start talking, a `Tyto #N risk=... smoothed=... -> ok` line appears every second. No lines means the collector is not in RoomIO.
- Stay quiet for a minute: `smoothed` stays well under 0.4 and the agent never nudges.
- Play a podcast next to the mic: within a few seconds `smoothed` crosses 0.4 and the agent cuts in, even mid-sentence, asking to pause it (the highest dimension should be `interfering_speech`), then logs `cooldown` for 60 s.
- Disconnect and start a new room: scores resume without errors.

Report to the user: files changed, thresholds used and how to tune them. Do not claim detection quality you did not observe.