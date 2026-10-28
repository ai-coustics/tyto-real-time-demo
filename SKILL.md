---
name: tyto-livekit
description: Add ai-coustics Tyto audio insight to an existing LiveKit Agents project, log the scores and make the agent ask the caller to pause background audio (podcast, TV, music) during the call. Use when the user wants in-call audio quality insight or reactions in LiveKit.
---

# Tyto audio insight for LiveKit Agents

Goal: diagnose the caller's audio during the call. Log Tyto scores every 5 s and, when background audio is sustained, have the agent politely ask the caller to pause or turn it down. Prompting the user only. Do not add model routing or toggle enhancement.

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
2. Per room, create `Analyzer(model=..., license_key=key, analysis_interval=5.0)` and chain its collector with the VAD: `FrameProcessorChain(vad.processor, analyzer.collector)`, passed as `noise_cancellation=` in `room_io.AudioInputOptions(sample_rate=16000, num_channels=1, ...)`.

Rules:

- The collector MUST be in RoomIO. An Analyzer whose collector is not in the audio path receives no audio.
- Tyto only buffers the original audio and analyzes on its own interval, so it runs in parallel to the pipeline. Keep 5 s unless the user measured CPU at their room count.
- Create a new Analyzer per room. Do not close components manually; RoomIO closes the chain.

## 4. Log the scores

Subscribe with `@analyzer.on("analysis_result")`. `event.result` carries `risk_score` plus the supporting metrics: `noise`, `interfering_speech`, `speaker_reverb`, `packet_loss`, `codec_degradation`, `speaker_loudness`. Log `risk_score` and the dimensions used for wording at info level so the user can tune the threshold.

## 5. React: nudge the caller

Use this logic (match the user's code style):

- The Tyto `risk_score` is the ONLY trigger. Do not trigger on `noise`, `interfering_speech` or any other single dimension.
- Constants: `RISK_THRESHOLD = 0.5` (start of Tyto's "higher risk" band; a quiet room stays around 0.0 to 0.25), `CONSECUTIVE_WINDOWS = 2` (about 10 s), `NUDGE_COOLDOWN = 60.0` seconds.
- Count consecutive windows with `risk_score > RISK_THRESHOLD`; reset the count on a window below it. Act only at `CONSECUTIVE_WINDOWS`, outside the cooldown, and when no nudge is already pending.
- Wording only: pick the highest of `interfering_speech`, `noise`, `speaker_reverb`, `packet_loss`, `codec_degradation` and tell the LLM that cause with a matching request (pause the podcast, move somewhere quieter, move closer to the mic, check the network). Skip `speaker_loudness`; higher is not worse. This avoids "I hear people talking" when the cause is noise.
- The nudge instructions also say: be brief and polite, continue where it left off, and ignore recent user messages that sound like background content.
- If `session.agent_state == "listening"`: call `session.generate_reply(instructions=..., allow_interruptions=False)`. Do not wait for user silence; background audio keeps the VAD in "speaking". Interruptions must be off or the background audio cuts the nudge off.
- Otherwise the agent is mid-reply: store the instructions in `agent.pending_nudge`. In the `Agent` subclass, override `on_user_turn_completed(self, turn_ctx, new_message)` and, when set, `turn_ctx.add_message(role="system", content=self.pending_nudge)` and clear it.
- Keep state (counter, last nudge time) inside the per-room entrypoint, never at module level.

## 6. Verify

- Run `python agent.py dev` (or `lk agent dev` with the LiveKit CLI), then open your project's Agents dashboard, pick `aic-vad-demo` and click Launch Console. Agent Console joins a real room, so audio goes through RoomIO and the collector. Needs `livekit-agents` 1.5.2 or later.
- Do not use CLI `console` mode (`python agent.py console`): it skips RoomIO, so Tyto gets no audio there. Agent Console (browser) and console mode (terminal) are different tools.
- Use the Console's Audio pane to confirm your mic reaches the agent and the Events pane to see the nudge reply. Tyto scores are in the worker logs, not in the Console.
- About 5 s after you start talking, `Tyto risk=...` lines appear every 5 s. No lines means the collector is not in RoomIO.
- Stay quiet for a minute: `risk` stays well under 0.5 and the agent never nudges.
- Play a podcast next to the mic: once `risk` stays above 0.5 for about 10 s, the agent asks to pause it (the highest dimension should be `interfering_speech`), once, then not again for 60 s.
- Disconnect and start a new room: scores resume without errors.

Report to the user: files changed, thresholds used and how to tune them. Do not claim detection quality you did not observe.