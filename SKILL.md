---
name: tyto-pipecat
description: Add ai-coustics Tyto audio insight to an existing Pipecat project, log the scores and make the bot ask the caller to pause background audio (podcast, TV, music) or move somewhere quieter during the call. Use when the user wants in-call audio quality insight or reactions in Pipecat.
---

# Tyto audio insight for Pipecat

Goal: diagnose the caller's audio during the call. Log Tyto scores every 5 s and, when background audio is sustained, have the bot politely ask the caller to pause it or turn it down. Prompting the user only. Do not add model routing or toggle enhancement.

Pipecat has no Tyto plugin. Pipecat ships ai-coustics enhancement (`AICFilter`) and VAD (`AICQuailVADAnalyzer`) only, so Tyto is wired with `aic-sdk` directly: a small `FrameProcessor` that copies mic audio into a Tyto collector, plus an asyncio task that analyzes on an interval.

Source of truth: https://docs.ai-coustics.com/models/audio-insight/real-time-analysis, https://docs.ai-coustics.com/models/audio-insight/tyto and https://docs.ai-coustics.com/models/get-started/pipecat-quickstart. If the ai-coustics docs MCP is available, use it to confirm APIs before writing code. Pipecat APIs move fast: check the installed version's source before trusting a frame or class name.

## 1. Check the project first

- Find the bot entrypoint (`Pipeline([...])`, `transport.input()`, `PipelineWorker` or the older `PipelineTask`, the runner) and the system prompt. Note the `pipecat-ai` and `aic-sdk` versions.
- Keep the user's existing transport, STT, LLM and TTS services and instructions.
- If the pipeline uses `AICFilter` as `audio_in_filter`, or any enhancement processor, STOP and tell the user: Tyto must score the original audio, and Pipecat applies `audio_in_filter` inside the transport, before any processor can see it. Either remove the filter or accept that Tyto scores enhanced audio (its meters then describe the enhancer, not the room).
- Note the input sample rate (`audio_in_sample_rate` on `PipelineParams` or the transport). 16 kHz is Tyto 1.1's native rate.

## 2. Install and provision

Versions known to work together (this repo): Python 3.11+, `aic-sdk>=3.1`, `pipecat-ai==1.8.1`. aic-sdk 3.x is required: Tyto 1.1 is refused by 2.x, and the model id must be `tyto-1.1-l-16khz` (the old `tyto-l-16khz` is refused).

Add `AIC_SDK_LICENSE` to the env file the project already loads (key from https://developers.ai-coustics.com/). Never hardcode it or put it in client code.

The model downloads from the ai-coustics CDN on first use. Add `models/` to `.gitignore`:

```python
model = aic.Model.from_file(aic.Model.download("tyto-1.1-l-16khz", "models"))
```

Load it once per process and share it across sessions.

## 3. Wire Tyto into the pipeline (the two Tyto parts)

Per session, create a pair and initialize the collector at the pipeline's input rate:

```python
collector, analyzer = aic.analyzer_pair(model, os.environ["AIC_SDK_LICENSE"])
config = aic.ProcessorConfig.optimal(model, sample_rate=SAMPLE_RATE)
collector.initialize(config)
```

1. **The tap.** A `FrameProcessor` placed directly after `transport.input()`. On `InputAudioRawFrame` it converts PCM16 to mono float32 (`np.frombuffer(frame.audio, "<i2").astype(np.float32) / 32768`), splits it into exact `config.block_size` chunks (carry the remainder to the next frame), calls `collector.buffer(chunk)` on each and counts buffered samples. It always pushes the frame on unchanged.
2. **The analysis loop.** An asyncio task started next to the worker and cancelled when the client disconnects. Every 5 s, if at least 5 s of real audio has been buffered since the last reset, run `result = await asyncio.to_thread(analyzer.analyze_buffered)`.

Rules:

- The tap MUST come before any enhancement, VAD or STT processor. Tyto scores the room, never the enhancer's output.
- Each `buffer()` call takes a flat mono `float32` array of exactly `block_size` samples. Not `(1, n)`. `ProcessorConfig.optimal` has no `num_channels`.
- Never run `analyze_buffered()` on the event loop or inside `process_frame`; it is CPU inference. `asyncio.to_thread` keeps the loop free and lets the result come back on the loop, where it is safe to queue frames.
- Warm-up gate: before a full window is buffered the analyzer pads with silence and the score is meaningless. Skip analysis until 5 s of audio is in.
- One pair per session. On session end, cancel the task and call `analyzer.terminate_session()`. To reuse a pair, call `analyzer.reset()` (clears the collector too) and zero the warm-up count.
- Keep 5 s unless the user measured CPU at their session count.

## 4. Log the scores

Each result carries `risk_score` (0..1, higher is worse) plus the dimensions: `noise`, `interfering_speech`, `speaker_reverb`, `speaker_loudness`, `packet_loss`, `codec_degradation`. Log at least `risk_score`, `noise` and `interfering_speech` at info level (Pipecat projects use `loguru`) so the user can tune thresholds.

`ProcessingNotAllowedError` can be raised mid-call (license, session terminated). Catch it in the loop, log it once and stop analyzing. Never let it kill the pipeline, and never keep acting on the last score.

## 5. React: nudge the caller

Use this logic (match the user's code style):

- Constants: `NOISE_THRESHOLD = 0.5`, `INTERFERING_SPEECH_THRESHOLD = 0.5`, `CONSECUTIVE_WINDOWS = 2` (about 10 s), `NUDGE_COOLDOWN = 60.0` seconds, `NUDGE_MAX_SECONDS = 8.0`.
- A window is noisy when `noise` or `interfering_speech` is above its threshold. A podcast or TV is speech, so it shows up mainly as `interfering_speech`.
- Count consecutive noisy windows; reset the count on a clean window. Act only at `CONSECUTIVE_WINDOWS`, outside the cooldown, and when no nudge is in flight.
- Pick the line from the stronger cause: background speech gets "It sounds like there's a TV or podcast playing near you. Could you pause it or turn it down?", noise gets "There's a lot of background noise on your side. Could you move somewhere quieter?".
- Speak it with `await worker.queue_frames([InterruptionFrame(), TTSSpeakFrame(text=line, append_to_context=True)])`. The interruption cuts a reply in progress; `append_to_context=True` records the line so the LLM knows it asked and does not repeat it. No LLM round trip, so it is fast.
- Close a gate in the tap (drop `InputAudioRawFrame` instead of pushing it) while the nudge plays. Otherwise the background audio triggers the user's turn and interrupts the nudge. Reopen it on `BotStoppedSpeakingFrame` (watch it with a `BaseObserver` on the worker, or let it pass through a processor after `transport.output()`), and also after `NUDGE_MAX_SECONDS` no matter what. A gate that never reopens looks exactly like a broken microphone.
- Pass `enable_interruptions` as the project already does; do not turn interruptions off globally to protect the nudge, the gate does that.
- Keep state (counter, last nudge time, gate flag) per session, inside the tap or the session object, never at module level.

Packet loss and codec degradation are network problems the caller cannot fix by moving. Do not nudge on them. `speaker_loudness` and `speaker_reverb` are informational only, never a nudge cause. Mention to the user they can be logged or added to the system prompt as context.

## 6. Verify

- Run the bot and connect from the client the project already uses (Pipecat's prebuilt WebRTC UI, Daily room, phone). Use headphones, or keep the client's echo cancellation on, so Tyto does not score the bot's own voice.
- About 5 s after you start talking, `Tyto risk=...` lines appear every 5 s. No lines means the tap is not after `transport.input()`, or the warm-up count never reaches 5 s (check the sample rate).
- Play a podcast next to the mic: `interfering_speech` climbs and after about 10 s the bot asks to pause it, once, then not again for 60 s. The nudge plays to the end without being cut off by the podcast.
- Disconnect and reconnect: scores resume without errors and no analysis task is left running from the old session.

Report to the user: files changed, thresholds used and how to tune them. Do not claim detection quality you did not observe.
