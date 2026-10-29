---
name: tyto-pipecat
description: Add ai-coustics Tyto audio insight to an existing Pipecat project, log the scores, ask the caller to fix bad audio (podcast, noise, echo) during the call and flag bad calls at the end. Use when the user wants in-call audio quality insight or reactions in Pipecat.
---

# Tyto audio insight for Pipecat

Goal: diagnose the caller's audio during the call. Log Tyto scores every 5 s, have the bot ask the caller to fix sustained bad audio, and flag the call at the end. Prompting the user only. Do not add model routing or toggle enhancement.

Tyto ships in Pipecat itself as `AICTytoObserver` (`pipecat.observers.aic_tyto_observer`), from `pipecat-ai` 1.15.0. No separate ai-coustics plugin is needed.

Source of truth: https://docs.ai-coustics.com/models/audio-insight/real-time-analysis and Pipecat's example https://github.com/pipecat-ai/pipecat/blob/v1.15.0/examples/voice/voice-aicoustics-tyto.py. If the ai-coustics docs MCP is available, use it to confirm APIs before writing code. Read the installed `pipecat/observers/aic_tyto_observer.py` if anything is unclear.

## 1. Check the project first

- Find the `Pipeline`, the `PipelineWorker` (or older `PipelineTask`) and the context aggregators. Note the `pipecat-ai` version. Below 1.15.0 there is no `AICTytoObserver`: ask the user before upgrading.
- Keep the user's existing transport, STT, LLM and TTS services and system prompt.
- If the transport uses `AICFilter` as `audio_in_filter`, pass the same filter as `audio_filter=` to the observer so Tyto scores the audio before enhancement. Without a filter, the observer scores the transport's `InputAudioRawFrame` audio.

## 2. Install and provision

- Python 3.11: `pipecat-ai[aic,...]==1.15.0` (the `aic` extra pulls `aic-sdk` 3.3.x). Keep the user's other extras.
- Add `AIC_SDK_LICENSE` to the env file the project already loads (key from https://developers.ai-coustics.com/). Never hardcode it or put it in client code.
- No manual model download: the observer fetches `tyto-1.1-l-16khz` to `~/.cache/pipecat/aic-models` on first run. Use `model_path=` only if the user wants a local file.

## 3. Add the observer

```python
from pipecat.observers.aic_tyto_observer import AICTytoObserver

tyto = AICTytoObserver(license_key=key, degraded_threshold=0.3)
worker = PipelineWorker(pipeline, observers=[tyto])
```

Rules:

- The observer must be in the worker's `observers`. It is not a pipeline processor.
- It analyzes once per 5 s of received audio in a background thread, so it never blocks the pipeline. The interval is fixed.
- Create a new observer per session (inside `bot()`), never at module level.

## 4. Log the scores

`@tyto.event_handler("on_analysis_result")` receives `(observer, scores, audio_secs)`. `scores` has `risk_score`, `noise`, `interfering_speech`, `speaker_reverb`, `packet_loss`, `codec_degradation`, `speaker_loudness`. Log `risk_score` and the dimensions used for wording at info level so the user can tune the threshold.

## 5. React: nudge the caller

- The Tyto `risk_score` is the ONLY trigger. Do not trigger on a single dimension.
- Constants: `RISK_THRESHOLD = 0.5` (start of Tyto's "higher risk" band; a quiet room stays around 0.0 to 0.25), `CONSECUTIVE_WINDOWS = 2` (about 10 s), `NUDGE_COOLDOWN = 60.0` seconds.
- Count consecutive windows with `risk_score > RISK_THRESHOLD`; reset on a window below it. Act only at `CONSECUTIVE_WINDOWS`, outside the cooldown, and when no nudge is pending.
- Wording only: pick the highest of `interfering_speech`, `noise`, `speaker_reverb`, `packet_loss`, `codec_degradation` and speak a fixed line for it (pause the podcast, move somewhere quieter, move closer to the mic, check the connection). Skip `speaker_loudness`; higher is not worse.
- Speak it as a `TTSSpeakFrame(text)` with `frame.interruptible = False` (set after construction, it is not an init argument). Background audio keeps the VAD in "speaking" and would cut an interruptible nudge off.
- If the bot is mid-reply, hold the frame and queue it on the assistant aggregator's `on_assistant_turn_stopped`. Track the reply with `on_assistant_turn_started` / `on_assistant_turn_stopped(aggregator, message)`. Otherwise `await worker.queue_frames([frame])`.
- Keep state (counter, last nudge time, pending frame) inside `bot()`, never at module level.

## 6. Flag the call

`@tyto.event_handler("on_call_summary")` receives `(observer, summary)` once at the end, with `windows`, `mean`, `p95`, `max`, `degraded_fraction` (share of windows with `risk_score >= degraded_threshold`) and `dominant_dimension`. Flag the call when `degraded_fraction >= 0.2` and log it (or send it to the user's analytics).

## 7. Verify

- Run `python bot.py`, open http://localhost:7860, allow the mic and connect.
- About 5 s after connecting, `Tyto risk=...` lines appear every 5 s. No lines means the observer is not in `observers` or setup failed (check for `Tyto setup failed` in the log, usually the license key).
- Stay quiet for a minute: `risk` stays well under 0.5 and the bot never nudges.
- Play a podcast next to the mic: once `risk` stays above 0.5 for about 10 s, the bot asks to pause it, once, then not again for 60 s.
- Disconnect: a `Tyto call summary ... flagged=` line prints. Reconnect: scores resume without errors.

Report to the user: files changed, thresholds used and how to tune them. Do not claim detection quality you did not observe.
