---
name: tyto-pipecat
description: Add ai-coustics Tyto audio insight to an existing Pipecat project, log the scores, ask the caller to fix bad audio (podcast, noise, echo) during the call and flag bad calls at the end. Use when the user wants in-call audio quality insight or reactions in Pipecat.
---

# Tyto audio insight for Pipecat

Goal: diagnose the caller's audio during the call. Log Tyto scores, have the bot ask the caller to fix sustained bad audio, and flag the call at the end. Prompting the user only. Do not add model routing or toggle enhancement.

Tyto ships in Pipecat itself as `AICTytoObserver` (`pipecat.observers.aic_tyto_observer`), from `pipecat-ai` 1.15.0 (Tyto 1.1) and 1.16.0 (Tyto 1.2). No separate ai-coustics plugin is needed.

Source of truth: https://docs.ai-coustics.com/models/audio-insight/real-time-analysis and Pipecat's example https://github.com/pipecat-ai/pipecat/blob/v1.16.0/examples/voice/voice-aicoustics-tyto.py. If the ai-coustics docs MCP is available, use it to confirm APIs before writing code. Read the installed `pipecat/observers/aic_tyto_observer.py` if anything is unclear.

## 1. Check the project first

- Find the `Pipeline`, the `PipelineWorker` (or older `PipelineTask`) and the context aggregators. Note the `pipecat-ai` version. Below 1.15.0 there is no `AICTytoObserver`, and 1.15.x runs Tyto 1.1: ask the user before upgrading to 1.16.0.
- Keep the user's existing transport, services and system prompt. Note whether the bot is a cascade (STT, LLM, TTS) or a speech-to-speech service such as `OpenAILiveLLMService` (GPT-Live 1); the nudge in step 5 differs.
- If the transport uses `AICFilter` as `audio_in_filter`, pass the same filter as `audio_filter=` to the observer so Tyto scores the audio before enhancement. Without a filter, the observer scores the transport's `InputAudioRawFrame` audio.

## 2. Install and provision

- Python 3.11: `pipecat-ai[aic,...]==1.16.0` (the `aic` extra pulls `aic-sdk` 3.4.x, model file format 8). Keep the user's other extras.
- `aic-sdk` 3.4 cannot download format 7 models such as `tyto-1.1-l-16khz`, `quail-vf-2.2-l-16khz` or `vad-2.1-xxs-16khz`. If the project sets `model_id=` on the observer, `AICFilter` or the AIC VAD, move it to a current model (`tyto-1.2-l-16khz`, `quail-vf-2.3-l-16khz`, `vad-ms-2.1-xxs-16khz`) or drop it to use the default.
- Add `AIC_SDK_LICENSE` to the env file the project already loads (key from https://developers.ai-coustics.com/). Never hardcode it or put it in client code.
- No manual model download: the observer fetches `tyto-1.2-l-16khz` to `~/.cache/pipecat/aic-models` on first run. Use `model_path=` only if the user wants a local file.

## 3. Add the observer

```python
from pipecat.observers.aic_tyto_observer import AICTytoObserver

tyto = AICTytoObserver(license_key=key, degraded_threshold=0.3)
worker = PipelineWorker(pipeline, observers=[tyto])
```

Rules:

- The observer must be in the worker's `observers`. It is not a pipeline processor.
- It analyzes the latest 5 s once per 5 s of received audio, in a background thread, so it never blocks the pipeline. There is no option for the interval. To react faster, set the module's private `aic_tyto_observer._TYTO_WINDOW_SECS = 1.0` before creating the observer: it then scores the rolling 5 s window every second (about 0.1 s of CPU per score).
- With the 1 s interval, the first scores come after 1-4 s of audio, padded with silence to 5 s. Tyto 1.2 scores those at risk 0.6-0.8 with `packet_loss` on top, even on clean audio: they nudge right after connect and count as degraded in the summary. Subclass the observer so the first score waits for a full window, as `bot.py` in this repo does:

  ```python
  class RollingTytoObserver(AICTytoObserver):
      def _reset_session(self):
          super()._reset_session()
          self._next_analysis_secs = 5.0
  ```

  Tell the user both overrides rely on private names in 1.16.0.
- Create a new observer per session (inside `bot()`), never at module level.

## 4. Log the scores

`@tyto.event_handler("on_analysis_result")` receives `(observer, scores, audio_secs)`. `scores` has `risk_score`, `noise`, `interfering_speech`, `speaker_reverb`, `packet_loss`, `codec_degradation`, `speaker_loudness`. Log `risk_score` and the dimensions used for wording at info level so the user can tune the threshold.

## 5. React: nudge the caller

- The Tyto `risk_score` is the ONLY trigger. Do not trigger on a single dimension.
- Constants: `NUDGE_COOLDOWN = 60.0` seconds. At the default 5 s interval, `RISK_THRESHOLD = 0.5` (start of Tyto's "higher risk" band; a quiet room stays around 0.0 to 0.25) and `CONSECUTIVE_WINDOWS = 2` (about 10 s). At a 1 s interval, smooth instead: `RISK_THRESHOLD = 0.4`, `SMOOTHING = 0.3`, `smoothed = SMOOTHING * risk + (1 - SMOOTHING) * smoothed`. Tyto 1.2 throws single-second spikes up to about 0.85 on calm audio; at 0.3 one spike stays under the threshold, a higher `SMOOTHING` lets it nudge.
- 5 s interval: count consecutive windows with `risk_score > RISK_THRESHOLD`, reset on a window below it, act at `CONSECUTIVE_WINDOWS`. 1 s interval: act when the smoothed risk is above `RISK_THRESHOLD`. Either way, only outside the cooldown.
- Wording only: pick the highest of `interfering_speech`, `noise`, `speaker_reverb`, `packet_loss`, `codec_degradation` and phrase the ask for it (pause the podcast, move somewhere quieter, move closer to the mic, check the connection). Skip `speaker_loudness`; higher is not worse.
- Speech-to-speech (GPT-Live 1, `OpenAILiveLLMService`): send the cause and the ask as commentary, `await llm.send_client_event(SessionCommentaryAppendEvent(delegation_id=None, content=...))` (from `pipecat.services.openai.live.events`). The model says it in its own words when it fits, and handles being talked over itself, so no pending or mid-reply logic is needed. Do not use `LLMMessagesAppendFrame`: as of 1.15.0 the Live service does not forward developer messages appended mid-session. Also tell it to ignore recent speech that sounds like background content.
- Cascade (STT, LLM, TTS): speak a fixed line as `TTSSpeakFrame(text)` with `frame.interruptible = False` (set after construction, it is not an init argument), or background audio keeps the VAD in "speaking" and cuts the nudge off. Queue it as `await worker.queue_frames([InterruptionFrame(), frame])`: the interruption cuts off any reply in progress (often an answer to the background audio itself), then the nudge plays.
- Keep state (counter, last nudge time) inside `bot()`, never at module level.

## 6. Flag the call

`@tyto.event_handler("on_call_summary")` receives `(observer, summary)` once at the end, with `windows`, `mean`, `p95`, `max`, `degraded_fraction` (share of windows with `risk_score >= degraded_threshold`) and `dominant_dimension`. Flag the call when `degraded_fraction >= 0.2` and log it (or send it to the user's analytics).

## 7. Verify

- Run `python bot.py`, open http://localhost:7860, allow the mic and connect.
- `Tyto ... risk=...` lines appear every 5 s (or every second with the faster interval), the first after 5 s of audio. No lines means the observer is not in `observers` or setup failed (check for `Tyto setup failed` in the log, usually the license key).
- Stay quiet for a minute: the bot never nudges. Single scores can still spike; the smoothed risk stays under the threshold.
- Play a podcast next to the mic: within about 10 s (5 s interval) or a few seconds (1 s interval) of sustained high risk, the bot asks to pause it, once, then not again for 60 s.
- Disconnect: a `Tyto call summary ... flagged=` line prints. Reconnect: scores resume without errors.

Report to the user: files changed, thresholds used and how to tune them. Do not claim detection quality you did not observe.
