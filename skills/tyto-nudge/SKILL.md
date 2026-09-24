---
name: tyto-nudge
description: Add ai-coustics Tyto real-time audio insight to an existing voice agent so it asks the user to fix their audio ("could you move somewhere quieter?") when background noise, other voices or a bad connection threaten the call. Works for Pipecat, LiveKit Agents, OpenAI Realtime, GPT-Live or a custom STT-LLM-TTS pipeline. Use when the user wants their agent to react to audio quality, background noise, or Tyto scores during a live call.
---

# Tyto nudge: make a voice agent react to bad audio

Goal: while the user talks, Tyto scores their audio. When the smoothed
`risk_score` is high and one fixable cause dominates, the agent stops and says
one line asking the user to fix it, then carries on.

Scope: **prompting the user only.** Do not add model routing, do not toggle
speech enhancement, do not change turn-taking, unless the user asks.

The policy is done. Copy [tyto_nudge.py](tyto_nudge.py) into the project
unchanged. Your job is the wiring: audio in, readings to the policy, the line
out through the agent.

## Before writing code

1. **Read the project first.** Find these four things and write each down with
   its file and line:
   - where the **user's inbound audio** is available, before any enhancement if
     possible (Tyto should score what the user actually sent)
   - how the app knows the **agent is speaking** (start and stop events)
   - how to **interrupt** the agent's current reply
   - how to make the agent **say a given line** (TTS speak, or a one-off
     response with instructions)
2. **Check versions.** Voice framework APIs change often. Look up the installed
   version's names in its source or docs. Do not rely on memory for frame,
   event or method names.
3. **Tyto setup.** `aic-sdk>=3.1` in Python (Node, Rust, C, C++ and WASM SDKs
   exist too), model `tyto-1.1-l-16khz`, key in `AIC_SDK_LICENSE`, kept on the
   server. Current API:
   <https://docs.ai-coustics.com/models/audio-insight/real-time-analysis>.
   If the ai-coustics docs MCP is available, query it.

## The wiring (every stack)

```
user audio ──> collector.buffer(block)          (audio path, cheap)
                     │
   worker, every N s: analyzer.analyze_buffered() ──> nudger.on_result(r) ──> line?
                                                                               │
                                             interrupt agent + mute mic ──> agent says line ──> resume
```

1. **Collector on the audio path, analyzer on a worker.** One pair per call:
   `collector, analyzer = aic.analyzer_pair(model, key)`. `collector.buffer()`
   takes exactly `config.block_size` mono float32 samples, so keep a residual
   buffer and emit fixed blocks. Never run `analyze_buffered()` inside the audio
   callback.
2. **Interval.** Start at 5 s, as the docs recommend. Go down to 1 s only after
   measuring CPU at your expected concurrency. If analysis runs longer than the
   interval, raise the interval instead of queueing jobs.
3. **Warm-up gate.** Count samples buffered since the last reset and do not
   analyze until you have 5 s of real audio. Before that the window is padded
   with silence. So the first reaction comes about 5 s into the user's speech.
   Tell the user this. It is expected, not a bug.
4. **Pause while the agent speaks.** On agent start: stop buffering. On agent
   stop: `analyzer.reset()`, `nudger.reset()`, zero the warm-up count. Otherwise
   the next score mixes old audio with new, or scores the agent's own voice
   through echo.
5. **Act on a line.** When `on_result` returns a string:
   1. Stop forwarding user audio to the agent (so noise does not start a turn).
   2. Interrupt the agent's current reply.
   3. Have the agent say the line exactly once, word for word.
   4. When that speech ends, resume listening and scoring (step 4).
   If the stack only offers "generate a reply with instructions", use:
   `Say exactly this in one short natural sentence and nothing else: "<line>"`.
6. **Thread safety.** Readings arrive on the analysis worker. Marshal the
   interrupt and speak calls onto the framework's event loop
   (`asyncio.run_coroutine_threadsafe`, `loop.call_soon_threadsafe`, or the
   framework's task queue).

## Stack notes

Verify every name below against the installed version before using it.

| Stack | User audio tap | Agent speaking | Interrupt | Say the line |
| --- | --- | --- | --- | --- |
| **LiveKit Agents** | Official plugin: `ai_coustics.Analyzer(model=..., analysis_interval=5.0)`, add `analyzer.collector` to the RoomIO frame processor (`FrameProcessorChain`). Results arrive as the `analysis_result` event, so no worker is needed | `AgentSession` `agent_state_changed` | `session.interrupt()` | `session.say(line)` with a TTS, or `session.generate_reply(instructions=...)` with a realtime model |
| **Pipecat** | A `FrameProcessor` that copies `InputAudioRawFrame.audio` (int16 PCM) into the collector and always passes the frame on | `BotStartedSpeakingFrame` / `BotStoppedSpeakingFrame` | push `InterruptionFrame` | push `TTSSpeakFrame(line)` (set `append_to_context=True` so the LLM knows it asked) |
| **OpenAI Realtime** (WebSocket) | The PCM you already send with `input_audio_buffer.append` | `response.output_audio.delta` / `response.done` | `response.cancel`, clear local playback, `input_audio_buffer.clear` | `response.create` with the line in `instructions` and `input: []` |
| **Custom STT, LLM, TTS** | Wherever raw mic frames enter the server | Your TTS playback start and end | Stop TTS playback and cancel the LLM stream | Send the line straight to TTS, skipping the LLM, and append it to the history as an assistant turn |

The reference repo has worked examples: branch `ver/python` (OpenAI Realtime and
GPT-Live), `ver/python-pipecat` and `ver/python-livekit` of
<https://github.com/ai-coustics/tyto-real-time-demo>.

## Rules to keep

- `risk_score` opens the gate, and a dimension picks the line. A high risk score
  with no dominant fixable cause never interrupts the user.
- Always act on the **smoothed** score (EMA, alpha 0.3), never on one raw
  window.
- `speaker_loudness` and `speaker_reverb` are informational. Never name them as
  the cause. `codec_degradation` is a transport problem, so never ask the user to
  fix it.
- One line per cause per episode, with a cooldown (`tyto_nudge.py` does both).
- The license key stays on the server. Never ship it to a browser or mobile
  client.
- Log every reading and every nudge (`risk_score`, cause, line, timestamp) so
  thresholds can be tuned on real calls. The defaults are demo values.

## Verify

1. Unit test the wiring with fake readings (dicts work with `on_result`): clean,
   then sustained noise, which should give one nudge; a single noisy spike, which
   should give none; high risk with only `codec_degradation`, which should give
   none.
2. Live: talk in a quiet room for 10 s, which should give no nudge. Then play a
   video or run a fan: the agent should stop and ask within about 5 to 10 s,
   once. Fix the noise and keep talking: it should not ask again.
3. Report which of these you ran and what you saw. If no key or mic was
   available, say that live verification is still to do.
