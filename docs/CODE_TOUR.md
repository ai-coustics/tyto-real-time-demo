# Code tour: make a voice agent react to bad audio

On a call, a person who cannot hear you well says so: "sorry, it's loud where
you are, can you move somewhere quieter?" A voice agent cannot do that on its
own because it only gets a transcript. Tyto, the ai-coustics audio insight
model, scores the user's microphone every few seconds. With that score the agent
can react the same way a person would.

This page walks through the code in the order you would build it. It covers
about 150 lines across three files. The rest of the repo is UI, transport and
optional extras.

## 1. Score the user's audio in real time

File: [src/tyto_voice/scorer.py](../src/tyto_voice/scorer.py)

Tyto gives you two objects. The **collector** takes audio on your audio path
and the **analyzer** scores the latest 5 second window on a worker thread.

```python
import aic_sdk as aic

model = aic.Model.from_file(aic.Model.download("tyto-1.1-l-16khz", "./models"))
collector, analyzer = aic.analyzer_pair(model, license_key)
config = aic.ProcessorConfig.optimal(model, sample_rate=24000)
collector.initialize(config)

collector.buffer(block)                 # audio path: exact block_size, mono float32
result = analyzer.analyze_buffered()    # worker thread: scores the last 5 s
```

The three parts to read:

- [`feed()`](../src/tyto_voice/scorer.py#L121) turns audio chunks of any length
  into the fixed-size blocks the collector needs. Call it from whatever already
  has the user's audio (a mic callback, a WebSocket, a Pipecat or LiveKit frame).
- [`_loop()`](../src/tyto_voice/scorer.py#L191) runs the analysis once per
  second, but only after 5 s of real audio has arrived. Before that the window
  is padded with silence and the score would mean nothing. **So the first
  reaction comes after about 5 seconds of speech.**
- [`resume()`](../src/tyto_voice/scorer.py#L147) resets the analyzer and the smoothing after
  the agent has spoken. Scoring pauses while the agent talks, so older audio
  would skew the next score, and an old average would re-ask about a problem
  the user has just fixed. The mic itself keeps flowing to the agent so the user
  can interrupt it; it is muted only while a nudge plays.

## 2. What Tyto returns

Every reading is one `Scores` object
([decision.py:67](../src/tyto_voice/decision.py#L67)), all values 0 to 1, higher
is worse:

| Field | What it tells you |
| --- | --- |
| `risk_score` | **The one to act on.** How likely the audio is to make the agent fail. Bands: below 0.30 good, 0.30 to 0.50 warn, above 0.50 bad |
| `noise` | Background noise |
| `interfering_speech` | Other voices: people nearby or a TV or radio |
| `packet_loss` | Dropouts on the connection |
| `codec_degradation` | Heavy compression in transport |
| `speaker_reverb`, `speaker_loudness` | Informational only, never a reason to react |

Run `uv run examples/score_mic.py` to watch these update live in the terminal
with nothing else attached. Talk normally, then turn on a fan or play a video.

## 3. Decide when to react

File: [src/tyto_voice/decision.py](../src/tyto_voice/decision.py). Pure Python,
no dependencies, so you can copy it into any stack.

Two rules keep the agent from nagging:

1. **Smooth first.** Each reading is blended with the previous ones using an
   exponential moving average with alpha 0.3
   ([`Scores.ema`](../src/tyto_voice/decision.py#L91)), so a single cough or door
   slam does not trigger anything.
2. **The risk score opens the gate. A dimension says what to ask for.**
   [`EnvMonitor.evaluate`](../src/tyto_voice/decision.py#L309) fires only when the
   smoothed `risk_score` is at least 0.40 **and** one dimension the user can fix
   clearly dominates ([`strongest_cause`](../src/tyto_voice/decision.py#L234)). A
   high risk score without a clear cause never interrupts the user.

The dominant dimension picks the line
([`EXPLANATIONS`](../src/tyto_voice/decision.py#L174)):

| Trigger (smoothed) | The agent says |
| --- | --- |
| risk ≥ 0.40 and `noise` > 0.45 | "Sorry, there is a lot of background noise. Could you move somewhere quieter?" |
| risk ≥ 0.40 and `interfering_speech` > 0.35 | "Sorry, I am hearing other voices in the background. Could you move somewhere quieter, or turn down anything playing nearby?" |
| risk ≥ 0.40 and `packet_loss` > 0.15 | "Sorry, your connection seems unstable. Could you check it and try again?" |

`codec_degradation` has no line on purpose. The user cannot fix compression, so
the agent is only told to confirm names and numbers.

## 4. Make the agent say it

File: [src/tyto_voice/controller.py](../src/tyto_voice/controller.py)

[`on_scores`](../src/tyto_voice/controller.py#L148) receives every smoothed
reading. When the monitor returns a nudge,
[`_fire_nudge`](../src/tyto_voice/controller.py#L251) does three things:

1. Mutes the mic, so the user's noisy audio does not start a new turn.
2. Interrupts whatever the agent is saying.
3. Has the agent speak the nudge, then resumes listening once it has finished.

Only the last step depends on your voice stack. Here it goes through a small
interface, [provider.py](../src/tyto_voice/provider.py), with one implementation
per backend:

| Stack | Interrupt | Speak the nudge |
| --- | --- | --- |
| OpenAI Realtime ([openai_realtime.py](../src/tyto_voice/openai_realtime.py#L152)) | `response.cancel` | `response.create` with the line as instructions |
| GPT-Live 1 ([openai_live.py](../src/tyto_voice/openai_live.py#L233)) | flush playback and hold output | `session.commentary.append` |
| Pipecat (branch `ver/python-pipecat`) | `InterruptionFrame` | `TTSSpeakFrame(text)` |
| LiveKit Agents (branch `ver/python-livekit`) | `session.interrupt()` | `session.generate_reply(instructions=...)` |

## What you can skip on first read

- **Jev** ([jev.py](../src/tyto_voice/jev.py)) is an optional judge that picks
  *when* to say the nudge (cut in now, or wait until the agent finishes its
  sentence). Without `AI_GATEWAY_API_KEY` the rule above fires on its own.
- The **Aware** room note and **Tuned** turn-taking in `on_scores` go beyond
  prompting the user. They are worth a look later.
- `examples/web/` is the demo UI.

## Try it

1. Get an SDK key at <https://developers.ai-coustics.com>.
2. `cp .env.example .env` and set `AIC_SDK_LICENSE` and `OPENAI_API_KEY`.
3. `uv pip install -e ".[web]" && uv run examples/web/server.py`, open
   <http://localhost:8080>, talk, then turn on some noise.

Tyto runs in the Python, Rust, Node.js, C, C++ and WASM SDKs, and LiveKit has a
plugin. Docs:
<https://docs.ai-coustics.com/models/audio-insight/real-time-analysis>.
