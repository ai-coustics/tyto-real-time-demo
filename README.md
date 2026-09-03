# Tyto voice agent, Python reference

A [Tyto](https://docs.ai-coustics.com) acoustics-aware voice-agent demo. You talk
to a live voice agent; in parallel **Tyto 1.1 scores your microphone in real
time** with the ai-coustics Python SDK, and the agent adapts to your acoustics on
three layers: it stays aware of your room, retunes turn-taking when it gets
noisy, and interrupts itself to say something when one problem takes over ("Could
you move somewhere quieter?").

The voice stack is a [Pipecat](https://pipecat.ai) cascade:

```
browser mic ──┬──▶ Tyto ──▶ scores ──▶ the three layers
              │      (always the raw microphone)
              └──▶ Voice Focus ──▶ Deepgram Flux ──▶ gpt-5-mini ──▶ Aura-2 ──▶ browser
                    (optional)      (STT + turns)                    (voice)
```

Tyto sits one hop after the microphone, so it scores exactly what the room is
doing, never what an enhancer made of it.

## What is in here

Two things you can run:

| Demo | What it shows | Needs |
| --- | --- | --- |
| [examples/pipecat/server.py](examples/pipecat/server.py) | The full demo with the browser UI. Tyto scoring, the agent, and all three keys run on the Python backend; the browser is a thin client. | all three keys |
| [examples/score_mic.py](examples/score_mic.py) | Live Tyto scoring of your mic in the terminal, with the three layer decisions printed. No agent, no voice stack. | ai-coustics key + a mic |

Start with the server demo. `score_mic.py` is the fastest way to see Tyto working
on its own, and it needs only the ai-coustics key.

The reusable library lives in [src/tyto_voice](src/tyto_voice), small and split by
job:

- `decision.py` - the scoring contract (`Scores`, tuned constants) and the
  decision layer (room note, turn-taking profile, nudge monitor). Pure Python,
  no dependencies, fully unit tested. Identical across every branch of this repo.
- `prompts.py` - the agent instructions, the Tyto background, and the greeting.
- `scorer.py` - `LiveTytoScorer`, real-time scoring over the aic-sdk streaming
  analyzer, with the warm-up gate and the pause/resume scoring gate.
- `provider.py` - `VoiceProvider`, the one interface every voice backend hides
  behind. Swap backends by writing one subclass.
- `controller.py` - `TytoController`, the provider-agnostic glue that turns a
  score stream into the three adaptations and answers `check_audio_quality`.
- `voicefocus.py` - `VoiceFocus`, optional Quail enhancement on the agent's
  input only. Off by default.
- `cascade.py` - `CascadeProvider`, the Pipecat cascade behind that seam. The one
  file that knows what Deepgram and OpenAI are.

## Run it

You need three keys. The Tyto model (`tyto-1.1-l-16khz`) is downloaded from the
ai-coustics CDN on first run into `./models`. Put the keys in a `.env`;
everything loads it automatically.

```bash
uv venv
cp .env.example .env    # then fill in the three keys

# the full demo with the browser UI (start here)
uv pip install -e .
uv run examples/pipecat/server.py    # then open http://localhost:8080

# or: live mic scoring in the terminal, no agent, no voice keys needed
uv run examples/score_mic.py
```

Real exported environment variables take precedence over `.env`. `.[dev]` adds
the test runner.

Python 3.11 or newer: Pipecat 1.8 requires it.

## How keys and secrets are handled

All three keys come from environment variables (or a `.env`) and stay on the
backend. The visitor's browser never sees a key, and never talks to Deepgram or
OpenAI directly; it exchanges audio only with your server.

- `AIC_SDK_LICENSE` runs the Tyto analyzer on the backend. Audio is scored on the
  server, on CPU, and nothing leaves the host for scoring.
- `DEEPGRAM_API_KEY` covers both ends of the cascade: Flux on the way in, Aura-2
  on the way out.
- `OPENAI_API_KEY` is gpt-5-mini in the middle.

## The three Tyto layers

All three run server-side, each one Pipecat frame:

1. **Aware** - a one-sentence room note is swapped into the agent's system
   message whenever the dominant cause changes, via `LLMMessagesTransformFrame`.
   The conversation history survives the swap, and the note never triggers a
   reply on its own.
2. **Tuned** - turn-taking switches between an `eager` and a `patient` profile
   when the room gets noisy, via `STTUpdateSettingsFrame`. Deepgram Flux does its
   own turn detection, so these are its real end-of-turn thresholds, changed
   mid-stream on the live socket. No separate VAD is involved.
3. **Reactive** - when the smoothed risk crosses the threshold and one cause
   dominates, an `InterruptionFrame` stops the reply mid-word and a
   `TTSSpeakFrame` puts one fixed line in the agent's mouth.

Layer 3 costs **no model round trip**. The nudge text is a constant in
`decision.py`, so it goes straight to the voice, which makes the Reactive layer
the fastest part of the demo rather than the slowest. The line is appended to the
context, so the agent knows it said it.

The `check_audio_quality` tool is registered on the LLM service, so the user can
ask "how do I sound?" at any time and get an answer from live Tyto numbers.

### Voice Focus

A switch in the UI, off by default, running the ai-coustics Quail VF 2.2
enhancement model (`quail-vf-2.2-l-16khz`) on the audio going to the agent.
Measured here at about 6% of one core for realtime.

**It never touches Tyto's copy.** In the pipeline the Tyto tap sits *before* the
Voice Focus processor, so the meters keep describing the room while the agent
hears the cleaned signal. Wiring it the other way round would have Tyto scoring
Quail's output: the meters would go green, the room note would go quiet and the
Reactive layer would stop firing, in a room that had not changed. It would still
look like it worked, which is why `test_cascade.py` asserts the ordering rather
than trusting a comment.

Enhance what the agent hears. Measure what the microphone heard.

It is off by default on purpose: the switch is what shows the difference, and a
difference needs a before. If the model does not load the switch is shown
disabled and everything else runs unchanged.

### Why a cascade

A cascade has more moving parts than a single speech-to-speech session, and buys
the ability to see and cancel every stage. That is what the Reactive layer needs:
this provider can stop a reply mid-word and speak its own line because both are
frames it owns, rather than state inside somebody else's session.

### gpt-5-mini settings

`reasoning_effort: "minimal"` and `verbosity: "low"` are load-bearing, not a
micro-optimization. Measured on this stack, three prompts each:

| settings | mean latency | reasoning tokens |
| --- | --- | --- |
| `minimal` + `low` | 1.23 s | 0 |
| defaults | 2.84 s | 64 to 400 |

On the defaults one reply spent its entire token budget reasoning and came back
**empty**, which a voice agent cannot use. `max_completion_tokens` is generous
(400) for the same reason: reasoning tokens are drawn from that budget.

## Limitations and notes

- **Echo cancellation.** The browser captures the mic with echo cancellation on,
  so speakers are fine and the agent's own voice is not in the signal. That is
  why the controller runs with `pause_scoring_while_speaking=False`: Tyto keeps
  measuring straight through a reply, which is what lets Layer 3 interrupt one.
- **Sample rates.** Capture is 16 kHz, which is both Tyto 1.1's optimal rate and
  Flux's native rate, so the same frames feed the scorer and the transcriber with
  no resampling. Playback is 24 kHz and only the browser hears it.
- **Tyto needs speech.** The analysis window is 5 s and the scorer will not
  report until it has a full window of real audio, so the agent opens with a
  question to get the visitor talking.
- **Verification.** The decision layer, controller state machine, and the frames
  each layer emits are covered by `pytest`. The aic-sdk path, the
  three network services, and the pipeline build are verified against the
  installed packages and live keys. The full live audio path needs your own keys,
  a mic, and a browser.

## Deploy story

A single uvicorn process serving the page and one WebRTC connection per visitor,
with keys in its environment. Put it behind TLS (the mic needs a secure origin,
so HTTPS or localhost) and it is shareable as a normal web app. Point
`AIC_MODELS_DIR` at a baked-in model directory so a container does not refetch
the model on every cold start.

Tyto itself runs anywhere Python runs, on CPU, with no audio leaving the host, so
it drops into an existing server-side voice pipeline by feeding the same
`LiveTytoScorer.feed` from whatever already has the user's audio.

## Develop

```bash
uv pip install -e ".[dev]"
uv run pytest -q
```

See [AGENTS.md](AGENTS.md) for an architecture map and the invariants to keep if
you extend this (it doubles as context for AI coding assistants).

## Links

- Tyto docs: <https://docs.ai-coustics.com>
- ai-coustics: <https://ai-coustics.com>
- Get an SDK key: <https://developers.ai-coustics.com>
- Python SDK: <https://github.com/ai-coustics/aic-sdk-py>
- Pipecat: <https://pipecat.ai>
