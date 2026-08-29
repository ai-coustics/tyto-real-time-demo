# Tyto voice agent, PhoneLLM reference

A Python port of the [Tyto](https://docs.ai-coustics.com) acoustics-aware
voice-agent demo. You talk to a live voice agent; in parallel **Tyto scores your
microphone in real time** with the ai-coustics Python SDK, and the agent adapts
to your acoustics on three layers: it stays aware of your room, retunes
turn-taking when it gets noisy, and nudges you when something is fixable ("Could
you turn the TV down?").

The agent on this branch is a cascade, the same stack as the Pipecat PhoneLLM
example:

```
Deepgram Flux (STT + turn-taking) → PhoneLLM Alpha 1 on Modal (LLM) → Deepgram Aura-2 (TTS)
```

[Pipecat PhoneLLM](https://huggingface.co/pipecat-ai/phonellm-alpha-1) is an
open-weights model from the Pipecat team, fine-tuned for voice agents that handle
phone calls: 32B total parameters with 3.5B active, accurate at tool calling,
fast, and cheap. Modal serves it as a one-command Auto Endpoint, which is what
makes it usable here without hosting anything.

The canonical browser reference in [index.html](index.html) (OpenAI Realtime over
WebRTC, fully client-side) is still the comparison point. The Tyto scoring
contract and the tuned constants are shared with it, so behavior is comparable
across stacks; where this branch deliberately diverges, it says so and why.

Both stacks run **Tyto 1.1** (`tyto-1.1-l-16khz`) on `aic-sdk` 3.x in Python and
`@ai-coustics/aic-sdk-wasm` 0.23.x in the browser. Tyto 1.1 keeps the same 5 s /
16 kHz mono contract as 1.0 while being much smaller and faster, and it changes
the metrics: the old background-media dimension is folded into
`interfering_speech` (competing speech from anything, live or a device),
`codec_degradation` is new, and the risk-score bands are now <0.30 good /
0.30-0.50 warn / >0.50 bad. Scores from 1.0 and 1.1 are not directly comparable.

## Tuned to be reactive

This branch is deliberately more twitchy than the browser reference, because the
interesting thing about a cascade is that every stage can be cancelled. Four
changes, all in [decision.py](src/tyto_voice/decision.py):

| Knob | Browser reference | Here | Why |
| --- | --- | --- | --- |
| `HOP_SECONDS` | 1.0 | **0.5** | Twice the readings, so half the delay before any layer reacts. Costs ~20% of one core. |
| `NUDGE_MIN_PERSIST` | 1 window (1 s) | **1 window (0.5 s)** | Fires on the first bad window. The EMA is what stops it being twitchy: one window only moves the smoothed score 30%. |
| `NUDGE_THRESHOLD_DEFAULT` | 0.40 | **0.31** | One point above the hysteresis floor, so the agent speaks up as soon as the room leaves the "good" band. |
| `NUDGE_COOLDOWN_SECONDS` | n/a | **10 s** | New here, and required: see below. |

And when a nudge trips, **both directions are cancelled, always**:

- **The audio in.** The mic is cut and the turn Flux was building is voided, so
  the half sentence the user was talked over is never transcribed and never
  answered.
- **The agent's own reply.** Any PhoneLLM request in flight is abandoned, the
  Aura-2 socket is cleared, and playback is flushed, so a reply already being
  spoken stops mid-word instead of the nudge queueing up behind it.

The web demo also keeps Tyto **measuring continuously**, including while the
agent talks (`pause_scoring_while_speaking=False`). The browser captures with
echo cancellation, so the agent's own voice is not in the signal and there is
nothing to protect against. This is what makes the second cancellation possible
at all: without readings during a reply, the Reactive layer could never fire
during one. It is also why the cooldown had to be added, since the browser
reference gets one for free from its re-warm after every agent turn.

## What is in here

Three things you can run:

| Demo | What it shows | Needs |
| --- | --- | --- |
| [examples/web/server.py](examples/web/server.py) | The full demo with the browser UI. Tyto scoring, the agent, and the keys all run on the Python backend; the browser is a thin client. | ai-coustics key + Modal endpoint + Deepgram key |
| [examples/score_mic.py](examples/score_mic.py) | Live Tyto scoring of your mic in the terminal, with the three layer decisions printed. No agent. | ai-coustics key + a mic |
| [examples/voice_agent.py](examples/voice_agent.py) | The full agent in the terminal (no UI), for headless or scripting use. | all three + headphones |

The web demo is the one to start with. It is the only one with echo
cancellation, so it is the only one where barge-in and mid-reply nudges work.

The reusable library lives in [src/tyto_voice](src/tyto_voice), split by job:

- `decision.py` - the scoring contract (`Scores`, tuned constants) and the
  decision layer (room note, turn-taking profile, nudge monitor). Pure Python,
  no dependencies, fully unit tested.
- `prompts.py` - the agent instructions and the Tyto background.
- `scorer.py` - `LiveTytoScorer`, real-time scoring over the aic-sdk streaming
  analyzer, with the warm-up gate and pause/resume that match the browser worker.
- `provider.py` - `VoiceProvider`, the one interface every voice backend hides
  behind. Swap backends by writing one subclass.
- `controller.py` - `TytoController`, the provider-agnostic glue that turns a
  score stream into the three adaptations and answers the `check_audio_quality`
  tool.
- `flux.py` - `FluxSTT`, Deepgram Flux: transcription and turn-taking in one
  websocket, including the eager end-of-turn events.
- `phonellm.py` - `PhoneLLMClient`, the Modal endpoint over plain
  `/v1/chat/completions`.
- `deepgram.py` - `DeepgramTTS`, the Aura-2 speak socket.
- `cascade.py` - `CascadeProvider`, the three of them wired behind the seam.
- `openai_realtime.py` - `OpenAIRealtimeProvider`, kept as the second
  implementation of the same seam.
- `audio.py` - `SounddeviceSink`, local speaker playback for the terminal agent.

## Provision PhoneLLM on Modal

```bash
uv tool install modal
modal setup
modal endpoint create --model pipecat-ai/phonellm-alpha-1
```

Modal provisions an Auto Endpoint: a production-ready, OpenAI-compatible
inference server. **Note the endpoint URL it prints**, you need it below. This
takes around 20 minutes, and the CLI cannot retrieve the URL afterwards
(`modal endpoint list` shows status but not the URL). If you lose it, it is on
the endpoint's page in the Modal dashboard (`modal dashboard`).

Check it is running, then create the proxy token the bot authenticates with:

```bash
modal endpoint list
modal curl <endpoint-url>/v1/models          # uses your local credentials

modal workspace proxy-tokens create          # save the ws-... secret now
modal workspace proxy-tokens allow <token-id> main   # only on RBAC workspaces
```

The token id and secret combine as `<token-id>.<token-secret>` (`wk-....ws-...`)
into the Bearer token the bot sends.

Endpoints scale to zero when idle, so the first request after a quiet period
returns 503 while the model spins up, which for a 30B model is minutes. The
cascade handles this: it warms the endpoint in the background at connect and
speaks a fixed greeting through the voice alone, so the demo is audible
immediately either way. Watch for `llm.ready` in the log panel.

To stop it permanently: `modal endpoint stop <endpoint-id>`.

## Run it

You also need an ai-coustics SDK license key from
<https://developers.ai-coustics.com> and a Deepgram key from
<https://console.deepgram.com>. The Tyto model (`tyto-1.1-l-16khz`) is downloaded
from the ai-coustics CDN on first run into `./models`. Put your keys in a `.env`;
everything loads it automatically.

```bash
uv venv
cp .env.example .env    # then fill in the four values

# the full demo with the browser UI (start here)
uv pip install -e ".[web]"
uv run examples/web/server.py        # then open http://localhost:8080

# or: live mic scoring in the terminal, no agent
uv pip install -e .
uv run examples/score_mic.py

# or: the agent in the terminal, no UI (use headphones)
uv pip install -e ".[agent]"
uv run examples/voice_agent.py
```

Real exported environment variables take precedence over `.env`. Install extras:
plain `.` for the mic scorer, `.[web]` for the browser demo, `.[agent]` for the
terminal agent, `.[dev]` for the tests.

## How keys and secrets are handled

Keys come from environment variables (or a `.env`) and stay on the backend. This
is the opposite of the browser reference, where each visitor pastes their own
keys: here the visitor's browser never sees a key.

- `AIC_SDK_LICENSE` runs the Tyto analyzer on the backend. Audio is scored on the
  server; nothing leaves it for scoring.
- `MODAL_ENDPOINT_URL` and `MODAL_API_KEY` reach PhoneLLM from the backend.
- `DEEPGRAM_API_KEY` opens both Deepgram sockets from the backend.

The browser only exchanges mic and agent audio with your server, so no key (or
ephemeral secret) is ever sent to it. Note that unlike the Tyto scoring, the
user's audio *does* leave your infrastructure: it goes to Deepgram for
transcription. Tyto itself does not.

## Which of the three Tyto layers are supported

All three, server-side, with the same tuned thresholds as the browser except
where the table above says otherwise:

1. **Aware** - a one-sentence room note is appended to the system message
   whenever the dominant cause changes. Fully supported. The examples pass
   `room_advice=False`: PhoneLLM is terse enough that it reads the note's advice
   out loud otherwise.
2. **Tuned** - turn-taking switches between an eager and a patient profile when
   the room is noisy. Fully supported: Flux takes its end-of-turn thresholds over
   a `Configure` control message, so profiles swap live with no reconnect. Eager
   ends turns at the lowest confidence Flux allows and speculates on the reply
   before the user has finished; patient raises the bar and stops speculating,
   because in a noisy room the guess is usually wrong.
3. **Reactive** - when the smoothed risk crosses the threshold and one cause the
   user can act on dominates, the agent interrupts with a single spoken nudge,
   then resumes. Fully supported, and cheaper than on any other branch: the nudge
   text is a fixed string from `decision.py`, so it goes straight to the voice
   with no model round trip. `codec_degradation` is deliberately excluded: it is
   a transport problem, so it only feeds the Aware note (confirm names and
   numbers) instead of asking the user to fix their room.

The `check_audio_quality` tool is wired as a function tool, so the user can ask
"how do I sound?" at any time. It is kept as a real tool call here rather than
pasted into every prompt because accurate tool calling is exactly what PhoneLLM
is fine-tuned for.

### Speculative turns

Flux emits `EagerEndOfTurn` when it thinks the user is probably done, before it
is sure, and guarantees that transcript will match the eventual `EndOfTurn` if
the user really has stopped. So the PhoneLLM request is fired on the guess:

- Guess holds: the reply is already in hand when the turn commits, and the
  model's latency disappears into the turn gap entirely.
- User keeps talking: `TurnResumed` throws the speculation away, and it cost
  nothing anybody heard.

An exact transcript match is the only licence to reuse a speculated reply.
Anything else is discarded and asked again, or the agent would answer a sentence
the user did not finish saying.

### Limitations and notes

- **Echo cancellation.** The web demo captures the mic in the browser with echo
  cancellation on, so speakers are fine, and barge-in and continuous scoring are
  both enabled there. The terminal agent plays through a raw output device with
  no cancellation, so use headphones, and both are off: with them on, Flux hears
  the agent as a turn, cuts it off, transcribes the echo, and the agent answers
  itself.
- **Audio transport.** Capture is PCM16 mono at 16 kHz, native for both Tyto and
  Flux, so nothing resamples on the way in. Aura-2 returns 24 kHz, so the browser
  keeps one AudioContext per direction.
- **Verification.** The decision layer, the controller state machine, the Flux
  turn-event routing and the cascade's speculation rules are covered by unit
  tests (`pytest`, no keys or hardware needed). The aic-sdk calls are verified
  against the installed package. The live audio path needs your own keys, a mic,
  and a browser to exercise, and has not been run end to end here.

## Deploy story

The web demo is the deployable one: a single aiohttp process serving the page and
one websocket per visitor, with keys in its environment. Put it behind TLS (the
mic needs a secure origin, so HTTPS or localhost) and it is shareable as a normal
web app. The terminal demos are local tools.

Tyto itself runs anywhere Python runs, on CPU, with no audio leaving the host, so
it also drops into an existing server-side voice pipeline (Pipecat, LiveKit
Agents, any cascaded stack) by feeding the same `LiveTytoScorer.feed` from
whatever already has the user's audio.

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
- PhoneLLM: <https://huggingface.co/pipecat-ai/phonellm-alpha-1>
- Deepgram Flux: <https://developers.deepgram.com/docs/flux/quickstart>
