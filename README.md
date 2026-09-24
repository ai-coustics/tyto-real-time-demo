# Tyto real-time demo

On a bad call, people adapt: "Sorry, it's loud where you are. Could you move somewhere quieter?" This demo gives a voice agent the same instinct. When the user's audio gets bad, the agent stops and asks them to fix it, then carries on.

[Try it live](https://ai-coustics-tyto-demo--tyto-demo.modal.run/) (calls are time-capped, use headphones or speakers with echo cancellation).

## What Tyto gives you

Tyto (`tyto-1.1-l-16khz`) is the ai-coustics audio insight model. It scores the user's microphone in real time, per 5 second window. It runs on your CPU through the Python SDK, so no audio leaves your host for scoring.

Act on one number: `risk_score`, 0 to 1, the likelihood the audio makes your agent fail (below 0.30 good, 0.30 to 0.50 warn, above 0.50 bad). Six dimensions say why. The score decides when. The dimensions only pick the words:

| Dimension | The agent asks |
| --- | --- |
| `noise` | "Could you move somewhere quieter?" |
| `interfering_speech` | "Could you turn down anything playing nearby?" |
| `packet_loss` | "Could you check your connection?" |

## How the agent acts

The backend smooths the risk score with a moving average, so one spike does nothing. When it reaches 0.40 and one fixable cause dominates, the agent (OpenAI GPT-Live 1 by default) stops and asks the user to fix it. The first reaction comes after about 5 seconds of speech, once Tyto has a full window.

Optional extras: room-aware prompting, patient turn-taking in noise, and a Jev judge (`AI_GATEWAY_API_KEY`) that decides when to cut in. Without them the rule above works on its own.

## Start

Get an SDK key at [developers.ai-coustics.com](https://developers.ai-coustics.com).

```bash
uv venv
cp .env.example .env            # set AIC_SDK_LICENSE and OPENAI_API_KEY
uv pip install -e ".[web]"
uv run examples/web/server.py   # open http://localhost:8080
uv run examples/score_mic.py    # scores only, needs just AIC_SDK_LICENSE
```

- [Code tour](docs/CODE_TOUR.md): the ~150 lines that make the agent react, in build order.
- [tyto-nudge skill](skills/tyto-nudge/SKILL.md): lets a coding agent add this to Pipecat, LiveKit or a custom stack.
- [index.html](index.html) (browser-only version) and [AGENTS.md](AGENTS.md) (contributor context).
- [Real-time analysis docs](https://docs.ai-coustics.com/models/audio-insight/real-time-analysis). Other SDKs: Rust, Node.js, C, C++, WASM, and a LiveKit plugin.

## What is in here

Three things you can run:

| Demo | What it shows | Needs |
| --- | --- | --- |
| [examples/web/server.py](examples/web/server.py) | The full demo with the browser UI, same as the reference. Tyto scoring, the agent, Jev, and the keys all run on the Python backend; the browser is a thin client. | ai-coustics key + OpenAI key (+ Vercel AI Gateway key for Jev) |
| [examples/score_mic.py](examples/score_mic.py) | Live Tyto scoring of your mic in the terminal, with the three layer decisions printed. No agent. | ai-coustics key + a mic |
| [examples/voice_agent.py](examples/voice_agent.py) | The full agent in the terminal (no UI), for headless or scripting use. | ai-coustics key + OpenAI key (+ gateway key) + headphones |

The web demo is the one to start with. Its page follows the ai-coustics design
system and the look of the Audio Insight post-call demo (tokens in
[examples/web/ds](examples/web/ds)): the risk score, what the agent is doing
about the room, Jev's verdict, the six dimensions, the conversation, and an
activity feed. Every key stays on the server and Tyto runs in Python.

The reusable library lives in [src/tyto_voice](src/tyto_voice). It is small and
split by job, mirroring the commented sections of the browser reference:

- `decision.py` - the scoring contract (`Scores`, tuned constants) and the
  decision layer (room note, turn-taking profile, nudge monitor). Pure Python,
  no dependencies, fully unit tested. This is the part that is identical across
  every branch.
- `prompts.py` - the agent instructions and the Tyto background.
- `scorer.py` - `LiveTytoScorer`, real-time scoring over the aic-sdk streaming
  analyzer, with the warm-up gate and pause/resume that match the browser worker.
- `provider.py` - `VoiceProvider`, the one interface every voice backend hides
  behind. Swap backends by writing one subclass.
- `controller.py` - `TytoController`, the provider-agnostic glue that turns a
  score stream into the three adaptations and answers the `check_audio_quality`
  tool.
- `openai_live.py` - `OpenAILiveProvider`, the GPT-Live 1 WebSocket backend and
  the default. GPT-Live is full duplex, streams one continuous audio track
  (silence included) and has no cancel event, so the module docstring explains
  how each seam command maps onto it. Audio playback is delegated to a sink so
  the same provider drives a local speaker or a browser.
- `openai_realtime.py` - `OpenAIRealtimeProvider`, the OpenAI Realtime WebSocket
  backend (`VOICE_BACKEND=realtime`).
- `jev.py` - `JevJudge` and the `Situation` / `Decision` contract. Tyto's gate
  decides *that*, Jev decides *how*. The state Jev sees is bucketed words, the
  combination policy is code, and any failure degrades to the tuned rule.
- `backends.py` - `make_provider` / `make_judge` from environment variables,
  shared by the web and terminal entry points.
- `audio.py` - `SounddeviceSink`, local speaker playback for the terminal agent.

## Run it

You need an ai-coustics SDK license key from
<https://developers.ai-coustics.com> (and an OpenAI key for the agent). The
model (`tyto-1.1-l-16khz`) is downloaded from the ai-coustics CDN on first run
into `./models`. Put your keys in a `.env`; everything loads it automatically.

```bash
uv venv
cp .env.example .env    # then edit AIC_SDK_LICENSE, OPENAI_API_KEY, AI_GATEWAY_API_KEY

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
- `OPENAI_API_KEY` opens the GPT-Live (or Realtime) WebSocket connection from the
  backend. The browser only exchanges mic and agent audio with your server, never
  with OpenAI, so no key (or ephemeral secret) is ever sent to the browser.
- `AI_GATEWAY_API_KEY` calls Jev through Vercel AI Gateway
  (`https://ai-gateway.vercel.sh/typesafe`, model `typesafe-ai/jev`), billed to
  your gateway account. Only a few bucketed words about the situation leave the
  server, never audio or raw scores. Optional: without it the tuned rule nudges
  on its own. A `TYPESAFE_API_KEY` works too, direct to TypeSafe.

All entry points auto-load a `.env` from the project root (see
[.env.example](.env.example)); exported environment variables override it.

## Which of the three Tyto layers are supported

All three, server-side, with the same tuned thresholds as the browser:

1. **Aware** - a one-sentence room note is swapped into the agent instructions
   via `session.update` whenever the dominant cause changes. Fully supported.
2. **Tuned** - turn-taking switches between an eager and a patient profile when
   the room is noisy. On Realtime these are real `turn_detection` settings
   (`semantic_vad` vs `server_vad` with longer end-of-speech and a higher
   threshold), the same as the browser. GPT-Live owns turn-taking itself and
   exposes no VAD knobs, so there the swap becomes one appended turn-taking
   instruction (wait for a clear pause, ignore faint background voices). That is
   a prompt-level adaptation and the UI labels it as such.
3. **Reactive** - when the smoothed risk crosses the threshold and one cause the
   user can act on dominates, the agent stops and speaks a single nudge, then
   resumes. On Realtime that is `response.cancel` plus a one-shot
   `response.create`. On GPT-Live there is no cancel event and the model finishes
   its sentence before it speaks an injected line (measured: 3 to 4 s), so the
   provider flushes the playback buffer and holds the model's audio back until it
   pauses and starts the nudge. The listener hears the agent stop at once and the
   nudge from its first word. `codec_degradation` is deliberately excluded here:
   it is a transport problem, so it only feeds the Aware note (confirm names and
   numbers) instead of asking the user to fix their room.

The `check_audio_quality` tool is wired as an OpenAI function tool on Realtime
and as client delegation on GPT-Live (the model delegates "how do I sound?" to
the backend, which answers with the live Tyto reading), so the user can ask at
any time.

### Optional: how Jev judges the nudge

Layer 3 has two stages when a gateway key is present:

1. **Tyto trips the gate.** Same rule as before, in Python: smoothed risk at or
   above the slider threshold and one actionable cause dominating.
2. **Jev judges how to act.** From the moment a cause shows up in the warn band
   the controller asks Jev, about once a second, one request with three atomic
   questions: a Choice over the allowed actions, and two Nouls (is the user's
   last line them already dealing with it, is the agent mid-number). The state
   is a handful of named buckets (cause, "severe", "about 10 seconds", "getting
   worse", who is speaking, last words, "asked about this: never"), never raw
   scores, because Jev reads numbers poorly. Python combines the answers: low
   confidence falls back to the rule, a user already fixing it means stay silent,
   an agent mid-detail means wait for the sentence. By the time the gate trips a
   fresh verdict is usually already cached, so the nudge fires with no added
   wait; otherwise it fires when the in-flight answer lands (~300 ms warm).

Actions: `ask_now` cuts in and nudges; `ask_after_sentence` nudges as soon as the
agent falls silent (at most 6 s later); `adapt_quietly` and `stay_silent` leave
the Aware note to do the work and hold the gate for 4 s. Every verdict, its
confidence, latency and reason land in the event log, and the nudge banner
names the verdict that fired it. A Jev timeout or error is logged as a fallback and the rule fires as before.

### Limitations and notes

- **Echo cancellation.** The web demo captures the mic in the browser with echo
  cancellation on, so speakers are fine. The terminal agent plays through a raw
  output device with no cancellation, so use headphones there. In both, Tyto
  pauses while the agent speaks, and the mic is muted only while a nudge plays,
  so the user can still interrupt an ordinary reply.
- **Audio transport.** The browser reference uses WebRTC straight to OpenAI; here
  audio is PCM16 mono at 24 kHz, relayed browser to backend to OpenAI and back.
  Tyto is fed the same 24 kHz frames and resamples internally to its 16 kHz rate.
  The extra hop adds a little latency in exchange for keeping all logic and keys
  on the server.
- **GPT-Live specifics.** The output track is continuous, so "the agent is
  speaking" is read off the audio (RMS gate, 0.5 s hangover) and only voiced
  stretches reach the browser. Transcripts arrive as timed fragments about a
  second behind the audio, with no final marker; lines are cut on punctuation or
  after 1.2 s of silence. Instructions are immutable after start, so the Aware
  note and the Tuned profile are appended (`session.instructions.append`) rather
  than replaced. Scoring pauses while the agent speaks (an invariant shared with
  the browser), so a trip usually lands while the agent is quiet and the nudge
  is spoken right away; the interrupt hold covers the case where the agent has
  just started talking.
- **Jev reachability.** The decision path needs the gateway, so the demo warms
  the connection at session start and caps each request at 1.5 s. If Jev is slow
  or down the rule fires unchanged and the event log says `jev.fallback`.
- **Verification.** The decision layer, the controller state machine with and
  without the judge, the Jev contract (state shape, option masking, parsing,
  policy, timeout fallback) and the GPT-Live event mapping (strict session
  start, speech segmentation, the interrupt hold, appends, delegation) are
  covered by unit tests (`pytest`, no network). The GPT-Live events and the Jev
  gateway request/response were verified against the live APIs on 2026-09-22.
  The live audio path needs your own keys, a mic, and a browser to exercise.

## Deploy to Modal

The web demo runs on Modal as a `web_server` function (aiohttp is not ASGI, and
Modal proxies the websocket to a plain port). The Tyto model is baked into the
image; keys come from a Modal secret in the `tyto-demo` environment.

```bash
uv tool install modal && modal setup           # once
modal secret create tyto-demo-live-keys \
    AIC_SDK_LICENSE=... OPENAI_API_KEY=... AI_GATEWAY_API_KEY=... -e tyto-demo
modal deploy deploy/modal_app.py -e tyto-demo   # https://ai-coustics-tyto-demo--tyto-demo.modal.run/
```

The public deploy spends your credits on every call, so the server guards
itself, per container: at most `MAX_SESSIONS` (8) calls at once and
`MAX_STARTS_PER_HOUR` (60) new calls an hour, and every call ends after
`MAX_SESSION_SECONDS` (300). When the server sees a public client address it also
caps each visitor with `MAX_SESSIONS_PER_IP` (2) and `MAX_STARTS_PER_IP_HOUR` (12).
`X-Forwarded-For` is read only from peers listed in `TRUSTED_PROXIES`; Modal's
proxy sends none, so there only the global caps apply. All of them are env vars.

Deploying to the `tyto-demo` app name replaces whatever version lived there
(the URL is pinned by label, so bookmarks survive); `modal app rollback
tyto-demo -e tyto-demo` brings the previous version back. See
[deploy/modal_app.py](deploy/modal_app.py) for sizing and the reasoning.

## Deploy story

The web demo is the deployable one: it is a single aiohttp process serving the
page and one websocket per visitor, with keys in its environment. Put it behind
TLS (the mic needs a secure origin, so HTTPS or localhost) and it is shareable as
a normal web app. The terminal demos are local tools.

Tyto itself runs anywhere Python runs, on CPU, with no audio leaving the host, so
it also drops into an existing server-side voice pipeline (Pipecat, LiveKit
Agents, a cascaded STT to LLM to TTS stack) by feeding the same
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
- GPT-Live: <https://developers.openai.com/api/docs/guides/live>
- Jev / TypeSafe: <https://docs.typesafe.ai>, on Vercel AI Gateway:
  <https://vercel.com/docs/ai-gateway/sdks-and-apis/typesafe>
