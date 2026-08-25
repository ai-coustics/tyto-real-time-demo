# Tyto voice agent, Python reference

A Python port of the [Tyto](https://docs.ai-coustics.com) acoustics-aware
voice-agent demo. You talk to a live voice agent; in parallel **Tyto scores your
microphone in real time** with the ai-coustics Python SDK, and the agent adapts
to your acoustics on three layers: it stays aware of your room, retunes
turn-taking when it gets noisy, and nudges you when something is fixable ("Could
you turn the TV down?").

This branch is the server-side sibling of the browser reference in
[index.html](index.html) (OpenAI Realtime over WebRTC, fully client-side). The
Tyto scoring contract and the tuned constants are identical to that reference so
behavior is comparable across stacks.

The agent here is a **cascade** rather than a speech-to-speech session:

```
mic ──> ai-coustics VAD 2.1 ──(utterance)──> Inkling-Small ──(text)──> Deepgram Aura-2 ──> speaker
         turn-taking                          hears audio,             the voice
                                              replies in text
```

Inkling-Small is multimodal, so the whole utterance goes to it as audio; that is
the only path that decides anything. A Deepgram speech-to-text call runs
alongside it purely as a caption for the UI.

The Tinker endpoint serves many other models, and `INKLING_MODEL` will point the
demo at any of them. Inkling-Small is the one it is built on, and the choice is
measured rather than assumed: see the benchmark table in
[inkling.py](src/tyto_voice/inkling.py). The short version is that Inkling is the
only model there that honours `reasoning_effort`, and every other one spent its
whole token budget thinking and returned an empty reply. `OpenAIRealtimeProvider` is
still in the tree as the second implementation of the same provider seam.

Both stacks run **Tyto 1.1** (`tyto-1.1-l-16khz`) on `aic-sdk` 3.x in Python and
`@ai-coustics/aic-sdk-wasm` 0.23.x in the browser. Tyto 1.1 keeps the same 5 s / 16 kHz mono contract as 1.0
while being much smaller and faster, and it changes the metrics: the old
background-media dimension is folded into `interfering_speech` (competing speech
from anything, live or a device), `codec_degradation` is new, and the risk-score
bands are now <0.30 good / 0.30-0.50 warn / >0.50 bad. Scores from 1.0 and 1.1
are not directly comparable.

## What is in here

Three things you can run:

| Demo | What it shows | Needs |
| --- | --- | --- |
| [examples/web/server.py](examples/web/server.py) | The full demo with the browser UI, same as the reference. Tyto scoring, the agent, and the keys all run on the Python backend; the browser is a thin client. | ai-coustics + Inkling + Deepgram keys |
| [examples/score_mic.py](examples/score_mic.py) | Live Tyto scoring of your mic in the terminal, with the three layer decisions printed. No agent. | ai-coustics key + a mic |
| [examples/voice_agent.py](examples/voice_agent.py) | The full agent in the terminal (no UI), for headless or scripting use. | ai-coustics + Inkling + Deepgram keys + headphones |

The web demo is the one to start with: it is the visual UI from the browser
reference, but every key stays on the server and Tyto runs in Python.

The reusable library lives in [src/tyto_voice](src/tyto_voice). It is small and
split by job, mirroring the commented sections of the browser reference:

- `decision.py` - the scoring contract (`Scores`, tuned constants) and the
  decision layer (room note, turn-taking profile, nudge monitor). Pure Python,
  no dependencies, fully unit tested. This is the part that is identical across
  every branch.
- `prompts.py` - the agent instructions. Short on purpose: it is re-sent every
  turn, and each rule in it exists because the model broke it without one.
- `vad.py` - `LiveVad`, turn-taking over the ai-coustics VAD 2.1 model. Turns a
  block stream into whole utterances, with a pre-roll so the first syllable
  survives.
- `inkling.py` - `InklingClient`, audio in and text out, with bounded history.
- `deepgram.py` - `DeepgramTTS`, the persistent Aura-2 speak socket, and
  `transcribe`, the caption for the UI panel.
- `cascade.py` - `CascadeProvider`, the VAD to Inkling to Deepgram backend
  behind the provider seam.
- `scorer.py` - `LiveTytoScorer`, real-time scoring over the aic-sdk streaming
  analyzer, with the warm-up gate and pause/resume that match the browser worker.
- `provider.py` - `VoiceProvider`, the one interface every voice backend hides
  behind. Swap backends by writing one subclass.
- `controller.py` - `TytoController`, the provider-agnostic glue that turns a
  score stream into the three adaptations and answers the `check_audio_quality`
  tool.
- `openai_realtime.py` - `OpenAIRealtimeProvider`, the OpenAI Realtime WebSocket
  backend, kept as the second implementation of the seam. Audio playback is
  delegated to a sink so the same provider drives a local speaker or a browser.
- `audio.py` - `SounddeviceSink`, local speaker playback for the terminal agent.

## Run it

You need an ai-coustics SDK license key from
<https://developers.ai-coustics.com>, plus an Inkling key and a Deepgram key for
the agent. Both models (`tyto-1.1-l-16khz` and `vad-2.1-xxs-16khz`) are
downloaded from the ai-coustics CDN on first run into `./models`. Put your keys
in a `.env`; everything loads it automatically.

```bash
uv venv
cp .env.example .env    # AIC_SDK_LICENSE, INKLING_API_KEY, DEEPGRAM_API_KEY

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

- `AIC_SDK_LICENSE` runs the Tyto analyzer and the VAD on the backend. Both are
  local; no audio leaves the server for scoring or for turn-taking.
- `INKLING_API_KEY` and `DEEPGRAM_API_KEY` are used from the backend only. The
  browser exchanges mic and agent audio with your server and nothing else, so no
  key (or ephemeral secret) is ever sent to the browser.

All entry points auto-load a `.env` from the project root (see
[.env.example](.env.example)); exported environment variables override it.

## Which of the three Tyto layers are supported

All three, server-side, with the same tuned thresholds as the browser:

1. **Aware** - a one-sentence room note is swapped into the agent's system
   message whenever the dominant cause changes. Fully supported.
2. **Tuned** - turn-taking switches between an eager and a patient VAD profile
   when the room is noisy. Because turn-taking is local here, the profiles are
   ai-coustics VAD parameters rather than OpenAI `turn_detection` dicts: patient
   raises the speech-probability threshold (0.50 to 0.70) and lengthens the
   end-of-turn silence (1.10 s to 1.50 s), so background sound stops ending the
   user's sentences for them. Same layer, same two names, same trigger.
3. **Reactive** - when the smoothed risk crosses the threshold and one cause the
   user can act on dominates, the agent interrupts itself with a single spoken
   nudge, then resumes. The interrupt is a Deepgram `Clear` plus a playback
   flush. The nudge text is a fixed string from `decision.py`, so it goes
   straight to the voice with no model round trip, which makes Reactive the
   fastest part of the demo instead of the slowest.
   `codec_degradation` is deliberately excluded here: it is a transport problem,
   so it only feeds the Aware note (confirm names and numbers) instead of asking
   the user to fix their room.

The current Tyto reading is attached to every turn as a private system line, so
the user can ask "how do I sound?" and be answered from live data in one round
trip. It carries its own age, and a reading older than `READING_MAX_AGE_SECONDS`
is withheld rather than passed off as current: see the note on staleness below. The agent is told to keep it to itself unless asked, and to describe it in
words rather than read the numbers out. `CHECK_AUDIO_QUALITY_TOOL` and the
controller's handler for it are still there for backends that prefer a tool call
(the OpenAI provider does), but the cascade does not use them.

### Latency

Measured against the live APIs, for a roughly 5 second utterance:

| Stage | Median |
| --- | --- |
| VAD end-of-turn silence (eager) | 1.10 s |
| Inkling-Small, audio in to text out | 1.1 s |
| Deepgram first audio, warm socket | 0.24 s |
| **Total, from when you stop talking** | **~2.4 s** |

The endpointing silence is the tunable half of that and the deliberate one: it
carries half a second of grace over the longest mid-sentence pause measured, so
thinking out loud does not end your turn. Every 0.1 s taken off `end_silence` in
`decision.py` is 0.1 s less dead air and 0.1 s more chance of being cut off.

Asking "how do I sound?" costs nothing extra: the live Tyto reading is attached
to every turn, so the answer comes back in the same single round trip. Routing
that through a tool call instead measured 2.2 to 2.8 s at the model, against
1.0 to 1.3 s inline.

Three things this depends on, all of them load-bearing:

- `reasoning_effort="none"`. With thinking on, a reply takes about 2.8 s instead
  of 1.2 s, and a reply truncated by `max_tokens` puts the raw chain of thought
  into `content`, where it gets spoken out loud.
- A persistent Deepgram speak socket. A fresh REST request is 0.60 s to first
  audio against 0.24 s on a warm socket.
- Bounded audio history. `AUDIO_HISTORY_TURNS` is 1, for correctness rather than
  speed; see the note below on why a retained utterance is a liability.

Streaming the model output would not help: the whole reply arrives as a single
chunk, so time-to-first-token equals time-to-completion.

### How fast Tyto reacts

The 5 s analysis window is fixed by the model and cannot be shortened. What can
be changed is how fast it slides, and that is what the demo actually feels:
`HOP_SECONDS` is **0.5 s** here, so a fresh reading covering the last five
seconds arrives twice a second. `analyze_buffered()` measures 116 ms, so that
costs about 23% of one core. The browser reference uses 1 s; this is a
deliberate divergence, paid for by `NUDGE_MIN_PERSIST` (below).

Two things had to be fixed before that mattered at all.

**Tyto is starved by short turns, and that is the honest trade.** The analyzer is
reset every time the agent speaks, so that stale audio can never skew a score,
and then needs a full fresh window before it will emit anything. Modelled on a
typical exchange (user talks 3 s, agent thinks 1.3 s and speaks 6 s), pausing
produces **zero readings in 62 seconds**.

`pause_scoring_while_speaking=False` removes that entirely and keeps Tyto warm,
but it is only safe where the microphone genuinely cannot hear the agent. On
speakers, whatever echo survives is measured as the user's room, which inflates
interfering speech and produces nudges about voices that are our own. Both demos
therefore pause by default. Talk for five continuous seconds and Tyto has a
reading; with headphones, turn the flag off and it has one twice a second.

**Faster hops must not mean twitchier nudges.** Halving the hop halves the time
to cross the nudge gate, which would silently make the agent twice as quick to
interrupt people. `NUDGE_MIN_PERSIST` is 2, so a cause must dominate two
consecutive windows: the same one second of sustained evidence as before.

Short turns still leave the reading itself thin on the terminal path, where
pausing is required:

That is fine for the Aware, Tuned and Reactive layers, which only ever act on a
score as it arrives. It is a trap for the inline reading, which is read on demand
and would otherwise hand over whatever was measured last. That is how the agent
ends up insisting the room is still noisy after the noise has stopped, quoting
the same cause it named minutes earlier.

So the reading travels with its age, and the age decides one thing only: whether
the agent may speak about the room at all. Past `READING_MAX_AGE_SECONDS` the
reading is withheld. None of that machinery is ever surfaced to the user, who
should never hear about measuring, windows or seconds; the agent simply says it
is not sure. Measured against the live model, with a history in which the agent
had already committed to "noisy":

| reading passed in | what the agent says |
| --- | --- |
| stale degraded, presented as current | "It's loud and messy, with lots of overlapping voices" |
| stale degraded, withheld | "I'm not sure right now." |
| fresh clean | "It sounds very clean and clear now with just a little room echo" |

Talk for five seconds or more in one turn and you get a real reading.

### Limitations and notes

- **Barge-in is off everywhere by default, including the browser.** It is
  implemented and it works, but it is not a small switch. Leaving the microphone
  open while the agent talks closes an acoustic loop unless echo cancellation is
  genuinely removing our own voice, and browser `getUserMedia` cancellation does
  not reliably cover Web Audio playback. What survives is enough to trip the
  VAD, and then the agent's own voice is heard as speech, barge-in cuts the
  reply off, the echo finishes as an "utterance", Inkling answers the agent's
  own words, and it goes round again. The symptom is the agent repeating itself.
  Set `allow_barge_in=True` only where the microphone cannot hear the speaker,
  which in practice means headphones.
  Independently of that switch, unmuting after the agent has spoken always
  discards whatever the VAD was holding, because it can only be echo.
- **Echo cancellation.** The web demo captures the mic in the browser with echo
  cancellation on, so speakers are fine. The terminal agent plays through a raw
  output device with no cancellation, so use headphones there. In both, the
  controller stops sending mic audio to the agent while the agent speaks.
- **Audio transport.** Capture is PCM16 mono at **16 kHz**, which is native for
  the VAD, Tyto and Inkling alike, so nothing resamples on the way in. The
  agent's voice comes back at **24 kHz**, so the browser keeps one AudioContext
  per direction.
- **The transcript is a caption, not an input.** Inkling hears the audio
  directly, so the Deepgram speech-to-text call exists only so the UI's "You"
  panel can show roughly what was said. It runs on its own thread, never gates a
  reply, and its text never reaches the prompt, so a wrong or late transcript can
  only make the panel wrong, never the conversation. `keyterm=Tyto` is needed
  there, or the demo's own name comes back as "Taito". If the call fails the
  panel falls back to the length of the turn.
- **Only the current utterance is ever sent as audio.** This one is worth
  understanding before changing it. The model does not treat a retained past
  utterance as history, it *listens* to it, and then describes the room it hears
  there as though it were the room now. With two turns retained, a recording
  made while a TV was on produced "you sound like you're in a quiet room with
  people talking in the background" in answer to an unrelated question, long
  after the TV was off. The live reading does not win that argument, because the
  model can hear the evidence against it. So `AUDIO_HISTORY_TURNS` is 1, and a
  test pins it. The agent's own replies stay in history as text, which carries
  the thread of the conversation for almost no tokens, and dropping the extra
  audio turn also saved roughly 0.25 s of latency per turn.
- **Verification.** The decision layer, the controller state machine, the VAD
  segmentation, the history compaction and the speak-socket flush matching are
  covered by unit tests (`pytest`, 68 of them, no SDK, keys or network needed).
  Beyond that, the VAD was run against real speech with the real model, and the
  whole server was driven end to end by a synthetic websocket client that
  streams audio in and checks agent audio comes back. Only the microphone and
  speaker devices themselves are untested.

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
