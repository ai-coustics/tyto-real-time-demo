# AGENTS.md

Context for AI coding assistants (and humans) working in this repo. Read this
before making changes. It follows the [agents.md](https://agents.md) convention
and is also useful as Claude Code / Cursor project context.

## What this repo is

The Python reference of the Tyto acoustics-aware voice-agent demo. A live voice
agent talks to the user while the ai-coustics **Tyto** model scores the user's
microphone in real time, and the agent adapts on three layers (Aware, Tuned,
Reactive). The canonical browser reference is [index.html](index.html); this
branch reproduces the same behavior server-side in Python. Keep the two
comparable.

The agent itself is a **cascade**, not a speech-to-speech session: the
ai-coustics VAD does turn-taking, Inkling-Small hears the utterance and answers
in text, Deepgram speaks it. `OpenAIRealtimeProvider` is still in the tree as
the second implementation of the provider seam, and is not used by either demo.

Tyto returns, per fixed 5 second window: a **risk_score** (0..1, higher is
worse) and six dimensions (`noise`, `speaker_reverb`, `speaker_loudness`,
`interfering_speech`, `packet_loss`, `codec_degradation`).

The current model is **Tyto 1.1** (`tyto-1.1-l-16khz`, model version 7), which
needs aic-sdk 3.x on the Python side and `@ai-coustics/aic-sdk-wasm` 0.23.x in
the browser. Coming from Tyto 1.0: the old `media_speech` dimension is folded
into `interfering_speech` (competing speech, live or from a device),
`codec_degradation` is new, and the risk score is recalibrated, so the docs'
bands moved to <0.30 good / 0.30-0.50 warn / >0.50 bad.

## Architecture and data flow

```
 mic ──> LiveTytoScorer.feed() ──(aic-sdk Collector)──┐
                                                       │ every ~1s
                                          Analyzer.analyze_buffered()
                                                       │  (smoothed, EMA 0.3)
                                                       v
 mic ──> provider.send_audio() ──> agent      TytoController.on_scores()
            (CascadeProvider)                          │
                ^   │ events                           ├─ Layer 1 Aware:    set_instructions(BASE + room note)
                │   v                                  ├─ Layer 2 Tuned:    set_turn_detection(eager | patient)
            VoiceProvider <───── commands ────────────-┴─ Layer 3 Reactive: interrupt() + nudge()

 inside CascadeProvider:
 send_audio ──> LiveVad.feed() ──(utterance)──> "cascade-turn" thread
                                                          ├──> Deepgram STT   (UI caption, off the critical path)
                                                    └──> Inkling-Small ──> DeepgramTTS ──> audio_out
```

- **scorer.py** owns the SDK analyzer and the audio buffering. It does not own
  the mic; callers push audio via `feed()`.
- **controller.py** is the brain. It is provider-agnostic and holds the
  mute/nudge state machine. It runs on two threads (scores arrive on the scorer
  thread, provider events on the transport thread), guarded by one re-entrant
  lock.
- **provider.py** is the seam. **cascade.py** (and **openai_realtime.py**) are
  the only files that know about a specific backend. Audio playback is delegated
  to callbacks (`audio_out` / `audio_done` / `audio_flush`), so the same provider
  drives a local speaker or a browser.
- **vad.py** owns turn-taking. It does not own the mic; the provider pushes
  blocks in and gets whole utterances back.
- **inkling.py** and **deepgram.py** are thin API clients with no Tyto knowledge.
  Transport is `urllib` plus `websockets`, so they add no dependency.
- **audio.py** is `SounddeviceSink`, the local-speaker player for the terminal
  agent. It also owns the "is the agent audible" signal (`on_agent_audio`).
- **decision.py** is the pure scoring contract and decision functions, shared
  and identical across branches.

### Frontends

- **examples/score_mic.py** - terminal mic scorer (no agent).
- **examples/voice_agent.py** - terminal agent; uses `SounddeviceSink`.
- **examples/web/** - the browser UI. `server.py` (aiohttp) is the whole brain
  per tab; the browser is a thin client. `index.html` is generated from the root
  reference (CSS + markup reused, BYOK gate removed); `app.js` is the transport
  (mic capture, agent playback, render). Mic audio goes browser -> backend, agent
  audio comes back the same way; keys stay in the server env and the browser
  never talks to Inkling or Deepgram. The player (browser) owns
  `on_agent_audio`, reported back over the socket.

### Who owns "agent audible" (on_agent_audio)

The component that plays the audio reports it: `SounddeviceSink` in the terminal,
the browser in the web demo. The provider never calls `on_agent_audio`; it only
reports generation lifecycle via `on_agent_speaking`. Keep this split when adding
a backend.

## The provider seam: adding a new voice backend

This is the main extension point. To add ElevenLabs, LiveKit, a cascaded
pipeline, etc., write one subclass of `VoiceProvider` (see
[src/tyto_voice/provider.py](src/tyto_voice/provider.py)) and nothing else changes.

1. Implement the commands: `connect`, `disconnect`, `set_instructions` (Aware),
   `set_turn_detection` (Tuned and the listen gate), `set_mic_enabled`,
   `interrupt`, `nudge` (Reactive), `request_response`, `send_tool_result`.
   In a cascaded backend, `set_turn_detection` configures your own VAD and
   `None` means stop listening; `interrupt` drops queued speech; `nudge` can
   speak the line directly, since the text is fixed and needs no model.
2. Drive the controller from the backend's events through a `Handlers` object:
   `on_ready`, `on_agent_speaking(active, nudge, cancelled)`,
   `on_user_transcript`, `on_agent_transcript`, `on_tool_call`. Call
   `controller.on_agent_audio(playing)` from whatever plays the audio, not the
   provider (see below).
3. Wire it like [examples/voice_agent.py](examples/voice_agent.py) (terminal) or
   [examples/web/server.py](examples/web/server.py) (browser).

If a backend cannot support a layer (for example it manages turn-taking itself),
do not fake it. Implement what you can and document the gap in the README.

## Sample rates

Capture is **16 kHz** end to end: the VAD model, Tyto, and Inkling all want
16 kHz mono, so nothing resamples between the microphone and the models. Agent
audio comes back from Deepgram at **24 kHz**. These are two separate rates on
purpose; the browser keeps one AudioContext per direction and the terminal demo
opens the input stream at 16 kHz and `SounddeviceSink` at 24 kHz. If you change
one, change it in `cascade.py`, `app.js` and `voice_agent.py` together.

## Invariants to preserve

These keep the demo correct and comparable across branches. Do not change them
casually.

- **Tuned constants are ground truth.** Window 5 s, hop ~2 s, EMA alpha 0.3
  (the value the Tyto docs recommend), the per-dimension thresholds, the nudge
  bands. They live in `decision.py` and match the browser byte for byte. If you
  change one, change it in every branch and say why.
- **Warm-up gate.** Never score until a full fresh 5 s window has been buffered
  since the last reset. On resume after the agent speaks, reset the analyzer and
  re-warm. Stale audio must never skew a reading.
- **Mute while the agent speaks.** The mic is muted (no frames sent to the
  agent) while the agent talks, and unmutes after.
  Pausing *scoring* is separate and conditional. It protects Tyto from scoring
  the agent's own voice on an open speaker, but every pause resets the analyzer
  and costs a fresh 5 s warm-up, which with short turns starves Tyto of readings
  completely (modelled: zero in 62 s). Where the capture path has echo
  cancellation, set `pause_scoring_while_speaking=False` and leave Tyto
  measuring. The web demo does; the terminal agent must not.
  In the cascade this starts when the turn is dispatched, not when audio begins,
  so the mic is also muted while the model is thinking. Barge-in does not weaken
  this: it lets the local VAD keep listening, but mic audio still never reaches
  the model mid-reply and scoring stays paused.
- **A turn must always end.** Every path out of a turn has to reach
  `_end_turn()`, including a model error, an empty reply, a dead speak socket,
  and an interrupt. Miss one and the mic stays muted and the demo goes silent
  for good. `tests/test_cascade.py` drives the real controller and provider
  through each of those paths; keep it that way.
- **Never block the aiohttp event loop.** `Session.start` and `Session.stop`
  download models and wait on threads, so `ws_handler` hands them to
  `asyncio.to_thread`. Anything else blocking belongs there too, or the writer
  task cannot deliver progress and other tabs are not served.
- **A nudge always needs a cause the user can act on.** A high risk_score alone
  never nudges; one dimension must dominate (`strongest_cause`), and it must be
  one with nudge text. `codec_degradation` deliberately has none: it is a
  transport problem, so it feeds the Aware note (confirm names and numbers) but
  is never spoken at the user.
- **`speaker_loudness` and `speaker_reverb` are informational only.** Never
  colored as a problem, never named as a cause, never the reason for a nudge.

## aic-sdk quick reference (verified against aic-sdk 3.1.0, core 0.23.0)

```python
import aic_sdk as aic
path = aic.Model.download("tyto-1.1-l-16khz", "./models")  # CDN, cached
model = aic.Model.from_file(path)
# streaming (live):
collector, analyzer = aic.analyzer_pair(model, license_key)
config = aic.ProcessorConfig.optimal(model, sample_rate=24000)
collector.initialize(config)
collector.buffer(np.zeros(config.block_size, dtype=np.float32))  # 1D mono, exact block_size
result = analyzer.analyze_buffered()   # rolling window; silence-padded if short
analyzer.reset()                       # clears analyzer AND collector
# result fields: risk_score, noise, speaker_reverb, speaker_loudness,
#                interfering_speech, packet_loss, codec_degradation
```

What changed from the 2.x API this repo used before (all of it applies here):
`num_frames` -> `block_size`, `ProcessorConfig` has no `num_channels` and
buffers are 1D mono, `media_speech` -> `codec_degradation`,
`get_optimal_num_frames` -> `get_optimal_block_size`, and the analyzer gained
`terminate_session()`. Model ids are dotted (`tyto-1.1-l-16khz`) even though the
CDN path is dashed.

This demo is real-time only. (`aic.FileAnalyzer(model, key).analyze(...)` exists
for offline batch scoring, but it is intentionally not part of this demo.)

`Model.download`, `ProcessorConfig.optimal(..., sample_rate=24000)` and the
`AnalysisResult` field names are confirmed against the installed package; only
the licensed analysis steps need a real key.

## The cascade's external APIs

All three were verified against the live services. These are the parts that bite.

### Inkling-Small (`inkling.py`)

`POST https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1/chat/completions`,
`Authorization: Bearer $INKLING_API_KEY`, OpenAI-compatible.

| Thing | Value, and why it matters |
| --- | --- |
| model id | exactly `thinkingmachines/Inkling-Small`. Bare `Inkling-Small` gives "Sampling is not supported". Override with `INKLING_MODEL`; the endpoint serves many models but Inkling is the only one that works for this demo, and `inkling.py` carries the benchmark that says why. |
| system messages | exactly ONE, always. Several models on this endpoint reject a second with "System message must be at the beginning", so per-turn context is appended to the first rather than sent separately. |
| audio input | `{"type": "input_audio", "input_audio": {"data": <base64 wav>, "format": "wav"}}`, 16 kHz mono |
| `reasoning_effort` | must be `"none"`. Anything else roughly doubles latency, and a reply truncated by `max_tokens` puts raw reasoning into `content`, where it gets spoken. |
| streaming | pointless: the whole reply arrives as one chunk |
| tools | standard `tools` / `tool_choice`, and `{"role": "tool", "tool_call_id": ...}` results |
| User-Agent | required. The endpoint is behind Cloudflare, which 403s urllib's default agent with error code 1010. |
| cost | audio is about 27 prompt tokens per second, and each retained audio turn adds about 0.25 s |
| history | `AUDIO_HISTORY_TURNS` must stay 1. The model listens to a retained past utterance and reports the room it hears there as the room now, which no amount of fresher context overrides. |

### Deepgram (`deepgram.py`)

`wss://api.deepgram.com/v1/speak?model=aura-2-orion-en&encoding=linear16&sample_rate=24000`,
`Authorization: Token $DEEPGRAM_API_KEY`.

| Concept | Outgoing / incoming |
| --- | --- |
| speak a line | `{"type": "Speak", "text": ...}` then `{"type": "Flush"}` |
| interrupt | `{"type": "Clear"}` |
| agent audio | binary frames, raw linear16, no WAV header |
| line finished | `{"type": "Flushed", "sequence_id": N}` |

- `/v2/speak` is rejected with HTTP 400. aura-2 is served from `/v1/speak`.
- Keep the socket open. Time-to-first-audio is 0.24 s warm against 0.60 s for a
  fresh REST request.
- Match `Flushed` by `sequence_id`. A line abandoned by `Clear` can still have a
  `Flushed` in flight, and without the check it ends the line that replaced it.
- **Never trust Deepgram's `container=wav`.** It writes a streaming placeholder
  length (`0x7fff...`) that reads back as about nineteen hours. Build WAV headers
  yourself; `inkling.encode_wav` does.
- STT (`/v1/listen`, `nova-3`) is a **caption only**. Inkling hears the user's
  audio directly, so this never gates a reply and its text must never reach the
  prompt. Keeping it out of history is deliberate: it is also what makes a late
  transcript harmless, since there is no question of which turn it belongs to.
  Pass `keyterm=Tyto`, or the demo's own name comes back as "Taito".

### The live reading (`decision.live_reading`)

The current Tyto scores are attached to every turn as a second system message,
so "how do I sound?" is answered from live data in one round trip. Measured, the
same answer via a `check_audio_quality` tool call cost 2.2 to 2.8 s against 1.0
to 1.3 s inline, because a tool call means a second request to Inkling.

It is per-turn context and is deliberately NOT stored in history: it describes
the room right now, and replaying a stale copy on later turns is worse than not
having it. `InklingClient.respond(context=...)` is the seam for that.

**A reading is usually not recent, and that is the trap.** Tyto needs a fresh 5 s
window and the analyzer is reset on every agent turn (the warm-up gate), so an
exchange of short utterances can produce one score a minute or none at all. The
other three layers are immune because they only act on a score as it arrives.
This one is read on demand, so it must carry its age: past
`READING_MAX_AGE_SECONDS` it is replaced by `NO_READING`. Skip that and the agent
confidently describes a room that stopped existing minutes ago, which is exactly
the bug it was written to fix. The age is an internal gate and nothing else: no
part of this machinery may reach the user, who should never hear about
measuring, windows, readings or seconds. `test_the_mechanics_never_reach_the_prompt_text`
pins that. The reading text also states that it overrides
anything said earlier, because the agent's own past replies are in history and it
will otherwise repeat them.

### ai-coustics VAD (`vad.py`)

Model `vad-2.1-xxs-16khz`, a separate `aic.Vad` instance, not `Processor`.

```python
vad = aic.Vad(model, license_key, aic.ProcessorConfig.optimal(model, sample_rate=16000))
ctx = vad.get_context()
ctx.set_parameter(aic.VadParameter.Sensitivity, 0.5)
ctx.set_parameter(aic.VadParameter.MinimumSpeechDuration, 0.06)  # range 0..1 s
ctx.set_parameter(aic.VadParameter.SpeechHoldDuration, 0.10)
vad.process(block)              # exactly block_size mono float32, returns None
ctx.is_speech_detected()        # the segmentation signal
ctx.reset()
```

**`is_speech_detected()` falling edge is NOT the end of the turn.** Measured on
real speech at 16 kHz, it drops out for 45 to 285 ms at ordinary pauses inside a
single sentence, and raising `SpeechHoldDuration` does not close those gaps
(it is a rolling majority over the last `hold * 2` seconds, so a longer window
can make a gap wider, not narrower). Ending on the falling edge split one 4.8 s
question into three utterances.

So `LiveVad` keeps `SpeechHoldDuration` short and runs its own hangover: a turn
ends only after `end_silence` seconds of continuous silence (1.10 eager, 1.50
patient, which is the worst gap plus half a second of grace), and that trailing
silence is trimmed off before the utterance is sent.
`end_silence` is ours, not an SDK parameter. Durations are rounded to the
model's window length, so reading a parameter back may not return what was
written.

Feed the VAD the original microphone audio, never enhanced output.

## Running and verifying

```bash
uv pip install -e ".[dev]"
uv run pytest -q                 # 68 tests: decision, controller, scorer, cascade
```

The unit tests need no SDK, key, network, or hardware. The end-to-end audio path
needs an ai-coustics key, an Inkling key, a Deepgram key, a mic, and headphones.

## Deploying

[deploy/modal_app.py](deploy/modal_app.py) is the only deployment artifact.
`examples/web/server.py` exposes `build_app(keys)` and `keys_from_env()` so the
routes and the session wiring have one definition shared by the local runner and
the deploy; keep it that way rather than restating routes in the Modal file.

Three things there are load bearing: the server binds `HOST` (0.0.0.0 on Modal,
loopback locally, so a laptop does not serve a microphone demo to its network);
both models are baked into the image and found through `AIC_MODELS_DIR`; and the
Modal app name is deliberately NOT `tyto-demo`, because deploying an app name
replaces that app's whole function set.

## Conventions

- Style: KISS and DRY, clean and minimal. Match the surrounding code.
- No em dashes in prose or comments. Use hyphens or rewrite.
- Secure context: the mic needs HTTPS or localhost in any browser-facing
  extension of this.
- Verify provider APIs against current docs; they change often. Do not trust
  training memory for method or event names.

## Related ai-coustics tooling

If your assistant has the ai-coustics MCP skills available, these help when
working with audio in this repo: `tyto` (score audio with the observability
model), `enhance` (run SDK enhancement), `transcribe` (STT across providers),
and `before-after` (comparison pages). They are separate tools from this demo
but share the same model family and license.
