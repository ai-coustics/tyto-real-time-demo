# AGENTS.md

Context for AI coding assistants (and humans) working in this repo. Read this
before making changes. It follows the [agents.md](https://agents.md) convention
and is also useful as Claude Code / Cursor project context.

## What this repo is

The Python reference of the Tyto acoustics-aware voice-agent demo. A live voice
agent talks to the user while the ai-coustics **Tyto** model scores the user's
microphone in real time, and the agent adapts on three layers (Aware, Tuned,
Reactive). The voice stack is a Pipecat cascade: Deepgram Flux to gpt-5-mini to
Deepgram Aura-2.

Tyto 1.1 returns, per fixed 5 second window: a **risk_score** (0..1, higher is
worse) and six dimensions (`noise`, `speaker_reverb`, `speaker_loudness`,
`interfering_speech`, `packet_loss`, `codec_degradation`).

Note for anyone porting from an older copy: Tyto 1.1 dropped `media_speech`,
merging it into `interfering_speech` (any competing speech, live or from a TV),
and added `codec_degradation`. It also needs aic-sdk 3.x and the model id
`tyto-1.1-l-16khz`; the old `tyto-l-16khz` is refused as an incompatible
version.

## Architecture and data flow

The Pipecat pipeline, per browser connection, is the whole demo:

```python
Pipeline([
    transport.input(),       # browser mic, PCM16 16 kHz over WebRTC
    TytoAudioTap,            # Tyto listens here, and this is the mic gate
    VoiceFocusProcessor,     # optional Quail enhancement, agent-only
    stt,                     # Deepgram Flux: transcription AND turn detection
    aggregators.user(),
    llm,                     # gpt-5-mini
    tts,                     # Deepgram Aura-2
    transport.output(),      # agent audio back to the browser
    aggregators.assistant(),
])
```

```
 mic ──> LiveTytoScorer.feed() ──(aic-sdk Collector)──┐
                                                       │ every 0.5s
                                          Analyzer.analyze_buffered()
                                                       │  (smoothed, EMA 0.3)
                                                       v
 mic ──> the cascade ──> agent                TytoController.on_scores()
             ^   │ frames                              │
             │   v                                     ├─ Layer 1 Aware:    set_instructions(BASE + room note)
        CascadeProvider <──── commands ────────────────┼─ Layer 2 Tuned:    set_turn_detection(eager | patient)
                                                       └─ Layer 3 Reactive: interrupt() + nudge()
```

Each layer is exactly one Pipecat frame:

| Layer | Frame | Effect |
| --- | --- | --- |
| Aware | `LLMMessagesTransformFrame` | rewrites the system message, keeps history |
| Tuned | `STTUpdateSettingsFrame` | retunes Flux end-of-turn thresholds mid-stream |
| Reactive | `InterruptionFrame` + `TTSSpeakFrame` | cuts the reply, speaks a fixed line |

- **scorer.py** owns the SDK analyzer and the audio buffering. It does not own
  the mic; callers push audio via `feed()`.
- **controller.py** is the brain. It is provider-agnostic and holds the
  mute/nudge state machine. It runs on two threads (scores arrive on the scorer
  thread, provider events on the transport thread), guarded by one re-entrant
  lock.
- **provider.py** is the seam. **cascade.py** is the only file that knows what
  Deepgram, OpenAI or Pipecat are. Everything above it is provider-agnostic.
- **voicefocus.py** is optional Quail enhancement on the agent's input. It is
  independent of the three layers and must never be in Tyto's path.
- **decision.py** is the pure scoring contract and decision functions, shared
  and identical across branches.

### Frontends

- **examples/score_mic.py** - terminal mic scorer (no agent, no voice keys).
- **examples/pipecat/** - the browser UI. `server.py` (FastAPI +
  `SmallWebRTCTransport`) is the whole brain per tab; the browser is a thin
  client that captures the mic, plays the agent, and renders. `app.js` carries
  the UI and the data-channel protocol. All three keys stay in the server env.

### The mic gate

There is exactly one gate, `TytoAudioTap.set_enabled`, sitting one hop after the
transport input. When it is shut, neither the scorer nor the transcriber sees a
sample, so the agent cannot be triggered by its own nudge. Do not add a second
gate at the transport or in the STT service.

### The UI has no layer cards

The three layers are not drawn as cards. Aware and Tuned show up in the event log
and in the agent's behaviour; Reactive announces itself in the banner and out
loud. If you add UI, do not re-add per-layer panels without asking: the current
layout is deliberate.

### Interim transcripts are snapshots, not deltas

Deepgram Flux re-sends the **whole** utterance on every update. The UI assigns
interim text (`userInterim = m.text`), it does not append. If you add a backend
that streams true deltas, accumulate them server-side before sending.

## The provider seam: adding a new voice backend

This is the main extension point. To add ElevenLabs, LiveKit, a cascaded
pipeline, etc., write one subclass of `VoiceProvider` (see
[src/tyto_voice/provider.py](src/tyto_voice/provider.py)) and nothing else changes.
[cascade.py](src/tyto_voice/cascade.py) is the worked example: it drives a
Pipecat cascade behind the seam, taps the mic into the scorer with a small
`FrameProcessor`, and turns pipeline frames into `Handlers` calls with a
`BaseObserver`. Where Pipecat owns a concern, the gap is documented, not faked.

1. Implement the commands: `connect`, `disconnect`, `set_instructions` (Aware),
   `set_turn_detection` (Tuned and the listen gate), `set_mic_enabled`,
   `interrupt`, `nudge` (Reactive), `request_response`, `send_tool_result`.
2. Drive the controller from the backend's events through a `Handlers` object:
   `on_ready`, `on_agent_speaking(active, nudge, cancelled)`,
   `on_user_transcript`, `on_agent_transcript`, `on_tool_call`. Call
   `controller.on_agent_audio(playing)` from whatever plays the audio, not the
   provider (see below).
3. Wire it like [examples/pipecat/server.py](examples/pipecat/server.py).

`TytoController` must stay synchronous and provider-agnostic: its tests drive a
plain `FakeProvider` with no event loop, and the nudge watchdog is a real
`threading.Timer`. Any asyncio hop belongs inside the provider.

If a backend cannot support a layer (for example it manages turn-taking itself),
do not fake it. Implement what you can and document the gap in the README.

## Invariants to preserve

These keep the demo correct and comparable across branches. Do not change them
casually.

- **Tuned constants are ground truth.** Window 5 s, hop 0.5 s, EMA alpha 0.3,
  the per-dimension thresholds, the risk bands (0.30 / 0.50). They live in
  `decision.py`. If you change one, change it in every branch and say why, and
  mirror it in `app.js`, which keeps its own copy for rendering.
- **decision.py imports nothing.** It is pure Python, shared across branches, and
  must never import pipecat. The cascade tests are where its intent is tied to
  real frames.
- **Warm-up gate.** Never score until a full fresh 5 s window has been buffered
  since the last reset. On resume after the agent speaks, reset the analyzer and
  re-warm. Stale audio must never skew a reading.
- **The scoring gate.** By default the mic is muted and scoring paused while the
  agent talks, so Tyto never scores the agent. This demo runs with
  `pause_scoring_while_speaking=False` because the browser cancels the echo, and
  that is deliberate: it is what lets Layer 3 interrupt a reply in progress. Do
  not turn it off anywhere the mic can hear the speaker.
- **Tyto never sees enhanced audio.** The Tyto tap is upstream of
  `VoiceFocusProcessor`, always. Reversing them makes the meters describe Quail's
  output instead of the room, the Reactive layer goes quiet, and the demo still
  looks like it works. Asserted in `test_cascade.py`.
- **A nudge must always give the mic back.** `NUDGE_MAX_SECONDS` arms a watchdog
  whenever the gate closes. A gate that never reopens looks exactly like a broken
  microphone, which is worse than the thing it guards.
- **A nudge always needs a cause.** A high risk_score alone never nudges; one
  dimension must dominate (`strongest_cause`), and it must be one the user can
  act on. `codec_degradation` and `packet_loss` are transport problems: they
  inform the agent (Aware) but `codec_degradation` never triggers a spoken nudge,
  because asking someone to change their room would be nonsense.
- **`speaker_loudness` and `speaker_reverb` are informational only.** Never
  colored as a problem, never named as a cause, never the reason for a nudge.

## aic-sdk quick reference (verified against aic-sdk 3.1.0, SDK 0.23.0)

```python
import aic_sdk as aic
path = aic.Model.download("tyto-1.1-l-16khz", "./models")  # CDN, cached
model = aic.Model.from_file(path)
# streaming (live):
collector, analyzer = aic.analyzer_pair(model, license_key)
config = aic.ProcessorConfig.optimal(model, sample_rate=16000)  # no num_channels
collector.initialize(config)
collector.buffer(np.zeros(config.block_size, dtype=np.float32))  # flat mono, exact
result = analyzer.analyze_buffered()   # rolling window; silence-padded if short
analyzer.reset()                       # clears analyzer AND collector
# result fields: risk_score, noise, speaker_reverb, speaker_loudness,
#                interfering_speech, packet_loss, codec_degradation
```

Three things changed with aic-sdk 3.x and will bite a naive upgrade: the model id
gained a version (`tyto-1.1-l-16khz`), `ProcessorConfig.optimal` dropped
`num_channels` and exposes `block_size` instead of `num_frames`, and `buffer()`
takes a flat mono array rather than a reshaped `(1, n)` one.

This demo is real-time only. (`aic.FileAnalyzer(model, key).analyze(...)` exists
for offline batch scoring, but it is intentionally not part of this demo.)

`Model.download` and `ProcessorConfig.optimal(..., sample_rate=24000)` are
confirmed to work; only the licensed analysis steps need a real key.

## Pipecat frame mapping (verified against pipecat-ai 1.8.1)

| Concept | Frame or call |
| --- | --- |
| set instructions (Aware) | `LLMMessagesTransformFrame(transform, run_llm=False)` |
| retune turn-taking (Tuned) | `STTUpdateSettingsFrame(delta=DeepgramFluxSTTService.Settings(...))` |
| nudge (Reactive) | `TTSSpeakFrame(text, append_to_context=True)` |
| interrupt | `InterruptionFrame()` |
| opening greeting | `TTSSpeakFrame` (no LLM round trip) |
| agent generating | `LLMFullResponseStartFrame` / `EndFrame` |
| agent audible | `BotStartedSpeakingFrame` / `BotStoppedSpeakingFrame` |
| user text | `TranscriptionFrame` / `InterimTranscriptionFrame` |
| agent text | `LLMTextFrame` |
| tool call | `llm.register_function(name, handler)` |

Notes that cost real debugging time:

- `PipelineTask` is deprecated since Pipecat 1.3.0. Use `PipelineWorker` from
  `pipecat.pipeline.worker`.
- Deepgram Flux does its own turn detection, so `TransportParams` needs no
  `vad_analyzer`. That is **not sufficient on its own**: `LLMContextAggregatorPair`
  defaults to `UserTurnStrategies()`, whose stop strategy constructs a
  `LocalSmartTurnAnalyzerV3` and loads an ONNX session at init. Pass
  `user_params=LLMUserAggregatorParams(user_turn_strategies=ExternalUserTurnStrategies(...))`
  or two components end up deciding when the user stopped talking.
- Pass `enable_rtvi=False` to `PipelineWorker`. On the default an `RTVIProcessor`
  is inserted at the head and consumes the browser data-channel messages this
  demo's UI protocol runs on.
- `STTUpdateSettingsFrame(settings={...})` is deprecated and warns. Use
  `delta=DeepgramFluxSTTService.Settings(...)`.
- "Stop speculating" cannot be expressed as `eager_eot_threshold=None`. An
  explicit null is rejected by Deepgram and fails silently; an omitted key reads
  as NOT_GIVEN and leaves the previous value live. Pin it to the profile's own
  `eot_threshold` instead. See `_flux_settings`.
- A nudge is a `TTSSpeakFrame`, so it produces none of the `LLMFullResponse`
  frames a normal reply does. The observer reports it as agent speech from the
  `Bot*SpeakingFrame` pair instead.

## Running and verifying

```bash
uv pip install -e ".[dev]"
uv run pytest -q                 # decision, controller, scorer, cascade, voice focus, server startup
```

Most tests need no SDK, key, or hardware. `tests/test_cascade.py` needs pipecat
installed and skips without it. The end-to-end audio path needs all three keys, a
mic, and a browser.

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
