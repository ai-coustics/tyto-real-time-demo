# AGENTS.md

Context for AI coding assistants (and humans) working in this repo. Read this
before making changes. It follows the [agents.md](https://agents.md) convention
and is also useful as Claude Code / Cursor project context.

## What this repo is

The PhoneLLM branch of the Tyto acoustics-aware voice-agent demo. A live voice
agent talks to the user while the ai-coustics **Tyto** model scores the user's
microphone in real time, and the agent adapts on three layers (Aware, Tuned,
Reactive). The canonical browser reference is [index.html](index.html); this
branch reproduces the same behavior server-side in Python. Keep the two
comparable, and say so in a comment wherever they deliberately differ.

The agent is a cascade: **Deepgram Flux** (transcription and turn-taking in one
socket) -> **Pipecat PhoneLLM Alpha 1 on Modal** (`pipecat-ai/phonellm-alpha-1`,
an OpenAI-compatible Auto Endpoint) -> **Deepgram Aura-2** (voice). The point of
a cascade here is that every stage is cancellable, which is what the Reactive
layer exploits.

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
                                                       │ every 0.5s
                                          Analyzer.analyze_buffered()
                                                       │  (smoothed, EMA 0.3)
                                                       v
 mic ──> provider.send_audio() ──> agent      TytoController.on_scores()
            (Flux -> PhoneLLM -> Aura-2)               │
                ^   │ events                           ├─ Layer 1 Aware:    set_instructions(BASE + room note)
                │   v                                  ├─ Layer 2 Tuned:    set_turn_detection(eager | patient)
            VoiceProvider <───── commands ────────────-┴─ Layer 3 Reactive: interrupt() + nudge()
```

- **scorer.py** owns the SDK analyzer and the audio buffering. It does not own
  the mic; callers push audio via `feed()`.
- **controller.py** is the brain. It is provider-agnostic and holds the
  mute/nudge state machine. It runs on two threads (scores arrive on the scorer
  thread, provider events on the transport thread), guarded by one re-entrant
  lock.
- **provider.py** is the seam. **cascade.py** is the live backend and the only
  place the three services are wired together; **flux.py**, **phonellm.py** and
  **deepgram.py** each know about exactly one of them. **openai_realtime.py** is
  the second implementation of the same seam, kept for comparison. Audio playback
  is delegated to callbacks (`audio_out` / `audio_done` / `audio_flush`), so the
  same provider drives a local speaker or a browser.
- **cascade.py** also owns *speculation*: Flux's `EagerEndOfTurn` fires a
  PhoneLLM request before the user has finished, and `EndOfTurn` reuses the reply
  only when the transcript matches exactly. Deepgram guarantees that match when
  no `TurnResumed` intervened; without the check the agent answers half a
  sentence.
- **audio.py** is `SounddeviceSink`, the local-speaker player for the terminal
  agent. It also owns the "is the agent audible" signal (`on_agent_audio`).
- **decision.py** is the pure scoring contract and decision functions. The
  contract (`Scores`, thresholds, bands, the decision rules) is shared with every
  branch. Two things here are branch-specific and marked as such: `VAD_PROFILES`
  holds Flux thresholds rather than OpenAI Realtime dicts, and the reactivity
  tuning (hop, nudge threshold, persist, cooldown) is deliberately hotter than
  the browser's. See the table in the README.

### Frontends

- **examples/score_mic.py** - terminal mic scorer (no agent).
- **examples/voice_agent.py** - terminal agent; uses `SounddeviceSink`.
- **examples/web/** - the browser UI. `server.py` (aiohttp) is the whole brain
  per tab; the browser is a thin client. `index.html` is generated from the root
  reference (CSS + markup reused, BYOK gate removed); `app.js` is the transport
  (mic capture, agent playback, render). Mic audio is relayed browser -> backend
  at 16 kHz and agent audio back at 24 kHz, one AudioContext per direction so
  neither side resamples; keys stay in the server env. The player (browser) owns
  `on_agent_audio`, reported back over the socket. This is the only frontend with
  echo cancellation, so it is the only one with barge-in and continuous scoring.

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
2. Drive the controller from the backend's events through a `Handlers` object:
   `on_ready`, `on_agent_speaking(active, nudge, cancelled)`,
   `on_user_transcript`, `on_agent_transcript`, `on_tool_call`. Call
   `controller.on_agent_audio(playing)` from whatever plays the audio, not the
   provider (see below).
3. Wire it like [examples/voice_agent.py](examples/voice_agent.py) (terminal) or
   [examples/web/server.py](examples/web/server.py) (browser).

If a backend cannot support a layer (for example it manages turn-taking itself),
do not fake it. Implement what you can and document the gap in the README.

## Invariants to preserve

These keep the demo correct and comparable across branches. Do not change them
casually.

- **Tuned constants are ground truth.** Window 5 s, EMA alpha 0.3 (the value the
  Tyto docs recommend), the per-dimension thresholds, the bands. They live in
  `decision.py`. Where this branch diverges from the browser it is written down,
  in the constant's own comment and in the README table: hop 0.5 s not 1 s, nudge
  threshold 0.31 not 0.40, plus a 10 s cooldown that only this branch needs. Do
  not add an undocumented divergence.
- **Warm-up gate.** Never score until a full fresh 5 s window has been buffered
  since the last reset. On resume after a pause, reset the analyzer and re-warm.
  Stale audio must never skew a reading.
- **The mic is muted while the agent speaks, unless barge-in is on.** With
  barge-in, audio keeps flowing so Flux can report the user talking over the
  agent; only turn that on where the microphone cannot hear the speaker.
- **Scoring pauses while the agent speaks, unless the capture cancels echo.**
  `pause_scoring_while_speaking=False` is what lets a nudge interrupt a reply in
  progress, and it is only safe where the agent's voice is not in the mic signal.
  It is on in the web demo and off in the terminal one, and both say why.
- **A speculated reply is only ever reused on an exact transcript match.**
  Anything else is discarded and asked again.
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

## Wire mapping (verified against current docs, 2026-08)

**Deepgram Flux**, `wss://api.deepgram.com/v2/listen`, header
`Authorization: Token <key>`, raw `linear16` binary in.

| Concept | Message |
| --- | --- |
| connect | query: `model=flux-general-en`, `encoding`, `sample_rate`, `eot_threshold`, `eager_eot_threshold`, `eot_timeout_ms`, `keyterm` |
| turn events | `TurnInfo` with `event` in StartOfTurn / Update / EagerEndOfTurn / TurnResumed / EndOfTurn, plus `turn_index` and `transcript` |
| retune (Layer 2) | `{"type": "Configure", "thresholds": {...}}` -> `ConfigureSuccess` |
| stop | `{"type": "CloseStream"}` |

Ranges Flux enforces: `eot_threshold` 0.5-1.0, `eager_eot_threshold` 0.3-0.9 and
`<= eot_threshold`, `eot_timeout_ms` 500-60000.

**PhoneLLM on Modal**, `<MODAL_ENDPOINT_URL>/v1/chat/completions`, header
`Authorization: Bearer <token-id>.<token-secret>`. Ordinary OpenAI-compatible
chat completions. Two body fields come from the model card and are required:
`temperature: 0` and `chat_template_kwargs: {"enable_thinking": false}`. Model id
must be exactly `pipecat-ai/phonellm-alpha-1`. The endpoint scales to zero, so
503 on the first request after a quiet period is normal; `/v1/models` is the
readiness probe.

**Deepgram Aura-2**, `wss://api.deepgram.com/v1/speak` (`/v2/speak` is a 400).
`Speak` + `Flush` to say a line, `Clear` to abandon one, `Flushed` (matched by
`sequence_id`) to know a line finished.

## Running and verifying

```bash
uv pip install -e ".[dev]"
uv run pytest -q                 # 48 tests: decision + controller + scorer + cascade
```

The unit tests need no SDK, key, network, or hardware. The end-to-end audio path
needs an ai-coustics key, a live Modal endpoint, a Deepgram key, a mic, and
either a browser or headphones.

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
