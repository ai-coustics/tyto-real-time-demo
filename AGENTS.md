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

The voice is **GPT-Live 1** (`gpt-live-1`, full duplex) by default, Realtime
(`gpt-realtime-2.1`) with `VOICE_BACKEND=realtime`. The Reactive layer has a
judge, **Jev** (TypeSafe AI System One) via Vercel AI Gateway: Tyto's rule
decides *that* there is a fixable problem, Jev decides *how* to act on it.

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
         (GPT-Live 1 | Realtime)                       │
                ^   │ events                           ├─ Layer 1 Aware:    set_instructions(BASE + room note)
                │   v                                  ├─ Layer 2 Tuned:    set_turn_detection(eager | patient)
            VoiceProvider <───── commands ────────────-┴─ Layer 3 Reactive: EnvMonitor gate ─> JevJudge.ask()
                                                                              (Tyto: that)      (Jev: how, ~300 ms)
                                                                                                  │ ask_now / ask_after_sentence /
                                                                                                  │ adapt_quietly / stay_silent
                                                                                                  v
                                                                                       interrupt() + nudge(), deferred, or held
```

- **scorer.py** owns the SDK analyzer and the audio buffering. It does not own
  the mic; callers push audio via `feed()`.
- **controller.py** is the brain. It is provider-agnostic and holds the
  mute/nudge state machine. It runs on two threads (scores arrive on the scorer
  thread, provider events on the transport thread), guarded by one re-entrant
  lock.
- **provider.py** is the seam. **openai_live.py** (GPT-Live 1, default) and
  **openai_realtime.py** are the only files that know about a specific backend.
  Audio playback is delegated to callbacks (`audio_out` / `audio_done` /
  `audio_flush`), so the same provider drives a local speaker or a browser.
- **jev.py** is the judge: `Situation` in (bucketed words), `Decision` out, one
  request in flight, worker-thread callback, fallback to the rule on any error.
  `build_state` / `build_questions` / `apply_policy` are pure and unit tested.
- **backends.py** builds the provider and the judge from env vars for both
  entry points.
- **audio.py** is `SounddeviceSink`, the local-speaker player for the terminal
  agent. It also owns the "is the agent audible" signal (`on_agent_audio`).
- **decision.py** is the pure scoring contract and decision functions, shared
  and identical across branches.

### Frontends

- **examples/score_mic.py** - terminal mic scorer (no agent).
- **examples/voice_agent.py** - terminal agent; uses `SounddeviceSink`.
- **examples/web/** - the browser UI. `server.py` (aiohttp) is the whole brain
  per tab; the browser is a thin client. `index.html` follows the ai-coustics
  design system (token CSS copied verbatim into `ds/`, served at `/ds`; the
  licensed Milling webfont is gitignored under `assets/fonts/` and the page
  falls back to Hanken Grotesk) and mirrors the Audio Insight post-call demo:
  top bar with the docs links, hero, four stat tiles (risk score, agent, decision,
  voice agent), six-dimension row, conversation, activity feed. `app.js` is the transport (mic capture,
  agent playback) plus the render. Audio is relayed browser <-> backend <->
  OpenAI; keys stay in the server env. The player (browser) owns
  `on_agent_audio`, reported back over the socket. The three layers have no
  boxes of their own: they surface as the Agent tile, Tyto lines in the
  conversation, and activity entries.

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
`openai_live.py` is the worked example of a hard case: no cancel event, no VAD
knobs, immutable instructions, a continuous audio track. Read its docstring
before adding another full-duplex backend.

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
- **Mute and pause while the agent speaks.** The mic is muted (no frames sent to
  the agent) and scoring is paused while the agent talks; both resume after.
- **A nudge always needs a cause the user can act on.** A high risk_score alone
  never nudges; one dimension must dominate (`strongest_cause`), and it must be
  one with nudge text. `codec_degradation` deliberately has none: it is a
  transport problem, so it feeds the Aware note (confirm names and numbers) but
  is never spoken at the user.
- **`speaker_loudness` and `speaker_reverb` are informational only.** Never
  colored as a problem, never named as a cause, never the reason for a nudge.
- **Jev judges, it never gates.** The `EnvMonitor` gate (threshold + dominant
  actionable cause) stays in Python and is what makes a nudge possible at all.
  Jev only picks between `ask_now`, `ask_after_sentence`, `adapt_quietly` and
  `stay_silent`, and the combination policy (`jev.apply_policy`) is code. Any
  Jev failure or low confidence falls back to the rule (ask now), never to
  silence.
- **Jev sees words, not numbers.** `build_state` sends buckets ("severe",
  "about 10 seconds", "never"), the two transcript tails and who is speaking.
  No scores, thresholds or audio ever go to the gateway. The unit test
  `test_state_is_named_buckets_without_raw_numbers` enforces this.
- **GPT-Live: the listener's stop is instant, the model's is not.** On an
  interrupt the provider flushes playback and holds (drops) the model's output
  until a ≥0.45 s pause followed by speech, a nudge keyword in the transcript,
  or a 6 s cap. Do not "fix" this by trusting the model to stop: measured live,
  it finishes its sentence (3 to 4 s) before it speaks a commentary.

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

## GPT-Live 1 event mapping (WebSocket, server-side, default)

Model `gpt-live-1` at `wss://api.openai.com/v1/live/sessions`, `Authorization:
Bearer`. Verified live on 2026-09-22 (two sessions, 26 s, $0.02):

| Concept | Outgoing / incoming |
| --- | --- |
| configure session | `session.start` with `session: {model, instructions, audio: {format: {type: audio/pcm, rate: 24000}, output: {voice}}, delegation: {type: client}}`. Strict: unknown fields are rejected; `model`, `instructions`, `audio` are immutable after start |
| ready | `session.started` |
| send mic | `session.input_audio.append` (base64 PCM16, 24 kHz). Muted = send zeros, keep the track continuous |
| agent audio | `session.output_audio.delta` (`delta` only, no timing fields on OpenAI's endpoint). Continuous, ~100 ms chunks, silence included: gate by RMS (speech > 0.01, silence < 0.001) |
| agent text | `session.output_transcript.delta` (`delta`, `start_ms`, `end_ms`), ~1 s behind the audio, no final marker |
| user text | `session.input_transcript.delta` |
| Aware / Tuned | `session.instructions.append` `{content, delegation_id: null}` (≤500 tokens). Does not make the model speak |
| open the call | `session.thinking.append` (quiet context). Made it speak in 1.2 s. `instructions.append` did not within 5 s |
| nudge | `session.commentary.append` (content to say aloud). Spoken at once when idle; when talking the model finishes its sentence first (3 to 4 s), hence the hold in `interrupt()` |
| interrupt | none in the API. Flush playback + hold output (see invariants) |
| tool call | `session.delegation.created` (`delegation.id`, `target: client`) -> answer with `session.commentary.append` carrying that `delegation_id` |
| usage / end | `session.usage.updated` (~1/min), `session.close` -> `session.closed` (final `usage.seconds`), `error` |

Prompting guides: <https://developers.openai.com/api/docs/guides/live>,
`.../live-prompting`, `.../live-delegation`; full event reference (Foundry
mirror): <https://learn.microsoft.com/azure/foundry/openai/gpt-live-reference>.

## Jev quick reference (verified against the gateway 2026-09-22)

```
POST https://ai-gateway.vercel.sh/typesafe/v1/systemone
Authorization: Bearer $AI_GATEWAY_API_KEY
{"model": "typesafe-ai/jev", "state": <str|object|array>,
 "questions": {"id": {"type": "choice", "instructions": "...", "criteria": {"opt": "meaning", ...}},
               "id2": {"type": "noul", "instructions": "..."},
               "id3": {"type": "score", "instructions": "...", "criteria": ["level 0", ..., "level n"]}}}
-> {"model": ..., "answers": {"id": {"type": "choice", "choice": "opt", "probabilities": {...}, "confidence": 0..1},
                              "id2": {"type": "noul", "noul": 0..1}, ...}, "usage": {...}, "provider_metadata": {...}}
GET  https://ai-gateway.vercel.sh/typesafe/v1/models      (also the connection warm-up)
```

Latency from Berlin through the gateway: 800 ms cold, ~300 ms warm (p50 298 ms
over 4 calls, one keep-alive client). Confidence for n options is
`(n * p_max - 1) / (n - 1)`. Known jagged edges of jev-1.13 that shaped the
design: unreliable with numbers, thresholds and durations, weaker with long
irrelevant state and double negatives. Docs: <https://docs.typesafe.ai>,
<https://docs.typesafe.ai/model-jaggedness/jev-1.13>.

## OpenAI Realtime event mapping (WebSocket, `VOICE_BACKEND=realtime`)

Model: `gpt-realtime-2.1` (the browser reference uses the same). Input
transcription stays on `gpt-4o-mini-transcribe`; `gpt-live-transcribe` is the
newer option if you want it.

| Concept | Outgoing / incoming |
| --- | --- |
| configure session | `session.update` (instructions, audio.input.turn_detection, transcription, audio.output.voice, tools) |
| send mic | `input_audio_buffer.append` (base64 PCM16, 24 kHz) |
| agent audio | `response.output_audio.delta` (also legacy `response.audio.delta`) |
| agent text | `response.output_audio_transcript.delta` / `.done` |
| user text | `conversation.item.input_audio_transcription.delta` / `.completed` |
| tool call | `response.function_call_arguments.done` |
| nudge | `response.create` with `metadata.tyto_purpose = "nudge"` |
| interrupt | `response.cancel` (and clear the local playback buffer) |

## Running and verifying

```bash
uv pip install -e ".[dev]"
uv run pytest -q                 # 68 tests: decision, controller (+judge), scorer, jev, gpt-live
```

The unit tests need no SDK, key, network or hardware (Jev is tested through an
httpx mock transport, GPT-Live through a captured send and a fake clock). The
end-to-end audio path needs an ai-coustics key, an OpenAI key, a gateway key, a
mic, and headphones.

## Deploy

`deploy/modal_app.py` ships the web demo to Modal: app `tyto-demo` in
environment `tyto-demo`, URL label pinned to `tyto-demo`, secret
`tyto-demo-live-keys` (AIC_SDK_LICENSE, OPENAI_API_KEY, AI_GATEWAY_API_KEY),
Tyto model baked into the image at `/models`. The server honours `HOST`,
`PORT` and `AIC_MODELS_DIR` for that. Deploy with `-e tyto-demo` or the app
lands in the profile's default environment. Roll back with
`modal app rollback tyto-demo -e tyto-demo`.

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
