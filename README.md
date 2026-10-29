# Tyto in Pipecat

A Pipecat voice bot that scores the caller's audio with Tyto every 5 s and asks the caller to fix it when the audio is likely to break the pipeline (background podcast, noise, echo, bad connection). At the end of the call it logs a summary and flags calls with bad audio.

Built on the official Pipecat ai-coustics integration (`AICTytoObserver`, Pipecat 1.15.0 and later).

## Docs

- Pipecat quickstart: https://docs.ai-coustics.com/models/get-started/pipecat-quickstart
- Real-time analysis with Tyto: https://docs.ai-coustics.com/models/audio-insight/real-time-analysis
- Pipecat's own Tyto example: https://github.com/pipecat-ai/pipecat/blob/v1.15.0/examples/voice/voice-aicoustics-tyto.py

## Requirements

- Python 3.11
- An ai-coustics SDK key from https://developers.ai-coustics.com/
- An OpenAI API key (STT, LLM and TTS)

## Setup

```sh
python3.11 -m venv .venv
.venv/bin/pip install "pipecat-ai[aic,openai,webrtc,runner]==1.15.0"
```

Create `.env.local`:

```sh
AIC_SDK_LICENSE=...
OPENAI_API_KEY=...
```

The Tyto and VAD models download on first run to `~/.cache/pipecat/aic-models`.

## Run

```sh
.venv/bin/python bot.py
```

Open http://localhost:7860, allow the mic and click Connect.

## What to expect

- About 5 s after you connect, the bot logs a `Tyto risk=...` line every 5 s.
- When `risk` stays above 0.5 for two windows (about 10 s), the bot asks you to fix the top cause, then waits 60 s before asking again. The background audio cannot interrupt that line.
- When you disconnect, it logs `Tyto call summary ... flagged=True` if 20% or more of the call had a risk of 0.3 or more.
- Tune `RISK_THRESHOLD`, `CONSECUTIVE_WINDOWS`, `NUDGE_COOLDOWN`, `DEGRADED_THRESHOLD` and `FLAG_FRACTION` in `bot.py`.

## Coding agents

`SKILL.md` adds the same Tyto setup to an existing Pipecat project. Drop it into Claude Code, Cursor or any coding agent.
