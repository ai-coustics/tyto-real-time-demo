# Tyto in Pipecat

A Pipecat voice bot (Deepgram Flux STT, OpenAI gpt-5-mini, Deepgram Aura-2 TTS) that scores the caller's audio with Tyto every second and asks the caller to fix it when the audio is likely to break the pipeline (background podcast, noise, echo, bad connection). At the end of the call it logs a summary and flags calls with bad audio.

Built on the official Pipecat ai-coustics integration (`AICTytoObserver`, Pipecat 1.15.0 and later).

## Docs

- Pipecat quickstart: https://docs.ai-coustics.com/models/get-started/pipecat-quickstart
- Real-time analysis with Tyto: https://docs.ai-coustics.com/models/audio-insight/real-time-analysis
- Pipecat's own Tyto example: https://github.com/pipecat-ai/pipecat/blob/v1.15.0/examples/voice/voice-aicoustics-tyto.py

## Requirements

- Python 3.11
- An ai-coustics SDK key from https://developers.ai-coustics.com/
- A Deepgram API key (STT and TTS)
- An OpenAI API key with access to `gpt-5-mini`

## Setup

```sh
python3.11 -m venv .venv
.venv/bin/pip install "pipecat-ai[aic,deepgram,openai,webrtc,runner]==1.15.0"
```

Create `.env.local`:

```sh
AIC_SDK_LICENSE=...
DEEPGRAM_API_KEY=...
OPENAI_API_KEY=...
```

The Tyto model downloads on first run to `~/.cache/pipecat/aic-models`.

## Run

```sh
.venv/bin/python bot.py
```

Open http://localhost:7860, allow the mic and click Connect.

## What to expect

- From about 1 s after you connect, the bot logs a `Tyto ... risk=... smoothed=... -> decision` line every second. Each one scores the latest 5 s of audio.
- When the smoothed risk (a moving average, so one bad second does not count) goes above 0.4, the bot asks you to fix the top cause, then waits 60 s before asking again. If it is mid-reply, it stops and asks right away. Background audio cannot interrupt that line.
- Each score also shows up in the browser client's Events pane as a server message.
- When you disconnect, it logs `Tyto call summary ... flagged=True` if 20% or more of the call had a risk of 0.3 or more.
- Each call is saved to `recordings/<date-time>.wav` on disconnect: your raw mic (what Tyto scored) on the left, the bot on the right.
- Tune `RISK_THRESHOLD`, `SMOOTHING`, `ANALYSIS_HOP`, `NUDGE_COOLDOWN`, `DEGRADED_THRESHOLD` and `FLAG_FRACTION` in `bot.py`.

## Coding agents

`SKILL.md` adds the same Tyto setup to an existing Pipecat project. Drop it into Claude Code, Cursor or any coding agent.
