# Tyto in LiveKit

A LiveKit voice agent that scores the caller's audio with Tyto every 5 s and asks the caller to fix it when the audio is likely to break the pipeline (background podcast, noise, echo, bad connection).

Built on the ai-coustics LiveKit extras plugin (`ai-coustics-livekit-extras`).

## Docs

- LiveKit quickstart: https://docs.ai-coustics.com/models/get-started/livekit-quickstart
- Real-time analysis with Tyto: https://docs.ai-coustics.com/models/audio-insight/real-time-analysis

## Requirements

- Python 3.11
- A LiveKit Cloud project
- An ai-coustics SDK key from https://developers.ai-coustics.com/

## Setup

```sh
python3.11 -m venv .venv
.venv/bin/pip install ai-coustics-livekit-extras==0.25.2 aic-sdk==3.3.2 "livekit-agents==1.8.2" python-dotenv
```

Create `.env.local`:

```sh
LIVEKIT_URL=...
LIVEKIT_API_KEY=...
LIVEKIT_API_SECRET=...
AIC_SDK_LICENSE=...
```

Download the models:

```sh
.venv/bin/python -c 'from aic_sdk import Model; from shutil import copyfile; copyfile(Model.download("vad-ms-2.1-xxs-16khz", "models"), "models/vad.aicmodel")'
.venv/bin/python -c 'from aic_sdk import Model; from shutil import copyfile; copyfile(Model.download("tyto-1.1-l-16khz", "models"), "models/tyto.aicmodel")'
```

## Run

```sh
.venv/bin/python agent.py dev
```

Then join from the LiveKit Agent Console and dispatch the agent by the name set in `agent.py` (`agent_name="aic-tyto-demo"`, change it to anything you like). The agent only joins rooms it is dispatched to by that name.

Do not use `agent.py console`: it skips RoomIO, so Tyto gets no audio.

## What to expect

- About 5 s after you start talking, the worker logs a `Tyto risk=...` line every 5 s.
- When `risk` stays above 0.5 for two windows (about 10 s), the agent asks you to fix the top cause, then waits 60 s before asking again.
- Tune `RISK_THRESHOLD`, `CONSECUTIVE_WINDOWS` and `NUDGE_COOLDOWN` in `agent.py`.

## Coding agents

`SKILL.md` adds the same Tyto setup to an existing LiveKit project. Drop it into Claude Code, Cursor or any coding agent.
