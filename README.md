# Tyto in LiveKit

A LiveKit voice agent that scores the caller's audio with Tyto 1.2 every second and asks the caller to fix it when the audio is likely to break the pipeline (background podcast, noise, echo, bad connection).

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
.venv/bin/pip install ai-coustics-livekit-extras==0.26.0 aic-sdk==3.4.0 "livekit-agents==1.8.2" python-dotenv
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
.venv/bin/python -c 'from aic_sdk import Model; from shutil import copyfile; copyfile(Model.download("tyto-1.2-l-16khz", "models"), "models/tyto.aicmodel")'
```

Upgrading from Tyto 1.1: run both commands again. aic-sdk 3.4.0 needs model file format 8, so the old VAD file fails to load too, even though its model ID is unchanged.

## Run

```sh
.venv/bin/python agent.py dev
```

Then join from the LiveKit Agent Console and dispatch the agent by the name set in `agent.py` (`agent_name="aic-tyto-demo"`, change it to anything you like). The agent only joins rooms it is dispatched to by that name.

Do not use `agent.py console`: it skips RoomIO, so Tyto gets no audio.

## What to expect

- About a second after audio reaches the agent, the worker logs a `Tyto #N ...` line every second. The first 5 lines say `warm-up`: Tyto 1.2 reads a partly filled 5 s window as high risk, so those scores are ignored.
- When the smoothed risk goes above 0.4, the agent cuts in with a fixed line for the top cause, then waits 60 s before asking again.
- Tune `RISK_THRESHOLD`, `SMOOTHING`, `WARMUP_RESULTS` and `NUDGE_COOLDOWN` in `agent.py`.

## Coding agents

`SKILL.md` adds the same Tyto setup to an existing LiveKit project. Drop it into Claude Code, Cursor or any coding agent.
