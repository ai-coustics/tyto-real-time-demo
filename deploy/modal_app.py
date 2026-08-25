"""Deploy the Tyto voice demo to Modal.

    modal deploy deploy/modal_app.py

The demo is an aiohttp process holding one websocket per browser tab, so it is
served with ``@modal.web_server`` rather than an ASGI adapter: aiohttp is not
ASGI, and Modal proxies the full websocket protocol to a plain listening port.
Modal terminates TLS for us, which matters because a browser will not hand over
a microphone on an insecure origin.

The two ai-coustics models are baked into the image at build time. Downloading
them needs no licence, only network, so a container starts with them already on
disk instead of fetching about 35 MB on every cold start.

Set up once, in the environment you deploy to:

    modal secret create tyto-demo-keys \\
        AIC_SDK_LICENSE=... INKLING_API_KEY=... DEEPGRAM_API_KEY=... -e tyto-demo

Note that ``tyto-demo`` is a Modal *environment* as well as the app name, and
secrets are per environment. The app URL reads
``/apps/<workspace>/<environment>/<app-id>``, which is easy to misread as an app
path.

Deploying an app name replaces the whole set of functions under it, and that
includes retiring the URLs of any function it no longer defines. A retired URL
answers "modal-http: invalid function call", so the label below is pinned rather
than derived from the function name. To go back:

    modal app rollback <app-id> -e tyto-demo
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import modal

# Deploying to an app name defines that app's entire set of functions, so a
# deploy under an existing name removes whatever else was there. Kept separate
# and obvious for that reason. Pair it with `-e tyto-demo`, or the deploy lands
# in whatever environment your profile defaults to.
APP_NAME = os.environ.get("MODAL_APP_NAME", "tyto-demo")

# Where the ai-coustics models live inside the image.
MODELS_DIR = "/models"
TYTO_MODEL = "tyto-1.1-l-16khz"
VAD_MODEL = "vad-2.1-xxs-16khz"

# The public URL is derived from the app name and this label, NOT from the name
# of the Python function. Pin it, or renaming the function silently moves the URL
# and every bookmark starts returning "modal-http: invalid function call".
URL_LABEL = "tyto-demo"

PORT = 8080
# Enough for a browser to load the page, open its socket, and for the server to
# pull down nothing (the models are baked), but generous in case a cold start
# lands behind an image pull.
STARTUP_TIMEOUT = 60.0

REPO = Path(__file__).parent.parent
WEB_DIR = REPO / "examples" / "web"
REMOTE_WEB = "/root/web"


def _bake_models() -> None:
    """Fetch both models during the image build. No licence needed to download."""
    import aic_sdk as aic

    for model_id in (TYTO_MODEL, VAD_MODEL):
        path = aic.Model.download(model_id, MODELS_DIR)
        print(f"baked {model_id} -> {path}")


image = (
    modal.Image.debian_slim(python_version="3.12")
    # Pinned to the same floors as pyproject.toml. sounddevice is deliberately
    # absent: it is only imported by the terminal demo, and it would need
    # PortAudio in the image for a code path that never runs on a server.
    .pip_install(
        "aic-sdk>=3.1",
        "numpy>=1.24",
        "aiohttp>=3.9",
        "websockets>=13",
    )
    .run_function(_bake_models)
    # The library, and the page and client script the server reads at runtime.
    .add_local_python_source("tyto_voice")
    .add_local_dir(WEB_DIR, remote_path=REMOTE_WEB)
)

app = modal.App(APP_NAME, image=image)

# Keys stay server side. The browser only ever exchanges audio with this app.
secrets = [modal.Secret.from_name("tyto-demo-keys")]


@app.function(
    secrets=secrets,
    # One websocket is one input, and each session runs its own Tyto analyzer
    # and VAD. Tyto measures 116 ms per window at a 0.5 s hop, so a session
    # costs roughly a quarter of a core once it is warm, plus the VAD.
    cpu=4.0,
    memory=2048,
    max_containers=10,
    # A browser tab holds its socket open, so do not tear the container down the
    # moment a request finishes.
    scaledown_window=300,
)
@modal.concurrent(max_inputs=8, target_inputs=4)
@modal.web_server(PORT, startup_timeout=STARTUP_TIMEOUT, label=URL_LABEL)
def web() -> None:
    """Start the demo's own aiohttp server and let Modal proxy to it.

    Run as a subprocess rather than in-process: ``web_server`` expects the
    decorated function to return once something is listening, and aiohttp's
    ``run_app`` owns the event loop for the lifetime of the process.
    """
    sys.path.insert(0, REMOTE_WEB)
    env = {
        **os.environ,
        "HOST": "0.0.0.0",          # Modal's proxy reaches the container, not loopback
        "PORT": str(PORT),
        "AIC_MODELS_DIR": MODELS_DIR,
        "PYTHONUNBUFFERED": "1",
    }
    subprocess.Popen(
        [sys.executable, str(Path(REMOTE_WEB) / "server.py")],
        env=env,
    )


@app.local_entrypoint()
def check() -> None:
    """`modal run deploy/modal_app.py` - confirm the image is sound before deploying."""
    print(f"app name        : {APP_NAME}")
    print(f"models baked at : {MODELS_DIR}")
    print(f"web dir mounted : {WEB_DIR} -> {REMOTE_WEB}")
    print("secret          : tyto-demo-keys")
