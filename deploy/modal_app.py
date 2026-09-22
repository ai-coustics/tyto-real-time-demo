"""Deploy the Tyto voice demo (GPT-Live 1 + Jev) to Modal.

    modal deploy deploy/modal_app.py -e tyto-demo

The demo is an aiohttp process holding one websocket per browser tab, so it is
served with ``@modal.web_server`` rather than an ASGI adapter: aiohttp is not
ASGI, and Modal proxies the full websocket protocol to a plain listening port.
Modal terminates TLS, which matters because a browser will not hand over a
microphone on an insecure origin.

The Tyto model is baked into the image at build time. Downloading it needs no
licence, only network, so a container starts with it on disk instead of
fetching it on every cold start.

Set up once, in the environment you deploy to (secrets are per environment):

    modal secret create tyto-demo-live-keys \\
        AIC_SDK_LICENSE=... OPENAI_API_KEY=... AI_GATEWAY_API_KEY=... -e tyto-demo

``tyto-demo`` is a Modal *environment* as well as the app name. The public URL
is ``https://<workspace>-<environment>--<label>.modal.run``, so for the
ai-coustics workspace this app answers at
https://ai-coustics-tyto-demo--tyto-demo.modal.run/.

Deploying an app name replaces the whole set of functions under it, including
the URLs of anything it no longer defines, so the label below is pinned rather
than derived from the function name. To go back to the previous version:

    modal app rollback tyto-demo -e tyto-demo
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import modal

# Deploying to an app name defines that app's entire set of functions, so a
# deploy under an existing name removes whatever else was there. Pair it with
# `-e tyto-demo`, or the deploy lands in your profile's default environment.
APP_NAME = os.environ.get("MODAL_APP_NAME", "tyto-demo")

# Where the Tyto model lives inside the image.
MODELS_DIR = "/models"
TYTO_MODEL = "tyto-1.1-l-16khz"

# The public URL is derived from the app name and this label, NOT from the name
# of the Python function. Pin it, or renaming the function silently moves the URL.
URL_LABEL = "tyto-demo"

PORT = 8080
# The page loads, the socket opens, nothing is downloaded (the model is baked),
# but be generous in case a cold start lands behind an image pull.
STARTUP_TIMEOUT = 60.0

REPO = Path(__file__).parent.parent
SRC_PKG = REPO / "src" / "tyto_voice"
WEB_DIR = REPO / "examples" / "web"
REMOTE_PKG = "/root/tyto_voice"  # /root is the working directory, so `import tyto_voice` just works
REMOTE_WEB = "/root/web"
SECRET_NAME = "tyto-demo-live-keys"


def _bake_model() -> None:
    """Fetch the Tyto model during the image build. No licence needed to download."""
    import aic_sdk as aic

    path = aic.Model.download(TYTO_MODEL, MODELS_DIR)
    print(f"baked {TYTO_MODEL} -> {path}")


image = (
    modal.Image.debian_slim(python_version="3.12")
    # Same floors as pyproject.toml. sounddevice is deliberately absent: only the
    # terminal demo imports it, and it would need PortAudio for a code path that
    # never runs on a server.
    .pip_install(
        "aic-sdk>=3.1",
        "numpy>=1.24",
        "aiohttp>=3.9",
        "websockets>=13",
        "httpx>=0.27",
    )
    .run_function(_bake_model)
    # The library as a plain directory (no local import needed at deploy time),
    # then the page, client script, design tokens and assets the server reads.
    .add_local_dir(SRC_PKG, remote_path=REMOTE_PKG, ignore=["**/__pycache__/**"])
    .add_local_dir(WEB_DIR, remote_path=REMOTE_WEB, ignore=["**/__pycache__/**"])
)

app = modal.App(APP_NAME, image=image)

# Keys stay server side. The browser only ever exchanges audio with this app.
secrets = [modal.Secret.from_name(SECRET_NAME)]


@app.function(
    secrets=secrets,
    # One websocket is one input, and each session runs its own Tyto analyzer
    # (about 100 ms per 5 s window at a 1 s hop, so a fraction of a core) plus
    # the GPT-Live socket and Jev calls, which are network-bound.
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
    env = {
        **os.environ,
        "HOST": "0.0.0.0",  # Modal's proxy reaches the container, not loopback
        "PORT": str(PORT),
        "AIC_MODELS_DIR": MODELS_DIR,
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": "/root",
    }
    subprocess.Popen([sys.executable, str(Path(REMOTE_WEB) / "server.py")], env=env)


@app.local_entrypoint()
def check() -> None:
    """`modal run deploy/modal_app.py -e tyto-demo` - build the image without deploying."""
    print(f"app name        : {APP_NAME}")
    print(f"url label       : {URL_LABEL}")
    print(f"model baked at  : {MODELS_DIR}/{TYTO_MODEL}")
    print(f"library         : {SRC_PKG} -> {REMOTE_PKG}")
    print(f"web dir         : {WEB_DIR} -> {REMOTE_WEB}")
    print(f"secret          : {SECRET_NAME}")
