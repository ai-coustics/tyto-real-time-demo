"""The demo server must be listening before it is ready.

One ordering bug, worth a file of its own because it is invisible. Loading the
Tyto and Voice Focus models takes a second or two (a manifest fetch, then a read
off disk), and the browser starts sending audio the moment the WebRTC connection
is up. If the session loads models before it builds the pipeline, that early
audio has nowhere to go: a visitor who speaks straight after clicking the mic
loses their first utterance and never sees a transcript for it.

Nothing about that failure is loud. The agent recovers on the next turn and the
logs look fine, so the only way it stays fixed is a test.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("pipecat", reason="the server needs the voice stack installed")
pytest.importorskip("fastapi")

SERVER_PY = Path(__file__).resolve().parents[1] / "examples" / "pipecat" / "server.py"


def load_server_module():
    spec = importlib.util.spec_from_file_location("_tyto_demo_server", SERVER_PY)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_pipeline_starts_before_the_models_load(monkeypatch):
    """connect() must happen before scorer.start(), not after."""
    import asyncio

    server = load_server_module()
    events = []

    class FakeScorer:
        scoring = True

        def __init__(self, *a, **kw):
            self.on_scores = None

        def start(self):
            events.append("scorer.start")

        def stop(self):
            pass

        def pause(self):
            pass

        def resume(self):
            pass

        def feed(self, mono):
            pass

    class FakeVoiceFocus:
        available = False
        enabled = False

        def __init__(self, *a, **kw):
            pass

        def start(self):
            events.append("voice_focus.start")
            return False

        def stop(self):
            pass

    class FakeProvider:
        def __init__(self, handlers, **kw):
            self.h = handlers
            self.audio_quality_fn = None

        def connect(self):
            events.append("provider.connect")

        def disconnect(self):
            pass

        def send_ui(self, message):
            pass

        def set_instructions(self, text):
            pass

        def set_turn_detection(self, td):
            pass

        def set_mic_enabled(self, on):
            pass

        def interrupt(self, clear_input=False):
            pass

        def nudge(self, text):
            pass

        def request_response(self):
            pass

        def send_tool_result(self, call_id, output):
            pass

        def set_voice_focus(self, on):
            return False

    monkeypatch.setattr(server, "LiveTytoScorer", FakeScorer)
    monkeypatch.setattr(server, "VoiceFocus", FakeVoiceFocus)
    monkeypatch.setattr(server, "CascadeProvider", FakeProvider)

    async def go():
        session = server.Session(object(), {"license": "l", "deepgram": "d", "openai": "o"})
        await session.start()
        # The loads are deliberately backgrounded, so let them run.
        if session._load_task:
            await session._load_task

    asyncio.run(go())

    assert "provider.connect" in events, "the pipeline was never started"
    assert "scorer.start" in events, "the scorer was never started"
    assert events.index("provider.connect") < events.index("scorer.start"), (
        "the pipeline must be live before the models load, or early audio is lost"
    )
    assert events.index("provider.connect") < events.index("voice_focus.start")


def test_model_loading_does_not_block_the_connection(monkeypatch):
    """A slow or hanging model load must not stop the agent coming up."""
    import asyncio

    server = load_server_module()
    connected = []

    class SlowScorer:
        scoring = True

        def __init__(self, *a, **kw):
            self.on_scores = None

        def start(self):
            import time

            time.sleep(0.4)  # stand-in for a manifest fetch plus a model read

        def stop(self):
            pass

        def pause(self):
            pass

        def resume(self):
            pass

        def feed(self, mono):
            pass

    class FakeVoiceFocus:
        available = False
        enabled = False

        def __init__(self, *a, **kw):
            pass

        def start(self):
            return False

        def stop(self):
            pass

    class FakeProvider:
        def __init__(self, handlers, **kw):
            self.h = handlers
            self.audio_quality_fn = None

        def connect(self):
            connected.append(True)

        def __getattr__(self, name):
            return lambda *a, **kw: None

    monkeypatch.setattr(server, "LiveTytoScorer", SlowScorer)
    monkeypatch.setattr(server, "VoiceFocus", FakeVoiceFocus)
    monkeypatch.setattr(server, "CascadeProvider", FakeProvider)

    async def go():
        import time

        session = server.Session(object(), {"license": "l", "deepgram": "d", "openai": "o"})
        t0 = time.time()
        await session.start()
        elapsed = time.time() - t0
        assert connected, "the pipeline never connected"
        assert elapsed < 0.3, f"start() blocked for {elapsed:.2f}s on a model load"
        if session._load_task:
            await session._load_task

    asyncio.run(go())
