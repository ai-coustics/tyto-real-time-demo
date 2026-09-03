"""Optional speech enhancement on the path to the agent, and only that path.

Quail VF is the ai-coustics enhancement model: it takes the microphone signal
and removes what is not the speaker. Here it is a switch the visitor can flip,
off by default, and it feeds exactly one consumer.

    mic ─┬──────────────────────────────────► Tyto     always raw
         └──► VoiceFocus (optional) ────────► Flux -> gpt-5-mini -> Aura-2

In the pipeline that ordering is literal: the Tyto tap sits before the Voice
Focus processor, so the scorer sees the microphone and Flux sees whatever the
switch says. See [cascade.py](cascade.py).

**Tyto must never be given the enhanced signal.** The demo exists to show what
the user's room is actually doing to their audio, and an enhancer in front of
the analyzer would have it scoring Quail's output instead: the meters would go
green, the room note would go quiet, and the Reactive layer would stop firing,
in a room that had not changed at all. The demo would still look like it worked,
which is what makes this the one wiring mistake here worth guarding in prose.
Enhance what the agent hears. Measure what the microphone heard.

Off by default for the same reason. Someone arriving at the page should meet
their room as it is; the switch is what shows the difference, and a difference
needs a before.

The model is ``quail-vf-2.2-l-16khz``, native at the 16 kHz the rest of the
capture chain runs at. Measured here at about 6% of one core for realtime audio,
adding 30 ms of delay, so it is affordable to leave switched on. This is the one
place that number is stated; the README quotes it.
"""

from __future__ import annotations

import os
import threading

import numpy as np

DEFAULT_MODEL = os.environ.get("AIC_VOICE_FOCUS_MODEL", "quail-vf-2.2-l-16khz")
DEFAULT_MODELS_DIR = os.environ.get("AIC_MODELS_DIR", "./models")


class VoiceFocus:
    """Block-aligned enhancement with a runtime on/off switch.

    ``process`` takes mono float32 of any length and returns the enhanced audio
    that is ready, carrying a residual between calls the way the scorer does:
    the SDK wants exactly ``block_size`` samples per call. While switched off it
    returns its input untouched and costs nothing.
    """

    def __init__(
        self,
        license_key: str,
        *,
        model_id: str = DEFAULT_MODEL,
        models_dir: str = DEFAULT_MODELS_DIR,
        sample_rate: int = 16000,
        on_log=None,
    ):
        self._license_key = license_key
        self._model_id = model_id
        self._models_dir = models_dir
        self._requested_rate = sample_rate
        self._on_log = on_log

        self.sample_rate = 0
        self.block_size = 0
        self.available = False

        self._processor = None
        self._lock = threading.Lock()
        self._enabled = False
        self._residual = np.empty(0, dtype=np.float32)

    # -- lifecycle ---------------------------------------------------------- #

    def start(self) -> bool:
        """Load the model. Returns False if it is unavailable, which is not
        fatal: the switch simply stays off and the demo runs without it."""
        try:
            import aic_sdk as aic

            path = aic.Model.download(self._model_id, self._models_dir)
            model = aic.Model.from_file(path)
            config = aic.ProcessorConfig.optimal(model, sample_rate=self._requested_rate)
            processor = aic.Processor(model, self._license_key)
            processor.initialize(config)
            self._processor = processor
            self.sample_rate = config.sample_rate
            self.block_size = config.block_size
            self.available = True
            self._log("vf.ready", f"{self._model_id} loaded, off by default")
        except Exception as err:  # noqa: BLE001 - an optional feature must not break the demo
            self._log("error", f"voice focus unavailable: {err}")
            self.available = False
        return self.available

    def stop(self) -> None:
        if self._processor is not None:
            try:
                self._processor.terminate_session()
            except Exception:  # noqa: BLE001 - teardown must not raise
                pass
            self._processor = None

    # -- control (any thread) ----------------------------------------------- #

    def set_enabled(self, on: bool) -> bool:
        """Returns the state actually reached, which is False if unavailable."""
        with self._lock:
            self._enabled = bool(on) and self.available
            if not self._enabled:
                self._residual = np.empty(0, dtype=np.float32)
            state = self._enabled
        self._log("vf.toggle", "on" if state else "off")
        return state

    @property
    def enabled(self) -> bool:
        return self._enabled

    # -- audio (mic thread) -------------------------------------------------- #

    def process(self, mono: np.ndarray) -> np.ndarray:
        """Enhanced audio when on, the input unchanged when off."""
        with self._lock:
            if not self._enabled or self._processor is None:
                return mono
            data = np.concatenate([self._residual, np.ascontiguousarray(mono, dtype=np.float32)])
            offset, n = 0, self.block_size
            out = []
            while len(data) - offset >= n:
                # process() may write in place, so never hand it a view of the
                # caller's buffer: Tyto is being fed the same array.
                block = np.array(data[offset : offset + n], dtype=np.float32)
                offset += n
                try:
                    out.append(np.asarray(self._processor.process(block), dtype=np.float32))
                except Exception as err:  # noqa: BLE001
                    self._log("error", f"voice focus: {err}")
                    self._enabled = False
                    # set_enabled's clear is bypassed on this path, so drop the
                    # residual here or a re-enable prepends stale audio.
                    self._residual = np.empty(0, dtype=np.float32)
                    return mono
            self._residual = data[offset:].copy()
        return np.concatenate(out) if out else np.empty(0, dtype=np.float32)

    def _log(self, kind: str, text: str) -> None:
        if self._on_log:
            self._on_log(kind, text)
