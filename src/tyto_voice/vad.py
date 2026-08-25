"""Turn-taking over the ai-coustics VAD, the local replacement for server-side
turn detection.

The OpenAI Realtime backend decided when the user had finished talking. A
cascaded stack has to do that itself, so this is where the demo's turn-taking
now lives: ``LiveVad`` runs the ``vad-2.1-xxs-16khz`` model over the mic and
turns a stream of blocks into whole utterances.

    feed(mono) -> None | np.ndarray     one utterance, when the user stops

Two details make the difference between this feeling right and feeling broken:

- **Pre-roll.** The VAD reports speech a little after it starts, because it
  needs ``minimum_speech_duration`` of evidence and the model itself lags by
  ``get_prediction_delay()`` samples. Without a pre-roll buffer the first
  syllable is missing from every utterance. We keep ``PREROLL_SECONDS`` of audio
  behind the write head and prepend it.

- **The falling edge is not the end of the turn.** Measured against real
  speech, ``is_speech_detected()`` drops out for 45 to 285 ms at ordinary pauses
  inside one sentence, and raising ``speech_hold_duration`` does not close those
  gaps. Ending on the falling edge split a single 4.8 s question into three
  utterances. So a turn ends only after ``end_silence`` seconds of *continuous*
  silence, which is the knob Layer 2 (Tuned) swaps: 1.10 s eager, 1.50 s
  patient. Those carry half a second of grace over the worst gap measured, so
  thinking mid-sentence does not end your turn. The trailing silence is trimmed
  off before the utterance is handed on, so it costs no tokens at the model.

The VAD model is tiny (xxs) and is designed to run in the audio path, so
:meth:`feed` is called straight from the mic callback, like the scorer.
"""

from __future__ import annotations

import os
import threading

import numpy as np

from .decision import VAD_PROFILES

DEFAULT_MODEL = "vad-2.1-xxs-16khz"

# Where downloaded models are cached. A deployment that bakes them into its
# image points AIC_MODELS_DIR at them so a container does not refetch on every
# cold start.
DEFAULT_MODELS_DIR = os.environ.get("AIC_MODELS_DIR", "./models")

# Audio kept behind the write head so an utterance never starts clipped.
PREROLL_SECONDS = 0.5
# Shorter than this and it was a cough, a door, or a stray click. Dropped.
MIN_UTTERANCE_SECONDS = 0.4
# A turn is forced after this long so one monologue cannot grow without bound.
# Also the memory ceiling: 30 s of 16 kHz float32 is under 2 MB.
MAX_UTTERANCE_SECONDS = 30.0
# Silence left on the end of an utterance after trimming. A little tail sounds
# natural to the model; the rest is dead weight in the prompt.
TAIL_SECONDS = 0.2


class LiveVad:
    """Segments a mono float32 stream into utterances. Not thread safe by
    itself; :meth:`feed` is expected to be called from one audio thread, while
    :meth:`set_profile` may be called from any thread.
    """

    def __init__(
        self,
        license_key: str,
        *,
        model_id: str = DEFAULT_MODEL,
        models_dir: str = DEFAULT_MODELS_DIR,
        sample_rate: int | None = None,
        profile: dict | None = None,
    ):
        self._license_key = license_key
        self._model_id = model_id
        self._models_dir = models_dir
        self._requested_rate = sample_rate

        self.sample_rate = 0
        self.block_size = 0

        self._vad = None
        self._ctx = None
        self._params = None  # aic.VadParameter, resolved at start()

        self._lock = threading.Lock()
        self._pending_profile = profile or VAD_PROFILES["eager"]
        self._enabled = True

        self._residual = np.empty(0, dtype=np.float32)
        self._preroll: list[np.ndarray] = []
        self._preroll_samples = 0
        self._utterance: list[np.ndarray] = []
        self._utterance_samples = 0
        self._speaking = False
        self._silence_run = 0  # consecutive silent samples while in a turn
        # Set from the profile once the sample rate is known. A profile without
        # an end_silence still gets a workable turn end rather than an error.
        self._end_silence_samples = 0

    # -- lifecycle ---------------------------------------------------------- #

    def start(self) -> None:
        """Download the model (cached) and initialize the detector."""
        import aic_sdk as aic

        model_path = aic.Model.download(self._model_id, self._models_dir)
        model = aic.Model.from_file(model_path)

        rate = self._requested_rate or model.get_optimal_sample_rate()
        config = aic.ProcessorConfig.optimal(model, sample_rate=rate)

        vad = aic.Vad(model, self._license_key, config)
        self._configure(vad, vad.get_context(), aic.VadParameter,
                        config.sample_rate, config.block_size)

    def _configure(self, vad, ctx, params, sample_rate: int, block_size: int) -> None:
        """Adopt a detector. Split out from :meth:`start` so the segmentation
        logic can be tested without the SDK or a licence."""
        self._vad = vad
        self._ctx = ctx
        self._params = params
        self.sample_rate = sample_rate
        self.block_size = block_size
        self._preroll_limit = round(PREROLL_SECONDS * sample_rate)
        self._min_samples = round(MIN_UTTERANCE_SECONDS * sample_rate)
        self._max_samples = round(MAX_UTTERANCE_SECONDS * sample_rate)
        self._tail_samples = round(TAIL_SECONDS * sample_rate)
        self._apply_pending_profile()

    def stop(self) -> None:
        if self._vad is not None:
            try:
                self._vad.terminate_session()
            except Exception:  # noqa: BLE001 - teardown must not raise
                pass
            self._vad = None
            self._ctx = None

    # -- control (any thread) ----------------------------------------------- #

    def set_profile(self, profile: dict | None) -> None:
        """Layer 2 - Tuned, and the listen gate.

        ``profile`` is a ``VAD_PROFILES`` entry. ``None`` means stop listening,
        which mirrors ``set_turn_detection(None)`` on the provider seam, and
        drops any utterance in progress.
        """
        if profile is None:
            with self._lock:
                already_off = not self._enabled
                self._enabled = False
            if not already_off:
                self.reset()
            return
        with self._lock:
            self._pending_profile = profile
            self._enabled = True
        self._apply_pending_profile()

    @property
    def enabled(self) -> bool:
        return self._enabled

    def reset(self) -> None:
        """Drop all buffered audio and clear the detector's state."""
        if self._ctx is not None:
            try:
                self._ctx.reset()
            except Exception:  # noqa: BLE001
                pass
        self._residual = np.empty(0, dtype=np.float32)
        self._preroll = []
        self._preroll_samples = 0
        self._utterance = []
        self._utterance_samples = 0
        self._speaking = False
        self._silence_run = 0

    def _apply_pending_profile(self) -> None:
        with self._lock:
            profile = self._pending_profile
            if profile is not None and self._ctx is not None:
                self._pending_profile = None
        if profile is None or self._ctx is None:
            return
        # The SDK rounds durations to the model's window length, so reading a
        # parameter back may not return what was written. That is expected.
        self._ctx.set_parameter(self._params.Sensitivity, profile["sensitivity"])
        self._ctx.set_parameter(
            self._params.MinimumSpeechDuration, profile["minimum_speech_duration"]
        )
        self._ctx.set_parameter(
            self._params.SpeechHoldDuration, profile["speech_hold_duration"]
        )
        # Ours, not the SDK's: how much continuous silence ends the turn.
        self._end_silence_samples = round(
            profile.get("end_silence", 0.6) * self.sample_rate
        )

    # -- audio in (mic thread) ---------------------------------------------- #

    def feed(self, mono: np.ndarray) -> np.ndarray | None:
        """Push mono float32 audio of any length.

        Returns one complete utterance when the user stops talking, otherwise
        None. The SDK needs exactly ``block_size`` samples per call, so a
        residual is carried between calls, like the scorer does.
        """
        if self._vad is None or not self._enabled:
            return None
        self._apply_pending_profile()

        data = np.concatenate([self._residual, np.ascontiguousarray(mono, dtype=np.float32)])
        offset, n = 0, self.block_size
        finished = None
        while len(data) - offset >= n:
            block = data[offset : offset + n]
            offset += n
            done = self._consume(block)
            if done is not None and finished is None:
                finished = done
        self._residual = data[offset:].copy()
        return finished

    def _consume(self, block: np.ndarray) -> np.ndarray | None:
        self._vad.process(block)  # does not modify the block
        speech = self._ctx.is_speech_detected()

        if speech and not self._speaking:  # silence -> speech
            self._speaking = True
            self._utterance = list(self._preroll)
            self._utterance_samples = self._preroll_samples

        if self._speaking:
            self._utterance.append(block.copy())
            self._utterance_samples += len(block)
            if speech:
                self._silence_run = 0
            else:
                # Short dropouts happen mid-sentence, so only a continuous run
                # of silence ends the turn.
                self._silence_run += len(block)
                if self._silence_run >= self._end_silence_samples:
                    self._speaking = False
                    return self._take_utterance()
            if self._utterance_samples >= self._max_samples:
                return self._take_utterance(keep_speaking=True)
            return None

        # Idle: keep a rolling pre-roll so the next utterance is not clipped.
        self._preroll.append(block.copy())
        self._preroll_samples += len(block)
        while self._preroll_samples - len(self._preroll[0]) >= self._preroll_limit:
            self._preroll_samples -= len(self._preroll.pop(0))
        return None

    def _take_utterance(self, keep_speaking: bool = False) -> np.ndarray | None:
        audio = np.concatenate(self._utterance) if self._utterance else None
        # Trim the silence that ended the turn, leaving a short natural tail.
        if audio is not None and self._silence_run > self._tail_samples:
            drop = self._silence_run - self._tail_samples
            audio = audio[:-drop] if drop < len(audio) else audio[:0]
        self._utterance = []
        self._utterance_samples = 0
        self._silence_run = 0
        self._preroll = []
        self._preroll_samples = 0
        # A forced cut mid-sentence keeps the turn open, so the rest of what the
        # user is saying becomes the next utterance instead of being dropped.
        self._speaking = keep_speaking
        if audio is None or len(audio) < self._min_samples:
            return None
        return audio

    # -- introspection ------------------------------------------------------ #

    @property
    def speaking(self) -> bool:
        return self._speaking

    @property
    def speech_samples(self) -> int:
        """Audio collected in the turn currently in progress, pre-roll included.

        Barge-in reads this to require a real run of speech before it believes
        the user has started talking over the agent.
        """
        return self._utterance_samples

    def probability(self) -> float:
        return self._ctx.raw_vad_probability() if self._ctx is not None else 0.0
