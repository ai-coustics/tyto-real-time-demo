"""Inkling-Small, the audio-in / text-out brain of the cascade.

Inkling-Small is multimodal, so it takes the user's utterance as audio and
returns the reply as text. That collapses speech-to-text and the LLM into one
hop, which is why this demo has no separate STT on the critical path.

Served over Thinking Machines' OpenAI-compatible endpoint, so the wire format is
ordinary ``/chat/completions``. Three things were established against the live
API and are load-bearing:

- The model id must be exactly ``thinkingmachines/Inkling-Small``. The bare
  ``Inkling-Small`` is rejected with "Sampling is not supported".
- ``reasoning_effort="none"`` is required. With thinking on, a reply takes about
  2.8 s instead of 1.2 s, and if it is cut short by ``max_tokens`` the raw
  chain of thought lands in ``content`` and gets spoken out loud.
- Streaming buys nothing. The whole reply arrives as a single chunk, so
  time-to-first-token equals time-to-completion and non-streaming is simpler.

Nothing transcribes the user, so history keeps the utterance itself. Audio is
roughly 27 prompt tokens per second and each retained audio turn adds about
0.25 s of latency, so only the most recent ``AUDIO_HISTORY_TURNS`` user turns are
carried; older ones fall out while the agent's own replies stay as text, which
keeps the thread of the conversation for almost nothing.

Only the standard library is used for transport, so this adds no dependency.
"""

from __future__ import annotations

import base64
import io
import json
import os
import threading
import urllib.error
import urllib.request
import wave
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

BASE_URL = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"

# The endpoint serves many models. Inkling-Small is the one this demo is built
# on and the only one measured to work well for it; see the note below. Override
# with INKLING_MODEL to try another.
MODEL = os.environ.get("INKLING_MODEL", "thinkingmachines/Inkling-Small")

# Benchmarked on this endpoint, same prompt, same question, median of 3, with a
# 120 token cap. Only Inkling accepts reasoning_effort, and that is what decides
# it: every other model here spent its whole budget thinking and returned an
# empty reply, which a voice agent cannot use.
#
#   thinkingmachines/Inkling-Small (text)   1.02s   40 tok   clean
#   thinkingmachines/Inkling-Small (audio)  1.15s   45 tok   clean
#   deepseek-ai/DeepSeek-V3.1               1.00s   28 tok   mangled: "TytĠisĠaĠspeech"
#   openai/gpt-oss-120b                     1.19s  118 tok   verbose, hit the cap
#   Qwen/Qwen3.5-4B                         1.85s  120 tok   empty, all reasoning
#   Qwen/Qwen3.6-35B-A3B                    2.19s  120 tok   empty, all reasoning
#   Qwen/Qwen3.6-27B                        2.42s  120 tok   empty, all reasoning
#   moonshotai/Kimi-K2.6                    2.37s  120 tok   empty, all reasoning
#   Qwen/Qwen3.5-9B                         2.62s  120 tok   empty, all reasoning
#   openai/gpt-oss-20b                      2.77s  120 tok   empty, all reasoning
#   nvidia/Nemotron-3-*                     unavailable for sampling on this key
#
# Note the audio penalty is only about 0.13 s against the same text. That is the
# number that decides the architecture: replacing audio input with a separate
# speech-to-text pass would save a tenth of a second and add a service.

# Inkling wants 16 kHz mono WAV, which is also what the VAD and Tyto run at, so
# the whole capture chain is 16 kHz and nothing resamples.
SAMPLE_RATE = 16000

# Spoken replies are short by instruction; this is only a runaway guard.
MAX_TOKENS = 160
REQUEST_TIMEOUT = 30.0

# The endpoint sits behind Cloudflare, which answers urllib's default
# "Python-urllib/3.x" agent with a 403 (error code 1010). Any ordinary agent
# string gets through; this one also makes the traffic identifiable.
USER_AGENT = "tyto-voice/0.1 (+https://github.com/ai-coustics)"

# Turns kept in history at all (a turn is one user utterance or one agent reply).
MAX_HISTORY_TURNS = 12
# How many of the user's utterances are carried as audio.
#
# This must be 1, and the reason is not latency. The model does not treat a past
# utterance as history, it *listens* to it, and then describes the room it hears
# there as though it were the room now. With 2, a turn recorded while a TV was on
# made the agent answer an unrelated question with "you sound like you're in a
# quiet room with people talking in the background", minutes after the TV was
# off. The live reading does not win that argument, because the model can hear
# the evidence against it.
#
# Nothing is really lost. The agent's own replies stay in history as text, which
# carries the thread of the conversation, and each retained audio turn also cost
# roughly 0.25 s of latency on every later turn.
AUDIO_HISTORY_TURNS = 1


def encode_wav(mono: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Mono float32 in, 16-bit PCM WAV out.

    We always build the header ourselves. Deepgram's own ``container=wav`` writes
    a streaming placeholder length (0x7fff...), which every WAV reader then
    misparses as roughly nineteen hours of audio.
    """
    pcm16 = (np.clip(mono, -1.0, 1.0) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm16.tobytes())
    return buf.getvalue()


@dataclass
class _UserTurn:
    audio_b64: str


@dataclass
class _AgentTurn:
    text: str


@dataclass
class Reply:
    text: str
    tool_calls: int = 0
    messages: list = field(default_factory=list)


class InklingClient:
    """One conversation. ``respond`` is blocking and expects a worker thread."""

    def __init__(
        self,
        api_key: str,
        *,
        instructions: str,
        tools: list | None = None,
        model: str = MODEL,
        base_url: str = BASE_URL,
        on_log: Callable[[str, str], None] | None = None,
    ):
        self._api_key = api_key
        self._model = model
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._tools = tools or []
        self._on_log = on_log

        self._lock = threading.Lock()
        self._instructions = instructions
        self._turns: list = []

    # -- conversation state -------------------------------------------------- #

    def set_instructions(self, text: str) -> None:
        """Layer 1 - Aware. Swapped in whole, applied on the next turn."""
        with self._lock:
            self._instructions = text

    def add_agent_line(self, text: str) -> None:
        """Record something the agent said outside a normal turn (a nudge, the
        opening greeting) so it knows it said it."""
        with self._lock:
            self._turns.append(_AgentTurn(text))
            self._trim()

    def reset(self) -> None:
        with self._lock:
            self._turns = []

    # -- one turn ------------------------------------------------------------ #

    def respond(
        self,
        audio: np.ndarray,
        *,
        sample_rate: int = SAMPLE_RATE,
        context: str | None = None,
        tool_handler: Callable[[str, str], dict] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> Reply | None:
        """Send one utterance and return the reply.

        ``context`` is live state for this turn only (the current Tyto reading).
        It is appended to the system message rather than sent as a second one,
        because several models on this endpoint reject more than one system
        message outright ("System message must be at the beginning"), and it is
        deliberately not stored in history: it describes the room right now, and
        replaying a stale copy of it on later turns would be worse than not
        having it.

        ``tool_handler(name, call_id)`` answers a tool call and its result is fed
        back for a second pass. ``cancelled()`` is polled between round trips so
        a Tyto nudge can abandon a turn already in flight.
        """
        audio_b64 = base64.b64encode(encode_wav(audio, sample_rate)).decode("ascii")
        with self._lock:
            self._turns.append(_UserTurn(audio_b64))
            self._trim()
            messages = self._build_messages(context)

        tool_calls = 0
        for _ in range(2):  # one tool round trip at most, then answer
            if cancelled is not None and cancelled():
                return None
            try:
                message = self._post(messages)
            except Exception as err:  # noqa: BLE001 - surfaced to the UI
                self._log("error", f"inkling: {err}")
                return None

            calls = message.get("tool_calls") or []
            if not calls or tool_handler is None:
                text = (message.get("content") or "").strip()
                if not text:
                    return None
                with self._lock:
                    self._turns.append(_AgentTurn(text))
                    self._trim()
                return Reply(text=text, tool_calls=tool_calls, messages=messages)

            # Tool calls are not kept in history: the answer text already carries
            # the information, and replaying them would cost tokens every turn.
            messages = messages + [{
                "role": "assistant",
                "content": None,
                "tool_calls": calls,
            }]
            for call in calls:
                name = (call.get("function") or {}).get("name", "")
                result = tool_handler(name, call.get("id", ""))
                tool_calls += 1
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.get("id", ""),
                    "content": json.dumps(result),
                })
        return None

    # -- internals ----------------------------------------------------------- #

    def _build_messages(self, context: str | None = None) -> list:
        """Render turns to wire format. Caller holds the lock.

        One system message, always. A second one is rejected by several of the
        models this endpoint serves, so per-turn context is appended to it.
        """
        system = self._instructions
        if context:
            system = f"{system}\n\n{context}"
        messages = [{"role": "system", "content": system}]
        user_indices = [i for i, t in enumerate(self._turns) if isinstance(t, _UserTurn)]
        keep_audio = set(user_indices[-AUDIO_HISTORY_TURNS:]) if AUDIO_HISTORY_TURNS else set()
        for i, turn in enumerate(self._turns):
            if isinstance(turn, _AgentTurn):
                messages.append({"role": "assistant", "content": turn.text})
            elif i in keep_audio:
                messages.append({
                    "role": "user",
                    "content": [{
                        "type": "input_audio",
                        "input_audio": {"data": turn.audio_b64, "format": "wav"},
                    }],
                })
            # Older user turns are dropped rather than re-sent as audio; keeping
            # them would undo the point of the window. The agent's own replies
            # still carry the thread of the conversation.
        return messages

    def _trim(self) -> None:
        """Caller holds the lock."""
        if len(self._turns) > MAX_HISTORY_TURNS:
            del self._turns[: len(self._turns) - MAX_HISTORY_TURNS]

    def _post(self, messages: list) -> dict:
        body = {
            "model": self._model,
            "messages": messages,
            "max_tokens": MAX_TOKENS,
            # Not optional. See the module docstring.
            "reasoning_effort": "none",
        }
        if self._tools:
            body["tools"] = self._tools
            body["tool_choice"] = "auto"
        request = urllib.request.Request(
            self._url,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "User-Agent": USER_AGENT,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
                payload = json.loads(response.read())
        except urllib.error.HTTPError as err:
            raise RuntimeError(f"HTTP {err.code}: {err.read()[:200].decode(errors='replace')}") from err
        choices = payload.get("choices") or []
        if not choices:
            raise RuntimeError(json.dumps(payload)[:200])
        return choices[0].get("message") or {}

    def _log(self, kind: str, text: str) -> None:
        if self._on_log:
            self._on_log(kind, text)
