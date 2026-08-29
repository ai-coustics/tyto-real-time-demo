"""Pipecat PhoneLLM on Modal, the brain of the cascade.

PhoneLLM Alpha 1 (``pipecat-ai/phonellm-alpha-1``) is an open-weights model from
the Pipecat team, fine-tuned for phone voice agents: short replies, accurate tool
calling, and a time-to-first-token low enough that a cascade can feel like a
speech-to-speech model. It is a mixture of experts, 32B total with 3.5B active,
which is why a 30B-class model can answer inside a turn gap at all.

Modal serves it as an Auto Endpoint: an ordinary OpenAI-compatible server, so
the wire format here is plain ``/v1/chat/completions``. Four things come from the
model card and are load-bearing:

- The model id must be exactly ``pipecat-ai/phonellm-alpha-1``.
- ``temperature`` must be 0. This is the recommended inference setting and it
  also makes the speculative path below sound sane: the reply the agent commits
  to is the same one it would have produced without speculating.
- Thinking must be off, via ``chat_template_kwargs={"enable_thinking": false}``.
  PhoneLLM was trained to call tools correctly *without* thinking, so leaving it
  on buys nothing and costs the whole reply in latency.
- ``max_tokens`` is a runaway guard, not a length control. Replies are kept short
  by instruction; the cap only stops a loop from being spoken out loud.

Replies are fetched whole rather than streamed. That reads like the wrong call
for a voice agent, and it would be, except that the turn is speculated on: by
the time Deepgram Flux commits the end of the turn the reply is usually already
in hand (see ``cascade.py``). Streaming would win back time only on the turns
speculation missed, at the cost of a partial-sentence chunker in front of the
text-to-speech socket.

History is plain text on both sides, so it is cheap to carry, and
``MAX_HISTORY_TURNS`` bounds it.

Only the standard library is used for transport, so this adds no dependency.
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable

MODEL = os.environ.get("PHONELLM_MODEL", "pipecat-ai/phonellm-alpha-1")

# Spoken replies are short by instruction; this is only a runaway guard.
MAX_TOKENS = 120
REQUEST_TIMEOUT = 30.0

# Turns kept in history (a turn is one user utterance or one agent reply).
MAX_HISTORY_TURNS = 16

# Modal Auto Endpoints scale to zero. The first request after a quiet period
# answers 503 while a container starts, which for a 30B model is minutes, not
# seconds. Retrying inside a turn is pointless (the user is waiting), so a turn
# gives up quickly; ``wait_until_ready`` is the one place that waits it out.
COLD_START_STATUS = (503, 502, 504)
READY_POLL_SECONDS = 5.0

USER_AGENT = "tyto-voice/0.1 (+https://github.com/ai-coustics)"


@dataclass
class Reply:
    text: str
    tool_calls: int = 0


class PhoneLLMClient:
    """One conversation. ``respond`` is blocking and expects a worker thread.

    ``respond`` deliberately does **not** touch history: a reply may be
    speculative and thrown away when the user turns out not to have finished
    talking. The caller commits with :meth:`commit` once the turn is real.
    """

    def __init__(
        self,
        endpoint_url: str,
        api_key: str,
        *,
        instructions: str,
        tools: list | None = None,
        model: str = MODEL,
        on_log: Callable[[str, str], None] | None = None,
    ):
        self._api_key = api_key
        self._model = model
        self._url = endpoint_url.rstrip("/") + "/v1/chat/completions"
        self._models_url = endpoint_url.rstrip("/") + "/v1/models"
        self._tools = _as_chat_tools(tools)
        self._on_log = on_log

        self._lock = threading.Lock()
        self._instructions = instructions
        self._turns: list[dict] = []

    # -- conversation state -------------------------------------------------- #

    def set_instructions(self, text: str) -> None:
        """Layer 1 - Aware. Swapped in whole, applied on the next turn."""
        with self._lock:
            self._instructions = text

    def commit(self, user_text: str, agent_text: str) -> None:
        """Record a completed exchange. Only committed turns reach the prompt."""
        with self._lock:
            if user_text:
                self._turns.append({"role": "user", "content": user_text})
            if agent_text:
                self._turns.append({"role": "assistant", "content": agent_text})
            self._trim()

    def add_agent_line(self, text: str) -> None:
        """Record something the agent said outside a normal turn (a nudge, the
        opening greeting) so it knows it said it."""
        with self._lock:
            self._turns.append({"role": "assistant", "content": text})
            self._trim()

    def reset(self) -> None:
        with self._lock:
            self._turns = []

    # -- readiness ----------------------------------------------------------- #

    def wait_until_ready(self, timeout: float = 600.0) -> bool:
        """Poll ``/v1/models`` until the endpoint answers, or give up.

        Worth doing once at connect: a cold Modal container takes minutes to load
        a 30B model, and a demo that fails the first turn instead of waiting for
        it looks broken.
        """
        deadline = time.monotonic() + timeout
        announced = False
        while time.monotonic() < deadline:
            request = urllib.request.Request(self._models_url, headers=self._headers())
            try:
                with urllib.request.urlopen(request, timeout=20.0):
                    return True
            except urllib.error.HTTPError as err:
                if err.code not in COLD_START_STATUS:
                    self._log("error", f"phonellm: HTTP {err.code}")
                    return False
            except Exception:  # noqa: BLE001 - network flake during a cold start
                pass
            if not announced:
                announced = True
                self._log("llm.cold", "waking the PhoneLLM endpoint, this can take minutes")
            time.sleep(READY_POLL_SECONDS)
        return False

    # -- one turn ------------------------------------------------------------ #

    def respond(
        self,
        text: str,
        *,
        tool_handler: Callable[[str, str], dict] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> Reply | None:
        """Answer one user utterance. Returns None if it was abandoned or failed.

        ``tool_handler(name, call_id)`` answers a tool call and its result is fed
        back for a second pass. PhoneLLM is trained for exactly this and is the
        reason ``check_audio_quality`` is still a real tool here rather than a
        line of context pasted into every prompt.

        ``cancelled()`` is polled around each round trip so a Tyto nudge, or the
        user carrying on talking, can abandon a turn already in flight.
        """
        with self._lock:
            messages = [{"role": "system", "content": self._instructions}] + list(self._turns)
        messages.append({"role": "user", "content": text})

        tool_calls = 0
        for _ in range(2):  # one tool round trip at most, then answer
            if cancelled is not None and cancelled():
                return None
            try:
                message = self._post(messages)
            except Exception as err:  # noqa: BLE001 - surfaced to the UI
                self._log("error", f"phonellm: {err}")
                return None

            calls = message.get("tool_calls") or []
            if not calls or tool_handler is None:
                reply = (message.get("content") or "").strip()
                return Reply(text=reply, tool_calls=tool_calls) if reply else None

            # Tool calls are not kept in history: the answer text already carries
            # the information, and replaying them would cost tokens every turn.
            messages = messages + [{"role": "assistant", "content": None, "tool_calls": calls}]
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

    def _trim(self) -> None:
        """Caller holds the lock."""
        if len(self._turns) > MAX_HISTORY_TURNS:
            del self._turns[: len(self._turns) - MAX_HISTORY_TURNS]

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        }

    def _post(self, messages: list) -> dict:
        body = {
            "model": self._model,
            "messages": messages,
            "max_tokens": MAX_TOKENS,
            # Both required by the model card. See the module docstring.
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if self._tools:
            body["tools"] = self._tools
            body["tool_choice"] = "auto"
        request = urllib.request.Request(
            self._url,
            data=json.dumps(body).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
                payload = json.loads(response.read())
        except urllib.error.HTTPError as err:
            detail = err.read()[:200].decode(errors="replace")
            if err.code in COLD_START_STATUS:
                raise RuntimeError(f"HTTP {err.code}: endpoint still waking up") from err
            raise RuntimeError(f"HTTP {err.code}: {detail}") from err
        choices = payload.get("choices") or []
        if not choices:
            raise RuntimeError(json.dumps(payload)[:200])
        return choices[0].get("message") or {}

    def _log(self, kind: str, text: str) -> None:
        if self._on_log:
            self._on_log(kind, text)


def _as_chat_tools(tools: list | None) -> list:
    """Accept the repo's flat tool dicts and emit chat-completions shape.

    The Realtime API takes ``{"type": "function", "name": ..., "parameters":
    ...}``; chat completions nests that under a ``function`` key. Converting here
    keeps ``CHECK_AUDIO_QUALITY_TOOL`` as the single definition in the repo.
    """
    wrapped = []
    for tool in tools or []:
        if "function" in tool:
            wrapped.append(tool)
            continue
        wrapped.append({
            "type": "function",
            "function": {
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
            },
        })
    return wrapped
