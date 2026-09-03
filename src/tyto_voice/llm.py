"""The agent's brain, behind one OpenAI-compatible chat-completions client.

Two backends are supported and they are not interchangeable by URL alone, which
is the whole reason this file has a ``Backend`` type instead of three settings.
Every request-body parameter PhoneLLM *requires* is one gpt-5-mini *rejects*:

    max_tokens            gpt-5-mini: 400, use max_completion_tokens
    temperature: 0        gpt-5-mini: 400, only the default 1 is supported
    chat_template_kwargs  gpt-5-mini: 400, unknown parameter

So a backend owns its own body shape, and swapping is one environment variable.

    LLM_BACKEND=phonellm     Pipecat PhoneLLM on a Modal Auto Endpoint
    LLM_BACKEND=gpt-5-mini   OpenAI (the default)

**PhoneLLM** (``pipecat-ai/phonellm-alpha-1``) is open weights, fine-tuned for
phone voice agents, and served from a Modal Auto Endpoint. It is the reason this
branch exists. Its one operational catch is that Modal Auto Endpoints scale to
zero: the first request after a quiet spell answers 503 while a 30B model loads,
which measured around 100 seconds here, and during that window the demo greets
you and then cannot answer anything. That is a deployment setting, not the
model, and it is fixed by keeping a container warm.

**gpt-5-mini** is a hosted reasoning model, so the settings that matter are the
ones that stop it thinking: ``reasoning_effort="minimal"`` and
``verbosity="low"`` measured 1.09 s against 2.75 s with both left at their
defaults, with zero reasoning tokens. A voice agent cannot afford the default.

Replies are fetched whole rather than streamed, because the turn is speculated
on: by the time Deepgram Flux commits the end of the turn the reply is usually
already in hand (see ``cascade.py``).

Only the standard library is used for transport, so this adds no dependency.
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable

# Spoken replies are short by instruction; the cap is only a runaway guard. It
# is generous for gpt-5-mini because reasoning tokens are drawn from the same
# budget, and a cap that a model spends entirely on thinking returns an empty
# reply that a voice agent cannot use.
MAX_TOKENS = 120
MAX_COMPLETION_TOKENS = 400
REQUEST_TIMEOUT = 30.0

# Turns kept in history (a turn is one user utterance or one agent reply).
MAX_HISTORY_TURNS = 16

# Statuses that mean "not up yet" rather than "wrong". Only PhoneLLM cold starts
# produce these, but retrying is harmless either way.
COLD_START_STATUS = (503, 502, 504)
READY_POLL_SECONDS = 5.0

USER_AGENT = "tyto-voice/0.1 (+https://github.com/ai-coustics)"


@dataclass(frozen=True)
class Backend:
    """Everything that differs between one chat-completions server and another."""

    name: str
    base_url: str
    model: str
    api_key: str
    # "max_tokens" or "max_completion_tokens"; see the module docstring.
    token_field: str
    token_budget: int
    # Merged into every request body verbatim.
    body: dict = field(default_factory=dict)
    # Whether the endpoint scales to zero and is worth waiting for on connect.
    cold_starts: bool = False


def phonellm_backend(endpoint_url: str, api_key: str, model: str | None = None) -> Backend:
    return Backend(
        name="phonellm",
        base_url=endpoint_url,
        model=model or os.environ.get("PHONELLM_MODEL", "pipecat-ai/phonellm-alpha-1"),
        api_key=api_key,
        token_field="max_tokens",
        token_budget=MAX_TOKENS,
        # Both from the model card and both load-bearing: PhoneLLM is trained to
        # call tools correctly without thinking, so leaving thinking on costs the
        # whole reply in latency and buys nothing.
        body={"temperature": 0, "chat_template_kwargs": {"enable_thinking": False}},
        cold_starts=True,
    )


def openai_backend(api_key: str, model: str = "gpt-5-mini") -> Backend:
    return Backend(
        name=model,
        base_url="https://api.openai.com",
        model=model,
        api_key=api_key,
        token_field="max_completion_tokens",
        token_budget=MAX_COMPLETION_TOKENS,
        # No temperature: this model accepts only its default, and sending 0 is a
        # 400. The two that are here are what keep it inside a turn gap.
        body={"reasoning_effort": "minimal", "verbosity": "low"},
    )


def backend_from_env(on_log: Callable[[str, str], None] | None = None) -> Backend:
    """Pick the backend from the environment. Raises if its keys are missing."""
    choice = os.environ.get("LLM_BACKEND", "gpt-5-mini").strip().lower()
    if choice == "phonellm":
        url, key = os.environ.get("MODAL_ENDPOINT_URL"), os.environ.get("MODAL_API_KEY")
        if not url or not key:
            raise SystemExit("LLM_BACKEND=phonellm needs MODAL_ENDPOINT_URL and MODAL_API_KEY.")
        return phonellm_backend(url, key)
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit(f"LLM_BACKEND={choice} needs OPENAI_API_KEY.")
    return openai_backend(key, model=choice)


@dataclass
class Reply:
    text: str
    tool_calls: int = 0


class LLMClient:
    """One conversation. ``respond`` is blocking and expects a worker thread.

    ``respond`` deliberately does **not** touch history: a reply may be
    speculative and thrown away when the user turns out not to have finished
    talking. The caller commits with :meth:`commit` once the turn is real.
    """

    def __init__(
        self,
        backend: Backend,
        *,
        instructions: str,
        tools: list | None = None,
        on_log: Callable[[str, str], None] | None = None,
    ):
        self.backend = backend
        self._url = backend.base_url.rstrip("/") + "/v1/chat/completions"
        self._models_url = backend.base_url.rstrip("/") + "/v1/models"
        self._tools = as_chat_tools(tools)
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

        Only worth doing for a backend that scales to zero. A hosted API is
        always up, so this returns immediately for one.
        """
        if not self.backend.cold_starts:
            return True
        deadline = time.monotonic() + timeout
        announced = False
        while time.monotonic() < deadline:
            request = urllib.request.Request(self._models_url, headers=self._headers())
            try:
                with urllib.request.urlopen(request, timeout=20.0):
                    return True
            except urllib.error.HTTPError as err:
                if err.code not in COLD_START_STATUS:
                    self._log("error", f"{self.backend.name}: HTTP {err.code}")
                    return False
            except Exception:  # noqa: BLE001 - network flake during a cold start
                pass
            if not announced:
                announced = True
                self._log("llm.cold", f"waking {self.backend.name}, this can take minutes")
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
        """Answer one user utterance. Returns None if abandoned or failed.

        ``tool_handler(name, call_id)`` answers a tool call and its result is fed
        back for a second pass, which is how ``check_audio_quality`` works.

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
                self._log("error", f"{self.backend.name}: {err}")
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
            "Authorization": f"Bearer {self.backend.api_key}",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        }

    def _post(self, messages: list) -> dict:
        backend = self.backend
        body = {
            "model": backend.model,
            "messages": messages,
            backend.token_field: backend.token_budget,
            **backend.body,
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


def as_chat_tools(tools: list | None) -> list:
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
