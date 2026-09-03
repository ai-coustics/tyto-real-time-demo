"""Tests for the Flux -> LLM -> Aura-2 cascade, with no network or audio.

Two things are worth pinning here, because neither is obvious from reading the
code and both are how the demo can go wrong in a way you only hear later:

- Speculation must be *exactly* opportunistic. Deepgram guarantees the eager
  transcript matches the committed one when the turn really was over, so a match
  is the only licence to reuse the reply. Anything else has to be thrown away, or
  the agent answers a sentence the user did not finish saying.

- A voided turn must stay dead. When the Reactive layer cuts the user off, Flux
  has not heard the end of the sentence but will still report one, and answering
  that half sentence is worse than answering nothing.
"""

import threading
import time

from tyto_voice.cascade import CascadeProvider
from tyto_voice.decision import VAD_PROFILES
from tyto_voice.flux import FluxSTT, _threshold_query
from tyto_voice.llm import Reply, as_chat_tools, phonellm_backend
from tyto_voice.provider import Handlers


# --------------------------------------------------------------------------- #
# Flux: turn-event routing                                                    #
# --------------------------------------------------------------------------- #


def make_stt(**callbacks) -> FluxSTT:
    seen = callbacks.setdefault("seen", [])
    return FluxSTT(
        "key",
        profile=VAD_PROFILES["eager"],
        on_start_of_turn=lambda i, t: seen.append(("start", i, t)),
        on_interim=lambda i, t: seen.append(("interim", i, t)),
        on_eager_end_of_turn=lambda i, t: seen.append(("eager", i, t)),
        on_turn_resumed=lambda i, t: seen.append(("resumed", i, t)),
        on_end_of_turn=lambda i, t: seen.append(("end", i, t)),
    )


def turn(event, index=0, transcript=""):
    return {"type": "TurnInfo", "event": event, "turn_index": index, "transcript": transcript}


def test_turn_events_route_to_their_callbacks():
    seen = []
    stt = make_stt(seen=seen)
    stt._receive(turn("StartOfTurn", 0))
    stt._receive(turn("Update", 0, "hello th"))
    stt._receive(turn("EagerEndOfTurn", 0, "hello there"))
    stt._receive(turn("EndOfTurn", 0, "hello there"))
    assert [s[0] for s in seen] == ["start", "interim", "eager", "end"]
    assert seen[-1] == ("end", 0, "hello there")


def test_discarded_turn_is_never_reported():
    seen = []
    stt = make_stt(seen=seen)
    stt._receive(turn("StartOfTurn", 0))
    stt.discard_turn()
    stt._receive(turn("EagerEndOfTurn", 0, "half a sent"))
    stt._receive(turn("EndOfTurn", 0, "half a sentence"))
    assert [s[0] for s in seen] == ["start"]


def test_discarding_only_kills_the_turn_in_progress():
    seen = []
    stt = make_stt(seen=seen)
    stt._receive(turn("StartOfTurn", 0))
    stt.discard_turn()
    stt._receive(turn("EndOfTurn", 0, "cut off"))
    stt._receive(turn("StartOfTurn", 1))
    stt._receive(turn("EndOfTurn", 1, "the next thing"))
    assert seen[-1] == ("end", 1, "the next thing")


def test_patient_profile_drops_eager_events_locally():
    # Flux has no "off" value for eager_eot_threshold, and a rejected Configure
    # silently keeps the old one, so this gate has to hold on our side.
    seen = []
    stt = make_stt(seen=seen)
    stt._receive(turn("EagerEndOfTurn", 0, "speculate on this"))
    assert [s[0] for s in seen] == ["eager"]

    stt.configure(VAD_PROFILES["patient"])
    stt._receive(turn("EagerEndOfTurn", 1, "but not on this"))
    stt._receive(turn("EndOfTurn", 1, "but not on this"))
    assert [s[0] for s in seen] == ["eager", "end"]

    stt.configure(VAD_PROFILES["eager"])
    stt._receive(turn("EagerEndOfTurn", 2, "back on"))
    assert [s[0] for s in seen] == ["eager", "end", "eager"]


def test_configure_never_sends_an_out_of_range_eager_threshold():
    sent = []
    stt = make_stt()
    stt._send = lambda obj: sent.append(obj)
    stt.configure(VAD_PROFILES["patient"])
    thresholds = sent[0]["thresholds"]
    # Omitted, not zeroed: 0 is outside Deepgram's 0.3-0.9 range and would be
    # rejected, leaving the eager profile's value quietly in force.
    assert "eager_eot_threshold" not in thresholds
    assert thresholds["eot_threshold"] == VAD_PROFILES["patient"]["eot_threshold"]


def test_eager_threshold_is_omitted_when_the_profile_has_none():
    eager = dict(_threshold_query(VAD_PROFILES["eager"]))
    patient = dict(_threshold_query(VAD_PROFILES["patient"]))
    assert "eager_eot_threshold" in eager
    assert "eager_eot_threshold" not in patient


# --------------------------------------------------------------------------- #
# Cascade: speculation                                                        #
# --------------------------------------------------------------------------- #


class FakeLLM:
    def __init__(self, reply="sure thing"):
        self.reply = reply
        self.asked = []
        self.committed = []
        self.lines = []
        self.gate = threading.Event()
        self.gate.set()

    def respond(self, text, tool_handler=None, cancelled=None):
        self.asked.append(text)
        self.gate.wait(timeout=2.0)
        if cancelled is not None and cancelled():
            return None
        return Reply(text=self.reply)

    def commit(self, user_text, agent_text):
        self.committed.append((user_text, agent_text))

    def add_agent_line(self, text):
        self.lines.append(text)

    def set_instructions(self, text):
        pass


class FakeTTS:
    def __init__(self):
        self.spoken = []
        self.cleared = 0
        self.closed = threading.Event()
        self.said = threading.Event()

    def speak(self, text):
        self.spoken.append(text)
        self.said.set()
        return True

    def clear(self):
        self.cleared += 1

    def close(self):
        pass


class FakeSTT:
    def __init__(self):
        self.discarded = 0
        self.configured = []

    def discard_turn(self):
        self.discarded += 1

    def configure(self, profile):
        self.configured.append(profile)

    def send_audio(self, mono):
        pass

    def close(self):
        pass


def build():
    provider = CascadeProvider(
        Handlers(),
        backend=phonellm_backend("http://example.invalid", "wk-x.ws-y"),
        deepgram_key="key",
        instructions="be brief",
        audio_out=lambda pcm: None,
    )
    provider.llm, provider.tts, provider.stt = FakeLLM(), FakeTTS(), FakeSTT()
    return provider


def wait_for_speech(provider, timeout=2.0):
    assert provider.tts.said.wait(timeout=timeout), "the agent never spoke"


def wait_for_ask(provider, n, timeout=2.0):
    """Block until the model has been asked n times.

    The speculation runs on its own thread, so without this the order of
    ``asked`` depends on which thread wins the race to append, and the test is
    flaky rather than wrong.
    """
    deadline = time.monotonic() + timeout
    while len(provider.llm.asked) < n and time.monotonic() < deadline:
        time.sleep(0.005)
    assert len(provider.llm.asked) >= n, f"only asked {len(provider.llm.asked)} times"


def test_matching_speculation_is_reused_so_the_model_is_asked_once():
    p = build()
    p._on_eager_end_of_turn(0, "what is tyto")
    p._on_end_of_turn(0, "what is tyto")
    wait_for_speech(p)
    assert p.llm.asked == ["what is tyto"]
    assert p.tts.spoken == ["sure thing"]
    assert p.llm.committed == [("what is tyto", "sure thing")]


def test_speculation_on_a_different_transcript_is_thrown_away():
    p = build()
    p.llm.gate.clear()  # hold the speculation open so it is still in flight
    p._on_eager_end_of_turn(0, "what is")
    wait_for_ask(p, 1)
    p.llm.gate.set()
    p._on_end_of_turn(0, "what is tyto exactly")
    wait_for_speech(p)
    # Asked twice: once on the guess, then again on what the user actually said.
    assert p.llm.asked == ["what is", "what is tyto exactly"]
    assert p.llm.committed == [("what is tyto exactly", "sure thing")]


def test_turn_resumed_cancels_the_speculation():
    p = build()
    p.llm.gate.clear()
    p._on_eager_end_of_turn(0, "hold on")
    wait_for_ask(p, 1)
    p._on_turn_resumed(0, "hold on")
    p.llm.gate.set()
    p._on_end_of_turn(0, "hold on i mean something else")
    wait_for_speech(p)
    assert p.llm.committed == [("hold on i mean something else", "sure thing")]


def test_interrupt_cancels_the_speculation_and_voids_the_input():
    p = build()
    p._on_eager_end_of_turn(0, "something")
    p.interrupt(clear_input=True)
    assert p._spec is None
    assert p.tts.cleared == 1
    assert p.stt.discarded == 1


def test_nudge_is_spoken_directly_with_no_model_round_trip():
    p = build()
    p.nudge("Could you turn the TV down?")
    assert p.tts.spoken == ["Could you turn the TV down?"]
    assert p.llm.asked == []
    assert p.llm.lines == ["Could you turn the TV down?"]


def test_listen_gate_stops_forwarding_audio_and_voids_the_turn():
    p = build()
    p.set_turn_detection(None)
    assert p.stt.discarded == 1
    p.send_audio([0.0] * 16)  # dropped: no configure, no audio
    assert p.stt.configured == []
    p.set_turn_detection(VAD_PROFILES["patient"])
    assert p.stt.configured == [VAD_PROFILES["patient"]]


# --------------------------------------------------------------------------- #
# LLM: the tool shape                                                    #
# --------------------------------------------------------------------------- #


def test_flat_repo_tools_become_chat_completions_tools():
    from tyto_voice.controller import CHECK_AUDIO_QUALITY_TOOL

    tools = as_chat_tools([CHECK_AUDIO_QUALITY_TOOL])
    assert tools[0]["type"] == "function"
    assert tools[0]["function"]["name"] == "check_audio_quality"
    assert tools[0]["function"]["parameters"]["type"] == "object"
    # Already-nested tools are passed through untouched.
    assert as_chat_tools(tools) == tools


# --------------------------------------------------------------------------- #
# LLM backends: the body shapes are mutually exclusive                        #
# --------------------------------------------------------------------------- #


def test_backend_body_shapes_do_not_leak_into_each_other():
    """Each parameter PhoneLLM requires is one gpt-5-mini rejects with a 400.

    Measured against both live APIs: gpt-5-mini refuses max_tokens ("use
    max_completion_tokens"), refuses temperature=0 ("only the default (1) is
    supported"), and refuses chat_template_kwargs ("unknown parameter"). Sending
    one body shape to the other backend fails every single turn, so this is
    pinned rather than left to a reviewer to notice.
    """
    from tyto_voice.llm import openai_backend, phonellm_backend

    phone = phonellm_backend("https://ep.example", "wk-x.ws-y")
    assert phone.token_field == "max_tokens"
    assert phone.body["temperature"] == 0
    assert phone.body["chat_template_kwargs"] == {"enable_thinking": False}
    assert phone.cold_starts is True  # Modal Auto Endpoints scale to zero

    gpt = openai_backend("sk-x")
    assert gpt.token_field == "max_completion_tokens"
    assert "temperature" not in gpt.body
    assert "chat_template_kwargs" not in gpt.body
    assert "max_tokens" not in gpt.body
    # Both required to keep a reasoning model inside a turn gap: 1.09 s with
    # them, 2.75 s without.
    assert gpt.body["reasoning_effort"] == "minimal"
    assert gpt.body["verbosity"] == "low"
    assert gpt.cold_starts is False  # a hosted API is always up


def test_backend_from_env_needs_the_keys_for_the_backend_it_picks(monkeypatch):
    from tyto_voice.llm import backend_from_env
    import pytest

    for name in ("LLM_BACKEND", "OPENAI_API_KEY", "MODAL_ENDPOINT_URL", "MODAL_API_KEY"):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(SystemExit):
        backend_from_env()  # default backend, no OPENAI_API_KEY

    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    assert backend_from_env().name == "gpt-5-mini"

    monkeypatch.setenv("LLM_BACKEND", "phonellm")
    with pytest.raises(SystemExit):
        backend_from_env()  # phonellm chosen, but no Modal keys

    monkeypatch.setenv("MODAL_ENDPOINT_URL", "https://ep.example")
    monkeypatch.setenv("MODAL_API_KEY", "wk-x.ws-y")
    assert backend_from_env().name == "phonellm"


def test_request_body_is_built_from_the_backend(monkeypatch):
    """The one place the two shapes could still cross: _post."""
    from tyto_voice.llm import LLMClient, openai_backend, phonellm_backend

    seen = {}

    def fake_post(self, messages):
        backend = self.backend
        seen[backend.name] = {
            "model": backend.model,
            backend.token_field: backend.token_budget,
            **backend.body,
        }
        return {"content": "ok"}

    monkeypatch.setattr(LLMClient, "_post", fake_post)
    for backend in (phonellm_backend("https://ep.example", "k"), openai_backend("sk-x")):
        LLMClient(backend, instructions="hi").respond("hello")

    assert "max_tokens" in seen["phonellm"] and "max_completion_tokens" not in seen["phonellm"]
    assert "max_completion_tokens" in seen["gpt-5-mini"] and "max_tokens" not in seen["gpt-5-mini"]
