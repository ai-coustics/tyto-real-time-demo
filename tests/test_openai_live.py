"""GPT-Live provider event mapping with a captured transport and a fake clock: no network, no audio device.

Covers what was verified live: the strict session.start shape, the continuous
audio track (silence included), speech segmentation by RMS, the interrupt hold
that makes the agent stop at once, and the append-based control events.
"""

import base64

import numpy as np

from tyto_voice.decision import VAD_PROFILES
from tyto_voice.openai_live import (
    CLEAN_NOTE,
    HOLD_RELEASE_GAP_S,
    LIVE_ADDENDUM,
    NUDGE_TIMEOUT_S,
    OPEN_NOTE,
    SILENT,
    SPEECH_HANGOVER_S,
    TURN_NOTES,
    OpenAILiveProvider,
)
from tyto_voice.provider import Handlers

BASE = "You are the host of the demo."
NUDGE = "Sorry, there is a lot of background noise. Could you move somewhere quieter?"


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


def build():
    events, sent, audio, clock = [], [], [], Clock()
    h = Handlers(
        on_ready=lambda: events.append(("ready",)),
        on_agent_speaking=lambda active, nudge=False, cancelled=False: events.append(("speaking", active, nudge, cancelled)),
        on_user_transcript=lambda t, f: events.append(("user", t, f)),
        on_agent_transcript=lambda t, f: events.append(("agent", t, f)),
        on_tool_call=lambda n, c: events.append(("tool", n, c)),
    )
    p = OpenAILiveProvider(
        h, api_key="sk-test", instructions=BASE, audio_out=audio.append,
        audio_done=lambda: events.append(("audio_done",)), audio_flush=lambda: events.append(("flush",)), clock=clock,
    )
    p._send = sent.append  # capture control events instead of writing to the socket
    p._ws = object()  # "connected"
    return p, events, sent, audio, clock


def delta(rms=0.1, ms=100) -> dict:
    """One server audio chunk: a sine at the given RMS, or digital silence."""
    n = 24 * ms
    x = (np.sin(np.arange(n) * 0.3) * rms * 1.414 * 32767).astype("<i2") if rms else np.zeros(n, "<i2")
    return {"type": "session.output_audio.delta", "delta": base64.b64encode(x.tobytes()).decode()}


def feed(p, clock, rms, seconds):
    """Stream 100 ms chunks in real time, ticking the watchdog like the loop does."""
    for _ in range(int(round(seconds * 10))):
        p._receive(delta(rms))
        clock.advance(0.1)
        p._tick(clock())


def speaking_events(events):
    return [e for e in events if e[0] == "speaking"]


def test_session_start_is_the_strict_config_with_client_delegation():
    p, *_ = build()
    s = p.session_start()["session"]
    assert s["model"] == "gpt-live-1"
    assert s["instructions"].startswith(BASE) and s["instructions"].endswith(LIVE_ADDENDUM)
    assert s["audio"] == {"format": {"type": "audio/pcm", "rate": 24000}, "output": {"voice": "marin"}}
    assert s["delegation"] == {"type": "client"}
    assert set(s) == {"model", "instructions", "audio", "delegation"}  # unknown fields are rejected


def test_started_reports_ready_and_the_opener_is_a_thinking_note():
    p, events, sent, *_ = build()
    p._receive({"type": "session.started", "session": {"id": "live_1"}})
    assert ("ready",) in events
    p.request_response()
    assert sent[-1] == {"type": "session.thinking.append", "delegation_id": None, "content": OPEN_NOTE}


def test_silence_is_dropped_and_speech_segments_drive_the_lifecycle():
    p, events, sent, audio, clock = build()
    feed(p, clock, 0.0, 0.5)  # the track runs even when nobody talks
    assert audio == [] and events == []
    feed(p, clock, 0.1, 0.5)
    assert speaking_events(events) == [("speaking", True, False, False)]
    assert len(audio) == 5
    feed(p, clock, 0.0, 0.2)  # a pause inside a sentence keeps playback continuous
    assert len(audio) == 7
    feed(p, clock, 0.0, SPEECH_HANGOVER_S + 0.1)
    assert ("audio_done",) in events and ("speaking", False, False, False) in events
    assert 7 <= len(audio) <= 7 + int(SPEECH_HANGOVER_S * 10) + 1


def test_interrupt_holds_the_tail_and_lets_the_nudge_through_after_the_pause():
    p, events, sent, audio, clock = build()
    feed(p, clock, 0.1, 0.5)  # the agent is mid-sentence
    events.clear(); audio.clear()
    p.interrupt(clear_input=True)
    p.nudge(NUDGE)
    assert ("flush",) in events and ("speaking", False, False, True) in events
    assert sent[-1] == {"type": "session.commentary.append", "delegation_id": None, "content": NUDGE}
    feed(p, clock, 0.1, 1.5)  # the model finishes its sentence anyway: the listener never hears it
    assert audio == [] and not any(e[1] for e in speaking_events(events))
    feed(p, clock, 0.0, HOLD_RELEASE_GAP_S + 0.1)  # it pauses before speaking the commentary
    feed(p, clock, 0.1, 0.3)  # the nudge: heard from its first chunk, flagged as the nudge
    assert len(audio) == 3 and ("speaking", True, True, False) in events
    feed(p, clock, 0.0, SPEECH_HANGOVER_S + 0.1)
    assert ("speaking", False, True, False) in events and ("audio_done",) in events


def test_interrupt_while_silent_lets_the_nudge_through_immediately():
    p, events, sent, audio, clock = build()
    p.interrupt()
    p.nudge(NUDGE)
    feed(p, clock, 0.1, 0.2)
    assert len(audio) == 2 and ("speaking", True, True, False) in events


def test_hold_gives_up_after_the_cap_so_the_nudge_is_never_lost():
    p, events, sent, audio, clock = build()
    feed(p, clock, 0.1, 0.3)
    p.interrupt(); p.nudge(NUDGE)
    feed(p, clock, 0.1, 6.5)  # the model never pauses
    assert audio and ("speaking", True, True, False) in events


def test_transcript_of_the_unheard_tail_is_hidden_but_the_nudge_releases_the_hold():
    p, events, sent, audio, clock = build()
    feed(p, clock, 0.1, 0.3)
    audio.clear()
    p.interrupt(); p.nudge(NUDGE)
    p._receive({"type": "session.output_transcript.delta", "delta": " swivel their heads"})
    assert not any(e[0] == "agent" for e in events)
    p._receive({"type": "session.output_transcript.delta", "delta": " background"})
    assert ("agent", " background", False) in events
    feed(p, clock, 0.1, 0.2)
    assert len(audio) == 2


def test_a_nudge_the_model_never_speaks_releases_the_state_machine():
    p, events, sent, audio, clock = build()
    p.interrupt(); p.nudge(NUDGE)
    clock.advance(NUDGE_TIMEOUT_S + 0.1)
    p._tick(clock())
    assert speaking_events(events)[-2:] == [("speaking", True, True, False), ("speaking", False, True, True)]


def test_set_instructions_appends_only_the_room_note():
    p, events, sent, *_ = build()
    p.set_instructions(BASE + "\n\nAudio note: degraded input, loud background noise.")
    assert sent[-1] == {"type": "session.instructions.append", "delegation_id": None, "content": "Audio note: degraded input, loud background noise." + SILENT}
    p.set_instructions(BASE)
    assert sent[-1]["content"] == CLEAN_NOTE


def test_turn_detection_becomes_one_turn_taking_note_per_profile():
    p, events, sent, *_ = build()
    p.set_turn_detection(VAD_PROFILES["patient"])
    assert sent[-1]["type"] == "session.instructions.append" and sent[-1]["content"] == TURN_NOTES["patient"]
    n = len(sent)
    p.set_turn_detection(VAD_PROFILES["patient"])  # unchanged: nothing appended
    p.set_turn_detection(None)  # the listen gate: nothing to gate on a full-duplex model
    assert len(sent) == n
    p.set_turn_detection(VAD_PROFILES["eager"])
    assert sent[-1]["content"] == TURN_NOTES["eager"]


def test_client_delegation_maps_to_the_audio_quality_tool_and_back():
    p, events, sent, *_ = build()
    p._receive({"type": "session.delegation.created", "delegation": {"id": "item_1", "type": "delegation", "target": "client"}})
    assert ("tool", "check_audio_quality", "item_1") in events
    p.send_tool_result("item_1", {"summary": "Audio is degraded. Biggest issue: Noise is high at 0.62.", "tyto_score": 0.55})
    m = sent[-1]
    assert m["type"] == "session.commentary.append" and m["delegation_id"] == "item_1"
    assert m["content"].startswith("Audio is degraded.") and "0.55" in m["content"]


def test_transcript_fragments_become_lines_on_punctuation_or_idle():
    p, events, sent, audio, clock = build()
    p._receive({"type": "session.input_transcript.delta", "delta": "Hello"})
    p._receive({"type": "session.input_transcript.delta", "delta": " there."})
    assert ("user", "Hello", False) in events and ("user", "Hello there.", True) in events
    p._receive({"type": "session.output_transcript.delta", "delta": " Hey"})
    clock.advance(1.3)
    p._tick(clock())
    assert ("agent", "Hey", True) in events


def test_muted_mic_sends_silence_so_the_input_track_stays_continuous():
    p, events, sent, *_ = build()
    p.set_mic_enabled(False)
    p.send_audio(np.ones(480, np.float32))
    m = sent[-1]
    pcm = base64.b64decode(m["audio"])
    assert m["type"] == "session.input_audio.append" and len(pcm) == 960 and set(pcm) == {0}
    p.set_mic_enabled(True)
    p.send_audio(np.ones(480, np.float32) * 0.5)
    assert set(base64.b64decode(sent[-1]["audio"])) != {0}


def test_errors_and_close_are_logged_and_close_ends_the_session():
    logs = []
    p, *_ = build()
    p._on_log = lambda k, t: logs.append((k, t))
    p._receive({"type": "error", "error": {"code": "invalid_audio", "message": "odd bytes"}})
    p._receive({"type": "session.closed", "reason": "close_requested", "usage": {"seconds": 12}})
    assert ("error", "invalid_audio odd bytes") in logs and p.closed.is_set()
