"""The web demo's abuse guard: concurrent cap, per-IP cap, hourly start limit."""

import sys
from pathlib import Path

import pytest

pytest.importorskip("aiohttp")
sys.path.insert(0, str(Path(__file__).parent.parent / "examples" / "web"))
from server import SessionGate, client_ip  # noqa: E402


def make(**kw):
    t = [0.0]
    limits = {"max_sessions": 3, "max_starts_per_hour": 100, "max_per_ip": 2, "max_starts_per_ip_hour": 4, **kw}
    return SessionGate(clock=lambda: t[0], **limits), t


def test_caps_total_sessions():
    gate, _ = make()
    assert [gate.admit(ip) for ip in ("a", "b", "c")] == [None, None, None]
    assert "busy" in gate.admit("d")
    gate.release("a")
    assert gate.admit("d") is None


def test_caps_sessions_per_ip():
    gate, _ = make()
    assert gate.admit("a") is None and gate.admit("a") is None
    assert "already" in gate.admit("a")


def test_hourly_start_limit_rolls_off():
    gate, t = make(max_sessions=10, max_per_ip=10)
    for _ in range(4):
        assert gate.admit("a") is None
        gate.release("a")
    assert "hour" in gate.admit("a")
    t[0] = 3601
    assert gate.admit("a") is None


def test_unknown_visitors_share_only_the_global_limits():
    # Behind Modal every visitor arrives from the proxy, so no per-visitor cap may apply.
    gate, t = make(max_starts_per_hour=5)
    assert [gate.admit(None) for _ in range(3)] == [None, None, None]
    assert "busy" in gate.admit(None)
    for _ in range(3):
        gate.release(None)
    assert gate.admit(None) is None and gate.admit(None) is None
    gate.release(None), gate.release(None)
    assert "hourly" in gate.admit(None)
    t[0] = 3601
    assert gate.admit(None) is None


class FakeRequest:
    def __init__(self, remote, headers=None):
        self.remote, self.headers = remote, headers or {}


def test_client_ip_ignores_private_proxy_hops():
    assert client_ip(FakeRequest("172.20.1.210")) is None
    assert client_ip(FakeRequest("172.20.1.210", {"X-Forwarded-For": "8.8.8.8, 10.0.0.1"})) == "8.8.8.8"
    assert client_ip(FakeRequest("1.1.1.1")) == "1.1.1.1"
