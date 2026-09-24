"""The web demo's abuse guard: concurrent cap, per-IP cap, hourly start limit."""

import sys
from pathlib import Path

import pytest

pytest.importorskip("aiohttp")
sys.path.insert(0, str(Path(__file__).parent.parent / "examples" / "web"))
from server import SessionGate, Settings, client_ip, origin_allowed  # noqa: E402


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
    def __init__(self, remote, headers=None, host="demo.example"):
        self.remote, self.headers, self.host = remote, headers or {}, host


PROXY = (__import__("ipaddress").ip_network("10.0.0.0/8"),)


def test_client_ip_ignores_private_peers_without_a_trusted_proxy():
    assert client_ip(FakeRequest("172.20.1.210"), trusted=()) is None  # Modal
    assert client_ip(FakeRequest("1.1.1.1"), trusted=()) == "1.1.1.1"


def test_forwarded_for_is_ignored_from_untrusted_peers():
    spoofed = FakeRequest("1.1.1.1", {"X-Forwarded-For": "8.8.8.8"})
    assert client_ip(spoofed, trusted=PROXY) == "1.1.1.1"


def test_forwarded_for_from_a_trusted_proxy_uses_the_last_hop_and_validates_it():
    assert client_ip(FakeRequest("10.0.0.5", {"X-Forwarded-For": "9.9.9.9, 8.8.8.8"}), trusted=PROXY) == "8.8.8.8"
    assert client_ip(FakeRequest("10.0.0.5", {"X-Forwarded-For": "not-an-ip"}), trusted=PROXY) is None


def test_visitor_table_stays_bounded():
    gate, t = make(max_sessions=10**6, max_starts_per_hour=10**6)
    for i in range(1000):
        gate.admit(f"ip{i}")
        gate.release(f"ip{i}")
    t[0] = 3601
    gate.admit("fresh")
    assert set(gate._starts) == {"*", "fresh"}


def test_websocket_origin_must_match_the_page_by_default():
    assert origin_allowed(FakeRequest("1.1.1.1", {"Origin": "https://demo.example"}), allowed=frozenset())
    assert not origin_allowed(FakeRequest("1.1.1.1", {"Origin": "https://evil.example"}), allowed=frozenset())
    assert not origin_allowed(FakeRequest("1.1.1.1"), allowed=frozenset())  # no Origin: not a browser page


def test_websocket_origin_allowlist_overrides_same_host():
    allowed = frozenset({"https://site.example"})
    assert origin_allowed(FakeRequest("1.1.1.1", {"Origin": "https://site.example/"}), allowed=allowed)
    assert not origin_allowed(FakeRequest("1.1.1.1", {"Origin": "https://demo.example"}), allowed=allowed)


def test_settings_are_read_when_built_not_at_import(monkeypatch):
    # main() builds Settings after load_env(), so values that only live in .env apply.
    monkeypatch.setenv("MAX_SESSIONS", "3")
    monkeypatch.setenv("ALLOWED_ORIGINS", "https://site.example/, https://b.example")
    monkeypatch.setenv("TRUSTED_PROXIES", "10.0.0.0/8")
    s = Settings.from_env()
    assert s.max_sessions == 3 and s.max_session_seconds == 300.0
    assert s.allowed_origins == {"https://site.example", "https://b.example"}
    assert client_ip(FakeRequest("10.1.2.3", {"X-Forwarded-For": "8.8.8.8"}), s.trusted_proxies) == "8.8.8.8"
