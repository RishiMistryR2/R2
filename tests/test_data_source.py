"""
tests/test_data_source.py — data source labeling: MOCK vs LIVE.

Verifies:
  - build_demo_state() stamps data_source="MOCK" — never silent fake prices.
  - RuntimeState.data_source defaults to "MOCK".
  - validate_credentials() returns False when keys are absent or API fails.
  - update_cycle stamps LIVE when Alpaca succeeds, MOCK when it fails.
  - startup() _validate_alpaca handles all credential combinations correctly.
  - Dashboard header always shows [DATA:MOCK] or [DATA:LIVE].
  - startup() remains non-fatal when memory or seed steps fail.

No real network calls — Alpaca is fully mocked.

Run: PYTHONPATH=. python tests/test_data_source.py
"""

import os
import sys
import traceback
import unittest.mock as mock
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

passed = failed = 0


def run(name, fn):
    global passed, failed
    try:
        fn()
        print(f"  PASS  {name}")
        passed += 1
    except Exception as e:
        print(f"  FAIL  {name}: {e}")
        traceback.print_exc()
        failed += 1


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _demo_state():
    from runtime.update_cycle import build_demo_state, reset_engine
    reset_engine()
    return build_demo_state("SPY")


# ═══════════════════════════════════════════════════════════
# build_demo_state tagging
# ═══════════════════════════════════════════════════════════

def test_build_demo_state_has_mock_label():
    state = _demo_state()
    assert hasattr(state, "data_source"), "MarketState missing data_source attribute"
    assert state.data_source == "MOCK"


def test_build_demo_state_price_is_fixed_545():
    state = _demo_state()
    close = state.indicators["5m"].close
    assert close == 545.0, f"Expected 545.0, got {close} — may be pulling live data!"


def test_runtime_state_defaults_to_mock():
    from runtime.runtime_state import RuntimeState
    assert RuntimeState().data_source == "MOCK"


# ═══════════════════════════════════════════════════════════
# validate_credentials()
# ═══════════════════════════════════════════════════════════

def test_validate_credentials_false_when_no_keys():
    from engine.alpaca_feed import validate_credentials
    with mock.patch("engine.alpaca_feed.ALPACA_API_KEY",    ""), \
         mock.patch("engine.alpaca_feed.ALPACA_SECRET_KEY", ""):
        assert validate_credentials() is False


def test_validate_credentials_false_on_http_error():
    from engine.alpaca_feed import validate_credentials
    with mock.patch("engine.alpaca_feed.ALPACA_API_KEY",    "key"), \
         mock.patch("engine.alpaca_feed.ALPACA_SECRET_KEY", "secret"), \
         mock.patch("engine.alpaca_feed._requests.get",
                    side_effect=Exception("connection refused")):
        assert validate_credentials() is False


def test_validate_credentials_false_on_bad_status():
    from engine.alpaca_feed import validate_credentials
    fake_resp = mock.Mock()
    fake_resp.status_code = 403
    fake_resp.text = "forbidden"
    with mock.patch("engine.alpaca_feed.ALPACA_API_KEY",    "key"), \
         mock.patch("engine.alpaca_feed.ALPACA_SECRET_KEY", "secret"), \
         mock.patch("engine.alpaca_feed._requests.get", return_value=fake_resp):
        assert validate_credentials() is False


def test_validate_credentials_true_on_valid_response():
    from engine.alpaca_feed import validate_credentials
    fake_resp = mock.Mock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {"bar": {"c": 735.12}}
    with mock.patch("engine.alpaca_feed.ALPACA_API_KEY",    "key"), \
         mock.patch("engine.alpaca_feed.ALPACA_SECRET_KEY", "secret"), \
         mock.patch("engine.alpaca_feed._requests.get", return_value=fake_resp):
        assert validate_credentials() is True


# ═══════════════════════════════════════════════════════════
# update_cycle data_source propagation
# ═══════════════════════════════════════════════════════════

def test_demo_mode_stamps_mock_on_runtime_state():
    from runtime.runtime_state import RuntimeState, SessionMode
    from runtime.update_cycle import run_update_cycle, reset_engine
    reset_engine()
    s = RuntimeState(session_mode=SessionMode.DEMO)
    run_update_cycle(s)
    assert s.data_source == "MOCK", f"Expected MOCK, got {s.data_source}"


def test_demo_mode_stamps_mock_on_market_state():
    from runtime.runtime_state import RuntimeState, SessionMode
    from runtime.update_cycle import run_update_cycle, reset_engine
    reset_engine()
    s = RuntimeState(session_mode=SessionMode.DEMO)
    run_update_cycle(s)
    assert getattr(s.market_state, "data_source", None) == "MOCK"


def test_live_mode_alpaca_success_sets_live():
    from runtime.runtime_state import RuntimeState, SessionMode
    from runtime.update_cycle import run_update_cycle, reset_engine
    reset_engine()

    fake_market = _demo_state()
    fake_market.data_source = "LIVE"
    fake_market.indicators["5m"].close = 735.50

    s = RuntimeState(session_mode=SessionMode.LIVE)
    with mock.patch("runtime.update_cycle.build_live_state",
                    return_value=fake_market):
        run_update_cycle(s)

    assert s.data_source == "LIVE", f"Expected LIVE, got {s.data_source}"
    assert s.market_state.indicators["5m"].close == 735.50


def test_live_mode_alpaca_failure_falls_back_to_mock():
    from runtime.runtime_state import RuntimeState, SessionMode
    from runtime.update_cycle import run_update_cycle, reset_engine
    reset_engine()
    s = RuntimeState(session_mode=SessionMode.LIVE)
    with mock.patch("runtime.update_cycle.build_live_state",
                    side_effect=RuntimeError("network timeout")):
        run_update_cycle(s)

    assert s.data_source == "MOCK", (
        f"Expected MOCK fallback after Alpaca failure, got {s.data_source!r}"
    )


def test_live_fallback_uses_mock_price_545():
    from runtime.runtime_state import RuntimeState, SessionMode
    from runtime.update_cycle import run_update_cycle, reset_engine
    reset_engine()
    s = RuntimeState(session_mode=SessionMode.LIVE)
    with mock.patch("runtime.update_cycle.build_live_state",
                    side_effect=Exception("down")):
        run_update_cycle(s)

    close = s.market_state.indicators["5m"].close
    assert close == 545.0, (
        f"After Alpaca failure, expected mock price 545.0, got {close}. "
        f"data_source={s.data_source}"
    )


# ═══════════════════════════════════════════════════════════
# Startup (_validate_alpaca)
# ═══════════════════════════════════════════════════════════

def test_validate_alpaca_demo_config_sets_mock():
    from runtime.run_r2 import StartupConfig, _validate_alpaca
    from runtime.runtime_state import RuntimeState, SessionMode
    s = RuntimeState(session_mode=SessionMode.DEMO)
    _validate_alpaca(s, StartupConfig(live=False))
    assert s.data_source  == "MOCK"
    assert s.session_mode == SessionMode.DEMO


def test_validate_alpaca_live_no_keys_falls_back():
    from runtime.run_r2 import StartupConfig, _validate_alpaca
    from runtime.runtime_state import RuntimeState, SessionMode
    s = RuntimeState(session_mode=SessionMode.LIVE)
    with mock.patch("config.settings.ALPACA_API_KEY",    ""), \
         mock.patch("config.settings.ALPACA_SECRET_KEY", ""):
        _validate_alpaca(s, StartupConfig(live=True))
    assert s.data_source  == "MOCK"
    assert s.session_mode == SessionMode.DEMO


def test_validate_alpaca_live_connection_failure_falls_back():
    from runtime.run_r2 import StartupConfig, _validate_alpaca
    from runtime.runtime_state import RuntimeState, SessionMode
    s = RuntimeState(session_mode=SessionMode.LIVE)
    with mock.patch("config.settings.ALPACA_API_KEY",    "key"), \
         mock.patch("config.settings.ALPACA_SECRET_KEY", "sec"), \
         mock.patch("engine.alpaca_feed.validate_credentials", return_value=False):
        _validate_alpaca(s, StartupConfig(live=True))
    assert s.data_source  == "MOCK"
    assert s.session_mode == SessionMode.DEMO


def test_validate_alpaca_live_success_sets_live():
    from runtime.run_r2 import StartupConfig, _validate_alpaca
    from runtime.runtime_state import RuntimeState, SessionMode
    s = RuntimeState(session_mode=SessionMode.DEMO)
    with mock.patch("config.settings.ALPACA_API_KEY",    "real_key"), \
         mock.patch("config.settings.ALPACA_SECRET_KEY", "real_sec"), \
         mock.patch("engine.alpaca_feed.validate_credentials", return_value=True):
        _validate_alpaca(s, StartupConfig(live=True))
    assert s.data_source  == "LIVE"
    assert s.session_mode == SessionMode.LIVE


def test_validate_alpaca_exception_falls_back():
    from runtime.run_r2 import StartupConfig, _validate_alpaca
    from runtime.runtime_state import RuntimeState, SessionMode
    s = RuntimeState(session_mode=SessionMode.LIVE)
    with mock.patch("config.settings.ALPACA_API_KEY",    "key"), \
         mock.patch("config.settings.ALPACA_SECRET_KEY", "sec"), \
         mock.patch("engine.alpaca_feed.validate_credentials",
                    side_effect=ImportError("alpaca-py not installed")):
        _validate_alpaca(s, StartupConfig(live=True))  # must not raise
    assert s.data_source  == "MOCK"
    assert s.session_mode == SessionMode.DEMO


def test_full_startup_memory_failure_non_fatal():
    from runtime.run_r2 import StartupConfig, startup
    from runtime.runtime_state import RuntimeStatus
    c = StartupConfig(mock_ai=True, show_dashboard=False)
    with mock.patch("runtime.run_r2._load_memory",
                    side_effect=Exception("disk error")):
        state, _ = startup(c)
    assert state.status == RuntimeStatus.RUNNING


def test_full_startup_seed_failure_non_fatal():
    from runtime.run_r2 import StartupConfig, startup
    from runtime.runtime_state import RuntimeStatus
    c = StartupConfig(mock_ai=True, show_dashboard=False)
    with mock.patch("runtime.run_r2._seed_market_state",
                    side_effect=Exception("feed down")):
        state, _ = startup(c)
    assert state.status == RuntimeStatus.RUNNING


# ═══════════════════════════════════════════════════════════
# Dashboard header shows DATA label
# ═══════════════════════════════════════════════════════════

def test_dashboard_header_shows_mock():
    from cli.dashboard import _render_header
    state = _demo_state()  # already tagged MOCK
    header = _render_header(state, datetime.now(tz=timezone.utc))
    assert "[DATA:MOCK]" in header, f"Header missing [DATA:MOCK]: {header!r}"


def test_dashboard_header_shows_live():
    from cli.dashboard import _render_header
    state = _demo_state()
    state.data_source = "LIVE"
    header = _render_header(state, datetime.now(tz=timezone.utc))
    assert "[DATA:LIVE]" in header, f"Header missing [DATA:LIVE]: {header!r}"


def test_dashboard_header_defaults_to_mock_if_attr_missing():
    from cli.dashboard import _render_header
    state = _demo_state()
    if hasattr(state, "data_source"):
        del state.data_source
    header = _render_header(state, datetime.now(tz=timezone.utc))
    assert "[DATA:MOCK]" in header, f"Should default to MOCK: {header!r}"


def test_full_dashboard_render_contains_data_label():
    from cli.dashboard import render_dashboard
    state = _demo_state()
    output = render_dashboard(state)
    assert "DATA:" in output, "Dashboard output missing DATA: label"


# ═══════════════════════════════════════════════════════════
# get_latest_price
# ═══════════════════════════════════════════════════════════

def test_get_latest_price_returns_none_on_failure():
    from engine.alpaca_feed import get_latest_price
    with mock.patch("engine.alpaca_feed._requests.get",
                    side_effect=Exception("timeout")):
        assert get_latest_price("SPY") is None


def test_get_latest_price_returns_float_on_success():
    from engine.alpaca_feed import get_latest_price
    fake_resp = mock.Mock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {"bar": {"c": 735.50}}
    with mock.patch("engine.alpaca_feed._requests.get", return_value=fake_resp):
        assert get_latest_price("SPY") == 735.50


# ═══════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    tests = [
        (n, f) for n, f in sorted(globals().items())
        if n.startswith("test_") and callable(f)
    ]
    print(f"\nData source tests ({len(tests)} total)\n{'─' * 60}")
    for name, fn in tests:
        run(name, fn)
    print(f"\n{'=' * 60}")
    print(f"{passed} passed, {failed} failed  ({passed + failed} total)")
    sys.exit(0 if failed == 0 else 1)