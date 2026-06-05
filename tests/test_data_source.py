"""
tests/test_data_source.py — data source, session, and current-price tests.

Covers:
  engine/market_session.py  : all six session states, DST-safe ET conversion,
                               predicates, to_session_phase bridge
  engine/alpaca_feed.py     : validate_credentials, get_latest_trade_price,
                               get_latest_quote, _stamp_current_price priority
                               (latest_trade > quote_midpoint > 1m_bar_fallback),
                               extended-hours labels
  runtime/update_cycle.py   : MOCK/LIVE stamping, fallback on Alpaca failure
  runtime/run_r2.py         : _validate_alpaca all paths, seed, non-fatal startup
  cli/dashboard.py          : price detail display (source / bid-ask / updated /
                               session), DATA:MOCK/LIVE header

No real network calls.

Run: PYTHONPATH=. python tests/test_data_source.py
"""

import os
import sys
import traceback
import unittest.mock as mock
from datetime import datetime, timezone, timedelta

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


def _utc(year, month, day, hour, minute=0, second=0) -> datetime:
    return datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)


def _demo_state():
    from runtime.update_cycle import build_demo_state, reset_engine
    reset_engine()
    return build_demo_state("SPY")


def _live_state(price: float = 735.50):
    s = _demo_state()
    s.data_source = "LIVE"
    for tf in s.indicators:
        s.indicators[tf].close = price
    return s


def _fake_resp(status: int, body: dict):
    r = mock.Mock()
    r.status_code = status
    r.text        = str(body)
    r.json.return_value = body
    return r


# ═══════════════════════════════════════════════════════════
# engine/market_session.py — session classification
# ═══════════════════════════════════════════════════════════

def test_session_premarket():
    from engine.market_session import get_market_session, MarketSession
    # Monday 2024-01-15  08:00 ET = 13:00 UTC (EST = UTC-5)
    dt = _utc(2024, 1, 15, 13, 0)
    assert get_market_session(dt) == MarketSession.PREMARKET


def test_session_open():
    from engine.market_session import get_market_session, MarketSession
    # Monday 09:45 ET = 14:45 UTC
    dt = _utc(2024, 1, 15, 14, 45)
    assert get_market_session(dt) == MarketSession.OPEN


def test_session_midday():
    from engine.market_session import get_market_session, MarketSession
    # Monday 13:00 ET = 18:00 UTC
    dt = _utc(2024, 1, 15, 18, 0)
    assert get_market_session(dt) == MarketSession.MIDDAY


def test_session_power_hour():
    from engine.market_session import get_market_session, MarketSession
    # Monday 15:30 ET = 20:30 UTC
    dt = _utc(2024, 1, 15, 20, 30)
    assert get_market_session(dt) == MarketSession.POWER_HOUR


def test_session_after_hours():
    from engine.market_session import get_market_session, MarketSession
    # Monday 17:00 ET = 22:00 UTC
    dt = _utc(2024, 1, 15, 22, 0)
    assert get_market_session(dt) == MarketSession.AFTER_HOURS


def test_session_closed_overnight():
    from engine.market_session import get_market_session, MarketSession
    # Monday 21:30 ET = Tuesday 02:30 UTC
    dt = _utc(2024, 1, 16, 2, 30)
    assert get_market_session(dt) == MarketSession.CLOSED


def test_session_closed_weekend_saturday():
    from engine.market_session import get_market_session, MarketSession
    # Saturday 2024-01-13 15:00 UTC (10:00 ET, would be OPEN on weekday)
    dt = _utc(2024, 1, 13, 15, 0)
    assert get_market_session(dt) == MarketSession.CLOSED


def test_session_closed_weekend_sunday():
    from engine.market_session import get_market_session, MarketSession
    dt = _utc(2024, 1, 14, 15, 0)
    assert get_market_session(dt) == MarketSession.CLOSED


def test_session_open_boundary_exact_930():
    from engine.market_session import get_market_session, MarketSession
    # Exactly 09:30 ET = 14:30 UTC → OPEN (inclusive lower bound)
    dt = _utc(2024, 1, 15, 14, 30)
    assert get_market_session(dt) == MarketSession.OPEN


def test_session_power_hour_boundary_exact_1500():
    from engine.market_session import get_market_session, MarketSession
    # Exactly 15:00 ET = 20:00 UTC → POWER_HOUR
    dt = _utc(2024, 1, 15, 20, 0)
    assert get_market_session(dt) == MarketSession.POWER_HOUR


def test_session_premarket_boundary_exact_0400():
    from engine.market_session import get_market_session, MarketSession
    # Exactly 04:00 ET = 09:00 UTC → PREMARKET
    dt = _utc(2024, 1, 15, 9, 0)
    assert get_market_session(dt) == MarketSession.PREMARKET


def test_session_after_hours_boundary_exact_1600():
    from engine.market_session import get_market_session, MarketSession
    # Exactly 16:00 ET = 21:00 UTC → AFTER_HOURS
    dt = _utc(2024, 1, 15, 21, 0)
    assert get_market_session(dt) == MarketSession.AFTER_HOURS


def test_session_closed_after_hours_end():
    from engine.market_session import get_market_session, MarketSession
    # 20:01 ET = 01:01 UTC next day → CLOSED
    dt = _utc(2024, 1, 16, 1, 1)
    assert get_market_session(dt) == MarketSession.CLOSED


def test_is_regular_hours_true():
    from engine.market_session import MarketSession, is_regular_hours
    for s in (MarketSession.OPEN, MarketSession.MIDDAY, MarketSession.POWER_HOUR):
        assert is_regular_hours(s) is True, s


def test_is_regular_hours_false():
    from engine.market_session import MarketSession, is_regular_hours
    for s in (MarketSession.PREMARKET, MarketSession.AFTER_HOURS, MarketSession.CLOSED):
        assert is_regular_hours(s) is False, s


def test_is_extended_hours():
    from engine.market_session import MarketSession, is_extended_hours
    assert is_extended_hours(MarketSession.PREMARKET)   is True
    assert is_extended_hours(MarketSession.AFTER_HOURS) is True
    assert is_extended_hours(MarketSession.OPEN)        is False
    assert is_extended_hours(MarketSession.CLOSED)      is False


def test_to_session_phase_all_map():
    from engine.market_session import get_market_session, to_session_phase, MarketSession
    from engine.models import SessionPhase
    for s in MarketSession:
        phase = to_session_phase(s)
        assert isinstance(phase, SessionPhase), f"no mapping for {s}"


def test_to_et_string_returns_et_time():
    from engine.market_session import to_et_string
    # 18:30 UTC on a weekday = 13:30 ET (EST, January)
    dt     = _utc(2024, 1, 15, 18, 30, 0)
    result = to_et_string(dt)
    assert "ET" in result
    assert "13:30:00" in result


def test_to_et_string_none():
    from engine.market_session import to_et_string
    assert to_et_string(None) == "—"


# ═══════════════════════════════════════════════════════════
# engine/alpaca_feed.py — validate / trade / quote / stamp
# ═══════════════════════════════════════════════════════════

def test_validate_credentials_false_no_keys():
    from engine.alpaca_feed import validate_credentials
    with mock.patch("engine.alpaca_feed._api_key",    return_value=""), \
         mock.patch("engine.alpaca_feed._secret_key", return_value=""):
        assert validate_credentials() is False


def test_validate_credentials_false_http_error():
    from engine.alpaca_feed import validate_credentials
    with mock.patch("engine.alpaca_feed._api_key",    return_value="k"), \
         mock.patch("engine.alpaca_feed._secret_key", return_value="s"), \
         mock.patch("engine.alpaca_feed._requests.get", side_effect=Exception("refused")):
        assert validate_credentials() is False


def test_validate_credentials_true():
    from engine.alpaca_feed import validate_credentials
    with mock.patch("engine.alpaca_feed._api_key",    return_value="k"), \
         mock.patch("engine.alpaca_feed._secret_key", return_value="s"), \
         mock.patch("engine.alpaca_feed._requests.get",
                    return_value=_fake_resp(200, {"bar": {"c": 735.12}})):
        assert validate_credentials() is True


def test_get_latest_trade_price_success():
    from engine.alpaca_feed import get_latest_trade_price
    body = {"trade": {"p": 735.50, "t": "2024-01-15T20:59:59Z", "s": 100}}
    with mock.patch("engine.alpaca_feed._api_key",    return_value="k"), \
         mock.patch("engine.alpaca_feed._secret_key", return_value="s"), \
         mock.patch("engine.alpaca_feed._requests.get",
                    return_value=_fake_resp(200, body)):
        result = get_latest_trade_price("SPY")
    assert result is not None
    price, ts = result
    assert price == 735.50
    assert isinstance(ts, datetime)


def test_get_latest_trade_price_none_on_error():
    from engine.alpaca_feed import get_latest_trade_price
    with mock.patch("engine.alpaca_feed._requests.get",
                    side_effect=Exception("timeout")):
        assert get_latest_trade_price("SPY") is None


def test_get_latest_trade_price_none_bad_status():
    from engine.alpaca_feed import get_latest_trade_price
    with mock.patch("engine.alpaca_feed._api_key",    return_value="k"), \
         mock.patch("engine.alpaca_feed._secret_key", return_value="s"), \
         mock.patch("engine.alpaca_feed._requests.get",
                    return_value=_fake_resp(404, {})):
        assert get_latest_trade_price("SPY") is None


def test_get_latest_trade_price_none_no_price_field():
    from engine.alpaca_feed import get_latest_trade_price
    with mock.patch("engine.alpaca_feed._api_key",    return_value="k"), \
         mock.patch("engine.alpaca_feed._secret_key", return_value="s"), \
         mock.patch("engine.alpaca_feed._requests.get",
                    return_value=_fake_resp(200, {"trade": {"t": "2024-01-15T20:59:59Z"}})):
        assert get_latest_trade_price("SPY") is None


def test_get_latest_quote_success():
    from engine.alpaca_feed import get_latest_quote
    body = {"quote": {"bp": 735.48, "ap": 735.52, "bs": 200, "as": 100,
                      "t": "2024-01-15T20:59:59Z"}}
    with mock.patch("engine.alpaca_feed._api_key",    return_value="k"), \
         mock.patch("engine.alpaca_feed._secret_key", return_value="s"), \
         mock.patch("engine.alpaca_feed._requests.get",
                    return_value=_fake_resp(200, body)):
        q = get_latest_quote("SPY")
    assert q is not None
    assert q["bid"]      == 735.48
    assert q["ask"]      == 735.52
    assert q["midpoint"] == round((735.48 + 735.52) / 2, 4)
    assert q["spread"]   == round(735.52 - 735.48, 4)
    assert isinstance(q["timestamp"], datetime)


def test_get_latest_quote_none_on_error():
    from engine.alpaca_feed import get_latest_quote
    with mock.patch("engine.alpaca_feed._requests.get",
                    side_effect=Exception("timeout")):
        assert get_latest_quote("SPY") is None


def test_get_latest_quote_none_missing_fields():
    from engine.alpaca_feed import get_latest_quote
    with mock.patch("engine.alpaca_feed._api_key",    return_value="k"), \
         mock.patch("engine.alpaca_feed._secret_key", return_value="s"), \
         mock.patch("engine.alpaca_feed._requests.get",
                    return_value=_fake_resp(200, {"quote": {"t": "2024-01-15T20:59:59Z"}})):
        assert get_latest_quote("SPY") is None


# ── _stamp_current_price priority ──────────────────────────

def _stamp_with_mocks(state, trade_result, quote_result, bar_fallback,
                      session_name="open"):
    """Helper: run _stamp_current_price with mocked dependencies."""
    from engine.alpaca_feed import _stamp_current_price
    from engine.market_session import MarketSession
    session_map = {
        "open":        MarketSession.OPEN,
        "midday":      MarketSession.MIDDAY,
        "power_hour":  MarketSession.POWER_HOUR,
        "after_hours": MarketSession.AFTER_HOURS,
        "premarket":   MarketSession.PREMARKET,
        "closed":      MarketSession.CLOSED,
    }
    sess = session_map.get(session_name, MarketSession.OPEN)
    with mock.patch("engine.alpaca_feed.get_latest_trade_price",
                    return_value=trade_result), \
         mock.patch("engine.alpaca_feed.get_latest_quote",
                    return_value=quote_result), \
         mock.patch("engine.alpaca_feed.get_market_session", return_value=sess):
        _stamp_current_price(state, "SPY", bar_fallback)


def test_stamp_trade_first_regular_hours():
    from engine.market_session import MarketSession
    state = _demo_state()
    ts    = _utc(2024, 1, 15, 18, 30)
    _stamp_with_mocks(state, (735.50, ts),
                      {"midpoint": 735.51, "bid": 735.49, "ask": 735.52,
                       "spread": 0.03, "timestamp": ts},
                      730.0, "midday")
    assert state.current_price        == 735.50
    assert state.current_price_source == "latest_trade"
    assert state.bid                  is None   # trade source: no bid/ask
    assert state.ask                  is None


def test_stamp_trade_extended_hours_label():
    state = _demo_state()
    ts    = _utc(2024, 1, 15, 22, 0)
    _stamp_with_mocks(state, (735.50, ts), None, 730.0, "after_hours")
    assert state.current_price_source == "extended_hours_trade"


def test_stamp_premarket_extended_label():
    state = _demo_state()
    ts    = _utc(2024, 1, 15, 13, 0)
    _stamp_with_mocks(state, (735.50, ts), None, 730.0, "premarket")
    assert state.current_price_source == "extended_hours_trade"


def test_stamp_quote_midpoint_when_no_trade():
    state = _demo_state()
    ts    = _utc(2024, 1, 15, 18, 30)
    quote = {"midpoint": 735.51, "bid": 735.49, "ask": 735.52,
             "spread": 0.03, "timestamp": ts}
    _stamp_with_mocks(state, None, quote, 730.0, "midday")
    assert state.current_price        == 735.51
    assert state.current_price_source == "quote_midpoint"
    assert state.bid                  == 735.49
    assert state.ask                  == 735.52
    assert state.spread               == 0.03


def test_stamp_extended_quote_label():
    state = _demo_state()
    ts    = _utc(2024, 1, 15, 22, 0)
    quote = {"midpoint": 735.00, "bid": 734.98, "ask": 735.02,
             "spread": 0.04, "timestamp": ts}
    _stamp_with_mocks(state, None, quote, 730.0, "after_hours")
    assert state.current_price_source == "extended_hours_quote_midpoint"


def test_stamp_bar_fallback_when_no_trade_or_quote():
    state = _demo_state()
    _stamp_with_mocks(state, None, None, 733.25, "midday")
    assert state.current_price        == 733.25
    assert state.current_price_source == "1m_bar_fallback"
    assert state.bid                  is None
    assert state.ask                  is None


def test_stamp_bar_fallback_closed_session():
    state = _demo_state()
    _stamp_with_mocks(state, None, None, 740.0, "closed")
    assert state.current_price_source == "1m_bar_fallback"


# ═══════════════════════════════════════════════════════════
# build_demo_state — uses real session classifier
# ═══════════════════════════════════════════════════════════

def test_build_demo_state_has_mock_label():
    assert _demo_state().data_source == "MOCK"


def test_build_demo_state_price_is_545():
    assert _demo_state().indicators["5m"].close == 545.0


def test_build_demo_state_has_market_session_attr():
    from engine.market_session import MarketSession
    state = _demo_state()
    ms = getattr(state, "market_session", None)
    assert ms is not None
    assert isinstance(ms, MarketSession)


def test_runtime_state_defaults_to_mock():
    from runtime.runtime_state import RuntimeState
    assert RuntimeState().data_source == "MOCK"


# ═══════════════════════════════════════════════════════════
# update_cycle data_source propagation
# ═══════════════════════════════════════════════════════════

def test_demo_cycle_stamps_mock_on_runtime():
    from runtime.runtime_state import RuntimeState, SessionMode
    from runtime.update_cycle import run_update_cycle, reset_engine
    reset_engine()
    s = RuntimeState(session_mode=SessionMode.DEMO)
    run_update_cycle(s)
    assert s.data_source == "MOCK"


def test_demo_cycle_stamps_mock_on_market_state():
    from runtime.runtime_state import RuntimeState, SessionMode
    from runtime.update_cycle import run_update_cycle, reset_engine
    reset_engine()
    s = RuntimeState(session_mode=SessionMode.DEMO)
    run_update_cycle(s)
    assert getattr(s.market_state, "data_source", None) == "MOCK"


def test_live_cycle_success_sets_live():
    from runtime.runtime_state import RuntimeState, SessionMode
    from runtime.update_cycle import run_update_cycle, reset_engine
    reset_engine()
    fake = _live_state(735.50)
    s    = RuntimeState(session_mode=SessionMode.LIVE)
    with mock.patch("runtime.update_cycle.build_live_state", return_value=fake):
        run_update_cycle(s)
    assert s.data_source == "LIVE"
    assert s.market_state.indicators["5m"].close == 735.50


def test_live_cycle_fallback_on_alpaca_fail():
    from runtime.runtime_state import RuntimeState, SessionMode
    from runtime.update_cycle import run_update_cycle, reset_engine
    reset_engine()
    s = RuntimeState(session_mode=SessionMode.LIVE)
    with mock.patch("runtime.update_cycle.build_live_state",
                    side_effect=RuntimeError("timeout")):
        run_update_cycle(s)
    assert s.data_source == "MOCK"
    assert s.market_state.indicators["5m"].close == 545.0


def test_live_cycle_error_does_not_stop_next():
    from runtime.runtime_state import RuntimeState, SessionMode
    from runtime.update_cycle import run_update_cycle, reset_engine
    reset_engine()
    s = RuntimeState(session_mode=SessionMode.LIVE)
    with mock.patch("runtime.update_cycle.build_live_state",
                    side_effect=Exception("transient")):
        ok1 = run_update_cycle(s)
    assert ok1 is True    # fallback to mock succeeds
    assert s.health.consecutive_errors == 0


# ═══════════════════════════════════════════════════════════
# run_r2 — _validate_alpaca + startup safety
# ═══════════════════════════════════════════════════════════

def test_validate_alpaca_demo_sets_mock():
    from runtime.run_r2 import StartupConfig, _validate_alpaca
    from runtime.runtime_state import RuntimeState, SessionMode
    s = RuntimeState(session_mode=SessionMode.DEMO)
    _validate_alpaca(s, StartupConfig(live=False))
    assert s.data_source == "MOCK"


def test_validate_alpaca_no_keys_falls_back():
    from runtime.run_r2 import StartupConfig, _validate_alpaca
    from runtime.runtime_state import RuntimeState, SessionMode
    s = RuntimeState(session_mode=SessionMode.LIVE)
    with mock.patch("os.getenv", return_value=""):
        _validate_alpaca(s, StartupConfig(live=True))
    assert s.data_source == "MOCK" and s.session_mode == SessionMode.DEMO


def test_validate_alpaca_conn_fail_falls_back():
    from runtime.run_r2 import StartupConfig, _validate_alpaca
    from runtime.runtime_state import RuntimeState, SessionMode
    s = RuntimeState(session_mode=SessionMode.LIVE)
    with mock.patch("os.getenv",
                    side_effect=lambda k, d="": "fake" if "KEY" in k else d), \
         mock.patch("engine.alpaca_feed.validate_credentials", return_value=False):
        _validate_alpaca(s, StartupConfig(live=True))
    assert s.data_source == "MOCK" and s.session_mode == SessionMode.DEMO


def test_validate_alpaca_success_sets_live():
    from runtime.run_r2 import StartupConfig, _validate_alpaca
    from runtime.runtime_state import RuntimeState, SessionMode
    s = RuntimeState(session_mode=SessionMode.DEMO)
    with mock.patch("os.getenv",
                    side_effect=lambda k, d="": "real" if "KEY" in k else d), \
         mock.patch("engine.alpaca_feed.validate_credentials", return_value=True):
        _validate_alpaca(s, StartupConfig(live=True))
    assert s.data_source == "LIVE" and s.session_mode == SessionMode.LIVE


def test_validate_alpaca_exception_non_fatal():
    from runtime.run_r2 import StartupConfig, _validate_alpaca
    from runtime.runtime_state import RuntimeState, SessionMode
    s = RuntimeState(session_mode=SessionMode.LIVE)
    with mock.patch("os.getenv",
                    side_effect=lambda k, d="": "k" if "KEY" in k else d), \
         mock.patch("engine.alpaca_feed.validate_credentials",
                    side_effect=ImportError("not installed")):
        _validate_alpaca(s, StartupConfig(live=True))
    assert s.data_source == "MOCK"


def test_startup_memory_failure_non_fatal():
    from runtime.run_r2 import StartupConfig, startup
    from runtime.runtime_state import RuntimeStatus
    with mock.patch("runtime.run_r2._load_memory", side_effect=Exception("disk")):
        state, _ = startup(StartupConfig(mock_ai=True, show_dashboard=False))
    assert state.status == RuntimeStatus.RUNNING


def test_startup_seed_failure_non_fatal():
    from runtime.run_r2 import StartupConfig, startup
    from runtime.runtime_state import RuntimeStatus
    with mock.patch("runtime.run_r2._seed_market_state", side_effect=Exception("feed")):
        state, _ = startup(StartupConfig(mock_ai=True, show_dashboard=False))
    assert state.status == RuntimeStatus.RUNNING


def test_startup_marks_update_done_after_seed():
    from runtime.run_r2 import StartupConfig, startup
    _, sched = startup(StartupConfig(mock_ai=True, show_dashboard=False))
    assert sched.task_due("update") is False


# ═══════════════════════════════════════════════════════════
# cli/dashboard.py — price detail + session display
# ═══════════════════════════════════════════════════════════

def test_dashboard_header_mock():
    from cli.dashboard import _render_header
    header = _render_header(_demo_state(), datetime.now(tz=timezone.utc))
    assert "[DATA:MOCK]" in header


def test_dashboard_header_live():
    from cli.dashboard import _render_header
    state = _demo_state()
    state.data_source = "LIVE"
    header = _render_header(state, datetime.now(tz=timezone.utc))
    assert "[DATA:LIVE]" in header


def test_dashboard_shows_current_price_not_bar_close():
    from cli.dashboard import _render_market_status
    state = _demo_state()
    state.current_price        = 735.50
    state.current_price_source = "latest_trade"
    state.bid = state.ask = state.spread = None
    state.current_price_timestamp = None
    output = _render_market_status(state, None)
    assert "735.50" in output
    assert "545"    not in output


def test_dashboard_shows_source_label():
    from cli.dashboard import _render_market_status
    state = _demo_state()
    state.current_price           = 735.50
    state.current_price_source    = "latest_trade"
    state.bid = state.ask = state.spread = None
    state.current_price_timestamp = None
    output = _render_market_status(state, None)
    assert "latest_trade" in output


def test_dashboard_shows_bid_ask_spread_for_quote():
    from cli.dashboard import _render_market_status
    state = _demo_state()
    state.current_price           = 735.50
    state.current_price_source    = "quote_midpoint"
    state.bid                     = 735.48
    state.ask                     = 735.52
    state.spread                  = 0.04
    state.current_price_timestamp = None
    output = _render_market_status(state, None)
    assert "735.48" in output
    assert "735.52" in output
    assert "0.04"   in output


def test_dashboard_shows_updated_timestamp():
    from cli.dashboard import _render_market_status
    state = _demo_state()
    state.current_price           = 735.50
    state.current_price_source    = "latest_trade"
    state.bid = state.ask = state.spread = None
    # Monday 13:30:00 ET = 18:30:00 UTC
    state.current_price_timestamp = _utc(2024, 1, 15, 18, 30, 0)
    output = _render_market_status(state, None)
    assert "Updated" in output
    assert "ET"      in output


def test_dashboard_shows_extended_hours_trade_label():
    from cli.dashboard import _render_market_status
    state = _demo_state()
    state.current_price           = 735.50
    state.current_price_source    = "extended_hours_trade"
    state.bid = state.ask = state.spread = None
    state.current_price_timestamp = None
    output = _render_market_status(state, None)
    assert "extended_hours_trade" in output


def test_dashboard_shows_1m_bar_fallback_label():
    from cli.dashboard import _render_market_status
    state = _demo_state()
    state.current_price           = 733.25
    state.current_price_source    = "1m_bar_fallback"
    state.bid = state.ask = state.spread = None
    state.current_price_timestamp = None
    output = _render_market_status(state, None)
    assert "733.25"       in output
    assert "1m_bar_fallback" in output


def test_dashboard_falls_back_to_bar_when_no_current_price():
    from cli.dashboard import _render_market_status
    state = _demo_state()
    for attr in ("current_price", "current_price_source", "bid", "ask", "spread",
                 "current_price_timestamp"):
        if hasattr(state, attr):
            delattr(state, attr)
    output = _render_market_status(state, None)
    assert "545" in output


def test_dashboard_shows_market_session_in_status():
    from cli.dashboard import _render_readiness
    from engine.market_session import MarketSession
    state = _demo_state()
    state.market_session = MarketSession.AFTER_HOURS
    output = _render_readiness(state, None)
    assert "AFTER HOURS" in output or "after_hours" in output.lower()


def test_dashboard_shows_premarket_session():
    from cli.dashboard import _render_readiness
    from engine.market_session import MarketSession
    state = _demo_state()
    state.market_session = MarketSession.PREMARKET
    output = _render_readiness(state, None)
    assert "PREMARKET" in output or "premarket" in output.lower()


def test_render_price_line_uses_current_price():
    from cli.dashboard import render_price_line
    state = _demo_state()
    state.current_price        = 735.50
    state.current_price_source = "latest_trade"
    state.bid = state.ask      = None
    line = render_price_line(state)
    assert "735.50" in line and "545" not in line


def test_render_price_line_shows_bid_ask():
    from cli.dashboard import render_price_line
    state = _demo_state()
    state.current_price        = 735.50
    state.current_price_source = "quote_midpoint"
    state.bid                  = 735.48
    state.ask                  = 735.52
    line = render_price_line(state)
    assert "735.48" in line and "735.52" in line


def test_render_price_line_bar_fallback():
    from cli.dashboard import render_price_line
    state = _demo_state()
    assert "545" in render_price_line(state)


def test_render_session_line_uses_market_session():
    from cli.dashboard import render_session_line
    from engine.market_session import MarketSession
    state = _demo_state()
    state.market_session = MarketSession.POWER_HOUR
    line = render_session_line(state)
    assert "POWER HOUR" in line or "power_hour" in line.lower()


def test_full_dashboard_contains_data_label():
    from cli.dashboard import render_dashboard
    output = render_dashboard(_demo_state())
    assert "DATA:" in output


# ═══════════════════════════════════════════════════════════
# get_latest_price (bar-based)
# ═══════════════════════════════════════════════════════════

def test_get_latest_price_none_on_error():
    from engine.alpaca_feed import get_latest_price
    with mock.patch("engine.alpaca_feed._requests.get",
                    side_effect=Exception("timeout")):
        assert get_latest_price("SPY") is None


def test_get_latest_price_returns_float():
    from engine.alpaca_feed import get_latest_price
    with mock.patch("engine.alpaca_feed._api_key",    return_value="k"), \
         mock.patch("engine.alpaca_feed._secret_key", return_value="s"), \
         mock.patch("engine.alpaca_feed._requests.get",
                    return_value=_fake_resp(200, {"bar": {"c": 735.50}})):
        assert get_latest_price("SPY") == 735.50


# ═══════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    tests = [
        (n, f) for n, f in sorted(globals().items())
        if n.startswith("test_") and callable(f)
    ]
    print(f"\nData source + session tests ({len(tests)} total)\n{'─' * 60}")
    for name, fn in tests:
        run(name, fn)
    print(f"\n{'=' * 60}")
    print(f"{passed} passed, {failed} failed  ({passed + failed} total)")
    sys.exit(0 if failed == 0 else 1)