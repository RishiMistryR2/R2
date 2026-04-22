"""
tests/test_phase2.py — Phase 2: core market engine.

Tests:
  - CandleStore: ingest, dedup, session tagging, VWAP, readiness, live candles
  - Readiness module: tier logic, penalties, confidence caps
  - classify_session_type: UTC timestamp to session label
  - IndicatorEngine: full compute() pipeline with readiness gating
  - AlpacaFeed: bar/timestamp parsing helpers (no network)
  - MarketEngine: event detection logic (no network)

Run: PYTHONPATH=. python tests/test_phase2.py
"""

import sys, os, asyncio, traceback
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import datetime, timezone, timedelta

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


def arun(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _ts(et_hour=10, minute=0, second=0) -> datetime:
    """UTC datetime for given ET hour (EDT = UTC-4) on 2025-04-09."""
    return datetime(2025, 4, 9, et_hour + 4, minute, second, tzinfo=timezone.utc)


def _bar(ts, o=545.0, h=546.0, l=544.0, c=545.5, v=10_000.0) -> dict:
    return {"timestamp": ts, "open": o, "high": h, "low": l, "close": c, "volume": v}


def _filled_store(n=60, base=545.0):
    """
    CandleStore with n rising bars on both 15m and 1h,
    so readiness is at least PARTIALLY_READY.
    """
    from engine.candle_store import CandleStore
    store = CandleStore()
    for i in range(n):
        b = _bar(_ts(10, i),
                 o=base + i*0.02, h=base + i*0.02 + 0.5,
                 l=base + i*0.02 - 0.3, c=base + i*0.02 + 0.1,
                 v=10_000 + i*100)
        arun(store.ingest_bar(b, "15m"))
        arun(store.ingest_bar(b, "1h"))
    return store


# ═══════════════════════════════════════════════════════════════════════════
# CandleStore
# ═══════════════════════════════════════════════════════════════════════════

def test_store_ingest_bar_accepted():
    from engine.candle_store import CandleStore
    store = CandleStore()
    ok = arun(store.ingest_bar(_bar(_ts(10, 0)), "5m"))
    assert ok is True
    assert store.candle_count("5m") == 1


def test_store_ingest_bar_dedup_same_ts():
    from engine.candle_store import CandleStore
    store = CandleStore()
    b = _bar(_ts(10, 0))
    arun(store.ingest_bar(b, "5m"))
    ok = arun(store.ingest_bar(b, "5m"))
    assert ok is False
    assert store.candle_count("5m") == 1


def test_store_ingest_bar_dedup_older_ts():
    from engine.candle_store import CandleStore
    store = CandleStore()
    arun(store.ingest_bar(_bar(_ts(10, 10)), "5m"))
    ok = arun(store.ingest_bar(_bar(_ts(10, 0)), "5m"))
    assert ok is False
    assert store.candle_count("5m") == 1


def test_store_multiple_bars_correct_order():
    from engine.candle_store import CandleStore
    store = CandleStore()
    for i in range(5):
        arun(store.ingest_bar(_bar(_ts(10, i), c=545.0 + i), "5m"))
    closes = store.get_closes("5m")
    assert closes == [545.0, 546.0, 547.0, 548.0, 549.0]


def test_store_get_closes_n():
    from engine.candle_store import CandleStore
    store = CandleStore()
    for i in range(10):
        arun(store.ingest_bar(_bar(_ts(10, i), c=545.0 + i), "15m"))
    last3 = store.get_closes("15m", n=3)
    assert len(last3) == 3
    assert last3[-1] == 554.0


def test_store_session_type_stored():
    from engine.candle_store import CandleStore
    store = CandleStore()
    arun(store.ingest_bar(_bar(_ts(5, 0)), "5m", session_type="premarket"))
    arun(store.ingest_bar(_bar(_ts(10, 1)), "5m", session_type="regular"))
    candles = store.get_candles("5m")
    assert candles[0].session_type == "premarket"
    assert candles[1].session_type == "regular"


def test_store_premarket_fraction():
    from engine.candle_store import CandleStore
    store = CandleStore()
    # premarket bars first (earlier timestamps)
    for i in range(2):
        arun(store.ingest_bar(_bar(_ts(5, i)), "5m", session_type="premarket"))
    for i in range(8):
        arun(store.ingest_bar(_bar(_ts(10, i)), "5m", session_type="regular"))
    frac = store.premarket_fraction("5m", lookback=10)
    assert abs(frac - 0.20) < 0.01


def test_store_premarket_fraction_zero():
    from engine.candle_store import CandleStore
    store = CandleStore()
    for i in range(10):
        arun(store.ingest_bar(_bar(_ts(10, i)), "5m", session_type="regular"))
    assert store.premarket_fraction("5m") == 0.0


def test_store_vwap_accumulates():
    from engine.candle_store import CandleStore
    store = CandleStore()
    arun(store.ingest_trade(545.0, 1000.0, _ts(10, 0), "regular"))
    arun(store.ingest_trade(547.0, 1000.0, _ts(10, 0, 30), "regular"))
    v = store.get_vwap()
    assert v is not None
    assert 545.0 < v < 547.0


def test_store_vwap_reset():
    from engine.candle_store import CandleStore
    store = CandleStore()
    arun(store.ingest_trade(545.0, 1000.0, _ts(10, 0), "regular"))
    store.reset_vwap()
    assert store.get_vwap() is None


def test_store_candle_counts_all_tfs():
    from engine.candle_store import CandleStore
    from config.settings import TIMEFRAMES
    store  = CandleStore()
    counts = store.candle_counts()
    assert set(counts.keys()) == set(TIMEFRAMES)
    assert all(v == 0 for v in counts.values())


def test_store_live_candle_created():
    from engine.candle_store import CandleStore
    store = CandleStore()
    t = datetime(2025, 4, 9, 14, 0, 0, tzinfo=timezone.utc)
    arun(store.ingest_trade(545.0, 100.0, t, "regular"))
    live = store.get_live_candle("1m")
    assert live is not None
    assert live.open == 545.0
    assert live.is_closed is False


def test_store_live_candle_closes_on_new_period():
    from engine.candle_store import CandleStore
    store = CandleStore()
    t1 = datetime(2025, 4, 9, 14, 0, 0, tzinfo=timezone.utc)
    t2 = datetime(2025, 4, 9, 14, 1, 5, tzinfo=timezone.utc)
    arun(store.ingest_trade(545.0, 100.0, t1, "regular"))
    closed = arun(store.ingest_trade(546.0, 100.0, t2, "regular"))
    assert "1m" in closed
    assert store.candle_count("1m") == 1


def test_store_get_readiness_not_ready():
    from engine.candle_store import CandleStore
    from engine.models import Readiness
    store = CandleStore()
    r, _ = store.get_readiness()
    assert r == Readiness.NOT_READY


def test_store_get_readiness_after_seeding():
    from engine.candle_store import CandleStore
    from engine.models import Readiness
    store = _filled_store(35)
    r, _ = store.get_readiness()
    assert r == Readiness.PARTIALLY_READY


# ═══════════════════════════════════════════════════════════════════════════
# Readiness module
# ═══════════════════════════════════════════════════════════════════════════

def test_readiness_empty():
    from engine.readiness import compute_readiness
    from engine.models import Readiness
    r, note = compute_readiness({})
    assert r == Readiness.NOT_READY
    assert "Seeding" in note


def test_readiness_missing_1h():
    from engine.readiness import compute_readiness
    from engine.models import Readiness
    r, _ = compute_readiness({"1d": 100, "15m": 50})
    assert r == Readiness.NOT_READY


def test_readiness_missing_15m():
    from engine.readiness import compute_readiness
    from engine.models import Readiness
    r, _ = compute_readiness({"1d": 100, "1h": 80})
    assert r == Readiness.NOT_READY


def test_readiness_partial():
    from engine.readiness import compute_readiness
    from engine.models import Readiness
    r, note = compute_readiness({"1d": 30, "1h": 50, "15m": 50, "5m": 30, "1m": 30})
    assert r == Readiness.PARTIALLY_READY
    assert "Warming" in note


def test_readiness_full():
    from engine.readiness import compute_readiness
    from engine.models import Readiness
    r, note = compute_readiness({"1d": 100, "1h": 150, "15m": 200, "5m": 300, "1m": 200})
    assert r == Readiness.FULLY_READY
    assert "ready" in note.lower()


def test_readiness_penalty_partial():
    from engine.readiness import readiness_score_penalty
    from engine.models import Readiness
    from config.settings import PARTIALLY_READY_PENALTY
    assert readiness_score_penalty(Readiness.PARTIALLY_READY) == PARTIALLY_READY_PENALTY


def test_readiness_penalty_full_zero():
    from engine.readiness import readiness_score_penalty
    from engine.models import Readiness
    assert readiness_score_penalty(Readiness.FULLY_READY) == 0.0


def test_readiness_confidence_cap_partial():
    from engine.readiness import readiness_regime_confidence_cap
    from engine.models import Readiness
    assert readiness_regime_confidence_cap(Readiness.PARTIALLY_READY) == 0.65


def test_readiness_confidence_cap_not_ready():
    from engine.readiness import readiness_regime_confidence_cap
    from engine.models import Readiness
    assert readiness_regime_confidence_cap(Readiness.NOT_READY) == 0.0


def test_readiness_confidence_cap_full():
    from engine.readiness import readiness_regime_confidence_cap
    from engine.models import Readiness
    assert readiness_regime_confidence_cap(Readiness.FULLY_READY) is None


def test_readiness_describe_missing_ema200():
    from engine.readiness import describe_missing_indicators
    missing = describe_missing_indicators({"1h": 50, "15m": 250})
    assert "1h" in missing
    assert "EMA-200" in missing["1h"]
    assert "15m" not in missing


def test_readiness_describe_missing_rsi():
    from engine.readiness import describe_missing_indicators
    missing = describe_missing_indicators({"5m": 10})
    assert "RSI" in missing.get("5m", [])


def test_readiness_premarket_penalty_above():
    from engine.readiness import premarket_confidence_reduction
    assert premarket_confidence_reduction(0.25, 0.20, 0.15) == 0.15


def test_readiness_premarket_penalty_below():
    from engine.readiness import premarket_confidence_reduction
    assert premarket_confidence_reduction(0.10, 0.20, 0.15) == 0.0


def test_readiness_premarket_penalty_at_boundary():
    from engine.readiness import premarket_confidence_reduction
    # exactly at threshold: NOT above → no penalty
    assert premarket_confidence_reduction(0.20, 0.20, 0.15) == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# classify_session_type
# ═══════════════════════════════════════════════════════════════════════════

def test_session_regular():
    from engine.candle_store import classify_session_type
    assert classify_session_type(datetime(2025, 4, 9, 14, 30, tzinfo=timezone.utc)) == "regular"


def test_session_open():
    from engine.candle_store import classify_session_type
    # 13:30 UTC = 09:30 EDT
    assert classify_session_type(datetime(2025, 4, 9, 13, 30, tzinfo=timezone.utc)) == "regular"


def test_session_premarket():
    from engine.candle_store import classify_session_type
    # 09:00 UTC = 05:00 EDT
    assert classify_session_type(datetime(2025, 4, 9, 9, 0, tzinfo=timezone.utc)) == "premarket"


def test_session_afterhours():
    from engine.candle_store import classify_session_type
    # 21:00 UTC = 17:00 EDT
    assert classify_session_type(datetime(2025, 4, 9, 21, 0, tzinfo=timezone.utc)) == "afterhours"


def test_session_overnight():
    from engine.candle_store import classify_session_type
    # 03:00 UTC = 23:00 EDT previous day
    assert classify_session_type(datetime(2025, 4, 9, 3, 0, tzinfo=timezone.utc)) == "overnight"


# ═══════════════════════════════════════════════════════════════════════════
# IndicatorEngine
# ═══════════════════════════════════════════════════════════════════════════

def test_ie_not_ready_empty_state():
    from engine.candle_store import CandleStore
    from engine.indicator_engine import IndicatorEngine
    from engine.models import Readiness
    store  = CandleStore()
    engine = IndicatorEngine(store)
    state  = engine.compute()
    assert state.indicators == {}
    assert state.trends     == {}
    assert state.session.readiness == Readiness.NOT_READY


def test_ie_partial_ready_populates():
    from engine.candle_store import CandleStore
    from engine.indicator_engine import IndicatorEngine
    store = CandleStore()
    for i in range(35):
        b = _bar(_ts(10, i))
        arun(store.ingest_bar(b, "1h"))
        arun(store.ingest_bar(b, "15m"))
    engine = IndicatorEngine(store)
    state  = engine.compute()
    assert "1h"  in state.indicators
    assert "15m" in state.indicators


def test_ie_core_indicators_present():
    store  = _filled_store(60)
    from engine.indicator_engine import IndicatorEngine
    engine = IndicatorEngine(store)
    state  = engine.compute()
    snap   = state.indicators.get("15m")
    assert snap is not None
    assert snap.close    > 0
    assert snap.ema_9    is not None
    assert snap.rsi      is not None
    assert snap.atr      is not None
    assert snap.macd_hist is not None


def test_ie_ema200_none_with_50_candles():
    from engine.candle_store import CandleStore
    from engine.indicator_engine import IndicatorEngine
    store = CandleStore()
    for i in range(50):
        b = _bar(_ts(10, i))
        arun(store.ingest_bar(b, "15m"))
        arun(store.ingest_bar(b, "1h"))
    engine = IndicatorEngine(store)
    state  = engine.compute()
    snap   = state.indicators.get("15m")
    assert snap is not None
    assert snap.ema_200 is None


def test_ie_trend_bullish_on_rising_prices():
    store  = _filled_store(60, base=540.0)
    from engine.indicator_engine import IndicatorEngine
    from engine.models import Direction
    engine = IndicatorEngine(store)
    state  = engine.compute()
    any_bull = any(t.direction == Direction.BULLISH for t in state.trends.values())
    assert any_bull


def test_ie_ema_stack_not_bear_on_rising():
    store  = _filled_store(60, base=500.0)
    from engine.indicator_engine import IndicatorEngine
    engine = IndicatorEngine(store)
    state  = engine.compute()
    snap   = state.indicators.get("15m")
    assert snap is not None
    assert snap.ema_stack != "bear_aligned"


def test_ie_sr_levels_are_lists():
    store  = _filled_store(60)
    from engine.indicator_engine import IndicatorEngine
    engine = IndicatorEngine(store)
    state  = engine.compute()
    assert isinstance(state.support_levels,    list)
    assert isinstance(state.resistance_levels, list)


def test_ie_htf_bias_valid():
    from engine.candle_store import CandleStore
    from engine.indicator_engine import IndicatorEngine
    from engine.models import Direction
    store = CandleStore()
    for i in range(60):
        b = _bar(_ts(10, i), c=540.0 + i * 0.05)
        arun(store.ingest_bar(b, "1h"))
        arun(store.ingest_bar(b, "1d"))
        arun(store.ingest_bar(b, "15m"))
    engine = IndicatorEngine(store)
    state  = engine.compute()
    assert state.htf_bias in list(Direction)


def test_ie_session_mix_is_bool():
    from engine.candle_store import CandleStore
    from engine.indicator_engine import IndicatorEngine
    store = CandleStore()
    for i in range(10):
        b = _bar(_ts(5, i))
        arun(store.ingest_bar(b, "15m", session_type="premarket"))
        arun(store.ingest_bar(b, "1h",  session_type="premarket"))
    for i in range(40):
        b = _bar(_ts(10, i))
        arun(store.ingest_bar(b, "15m", session_type="regular"))
        arun(store.ingest_bar(b, "1h",  session_type="regular"))
    engine = IndicatorEngine(store)
    state  = engine.compute()
    snap   = state.indicators.get("15m")
    assert snap is not None
    assert isinstance(snap.session_mix, bool)


def test_ie_readiness_on_state():
    from engine.indicator_engine import IndicatorEngine
    from engine.models import Readiness
    store  = _filled_store(60)
    engine = IndicatorEngine(store)
    state  = engine.compute()
    assert state.session is not None
    assert state.session.readiness in list(Readiness)


def test_ie_state_json_serializable():
    import json
    from engine.indicator_engine import IndicatorEngine
    store  = _filled_store(60)
    engine = IndicatorEngine(store)
    state  = engine.compute()
    json.dumps(state.to_dict())  # must not raise


# ═══════════════════════════════════════════════════════════════════════════
# AlpacaFeed parsing helpers (no network)
# ═══════════════════════════════════════════════════════════════════════════

def test_alpaca_parse_ts_z():
    from engine.alpaca_feed import AlpacaFeed
    ts = AlpacaFeed._parse_ts("2025-04-09T14:30:00Z")
    assert ts.tzinfo is not None
    assert ts.hour == 14 and ts.minute == 30


def test_alpaca_parse_ts_offset():
    from engine.alpaca_feed import AlpacaFeed
    ts = AlpacaFeed._parse_ts("2025-04-09T14:30:00+00:00")
    assert ts.hour == 14


def test_alpaca_parse_bar_values():
    from engine.alpaca_feed import AlpacaFeed
    raw = {"t": "2025-04-09T14:30:00Z",
           "o": "545.10", "h": "546.50",
           "l": "544.80", "c": "546.00", "v": "125000"}
    bar = AlpacaFeed._parse_bar(raw)
    assert isinstance(bar["timestamp"], datetime)
    assert bar["open"]   == 545.10
    assert bar["high"]   == 546.50
    assert bar["volume"] == 125_000.0


def test_alpaca_parse_bar_all_keys():
    from engine.alpaca_feed import AlpacaFeed
    raw = {"t": "2025-04-09T14:30:00Z",
           "o": "1", "h": "2", "l": "0.5", "c": "1.5", "v": "99"}
    bar = AlpacaFeed._parse_bar(raw)
    for k in ("timestamp", "open", "high", "low", "close", "volume"):
        assert k in bar


# ═══════════════════════════════════════════════════════════════════════════
# MarketEngine event detection (no network)
# ═══════════════════════════════════════════════════════════════════════════

def test_me_instantiates():
    from engine.market_engine import MarketEngine
    from engine.models import Readiness
    eng = MarketEngine()
    assert eng.get_state()     is None
    assert eng.get_readiness() == Readiness.NOT_READY


def test_me_htf_bias_flip():
    from engine.market_engine import MarketEngine
    from engine.models import MarketState, Direction
    eng = MarketEngine()
    now = datetime.now(tz=timezone.utc)
    prev = MarketState(symbol="SPY", timestamp=now)
    curr = MarketState(symbol="SPY", timestamp=now)
    prev.htf_bias = Direction.BULLISH
    curr.htf_bias = Direction.BEARISH
    eng._prev_state = prev
    eng._state      = curr
    events = eng._detect_events([])
    htf = [e for e in events if e.event_type == "htf_bias_flip"]
    assert len(htf) == 1
    assert htf[0].payload["from"] == "bullish"
    assert htf[0].payload["to"]   == "bearish"


def test_me_trend_flip():
    from engine.market_engine import MarketEngine
    from engine.models import MarketState, TrendState, Direction
    eng = MarketEngine()
    now = datetime.now(tz=timezone.utc)
    prev = MarketState(symbol="SPY", timestamp=now)
    curr = MarketState(symbol="SPY", timestamp=now)
    prev.trends["15m"] = TrendState("15m", direction=Direction.BULLISH)
    curr.trends["15m"] = TrendState("15m", direction=Direction.BEARISH)
    eng._prev_state = prev
    eng._state      = curr
    flips = [e for e in eng._detect_events(["15m"]) if e.event_type == "trend_flip"]
    assert len(flips) == 1
    assert flips[0].payload["timeframe"] == "15m"


def test_me_weakening_event():
    from engine.market_engine import MarketEngine
    from engine.models import MarketState, TrendState, Direction
    eng = MarketEngine()
    now = datetime.now(tz=timezone.utc)
    prev = MarketState(symbol="SPY", timestamp=now)
    curr = MarketState(symbol="SPY", timestamp=now)
    prev.trends["1h"] = TrendState("1h", direction=Direction.BULLISH, is_weakening=False)
    curr.trends["1h"] = TrendState("1h", direction=Direction.BULLISH, is_weakening=True)
    eng._prev_state = prev
    eng._state      = curr
    weak = [e for e in eng._detect_events([]) if e.event_type == "trend_weakening"]
    assert len(weak) == 1


def test_me_candle_close_fires_for_5m_15m_1h():
    from engine.market_engine import MarketEngine
    from engine.models import MarketState
    eng = MarketEngine()
    now = datetime.now(tz=timezone.utc)
    eng._prev_state = MarketState(symbol="SPY", timestamp=now)
    eng._state      = MarketState(symbol="SPY", timestamp=now)
    events = eng._detect_events(["5m", "15m", "1h"])
    tfs = {e.payload["timeframe"] for e in events if e.event_type == "candle_close"}
    assert {"5m", "15m", "1h"} == tfs


def test_me_1m_close_not_an_event():
    from engine.market_engine import MarketEngine
    from engine.models import MarketState
    eng = MarketEngine()
    now = datetime.now(tz=timezone.utc)
    eng._prev_state = MarketState(symbol="SPY", timestamp=now)
    eng._state      = MarketState(symbol="SPY", timestamp=now)
    events = eng._detect_events(["1m"])
    close_1m = [e for e in events
                if e.event_type == "candle_close" and e.payload.get("timeframe") == "1m"]
    assert len(close_1m) == 0


def test_me_structure_break_no_duplicate():
    from engine.market_engine import MarketEngine
    from engine.models import MarketState, TrendState, Direction
    eng = MarketEngine()
    now = datetime.now(tz=timezone.utc)

    def _s(brk):
        s = MarketState(symbol="SPY", timestamp=now)
        t = TrendState("15m", direction=Direction.BULLISH)
        t.structure_break = brk
        s.trends["15m"] = t
        return s

    eng._prev_state = _s(False)
    eng._state      = _s(True)
    ev1 = [e for e in eng._detect_events([]) if e.event_type == "structure_break"]
    assert len(ev1) == 1

    eng._prev_state = eng._state
    eng._state      = _s(True)
    ev2 = [e for e in eng._detect_events([]) if e.event_type == "structure_break"]
    assert len(ev2) == 0   # already seen — no duplicate


def test_me_structure_break_re_fires_after_clear():
    from engine.market_engine import MarketEngine
    from engine.models import MarketState, TrendState, Direction
    eng = MarketEngine()
    now = datetime.now(tz=timezone.utc)

    def _s(brk):
        s = MarketState(symbol="SPY", timestamp=now)
        t = TrendState("15m", direction=Direction.BULLISH)
        t.structure_break = brk
        s.trends["15m"] = t
        return s

    eng._prev_state = _s(False); eng._state = _s(True)
    eng._detect_events([])                            # first fire
    eng._prev_state = eng._state; eng._state = _s(False)
    eng._detect_events([])                            # clear
    eng._prev_state = eng._state; eng._state = _s(True)
    ev = [e for e in eng._detect_events([]) if e.event_type == "structure_break"]
    assert len(ev) == 1                               # fires again


# ═══════════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    print(f"\nPhase 2 tests ({len(tests)} total)\n{'─'*52}")
    for name, fn in tests:
        run(name, fn)
    print(f"\n{'='*52}")
    print(f"{passed} passed, {failed} failed  ({passed+failed} total)")
    sys.exit(0 if failed == 0 else 1)