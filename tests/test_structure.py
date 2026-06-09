"""
tests/test_structure.py — Market Structure Engine v2.0A tests.

Coverage:
  detect_swing_points()      — pivot high/low detection
  classify_trend_structure() — HH/HL/LH/LL sequence + direction
  classify_market_state()    — trend/pullback/compression/range/breakout/breakdown
  detect_key_levels()        — session H/L, prior day H/L, VWAP
  detect_behavioral_events() — VWAP reclaim/rejection, failed breakdown/breakout,
                                liquidity sweep, trend exhaustion
  analyze()                  — full pipeline, fallback safety
  StructureState.to_dict()   — AI packaging format
  update_cycle integration   — structure_state set on RuntimeState
  AI state_packager          — structure included in package

Run: PYTHONPATH=. python tests/test_structure.py
"""

import os
import sys
import traceback
from datetime import datetime, timezone, timedelta
from typing import List

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
# Bar builders
# ──────────────────────────────────────────────

def _now():
    return datetime.now(tz=timezone.utc)


def _bar(close, high=None, low=None, open_=None, volume=100_000, i=0):
    h = high  if high  is not None else close + 0.5
    l = low   if low   is not None else close - 0.5
    o = open_ if open_ is not None else close - 0.1
    return {
        "timestamp": _now() - timedelta(minutes=(100 - i) * 5),
        "open": o, "high": h, "low": l, "close": close,
        "volume": float(volume), "vwap": close,
    }


def _bars_uptrend(start=540.0, step=0.3, n=40, atr=1.0) -> List[dict]:
    import math
    bars = []
    for i in range(n):
        mid = start + i * step + math.sin(i * 0.45) * atr * 1.1
        bars.append(_bar(round(mid, 2),
                         high=round(mid + atr * 0.55, 2),
                         low=round(mid  - atr * 0.55, 2), i=i))
    return bars


def _bars_downtrend(start=560.0, step=0.3, n=40, atr=1.0) -> List[dict]:
    import math
    bars = []
    for i in range(n):
        mid = start - i * step + math.sin(i * 0.45) * atr * 1.1
        bars.append(_bar(round(mid, 2),
                         high=round(mid + atr * 0.55, 2),
                         low=round(mid  - atr * 0.55, 2), i=i))
    return bars


def _bars_flat(price=545.0, n=30, noise=0.3) -> List[dict]:
    import math
    bars = []
    for i in range(n):
        mid = price + math.sin(i * 0.7) * noise
        bars.append(_bar(round(mid, 2), i=i))
    return bars


def _bars_compression(price=545.0, n=40) -> List[dict]:
    bars = []
    for i in range(n):
        atr = 2.0 if i < 25 else max(0.15, 2.0 * (0.7 ** (i - 25)))
        bars.append(_bar(price, high=round(price + atr, 2),
                         low=round(price - atr, 2), i=i))
    return bars


# ══════════════════════════════════════════════════════════════
# detect_swing_points
# ══════════════════════════════════════════════════════════════

def test_swing_detects_clear_high():
    from engine.structure_engine import detect_swing_points
    bars = [_bar(544, i=0), _bar(545, i=1), _bar(548, high=548.5, i=2),
            _bar(545, i=3), _bar(544, i=4), _bar(543, i=5)]
    highs, _ = detect_swing_points(bars, left=2, right=2)
    assert any(abs(h.price - 548.5) < 0.1 for h in highs)


def test_swing_detects_clear_low():
    from engine.structure_engine import detect_swing_points
    bars = [_bar(547, i=0), _bar(545, i=1), _bar(541, low=540.5, i=2),
            _bar(545, i=3), _bar(547, i=4), _bar(548, i=5)]
    _, lows = detect_swing_points(bars, left=2, right=2)
    assert any(abs(l.price - 540.5) < 0.1 for l in lows)


def test_swing_empty_bars():
    from engine.structure_engine import detect_swing_points
    h, l = detect_swing_points([])
    assert h == [] and l == []


def test_swing_too_few_bars():
    from engine.structure_engine import detect_swing_points
    bars = [_bar(545, i=i) for i in range(4)]
    h, l = detect_swing_points(bars, left=3, right=2)
    assert len(h) == 0 and len(l) == 0


def test_swing_flat_bars_no_clear_pivot():
    from engine.structure_engine import detect_swing_points
    bars = [_bar(545, high=546, low=544, i=i) for i in range(20)]
    h, _ = detect_swing_points(bars, left=3, right=2)
    assert len(h) == 0


def test_swing_uptrend_finds_multiple():
    from engine.structure_engine import detect_swing_points
    highs, lows = detect_swing_points(_bars_uptrend(n=40))
    assert len(highs) >= 2 and len(lows) >= 2


def test_swing_points_sorted_ascending():
    from engine.structure_engine import detect_swing_points
    highs, lows = detect_swing_points(_bars_uptrend(n=40))
    for i in range(1, len(highs)):
        assert highs[i].bar_index >= highs[i-1].bar_index


def test_swing_point_has_timestamp():
    from engine.structure_engine import detect_swing_points
    highs, _ = detect_swing_points(_bars_uptrend(n=40))
    if highs:
        assert highs[0].timestamp is not None


# ══════════════════════════════════════════════════════════════
# classify_trend_structure
# ══════════════════════════════════════════════════════════════

def test_trend_uptrend_hh_hl():
    from engine.structure_engine import detect_swing_points, classify_trend_structure
    highs, lows = detect_swing_points(_bars_uptrend(n=50))
    trend = classify_trend_structure(highs, lows)
    assert trend.direction == "uptrend"
    assert "HH" in trend.sequence and "HL" in trend.sequence


def test_trend_downtrend_lh_ll():
    from engine.structure_engine import detect_swing_points, classify_trend_structure
    highs, lows = detect_swing_points(_bars_downtrend(n=50))
    trend = classify_trend_structure(highs, lows)
    assert trend.direction == "downtrend"
    assert "LH" in trend.sequence and "LL" in trend.sequence


def test_trend_empty_swings_returns_ranging():
    from engine.structure_engine import classify_trend_structure
    trend = classify_trend_structure([], [])
    assert trend.direction == "ranging" and trend.sequence == []


def test_trend_sequence_limited_to_8():
    from engine.structure_engine import detect_swing_points, classify_trend_structure
    highs, lows = detect_swing_points(_bars_uptrend(n=80))
    trend = classify_trend_structure(highs, lows)
    assert len(trend.sequence) <= 8


def test_trend_intact_uptrend():
    from engine.structure_engine import detect_swing_points, classify_trend_structure
    highs, lows = detect_swing_points(_bars_uptrend(n=50))
    trend = classify_trend_structure(highs, lows)
    if trend.direction == "uptrend":
        assert trend.trend_intact is True


def test_trend_last_hh_positive():
    from engine.structure_engine import detect_swing_points, classify_trend_structure
    highs, lows = detect_swing_points(_bars_uptrend(n=50))
    trend = classify_trend_structure(highs, lows)
    if trend.direction == "uptrend" and trend.last_hh:
        assert trend.last_hh >= 540.0


def test_trend_last_hl_positive():
    from engine.structure_engine import detect_swing_points, classify_trend_structure
    highs, lows = detect_swing_points(_bars_uptrend(n=50))
    trend = classify_trend_structure(highs, lows)
    if trend.direction == "uptrend" and trend.last_hl:
        assert trend.last_hl > 0


def test_trend_sequence_str():
    from engine.structure_engine import TrendStructure
    t = TrendStructure("uptrend", ["HH","HL","HH","HL"], [], [])
    assert "HH" in t.sequence_str() and "HL" in t.sequence_str()


# ══════════════════════════════════════════════════════════════
# classify_market_state
# ══════════════════════════════════════════════════════════════

def test_market_state_uptrend():
    from engine.structure_engine import (
        detect_swing_points, classify_trend_structure,
        classify_market_state, KeyLevels, MarketStateLabel,
    )
    bars  = _bars_uptrend(n=50)
    highs, lows = detect_swing_points(bars)
    trend = classify_trend_structure(highs, lows)
    state = classify_market_state(bars, trend, 1.0, bars[-1]["close"], None, KeyLevels())
    assert state.state in (MarketStateLabel.TREND, MarketStateLabel.PULLBACK)


def test_market_state_compression():
    from engine.structure_engine import (
        detect_swing_points, classify_trend_structure,
        classify_market_state, KeyLevels, MarketStateLabel,
    )
    bars  = _bars_compression(n=42)
    highs, lows = detect_swing_points(bars)
    trend = classify_trend_structure(highs, lows)
    state = classify_market_state(bars, trend, 0.5, bars[-1]["close"], None, KeyLevels())
    assert state.state == MarketStateLabel.COMPRESSION


def test_market_state_breakout():
    from engine.structure_engine import (
        detect_swing_points, classify_trend_structure,
        classify_market_state, KeyLevels, MarketStateLabel,
    )
    bars    = _bars_uptrend(n=30)
    highs, lows = detect_swing_points(bars)
    trend   = classify_trend_structure(highs, lows)
    current = bars[-1]["close"]
    state   = classify_market_state(bars, trend, 1.0, current, None,
                                    KeyLevels(range_high=current - 0.5))
    assert state.state == MarketStateLabel.BREAKOUT


def test_market_state_breakdown():
    from engine.structure_engine import (
        detect_swing_points, classify_trend_structure,
        classify_market_state, KeyLevels, MarketStateLabel,
    )
    bars    = _bars_downtrend(n=30)
    highs, lows = detect_swing_points(bars)
    trend   = classify_trend_structure(highs, lows)
    current = bars[-1]["close"]
    state   = classify_market_state(bars, trend, 1.0, current, None,
                                    KeyLevels(range_low=current + 0.5))
    assert state.state == MarketStateLabel.BREAKDOWN


def test_market_state_insufficient_bars():
    from engine.structure_engine import (
        classify_market_state, KeyLevels, TrendStructure, MarketStateLabel,
    )
    bars  = [_bar(545, i=i) for i in range(4)]
    trend = TrendStructure("ranging", [], [], [])
    state = classify_market_state(bars, trend, 1.0, 545.0, None, KeyLevels())
    assert state.state == MarketStateLabel.RANGE
    assert state.confidence < 0.5


# ══════════════════════════════════════════════════════════════
# detect_key_levels
# ══════════════════════════════════════════════════════════════

def test_key_levels_vwap_stored():
    from engine.structure_engine import detect_key_levels
    levels = detect_key_levels([], [], vwap=543.5, now=_now())
    assert levels.vwap == 543.5


def test_key_levels_prior_day():
    from engine.structure_engine import detect_key_levels
    today = _now()
    daily = [
        {"timestamp": today - timedelta(days=3), "high": 548.0, "low": 539.0,
         "open": 540.0, "close": 547.0, "volume": 1e6, "vwap": 543.5},
        {"timestamp": today - timedelta(days=2), "high": 549.0, "low": 540.0,
         "open": 541.0, "close": 548.0, "volume": 1e6, "vwap": 544.5},
        {"timestamp": today - timedelta(days=1), "high": 550.0, "low": 541.0,
         "open": 542.0, "close": 549.0, "volume": 1e6, "vwap": 545.5},
    ]
    levels = detect_key_levels(daily, [], vwap=None, now=today)
    assert levels.prior_day_high == 550.0
    assert levels.prior_day_low  == 541.0


def test_key_levels_empty_safe():
    from engine.structure_engine import detect_key_levels
    levels = detect_key_levels([], [], vwap=None, now=_now())
    assert levels.session_high is None
    assert levels.prior_day_high is None


def test_key_levels_to_dict_excludes_none():
    from engine.structure_engine import KeyLevels
    kl = KeyLevels(vwap=543.5, session_high=546.0)
    d  = kl.to_dict()
    assert "vwap" in d and "session_high" in d
    assert "session_low" not in d


def test_key_levels_nearest_to():
    from engine.structure_engine import KeyLevels
    kl = KeyLevels(session_high=546.0, session_low=541.0, vwap=543.5)
    _, price = kl.nearest_to(543.0)
    assert price is not None and abs(price - 543.5) < 1.0


# ══════════════════════════════════════════════════════════════
# detect_behavioral_events
# ══════════════════════════════════════════════════════════════

def _vwap_reclaim_bars(vwap=543.5):
    bars = []
    for i in range(15):
        c = vwap - 1.0 if i < 8 else vwap + 0.5
        bars.append(_bar(c, high=c + 0.4, low=c - 0.4, i=i))
    return bars


def _vwap_rejection_bars(vwap=543.5):
    bars = []
    for i in range(12):
        c = vwap - 0.3 if i == 8 else vwap - 1.0
        h = vwap + 0.1 if i == 8 else c + 0.4
        bars.append(_bar(c, high=h, low=c - 0.4, i=i))
    return bars


def _failed_breakdown_bars(support=541.0, atr=1.0):
    bars = []
    for i in range(15):
        c = support - atr * 0.3 if i == 8 else (support + 0.5 if i > 8 else support + 1.0)
        bars.append(_bar(c, high=c + 0.4, low=c - 0.4, i=i))
    return bars


def _liquidity_sweep_bars(key_level=541.0, atr=1.0):
    bars = []
    for i in range(10):
        if i == 7:
            bars.append({
                "timestamp": _now() - timedelta(minutes=(10 - i) * 5),
                "open": key_level + 0.5, "high": key_level + 0.8,
                "low":  key_level - atr * 0.3,
                "close": key_level + 0.3,
                "volume": 500_000.0, "vwap": key_level,
            })
        else:
            c = key_level + 1.0
            bars.append(_bar(c, low=c - 0.4, i=i))
    return bars


def test_vwap_reclaim_detected():
    from engine.structure_engine import detect_behavioral_events, KeyLevels, BehavioralEventType
    vwap = 543.5
    events = detect_behavioral_events(
        _vwap_reclaim_bars(vwap), KeyLevels(vwap=vwap), vwap, 1.0
    )
    assert BehavioralEventType.VWAP_RECLAIM in [e.event_type for e in events]


def test_vwap_reclaim_bars_ago_positive():
    from engine.structure_engine import detect_behavioral_events, KeyLevels, BehavioralEventType
    vwap = 543.5
    events = detect_behavioral_events(
        _vwap_reclaim_bars(vwap), KeyLevels(vwap=vwap), vwap, 1.0
    )
    reclaims = [e for e in events if e.event_type == BehavioralEventType.VWAP_RECLAIM]
    assert reclaims[0].bars_ago >= 1


def test_vwap_rejection_detected():
    from engine.structure_engine import detect_behavioral_events, KeyLevels, BehavioralEventType
    vwap = 543.5
    events = detect_behavioral_events(
        _vwap_rejection_bars(vwap), KeyLevels(vwap=vwap), vwap, 1.0
    )
    assert BehavioralEventType.VWAP_REJECTION in [e.event_type for e in events]


def test_failed_breakdown_detected():
    from engine.structure_engine import detect_behavioral_events, KeyLevels, BehavioralEventType
    support = 541.0
    events  = detect_behavioral_events(
        _failed_breakdown_bars(support), KeyLevels(session_low=support, range_low=support), None, 1.0
    )
    assert BehavioralEventType.FAILED_BREAKDOWN in [e.event_type for e in events]


def test_liquidity_sweep_detected():
    from engine.structure_engine import detect_behavioral_events, KeyLevels, BehavioralEventType
    key_level = 541.0
    events    = detect_behavioral_events(
        _liquidity_sweep_bars(key_level, 1.0), KeyLevels(session_low=key_level), None, 1.0
    )
    assert BehavioralEventType.LIQUIDITY_SWEEP_LOW in [e.event_type for e in events]


def test_behavioral_empty_bars_safe():
    from engine.structure_engine import detect_behavioral_events, KeyLevels
    assert detect_behavioral_events([], KeyLevels(), None, 1.0) == []


def test_behavioral_events_have_detail():
    from engine.structure_engine import detect_behavioral_events, KeyLevels
    events = detect_behavioral_events(
        _vwap_reclaim_bars(543.5), KeyLevels(vwap=543.5), 543.5, 1.0
    )
    for e in events:
        assert isinstance(e.detail, str) and len(e.detail) > 0


# ══════════════════════════════════════════════════════════════
# Full analyze() pipeline
# ══════════════════════════════════════════════════════════════

class _FakeMS:
    def __init__(self, bars_5m, bars_1d=None, vwap=543.5, atr=1.45, close=545.0):
        self.symbol    = "SPY"
        self.timestamp = _now()
        self.bars      = {"5m": bars_5m, "1d": bars_1d or [], "15m": [], "1h": [], "1m": []}
        self.indicators = {}
        self.trends     = {}
        class _Ind:
            pass
        ind = _Ind(); ind.vwap = vwap; ind.atr = atr; ind.close = close
        self.indicators["5m"]  = ind
        self.indicators["15m"] = ind


def test_analyze_uptrend():
    from engine.structure_engine import analyze
    result = analyze(_FakeMS(_bars_uptrend(n=50)))
    assert result.trend.direction in ("uptrend", "ranging")
    assert result.market_state.state.value in ("trend","pullback","range","compression")


def test_analyze_returns_structure_state():
    from engine.structure_engine import analyze, StructureState
    assert isinstance(analyze(_FakeMS(_bars_uptrend(n=40))), StructureState)


def test_analyze_narrative_not_empty():
    from engine.structure_engine import analyze
    result = analyze(_FakeMS(_bars_uptrend(n=50)))
    assert len(result.structure_narrative)  > 0
    assert len(result.buyer_context)        > 0
    assert len(result.seller_context)       > 0
    assert len(result.missing_confirmation) > 0


def test_analyze_without_bars_fallback():
    from engine.structure_engine import analyze, StructureState
    from runtime.update_cycle import build_demo_state, reset_engine
    reset_engine()
    state = build_demo_state()
    del state.bars
    result = analyze(state)
    assert isinstance(result, StructureState) and result.symbol == "SPY"


def test_analyze_empty_state_safe():
    from engine.structure_engine import analyze, StructureState
    class Empty:
        symbol = "SPY"; timestamp = _now(); bars = {}; indicators = {}; trends = {}
    result = analyze(Empty())
    assert isinstance(result, StructureState)


def test_analyze_never_raises():
    from engine.structure_engine import analyze
    for bad in [None, object(), 42, "string"]:
        try:
            r = analyze(bad)
            assert r is not None
        except Exception as e:
            assert False, f"analyze raised on {bad!r}: {e}"


# ══════════════════════════════════════════════════════════════
# StructureState.to_dict()
# ══════════════════════════════════════════════════════════════

def test_to_dict_required_keys():
    from engine.structure_engine import analyze
    from runtime.update_cycle import build_demo_state, reset_engine
    reset_engine()
    d = analyze(build_demo_state()).to_dict()
    for key in ("direction","sequence","market_state","buyer_context",
                "seller_context","missing_confirmation","structure_narrative","key_levels"):
        assert key in d, f"Missing: {key}"


def test_to_dict_key_levels_dict():
    from engine.structure_engine import analyze
    from runtime.update_cycle import build_demo_state, reset_engine
    reset_engine()
    assert isinstance(analyze(build_demo_state()).to_dict()["key_levels"], dict)


def test_to_dict_recent_events_list():
    from engine.structure_engine import analyze
    from runtime.update_cycle import build_demo_state, reset_engine
    reset_engine()
    d = analyze(build_demo_state()).to_dict()
    assert isinstance(d.get("recent_events", []), list)


# ══════════════════════════════════════════════════════════════
# Integration: update_cycle + RuntimeState
# ══════════════════════════════════════════════════════════════

def test_update_cycle_sets_structure_state():
    from runtime.runtime_state import RuntimeState
    from runtime.update_cycle import run_update_cycle, reset_engine
    reset_engine()
    s = RuntimeState()
    run_update_cycle(s)
    assert s.structure_state is not None


def test_update_cycle_structure_has_direction():
    from runtime.runtime_state import RuntimeState
    from runtime.update_cycle import run_update_cycle, reset_engine
    reset_engine()
    s = RuntimeState()
    run_update_cycle(s)
    assert s.structure_state.trend.direction in ("uptrend","downtrend","ranging")


def test_update_cycle_structure_has_narrative():
    from runtime.runtime_state import RuntimeState
    from runtime.update_cycle import run_update_cycle, reset_engine
    reset_engine()
    s = RuntimeState()
    run_update_cycle(s)
    assert len(s.structure_state.structure_narrative) > 0


def test_structure_failure_does_not_halt_cycle():
    import unittest.mock as mock
    from runtime.runtime_state import RuntimeState
    from runtime.update_cycle import run_update_cycle, reset_engine
    reset_engine()
    s = RuntimeState()
    with mock.patch("runtime.update_cycle._run_structure_engine",
                    side_effect=RuntimeError("structure exploded")):
        ok = run_update_cycle(s)
    assert ok is True and s.market_state is not None


def test_build_demo_state_has_bars():
    from runtime.update_cycle import build_demo_state, reset_engine
    reset_engine()
    state = build_demo_state()
    bars  = getattr(state, "bars", None)
    assert bars is not None
    for tf in ("5m", "15m", "1d"):
        assert tf in bars
    assert len(bars["5m"]) >= 30


def test_demo_bars_have_ohlcv_keys():
    from runtime.update_cycle import build_demo_state, reset_engine
    reset_engine()
    for bar in build_demo_state().bars["5m"][:3]:
        for key in ("timestamp","open","high","low","close","volume","vwap"):
            assert key in bar, f"Missing bar key: {key}"


def test_demo_bars_produce_uptrend():
    from engine.structure_engine import analyze
    from runtime.update_cycle import build_demo_state, reset_engine
    reset_engine()
    result = analyze(build_demo_state())
    assert result.trend.direction == "uptrend"
    assert len(result.trend.swing_highs) >= 2
    assert len(result.trend.swing_lows)  >= 2


# ══════════════════════════════════════════════════════════════
# Integration: AI packaging
# ══════════════════════════════════════════════════════════════

def test_state_packager_includes_structure():
    from runtime.update_cycle import build_demo_state, reset_engine
    from engine.structure_engine import analyze
    from ai.state_packager import build_full_package
    reset_engine()
    state  = build_demo_state()
    struct = analyze(state)
    pkg    = build_full_package(state=state, structure_state=struct,
                                request_type="market_commentary")
    d = pkg.to_dict()
    assert "structure" in d
    assert d["structure"]["direction"] in ("uptrend","downtrend","ranging")


def test_state_packager_none_structure_safe():
    from runtime.update_cycle import build_demo_state, reset_engine
    from ai.state_packager import build_full_package
    reset_engine()
    pkg = build_full_package(state=build_demo_state(), structure_state=None)
    d   = pkg.to_dict()
    assert "structure" not in d


def test_structure_package_has_narrative():
    from runtime.update_cycle import build_demo_state, reset_engine
    from engine.structure_engine import analyze
    from ai.state_packager import build_full_package
    reset_engine()
    state  = build_demo_state()
    pkg    = build_full_package(state=state, structure_state=analyze(state))
    d      = pkg.to_dict()
    assert len(d["structure"]["buyer_context"])        > 0
    assert len(d["structure"]["seller_context"])       > 0
    assert len(d["structure"]["missing_confirmation"]) > 0


def test_commentary_cycle_uses_structure_state():
    """structure_state from update_cycle is accessible in generate_commentary."""
    from runtime.runtime_state import RuntimeState
    from runtime.update_cycle import run_update_cycle, reset_engine
    from runtime.commentary_cycle import generate_commentary, TRIGGER_PERIODIC
    reset_engine()
    s = RuntimeState(mock_ai=True)
    s.mark_started()
    run_update_cycle(s)
    assert s.structure_state is not None
    result = generate_commentary(s, trigger=TRIGGER_PERIODIC)
    assert result is None or isinstance(result, str)


# ══════════════════════════════════════════════════════════════
# Runner
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    tests = [
        (n, f) for n, f in sorted(globals().items())
        if n.startswith("test_") and callable(f)
    ]
    print(f"\nStructure engine tests ({len(tests)} total)\n{'─' * 60}")
    for name, fn in tests:
        run(name, fn)
    print(f"\n{'=' * 60}")
    print(f"{passed} passed, {failed} failed  ({passed + failed} total)")
    import sys; sys.exit(0 if failed == 0 else 1)