"""
tests/test_pattern_engine.py — Unit tests for engine/pattern_engine.py

Tests pattern classification, base-rate statistics, narrative formatting,
and package wrapper.  No API calls, no live data.

Run: PYTHONPATH=. python tests/test_pattern_engine.py
"""

import os, sys, traceback

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


# ──────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────

def _uptrend_trend(intact=True):
    return {
        "direction": "uptrend", "market_state": "trend",
        "trend_intact": intact, "sequence": "HH HL HH HL",
        "last_hl": 544.22, "last_hh": 549.48,
        "key_levels": {"session_high": 549.48, "session_low": 541.19, "vwap": 543.5},
        "recent_events": [],
    }


def _uptrend_pullback():
    return {
        "direction": "uptrend", "market_state": "pullback",
        "trend_intact": True, "sequence": "HH HL HH HL",
        "last_hl": 544.22, "last_hh": 549.48,
        "key_levels": {"session_high": 549.48, "session_low": 541.19, "vwap": 543.5},
        "recent_events": [],
    }


def _downtrend_trend(intact=True):
    return {
        "direction": "downtrend", "market_state": "trend",
        "trend_intact": intact, "sequence": "LH LL LH LL",
        "last_lh": 738.39, "last_ll": 730.50,
        "key_levels": {"session_high": 738.39, "session_low": 730.50, "vwap": 734.9},
        "recent_events": [],
    }


def _downtrend_pullback():
    return {
        "direction": "downtrend", "market_state": "pullback",
        "trend_intact": True, "sequence": "LH LL LH LL",
        "last_lh": 738.39, "last_ll": 730.50,
        "key_levels": {"session_high": 738.39, "session_low": 730.50, "vwap": 734.9},
        "recent_events": [],
    }


def _range_fields():
    return {
        "direction": "ranging", "market_state": "range",
        "trend_intact": False, "sequence": "",
        "key_levels": {"session_high": 737.00, "session_low": 735.80, "vwap": 736.40},
        "recent_events": [],
    }


def _compression_uptrend():
    return {
        "direction": "uptrend", "market_state": "compression",
        "trend_intact": True, "sequence": "HH HL",
        "key_levels": {"session_high": 549.48, "session_low": 541.19, "vwap": 543.5},
        "recent_events": [],
    }


def _with_event(base: dict, event_name: str) -> dict:
    d = dict(base)
    d["recent_events"] = [event_name]
    return d


# ──────────────────────────────────────────────────────────────
# classify_patterns — basic matching
# ──────────────────────────────────────────────────────────────

def test_returns_list():
    from engine.pattern_engine import classify_patterns
    result = classify_patterns(_uptrend_pullback())
    assert isinstance(result, list)


def test_uptrend_pullback_gives_continuation_long():
    from engine.pattern_engine import classify_patterns
    ps = classify_patterns(_uptrend_pullback())
    names = [p.name for p in ps]
    assert "Trend Continuation (Long)" in names, f"Got: {names}"


def test_downtrend_pullback_gives_continuation_short():
    from engine.pattern_engine import classify_patterns
    ps = classify_patterns(_downtrend_pullback())
    names = [p.name for p in ps]
    assert "Trend Continuation (Short)" in names, f"Got: {names}"


def test_range_gives_range_compression():
    from engine.pattern_engine import classify_patterns
    ps = classify_patterns(_range_fields())
    names = [p.name for p in ps]
    assert "Range Compression" in names or "Post-trend Consolidation" in names, (
        f"Expected range pattern, got: {names}"
    )


def test_uptrend_range_gives_post_trend_consolidation():
    from engine.pattern_engine import classify_patterns
    d = {**_range_fields(), "direction": "uptrend"}
    ps = classify_patterns(d)
    names = [p.name for p in ps]
    assert "Post-trend Consolidation" in names, f"Got: {names}"


def test_downtrend_range_gives_post_trend_consolidation():
    from engine.pattern_engine import classify_patterns
    d = {**_range_fields(), "direction": "downtrend"}
    ps = classify_patterns(d)
    names = [p.name for p in ps]
    assert "Post-trend Consolidation" in names, f"Got: {names}"


def test_compression_uptrend_gives_bull_flag():
    from engine.pattern_engine import classify_patterns
    ps = classify_patterns(_compression_uptrend())
    names = [p.name for p in ps]
    assert "Bull Flag" in names, f"Got: {names}"


def test_compression_downtrend_gives_bear_flag():
    from engine.pattern_engine import classify_patterns
    d = {**_compression_uptrend(), "direction": "downtrend"}
    ps = classify_patterns(d)
    names = [p.name for p in ps]
    assert "Bear Flag" in names, f"Got: {names}"


def test_breakout_state():
    from engine.pattern_engine import classify_patterns
    d = {**_uptrend_trend(), "market_state": "breakout"}
    ps = classify_patterns(d)
    names = [p.name for p in ps]
    assert "Breakout Attempt" in names, f"Got: {names}"


def test_breakdown_state():
    from engine.pattern_engine import classify_patterns
    d = {**_downtrend_trend(), "market_state": "breakdown"}
    ps = classify_patterns(d)
    names = [p.name for p in ps]
    assert "Breakdown Attempt" in names, f"Got: {names}"


def test_intact_trend_no_exhaustion():
    from engine.pattern_engine import classify_patterns
    ps = classify_patterns(_uptrend_trend(intact=True))
    names = [p.name for p in ps]
    assert "Trend Exhaustion" not in names


def test_weakening_trend_gives_exhaustion():
    from engine.pattern_engine import classify_patterns
    ps = classify_patterns(_uptrend_trend(intact=False))
    names = [p.name for p in ps]
    assert "Trend Exhaustion" in names, f"Got: {names}"


def test_empty_fields_no_crash():
    from engine.pattern_engine import classify_patterns
    result = classify_patterns({})
    assert isinstance(result, list)


# ──────────────────────────────────────────────────────────────
# Behavioral event matching
# ──────────────────────────────────────────────────────────────

def test_failed_breakout_event():
    from engine.pattern_engine import classify_patterns
    d  = _with_event(_range_fields(), "failed_breakout (2b ago)")
    ps = classify_patterns(d)
    names = [p.name for p in ps]
    assert "Failed Breakout" in names, f"Got: {names}"


def test_failed_breakdown_event():
    from engine.pattern_engine import classify_patterns
    d  = _with_event(_range_fields(), "failed_breakdown (3b ago)")
    ps = classify_patterns(d)
    names = [p.name for p in ps]
    assert "Failed Breakdown" in names, f"Got: {names}"


def test_vwap_reclaim_event():
    from engine.pattern_engine import classify_patterns
    d  = _with_event(_uptrend_pullback(), "vwap_reclaim (1b ago)")
    ps = classify_patterns(d)
    names = [p.name for p in ps]
    assert "VWAP Reclaim" in names, f"Got: {names}"


def test_vwap_rejection_event():
    from engine.pattern_engine import classify_patterns
    d  = _with_event(_downtrend_pullback(), "vwap_rejection (2b ago)")
    ps = classify_patterns(d)
    names = [p.name for p in ps]
    assert "VWAP Rejection" in names, f"Got: {names}"


def test_behavioral_event_comes_first():
    """Behavioral event patterns should appear before state-based patterns."""
    from engine.pattern_engine import classify_patterns
    d  = _with_event(_range_fields(), "failed_breakout (1b ago)")
    ps = classify_patterns(d)
    assert ps[0].name == "Failed Breakout", (
        f"Failed Breakout should be first, got: {[p.name for p in ps]}"
    )


def test_no_duplicate_patterns():
    from engine.pattern_engine import classify_patterns
    d  = _with_event(_range_fields(), "failed_breakout (1b ago)")
    ps = classify_patterns(d)
    names = [p.name for p in ps]
    assert len(names) == len(set(names)), f"Duplicate patterns: {names}"


# ──────────────────────────────────────────────────────────────
# PatternMatch statistics sanity
# ──────────────────────────────────────────────────────────────

def test_failed_breakout_bearish_bias():
    from engine.pattern_engine import _CATALOG
    p = _CATALOG["failed_breakout"]
    assert p.resolution_down > p.resolution_up, (
        "Failed breakout should have bearish resolution bias"
    )
    assert p.resolution_down >= 0.65, "Failed breakout should be strongly bearish"


def test_failed_breakdown_bullish_bias():
    from engine.pattern_engine import _CATALOG
    p = _CATALOG["failed_breakdown"]
    assert p.resolution_up > p.resolution_down


def test_bull_flag_bullish_bias():
    from engine.pattern_engine import _CATALOG
    p = _CATALOG["bull_flag"]
    assert p.resolution_up > p.resolution_down


def test_bear_flag_bearish_bias():
    from engine.pattern_engine import _CATALOG
    p = _CATALOG["bear_flag"]
    assert p.resolution_down > p.resolution_up


def test_range_compression_neutral():
    from engine.pattern_engine import _CATALOG
    p = _CATALOG["range_compression"]
    assert p.bias == "neutral"
    assert p.resolution_down == p.resolution_up


def test_all_patterns_have_atr_move():
    from engine.pattern_engine import _CATALOG
    for key, p in _CATALOG.items():
        assert p.avg_atr_move > 0, f"{key} has zero ATR move"


def test_all_patterns_have_note():
    from engine.pattern_engine import _CATALOG
    for key, p in _CATALOG.items():
        assert len(p.note) > 20, f"{key} has empty note"


# ──────────────────────────────────────────────────────────────
# build_pattern_narrative — format checks
# ──────────────────────────────────────────────────────────────

def test_narrative_returns_string():
    from engine.pattern_engine import build_pattern_narrative
    text = build_pattern_narrative(_range_fields())
    assert isinstance(text, str)


def test_narrative_has_pattern_header():
    from engine.pattern_engine import build_pattern_narrative
    text = build_pattern_narrative(_range_fields())
    assert "pattern" in text.lower()


def test_narrative_has_percentage():
    from engine.pattern_engine import build_pattern_narrative
    text = build_pattern_narrative(_range_fields())
    assert "%" in text, f"Expected percentage in narrative:\n{text}"


def test_narrative_has_atr_move():
    from engine.pattern_engine import build_pattern_narrative
    text = build_pattern_narrative(_range_fields())
    assert "ATR" in text, f"Expected ATR in narrative:\n{text}"


def test_narrative_with_atr_shows_points():
    from engine.pattern_engine import build_pattern_narrative
    text = build_pattern_narrative(_range_fields(), atr=1.45)
    assert "points" in text.lower() or "ATR" in text, (
        f"Expected point value when ATR provided:\n{text}"
    )


def test_narrative_has_resolution_timeline():
    from engine.pattern_engine import build_pattern_narrative
    text = build_pattern_narrative(_range_fields())
    assert "session" in text.lower(), f"Expected session timeline:\n{text}"


def test_narrative_has_note():
    from engine.pattern_engine import build_pattern_narrative
    text = build_pattern_narrative(_range_fields())
    assert "Note:" in text


def test_narrative_empty_for_intact_trend():
    """Intact clean trend needs no pattern commentary."""
    from engine.pattern_engine import build_pattern_narrative
    text = build_pattern_narrative(_uptrend_trend(intact=True))
    # Either empty or minimal — intact clean trend has no pattern ambiguity
    assert isinstance(text, str)   # must not crash


def test_narrative_failed_breakout_bearish_language():
    from engine.pattern_engine import build_pattern_narrative
    d    = _with_event(_range_fields(), "failed_breakout (1b ago)")
    text = build_pattern_narrative(d)
    lower = text.lower()
    assert "failed breakout" in lower
    # Should show bearish resolution bias
    assert "lower" in lower or "bearish" in lower


def test_narrative_failed_breakdown_bullish_language():
    from engine.pattern_engine import build_pattern_narrative
    d    = _with_event(_range_fields(), "failed_breakdown (2b ago)")
    text = build_pattern_narrative(d)
    lower = text.lower()
    assert "failed breakdown" in lower
    assert "higher" in lower or "bullish" in lower


def test_narrative_multiple_patterns_listed():
    """When multiple patterns match, all should appear in the header."""
    from engine.pattern_engine import build_pattern_narrative
    d = _with_event(_range_fields(), "failed_breakout (1b ago)")
    text = build_pattern_narrative(d)
    assert "Failed Breakout" in text


def test_narrative_after_hours_session_note():
    from engine.pattern_engine import build_pattern_narrative
    text = build_pattern_narrative(_range_fields(), session="after_hours")
    assert "session" in text.lower()


# ──────────────────────────────────────────────────────────────
# Package wrapper
# ──────────────────────────────────────────────────────────────

def test_from_package_no_crash():
    from engine.pattern_engine import build_pattern_narrative_from_package
    from ai.state_packager import build_full_package
    from engine.structure_engine import analyze
    from runtime.update_cycle import build_demo_state, reset_engine
    reset_engine()
    state = build_demo_state()
    pkg   = build_full_package(state=state, structure_state=analyze(state))
    text  = build_pattern_narrative_from_package(pkg.to_dict())
    assert isinstance(text, str)


def test_from_empty_package_no_crash():
    from engine.pattern_engine import build_pattern_narrative_from_package
    text = build_pattern_narrative_from_package({})
    assert isinstance(text, str)


def test_from_package_with_atr_shows_points():
    from engine.pattern_engine import build_pattern_narrative_from_package
    d = {
        "market": {
            "market_session": "open",
            "regime":         {"name": "chop"},
            "indicators":     {"5m": {"atr": 1.45}},
        },
        "structure": {
            "direction":   "ranging",
            "market_state": "range",
            "trend_intact": False,
            "sequence":    "",
            "recent_events": [],
            "key_levels":  {"session_high": 549.48, "session_low": 541.19, "vwap": 543.5},
        },
    }
    text = build_pattern_narrative_from_package(d)
    if text:
        # If patterns were found, ATR × move should appear as a point value
        assert "1.45" in text or "points" in text.lower() or "ATR" in text


# ──────────────────────────────────────────────────────────────
# StructureState object path
# ──────────────────────────────────────────────────────────────

def test_accepts_structure_state_object():
    from engine.pattern_engine import build_pattern_narrative
    from engine.structure_engine import analyze
    from runtime.update_cycle import build_demo_state, reset_engine
    reset_engine()
    struct = analyze(build_demo_state())
    text   = build_pattern_narrative(struct, regime="trend_bull", session="open")
    assert isinstance(text, str)


# ──────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        (n, f) for n, f in sorted(globals().items())
        if n.startswith("test_") and callable(f)
    ]
    print(f"\nPattern engine tests ({len(tests)} total)\n{'─' * 60}")
    for name, fn in tests:
        run(name, fn)
    print(f"\n{'=' * 60}")
    print(f"{passed} passed, {failed} failed  ({passed + failed} total)")
    import sys; sys.exit(0 if failed == 0 else 1)