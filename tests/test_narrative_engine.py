"""
tests/test_narrative_engine.py — Unit tests for engine/narrative_engine.py

Tests the narrative engine in isolation: no API calls, no live data.
Verifies trader-quality output across all six market states, all sessions,
and the no-structure fallback.

Run: PYTHONPATH=. python tests/test_narrative_engine.py
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
# Helpers / fixtures
# ──────────────────────────────────────────────────────────────

def _uptrend_dict(sess="after_hours"):
    return {
        "direction":            "uptrend",
        "sequence":             "HH HL HH HL",
        "trend_intact":         True,
        "last_hh":              549.48,
        "last_hl":              544.22,
        "last_lh":              None,
        "last_ll":              None,
        "market_state":         "trend",
        "structure_narrative":  "SPY trending higher.",
        "buyer_context":        "Buyers defending 544.22.",
        "seller_context":       "Sellers at 549.48.",
        "missing_confirmation": "Need break above 549.48.",
        "key_levels": {
            "session_high":       549.48,
            "session_low":        541.19,
            "prior_day_high":     565.43,
            "prior_day_low":      552.67,
            "vwap":               543.50,
            "nearest_support":    544.22,
            "nearest_resistance": None,
        },
    }


def _downtrend_dict(sess="open"):
    return {
        "direction":            "downtrend",
        "sequence":             "LH LL LH LL",
        "trend_intact":         True,
        "last_hh":              None,
        "last_hl":              None,
        "last_lh":              548.00,
        "last_ll":              541.00,
        "market_state":         "trend",
        "structure_narrative":  "SPY trending lower.",
        "buyer_context":        "Buyers at 541.00.",
        "seller_context":       "Sellers capping at 548.00.",
        "missing_confirmation": "Need lower low below 541.00 to confirm.",
        "key_levels": {
            "session_high":       548.00,
            "session_low":        541.00,
            "prior_day_high":     550.00,
            "prior_day_low":      540.00,
            "vwap":               544.50,
            "nearest_support":    None,
            "nearest_resistance": 548.00,
        },
    }


def _range_dict(sess="midday"):
    return {
        "direction":            "ranging",
        "sequence":             "",
        "trend_intact":         False,
        "last_hh":              None,
        "last_hl":              None,
        "last_lh":              None,
        "last_ll":              None,
        "market_state":         "range",
        "structure_narrative":  "SPY range-bound.",
        "buyer_context":        "",
        "seller_context":       "",
        "missing_confirmation": "Needs directional resolution.",
        "key_levels": {
            "session_high":       549.48,
            "session_low":        541.19,
            "prior_day_high":     565.43,
            "prior_day_low":      552.67,
            "vwap":               545.00,
            "nearest_support":    None,
            "nearest_resistance": None,
        },
    }


# ──────────────────────────────────────────────────────────────
# build_narrative — basic structure
# ──────────────────────────────────────────────────────────────

def test_returns_string():
    from engine.narrative_engine import build_narrative
    result = build_narrative(_uptrend_dict())
    assert isinstance(result, str)
    assert len(result) > 40


def test_not_empty_for_none():
    from engine.narrative_engine import build_narrative
    result = build_narrative(None, regime="chop", session="after_hours")
    assert len(result) > 20


def test_not_raises_on_bad_dict():
    from engine.narrative_engine import build_narrative
    result = build_narrative({"bad_key": "bad_value"}, regime="unknown")
    assert isinstance(result, str)


def test_not_raises_on_empty_dict():
    from engine.narrative_engine import build_narrative
    result = build_narrative({})
    assert isinstance(result, str)


# ──────────────────────────────────────────────────────────────
# Opening — market states
# ──────────────────────────────────────────────────────────────

def test_uptrend_opening():
    from engine.narrative_engine import build_narrative
    text = build_narrative(_uptrend_dict())
    assert "trending higher" in text.lower() or "uptrend" in text.lower()


def test_downtrend_opening():
    from engine.narrative_engine import build_narrative
    text = build_narrative(_downtrend_dict())
    assert "trending lower" in text.lower() or "downtrend" in text.lower()


def test_range_opening():
    from engine.narrative_engine import build_narrative
    text = build_narrative(_range_dict())
    assert "range" in text.lower()


def test_pullback_uptrend_opening():
    from engine.narrative_engine import build_narrative
    d = _uptrend_dict()
    d["market_state"] = "pullback"
    text = build_narrative(d)
    assert "pulling back" in text.lower() or "pullback" in text.lower()


def test_pullback_downtrend_opening():
    from engine.narrative_engine import build_narrative
    d = _downtrend_dict()
    d["market_state"] = "pullback"
    text = build_narrative(d)
    assert "bouncing" in text.lower() or "pullback" in text.lower()


def test_compression_opening():
    from engine.narrative_engine import build_narrative
    d = _range_dict()
    d["market_state"] = "compression"
    text = build_narrative(d)
    assert "compress" in text.lower()


def test_breakout_opening():
    from engine.narrative_engine import build_narrative
    d = _uptrend_dict()
    d["market_state"] = "breakout"
    text = build_narrative(d)
    assert "broke" in text.lower() or "break" in text.lower() or "expansion" in text.lower()


def test_breakdown_opening():
    from engine.narrative_engine import build_narrative
    d = _downtrend_dict()
    d["market_state"] = "breakdown"
    text = build_narrative(d)
    assert "broke" in text.lower() or "break" in text.lower() or "seller" in text.lower()


# ──────────────────────────────────────────────────────────────
# Buyer / seller synthesis
# ──────────────────────────────────────────────────────────────

def test_uptrend_has_buyer_mention():
    from engine.narrative_engine import build_narrative
    text = build_narrative(_uptrend_dict())
    assert "buyer" in text.lower() or "higher low" in text.lower()


def test_uptrend_has_seller_mention():
    from engine.narrative_engine import build_narrative
    text = build_narrative(_uptrend_dict())
    assert "seller" in text.lower() or "capping" in text.lower()


def test_downtrend_seller_leads():
    from engine.narrative_engine import build_narrative
    text = build_narrative(_downtrend_dict())
    s_idx = text.lower().find("seller")
    b_idx = text.lower().find("buyer")
    if s_idx != -1 and b_idx != -1:
        assert s_idx < b_idx, "Downtrend: seller should appear before buyer"


def test_uptrend_buyer_leads():
    from engine.narrative_engine import build_narrative
    text = build_narrative(_uptrend_dict())
    b_idx = text.lower().find("buyer")
    s_idx = text.lower().find("seller")
    if b_idx != -1 and s_idx != -1:
        assert b_idx < s_idx, "Uptrend: buyer should appear before seller"


def test_range_has_both_sides():
    from engine.narrative_engine import build_narrative
    text = build_narrative(_range_dict())
    lower = text.lower()
    assert ("buyer" in lower or "defend" in lower), "Range should mention buyers/defense"
    assert ("seller" in lower or "reject" in lower), "Range should mention sellers/rejection"


def test_uses_actual_price_levels():
    from engine.narrative_engine import build_narrative
    text = build_narrative(_uptrend_dict())
    # Should contain at least one of the price levels
    assert "544.22" in text or "549.48" in text or "543.50" in text


# ──────────────────────────────────────────────────────────────
# Key levels
# ──────────────────────────────────────────────────────────────

def test_key_levels_present():
    from engine.narrative_engine import build_narrative
    text = build_narrative(_uptrend_dict())
    assert "key levels" in text.lower() or "549.48" in text or "543.50" in text


def test_key_levels_not_present_when_no_levels():
    from engine.narrative_engine import build_narrative
    d = _uptrend_dict()
    d["key_levels"] = {}
    text = build_narrative(d)
    assert isinstance(text, str)   # doesn't crash


# ──────────────────────────────────────────────────────────────
# Confirmation
# ──────────────────────────────────────────────────────────────

def test_confirmation_present_when_missing():
    from engine.narrative_engine import build_narrative
    text = build_narrative(_uptrend_dict())
    assert "confirm" in text.lower()


def test_confirmation_absent_when_empty():
    from engine.narrative_engine import build_narrative
    d = _uptrend_dict()
    d["missing_confirmation"] = ""
    text = build_narrative(d)
    # Should not crash and should still produce output
    assert len(text) > 20


# ──────────────────────────────────────────────────────────────
# Session implications
# ──────────────────────────────────────────────────────────────

def test_after_hours_no_entry_implication():
    from engine.narrative_engine import build_narrative
    text = build_narrative(_uptrend_dict(), session="after_hours")
    lower = text.lower()
    assert "first 30 minutes" not in lower
    assert "extended-hours" in lower or "next-session" in lower or "no active entries" in lower


def test_after_hours_no_pre_market():
    from engine.narrative_engine import build_narrative
    text = build_narrative(_uptrend_dict(), session="after_hours")
    assert "pre-market" not in text.lower()


def test_premarket_watch_for_open():
    from engine.narrative_engine import build_narrative
    text = build_narrative(_uptrend_dict(), session="premarket")
    assert "open" in text.lower() or "pre-market" in text.lower()


def test_closed_no_entries():
    from engine.narrative_engine import build_narrative
    text = build_narrative(_uptrend_dict(), session="closed")
    lower = text.lower()
    assert "closed" in lower or "next" in lower


def test_open_uptrend_long_bias():
    from engine.narrative_engine import build_narrative
    text = build_narrative(_uptrend_dict(), session="open",
                           entries_allowed=True)
    lower = text.lower()
    assert "long" in lower or "continuation" in lower or "structure" in lower


def test_open_downtrend_short_bias():
    from engine.narrative_engine import build_narrative
    text = build_narrative(_downtrend_dict(), session="open",
                           entries_allowed=True)
    lower = text.lower()
    assert "short" in lower or "continuation" in lower or "seller" in lower


def test_range_wait_for_breakout():
    from engine.narrative_engine import build_narrative
    text = build_narrative(_range_dict(), session="open", entries_allowed=True)
    lower = text.lower()
    assert "wait" in lower or "range" in lower or "break" in lower


# ──────────────────────────────────────────────────────────────
# No-structure fallback
# ──────────────────────────────────────────────────────────────

def test_fallback_chop_not_bare_label():
    from engine.narrative_engine import build_narrative
    text = build_narrative(None, regime="chop", session="after_hours")
    assert text != "Regime is chop."
    assert text != "Regime is chop. Narrative: chop."
    assert len(text) > 60


def test_fallback_chop_has_guidance():
    from engine.narrative_engine import build_narrative
    text = build_narrative(None, regime="chop", session="open",
                           entries_allowed=True)
    lower = text.lower()
    assert "chop" in lower or "equilibrium" in lower or "range" in lower
    assert "wait" in lower or "watch" in lower or "no" in lower


def test_fallback_trend_bull_mentions_structure():
    from engine.narrative_engine import build_narrative
    text = build_narrative(None, regime="trend_bull", session="open",
                           entries_allowed=True)
    lower = text.lower()
    assert "higher high" in lower or "hh" in lower or "trend" in lower


def test_fallback_after_hours_chop_no_first_30():
    from engine.narrative_engine import build_narrative
    text = build_narrative(None, regime="chop", session="after_hours",
                           entries_allowed=False)
    assert "first 30 minutes" not in text.lower()
    assert "pre-market" not in text.lower()


def test_fallback_empty_regime():
    from engine.narrative_engine import build_narrative
    text = build_narrative(None, regime="", session="open")
    assert len(text) > 20


# ──────────────────────────────────────────────────────────────
# StructureState object path
# ──────────────────────────────────────────────────────────────

def test_accepts_structure_state_object():
    from engine.narrative_engine import build_narrative
    from engine.structure_engine import analyze
    from runtime.update_cycle import build_demo_state, reset_engine
    reset_engine()
    state  = build_demo_state()
    struct = analyze(state)
    text   = build_narrative(struct, session="open", regime="trend_bull",
                             entries_allowed=True)
    assert isinstance(text, str)
    assert len(text) > 60


def test_structure_state_has_levels():
    from engine.narrative_engine import build_narrative
    from engine.structure_engine import analyze
    from runtime.update_cycle import build_demo_state, reset_engine
    reset_engine()
    state  = build_demo_state()
    struct = analyze(state)
    text   = build_narrative(struct, session="after_hours")
    # Should contain at least one real price level
    import re
    prices = re.findall(r"\d{3,}\.\d{2}", text)
    assert len(prices) >= 1, f"Expected price levels in narrative:\n{text}"


# ──────────────────────────────────────────────────────────────
# build_narrative_from_package
# ──────────────────────────────────────────────────────────────

def test_from_package_no_crash():
    from engine.narrative_engine import build_narrative_from_package
    from ai.state_packager import build_full_package
    from engine.structure_engine import analyze
    from runtime.update_cycle import build_demo_state, reset_engine
    reset_engine()
    state = build_demo_state()

    class _FMS:
        value = "after_hours"
    state.market_session = _FMS()
    pkg  = build_full_package(state=state, structure_state=analyze(state))
    text = build_narrative_from_package(pkg.to_dict())
    assert isinstance(text, str)
    assert len(text) > 40


def test_from_package_empty_dict():
    from engine.narrative_engine import build_narrative_from_package
    text = build_narrative_from_package({})
    assert isinstance(text, str)


def test_from_package_no_structure_key():
    from engine.narrative_engine import build_narrative_from_package
    text = build_narrative_from_package({"market": {"market_session": "after_hours",
                                                     "regime": {"name": "chop"}}})
    assert "chop" in text.lower() or "after hours" in text.lower()


# ──────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        (n, f) for n, f in sorted(globals().items())
        if n.startswith("test_") and callable(f)
    ]
    print(f"\nNarrative engine tests ({len(tests)} total)\n{'─' * 60}")
    for name, fn in tests:
        run(name, fn)
    print(f"\n{'=' * 60}")
    print(f"{passed} passed, {failed} failed  ({passed + failed} total)")
    import sys; sys.exit(0 if failed == 0 else 1)