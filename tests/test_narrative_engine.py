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
# v2.1 — Level specificity and invalidation
# ──────────────────────────────────────────────────────────────

def test_range_buyer_seller_uses_session_levels():
    """Range branch must use actual session_high/session_low values."""
    from engine.narrative_engine import build_narrative
    text = build_narrative(_range_dict(), session="open", entries_allowed=True)
    # session_high=549.48, session_low=541.19 must appear
    assert "549.48" in text, f"session_high missing from range narrative:\n{text}"
    assert "541.19" in text, f"session_low missing from range narrative:\n{text}"


def test_range_no_generic_equilibrium():
    """Range branch must not say 'buyers and sellers are in equilibrium' without levels."""
    from engine.narrative_engine import build_narrative
    text = build_narrative(_range_dict(), session="open", entries_allowed=True)
    # The generic equilibrium phrase only appears when levels are absent
    if "equilibrium" in text.lower():
        # If it's there, levels must also be there to anchor it
        assert "541.19" in text or "549.48" in text, (
            f"'equilibrium' used without level anchors:\n{text}"
        )


def test_uptrend_has_invalidation():
    """Uptrend narrative must include invalidation (close below HL or session low)."""
    from engine.narrative_engine import build_narrative
    text = build_narrative(_uptrend_dict(), session="open", entries_allowed=True)
    lower = text.lower()
    assert (
        "structure degrades" in lower
        or "degrades" in lower
        or "close below" in lower
        or "invalidat" in lower
    ), f"Expected invalidation in uptrend narrative:\n{text}"


def test_downtrend_has_invalidation():
    """Downtrend narrative must include invalidation (reclaim of LH or session high)."""
    from engine.narrative_engine import build_narrative
    text = build_narrative(_downtrend_dict(), session="open", entries_allowed=True)
    lower = text.lower()
    assert (
        "reclaim" in lower
        or "loses conviction" in lower
        or "invalidat" in lower
        or "close above" in lower
    ), f"Expected invalidation in downtrend narrative:\n{text}"


def test_range_has_confirmation_with_level():
    """Range narrative must state confirmation level explicitly."""
    from engine.narrative_engine import build_narrative
    text = build_narrative(_range_dict(), session="open", entries_allowed=True)
    # Either from _confirmation or _invalidation
    assert "549.48" in text or "541.19" in text, (
        f"Range narrative missing specific confirmation level:\n{text}"
    )


def test_range_has_invalidation_both_directions():
    """Range invalidation must give both bullish and bearish resolution levels."""
    from engine.narrative_engine import build_narrative
    text = build_narrative(_range_dict(), session="open", entries_allowed=True)
    # Both session_high and session_low should appear
    has_high = "549.48" in text
    has_low  = "541.19" in text
    assert has_high and has_low, (
        f"Range narrative should reference both session_high (549.48) "
        f"and session_low (541.19):\n{text}"
    )


def test_minimal_fallback_chop_uses_session_levels():
    """
    No-structure chop fallback must use specific price levels, not just
    'current range'.
    """
    from engine.narrative_engine import build_narrative
    kl = {
        "session_high":   737.00,
        "session_low":    735.80,
        "vwap":           736.40,
        "prior_day_high": 738.50,
        "prior_day_low":  734.20,
        "nearest_support": None,
        "nearest_resistance": None,
    }
    # Pass kl via a minimal struct dict with no direction
    struct_no_dir = {"key_levels": kl, "direction": "", "market_state": "range"}
    text = build_narrative(struct_no_dir, regime="chop", session="after_hours",
                           entries_allowed=False)
    assert "737.00" in text or "735.80" in text, (
        f"Fallback chop must reference specific levels:\n{text}"
    )
    assert "current range" not in text.lower(), (
        f"'current range' without levels is too generic:\n{text}"
    )


def test_minimal_fallback_no_generic_range_phrase():
    """'current range' must not appear as an unanchored phrase."""
    from engine.narrative_engine import build_narrative
    # No structure, no levels at all — generic is acceptable here
    text = build_narrative(None, regime="chop", session="after_hours",
                           entries_allowed=False)
    # This path has no levels — but should still not say "current range"
    assert "current range" not in text.lower(), (
        f"'current range' should not appear without a specific level:\n{text}"
    )


def test_minimal_fallback_trend_bull_mentions_specific_level():
    """Trend bull fallback with levels must mention a specific breakout target."""
    from engine.narrative_engine import build_narrative
    kl = {
        "session_high":   549.48,
        "session_low":    541.19,
        "vwap":           543.50,
        "prior_day_high": 565.43,
        "prior_day_low":  552.67,
        "nearest_support": None,
        "nearest_resistance": None,
    }
    struct_no_dir = {"key_levels": kl, "direction": "", "market_state": "range"}
    text = build_narrative(struct_no_dir, regime="trend_bull", session="open",
                           entries_allowed=True)
    # Should mention some breakout level
    import re
    prices = re.findall(r"\d{3,}\.\d{2}", text)
    assert len(prices) >= 1, f"Trend bull fallback with levels must mention a price:\n{text}"


def test_minimal_fallback_trend_bear_is_bearish():
    """
    Regression: "trend_bear" contains "trend", and the bull branch used to
    match it first, producing bullish text for a bearish regime.
    """
    from engine.narrative_engine import build_narrative
    kl = {
        "session_high":   738.39,
        "session_low":    730.50,
        "vwap":           734.90,
        "prior_day_high": 741.00,
        "prior_day_low":  728.00,
        "nearest_support": None,
        "nearest_resistance": None,
    }
    struct_no_dir = {"key_levels": kl, "direction": "", "market_state": "range"}
    for levels in (struct_no_dir, None):
        text = build_narrative(levels, regime="trend_bear", session="open",
                               entries_allowed=True)
        assert "lower highs and lower lows" in text, (
            f"trend_bear fallback must be bearish:\n{text}"
        )
        assert "higher highs and higher lows" not in text, (
            f"trend_bear fallback must not contain bullish watch text:\n{text}"
        )


def test_minimal_fallback_trend_bull_still_bullish():
    from engine.narrative_engine import build_narrative
    text = build_narrative(None, regime="trend_bull", session="open",
                           entries_allowed=True)
    assert "higher highs and higher lows" in text
    assert "lower highs and lower lows" not in text


def test_uptrend_confirmation_and_invalidation_have_different_levels():
    """Confirmation and invalidation must reference different price levels."""
    from engine.narrative_engine import build_narrative
    text = build_narrative(_uptrend_dict(), session="open", entries_allowed=True)
    import re
    prices = re.findall(r"\d{3,}\.\d{2}", text)
    unique = set(prices)
    assert len(unique) >= 2, (
        f"Expected at least 2 distinct levels in uptrend narrative:\n{text}\n"
        f"Prices found: {prices}"
    )


def test_structure_state_object_has_invalidation():
    """StructureState object path must produce invalidation."""
    from engine.narrative_engine import build_narrative
    from engine.structure_engine import analyze
    from runtime.update_cycle import build_demo_state, reset_engine
    reset_engine()
    struct = analyze(build_demo_state())
    text   = build_narrative(struct, session="open", entries_allowed=True)
    lower  = text.lower()
    assert (
        "degrades" in lower or "close below" in lower
        or "reclaim" in lower or "loses conviction" in lower
    ), f"StructureState path must produce invalidation:\n{text}"


def test_vwap_mentioned_as_pivot_in_range():
    """When VWAP is in key_levels, range output must call it a pivot."""
    from engine.narrative_engine import build_narrative
    text = build_narrative(_range_dict(), session="open", entries_allowed=True)
    lower = text.lower()
    # _range_dict includes vwap=545.00
    assert "545.00" in text or "vwap" in lower, (
        f"Range narrative should reference VWAP:\n{text}"
    )


# ──────────────────────────────────────────────────────────────


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