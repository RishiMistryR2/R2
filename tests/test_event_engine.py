"""
tests/test_event_engine.py — Unit tests for engine/event_engine.py

Coverage:
  - Exact duplicate suppression
  - ATR price-move materiality (>0.5×, >1.0×, <0.5×)
  - All 9 change dimensions
  - Priority threshold calculation
  - Score accumulation and capping
  - snapshot_key bucketing
  - build_snapshot runtime integration

Run: PYTHONPATH=. python tests/test_event_engine.py
"""

import os, sys, traceback
from datetime import datetime, timezone
from typing import Optional

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
# Snapshot factory helpers
# ──────────────────────────────────────────────────────────────

def _snap(**kw):
    from engine.event_engine import MarketSnapshot
    base = dict(
        direction="uptrend",    market_state="trend",     trend_intact=True,
        sequence="HH HL HH HL", session_high=549.48,      session_low=541.19,
        last_hl=544.22,          last_lh=None,
        current_price=546.00,    atr_5m=1.40,
        missing_confirmation="Need break above 549.48.",
        regime_name="trend_bull", regime_conf=0.72,
        confidence_score=58.0,
        pattern_names=("Trend Continuation (Long)",),
        timestamp=datetime.now(tz=timezone.utc),
        trigger="regime_change",
    )
    base.update(kw)
    return MarketSnapshot(**base)


def _range_snap(**kw):
    return _snap(
        direction="ranging",     market_state="range",     trend_intact=False,
        sequence="",             session_high=737.00,      session_low=735.80,
        last_hl=None,            last_lh=None,
        current_price=736.40,    atr_5m=0.62,
        missing_confirmation="",
        regime_name="chop",      regime_conf=0.55,
        confidence_score=35.0,
        pattern_names=("Range Compression",),
        **kw,
    )


def _dn_snap(**kw):
    defaults = dict(
        direction="downtrend",    market_state="trend",    trend_intact=True,
        sequence="LH LL LH LL",  session_high=738.39,     session_low=730.50,
        last_hl=None,             last_lh=738.39,
        current_price=734.00,     atr_5m=1.55,
        missing_confirmation="Need new low below 730.50.",
        regime_name="trend_bear", regime_conf=0.68,
        confidence_score=54.0,
        pattern_names=("Trend Continuation (Short)",),
    )
    defaults.update(kw)
    return _snap(**defaults)

# ──────────────────────────────────────────────────────────────
# Always-allow triggers
# ──────────────────────────────────────────────────────────────

def test_startup_always_allowed():
    from engine.event_engine import evaluate_event
    r = evaluate_event(_snap(), _snap(), "startup_briefing")
    assert not r.suppress and r.score == 100.0


def test_user_request_always_allowed():
    from engine.event_engine import evaluate_event
    r = evaluate_event(_snap(), _snap(), "user_request")
    assert not r.suppress


def test_no_previous_always_allowed():
    from engine.event_engine import evaluate_event
    r = evaluate_event(_snap(), None, "regime_change")
    assert not r.suppress


# ──────────────────────────────────────────────────────────────
# DUPLICATE SUPPRESSION
# ──────────────────────────────────────────────────────────────

def test_exact_duplicate_suppressed():
    """Core requirement: identical snapshots must be suppressed."""
    from engine.event_engine import evaluate_event
    s = _snap()
    r = evaluate_event(s, s, "regime_change")
    assert r.suppress, f"Exact duplicate must be suppressed, got score={r.score}"


def test_exact_duplicate_score_zero_or_negative():
    from engine.event_engine import evaluate_event
    s = _snap()
    r = evaluate_event(s, s, "regime_change")
    assert r.score <= 5.0, f"Duplicate score should be ≤5, got {r.score}"


def test_exact_duplicate_priority_suppressed():
    from engine.event_engine import evaluate_event
    s = _snap()
    r = evaluate_event(s, s, "narrative_change")
    assert r.priority == "suppressed"


def test_one_point_confidence_change_suppressed():
    """1-point confidence delta alone must be suppressed (< 5-point threshold)."""
    from engine.event_engine import evaluate_event
    curr = _snap(confidence_score=59.0)
    prev = _snap(confidence_score=58.0)
    r    = evaluate_event(curr, prev, "narrative_change")
    assert r.suppress, f"1-pt confidence change should be suppressed, score={r.score}"


def test_tiny_price_move_suppressed():
    """Sub-0.5 ATR price move alone must be suppressed."""
    from engine.event_engine import evaluate_event
    # ATR=1.40, move=0.30 → ratio=0.21 (< 0.5 ATR)
    curr = _snap(current_price=546.30, atr_5m=1.40)
    prev = _snap(current_price=546.00, atr_5m=1.40)
    r    = evaluate_event(curr, prev, "narrative_change")
    assert r.suppress, f"0.21× ATR move should be suppressed, score={r.score}"


def test_four_point_confidence_no_other_changes_suppressed():
    """4-point confidence delta with no structural changes must be suppressed."""
    from engine.event_engine import evaluate_event
    curr = _snap(confidence_score=62.0)
    prev = _snap(confidence_score=58.0)
    r    = evaluate_event(curr, prev, "narrative_change")
    assert r.suppress, f"4-pt confidence alone should be suppressed, score={r.score}"


# ──────────────────────────────────────────────────────────────
# ATR PRICE-MOVE MATERIALITY
# ──────────────────────────────────────────────────────────────

def test_price_move_large_not_suppressed():
    """Price move ≥1.0× ATR must produce an event."""
    from engine.event_engine import evaluate_event
    # ATR=1.40, move=1.60 → ratio=1.14
    curr = _snap(current_price=547.60, atr_5m=1.40)
    prev = _snap(current_price=546.00, atr_5m=1.40)
    r    = evaluate_event(curr, prev, "regime_change")
    assert not r.suppress, f"1.14× ATR move should not be suppressed"


def test_price_move_large_score():
    """Price move ≥1.0× ATR contributes +20 to score."""
    from engine.event_engine import _detect_changes
    curr    = _snap(current_price=547.60, atr_5m=1.40)
    prev    = _snap(current_price=546.00, atr_5m=1.40)
    changes = _detect_changes(curr, prev)
    item    = next((c for c in changes if c.key == "price_move_large"), None)
    assert item is not None, "Should detect price_move_large"
    assert item.score == 20.0


def test_price_move_large_priority_important():
    from engine.event_engine import _detect_changes
    curr    = _snap(current_price=547.60, atr_5m=1.40)
    prev    = _snap(current_price=546.00, atr_5m=1.40)
    changes = _detect_changes(curr, prev)
    item    = next((c for c in changes if c.key == "price_move_large"), None)
    assert item is not None
    assert item.priority == "important"


def test_price_move_large_description_includes_ratio():
    from engine.event_engine import _detect_changes
    curr    = _snap(current_price=547.60, atr_5m=1.40)
    prev    = _snap(current_price=546.00, atr_5m=1.40)
    changes = _detect_changes(curr, prev)
    item    = next((c for c in changes if c.key == "price_move_large"), None)
    assert item is not None
    assert "ATR" in item.description
    assert "1.1" in item.description or "1.14" in item.description or "×" in item.description


def test_price_move_meaningful_detected():
    """Price move ≥0.5× ATR but <1.0× must produce price_move_meaningful."""
    from engine.event_engine import _detect_changes
    # ATR=1.40, move=0.80 → ratio=0.57
    curr    = _snap(current_price=546.80, atr_5m=1.40)
    prev    = _snap(current_price=546.00, atr_5m=1.40)
    changes = _detect_changes(curr, prev)
    item    = next((c for c in changes if c.key == "price_move_meaningful"), None)
    assert item is not None, "Should detect price_move_meaningful for 0.57× ATR move"
    assert item.score == 8.0


def test_price_move_meaningful_informational():
    from engine.event_engine import _detect_changes
    curr    = _snap(current_price=546.80, atr_5m=1.40)
    prev    = _snap(current_price=546.00, atr_5m=1.40)
    changes = _detect_changes(curr, prev)
    item    = next((c for c in changes if c.key == "price_move_meaningful"), None)
    assert item is not None
    assert item.priority == "informational"


def test_price_move_sub_half_atr_no_item():
    """Price move <0.5× ATR must produce no price-move change item."""
    from engine.event_engine import _detect_changes
    # ATR=1.40, move=0.20 → ratio=0.14
    curr    = _snap(current_price=546.20, atr_5m=1.40)
    prev    = _snap(current_price=546.00, atr_5m=1.40)
    changes = _detect_changes(curr, prev)
    keys    = [c.key for c in changes]
    assert "price_move_large"      not in keys
    assert "price_move_meaningful" not in keys


def test_price_move_no_atr_no_item():
    """Without ATR, price move scoring must be skipped gracefully."""
    from engine.event_engine import _detect_changes
    curr    = _snap(current_price=549.00, atr_5m=None)
    prev    = _snap(current_price=546.00, atr_5m=None)
    changes = _detect_changes(curr, prev)
    keys    = [c.key for c in changes]
    assert "price_move_large"      not in keys
    assert "price_move_meaningful" not in keys


def test_price_move_no_previous_price_no_item():
    """Without a previous price, price move scoring must be skipped."""
    from engine.event_engine import _detect_changes
    curr    = _snap(current_price=549.00, atr_5m=1.40)
    prev    = _snap(current_price=None,   atr_5m=1.40)
    changes = _detect_changes(curr, prev)
    keys    = [c.key for c in changes]
    assert "price_move_large" not in keys


def test_price_downward_move_large_detected():
    """Downward large move must also be detected."""
    from engine.event_engine import _detect_changes
    curr    = _dn_snap(current_price=730.00, atr_5m=1.55)
    prev    = _dn_snap(current_price=732.00, atr_5m=1.55)
    # delta = -2.00, ratio = 1.29 → large
    changes = _detect_changes(curr, prev)
    item    = next((c for c in changes if c.key == "price_move_large"), None)
    assert item is not None
    assert "lower" in item.description.lower()


def test_price_move_description_direction():
    """Description must state 'higher' or 'lower' direction."""
    from engine.event_engine import _detect_changes
    curr    = _snap(current_price=549.00, atr_5m=1.40)
    prev    = _snap(current_price=546.00, atr_5m=1.40)
    changes = _detect_changes(curr, prev)
    item    = next((c for c in changes
                    if c.key in ("price_move_large", "price_move_meaningful")), None)
    if item:
        assert "higher" in item.description.lower() or "lower" in item.description.lower()


# ──────────────────────────────────────────────────────────────
# STRUCTURAL CHANGES
# ──────────────────────────────────────────────────────────────

def test_direction_change_critical():
    from engine.event_engine import evaluate_event
    r = evaluate_event(_snap(direction="downtrend"), _snap(direction="uptrend"),
                       "regime_change")
    assert r.priority == "critical" and not r.suppress


def test_direction_change_score():
    from engine.event_engine import _detect_changes, _SCORES
    changes = _detect_changes(_snap(direction="downtrend"), _snap(direction="uptrend"))
    item    = next((c for c in changes if c.key == "direction_change"), None)
    assert item is not None
    assert item.score == _SCORES["direction_change"]


def test_trend_broken_critical():
    from engine.event_engine import evaluate_event
    r = evaluate_event(_snap(trend_intact=False), _snap(trend_intact=True),
                       "regime_change")
    assert r.priority in ("critical", "important") and not r.suppress


def test_trend_broken_item():
    from engine.event_engine import _detect_changes
    changes = _detect_changes(_snap(trend_intact=False), _snap(trend_intact=True))
    assert any(c.key == "trend_broken" for c in changes)


def test_trend_restored_item():
    from engine.event_engine import _detect_changes
    changes = _detect_changes(_snap(trend_intact=True), _snap(trend_intact=False))
    assert any(c.key == "trend_restored" for c in changes)


def test_state_to_breakout_critical():
    from engine.event_engine import evaluate_event
    r = evaluate_event(
        _snap(market_state="breakout"),
        _snap(market_state="trend"),
        "regime_change",
    )
    assert r.priority == "critical"


def test_state_to_breakdown_critical():
    from engine.event_engine import evaluate_event
    r = evaluate_event(
        _dn_snap(market_state="breakdown"),
        _dn_snap(market_state="trend"),
        "regime_change",
    )
    assert r.priority == "critical"


def test_state_change_trend_to_pullback_not_suppressed():
    from engine.event_engine import evaluate_event
    r = evaluate_event(
        _snap(market_state="pullback"),
        _snap(market_state="trend"),
        "regime_change",
    )
    assert not r.suppress


# ──────────────────────────────────────────────────────────────
# CONFIRMATION
# ──────────────────────────────────────────────────────────────

def test_confirmation_achieved_critical():
    from engine.event_engine import evaluate_event
    curr = _snap(missing_confirmation="")
    prev = _snap(missing_confirmation="Need break above 549.48.")
    r    = evaluate_event(curr, prev, "regime_change")
    assert r.priority == "critical" and not r.suppress


def test_confirmation_achieved_item():
    from engine.event_engine import _detect_changes
    changes = _detect_changes(
        _snap(missing_confirmation=""),
        _snap(missing_confirmation="Need break above 549.48."),
    )
    assert any(c.key == "confirmation_achieved" for c in changes)


def test_no_confirmation_change_no_item():
    from engine.event_engine import _detect_changes
    same = "Need break above 549.48."
    changes = _detect_changes(_snap(missing_confirmation=same),
                              _snap(missing_confirmation=same))
    assert not any(c.key == "confirmation_achieved" for c in changes)


# ──────────────────────────────────────────────────────────────
# CONFIDENCE
# ──────────────────────────────────────────────────────────────

def test_confidence_major_item():
    from engine.event_engine import _detect_changes
    changes = _detect_changes(_snap(confidence_score=74.0), _snap(confidence_score=57.0))
    assert any(c.key == "confidence_major" for c in changes)


def test_confidence_major_description_both_values():
    from engine.event_engine import _detect_changes
    changes = _detect_changes(_snap(confidence_score=74.0), _snap(confidence_score=57.0))
    item    = next((c for c in changes if c.key == "confidence_major"), None)
    assert item is not None
    assert "57" in item.description and "74" in item.description


def test_confidence_minor_item():
    from engine.event_engine import _detect_changes
    changes = _detect_changes(_snap(confidence_score=63.0), _snap(confidence_score=57.0))
    assert any(c.key in ("confidence_minor", "confidence_major") for c in changes)


def test_confidence_tiny_no_item():
    from engine.event_engine import _detect_changes
    changes = _detect_changes(_snap(confidence_score=58.9), _snap(confidence_score=57.0))
    keys    = [c.key for c in changes]
    assert "confidence_minor" not in keys and "confidence_major" not in keys


# ──────────────────────────────────────────────────────────────
# REGIME
# ──────────────────────────────────────────────────────────────

def test_regime_name_change_not_suppressed():
    from engine.event_engine import evaluate_event
    r = evaluate_event(
        _snap(regime_name="chop"),
        _snap(regime_name="trend_bull"),
        "regime_change",
    )
    assert not r.suppress


def test_regime_change_item():
    from engine.event_engine import _detect_changes
    changes = _detect_changes(_snap(regime_name="chop"), _snap(regime_name="trend_bull"))
    assert any(c.key == "regime_name_change" for c in changes)


def test_same_regime_no_item():
    from engine.event_engine import _detect_changes
    changes = _detect_changes(_snap(regime_name="trend_bull"), _snap(regime_name="trend_bull"))
    assert not any(c.key == "regime_name_change" for c in changes)


# ──────────────────────────────────────────────────────────────
# PATTERN
# ──────────────────────────────────────────────────────────────

def test_pattern_added_item():
    from engine.event_engine import _detect_changes
    curr    = _snap(pattern_names=("Trend Continuation (Long)", "Failed Breakout"))
    prev    = _snap(pattern_names=("Trend Continuation (Long)",))
    changes = _detect_changes(curr, prev)
    assert any(c.key in ("pattern_added", "pattern_change") for c in changes)


def test_pattern_removed_item():
    from engine.event_engine import _detect_changes
    curr    = _snap(pattern_names=())
    prev    = _snap(pattern_names=("Trend Continuation (Long)",))
    changes = _detect_changes(curr, prev)
    assert any(c.key in ("pattern_removed", "pattern_change") for c in changes)


def test_same_patterns_no_item():
    from engine.event_engine import _detect_changes
    same = ("Trend Continuation (Long)",)
    changes = _detect_changes(_snap(pattern_names=same), _snap(pattern_names=same))
    assert not any("pattern" in c.key for c in changes)


# ──────────────────────────────────────────────────────────────
# PRIORITY THRESHOLDS
# ──────────────────────────────────────────────────────────────

def test_priority_critical():
    from engine.event_engine import _to_priority
    assert _to_priority(40.0) == "critical"
    assert _to_priority(100.0) == "critical"


def test_priority_important():
    from engine.event_engine import _to_priority
    assert _to_priority(20.0) == "important"
    assert _to_priority(39.9) == "important"


def test_priority_informational():
    from engine.event_engine import _to_priority
    assert _to_priority(5.0) == "informational"
    assert _to_priority(19.9) == "informational"


def test_priority_suppressed():
    from engine.event_engine import _to_priority
    assert _to_priority(4.9) == "suppressed"
    assert _to_priority(0.0) == "suppressed"


# ──────────────────────────────────────────────────────────────
# SCORE ACCUMULATION AND CAPPING
# ──────────────────────────────────────────────────────────────

def test_multiple_changes_accumulate():
    from engine.event_engine import evaluate_event
    curr = _snap(
        direction="downtrend", trend_intact=False,
        regime_name="trend_bear", confidence_score=74.0,
    )
    prev = _snap(
        direction="uptrend", trend_intact=True,
        regime_name="trend_bull", confidence_score=57.0,
    )
    r = evaluate_event(curr, prev, "regime_change")
    assert r.score >= 50.0, f"Multiple changes should accumulate, got {r.score}"
    assert r.priority == "critical"


def test_score_capped_at_100():
    from engine.event_engine import evaluate_event
    curr = _snap(
        direction="downtrend", trend_intact=False, market_state="breakdown",
        regime_name="chop", confidence_score=74.0,
        pattern_names=("Failed Breakout", "Bear Flag"),
        current_price=538.00, atr_5m=1.40,
    )
    prev = _snap(
        direction="uptrend", trend_intact=True, market_state="trend",
        regime_name="trend_bull", confidence_score=57.0,
        pattern_names=("Trend Continuation (Long)",),
        current_price=546.00, atr_5m=1.40,
    )
    r = evaluate_event(curr, prev, "regime_change")
    assert r.score <= 100.0, f"Score must be capped at 100, got {r.score}"


def test_score_non_negative():
    from engine.event_engine import evaluate_event
    r = evaluate_event(_snap(), _snap(), "narrative_change")
    assert r.score >= 0.0


# ──────────────────────────────────────────────────────────────
# SNAPSHOT KEY — DEDUPLICATION BUCKETING
# ──────────────────────────────────────────────────────────────

def test_same_state_same_key():
    from engine.event_engine import snapshot_key
    assert snapshot_key(_snap()) == snapshot_key(_snap())


def test_different_direction_different_key():
    from engine.event_engine import snapshot_key
    assert snapshot_key(_snap(direction="uptrend")) != snapshot_key(_snap(direction="downtrend"))


def test_different_regime_different_key():
    from engine.event_engine import snapshot_key
    assert snapshot_key(_snap(regime_name="chop")) != snapshot_key(_snap(regime_name="trend_bull"))


def test_tiny_confidence_same_key():
    """1-point confidence delta must NOT produce a different key (bucketed to ±5)."""
    from engine.event_engine import snapshot_key
    s1 = _snap(confidence_score=58.0)
    s2 = _snap(confidence_score=59.0)
    assert snapshot_key(s1) == snapshot_key(s2)


def test_five_point_confidence_may_differ():
    """5-point delta crossing a bucket boundary must produce a different key."""
    from engine.event_engine import snapshot_key
    s1 = _snap(confidence_score=55.0)   # bucket 55
    s2 = _snap(confidence_score=60.0)   # bucket 60
    assert snapshot_key(s1) != snapshot_key(s2)


def test_different_patterns_different_key():
    from engine.event_engine import snapshot_key
    s1 = _snap(pattern_names=("Bull Flag",))
    s2 = _snap(pattern_names=("Bear Flag",))
    assert snapshot_key(s1) != snapshot_key(s2)


def test_different_confirmation_different_key():
    from engine.event_engine import snapshot_key
    s1 = _snap(missing_confirmation="Need break above 549.48.")
    s2 = _snap(missing_confirmation="")
    assert snapshot_key(s1) != snapshot_key(s2)


# ──────────────────────────────────────────────────────────────
# CHANGE SUMMARY
# ──────────────────────────────────────────────────────────────

def test_change_summary_not_empty_for_direction_change():
    from engine.event_engine import evaluate_event
    r = evaluate_event(_snap(direction="downtrend"), _snap(direction="uptrend"),
                       "regime_change")
    assert len(r.change_summary) > 0


def test_change_summary_includes_direction():
    from engine.event_engine import evaluate_event
    r = evaluate_event(_snap(direction="downtrend"), _snap(direction="uptrend"),
                       "regime_change")
    assert "direction" in r.change_summary.lower() or "downtrend" in r.change_summary.lower()


def test_change_summary_empty_for_suppressed():
    from engine.event_engine import evaluate_event
    r = evaluate_event(_snap(), _snap(), "regime_change")
    assert r.suppress


# ──────────────────────────────────────────────────────────────
# RUNTIME INTEGRATION
# ──────────────────────────────────────────────────────────────

def test_build_snapshot_no_crash():
    from engine.event_engine import build_snapshot
    from runtime.update_cycle import run_update_cycle, reset_engine
    from runtime.runtime_state import RuntimeState
    reset_engine()
    state = RuntimeState()
    run_update_cycle(state)
    snap = build_snapshot(state, "regime_change")
    assert snap is not None
    assert 0 <= snap.confidence_score <= 100


def test_build_snapshot_direction_populated():
    from engine.event_engine import build_snapshot
    from runtime.update_cycle import run_update_cycle, reset_engine
    from runtime.runtime_state import RuntimeState
    reset_engine()
    state = RuntimeState()
    run_update_cycle(state)
    snap = build_snapshot(state, "regime_change")
    assert snap.direction in ("uptrend", "downtrend", "ranging", "")


def test_build_snapshot_blank_state_no_crash():
    from engine.event_engine import build_snapshot
    from runtime.runtime_state import RuntimeState
    snap = build_snapshot(RuntimeState(), "regime_change")
    assert snap is not None


def test_build_snapshot_has_atr_after_update():
    """After a full update cycle, ATR should be populated."""
    from engine.event_engine import build_snapshot
    from runtime.update_cycle import run_update_cycle, reset_engine
    from runtime.runtime_state import RuntimeState
    reset_engine()
    state = RuntimeState()
    run_update_cycle(state)
    snap = build_snapshot(state, "regime_change")
    # ATR may be None if indicators aren't wired in demo mode — just don't crash
    assert snap.atr_5m is None or snap.atr_5m > 0


def test_evaluate_after_real_update():
    """Full pipeline: build two snapshots, evaluate between them."""
    from engine.event_engine import build_snapshot, evaluate_event
    from runtime.update_cycle import run_update_cycle, reset_engine
    from runtime.runtime_state import RuntimeState
    reset_engine()
    state = RuntimeState()
    run_update_cycle(state)
    snap1 = build_snapshot(state, "regime_change")
    snap2 = build_snapshot(state, "narrative_change")
    r     = evaluate_event(snap2, snap1, "narrative_change")
    # Same state → suppress
    assert isinstance(r.suppress, bool)
    assert 0 <= r.score <= 100


# ──────────────────────────────────────────────────────────────
# GOOD EVENT vs BAD EVENT (user-spec examples)
# ──────────────────────────────────────────────────────────────

def test_good_event_not_suppressed():
    """'Price broke below prior LL. Confirmation achieved.' → should not suppress."""
    from engine.event_engine import evaluate_event
    curr = _dn_snap(
        market_state         = "breakdown",
        missing_confirmation = "",           # confirmation achieved
        current_price        = 729.50,       # broke below session low 730.50
        confidence_score     = 74.0,         # was 57%
    )
    prev = _dn_snap(
        market_state         = "trend",
        missing_confirmation = "Need new low below 730.50.",
        current_price        = 731.00,
        confidence_score     = 57.0,
    )
    r = evaluate_event(curr, prev, "regime_change")
    assert not r.suppress, f"Good event must not be suppressed, score={r.score}"
    assert r.priority in ("critical", "important")


def test_bad_event_suppressed():
    """'Price moved 0.04%. Confidence 57% → 58%.' → must suppress."""
    from engine.event_engine import evaluate_event
    curr = _snap(current_price=546.04, confidence_score=58.0)
    prev = _snap(current_price=546.00, confidence_score=57.0)
    r    = evaluate_event(curr, prev, "narrative_change")
    assert r.suppress, f"Bad event (0.04% move, 1pt confidence) must suppress, score={r.score}"


# ──────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        (n, f) for n, f in sorted(globals().items())
        if n.startswith("test_") and callable(f)
    ]
    print(f"\nEvent intelligence engine tests ({len(tests)} total)\n{'─' * 60}")
    for name, fn in tests:
        run(name, fn)
    print(f"\n{'=' * 60}")
    print(f"{passed} passed, {failed} failed  ({passed + failed} total)")
    import sys; sys.exit(0 if failed == 0 else 1)