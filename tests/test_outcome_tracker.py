"""
tests/test_outcome_tracker.py — Tests for memory/outcome_tracker.py

Coverage:
  ResolutionStrategy subclasses (SessionBreakStrategy, ATRTargetStrategy,
    TimeExpirationStrategy, StructureContinuationStrategy)
  OutcomeTracker.record_observation  — fields, prediction, snapshot duck-typing
  OutcomeTracker.update_pending      — price trackers, resolution, MFE/MAE
  OutcomeTracker.record_outcome      — manual override
  PatternStats                       — rates, accuracy, str()
  get_stats / all_stats / similar_patterns / summary_text
  pending_observations / recent_observations
  Persistence (save + load round-trip)
  Rolling trim (max_observations, eviction order)
  Singleton (get_tracker / reset_tracker)
  PatternObservation properties (elapsed_minutes, prediction_correct)

Run: PYTHONPATH=. python tests/test_outcome_tracker.py
"""

import os
import sys
import tempfile
import traceback
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


# ──────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────

def _tracker(timeout=120.0, strategy=None):
    from memory.outcome_tracker import OutcomeTracker, SessionBreakStrategy
    s = strategy or SessionBreakStrategy(timeout_minutes=timeout)
    return OutcomeTracker(path=":memory:", strategy=s)


def _snap(**kw):
    """Minimal snapshot dict compatible with record_observation."""
    base = dict(
        direction="downtrend",  market_state="trend",
        sequence="LH LL LH LL", confidence_score=54.0,
        regime_name="trend_bear", trigger="regime_change",
        session_high=738.39,    session_low=730.50,
        current_price=734.00,   atr_5m=1.55,
        pattern_names=["Trend Continuation (Short)"],
    )
    base.update(kw)
    return base


def _pat(name: str, bias: str = "bearish"):
    """Minimal PatternMatch-like object."""
    class _P:
        pass
    p = _P()
    p.name = name
    p.bias = bias
    return p


def _pat_dict(name: str, bias: str = "bearish"):
    return {"name": name, "bias": bias}


# ══════════════════════════════════════════════════════════════
# ResolutionStrategy — SessionBreakStrategy
# ══════════════════════════════════════════════════════════════

def test_session_break_bullish_when_price_at_session_high():
    from memory.outcome_tracker import SessionBreakStrategy, _dict_to_obs
    s   = SessionBreakStrategy(timeout_minutes=120.0)
    obs = _dict_to_obs(_obs_fields(session_high=738.39, session_low=730.50))
    r   = s.resolve(obs, current_price=738.39, atr=1.55, elapsed_minutes=10.0)
    assert r == "bullish"


def test_session_break_bullish_when_price_above_session_high():
    from memory.outcome_tracker import SessionBreakStrategy, _dict_to_obs
    s   = SessionBreakStrategy()
    obs = _dict_to_obs(_obs_fields(session_high=738.39, session_low=730.50))
    r   = s.resolve(obs, current_price=740.00, atr=1.55, elapsed_minutes=5.0)
    assert r == "bullish"


def test_session_break_bearish_when_price_at_session_low():
    from memory.outcome_tracker import SessionBreakStrategy, _dict_to_obs
    s   = SessionBreakStrategy()
    obs = _dict_to_obs(_obs_fields(session_high=738.39, session_low=730.50))
    r   = s.resolve(obs, current_price=730.50, atr=1.55, elapsed_minutes=5.0)
    assert r == "bearish"


def test_session_break_bearish_when_price_below_session_low():
    from memory.outcome_tracker import SessionBreakStrategy, _dict_to_obs
    s   = SessionBreakStrategy()
    obs = _dict_to_obs(_obs_fields(session_high=738.39, session_low=730.50))
    r   = s.resolve(obs, current_price=729.00, atr=1.55, elapsed_minutes=5.0)
    assert r == "bearish"


def test_session_break_none_when_inside_range():
    from memory.outcome_tracker import SessionBreakStrategy, _dict_to_obs
    s   = SessionBreakStrategy()
    obs = _dict_to_obs(_obs_fields(session_high=738.39, session_low=730.50))
    r   = s.resolve(obs, current_price=734.00, atr=1.55, elapsed_minutes=5.0)
    assert r is None


def test_session_break_inconclusive_on_timeout():
    from memory.outcome_tracker import SessionBreakStrategy, _dict_to_obs
    s   = SessionBreakStrategy(timeout_minutes=60.0)
    obs = _dict_to_obs(_obs_fields(session_high=738.39, session_low=730.50))
    r   = s.resolve(obs, current_price=734.00, atr=1.55, elapsed_minutes=60.0)
    assert r == "inconclusive"


def test_session_break_timeout_takes_priority_over_price():
    """When elapsed >= timeout AND price crossed, timeout wins first."""
    from memory.outcome_tracker import SessionBreakStrategy, _dict_to_obs
    s   = SessionBreakStrategy(timeout_minutes=5.0)
    obs = _dict_to_obs(_obs_fields(session_high=738.39, session_low=730.50))
    r   = s.resolve(obs, current_price=740.00, atr=1.55, elapsed_minutes=10.0)
    assert r == "inconclusive"


# ══════════════════════════════════════════════════════════════
# ResolutionStrategy — ATRTargetStrategy
# ══════════════════════════════════════════════════════════════

def test_atr_target_bullish_when_price_rises_target():
    from memory.outcome_tracker import ATRTargetStrategy, _dict_to_obs
    s   = ATRTargetStrategy(target_atr=1.0, timeout_minutes=120.0)
    obs = _dict_to_obs(_obs_fields(price_at_record=734.00, atr_at_record=1.55))
    r   = s.resolve(obs, current_price=735.55, atr=1.55, elapsed_minutes=10.0)
    assert r == "bullish"


def test_atr_target_bearish_when_price_falls_target():
    from memory.outcome_tracker import ATRTargetStrategy, _dict_to_obs
    s   = ATRTargetStrategy(target_atr=1.0, timeout_minutes=120.0)
    obs = _dict_to_obs(_obs_fields(price_at_record=734.00, atr_at_record=1.55))
    r   = s.resolve(obs, current_price=732.45, atr=1.55, elapsed_minutes=10.0)
    assert r == "bearish"


def test_atr_target_none_when_move_too_small():
    from memory.outcome_tracker import ATRTargetStrategy, _dict_to_obs
    s   = ATRTargetStrategy(target_atr=1.0, timeout_minutes=120.0)
    obs = _dict_to_obs(_obs_fields(price_at_record=734.00, atr_at_record=1.55))
    r   = s.resolve(obs, current_price=734.50, atr=1.55, elapsed_minutes=5.0)
    assert r is None


def test_atr_target_inconclusive_on_timeout():
    from memory.outcome_tracker import ATRTargetStrategy, _dict_to_obs
    s   = ATRTargetStrategy(target_atr=1.0, timeout_minutes=30.0)
    obs = _dict_to_obs(_obs_fields(price_at_record=734.00, atr_at_record=1.55))
    r   = s.resolve(obs, current_price=734.50, atr=1.55, elapsed_minutes=31.0)
    assert r == "inconclusive"


def test_atr_target_none_when_no_atr():
    """Without ATR, strategy cannot compute — returns None."""
    from memory.outcome_tracker import ATRTargetStrategy, _dict_to_obs
    s   = ATRTargetStrategy(target_atr=1.0)
    obs = _dict_to_obs(_obs_fields(price_at_record=734.00, atr_at_record=None))
    r   = s.resolve(obs, current_price=738.00, atr=None, elapsed_minutes=5.0)
    assert r is None


# ══════════════════════════════════════════════════════════════
# ResolutionStrategy — TimeExpirationStrategy
# ══════════════════════════════════════════════════════════════

def test_time_expiry_none_before_expiry():
    from memory.outcome_tracker import TimeExpirationStrategy, _dict_to_obs
    s   = TimeExpirationStrategy(expiry_minutes=60.0)
    obs = _dict_to_obs(_obs_fields(price_at_record=734.00))
    r   = s.resolve(obs, current_price=734.00, atr=1.55, elapsed_minutes=30.0)
    assert r is None


def test_time_expiry_bullish_when_price_above_entry():
    from memory.outcome_tracker import TimeExpirationStrategy, _dict_to_obs
    s   = TimeExpirationStrategy(expiry_minutes=60.0)
    obs = _dict_to_obs(_obs_fields(price_at_record=734.00))
    r   = s.resolve(obs, current_price=737.00, atr=1.55, elapsed_minutes=61.0)
    assert r == "bullish"


def test_time_expiry_bearish_when_price_below_entry():
    from memory.outcome_tracker import TimeExpirationStrategy, _dict_to_obs
    s   = TimeExpirationStrategy(expiry_minutes=60.0)
    obs = _dict_to_obs(_obs_fields(price_at_record=734.00))
    r   = s.resolve(obs, current_price=731.00, atr=1.55, elapsed_minutes=61.0)
    assert r == "bearish"


def test_time_expiry_inconclusive_when_price_unchanged():
    from memory.outcome_tracker import TimeExpirationStrategy, _dict_to_obs
    s   = TimeExpirationStrategy(expiry_minutes=60.0)
    obs = _dict_to_obs(_obs_fields(price_at_record=734.00))
    r   = s.resolve(obs, current_price=734.00, atr=1.55, elapsed_minutes=61.0)
    assert r == "inconclusive"


# ══════════════════════════════════════════════════════════════
# ResolutionStrategy — StructureContinuationStrategy
# ══════════════════════════════════════════════════════════════

def test_structure_continuation_delegates_to_session_break():
    from memory.outcome_tracker import StructureContinuationStrategy, _dict_to_obs
    s   = StructureContinuationStrategy(timeout_minutes=120.0)
    obs = _dict_to_obs(_obs_fields(session_high=738.39, session_low=730.50))
    r   = s.resolve(obs, current_price=739.00, atr=1.55, elapsed_minutes=5.0)
    assert r == "bullish"


def test_structure_continuation_inconclusive_on_timeout():
    from memory.outcome_tracker import StructureContinuationStrategy, _dict_to_obs
    s   = StructureContinuationStrategy(timeout_minutes=30.0)
    obs = _dict_to_obs(_obs_fields(session_high=738.39, session_low=730.50))
    r   = s.resolve(obs, current_price=734.00, atr=1.55, elapsed_minutes=31.0)
    assert r == "inconclusive"


# ══════════════════════════════════════════════════════════════
# record_observation — field population
# ══════════════════════════════════════════════════════════════

def test_record_returns_obs_id():
    t   = _tracker()
    oid = t.record_observation(_snap(), [_pat("Failed Breakout")])
    assert isinstance(oid, str) and len(oid) > 0


def test_record_increments_count():
    t = _tracker()
    t.record_observation(_snap(), [_pat("X")])
    assert t.observation_count == 1


def test_record_fields_from_dict_snap():
    t   = _tracker()
    oid = t.record_observation(
        _snap(direction="downtrend", session_high=738.39, session_low=730.50,
              current_price=734.00, atr_5m=1.55,
              regime_name="trend_bear", sequence="LH LL",
              confidence_score=54.0, trigger="regime_change"),
        [_pat("Failed Breakout", "bearish")],
        session="after_hours", symbol="SPY",
    )
    obs = t.get_observation(oid)
    assert obs.direction        == "downtrend"
    assert obs.session_high     == 738.39
    assert obs.session_low      == 730.50
    assert obs.price_at_record  == 734.00
    assert obs.atr_at_record    == 1.55
    assert obs.session          == "after_hours"
    assert obs.symbol           == "SPY"
    assert obs.prediction       == "bearish"
    assert "Failed Breakout"    in obs.pattern_names
    assert obs.confidence_score == 54.0


def test_record_prediction_bearish():
    t   = _tracker()
    oid = t.record_observation(_snap(), [_pat("Failed Breakout", "bearish")])
    assert t.get_observation(oid).prediction == "bearish"


def test_record_prediction_bullish():
    t   = _tracker()
    oid = t.record_observation(_snap(), [_pat("VWAP Reclaim", "bullish")])
    assert t.get_observation(oid).prediction == "bullish"


def test_record_prediction_neutral_no_patterns():
    t   = _tracker()
    oid = t.record_observation(_snap(), [])
    assert t.get_observation(oid).prediction == "neutral"


def test_record_multiple_pattern_names():
    t   = _tracker()
    oid = t.record_observation(
        _snap(),
        [_pat("Failed Breakout", "bearish"), _pat("VWAP Rejection", "bearish")],
    )
    obs = t.get_observation(oid)
    assert "Failed Breakout"  in obs.pattern_names
    assert "VWAP Rejection"   in obs.pattern_names


def test_record_accepts_dict_patterns():
    t   = _tracker()
    oid = t.record_observation(_snap(), [_pat_dict("Failed Breakout", "bearish")])
    obs = t.get_observation(oid)
    assert obs.prediction == "bearish"
    assert "Failed Breakout" in obs.pattern_names


def test_record_starts_unresolved():
    t   = _tracker()
    oid = t.record_observation(_snap(), [_pat("X")])
    obs = t.get_observation(oid)
    assert obs.resolved is False
    assert obs.outcome is None


def test_record_price_trackers_init_to_entry():
    t   = _tracker()
    oid = t.record_observation(_snap(current_price=734.00), [_pat("X")])
    obs = t.get_observation(oid)
    assert obs.price_high_seen == 734.00
    assert obs.price_low_seen  == 734.00


def test_record_accepts_object_snapshot():
    """Duck-typed snapshot object (e.g. MarketSnapshot)."""
    t = _tracker()

    class _Snap:
        direction        = "uptrend"
        market_state     = "trend"
        sequence         = "HH HL HH HL"
        confidence_score = 60.0
        regime_name      = "trend_bull"
        trigger          = "regime_change"
        session_high     = 549.48
        session_low      = 541.19
        current_price    = 546.00
        atr_5m           = 1.40
        pattern_names    = ("Trend Continuation (Long)",)

    oid = t.record_observation(_Snap(), [_pat("Trend Continuation (Long)", "bullish")])
    obs = t.get_observation(oid)
    assert obs.direction    == "uptrend"
    assert obs.prediction   == "bullish"
    assert obs.session_high == 549.48


# ══════════════════════════════════════════════════════════════
# update_pending — resolution via strategy
# ══════════════════════════════════════════════════════════════

def test_update_pending_returns_zero_when_empty():
    t = _tracker()
    assert t.update_pending(735.00) == 0


def test_update_pending_resolves_bullish():
    t   = _tracker()
    oid = t.record_observation(
        _snap(session_high=738.39, session_low=730.50, current_price=734.00),
        [_pat("X", "bullish")],
    )
    n   = t.update_pending(739.00)
    obs = t.get_observation(oid)
    assert n == 1
    assert obs.resolved is True
    assert obs.outcome  == "bullish"


def test_update_pending_resolves_bearish():
    t   = _tracker()
    oid = t.record_observation(
        _snap(session_high=738.39, session_low=730.50, current_price=734.00),
        [_pat("X", "bearish")],
    )
    n   = t.update_pending(730.00)
    obs = t.get_observation(oid)
    assert n == 1
    assert obs.outcome == "bearish"


def test_update_pending_resolves_inconclusive_on_timeout():
    t   = _tracker(timeout=0.0001)
    oid = t.record_observation(
        _snap(session_high=738.39, session_low=730.50, current_price=734.00),
        [_pat("X", "neutral")],
    )
    n   = t.update_pending(734.00)
    obs = t.get_observation(oid)
    assert n == 1
    assert obs.outcome == "inconclusive"


def test_update_pending_no_resolve_inside_range():
    t   = _tracker(timeout=120.0)
    oid = t.record_observation(
        _snap(session_high=738.39, session_low=730.50, current_price=734.00),
        [_pat("X")],
    )
    n   = t.update_pending(735.00)
    obs = t.get_observation(oid)
    assert n == 0
    assert obs.resolved is False


def test_update_pending_records_resolution_price():
    t   = _tracker()
    oid = t.record_observation(
        _snap(session_high=738.39, session_low=730.50, current_price=734.00),
        [_pat("X")],
    )
    t.update_pending(739.00)
    assert t.get_observation(oid).resolution_price == 739.00


def test_update_pending_does_not_re_resolve():
    t   = _tracker()
    oid = t.record_observation(
        _snap(session_high=738.39, session_low=730.50, current_price=734.00),
        [_pat("X")],
    )
    t.update_pending(739.00)
    n = t.update_pending(739.00)
    assert n == 0


def test_update_pending_resolves_multiple():
    t = _tracker()
    for _ in range(3):
        t.record_observation(
            _snap(session_high=738.39, session_low=730.50, current_price=734.00),
            [_pat("X")],
        )
    n = t.update_pending(739.00)
    assert n == 3


# ══════════════════════════════════════════════════════════════
# update_pending — price trackers
# ══════════════════════════════════════════════════════════════

def test_price_high_seen_updated():
    t   = _tracker(timeout=120.0)
    oid = t.record_observation(
        _snap(session_high=738.39, session_low=730.50, current_price=734.00),
        [_pat("X")],
    )
    t.update_pending(736.00)
    assert t.get_observation(oid).price_high_seen == 736.00


def test_price_low_seen_updated():
    t   = _tracker(timeout=120.0)
    oid = t.record_observation(
        _snap(session_high=738.39, session_low=730.50, current_price=734.00),
        [_pat("X")],
    )
    t.update_pending(732.00)
    assert t.get_observation(oid).price_low_seen == 732.00


def test_price_high_seen_tracks_maximum():
    t   = _tracker(timeout=120.0)
    oid = t.record_observation(
        _snap(session_high=738.39, session_low=730.50, current_price=734.00),
        [_pat("X")],
    )
    t.update_pending(736.00)
    t.update_pending(733.00)   # lower — should not overwrite high
    assert t.get_observation(oid).price_high_seen == 736.00


def test_price_low_seen_tracks_minimum():
    t   = _tracker(timeout=120.0)
    oid = t.record_observation(
        _snap(session_high=738.39, session_low=730.50, current_price=734.00),
        [_pat("X")],
    )
    t.update_pending(732.00)
    t.update_pending(736.00)   # higher — should not overwrite low
    assert t.get_observation(oid).price_low_seen == 732.00


# ══════════════════════════════════════════════════════════════
# update_pending — MFE / MAE
# ══════════════════════════════════════════════════════════════

def test_bearish_mfe_positive_when_price_drops():
    t   = _tracker()
    oid = t.record_observation(
        _snap(session_high=738.39, session_low=730.50,
              current_price=734.00, atr_5m=1.55),
        [_pat("Failed Breakout", "bearish")],
    )
    t.update_pending(732.00)          # favorable move down
    t.update_pending(730.00, atr=1.55)  # resolution
    obs = t.get_observation(oid)
    assert obs.mfe_atr is not None
    assert obs.mfe_atr > 0.0


def test_bullish_mfe_positive_when_price_rises():
    t   = _tracker()
    oid = t.record_observation(
        _snap(session_high=738.39, session_low=730.50,
              current_price=734.00, atr_5m=1.55),
        [_pat("VWAP Reclaim", "bullish")],
    )
    t.update_pending(736.00)
    t.update_pending(739.00, atr=1.55)
    obs = t.get_observation(oid)
    assert obs.mfe_atr is not None
    assert obs.mfe_atr > 0.0


def test_mfe_none_when_no_atr_available():
    t   = _tracker()
    oid = t.record_observation(
        _snap(session_high=738.39, session_low=730.50,
              current_price=734.00, atr_5m=None),
        [_pat("X", "bearish")],
    )
    t.update_pending(730.00, atr=None)
    obs = t.get_observation(oid)
    assert obs.mfe_atr is None
    assert obs.mae_atr is None


def test_mfe_mae_computed_at_resolution():
    """Both MFE and MAE must be present after a resolved observation."""
    t   = _tracker()
    oid = t.record_observation(
        _snap(session_high=738.39, session_low=730.50,
              current_price=734.00, atr_5m=1.55),
        [_pat("X", "bearish")],
    )
    t.update_pending(736.00, atr=1.55)   # adverse (price goes up)
    t.update_pending(730.00, atr=1.55)   # resolution (bearish)
    obs = t.get_observation(oid)
    assert obs.mfe_atr is not None
    assert obs.mae_atr is not None
    # MAE should reflect the adverse move from 734 → 736
    assert obs.mae_atr > 0.0


# ══════════════════════════════════════════════════════════════
# record_outcome (manual override)
# ══════════════════════════════════════════════════════════════

def test_record_outcome_returns_true_when_found():
    t   = _tracker()
    oid = t.record_observation(_snap(), [_pat("X")])
    assert t.record_outcome(oid, "bearish") is True


def test_record_outcome_returns_false_when_not_found():
    t = _tracker()
    assert t.record_outcome("nonexistent_id", "bearish") is False


def test_record_outcome_marks_resolved():
    t   = _tracker()
    oid = t.record_observation(_snap(), [_pat("X")])
    t.record_outcome(oid, "bullish", resolution_minutes=38.0,
                     mfe_atr=1.3, mae_atr=0.2)
    obs = t.get_observation(oid)
    assert obs.resolved           is True
    assert obs.outcome            == "bullish"
    assert obs.resolution_minutes == 38.0
    assert obs.mfe_atr            == 1.3
    assert obs.mae_atr            == 0.2


def test_record_outcome_appears_in_stats():
    t   = _tracker()
    oid = t.record_observation(_snap(), [_pat("Failed Breakout", "bearish")])
    t.record_outcome(oid, "bearish", resolution_minutes=46.0)
    s   = t.get_stats("Failed Breakout")
    assert s is not None
    assert s.bearish_outcomes         == 1
    assert s.avg_resolution_minutes   == 46.0


# ══════════════════════════════════════════════════════════════
# PatternStats — rates and accuracy
# ══════════════════════════════════════════════════════════════

def test_get_stats_returns_none_when_no_resolved():
    t = _tracker()
    t.record_observation(_snap(), [_pat("Failed Breakout")])
    assert t.get_stats("Failed Breakout") is None


def test_get_stats_returns_none_for_unknown_pattern():
    t = _tracker()
    assert t.get_stats("Unknown Pattern XYZ") is None


def test_get_stats_after_resolution():
    t   = _tracker()
    oid = t.record_observation(
        _snap(session_high=738.39, session_low=730.50, current_price=734.00),
        [_pat("Failed Breakout", "bearish")],
    )
    t.update_pending(730.00)
    s = t.get_stats("Failed Breakout")
    assert s is not None
    assert s.occurrences      == 1
    assert s.resolved_count   == 1
    assert s.bearish_outcomes == 1
    assert s.bullish_outcomes == 0


def test_get_stats_occurrences_includes_pending():
    t = _tracker()
    for price in (730.00, None):
        oid = t.record_observation(
            _snap(session_high=738.39, session_low=730.50, current_price=734.00),
            [_pat("Failed Breakout", "bearish")],
        )
        if price:
            t.update_pending(price)
    s = t.get_stats("Failed Breakout")
    assert s.occurrences    == 2
    assert s.resolved_count == 1


def test_stats_correct_prediction_counted():
    t   = _tracker()
    oid = t.record_observation(
        _snap(session_high=738.39, session_low=730.50, current_price=734.00),
        [_pat("Failed Breakout", "bearish")],
    )
    t.update_pending(730.00)   # bearish prediction + bearish outcome = correct
    s = t.get_stats("Failed Breakout")
    assert s.correct_predictions == 1
    assert s.total_directional   == 1
    assert s.prediction_accuracy == 1.0


def test_stats_wrong_prediction():
    t   = _tracker()
    oid = t.record_observation(
        _snap(session_high=738.39, session_low=730.50, current_price=734.00),
        [_pat("Failed Breakout", "bearish")],
    )
    t.update_pending(739.00)   # bearish prediction + bullish outcome = wrong
    s = t.get_stats("Failed Breakout")
    assert s.correct_predictions == 0
    assert s.prediction_accuracy == 0.0


def test_stats_inconclusive_excluded_from_directional():
    t   = _tracker(timeout=0.0001)
    oid = t.record_observation(
        _snap(session_high=738.39, session_low=730.50, current_price=734.00),
        [_pat("Failed Breakout", "bearish")],
    )
    t.update_pending(734.00)   # timeout → inconclusive
    s = t.get_stats("Failed Breakout")
    assert s.inconclusive_count == 1
    assert s.total_directional  == 0


def test_stats_neutral_excluded_from_directional():
    t   = _tracker()
    oid = t.record_observation(
        _snap(session_high=738.39, session_low=730.50, current_price=734.00),
        [_pat("Range Compression", "neutral")],
    )
    t.update_pending(739.00)
    s = t.get_stats("Range Compression")
    assert s.total_directional == 0


def test_stats_bullish_rate_calculation():
    t = _tracker()
    for price, bias in [(739.00, "bullish"), (739.00, "bullish"), (730.00, "bearish")]:
        oid = t.record_observation(
            _snap(session_high=738.39, session_low=730.50, current_price=734.00),
            [_pat("X", bias)],
        )
        t.update_pending(price)
    s = t.get_stats("X")
    assert s.resolved_count    == 3
    assert s.bullish_outcomes  == 2
    assert s.bearish_outcomes  == 1
    assert abs(s.bullish_rate - 2/3) < 0.01
    assert abs(s.bearish_rate - 1/3) < 0.01


def test_stats_avg_confidence():
    t = _tracker()
    for conf in (60.0, 80.0):
        oid = t.record_observation(
            _snap(session_high=738.39, session_low=730.50, current_price=734.00,
                  confidence_score=conf),
            [_pat("X")],
        )
        t.update_pending(739.00)
    s = t.get_stats("X")
    assert abs(s.avg_confidence - 70.0) < 0.01


def test_stats_str_has_required_fields():
    t   = _tracker()
    oid = t.record_observation(
        _snap(session_high=738.39, session_low=730.50, current_price=734.00),
        [_pat("Failed Breakout")],
    )
    t.update_pending(730.00)
    txt = str(t.get_stats("Failed Breakout"))
    assert "Failed Breakout"   in txt
    assert "Occurrences"       in txt
    assert "Bullish"           in txt
    assert "Bearish"           in txt
    assert "Average"           in txt


def test_rates_default_half_when_no_resolved():
    from memory.outcome_tracker import PatternStats
    s = PatternStats(
        pattern_name="X", occurrences=0, resolved_count=0,
        bullish_outcomes=0, bearish_outcomes=0, inconclusive_count=0,
        correct_predictions=0, total_directional=0,
        avg_resolution_minutes=0.0, avg_mfe_atr=0.0,
        avg_mae_atr=0.0, avg_confidence=0.0,
    )
    assert s.bullish_rate        == 0.5
    assert s.bearish_rate        == 0.5
    assert s.prediction_accuracy == 0.5


# ══════════════════════════════════════════════════════════════
# all_stats / similar_patterns / summary_text
# ══════════════════════════════════════════════════════════════

def test_all_stats_empty_when_no_resolved():
    t = _tracker()
    t.record_observation(_snap(), [_pat("X")])
    assert t.all_stats() == {}


def test_all_stats_includes_resolved_patterns():
    t = _tracker()
    for name, price in [("Failed Breakout", 730.00), ("VWAP Reclaim", 739.00)]:
        oid = t.record_observation(
            _snap(session_high=738.39, session_low=730.50, current_price=734.00),
            [_pat(name)],
        )
        t.update_pending(price)
    stats = t.all_stats()
    assert "Failed Breakout" in stats
    assert "VWAP Reclaim"    in stats


def test_similar_patterns_from_dict():
    t   = _tracker()
    oid = t.record_observation(
        _snap(session_high=738.39, session_low=730.50, current_price=734.00),
        [_pat("Failed Breakout", "bearish")],
    )
    t.update_pending(730.00)
    result = t.similar_patterns({"pattern_names": ["Failed Breakout"]})
    assert len(result) == 1
    assert result[0].pattern_name == "Failed Breakout"


def test_similar_patterns_from_object():
    t   = _tracker()
    oid = t.record_observation(
        _snap(session_high=738.39, session_low=730.50, current_price=734.00),
        [_pat("Range Compression", "neutral")],
    )
    t.update_pending(739.00)

    class _S:
        pattern_names = ("Range Compression",)

    result = t.similar_patterns(_S())
    assert len(result) == 1


def test_similar_patterns_ordered_by_occurrences():
    t = _tracker()
    for _ in range(2):
        oid = t.record_observation(
            _snap(session_high=738.39, session_low=730.50, current_price=734.00),
            [_pat("X", "bearish")],
        )
        t.update_pending(730.00)
    oid = t.record_observation(
        _snap(session_high=738.39, session_low=730.50, current_price=734.00),
        [_pat("Y", "bullish")],
    )
    t.update_pending(739.00)
    result = t.similar_patterns({"pattern_names": ["X", "Y"]})
    assert result[0].pattern_name == "X"


def test_similar_patterns_empty_for_unknown():
    t = _tracker()
    assert t.similar_patterns({"pattern_names": ["No Such Pattern"]}) == []


def test_summary_text_has_pattern_name():
    t   = _tracker()
    oid = t.record_observation(
        _snap(session_high=738.39, session_low=730.50, current_price=734.00),
        [_pat("Failed Breakout")],
    )
    t.update_pending(730.00)
    assert "Failed Breakout" in t.summary_text("Failed Breakout")


def test_summary_text_no_obs():
    t = _tracker()
    assert "no resolved" in t.summary_text("Unknown").lower()


# ══════════════════════════════════════════════════════════════
# pending_observations / recent_observations
# ══════════════════════════════════════════════════════════════

def test_pending_returns_unresolved():
    t = _tracker()
    t.record_observation(_snap(), [_pat("X")])
    t.record_observation(_snap(), [_pat("Y")])
    assert len(t.pending_observations()) == 2


def test_pending_excludes_resolved():
    t = _tracker()
    oid = t.record_observation(
        _snap(session_high=738.39, session_low=730.50, current_price=734.00),
        [_pat("X")],
    )
    t.record_observation(_snap(session_high=None, session_low=None), [_pat("Y")])
    t.update_pending(730.00)   # resolves first only; second has no key levels
    pending = t.pending_observations()
    assert all(not o.resolved for o in pending)
    assert len(pending) == 1


def test_recent_honours_n():
    t = _tracker()
    for _ in range(5):
        t.record_observation(_snap(), [_pat("X")])
    assert len(t.recent_observations(n=3)) == 3


def test_recent_newest_first():
    t = _tracker()
    for _ in range(3):
        t.record_observation(_snap(), [_pat("X")])
    recent = t.recent_observations(n=3)
    ts = [r.recorded_at for r in recent]
    assert ts == sorted(ts, reverse=True)


# ══════════════════════════════════════════════════════════════
# Persistence (save + load round-trip)
# ══════════════════════════════════════════════════════════════

def test_save_load_roundtrip():
    from memory.outcome_tracker import OutcomeTracker
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as fh:
        path = fh.name
    try:
        t1  = OutcomeTracker(path=path)
        oid = t1.record_observation(
            _snap(session_high=738.39, session_low=730.50, current_price=734.00),
            [_pat("Failed Breakout", "bearish")],
            session="open",
        )
        t1.record_outcome(oid, "bearish", resolution_minutes=38.0,
                          mfe_atr=1.2, mae_atr=0.3)
        t1.save()

        t2  = OutcomeTracker(path=path)
        assert t2.observation_count == 1
        obs = t2.get_observation(oid)
        assert obs is not None
        assert obs.outcome            == "bearish"
        assert obs.resolution_minutes == 38.0
        assert obs.mfe_atr            == 1.2
        assert obs.session            == "open"
        assert "Failed Breakout" in obs.pattern_names
    finally:
        os.unlink(path)
        try: os.unlink(path + ".tmp")
        except FileNotFoundError: pass


def test_load_missing_file_no_crash():
    from memory.outcome_tracker import OutcomeTracker
    t = OutcomeTracker(path="/nonexistent/path/outcome_tracker.json")
    assert t.observation_count == 0


def test_memory_mode_save_noop():
    t = _tracker()
    t.record_observation(_snap(), [_pat("X")])
    t.save()     # must not raise
    assert t._path is None


# ══════════════════════════════════════════════════════════════
# Rolling trim
# ══════════════════════════════════════════════════════════════

def test_trim_keeps_max():
    from memory.outcome_tracker import OutcomeTracker
    t = OutcomeTracker(path=":memory:", max_observations=3)
    for _ in range(5):
        t.record_observation(_snap(), [_pat("X")])
    assert t.observation_count == 3


def test_trim_evicts_resolved_before_pending():
    from memory.outcome_tracker import OutcomeTracker
    t    = OutcomeTracker(path=":memory:", max_observations=2)
    oid1 = t.record_observation(
        _snap(session_high=738.39, session_low=730.50, current_price=734.00),
        [_pat("X")],
    )
    t.update_pending(739.00)     # resolve oid1
    oid2 = t.record_observation(_snap(), [_pat("Y")])   # pending
    oid3 = t.record_observation(_snap(), [_pat("Z")])   # pending → triggers trim
    assert t.observation_count     == 2
    assert t.get_observation(oid1) is None   # evicted (resolved, oldest)
    assert t.get_observation(oid2) is not None
    assert t.get_observation(oid3) is not None


# ══════════════════════════════════════════════════════════════
# Singleton
# ══════════════════════════════════════════════════════════════

def test_get_tracker_same_instance():
    from memory.outcome_tracker import get_tracker, reset_tracker
    reset_tracker(":memory:")
    assert get_tracker() is get_tracker()


def test_reset_tracker_fresh_store():
    from memory.outcome_tracker import get_tracker, reset_tracker
    t1 = reset_tracker(":memory:")
    t1.record_observation(_snap(), [_pat("X")])
    t2 = reset_tracker(":memory:")
    assert t2.observation_count == 0


# ══════════════════════════════════════════════════════════════
# PatternObservation properties
# ══════════════════════════════════════════════════════════════

def test_prediction_correct_true():
    obs = _make_obs(prediction="bearish", outcome="bearish")
    assert obs.prediction_correct is True


def test_prediction_correct_false():
    obs = _make_obs(prediction="bearish", outcome="bullish")
    assert obs.prediction_correct is False


def test_prediction_correct_none_for_inconclusive():
    obs = _make_obs(prediction="bearish", outcome="inconclusive")
    assert obs.prediction_correct is None


def test_prediction_correct_none_for_neutral():
    obs = _make_obs(prediction="neutral", outcome="bullish")
    assert obs.prediction_correct is None


def test_prediction_correct_none_when_unresolved():
    obs = _make_obs(prediction="bearish", outcome=None)
    assert obs.prediction_correct is None


def test_elapsed_minutes_positive():
    obs = _make_obs()
    assert obs.elapsed_minutes >= 0.0


# ══════════════════════════════════════════════════════════════
# ATRTargetStrategy integration via OutcomeTracker
# ══════════════════════════════════════════════════════════════

def test_atr_strategy_resolves_via_tracker():
    from memory.outcome_tracker import OutcomeTracker, ATRTargetStrategy
    t   = OutcomeTracker(
        path=":memory:",
        strategy=ATRTargetStrategy(target_atr=1.0, timeout_minutes=120.0),
    )
    oid = t.record_observation(
        _snap(current_price=734.00, atr_5m=1.55,
              session_high=800.00, session_low=700.00),  # wide — won't trigger session break
        [_pat("X", "bullish")],
    )
    t.update_pending(735.60, atr=1.55)   # 1.6 pts / 1.55 ATR ≈ 1.03 → bullish
    obs = t.get_observation(oid)
    assert obs.resolved is True
    assert obs.outcome  == "bullish"


def test_time_expiry_strategy_via_tracker():
    from memory.outcome_tracker import OutcomeTracker, TimeExpirationStrategy
    t   = OutcomeTracker(
        path=":memory:",
        strategy=TimeExpirationStrategy(expiry_minutes=0.0001),
    )
    oid = t.record_observation(
        _snap(current_price=734.00, session_high=800.00, session_low=700.00),
        [_pat("X", "bullish")],
    )
    t.update_pending(737.00, atr=1.55)   # expiry elapsed + price above entry
    obs = t.get_observation(oid)
    assert obs.resolved is True
    assert obs.outcome  == "bullish"


# ──────────────────────────────────────────────────────────────
# Internal test helpers
# ──────────────────────────────────────────────────────────────

def _obs_fields(**kw) -> dict:
    """Minimal dict to pass to _dict_to_obs for strategy unit tests."""
    base = dict(
        obs_id="test-obs", recorded_at=datetime.now(tz=timezone.utc).isoformat(),
        symbol="SPY", session="open", trigger="t",
        regime_name="trend_bear", direction="downtrend",
        market_state="trend", sequence="LH LL",
        confidence_score=54.0, pattern_names=["X"], prediction="bearish",
        session_high=None, session_low=None,
        price_at_record=None, atr_at_record=None,
    )
    base.update(kw)
    return base


def _make_obs(**kw) -> "PatternObservation":
    from memory.outcome_tracker import PatternObservation
    base = dict(
        obs_id="test", recorded_at=datetime.now(tz=timezone.utc).isoformat(),
        symbol="SPY", session="open", trigger="t",
        regime_name="", direction="", market_state="", sequence="",
        confidence_score=50.0, pattern_names=[], prediction="bearish",
        session_high=None, session_low=None,
        price_at_record=None, atr_at_record=None,
        resolved=True, outcome=None,
    )
    base.update(kw)
    return PatternObservation(**base)


# ──────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        (n, f) for n, f in sorted(globals().items())
        if n.startswith("test_") and callable(f)
    ]
    print(f"\nOutcome tracker tests ({len(tests)} total)\n{'─' * 60}")
    for name, fn in tests:
        run(name, fn)
    print(f"\n{'=' * 60}")
    print(f"{passed} passed, {failed} failed  ({passed + failed} total)")
    import sys; sys.exit(0 if failed == 0 else 1)