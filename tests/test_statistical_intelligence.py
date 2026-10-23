"""
tests/test_statistical_intelligence.py — Tests for memory/statistical_intelligence.py

Coverage:
  get_pattern_stats                 — delegates to OutcomeTracker
  get_confidence_bucket_stats       — bucketing, rates, accuracy
  get_directional_stats             — win rate, exclusion of inconclusive
  get_historical_context            — aggregation, dominant outcome, accuracy
  get_historical_context_from_package — package dict path
  calibrate_confidence              — adjustment magnitude, clamping, explanation
  summary_text                      — format, insufficient history, full block
  HistoricalContext.has_sufficient_history
  CalibrationResult fields
  Edge cases: empty tracker, single obs, all inconclusive, neutral prediction

Run: PYTHONPATH=. python tests/test_statistical_intelligence.py
"""

import os
import sys
import traceback

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

def _fresh_tracker():
    from memory.outcome_tracker import OutcomeTracker
    return OutcomeTracker(path=":memory:")


def _pat(name: str, bias: str = "bearish"):
    class _P:
        pass
    p = _P()
    p.name = name
    p.bias = bias
    return p


def _snap_dict(**kw):
    base = dict(
        session_high=738.39, session_low=730.50,
        current_price=734.00, atr_5m=1.55,
        confidence_score=54.0,
        pattern_names=["Failed Breakout"],
        regime_name="trend_bear",
        trigger="regime_change",
    )
    base.update(kw)
    return base


def _record_and_resolve(tracker, pattern_name, bias, outcome,
                        confidence=60.0, mfe_atr=1.2, mae_atr=0.3,
                        resolution_minutes=42.0):
    """Helper: record one observation and manually resolve it."""
    oid = tracker.record_observation(
        _snap_dict(
            session_high=738.39, session_low=730.50,
            current_price=734.00, atr_5m=1.55,
            confidence_score=confidence,
        ),
        [_pat(pattern_name, bias)],
    )
    tracker.record_outcome(
        oid, outcome,
        resolution_minutes=resolution_minutes,
        mfe_atr=mfe_atr,
        mae_atr=mae_atr,
    )
    return oid


def _populated_tracker(n_bearish=7, n_bullish=3):
    """
    Returns a tracker with n_bearish resolved-bearish and n_bullish
    resolved-bullish observations, all for 'Failed Breakout' pattern.
    """
    t = _fresh_tracker()
    for _ in range(n_bearish):
        _record_and_resolve(t, "Failed Breakout", "bearish", "bearish",
                            confidence=68.0, mfe_atr=1.3, mae_atr=0.2,
                            resolution_minutes=42.0)
    for _ in range(n_bullish):
        _record_and_resolve(t, "Failed Breakout", "bearish", "bullish",
                            confidence=68.0, mfe_atr=0.4, mae_atr=0.8,
                            resolution_minutes=55.0)
    return t


# ══════════════════════════════════════════════════════════════
# get_pattern_stats — delegates to OutcomeTracker
# ══════════════════════════════════════════════════════════════

def test_get_pattern_stats_none_when_empty():
    from memory.statistical_intelligence import get_pattern_stats
    t = _fresh_tracker()
    assert get_pattern_stats("Failed Breakout", tracker=t) is None


def test_get_pattern_stats_returns_stats():
    from memory.statistical_intelligence import get_pattern_stats
    t = _populated_tracker(n_bearish=3, n_bullish=1)
    s = get_pattern_stats("Failed Breakout", tracker=t)
    assert s is not None
    assert s.pattern_name   == "Failed Breakout"
    assert s.resolved_count == 4


def test_get_pattern_stats_unknown_pattern():
    from memory.statistical_intelligence import get_pattern_stats
    t = _populated_tracker()
    assert get_pattern_stats("Nonexistent Pattern XYZ", tracker=t) is None


# ══════════════════════════════════════════════════════════════
# get_confidence_bucket_stats — bucketing
# ══════════════════════════════════════════════════════════════

def test_bucket_none_when_no_obs():
    from memory.statistical_intelligence import get_confidence_bucket_stats
    t = _fresh_tracker()
    assert get_confidence_bucket_stats(68.0, tracker=t) is None


def test_bucket_returns_stats_for_correct_bucket():
    from memory.statistical_intelligence import get_confidence_bucket_stats
    t = _fresh_tracker()
    # confidence=68 → bucket 65–74
    _record_and_resolve(t, "X", "bearish", "bearish", confidence=68.0)
    _record_and_resolve(t, "X", "bearish", "bearish", confidence=70.0)
    s = get_confidence_bucket_stats(68.0, tracker=t)
    assert s is not None
    assert s.bucket_low  == 65
    assert s.bucket_high == 74
    assert s.sample_size == 2


def test_bucket_does_not_mix_obs_from_other_bucket():
    from memory.statistical_intelligence import get_confidence_bucket_stats
    t = _fresh_tracker()
    _record_and_resolve(t, "X", "bearish", "bearish", confidence=68.0)  # 65–74
    _record_and_resolve(t, "X", "bearish", "bullish", confidence=55.0)  # 55–64
    s = get_confidence_bucket_stats(68.0, tracker=t)
    assert s.sample_size == 1


def test_bucket_bullish_rate():
    from memory.statistical_intelligence import get_confidence_bucket_stats
    t = _fresh_tracker()
    for _ in range(3):
        _record_and_resolve(t, "X", "bullish", "bullish", confidence=60.0)
    for _ in range(1):
        _record_and_resolve(t, "X", "bullish", "bearish", confidence=62.0)
    s = get_confidence_bucket_stats(60.0, tracker=t)
    assert abs(s.bullish_rate - 0.75) < 0.01
    assert abs(s.bearish_rate - 0.25) < 0.01


def test_bucket_prediction_accuracy():
    from memory.statistical_intelligence import get_confidence_bucket_stats
    t = _fresh_tracker()
    # 3 correct bearish predictions, 1 wrong (resolved bullish)
    for _ in range(3):
        _record_and_resolve(t, "X", "bearish", "bearish", confidence=68.0)
    _record_and_resolve(t, "X", "bearish", "bullish", confidence=70.0)
    s = get_confidence_bucket_stats(68.0, tracker=t)
    assert abs(s.prediction_accuracy - 0.75) < 0.01


def test_bucket_inconclusive_excluded_from_accuracy():
    from memory.statistical_intelligence import get_confidence_bucket_stats
    t = _fresh_tracker()
    _record_and_resolve(t, "X", "bearish", "bearish",     confidence=68.0)
    _record_and_resolve(t, "X", "bearish", "inconclusive", confidence=68.0)
    s = get_confidence_bucket_stats(68.0, tracker=t)
    # Only 1 directional observation → accuracy = 1.0
    assert s.prediction_accuracy == 1.0
    assert s.sample_size         == 2
    assert abs(s.inconclusive_rate - 0.5) < 0.01


def test_bucket_avg_mfe_atr():
    from memory.statistical_intelligence import get_confidence_bucket_stats
    t = _fresh_tracker()
    _record_and_resolve(t, "X", "bearish", "bearish", confidence=68.0,
                        mfe_atr=1.0)
    _record_and_resolve(t, "X", "bearish", "bearish", confidence=70.0,
                        mfe_atr=2.0)
    s = get_confidence_bucket_stats(68.0, tracker=t)
    assert abs(s.avg_mfe_atr - 1.5) < 0.01


def test_bucket_label_format():
    from memory.statistical_intelligence import get_confidence_bucket_stats
    t = _fresh_tracker()
    _record_and_resolve(t, "X", "bearish", "bearish", confidence=68.0)
    s = get_confidence_bucket_stats(68.0, tracker=t)
    assert "65" in s.bucket_label
    assert "74" in s.bucket_label


def test_bucket_none_for_out_of_range():
    from memory.statistical_intelligence import get_confidence_bucket_stats
    t = _fresh_tracker()
    # confidence 95 is outside any defined bucket
    assert get_confidence_bucket_stats(95.0, tracker=t) is None


# ══════════════════════════════════════════════════════════════
# get_directional_stats — win rate
# ══════════════════════════════════════════════════════════════

def test_directional_none_when_no_obs():
    from memory.statistical_intelligence import get_directional_stats
    t = _fresh_tracker()
    assert get_directional_stats("bearish", tracker=t) is None


def test_directional_none_for_neutral():
    from memory.statistical_intelligence import get_directional_stats
    t = _populated_tracker()
    assert get_directional_stats("neutral", tracker=t) is None


def test_directional_win_rate_bearish():
    from memory.statistical_intelligence import get_directional_stats
    t = _populated_tracker(n_bearish=7, n_bullish=3)
    s = get_directional_stats("bearish", tracker=t)
    assert s is not None
    assert abs(s.win_rate - 0.7) < 0.01
    assert s.sample_size == 10


def test_directional_win_rate_bullish():
    from memory.statistical_intelligence import get_directional_stats
    t = _fresh_tracker()
    # 4 bullish predictions, 3 resolved bullish (correct), 1 bearish (wrong)
    for _ in range(3):
        _record_and_resolve(t, "X", "bullish", "bullish")
    _record_and_resolve(t, "X", "bullish", "bearish")
    s = get_directional_stats("bullish", tracker=t)
    assert abs(s.win_rate - 0.75) < 0.01


def test_directional_excludes_inconclusive():
    from memory.statistical_intelligence import get_directional_stats
    t = _fresh_tracker()
    _record_and_resolve(t, "X", "bearish", "bearish")
    _record_and_resolve(t, "X", "bearish", "inconclusive")
    s = get_directional_stats("bearish", tracker=t)
    # inconclusive excluded → only 1 conclusive obs
    assert s.sample_size == 1
    assert s.win_rate    == 1.0


def test_directional_avg_mfe():
    from memory.statistical_intelligence import get_directional_stats
    t = _fresh_tracker()
    _record_and_resolve(t, "X", "bearish", "bearish", mfe_atr=1.0)
    _record_and_resolve(t, "X", "bearish", "bearish", mfe_atr=3.0)
    s = get_directional_stats("bearish", tracker=t)
    assert abs(s.avg_mfe_atr - 2.0) < 0.01


def test_directional_avg_resolution_minutes():
    from memory.statistical_intelligence import get_directional_stats
    t = _fresh_tracker()
    _record_and_resolve(t, "X", "bearish", "bearish", resolution_minutes=40.0)
    _record_and_resolve(t, "X", "bearish", "bearish", resolution_minutes=60.0)
    s = get_directional_stats("bearish", tracker=t)
    assert abs(s.avg_resolution_minutes - 50.0) < 0.1


# ══════════════════════════════════════════════════════════════
# get_historical_context — aggregation
# ══════════════════════════════════════════════════════════════

def test_context_empty_tracker():
    from memory.statistical_intelligence import get_historical_context
    t   = _fresh_tracker()
    ctx = get_historical_context(
        {"pattern_names": ["Failed Breakout"], "confidence_score": 68.0,
         "prediction": "bearish"},
        tracker=t,
    )
    assert ctx.total_sample            == 0
    assert ctx.has_sufficient_history  is False
    assert ctx.matched_patterns        == []
    assert ctx.dominant_outcome        is None


def test_context_total_sample():
    from memory.statistical_intelligence import get_historical_context
    t = _populated_tracker(n_bearish=7, n_bullish=3)
    ctx = get_historical_context(
        {"pattern_names": ["Failed Breakout"], "confidence_score": 68.0,
         "prediction": "bearish"},
        tracker=t,
    )
    assert ctx.total_sample == 10


def test_context_dominant_outcome_bearish():
    from memory.statistical_intelligence import get_historical_context
    t   = _populated_tracker(n_bearish=7, n_bullish=3)
    ctx = get_historical_context(
        {"pattern_names": ["Failed Breakout"], "prediction": "bearish"},
        tracker=t,
    )
    assert ctx.dominant_outcome == "bearish"
    assert abs(ctx.dominant_rate - 0.7) < 0.01


def test_context_dominant_outcome_bullish():
    from memory.statistical_intelligence import get_historical_context
    t   = _populated_tracker(n_bearish=2, n_bullish=8)
    ctx = get_historical_context(
        {"pattern_names": ["Failed Breakout"], "prediction": "bearish"},
        tracker=t,
    )
    assert ctx.dominant_outcome == "bullish"


def test_context_overall_accuracy():
    from memory.statistical_intelligence import get_historical_context
    # 7 correct bearish, 3 wrong → accuracy 0.7
    t   = _populated_tracker(n_bearish=7, n_bullish=3)
    ctx = get_historical_context(
        {"pattern_names": ["Failed Breakout"], "prediction": "bearish"},
        tracker=t,
    )
    assert abs(ctx.overall_accuracy - 0.7) < 0.01


def test_context_avg_mfe_weighted():
    from memory.statistical_intelligence import get_historical_context
    t = _fresh_tracker()
    # Pattern A: 2 obs, mfe=1.0  →  contributes weight 2
    for _ in range(2):
        _record_and_resolve(t, "PatA", "bearish", "bearish", mfe_atr=1.0)
    # Pattern B: 2 obs, mfe=3.0  →  contributes weight 2
    for _ in range(2):
        _record_and_resolve(t, "PatB", "bearish", "bearish", mfe_atr=3.0)
    ctx = get_historical_context(
        {"pattern_names": ["PatA", "PatB"], "prediction": "bearish"},
        tracker=t,
    )
    # Weighted avg = (2×1.0 + 2×3.0) / 4 = 2.0
    assert abs(ctx.avg_mfe_atr - 2.0) < 0.05


def test_context_has_sufficient_history_true():
    from memory.statistical_intelligence import get_historical_context, MIN_SAMPLE_SIZE
    t = _fresh_tracker()
    for _ in range(MIN_SAMPLE_SIZE):
        _record_and_resolve(t, "X", "bearish", "bearish")
    ctx = get_historical_context(
        {"pattern_names": ["X"], "prediction": "bearish"}, tracker=t
    )
    assert ctx.has_sufficient_history is True


def test_context_has_sufficient_history_false():
    from memory.statistical_intelligence import get_historical_context, MIN_SAMPLE_SIZE
    t = _fresh_tracker()
    for _ in range(MIN_SAMPLE_SIZE - 1):
        _record_and_resolve(t, "X", "bearish", "bearish")
    ctx = get_historical_context(
        {"pattern_names": ["X"], "prediction": "bearish"}, tracker=t
    )
    assert ctx.has_sufficient_history is False


def test_context_includes_confidence_bucket():
    from memory.statistical_intelligence import get_historical_context
    t = _fresh_tracker()
    _record_and_resolve(t, "X", "bearish", "bearish", confidence=68.0)
    ctx = get_historical_context(
        {"pattern_names": ["X"], "confidence_score": 68.0, "prediction": "bearish"},
        tracker=t,
    )
    assert ctx.confidence_bucket is not None
    assert ctx.confidence_bucket.bucket_low == 65


def test_context_includes_directional():
    from memory.statistical_intelligence import get_historical_context
    t = _fresh_tracker()
    for _ in range(3):
        _record_and_resolve(t, "X", "bearish", "bearish")
    ctx = get_historical_context(
        {"pattern_names": ["X"], "prediction": "bearish"}, tracker=t
    )
    assert ctx.directional is not None
    assert ctx.directional.prediction == "bearish"


def test_context_no_directional_for_neutral():
    from memory.statistical_intelligence import get_historical_context
    t = _fresh_tracker()
    _record_and_resolve(t, "X", "neutral", "bearish")
    ctx = get_historical_context(
        {"pattern_names": ["X"], "prediction": "neutral"}, tracker=t
    )
    assert ctx.directional is None


def test_context_accepts_object_snapshot():
    from memory.statistical_intelligence import get_historical_context
    t = _fresh_tracker()
    for _ in range(5):
        _record_and_resolve(t, "Range Compression", "neutral", "bearish")

    class _Snap:
        pattern_names    = ("Range Compression",)
        confidence_score = 50.0
        prediction       = "neutral"

    ctx = get_historical_context(_Snap(), tracker=t)
    assert ctx.total_sample == 5


def test_context_multiple_patterns_aggregated():
    from memory.statistical_intelligence import get_historical_context
    t = _fresh_tracker()
    for _ in range(4):
        _record_and_resolve(t, "Pattern A", "bearish", "bearish")
    for _ in range(6):
        _record_and_resolve(t, "Pattern B", "bearish", "bearish")
    ctx = get_historical_context(
        {"pattern_names": ["Pattern A", "Pattern B"], "prediction": "bearish"},
        tracker=t,
    )
    assert ctx.total_sample         == 10
    assert len(ctx.matched_patterns) == 2


def test_context_empty_pattern_names():
    from memory.statistical_intelligence import get_historical_context
    t   = _populated_tracker()
    ctx = get_historical_context(
        {"pattern_names": [], "prediction": "bearish"}, tracker=t
    )
    assert ctx.total_sample == 0
    assert ctx.matched_patterns == []


# ══════════════════════════════════════════════════════════════
# get_historical_context_from_package
# ══════════════════════════════════════════════════════════════

def test_from_package_no_crash_empty():
    from memory.statistical_intelligence import get_historical_context_from_package
    t   = _fresh_tracker()
    ctx = get_historical_context_from_package({}, tracker=t)
    assert ctx is not None
    assert ctx.total_sample == 0


def test_from_package_uses_structure():
    from memory.statistical_intelligence import get_historical_context_from_package
    t   = _fresh_tracker()
    pkg = {
        "market": {
            "market_session": "open",
            "regime": {"name": "trend_bear", "confidence": 0.7},
            "session": {"entries_allowed": True},
        },
        "structure": {
            "direction": "downtrend", "market_state": "pullback",
            "trend_intact": True, "sequence": "LH LL LH LL",
            "recent_events": [],
            "key_levels": {
                "session_high": 738.39, "session_low": 730.50,
                "vwap": 734.90, "prior_day_high": 741.00,
                "prior_day_low": 728.00,
                "nearest_support": None, "nearest_resistance": 738.39,
            },
        },
    }
    ctx = get_historical_context_from_package(pkg, tracker=t)
    assert ctx is not None


def test_from_package_returns_historical_context():
    from memory.statistical_intelligence import get_historical_context_from_package, HistoricalContext
    t   = _fresh_tracker()
    ctx = get_historical_context_from_package({"market": {}, "structure": {}}, tracker=t)
    assert isinstance(ctx, HistoricalContext)


# ══════════════════════════════════════════════════════════════
# calibrate_confidence
# ══════════════════════════════════════════════════════════════

def _ctx_with_accuracy(accuracy: float, n: int, pred_n: int = None,
                       pred_win: float = None):
    """Build a minimal HistoricalContext for calibration tests."""
    from memory.statistical_intelligence import HistoricalContext, DirectionalStats
    if pred_n is not None and pred_win is not None:
        ds = DirectionalStats(
            prediction="bearish", sample_size=pred_n, win_rate=pred_win,
            avg_mfe_atr=1.0, avg_mae_atr=0.3, avg_resolution_minutes=40.0,
        )
    else:
        ds = None
    return HistoricalContext(
        matched_patterns=[], confidence_bucket=None, directional=ds,
        total_sample=n, dominant_outcome="bearish", dominant_rate=accuracy,
        overall_accuracy=accuracy, avg_mfe_atr=1.2, avg_mae_atr=0.3,
        avg_resolution_minutes=42.0, has_sufficient_history=(n >= 10),
    )


def test_calibrate_no_adjustment_below_min_sample():
    from memory.statistical_intelligence import calibrate_confidence, MIN_SAMPLE_SIZE
    ctx = _ctx_with_accuracy(0.75, n=MIN_SAMPLE_SIZE - 1)
    r   = calibrate_confidence(70.0, ctx)
    assert r.adjustment         == 0.0
    assert r.final_confidence   == 70.0
    assert r.sufficient_history is False


def test_calibrate_increases_confidence_when_history_supports():
    from memory.statistical_intelligence import calibrate_confidence
    ctx = _ctx_with_accuracy(0.75, n=30)
    r   = calibrate_confidence(70.0, ctx)
    assert r.adjustment       > 0.0
    assert r.final_confidence > 70.0


def test_calibrate_decreases_confidence_when_history_contradicts():
    from memory.statistical_intelligence import calibrate_confidence
    ctx = _ctx_with_accuracy(0.25, n=30)
    r   = calibrate_confidence(70.0, ctx)
    assert r.adjustment       < 0.0
    assert r.final_confidence < 70.0


def test_calibrate_no_adjustment_at_50pct_win_rate():
    from memory.statistical_intelligence import calibrate_confidence
    ctx = _ctx_with_accuracy(0.50, n=30)
    r   = calibrate_confidence(70.0, ctx)
    assert abs(r.adjustment) < 0.5


def test_calibrate_clamped_at_90():
    from memory.statistical_intelligence import calibrate_confidence
    ctx = _ctx_with_accuracy(1.0, n=30)
    r   = calibrate_confidence(88.0, ctx)
    assert r.final_confidence <= 90.0


def test_calibrate_clamped_at_15():
    from memory.statistical_intelligence import calibrate_confidence
    ctx = _ctx_with_accuracy(0.0, n=30)
    r   = calibrate_confidence(20.0, ctx)
    assert r.final_confidence >= 15.0


def test_calibrate_max_adjustment_bounded():
    from memory.statistical_intelligence import (
        calibrate_confidence, MAX_CALIBRATION_ADJUSTMENT,
    )
    ctx = _ctx_with_accuracy(1.0, n=100)
    r   = calibrate_confidence(50.0, ctx)
    assert abs(r.adjustment) <= MAX_CALIBRATION_ADJUSTMENT


def test_calibrate_weight_ramps_with_sample_size():
    from memory.statistical_intelligence import calibrate_confidence
    ctx_small = _ctx_with_accuracy(0.75, n=10)
    ctx_large = _ctx_with_accuracy(0.75, n=30)
    r_small   = calibrate_confidence(70.0, ctx_small)
    r_large   = calibrate_confidence(70.0, ctx_large)
    assert r_large.adjustment > r_small.adjustment


def test_calibrate_uses_directional_win_rate_when_sufficient():
    from memory.statistical_intelligence import calibrate_confidence, MIN_SAMPLE_SIZE
    # directional win_rate=0.80 overrides overall_accuracy=0.60
    ctx = _ctx_with_accuracy(0.60, n=30, pred_n=MIN_SAMPLE_SIZE, pred_win=0.80)
    r   = calibrate_confidence(70.0, ctx)
    assert r.win_rate == 0.80


def test_calibrate_falls_back_to_overall_when_directional_insufficient():
    from memory.statistical_intelligence import calibrate_confidence, MIN_SAMPLE_SIZE
    ctx = _ctx_with_accuracy(0.75, n=30, pred_n=MIN_SAMPLE_SIZE - 1, pred_win=0.80)
    r   = calibrate_confidence(70.0, ctx)
    assert r.win_rate == 0.75   # overall_accuracy used


def test_calibrate_explanation_not_empty():
    from memory.statistical_intelligence import calibrate_confidence
    ctx = _ctx_with_accuracy(0.75, n=30)
    r   = calibrate_confidence(70.0, ctx)
    assert len(r.explanation) > 20


def test_calibrate_explanation_mentions_sample_size():
    from memory.statistical_intelligence import calibrate_confidence
    ctx = _ctx_with_accuracy(0.75, n=25)
    r   = calibrate_confidence(70.0, ctx)
    assert "25" in r.explanation


def test_calibrate_result_fields():
    from memory.statistical_intelligence import calibrate_confidence
    ctx = _ctx_with_accuracy(0.75, n=30)
    r   = calibrate_confidence(72.0, ctx)
    assert r.base_confidence  == 72.0
    assert isinstance(r.final_confidence, float)
    assert isinstance(r.adjustment,       float)
    assert isinstance(r.sufficient_history, bool)
    assert isinstance(r.win_rate,          float)
    assert isinstance(r.sample_size,       int)
    assert isinstance(r.explanation,       str)


# ══════════════════════════════════════════════════════════════
# summary_text
# ══════════════════════════════════════════════════════════════

def test_summary_no_history():
    from memory.statistical_intelligence import summary_text, HistoricalContext
    ctx = HistoricalContext(
        matched_patterns=[], confidence_bucket=None, directional=None,
        total_sample=0, dominant_outcome=None, dominant_rate=0.0,
        overall_accuracy=0.5, avg_mfe_atr=0.0, avg_mae_atr=0.0,
        avg_resolution_minutes=0.0, has_sufficient_history=False,
    )
    txt = summary_text(ctx)
    assert "no resolved" in txt.lower() or "no" in txt.lower()


def test_summary_insufficient_history_shows_count():
    from memory.statistical_intelligence import summary_text, HistoricalContext
    ctx = HistoricalContext(
        matched_patterns=[], confidence_bucket=None, directional=None,
        total_sample=5, dominant_outcome=None, dominant_rate=0.0,
        overall_accuracy=0.5, avg_mfe_atr=0.0, avg_mae_atr=0.0,
        avg_resolution_minutes=0.0, has_sufficient_history=False,
    )
    txt = summary_text(ctx)
    assert "5" in txt


def test_summary_sufficient_history_has_header():
    from memory.statistical_intelligence import (
        get_historical_context, summary_text,
    )
    t = _populated_tracker(n_bearish=8, n_bullish=2)
    ctx = get_historical_context(
        {"pattern_names": ["Failed Breakout"], "prediction": "bearish",
         "confidence_score": 68.0},
        tracker=t,
    )
    txt = summary_text(ctx)
    assert "Historical memory" in txt


def test_summary_contains_pattern_name():
    from memory.statistical_intelligence import get_historical_context, summary_text
    t   = _populated_tracker()
    ctx = get_historical_context(
        {"pattern_names": ["Failed Breakout"], "prediction": "bearish"},
        tracker=t,
    )
    txt = summary_text(ctx)
    assert "Failed Breakout" in txt


def test_summary_contains_obs_count():
    from memory.statistical_intelligence import get_historical_context, summary_text
    t   = _populated_tracker(n_bearish=7, n_bullish=3)
    ctx = get_historical_context(
        {"pattern_names": ["Failed Breakout"], "prediction": "bearish"},
        tracker=t,
    )
    txt = summary_text(ctx)
    assert "10" in txt


def test_summary_contains_bearish_rate():
    from memory.statistical_intelligence import get_historical_context, summary_text
    t   = _populated_tracker(n_bearish=7, n_bullish=3)
    ctx = get_historical_context(
        {"pattern_names": ["Failed Breakout"], "prediction": "bearish"},
        tracker=t,
    )
    txt = summary_text(ctx)
    assert "bearish" in txt.lower()


def test_summary_contains_avg_move():
    from memory.statistical_intelligence import get_historical_context, summary_text
    t   = _populated_tracker()
    ctx = get_historical_context(
        {"pattern_names": ["Failed Breakout"], "prediction": "bearish"},
        tracker=t,
    )
    txt = summary_text(ctx)
    assert "ATR" in txt or "move" in txt.lower()


def test_summary_with_calibration_includes_explanation():
    from memory.statistical_intelligence import (
        get_historical_context, calibrate_confidence, summary_text,
    )
    t   = _populated_tracker(n_bearish=8, n_bullish=2)
    ctx = get_historical_context(
        {"pattern_names": ["Failed Breakout"], "prediction": "bearish",
         "confidence_score": 68.0},
        tracker=t,
    )
    calib = calibrate_confidence(72.0, ctx)
    txt   = summary_text(ctx, calibration=calib)
    assert calib.explanation[:30] in txt or "Confidence" in txt


def test_summary_without_calibration_no_crash():
    from memory.statistical_intelligence import get_historical_context, summary_text
    t   = _populated_tracker()
    ctx = get_historical_context(
        {"pattern_names": ["Failed Breakout"], "prediction": "bearish"},
        tracker=t,
    )
    txt = summary_text(ctx, calibration=None)
    assert isinstance(txt, str)
    assert len(txt) > 0


# ══════════════════════════════════════════════════════════════
# Edge cases
# ══════════════════════════════════════════════════════════════

def test_all_inconclusive_accuracy_defaults_half():
    from memory.statistical_intelligence import get_directional_stats
    t = _fresh_tracker()
    for _ in range(5):
        _record_and_resolve(t, "X", "bearish", "inconclusive")
    s = get_directional_stats("bearish", tracker=t)
    # All inconclusive → no conclusive obs → returns None
    assert s is None


def test_single_observation_stats():
    from memory.statistical_intelligence import get_historical_context
    t = _fresh_tracker()
    _record_and_resolve(t, "X", "bearish", "bearish", mfe_atr=1.5,
                        resolution_minutes=38.0)
    ctx = get_historical_context(
        {"pattern_names": ["X"], "prediction": "bearish"}, tracker=t
    )
    assert ctx.total_sample           == 1
    assert ctx.dominant_outcome       == "bearish"
    assert ctx.has_sufficient_history is False   # n=1 < MIN_SAMPLE_SIZE


def test_unknown_pattern_gives_empty_context():
    from memory.statistical_intelligence import get_historical_context
    t   = _populated_tracker()
    ctx = get_historical_context(
        {"pattern_names": ["No Such Pattern"], "prediction": "bearish"},
        tracker=t,
    )
    assert ctx.total_sample    == 0
    assert ctx.matched_patterns == []


def test_calibrate_with_empty_context_no_crash():
    from memory.statistical_intelligence import calibrate_confidence, HistoricalContext
    ctx = HistoricalContext(
        matched_patterns=[], confidence_bucket=None, directional=None,
        total_sample=0, dominant_outcome=None, dominant_rate=0.0,
        overall_accuracy=0.5, avg_mfe_atr=0.0, avg_mae_atr=0.0,
        avg_resolution_minutes=0.0, has_sufficient_history=False,
    )
    r = calibrate_confidence(65.0, ctx)
    assert r.final_confidence == 65.0
    assert r.adjustment       == 0.0


def test_confidence_bucket_all_eight_defined():
    from memory.statistical_intelligence import CONFIDENCE_BUCKETS
    assert len(CONFIDENCE_BUCKETS) == 8
    lows = [b[0] for b in CONFIDENCE_BUCKETS]
    assert lows == sorted(lows)


def test_bucket_for_boundary_values():
    from memory.statistical_intelligence import get_confidence_bucket_stats
    t = _fresh_tracker()
    _record_and_resolve(t, "X", "bearish", "bearish", confidence=15.0)
    _record_and_resolve(t, "X", "bearish", "bearish", confidence=90.0)
    s15 = get_confidence_bucket_stats(15.0, tracker=t)
    s90 = get_confidence_bucket_stats(90.0, tracker=t)
    assert s15 is not None
    assert s90 is not None
    assert s15.bucket_low  == 15
    assert s90.bucket_high == 90


# ══════════════════════════════════════════════════════════════
# v2.7 — staleness filter
# ══════════════════════════════════════════════════════════════

def _age_obs(tracker, oid, days):
    """Backdate one observation's recorded_at by `days`."""
    from datetime import datetime, timedelta, timezone
    obs = tracker.get_observation(oid)
    obs.recorded_at = (datetime.now(tz=timezone.utc) - timedelta(days=days)).isoformat()


def _tracker_with_stale(n_fresh, n_stale, stale_days=40, outcome="bearish"):
    """Tracker with n_fresh current + n_stale backdated resolved observations."""
    t = _fresh_tracker()
    for _ in range(n_fresh):
        _record_and_resolve(t, "Failed Breakout", "bearish", outcome, confidence=68.0)
    for _ in range(n_stale):
        oid = _record_and_resolve(t, "Failed Breakout", "bearish", outcome, confidence=68.0)
        _age_obs(t, oid, stale_days)
    return t


def test_stale_observations_excluded_from_pattern_stats():
    from memory.statistical_intelligence import get_historical_context
    t   = _tracker_with_stale(n_fresh=4, n_stale=6)
    ctx = get_historical_context(_snap_dict(confidence_score=68.0), tracker=t)
    assert ctx.total_sample       == 4
    assert ctx.age_filtered_count == 6


def test_stale_filter_can_drop_below_min_sample():
    # 12 obs total would be sufficient, but only 5 are recent.
    from memory.statistical_intelligence import get_historical_context
    t   = _tracker_with_stale(n_fresh=5, n_stale=7)
    ctx = get_historical_context(_snap_dict(confidence_score=68.0), tracker=t)
    assert ctx.has_sufficient_history is False


def test_observations_within_30_days_kept():
    from memory.statistical_intelligence import get_historical_context
    t   = _tracker_with_stale(n_fresh=0, n_stale=10, stale_days=29)
    ctx = get_historical_context(_snap_dict(confidence_score=68.0), tracker=t)
    assert ctx.total_sample       == 10
    assert ctx.age_filtered_count == 0


def test_bucket_stats_exclude_stale():
    from memory.statistical_intelligence import get_confidence_bucket_stats
    t = _tracker_with_stale(n_fresh=3, n_stale=5)
    s = get_confidence_bucket_stats(68.0, tracker=t)
    assert s.sample_size == 3


def test_directional_stats_exclude_stale():
    from memory.statistical_intelligence import get_directional_stats
    t = _tracker_with_stale(n_fresh=3, n_stale=5)
    s = get_directional_stats("bearish", tracker=t)
    assert s.sample_size == 3


def test_all_stale_returns_none_for_bucket_and_directional():
    from memory.statistical_intelligence import (
        get_confidence_bucket_stats, get_directional_stats,
    )
    t = _tracker_with_stale(n_fresh=0, n_stale=5)
    assert get_confidence_bucket_stats(68.0, tracker=t) is None
    assert get_directional_stats("bearish", tracker=t) is None


def test_max_age_days_none_disables_filter():
    from memory.statistical_intelligence import (
        get_historical_context, get_confidence_bucket_stats, get_directional_stats,
    )
    t   = _tracker_with_stale(n_fresh=4, n_stale=6)
    ctx = get_historical_context(_snap_dict(confidence_score=68.0),
                                 tracker=t, max_age_days=None)
    assert ctx.total_sample       == 10
    assert ctx.age_filtered_count == 0
    assert get_confidence_bucket_stats(68.0, tracker=t, max_age_days=None).sample_size == 10
    assert get_directional_stats("bearish", tracker=t, max_age_days=None).sample_size == 10


def test_stale_pending_observations_not_counted():
    # Only resolved observations would have influenced calibration, so only
    # those count toward age_filtered_count.
    from memory.statistical_intelligence import get_historical_context
    t   = _tracker_with_stale(n_fresh=2, n_stale=3)
    oid = t.record_observation(_snap_dict(), [_pat("Failed Breakout", "bearish")])
    _age_obs(t, oid, 40)
    ctx = get_historical_context(_snap_dict(confidence_score=68.0), tracker=t)
    assert ctx.age_filtered_count == 3


def test_unreadable_timestamp_kept():
    from memory.statistical_intelligence import get_historical_context
    t   = _fresh_tracker()
    oid = _record_and_resolve(t, "Failed Breakout", "bearish", "bearish", confidence=68.0)
    t.get_observation(oid).recorded_at = "not-a-timestamp"
    ctx = get_historical_context(_snap_dict(confidence_score=68.0), tracker=t)
    assert ctx.total_sample       == 1
    assert ctx.age_filtered_count == 0


def test_calibration_carries_age_filtered_count():
    from memory.statistical_intelligence import get_historical_context, calibrate_confidence
    t   = _tracker_with_stale(n_fresh=10, n_stale=4)
    ctx = get_historical_context(_snap_dict(confidence_score=68.0), tracker=t)
    r   = calibrate_confidence(68.0, ctx)
    assert r.age_filtered_count == 4
    assert "4 observations older than 30 days excluded" in r.explanation


def test_calibration_insufficient_history_mentions_stale():
    from memory.statistical_intelligence import get_historical_context, calibrate_confidence
    t   = _tracker_with_stale(n_fresh=2, n_stale=1)
    ctx = get_historical_context(_snap_dict(confidence_score=68.0), tracker=t)
    r   = calibrate_confidence(68.0, ctx)
    assert r.sufficient_history is False
    assert r.age_filtered_count == 1
    assert "1 observation older than 30 days excluded" in r.explanation


def test_no_stale_note_when_nothing_filtered():
    from memory.statistical_intelligence import calibrate_confidence
    r = calibrate_confidence(70.0, _ctx_with_accuracy(0.75, n=30))
    assert "older than" not in r.explanation


# ══════════════════════════════════════════════════════════════
# v2.7 — conflict dampening
# ══════════════════════════════════════════════════════════════

def _ctx_dominant(dominant, accuracy=0.75, n=30):
    """_ctx_with_accuracy, but with a chosen dominant historical outcome."""
    ctx = _ctx_with_accuracy(accuracy, n=n)
    ctx.dominant_outcome = dominant
    return ctx


def test_bearish_history_vs_uptrend_dampened():
    from memory.statistical_intelligence import calibrate_confidence, CONFLICT_DAMPEN_FACTOR
    r = calibrate_confidence(70.0, _ctx_dominant("bearish"), structure_direction="uptrend")
    assert r.conflict_dampened is True
    assert r.raw_adjustment > 0
    assert abs(r.adjustment - r.raw_adjustment * CONFLICT_DAMPEN_FACTOR) <= 0.1
    assert r.final_confidence == round(70.0 + r.adjustment, 1)


def test_bullish_history_vs_downtrend_dampened():
    from memory.statistical_intelligence import calibrate_confidence
    r = calibrate_confidence(70.0, _ctx_dominant("bullish"), structure_direction="downtrend")
    assert r.conflict_dampened is True
    assert abs(r.adjustment) < abs(r.raw_adjustment)


def test_negative_adjustment_also_dampened():
    from memory.statistical_intelligence import calibrate_confidence
    r = calibrate_confidence(70.0, _ctx_dominant("bearish", accuracy=0.25),
                             structure_direction="uptrend")
    assert r.conflict_dampened is True
    assert r.raw_adjustment < r.adjustment < 0


def test_aligned_history_not_dampened():
    from memory.statistical_intelligence import calibrate_confidence
    for dominant, direction in (("bearish", "downtrend"), ("bullish", "uptrend")):
        r = calibrate_confidence(70.0, _ctx_dominant(dominant), structure_direction=direction)
        assert r.conflict_dampened is False
        assert r.adjustment    == r.raw_adjustment
        assert r.dampen_reason == ""


def test_no_dampening_without_clear_direction():
    from memory.statistical_intelligence import calibrate_confidence
    for direction in (None, "", "ranging"):
        r = calibrate_confidence(70.0, _ctx_dominant("bearish"), structure_direction=direction)
        assert r.conflict_dampened is False


def test_no_dampening_for_inconclusive_history():
    from memory.statistical_intelligence import calibrate_confidence
    r = calibrate_confidence(70.0, _ctx_dominant("inconclusive"), structure_direction="uptrend")
    assert r.conflict_dampened is False


def test_dampen_reason_is_plain_english():
    from memory.statistical_intelligence import calibrate_confidence
    r = calibrate_confidence(70.0, _ctx_dominant("bearish"), structure_direction="uptrend")
    assert "bearish" in r.dampen_reason and "uptrend" in r.dampen_reason
    assert "reduced from" in r.explanation
    assert f"{r.raw_adjustment:+.1f}" in r.explanation
    assert f"{r.adjustment:+.1f}" in r.explanation


def test_dampened_result_still_clamped():
    from memory.statistical_intelligence import calibrate_confidence
    r = calibrate_confidence(88.0, _ctx_dominant("bearish", accuracy=1.0),
                             structure_direction="uptrend")
    assert r.final_confidence <= 90.0


def test_insufficient_history_never_dampened():
    from memory.statistical_intelligence import calibrate_confidence
    r = calibrate_confidence(70.0, _ctx_dominant("bearish", n=5), structure_direction="uptrend")
    assert r.sufficient_history is False
    assert r.conflict_dampened  is False
    assert r.adjustment == 0.0 and r.raw_adjustment == 0.0


# ══════════════════════════════════════════════════════════════
# v2.7 — new CalibrationResult fields / backward compatibility
# ══════════════════════════════════════════════════════════════

def test_raw_adjustment_matches_adjustment_by_default():
    # Old two-argument call: no direction, so no dampening.
    from memory.statistical_intelligence import calibrate_confidence
    r = calibrate_confidence(70.0, _ctx_with_accuracy(0.75, n=30))
    assert r.raw_adjustment    == r.adjustment
    assert r.conflict_dampened is False
    assert r.dampen_reason     == ""
    assert r.age_filtered_count == 0


def test_new_fields_have_correct_types():
    from memory.statistical_intelligence import calibrate_confidence
    r = calibrate_confidence(70.0, _ctx_dominant("bearish"), structure_direction="uptrend")
    assert isinstance(r.raw_adjustment,     float)
    assert isinstance(r.conflict_dampened,  bool)
    assert isinstance(r.dampen_reason,      str)
    assert isinstance(r.age_filtered_count, int)


def test_calibration_result_constructible_without_new_fields():
    from memory.statistical_intelligence import CalibrationResult
    r = CalibrationResult(
        base_confidence=60.0, final_confidence=60.0, adjustment=0.0,
        sufficient_history=False, win_rate=0.0, sample_size=0, explanation="x",
    )
    assert r.raw_adjustment == 0.0
    assert r.conflict_dampened is False
    assert r.dampen_reason == ""
    assert r.age_filtered_count == 0


def test_historical_context_age_filtered_defaults_to_zero():
    ctx = _ctx_with_accuracy(0.6, n=12)
    assert ctx.age_filtered_count == 0


# ──────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        (n, f) for n, f in sorted(globals().items())
        if n.startswith("test_") and callable(f)
    ]
    print(f"\nStatistical intelligence tests ({len(tests)} total)\n{'─' * 60}")
    for name, fn in tests:
        run(name, fn)
    print(f"\n{'=' * 60}")
    print(f"{passed} passed, {failed} failed  ({passed + failed} total)")
    import sys; sys.exit(0 if failed == 0 else 1)