"""
memory/statistical_intelligence.py — R2 Statistical Intelligence Engine v2.6

Reads resolved observations from OutcomeTracker and produces structured
historical context for commentary enrichment and confidence calibration.

This module never writes to the tracker.  It is a pure read/aggregation
layer.  All computation is deterministic — no ML, no randomness.

Architecture
------------
ConfidenceBucketStats
    Statistics for observations whose confidence_score fell in a fixed
    10-point bucket (e.g. 60–69).  Answers: "When the system was 60–69%
    confident, how often was the prediction correct?"

DirectionalStats
    Statistics for observations with a given prediction bias ("bullish" or
    "bearish").  Answers: "When the system predicted bearish, how often did
    price resolve bearish?"

HistoricalContext
    Unified view aggregating pattern-level stats, confidence-bucket stats,
    and directional stats for the current market situation.

CalibrationResult
    Result of adjusting base confidence using historical win rate.
    Always fully explainable in plain text.

Public API
----------
  get_pattern_stats(name, tracker=None)              → PatternStats | None
  get_confidence_bucket_stats(conf, tracker=None)    → ConfidenceBucketStats | None
  get_directional_stats(prediction, tracker=None)    → DirectionalStats | None
  get_historical_context(snap, conf, pred, tracker)  → HistoricalContext
  get_historical_context_from_package(pkg, tracker)  → HistoricalContext
  calibrate_confidence(base, context)                → CalibrationResult
  summary_text(context, calibration=None)            → str

Constants
---------
  MIN_SAMPLE_SIZE              = 10    minimum resolved obs for inference
  CALIBRATION_FULL_WEIGHT_N    = 30    sample size for full adjustment weight
  MAX_CALIBRATION_ADJUSTMENT   = 15.0  maximum pts of adjustment (±)
  CONFIDENCE_BUCKETS           = [(15,24),(25,34),(35,44),(45,54),
                                   (55,64),(65,74),(75,84),(85,90)]
  MAX_OBSERVATION_AGE_DAYS     = 30    v2.7: older observations are excluded
  CONFLICT_DAMPEN_FACTOR       = 0.5   v2.7: adjustment scale when history
                                       opposes current structure direction
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger("memory.statistical_intelligence")

# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────

MIN_SAMPLE_SIZE            = 10
CALIBRATION_FULL_WEIGHT_N  = 30
MAX_CALIBRATION_ADJUSTMENT = 15.0
_MAX_OBS                   = 500   # matches OutcomeTracker rolling cap

# v2.7 — staleness filter: market behaviour from more than a month ago is
# not reliable evidence about today's structure.
MAX_OBSERVATION_AGE_DAYS   = 30

# v2.7 — conflict dampening: when history leans the opposite way to the
# current structure, live structure is the fresher evidence, so the
# historical adjustment is scaled down by this factor.
CONFLICT_DAMPEN_FACTOR     = 0.5

# 8-point buckets across the 15–90 confidence range
CONFIDENCE_BUCKETS: list[tuple[int, int]] = [
    (15, 24), (25, 34), (35, 44), (45, 54),
    (55, 64), (65, 74), (75, 84), (85, 90),
]


# ══════════════════════════════════════════════════════════════
# Data types
# ══════════════════════════════════════════════════════════════

@dataclass
class ConfidenceBucketStats:
    """
    Statistics for all resolved observations whose confidence score fell
    in a fixed bucket (e.g. 65–74).
    """
    bucket_label:        str    # "65–74"
    bucket_low:          int
    bucket_high:         int
    sample_size:         int    # resolved observations in this bucket
    prediction_accuracy: float  # correct directional / total directional
    bullish_rate:        float  # bullish / resolved
    bearish_rate:        float  # bearish / resolved
    inconclusive_rate:   float  # inconclusive / resolved
    avg_mfe_atr:         float
    avg_mae_atr:         float


@dataclass
class DirectionalStats:
    """
    Statistics for all resolved observations with a given prediction bias.
    Inconclusive outcomes are excluded from win_rate.
    """
    prediction:             str     # "bullish" | "bearish"
    sample_size:            int     # conclusively resolved (excl. inconclusive)
    win_rate:               float   # prediction == outcome / sample_size
    avg_mfe_atr:            float
    avg_mae_atr:            float
    avg_resolution_minutes: float


@dataclass
class HistoricalContext:
    """
    Aggregated historical view of the current market situation.
    Built from pattern-level stats, confidence-bucket stats, and
    directional stats for the current snapshot.
    """
    # Pattern-level
    matched_patterns:    list           # list[PatternStats] from OutcomeTracker

    # Confidence-bucket
    confidence_bucket:   Optional[ConfidenceBucketStats]

    # Directional (for current prediction)
    directional:         Optional[DirectionalStats]

    # Aggregated across all matched patterns
    total_sample:           int
    dominant_outcome:       Optional[str]   # "bullish" | "bearish" | "inconclusive"
    dominant_rate:          float
    overall_accuracy:       float           # correct_predictions / total_directional
    avg_mfe_atr:            float
    avg_mae_atr:            float
    avg_resolution_minutes: float

    # Meta
    has_sufficient_history: bool    # total_sample >= MIN_SAMPLE_SIZE
    age_filtered_count:     int = 0 # v2.7: resolved obs excluded as stale


@dataclass
class CalibrationResult:
    """
    Deterministic adjustment of base confidence using historical win rate.
    Every adjustment is bounded, weighted by sample size, and fully
    explained in plain English.
    """
    base_confidence:    float
    final_confidence:   float
    adjustment:         float   # positive = increased, negative = decreased
    sufficient_history: bool
    win_rate:           float   # historical rate that drove the adjustment
    sample_size:        int
    explanation:        str
    # v2.7 — all defaulted for backward compatibility
    raw_adjustment:     float = 0.0    # before conflict dampening
    conflict_dampened:  bool  = False  # True when history opposed structure
    dampen_reason:      str   = ""     # plain-English why, "" if not dampened
    age_filtered_count: int   = 0      # resolved obs excluded as > 30 days old


# ══════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════

def get_pattern_stats(
    pattern_name: str,
    tracker=None,
):
    """
    Return PatternStats for one pattern name.
    Delegates directly to OutcomeTracker.get_stats().
    Returns None if the pattern has no resolved observations.
    """
    t = _get_tracker(tracker)
    return t.get_stats(pattern_name)


def get_confidence_bucket_stats(
    confidence_score: float,
    tracker=None,
    max_age_days:     Optional[int] = MAX_OBSERVATION_AGE_DAYS,
) -> Optional[ConfidenceBucketStats]:
    """
    Return statistics for all resolved observations whose confidence_score
    fell in the same bucket as confidence_score.

    Returns None if no resolved observations exist in that bucket.
    """
    bucket = _bucket_for(confidence_score)
    if bucket is None:
        return None

    b_low, b_high = bucket
    t             = _get_tracker(tracker)
    all_obs, _    = _fresh_observations(t, max_age_days)

    bucket_obs = [
        o for o in all_obs
        if o.resolved and b_low <= o.confidence_score <= b_high
    ]
    if not bucket_obs:
        return None

    n         = len(bucket_obs)
    bullish   = sum(1 for o in bucket_obs if o.outcome == "bullish")
    bearish   = sum(1 for o in bucket_obs if o.outcome == "bearish")
    inconc    = sum(1 for o in bucket_obs if o.outcome == "inconclusive")

    directional = [
        o for o in bucket_obs
        if o.prediction != "neutral" and o.outcome not in (None, "inconclusive")
    ]
    correct  = sum(1 for o in directional if o.prediction == o.outcome)
    accuracy = correct / len(directional) if directional else 0.5

    mfes = [o.mfe_atr for o in bucket_obs if o.mfe_atr is not None]
    maes = [o.mae_atr for o in bucket_obs if o.mae_atr is not None]

    return ConfidenceBucketStats(
        bucket_label        = f"{b_low}–{b_high}",
        bucket_low          = b_low,
        bucket_high         = b_high,
        sample_size         = n,
        prediction_accuracy = round(accuracy, 4),
        bullish_rate        = round(bullish / n, 4),
        bearish_rate        = round(bearish / n, 4),
        inconclusive_rate   = round(inconc  / n, 4),
        avg_mfe_atr         = round(sum(mfes) / len(mfes), 2) if mfes else 0.0,
        avg_mae_atr         = round(sum(maes) / len(maes), 2) if maes else 0.0,
    )


def get_directional_stats(
    prediction: str,
    tracker=None,
    max_age_days: Optional[int] = MAX_OBSERVATION_AGE_DAYS,
) -> Optional[DirectionalStats]:
    """
    Return statistics for all observations that predicted the given direction
    and resolved conclusively (inconclusive outcomes are excluded).

    Returns None when no conclusive observations exist for this prediction.
    """
    if prediction == "neutral":
        return None

    t       = _get_tracker(tracker)
    all_obs, _ = _fresh_observations(t, max_age_days)

    obs = [
        o for o in all_obs
        if (o.resolved
            and o.prediction == prediction
            and o.outcome not in (None, "inconclusive"))
    ]
    if not obs:
        return None

    correct = sum(1 for o in obs if o.prediction == o.outcome)
    mfes    = [o.mfe_atr            for o in obs if o.mfe_atr            is not None]
    maes    = [o.mae_atr            for o in obs if o.mae_atr            is not None]
    times   = [o.resolution_minutes for o in obs if o.resolution_minutes is not None]

    return DirectionalStats(
        prediction             = prediction,
        sample_size            = len(obs),
        win_rate               = round(correct / len(obs), 4),
        avg_mfe_atr            = round(sum(mfes)  / len(mfes),  2) if mfes  else 0.0,
        avg_mae_atr            = round(sum(maes)  / len(maes),  2) if maes  else 0.0,
        avg_resolution_minutes = round(sum(times) / len(times), 1) if times else 0.0,
    )


def get_historical_context(
    snapshot_or_dict,
    confidence_score: Optional[float] = None,
    prediction:       Optional[str]   = None,
    tracker=None,
    max_age_days:     Optional[int]   = MAX_OBSERVATION_AGE_DAYS,
) -> HistoricalContext:
    """
    Build a HistoricalContext for the current market situation.

    snapshot_or_dict may be:
      - a MarketSnapshot (from event_engine)  — uses .pattern_names, .confidence_score
      - a plain dict                          — uses ["pattern_names"], ["confidence_score"]
      - any duck-typed object with the above attributes

    confidence_score and prediction override what is read from the snapshot.
    """
    # ── Extract metadata ──────────────────────────────────────
    if isinstance(snapshot_or_dict, dict):
        snap_names  = list(snapshot_or_dict.get("pattern_names", []))
        snap_conf   = snapshot_or_dict.get("confidence_score", 50.0)
        snap_pred   = snapshot_or_dict.get("prediction", "neutral")
    else:
        snap_names  = list(getattr(snapshot_or_dict, "pattern_names", ()))
        snap_conf   = getattr(snapshot_or_dict, "confidence_score", 50.0)
        snap_pred   = getattr(snapshot_or_dict, "prediction", "neutral")

    effective_conf = confidence_score if confidence_score is not None else snap_conf
    effective_pred = prediction       if prediction       is not None else snap_pred

    t = _get_tracker(tracker)

    # ── Staleness filter (v2.7) ───────────────────────────────
    fresh_obs, age_filtered = _fresh_observations(t, max_age_days)

    # ── Pattern stats (fresh observations only) ───────────────
    matched = [
        s for name in snap_names
        for s in [_pattern_stats_from(name, fresh_obs)]
        if s is not None
    ]

    # ── Confidence bucket ─────────────────────────────────────
    cb = (get_confidence_bucket_stats(effective_conf, tracker=t,
                                      max_age_days=max_age_days)
          if effective_conf is not None else None)

    # ── Directional stats ─────────────────────────────────────
    ds = (get_directional_stats(effective_pred, tracker=t,
                                max_age_days=max_age_days)
          if effective_pred and effective_pred != "neutral" else None)

    # ── Aggregate across matched patterns ─────────────────────
    total_sample = sum(s.resolved_count for s in matched)

    if matched and total_sample > 0:
        total_bullish  = sum(s.bullish_outcomes    for s in matched)
        total_bearish  = sum(s.bearish_outcomes    for s in matched)
        total_inconc   = sum(s.inconclusive_count  for s in matched)
        total_dir      = sum(s.total_directional   for s in matched)
        total_correct  = sum(s.correct_predictions for s in matched)
        counts         = [s.resolved_count         for s in matched]

        # Dominant outcome
        candidates = [
            ("bullish",      total_bullish),
            ("bearish",      total_bearish),
            ("inconclusive", total_inconc),
        ]
        dominant, dom_n = max(candidates, key=lambda x: x[1])
        dominant_rate   = dom_n / total_sample

        overall_accuracy = total_correct / total_dir if total_dir else 0.5

        avg_mfe = _weighted_avg(
            [s.avg_mfe_atr            for s in matched], counts)
        avg_mae = _weighted_avg(
            [s.avg_mae_atr            for s in matched], counts)
        avg_res = _weighted_avg(
            [s.avg_resolution_minutes for s in matched], counts)
    else:
        dominant         = None
        dominant_rate    = 0.0
        overall_accuracy = 0.5
        avg_mfe          = 0.0
        avg_mae          = 0.0
        avg_res          = 0.0

    return HistoricalContext(
        matched_patterns        = matched,
        confidence_bucket       = cb,
        directional             = ds,
        total_sample            = total_sample,
        dominant_outcome        = dominant,
        dominant_rate           = round(dominant_rate, 4),
        overall_accuracy        = round(overall_accuracy, 4),
        avg_mfe_atr             = round(avg_mfe, 2),
        avg_mae_atr             = round(avg_mae, 2),
        avg_resolution_minutes  = round(avg_res, 1),
        has_sufficient_history  = total_sample >= MIN_SAMPLE_SIZE,
        age_filtered_count      = age_filtered,
    )


def get_historical_context_from_package(
    package_dict: dict,
    tracker=None,
) -> HistoricalContext:
    """
    Build HistoricalContext from a StatePackage.to_dict() result.

    Extracts pattern names via the pattern engine and confidence score
    via the confidence engine.  Both imports are non-fatal — falls back
    gracefully when the engines are unavailable.
    """
    market = package_dict.get("market", {}) or {}
    struct = package_dict.get("structure", {}) or {}

    # ── Pattern names ─────────────────────────────────────────
    pattern_names: list[str] = []
    try:
        from engine.pattern_engine import classify_patterns
        patterns      = classify_patterns(struct)
        pattern_names = [p.name for p in patterns]
    except Exception:
        pass

    # ── Confidence score ──────────────────────────────────────
    confidence_score = 50.0
    try:
        from engine.confidence_engine import compute_confidence
        from engine.narrative_engine  import _session_key
        regime_d    = market.get("regime", {}) or {}
        regime_name = regime_d.get("name", "")
        regime_conf = float(regime_d.get("confidence", 0.5))
        sess        = market.get("market_session", "")
        entries     = (market.get("session", {}) or {}).get("entries_allowed", True)
        cr          = compute_confidence(struct, regime_name, regime_conf, sess, entries)
        confidence_score = cr.score
    except Exception:
        pass

    # ── Prediction (from primary pattern bias) ─────────────────
    prediction = "neutral"
    try:
        from engine.pattern_engine import classify_patterns
        patterns = classify_patterns(struct)
        if patterns:
            prediction = getattr(patterns[0], "bias", "neutral") or "neutral"
    except Exception:
        pass

    snap_dict = {
        "pattern_names":    pattern_names,
        "confidence_score": confidence_score,
        "prediction":       prediction,
    }
    return get_historical_context(snap_dict, tracker=tracker)


def calibrate_confidence(
    base_confidence:     float,
    context:             HistoricalContext,
    structure_direction: Optional[str] = None,
) -> CalibrationResult:
    """
    Deterministically adjust base_confidence using historical win rate.

    Rules:
      - If sample_size < MIN_SAMPLE_SIZE: no adjustment.
      - weight ramps linearly from 0 → 1 as sample_size goes 0 → CALIBRATION_FULL_WEIGHT_N.
      - adjustment = (win_rate − 0.5) × 2 × MAX_CALIBRATION_ADJUSTMENT × weight
      - Final confidence is clamped to [15, 90].

    The chosen win_rate is:
      1. directional.win_rate  if directional sample_size >= MIN_SAMPLE_SIZE
      2. overall_accuracy      otherwise

    v2.7 conflict dampening: if structure_direction is given and the
    historical dominant outcome opposes it (bullish history vs downtrend,
    bearish history vs uptrend), the adjustment is scaled by
    CONFLICT_DAMPEN_FACTOR.  raw_adjustment keeps the undampened value.
    """
    stale_note = _stale_note(context.age_filtered_count)

    if not context.has_sufficient_history:
        return CalibrationResult(
            base_confidence    = base_confidence,
            final_confidence   = round(base_confidence, 1),
            adjustment         = 0.0,
            sufficient_history = False,
            win_rate           = 0.0,
            sample_size        = context.total_sample,
            explanation        = (
                f"Insufficient history ({context.total_sample} obs, "
                f"minimum {MIN_SAMPLE_SIZE} required). "
                "Confidence unchanged." + stale_note
            ),
            age_filtered_count = context.age_filtered_count,
        )

    # Select the most specific win rate available
    if (context.directional is not None
            and context.directional.sample_size >= MIN_SAMPLE_SIZE):
        win_rate    = context.directional.win_rate
        sample_size = context.directional.sample_size
        rate_source = "directional"
    else:
        win_rate    = context.overall_accuracy
        sample_size = context.total_sample
        rate_source = "overall"

    # Compute weighted adjustment
    weight     = min(sample_size / CALIBRATION_FULL_WEIGHT_N, 1.0)
    raw_adj    = (win_rate - 0.5) * 2.0 * MAX_CALIBRATION_ADJUSTMENT * weight
    raw_adj    = max(-MAX_CALIBRATION_ADJUSTMENT,
                     min(MAX_CALIBRATION_ADJUSTMENT, raw_adj))

    # Conflict dampening (v2.7)
    conflict, dampen_reason = _history_conflict(
        context.dominant_outcome, structure_direction,
    )
    adjustment = raw_adj * CONFLICT_DAMPEN_FACTOR if conflict else raw_adj

    final      = max(15.0, min(90.0, base_confidence + adjustment))
    final      = round(final, 1)
    adjustment = round(adjustment, 1)
    raw_adj    = round(raw_adj, 1)

    # Plain-English explanation
    if abs(adjustment) < 0.5:
        explanation = (
            f"Historical {rate_source} win rate ({win_rate:.0%}, "
            f"n={sample_size}) is near 50% — no meaningful adjustment. "
            f"Confidence remains {final:.0f}%."
        )
    elif adjustment > 0:
        explanation = (
            f"Historical {rate_source} win rate of {win_rate:.0%} "
            f"({sample_size} obs) supports the thesis. "
            f"Confidence increased from {base_confidence:.0f}% "
            f"to {final:.0f}% (+{adjustment:.1f} pts)."
        )
    else:
        explanation = (
            f"Historical {rate_source} win rate of {win_rate:.0%} "
            f"({sample_size} obs) contradicts the thesis. "
            f"Confidence decreased from {base_confidence:.0f}% "
            f"to {final:.0f}% ({adjustment:.1f} pts)."
        )

    if conflict:
        explanation += (
            f" {dampen_reason}, so the historical adjustment was reduced "
            f"from {raw_adj:+.1f} to {adjustment:+.1f} pts."
        )
    explanation += stale_note

    return CalibrationResult(
        base_confidence    = base_confidence,
        final_confidence   = final,
        adjustment         = adjustment,
        sufficient_history = True,
        win_rate           = win_rate,
        sample_size        = sample_size,
        explanation        = explanation,
        raw_adjustment     = raw_adj,
        conflict_dampened  = conflict,
        dampen_reason      = dampen_reason,
        age_filtered_count = context.age_filtered_count,
    )


def summary_text(
    context:     HistoricalContext,
    calibration: Optional[CalibrationResult] = None,
) -> str:
    """
    Generate a compact historical memory block for commentary.

    Example output:
        Historical memory (Failed Breakout, 48 obs):
          · 71% resolved bearish  ·  29% resolved bullish
          · Average move: 1.3 ATR  ·  Average resolution: 42 min
          · Directional accuracy: 71%
          Historical win rate of 71% (48 obs) supports the thesis.
          Confidence increased from 72.0% to 78.0% (+6.0 pts).
    """
    if not context.has_sufficient_history:
        n = context.total_sample
        if n == 0:
            return "Historical memory: no resolved observations yet."
        return (
            f"Historical memory: {n} observation"
            + ("s" if n != 1 else "")
            + f" (minimum {MIN_SAMPLE_SIZE} required for inference)."
        )

    lines: list[str] = []

    # ── Header ────────────────────────────────────────────────
    if context.matched_patterns:
        names = ", ".join(p.pattern_name for p in context.matched_patterns)
        lines.append(f"Historical memory ({names}, {context.total_sample} obs):")
    else:
        lines.append(f"Historical memory ({context.total_sample} obs):")

    # ── Outcome rates ─────────────────────────────────────────
    if context.matched_patterns and context.total_sample > 0:
        total_bullish = sum(p.bullish_outcomes for p in context.matched_patterns)
        total_bearish = sum(p.bearish_outcomes for p in context.matched_patterns)
        n             = context.total_sample
        lines.append(
            f"  · {total_bearish/n:.0%} resolved bearish  "
            f"·  {total_bullish/n:.0%} resolved bullish"
        )

    # ── Move and resolution time ──────────────────────────────
    if context.avg_mfe_atr > 0 or context.avg_resolution_minutes > 0:
        move_str = (f"Average move: {context.avg_mfe_atr:.1f} ATR  "
                    if context.avg_mfe_atr > 0 else "")
        time_str = (f"·  Average resolution: {context.avg_resolution_minutes:.0f} min"
                    if context.avg_resolution_minutes > 0 else "")
        if move_str or time_str:
            lines.append(f"  · {move_str}{time_str}".rstrip())

    # ── Directional accuracy ──────────────────────────────────
    if context.overall_accuracy and context.overall_accuracy != 0.5:
        lines.append(
            f"  · Directional accuracy: {context.overall_accuracy:.0%}"
        )

    # ── Calibration explanation ───────────────────────────────
    if calibration is not None and calibration.sufficient_history:
        lines.append(f"  {calibration.explanation}")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════
# Internal helpers
# ══════════════════════════════════════════════════════════════

def _bucket_for(confidence_score: float) -> Optional[tuple[int, int]]:
    """Return the (low, high) bucket tuple for a confidence score."""
    for low, high in CONFIDENCE_BUCKETS:
        if low <= confidence_score <= high:
            return (low, high)
    return None


def _fresh_observations(tracker, max_age_days: Optional[int]) -> tuple[list, int]:
    """
    Return (observations recorded within max_age_days, number of resolved
    observations excluded as stale).  max_age_days=None disables the filter.
    Observations with an unreadable timestamp are kept: only observations
    we can show are old get excluded.
    """
    all_obs = tracker.recent_observations(n=_MAX_OBS)
    if max_age_days is None:
        return all_obs, 0

    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=max_age_days)
    fresh: list = []
    stale = 0
    for o in all_obs:
        try:
            rec = datetime.fromisoformat(o.recorded_at)
            if rec.tzinfo is None:
                rec = rec.replace(tzinfo=timezone.utc)
        except (TypeError, ValueError):
            fresh.append(o)
            continue
        if rec < cutoff:
            if o.resolved:
                stale += 1
        else:
            fresh.append(o)
    return fresh, stale


def _pattern_stats_from(pattern_name: str, observations: list):
    """
    PatternStats for one pattern over a given observation list.
    Same aggregation as OutcomeTracker.get_stats(), but over the
    age-filtered list instead of every stored observation.
    """
    # TODO(future version): promote _compute_stats to a public function in
    # memory/outcome_tracker.py; importing a private helper across modules
    # is a stopgap to avoid duplicating its aggregation logic here.
    from memory.outcome_tracker import _compute_stats
    named    = [o for o in observations if pattern_name in o.pattern_names]
    resolved = [o for o in named if o.resolved]
    if not resolved:
        return None
    return _compute_stats(pattern_name, resolved, len(named))


_OPPOSING_HISTORY = {("bullish", "downtrend"), ("bearish", "uptrend")}


def _history_conflict(
    dominant_outcome:    Optional[str],
    structure_direction: Optional[str],
) -> tuple[bool, str]:
    """True + plain-English reason when history opposes current structure."""
    if (dominant_outcome, structure_direction) in _OPPOSING_HISTORY:
        return True, (
            f"Historically these setups mostly resolved {dominant_outcome}, "
            f"but current structure is a {structure_direction}"
        )
    return False, ""


def _stale_note(age_filtered_count: int) -> str:
    """Explanation suffix noting how many stale observations were excluded."""
    if age_filtered_count <= 0:
        return ""
    plural = "s" if age_filtered_count != 1 else ""
    return (
        f" {age_filtered_count} observation{plural} older than "
        f"{MAX_OBSERVATION_AGE_DAYS} days excluded."
    )


def _weighted_avg(values: list[float], weights: list[float]) -> float:
    """Weighted average.  Returns 0.0 when total weight is zero."""
    total_weight = sum(weights)
    if total_weight == 0:
        return 0.0
    return sum(v * w for v, w in zip(values, weights)) / total_weight


def _get_tracker(tracker=None):
    """Return the provided tracker or the global singleton."""
    if tracker is not None:
        return tracker
    from memory.outcome_tracker import get_tracker
    return get_tracker()