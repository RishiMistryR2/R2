"""
engine/readiness.py — readiness state computation.

Pure functions: (candle_counts: dict) -> Readiness + detail.
No network, no side effects. Fully testable.

Readiness gates:
  NOT_READY:       fewer than 30 candles on 1h OR 15m
                   → no indicators, no regime, no entry evaluation
  PARTIALLY_READY: ≥30 on both primary TFs, but below full targets
                   → all indicators computed with available data
                   → score penalty of PARTIALLY_READY_PENALTY applied
                   → regime confidence capped at 0.65
  FULLY_READY:     all TFs at or above full targets
                   → all operations unrestricted
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from engine.models import Readiness
from config.settings import (
    READINESS_THRESHOLDS,
    PARTIALLY_READY_PENALTY,
)


def compute_readiness(candle_counts: Dict[str, int]) -> Tuple[Readiness, str]:
    """
    Determine system readiness from current candle counts.

    Returns (Readiness, human-readable note).

    candle_counts: {timeframe: count_of_closed_candles}
    Missing timeframes are treated as 0.
    """
    fully   = READINESS_THRESHOLDS["fully_ready"]
    partial = READINESS_THRESHOLDS["partially_ready"]

    # Check NOT_READY: either primary TF below bare minimum
    primary_tfs = ("1h", "15m")
    for tf in primary_tfs:
        count   = candle_counts.get(tf, 0)
        minimum = partial[tf]
        if count < minimum:
            short = minimum - count
            return (
                Readiness.NOT_READY,
                f"Seeding {tf}: {count}/{minimum} candles ({short} needed)"
            )

    # Check FULLY_READY: all configured TFs at target
    not_full = []
    for tf, target in fully.items():
        count = candle_counts.get(tf, 0)
        if count < target:
            not_full.append(f"{tf}: {count}/{target}")

    if not_full:
        return (
            Readiness.PARTIALLY_READY,
            f"Warming up — {', '.join(not_full[:3])}"
        )

    return Readiness.FULLY_READY, "All timeframes ready"


def readiness_score_penalty(readiness: Readiness) -> float:
    """
    Points to subtract from SetupScore.adjusted_total based on readiness.
    FULLY_READY → 0.0 penalty.
    PARTIALLY_READY → PARTIALLY_READY_PENALTY (default 10.0).
    NOT_READY → not evaluated (entry gates block before scoring).
    """
    if readiness == Readiness.PARTIALLY_READY:
        return PARTIALLY_READY_PENALTY
    return 0.0


def readiness_regime_confidence_cap(readiness: Readiness) -> Optional[float]:
    """
    Maximum regime confidence allowed at this readiness level.
    Returns None if no cap (FULLY_READY).
    """
    if readiness == Readiness.NOT_READY:
        return 0.0
    if readiness == Readiness.PARTIALLY_READY:
        return 0.65
    return None


def describe_missing_indicators(candle_counts: Dict[str, int]) -> Dict[str, list]:
    """
    For PARTIALLY_READY state: identify which EMA periods cannot yet be computed
    on each timeframe, so the AI context can note them explicitly.

    Returns {timeframe: [missing_indicator_names]}
    """
    # EMA-N requires N candles minimum (Wilder needs more but this is the hard floor)
    ema_requirements = {9: 9, 20: 20, 50: 50, 200: 200}
    result: Dict[str, list] = {}

    for tf, count in candle_counts.items():
        missing = []
        for period, req in ema_requirements.items():
            if count < req:
                missing.append(f"EMA-{period}")
        if count < 14:
            missing.append("RSI")
        if count < 35:   # MACD needs ~26 + 9 signal periods
            missing.append("MACD")
        if count < 15:
            missing.append("ATR")
        if missing:
            result[tf] = missing

    return result


def premarket_confidence_reduction(
    premarket_fraction: float,
    threshold: float = 0.20,
    penalty: float = 0.15,
) -> float:
    """
    Returns the confidence reduction to apply when the classification window
    contains more than `threshold` fraction of premarket candles.

    Returns 0.0 if below threshold, `penalty` if above.
    """
    if premarket_fraction > threshold:
        return penalty
    return 0.0