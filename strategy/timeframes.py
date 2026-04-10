"""
Timeframe weighting and consensus logic.

The hierarchy (high to low authority):
  Daily  (40%) → macro context, veto power
  1h     (30%) → dominant trend direction
  15m    (20%) → setup formation, entry prerequisite
  5m      (7%) → entry/exit timing
  1m      (3%) → stop placement precision

Rules encoded here (not in the strategy engine) so they can be
tested independently and tuned without touching trade logic.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional

from engine.models import Direction, TrendState, TrendStrength


# ──────────────────────────────────────────────
# Weights (must sum to 1.0)
# ──────────────────────────────────────────────

TF_WEIGHTS: Dict[str, float] = {
    "1d":  0.40,
    "1h":  0.30,
    "15m": 0.20,
    "5m":  0.07,
    "1m":  0.03,
}

# Timeframes that set overall bias (higher authority)
HTF_TIMEFRAMES = ("1d", "1h")

# Timeframes used for setup and entry confirmation
SETUP_TIMEFRAMES = ("15m", "5m")

# Timeframe used only for timing / stop placement
TIMING_TIMEFRAMES = ("1m",)


@dataclass
class TFConsensus:
    """
    Result of aggregating all timeframe trend assessments.
    """
    weighted_score: float       # positive = bullish, negative = bearish
    direction:      Direction
    strength:       TrendStrength

    # Alignment quality
    htf_agreement:    bool = False   # daily + 1h point same way
    htf_conflict:     bool = False   # daily and 1h disagree
    setup_confirmed:  bool = False   # 15m aligns with HTF bias
    tradeable:        bool = False   # meets minimum requirements to look for entry

    # Readable explanation
    summary: str = ""


def compute_tf_consensus(trends: Dict[str, TrendState]) -> TFConsensus:
    """
    Aggregate per-timeframe trend states into a single consensus.

    Scoring:
      +weight  for bullish TF
      -weight  for bearish TF
       0       for neutral TF

    Multipliers:
      strong trend:   ×1.0
      moderate trend: ×0.7
      weak trend:     ×0.4
      no trend:       ×0.0

    Final score range: [-1.0, +1.0]
    """
    score = 0.0
    notes = []

    for tf, weight in TF_WEIGHTS.items():
        trend = trends.get(tf)
        if trend is None:
            continue

        strength_mult = _strength_multiplier(trend.strength)
        if trend.direction == Direction.BULLISH:
            contribution = weight * strength_mult
            score += contribution
            if contribution > 0.05:
                notes.append(f"{tf} bull({trend.strength.value})")
        elif trend.direction == Direction.BEARISH:
            contribution = weight * strength_mult
            score -= contribution
            if contribution > 0.05:
                notes.append(f"{tf} bear({trend.strength.value})")

    # ── Direction from score ───────────────────
    if score >= 0.12:
        direction = Direction.BULLISH
    elif score <= -0.12:
        direction = Direction.BEARISH
    else:
        direction = Direction.NEUTRAL

    # ── Strength from magnitude ────────────────
    abs_score = abs(score)
    if abs_score >= 0.55:
        strength = TrendStrength.STRONG
    elif abs_score >= 0.30:
        strength = TrendStrength.MODERATE
    elif abs_score >= 0.12:
        strength = TrendStrength.WEAK
    else:
        strength = TrendStrength.NONE

    # ── HTF agreement / conflict ───────────────
    daily_dir = trends.get("1d", TrendState("1d")).direction
    hourly_dir = trends.get("1h", TrendState("1h")).direction

    htf_agreement = (
        daily_dir != Direction.NEUTRAL
        and hourly_dir != Direction.NEUTRAL
        and daily_dir == hourly_dir
    )
    htf_conflict = (
        daily_dir != Direction.NEUTRAL
        and hourly_dir != Direction.NEUTRAL
        and daily_dir != hourly_dir
    )

    # ── Setup confirmation (15m aligns with HTF) ──
    htf_bias = _resolve_htf_bias(daily_dir, hourly_dir)
    tf15_dir = trends.get("15m", TrendState("15m")).direction
    setup_confirmed = (
        htf_bias != Direction.NEUTRAL
        and tf15_dir == htf_bias
    )

    # ── Tradeable ─────────────────────────────────
    # Tradeable = no HTF conflict AND we have at least a 1h direction
    # AND 15m confirms
    tradeable = (
        not htf_conflict
        and hourly_dir != Direction.NEUTRAL
        and setup_confirmed
    )

    summary = " | ".join(notes) if notes else "no clear signal"

    return TFConsensus(
        weighted_score=round(score, 3),
        direction=direction,
        strength=strength,
        htf_agreement=htf_agreement,
        htf_conflict=htf_conflict,
        setup_confirmed=setup_confirmed,
        tradeable=tradeable,
        summary=summary,
    )


def get_htf_bias(trends: Dict[str, TrendState]) -> Direction:
    """
    Simple HTF bias: what do daily + 1h say together?

    Rules:
      daily bull + 1h bull → BULLISH
      daily bear + 1h bear → BEARISH
      daily neutral + 1h bull → BULLISH (allowed, tighter sizing)
      daily neutral + 1h bear → BEARISH (allowed, tighter sizing)
      daily bull + 1h bear → NEUTRAL (conflict, no trade)
      daily bear + 1h bull → NEUTRAL (conflict, no trade)
    """
    daily_dir  = trends.get("1d", TrendState("1d")).direction
    hourly_dir = trends.get("1h", TrendState("1h")).direction
    return _resolve_htf_bias(daily_dir, hourly_dir)


def is_entry_allowed(
    trends:    Dict[str, TrendState],
    direction: Direction,
) -> tuple[bool, str]:
    """
    Returns (allowed, reason).
    Checks all override rules before any entry is considered.
    """
    daily_dir  = trends.get("1d", TrendState("1d")).direction
    hourly_dir = trends.get("1h", TrendState("1h")).direction
    tf15_dir   = trends.get("15m", TrendState("15m")).direction

    # Rule 1: daily + 1h conflict → no trade
    if (daily_dir != Direction.NEUTRAL
            and hourly_dir != Direction.NEUTRAL
            and daily_dir != hourly_dir):
        return False, f"HTF conflict: daily={daily_dir.value} vs 1h={hourly_dir.value}"

    # Rule 2: trading against daily when daily is directional → no trade
    if daily_dir != Direction.NEUTRAL and daily_dir != direction:
        return False, f"Trading against daily ({daily_dir.value})"

    # Rule 3: 1h must be in the direction (or neutral)
    if hourly_dir != Direction.NEUTRAL and hourly_dir != direction:
        return False, f"1h trend ({hourly_dir.value}) opposes direction"

    # Rule 4: 15m must confirm direction
    if tf15_dir != direction:
        return False, f"15m not confirmed ({tf15_dir.value})"

    return True, "conditions met"


def get_position_size_modifier(trends: Dict[str, TrendState]) -> float:
    """
    Return a multiplier (0.5–1.0) for position sizing based on TF confidence.

    Full size (1.0):  daily + 1h + 15m all aligned
    Reduced (0.75):   daily neutral, 1h + 15m aligned
    Half size (0.5):  daily neutral, 1h weak, 15m aligned
    """
    daily_dir  = trends.get("1d", TrendState("1d")).direction
    hourly_dir = trends.get("1h", TrendState("1h")).direction
    hourly_str = trends.get("1h", TrendState("1h")).strength

    if daily_dir != Direction.NEUTRAL and hourly_dir == daily_dir:
        return 1.0
    if daily_dir == Direction.NEUTRAL and hourly_str in (TrendStrength.STRONG, TrendStrength.MODERATE):
        return 0.75
    return 0.5


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _strength_multiplier(strength: TrendStrength) -> float:
    return {
        TrendStrength.STRONG:   1.0,
        TrendStrength.MODERATE: 0.7,
        TrendStrength.WEAK:     0.4,
        TrendStrength.NONE:     0.0,
    }.get(strength, 0.0)


def _resolve_htf_bias(daily: Direction, hourly: Direction) -> Direction:
    """Core override logic for daily vs 1h interaction."""
    # Conflict
    if (daily != Direction.NEUTRAL
            and hourly != Direction.NEUTRAL
            and daily != hourly):
        return Direction.NEUTRAL

    # Both agree
    if daily == hourly and daily != Direction.NEUTRAL:
        return daily

    # Daily neutral: defer to hourly
    if daily == Direction.NEUTRAL and hourly != Direction.NEUTRAL:
        return hourly

    # Hourly neutral: defer to daily
    if hourly == Direction.NEUTRAL and daily != Direction.NEUTRAL:
        return daily

    return Direction.NEUTRAL
