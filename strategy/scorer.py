"""
Setup Quality Scorer.

Produces a 0–100 score with four components, each independently visible
so the AI can say "this setup scores 72 — strong alignment (28/30) and
structure (22/25) but mediocre timing (12/20)."

Component breakdown:
  trend_alignment   (max 30): how many TFs agree with the direction
  momentum_quality  (max 25): RSI zone + MACD quality + no divergence
  structure_context (max 25): key level + VWAP + S/R clearance
  entry_timing      (max 20): 5m candle quality + ATR context + 1m confirm
"""

from __future__ import annotations
from typing import Optional

from engine.models import (
    Direction,
    IndicatorSnapshot,
    MarketState,
    SetupScore,
    TradeDirection,
    TrendState,
    TrendStrength,
)
from strategy.timeframes import get_position_size_modifier


def score_setup(
    state:     MarketState,
    direction: TradeDirection,
) -> SetupScore:
    """
    Score a potential setup in the given direction.
    Returns a SetupScore with total and all components filled.
    """
    score = SetupScore()

    score.trend_alignment,  score.alignment_detail  = _score_alignment(state, direction)
    score.momentum_quality, score.momentum_detail   = _score_momentum(state, direction)
    score.structure_context, score.structure_detail = _score_structure(state, direction)
    score.entry_timing,     score.timing_detail     = _score_timing(state, direction)

    score.total = (
        score.trend_alignment
        + score.momentum_quality
        + score.structure_context
        + score.entry_timing
    )
    # Apply regime multiplier (if regime is set on state)
    if hasattr(state, "regime") and state.regime:
        score.regime_multiplier = state.regime.score_multiplier
    # Apply session modifier (if session is set on state)
    if hasattr(state, "session") and state.session:
        score.session_multiplier = state.session.score_modifier
    score.adjusted_total = round(
        score.total * score.regime_multiplier * score.session_multiplier, 1
    )
    return score


# ──────────────────────────────────────────────
# Component 1: Trend alignment (0–30)
# ──────────────────────────────────────────────

def _score_alignment(
    state:     MarketState,
    direction: TradeDirection,
) -> tuple[float, str]:
    """
    How many timeframes agree with the trade direction?

    Daily + 1h + 15m all aligned:   30
    1h + 15m aligned:                20
    1h aligned, daily neutral:       16
    15m aligned only:                 8
    Any conflict with 1h or daily:    0
    """
    target_dir = Direction.BULLISH if direction == TradeDirection.CALLS else Direction.BEARISH

    daily_dir  = state.trends.get("1d",  TrendState("1d")).direction
    hourly_dir = state.trends.get("1h",  TrendState("1h")).direction
    tf15_dir   = state.trends.get("15m", TrendState("15m")).direction
    tf5_dir    = state.trends.get("5m",  TrendState("5m")).direction

    # Conflict = 0
    if (daily_dir != Direction.NEUTRAL
            and hourly_dir != Direction.NEUTRAL
            and daily_dir != hourly_dir):
        return 0.0, f"HTF conflict: daily={daily_dir.value} vs 1h={hourly_dir.value}"

    aligned_tfs = []
    conflict_tfs = []

    for tf, d in [("daily", daily_dir), ("1h", hourly_dir), ("15m", tf15_dir), ("5m", tf5_dir)]:
        if d == target_dir:
            aligned_tfs.append(tf)
        elif d != Direction.NEUTRAL:
            conflict_tfs.append(tf)

    if conflict_tfs:
        return 0.0, f"Conflicting TF(s): {conflict_tfs}"

    # Score by combination
    has_daily  = "daily" in aligned_tfs
    has_1h     = "1h" in aligned_tfs
    has_15m    = "15m" in aligned_tfs
    has_5m     = "5m" in aligned_tfs

    if has_daily and has_1h and has_15m:
        points = 30.0
        bonus = 2.0 if has_5m else 0.0
        detail = "daily+1h+15m all aligned"
    elif has_1h and has_15m and has_daily:
        points = 26.0
        detail = "1h+15m aligned, daily supporting"
    elif has_1h and has_15m:
        points = 20.0
        detail = "1h+15m aligned, daily neutral"
    elif has_1h:
        points = 12.0
        detail = "1h aligned only — 15m not yet confirmed"
    elif has_15m:
        points = 8.0
        detail = "15m aligned only — wait for 1h confirmation"
    else:
        points = 0.0
        detail = "no TF alignment"

    # Strength bonus on 1h (up to +3)
    trend_1h = state.trends.get("1h")
    if trend_1h and trend_1h.direction == target_dir:
        if trend_1h.strength == TrendStrength.STRONG:
            points = min(30.0, points + 3.0)
            detail += " + strong 1h"
        elif trend_1h.strength == TrendStrength.MODERATE:
            points = min(30.0, points + 1.5)
            detail += " + moderate 1h"

    return round(points, 1), detail


# ──────────────────────────────────────────────
# Component 2: Momentum quality (0–25)
# ──────────────────────────────────────────────

def _score_momentum(
    state:     MarketState,
    direction: TradeDirection,
) -> tuple[float, str]:
    """
    RSI zone quality + MACD quality + no divergence

    RSI in ideal zone:              +10  (45-62 for longs, 38-55 for shorts)
    RSI in acceptable zone:          +6  (35-44 or 63-68 for longs)
    RSI extreme (overbought/sold):    +0

    MACD hist positive+growing:      +8  (longs)
    MACD hist positive:              +5
    MACD hist near 0 (< 0.001):      +2
    MACD hist negative:               0

    No trend weakening on 5m/15m:    +7
    """
    ind_5m   = state.indicators.get("5m")
    ind_15m  = state.indicators.get("15m")
    points   = 0.0
    details  = []

    if not ind_5m:
        return 0.0, "no 5m data"

    # ── RSI ───────────────────────────────────
    rsi = ind_5m.rsi
    if rsi is not None:
        if direction == TradeDirection.CALLS:
            if 45 <= rsi <= 62:
                points += 10.0; details.append(f"RSI ideal ({rsi:.1f})")
            elif 35 <= rsi < 45 or 62 < rsi <= 68:
                points += 6.0;  details.append(f"RSI acceptable ({rsi:.1f})")
            elif rsi < 35:
                points += 2.0;  details.append(f"RSI low ({rsi:.1f})")
            # >68: 0 points
        else:
            if 38 <= rsi <= 55:
                points += 10.0; details.append(f"RSI ideal ({rsi:.1f})")
            elif 55 < rsi <= 65 or 32 <= rsi < 38:
                points += 6.0;  details.append(f"RSI acceptable ({rsi:.1f})")
            elif rsi > 65:
                points += 2.0;  details.append(f"RSI high ({rsi:.1f})")

    # ── MACD ──────────────────────────────────
    hist = ind_5m.macd_hist
    if hist is not None:
        if direction == TradeDirection.CALLS:
            if hist > 0.002:
                points += 8.0; details.append(f"MACD hist bullish ({hist:.4f})")
            elif hist > 0:
                points += 5.0; details.append(f"MACD hist marginally positive ({hist:.4f})")
            elif abs(hist) <= 0.001:
                points += 2.0; details.append("MACD hist near zero")
        else:
            if hist < -0.002:
                points += 8.0; details.append(f"MACD hist bearish ({hist:.4f})")
            elif hist < 0:
                points += 5.0; details.append(f"MACD hist marginally negative ({hist:.4f})")
            elif abs(hist) <= 0.001:
                points += 2.0; details.append("MACD hist near zero")

    # ── No weakening ──────────────────────────
    trend_5m  = state.trends.get("5m")
    trend_15m = state.trends.get("15m")
    weakening = (
        (trend_5m  and trend_5m.is_weakening)
        or (trend_15m and trend_15m.is_weakening)
    )
    if not weakening:
        points += 7.0
        details.append("no weakening signals")
    else:
        details.append("weakening detected — momentum penalty")

    return round(min(25.0, points), 1), " | ".join(details)


# ──────────────────────────────────────────────
# Component 3: Structure context (0–25)
# ──────────────────────────────────────────────

def _score_structure(
    state:     MarketState,
    direction: TradeDirection,
) -> tuple[float, str]:
    """
    Pullback to key level + VWAP position + S/R clearance

    Clean pullback to EMA-20 or swing level:  +10
    Price on correct side of VWAP:             +8
    No immediate S/R in path (1 ATR clear):    +7
    """
    ind_5m  = state.indicators.get("5m")
    ind_15m = state.indicators.get("15m")
    points  = 0.0
    details = []

    # Use 15m for structure, 5m for precision
    ind = ind_15m or ind_5m
    if not ind:
        return 0.0, "no indicator data"

    price = ind.close
    atr   = ind.atr or (price * 0.002)

    # ── Pullback quality ──────────────────────
    if direction == TradeDirection.CALLS:
        ema20 = ind.ema_20
        ema9  = ind.ema_9
        if ema20 and abs(price - ema20) / ema20 < 0.003:
            points += 10.0; details.append("pullback to EMA-20")
        elif ema9 and abs(price - ema9) / ema9 < 0.002:
            points += 7.0;  details.append("pullback to EMA-9")
        elif ema20 and price > ema20:
            points += 4.0;  details.append("above EMA-20")
    else:
        ema20 = ind.ema_20
        ema9  = ind.ema_9
        if ema20 and abs(price - ema20) / ema20 < 0.003:
            points += 10.0; details.append("bounce to EMA-20")
        elif ema9 and abs(price - ema9) / ema9 < 0.002:
            points += 7.0;  details.append("bounce to EMA-9")
        elif ema20 and price < ema20:
            points += 4.0;  details.append("below EMA-20")

    # ── VWAP position ─────────────────────────
    if ind.vwap:
        if direction == TradeDirection.CALLS and price > ind.vwap:
            points += 8.0; details.append("above VWAP")
        elif direction == TradeDirection.PUTS and price < ind.vwap:
            points += 8.0; details.append("below VWAP")
        else:
            details.append("wrong side of VWAP (-8)")

    # ── S/R clearance ─────────────────────────
    if direction == TradeDirection.CALLS:
        blockers = [r for r in state.resistance_levels if r.price > price and r.price - price < atr]
        if not blockers:
            points += 7.0; details.append("path clear above (>1 ATR)")
        else:
            details.append(f"resistance at {blockers[0].price:.2f} within ATR")
    else:
        blockers = [s for s in state.support_levels if s.price < price and price - s.price < atr]
        if not blockers:
            points += 7.0; details.append("path clear below (>1 ATR)")
        else:
            details.append(f"support at {blockers[0].price:.2f} within ATR")

    return round(min(25.0, points), 1), " | ".join(details)


# ──────────────────────────────────────────────
# Component 4: Entry timing (0–20)
# ──────────────────────────────────────────────

def _score_timing(
    state:     MarketState,
    direction: TradeDirection,
) -> tuple[float, str]:
    """
    5m candle close quality + ATR context + 1m confirmation

    5m close above EMA-9 (long) or below (short):  +8
    ATR not extended (not chasing):                 +6
    1m momentum aligning:                           +6
    """
    ind_5m = state.indicators.get("5m")
    ind_1m = state.indicators.get("1m")
    points = 0.0
    details = []

    if not ind_5m:
        return 0.0, "no 5m data"

    price = ind_5m.close
    ema9  = ind_5m.ema_9

    # ── 5m close vs EMA-9 ────────────────────
    if ema9:
        if direction == TradeDirection.CALLS and price > ema9:
            points += 8.0; details.append(f"5m above EMA-9 ({price:.2f}>{ema9:.2f})")
        elif direction == TradeDirection.PUTS and price < ema9:
            points += 8.0; details.append(f"5m below EMA-9 ({price:.2f}<{ema9:.2f})")
        else:
            details.append("5m not past EMA-9 yet")

    # ── ATR extension check ───────────────────
    if ind_5m.atr and ema9:
        dist_from_ema = abs(price - ema9)
        if dist_from_ema < ind_5m.atr * 0.5:
            points += 6.0; details.append("not extended — good ATR entry")
        elif dist_from_ema < ind_5m.atr:
            points += 3.0; details.append("slightly extended")
        else:
            details.append("extended >1 ATR from EMA — chasing penalty")

    # ── 1m confirmation ───────────────────────
    if ind_1m:
        if direction == TradeDirection.CALLS:
            if (ind_1m.macd_hist and ind_1m.macd_hist > 0
                    and ind_1m.rsi and 40 < ind_1m.rsi < 70):
                points += 6.0; details.append("1m confirming bullish")
            elif ind_1m.ema_stack == "bull_aligned":
                points += 4.0; details.append("1m EMAs bull-aligned")
        else:
            if (ind_1m.macd_hist and ind_1m.macd_hist < 0
                    and ind_1m.rsi and 30 < ind_1m.rsi < 60):
                points += 6.0; details.append("1m confirming bearish")
            elif ind_1m.ema_stack == "bear_aligned":
                points += 4.0; details.append("1m EMAs bear-aligned")

    return round(min(20.0, points), 1), " | ".join(details)
