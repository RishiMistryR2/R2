"""
Market Narrative Classifier.

Determines the dominant market condition from the full MarketState.
This is a pure function: MarketState -> NarrativeState.
No state, no side effects. Completely deterministic.

The narrative answers: "What kind of market is this right now?"
The AI uses this as the first sentence of every response.
"""

from __future__ import annotations
from typing import Dict, Optional

from engine.models import (
    Direction,
    IndicatorSnapshot,
    MarketNarrative,
    MarketState,
    NarrativeState,
    TrendState,
    TrendStrength,
)
from strategy.timeframes import TFConsensus, compute_tf_consensus


def classify_narrative(state: MarketState) -> NarrativeState:
    """
    Classify the current market narrative from a full MarketState.

    Priority order (first match wins):
      1. BREAKOUT / BREAKDOWN  — structural, high-confidence
      2. REVERSAL              — confirmed flip
      3. TRENDING              — sustained directional move
      4. TRANSITION            — weakening trend, caution
      5. CHOP                  — no edge
    """
    consensus = compute_tf_consensus(state.trends)
    ind_1h    = state.indicators.get("1h")
    ind_15m   = state.indicators.get("15m")
    ind_5m    = state.indicators.get("5m")
    trend_1h  = state.trends.get("1h")
    trend_15m = state.trends.get("15m")

    # ── 1. Breakout / Breakdown ────────────────
    breakout_result = _check_breakout(state, consensus, ind_15m, ind_5m)
    if breakout_result:
        return breakout_result

    # ── 2. Reversal ────────────────────────────
    reversal_result = _check_reversal(state, consensus, trend_1h, trend_15m)
    if reversal_result:
        return reversal_result

    # ── 3. Trending ────────────────────────────
    trending_result = _check_trending(consensus, trend_1h, trend_15m, ind_1h)
    if trending_result:
        return trending_result

    # ── 4. Transition ──────────────────────────
    transition_result = _check_transition(consensus, trend_1h, trend_15m, ind_1h)
    if transition_result:
        return transition_result

    # ── 5. Default: Chop ───────────────────────
    return NarrativeState(
        narrative=MarketNarrative.CHOP,
        confidence=0.6,
        description="Range-bound — no clear directional edge. Avoid directional trades.",
        key_levels=_format_key_levels(state),
        bias_source=consensus.summary,
        trade_posture="stand_aside",
    )


# ──────────────────────────────────────────────
# Condition checks (each returns NarrativeState or None)
# ──────────────────────────────────────────────

def _check_breakout(
    state:    MarketState,
    consensus: TFConsensus,
    ind_15m:  Optional[IndicatorSnapshot],
    ind_5m:   Optional[IndicatorSnapshot],
) -> Optional[NarrativeState]:
    """
    Breakout: price clearing resistance with ATR expansion.
    Breakdown: price breaching support with ATR expansion.

    Conditions (breakout):
      - Price above nearest resistance level (within last 3 candles)
      - ATR on 15m or 5m expanded (> 1.2x recent average)
      - 15m bullish + HTF not bearish
      - Consensus weighted score >= 0.35

    Breakdown is the mirror.
    """
    if not ind_15m or not ind_5m:
        return None

    price = ind_15m.close
    atr   = ind_15m.atr

    # Check if price recently broke a resistance level
    broke_resistance = any(
        abs(price - r.price) / r.price < 0.003 or price > r.price
        for r in state.resistance_levels[:3]
    )
    broke_support = any(
        abs(price - s.price) / s.price < 0.003 or price < s.price
        for s in state.support_levels[:3]
    )

    # ATR expansion: current ATR vs EMA-20 derived typical range
    atr_expanded = False
    if atr and ind_15m.ema_20:
        # Approximate: ATR > 0.25% of price is elevated intraday
        atr_expanded = atr > (price * 0.0025)

    trend_15m = state.trends.get("15m")

    # Breakout
    if (broke_resistance
            and atr_expanded
            and trend_15m and trend_15m.direction == Direction.BULLISH
            and consensus.weighted_score >= 0.30
            and state.htf_bias != Direction.BEARISH):
        return NarrativeState(
            narrative=MarketNarrative.BREAKOUT,
            confidence=min(0.9, 0.6 + consensus.weighted_score),
            description=f"Breakout in progress — price clearing resistance near {price:.2f} with ATR expansion.",
            key_levels=_format_key_levels(state),
            bias_source=consensus.summary,
            trade_posture="long_bias",
        )

    # Breakdown
    if (broke_support
            and atr_expanded
            and trend_15m and trend_15m.direction == Direction.BEARISH
            and consensus.weighted_score <= -0.30
            and state.htf_bias != Direction.BULLISH):
        return NarrativeState(
            narrative=MarketNarrative.BREAKDOWN,
            confidence=min(0.9, 0.6 + abs(consensus.weighted_score)),
            description=f"Breakdown in progress — price breaching support near {price:.2f} with ATR expansion.",
            key_levels=_format_key_levels(state),
            bias_source=consensus.summary,
            trade_posture="short_bias",
        )

    return None


def _check_reversal(
    state:     MarketState,
    consensus: TFConsensus,
    trend_1h:  Optional[TrendState],
    trend_15m: Optional[TrendState],
) -> Optional[NarrativeState]:
    """
    Reversal: structure break confirmed + new direction holding.

    Conditions (bull reversal):
      - 1h or 15m structure_break == True
      - Consensus direction switched to BULLISH
      - Price reclaimed last swing low (held as support)
      - 15m direction BULLISH
    """
    if not trend_1h or not trend_15m:
        return None

    # Bull reversal: prior bear structure broken, now bullish
    if (trend_15m.structure_break
            and trend_15m.direction == Direction.BULLISH
            and consensus.direction == Direction.BULLISH
            and state.htf_bias != Direction.BEARISH):
        return NarrativeState(
            narrative=MarketNarrative.REVERSAL_BULL,
            confidence=0.65,
            description="Bullish reversal forming — prior bearish structure broken, 15m turning higher.",
            key_levels=_format_key_levels(state),
            bias_source=consensus.summary,
            trade_posture="long_bias",
        )

    # Bear reversal
    if (trend_15m.structure_break
            and trend_15m.direction == Direction.BEARISH
            and consensus.direction == Direction.BEARISH
            and state.htf_bias != Direction.BULLISH):
        return NarrativeState(
            narrative=MarketNarrative.REVERSAL_BEAR,
            confidence=0.65,
            description="Bearish reversal forming — prior bullish structure broken, 15m turning lower.",
            key_levels=_format_key_levels(state),
            bias_source=consensus.summary,
            trade_posture="short_bias",
        )

    return None


def _check_trending(
    consensus: TFConsensus,
    trend_1h:  Optional[TrendState],
    trend_15m: Optional[TrendState],
    ind_1h:    Optional[IndicatorSnapshot],
) -> Optional[NarrativeState]:
    """
    Trending: sustained directional move, strong consensus, aligned structure.

    Conditions (bull trend):
      - Consensus direction BULLISH, strength STRONG or MODERATE
      - 1h and 15m both BULLISH
      - EMA stack aligned on 1h
      - No weakening signals on 1h
    """
    if not trend_1h or not trend_15m:
        return None

    bull_trending = (
        consensus.direction == Direction.BULLISH
        and consensus.strength in (TrendStrength.STRONG, TrendStrength.MODERATE)
        and trend_1h.direction == Direction.BULLISH
        and trend_15m.direction == Direction.BULLISH
        and not trend_1h.is_weakening
        and ind_1h and ind_1h.ema_stack in ("bull_aligned", "mixed")
    )

    bear_trending = (
        consensus.direction == Direction.BEARISH
        and consensus.strength in (TrendStrength.STRONG, TrendStrength.MODERATE)
        and trend_1h.direction == Direction.BEARISH
        and trend_15m.direction == Direction.BEARISH
        and not trend_1h.is_weakening
        and ind_1h and ind_1h.ema_stack in ("bear_aligned", "mixed")
    )

    if bull_trending:
        conf = 0.7 + (0.2 if consensus.strength == TrendStrength.STRONG else 0.0)
        return NarrativeState(
            narrative=MarketNarrative.TRENDING_BULL,
            confidence=round(conf, 2),
            description=f"Bullish trend in progress ({consensus.strength.value} conviction). "
                        f"Look for pullback entries on 5m/15m.",
            key_levels=_format_key_levels(state=None),
            bias_source=consensus.summary,
            trade_posture="long_bias",
        )

    if bear_trending:
        conf = 0.7 + (0.2 if consensus.strength == TrendStrength.STRONG else 0.0)
        return NarrativeState(
            narrative=MarketNarrative.TRENDING_BEAR,
            confidence=round(conf, 2),
            description=f"Bearish trend in progress ({consensus.strength.value} conviction). "
                        f"Look for bounce rejections on 5m/15m.",
            key_levels=_format_key_levels(state=None),
            bias_source=consensus.summary,
            trade_posture="short_bias",
        )

    return None


def _check_transition(
    consensus: TFConsensus,
    trend_1h:  Optional[TrendState],
    trend_15m: Optional[TrendState],
    ind_1h:    Optional[IndicatorSnapshot],
) -> Optional[NarrativeState]:
    """
    Transition: a trend that is losing conviction.
    Classic signals: weakening detected, MACD histogram shrinking,
    mixed EMA stack, or 1h and 15m diverging.
    """
    if not trend_1h or not trend_15m:
        return None

    # Trend was directional but now weakening
    weakening = trend_1h.is_weakening or trend_15m.is_weakening

    # EMA stack went from aligned to mixed
    stack_mixed = ind_1h and ind_1h.ema_stack == "mixed"

    # 1h and 15m pointing different directions
    tf_divergence = (
        trend_1h.direction != Direction.NEUTRAL
        and trend_15m.direction != Direction.NEUTRAL
        and trend_1h.direction != trend_15m.direction
    )

    # Consensus weak
    consensus_weak = consensus.strength in (TrendStrength.WEAK, TrendStrength.NONE)

    transition_signals = sum([weakening, stack_mixed, tf_divergence, consensus_weak])

    if transition_signals >= 2:
        prior_direction = trend_1h.direction.value if trend_1h.direction != Direction.NEUTRAL else "unknown"
        return NarrativeState(
            narrative=MarketNarrative.TRANSITION,
            confidence=0.5 + (0.1 * transition_signals),
            description=f"Transition phase — prior {prior_direction} trend losing conviction. "
                        f"Reduce size, wait for clarity before new entries.",
            key_levels=_format_key_levels(state=None),
            bias_source=consensus.summary,
            trade_posture="neutral",
        )

    return None


# ──────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────

def _format_key_levels(state: Optional[MarketState]) -> str:
    if not state:
        return ""
    parts = []
    if state.resistance_levels:
        r = state.resistance_levels[0]
        parts.append(f"resistance {r.price:.2f} ({r.strength})")
    if state.support_levels:
        s = state.support_levels[0]
        parts.append(f"support {s.price:.2f} ({s.strength})")
    return ", ".join(parts) if parts else ""
