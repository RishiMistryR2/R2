"""
Market Regime Engine.

Classifies the current market environment into one of seven regimes.
Uses hysteresis — requires N candles of consistent signals before
switching regimes, preventing thrashing on noisy data.

Regime vs Narrative distinction:
  MarketNarrative: instantaneous condition, changes per candle
  MarketRegime:    persistent environment, requires 3+ confirming candles

The regime is the lens through which all setup scoring happens.
A B-grade setup in a TREND_BULL regime trades better than an A-grade
setup in CHOP. The scoring multiplier makes this quantitative.

Classification logic (priority order):
  1. CHOP:               low ATR, EMA stack mixed, no clear HH/HL or LH/LL
  2. BREAKOUT_ATTEMPT:   price testing key resistance/support, no close through yet
  3. BREAKOUT_CONFIRMED: closed through key level + retest attempt + holding
  4. TRANSITION:         prior trend losing conviction across 2+ timeframes
  5. TREND_BULL:         HH+HL structure, aligned EMAs, sustained momentum
  6. TREND_BEAR:         LH+LL structure, aligned EMAs, sustained momentum
  7. RANGE:              defined S/R boundaries, oscillating, no expansion
"""

from __future__ import annotations
from collections import deque
from datetime import datetime, timezone
from typing import Deque, Dict, Optional

from engine.models import (
    Direction,
    IndicatorSnapshot,
    MarketRegime,
    MarketState,
    REGIME_ENTRY_PERMITTED,
    REGIME_PREFERRED_SETUPS,
    REGIME_SCORE_MULTIPLIERS,
    RegimeState,
    TrendState,
    TrendStrength,
)


# Minimum candles in a condition before regime switches
REGIME_PERSISTENCE = 3


class MarketRegimeEngine:
    """
    Classifies the market regime with hysteresis.

    Usage:
        engine = MarketRegimeEngine()
        regime_state = engine.classify(market_state)
    """

    def __init__(self) -> None:
        self._history: Deque[MarketRegime] = deque(maxlen=10)
        self._current: Optional[RegimeState] = None
        self._candles_in_regime: int = 0

    def classify(self, state: MarketState) -> RegimeState:
        """
        Classify current regime. Respects hysteresis — regime only changes
        after REGIME_PERSISTENCE consistent signals.
        """
        candidate = self._compute_candidate(state)

        if self._current is None:
            # First call
            self._current = self._build_regime_state(candidate, state, 0)
            self._candles_in_regime = 1
        elif candidate == self._current.regime:
            # Same regime — increment persistence
            self._candles_in_regime += 1
            self._current.candles_in_regime = self._candles_in_regime
        else:
            # Potential regime change — check persistence
            self._history.append(candidate)
            recent = list(self._history)[-REGIME_PERSISTENCE:]
            if len(recent) >= REGIME_PERSISTENCE and all(r == candidate for r in recent):
                # Regime confirmed — switch
                self._current = self._build_regime_state(candidate, state, 0)
                self._candles_in_regime = 1
            # else: keep current regime (hysteresis)

        return self._current

    # ──────────────────────────────────────────
    # Classification logic
    # ──────────────────────────────────────────

    def _compute_candidate(self, state: MarketState) -> MarketRegime:
        """Compute the most likely regime from current state. No hysteresis here."""
        trend_1h  = state.trends.get("1h")
        trend_15m = state.trends.get("15m")
        trend_5m  = state.trends.get("5m")
        ind_1h    = state.indicators.get("1h")
        ind_15m   = state.indicators.get("15m")

        # ── 1. Chop check (priority — no other regime if choppy) ──
        if self._is_chop(state, trend_1h, trend_15m, ind_1h):
            return MarketRegime.CHOP

        # ── 2. Breakout checks ──────────────────
        if self._is_breakout_confirmed(state, trend_15m, ind_15m):
            return MarketRegime.BREAKOUT_CONFIRMED
        if self._is_breakout_attempt(state, trend_15m, ind_15m):
            return MarketRegime.BREAKOUT_ATTEMPT

        # ── 3. Transition check ─────────────────
        if self._is_transition(trend_1h, trend_15m, ind_1h):
            return MarketRegime.TRANSITION

        # ── 4. Trend checks ─────────────────────
        if self._is_trending_bull(state, trend_1h, trend_15m, ind_1h):
            return MarketRegime.TREND_BULL
        if self._is_trending_bear(state, trend_1h, trend_15m, ind_1h):
            return MarketRegime.TREND_BEAR

        # ── 5. Range ────────────────────────────
        return MarketRegime.RANGE

    # ──────────────────────────────────────────
    # Individual regime classifiers
    # ──────────────────────────────────────────

    def _is_chop(
        self,
        state:     MarketState,
        trend_1h:  Optional[TrendState],
        trend_15m: Optional[TrendState],
        ind_1h:    Optional[IndicatorSnapshot],
    ) -> bool:
        """
        Chop: all of:
          - 1h direction NEUTRAL or EMA stack mixed
          - 15m direction NEUTRAL
          - Both trends showing TrendStrength.NONE or WEAK
          - No clear HH/HL or LH/LL structure
        """
        if not trend_1h or not trend_15m:
            return False

        both_neutral = (
            trend_1h.direction == Direction.NEUTRAL
            and trend_15m.direction == Direction.NEUTRAL
        )
        both_weak = (
            trend_1h.strength in (TrendStrength.NONE, TrendStrength.WEAK)
            and trend_15m.strength in (TrendStrength.NONE, TrendStrength.WEAK)
        )
        ema_mixed = ind_1h and ind_1h.ema_stack == "mixed"
        no_structure = (
            not trend_1h.higher_highs and not trend_1h.lower_lows
        )

        return (both_neutral or (both_weak and ema_mixed)) and no_structure

    def _is_breakout_confirmed(
        self,
        state:     MarketState,
        trend_15m: Optional[TrendState],
        ind_15m:   Optional[IndicatorSnapshot],
    ) -> bool:
        """
        Breakout confirmed: price closed through a key level,
        attempted retest, and is holding.
        Proxy: resistance_levels empty (price through all of them) + ATR expanded.
        """
        if not trend_15m or not ind_15m:
            return False

        # Price above all tracked resistance OR below all tracked support
        if ind_15m.close and state.resistance_levels:
            above_all_resistance = all(
                ind_15m.close > r.price for r in state.resistance_levels[:3]
            )
            if above_all_resistance and trend_15m.direction == Direction.BULLISH:
                return True

        if ind_15m.close and state.support_levels:
            below_all_support = all(
                ind_15m.close < s.price for s in state.support_levels[:3]
            )
            if below_all_support and trend_15m.direction == Direction.BEARISH:
                return True

        return False

    def _is_breakout_attempt(
        self,
        state:     MarketState,
        trend_15m: Optional[TrendState],
        ind_15m:   Optional[IndicatorSnapshot],
    ) -> bool:
        """
        Breakout attempt: price within 0.5% of a key level but hasn't closed through.
        """
        if not ind_15m or not ind_15m.close:
            return False

        price = ind_15m.close
        atr   = ind_15m.atr or (price * 0.002)

        # Testing resistance
        for r in state.resistance_levels[:2]:
            if abs(price - r.price) < atr * 0.5 and price < r.price:
                return True

        # Testing support
        for s in state.support_levels[:2]:
            if abs(price - s.price) < atr * 0.5 and price > s.price:
                return True

        return False

    def _is_transition(
        self,
        trend_1h:  Optional[TrendState],
        trend_15m: Optional[TrendState],
        ind_1h:    Optional[IndicatorSnapshot],
    ) -> bool:
        """
        Transition: prior trend losing conviction.
        2+ of: weakening on 1h, mixed EMA stack, 1h/15m divergence.
        """
        if not trend_1h or not trend_15m:
            return False

        signals = 0
        if trend_1h.is_weakening:
            signals += 1
        if trend_15m.is_weakening:
            signals += 1
        if ind_1h and ind_1h.ema_stack == "mixed":
            signals += 1
        if (trend_1h.direction != Direction.NEUTRAL
                and trend_15m.direction != Direction.NEUTRAL
                and trend_1h.direction != trend_15m.direction):
            signals += 2   # strong signal

        return signals >= 2

    def _is_trending_bull(
        self,
        state:     MarketState,
        trend_1h:  Optional[TrendState],
        trend_15m: Optional[TrendState],
        ind_1h:    Optional[IndicatorSnapshot],
    ) -> bool:
        if not trend_1h or not trend_15m:
            return False
        return (
            trend_1h.direction == Direction.BULLISH
            and trend_1h.strength in (TrendStrength.STRONG, TrendStrength.MODERATE)
            and trend_15m.direction == Direction.BULLISH
            and trend_1h.higher_highs
            and not trend_1h.is_weakening
            and (ind_1h is None or ind_1h.ema_stack in ("bull_aligned", "mixed"))
        )

    def _is_trending_bear(
        self,
        state:     MarketState,
        trend_1h:  Optional[TrendState],
        trend_15m: Optional[TrendState],
        ind_1h:    Optional[IndicatorSnapshot],
    ) -> bool:
        if not trend_1h or not trend_15m:
            return False
        return (
            trend_1h.direction == Direction.BEARISH
            and trend_1h.strength in (TrendStrength.STRONG, TrendStrength.MODERATE)
            and trend_15m.direction == Direction.BEARISH
            and trend_1h.lower_lows
            and not trend_1h.is_weakening
            and (ind_1h is None or ind_1h.ema_stack in ("bear_aligned", "mixed"))
        )

    # ──────────────────────────────────────────
    # RegimeState builder
    # ──────────────────────────────────────────

    def _build_regime_state(
        self,
        regime:             MarketRegime,
        state:              MarketState,
        candles_in_regime:  int,
    ) -> RegimeState:
        descriptions = {
            MarketRegime.TREND_BULL:         "Bullish trend — favor continuations and pullback entries",
            MarketRegime.TREND_BEAR:         "Bearish trend — favor continuations and bounce rejections",
            MarketRegime.RANGE:              "Range-bound — prefer mean-reversion plays at boundaries",
            MarketRegime.CHOP:               "Choppy — no edge. Stand aside until clarity emerges",
            MarketRegime.BREAKOUT_ATTEMPT:   "Breakout attempt in progress — wait for close confirmation",
            MarketRegime.BREAKOUT_CONFIRMED: "Breakout confirmed — enter on first clean pullback",
            MarketRegime.TRANSITION:         "Regime transition — reduce size, wait for new structure",
        }
        cautions = {
            MarketRegime.CHOP:             "Avoid forcing entries. Scoring heavily penalized.",
            MarketRegime.BREAKOUT_ATTEMPT: "Entries blocked until breakout confirmed with a close.",
            MarketRegime.TRANSITION:       "Prior setups may no longer apply. Rebuild thesis.",
            MarketRegime.RANGE:            "Do not chase — only trade at the edges.",
        }
        confidence = _regime_confidence(state)
        prev = self._current.regime if self._current else None
        return RegimeState(
            regime=regime,
            confidence=confidence,
            score_multiplier=REGIME_SCORE_MULTIPLIERS[regime],
            entry_permitted=REGIME_ENTRY_PERMITTED[regime],
            description=descriptions.get(regime, ""),
            preferred_setups=REGIME_PREFERRED_SETUPS.get(regime, []),
            caution_note=cautions.get(regime, ""),
            candles_in_regime=candles_in_regime,
            raw_scores=_compute_raw_scores(state),
            prev_regime=prev,
        )


def _regime_confidence(state: MarketState) -> float:
    """Rough confidence based on TF alignment."""
    from strategy.timeframes import compute_tf_consensus
    consensus = compute_tf_consensus(state.trends)
    base = abs(consensus.weighted_score)
    return round(min(0.95, 0.4 + base * 0.9), 2)


def _compute_raw_scores(state: MarketState) -> dict:
    """
    Score each regime using the signal table from the design spec.
    Returns {regime_value: int_score} for transparency in AI context.
    """
    trend_1h  = state.trends.get("1h")
    trend_15m = state.trends.get("15m")
    ind_1h    = state.indicators.get("1h")

    scores = {r.value: 0 for r in MarketRegime}

    if not trend_1h or not trend_15m:
        return scores

    # S1: higher_highs on 1h
    if trend_1h.higher_highs:
        scores["trend_bull"] += 3
        scores["trend_bear"] -= 1
    # S2: lower_lows on 1h
    if trend_1h.lower_lows:
        scores["trend_bear"] += 3
        scores["trend_bull"] -= 1
    # Neither S1 nor S2
    if not trend_1h.higher_highs and not trend_1h.lower_lows:
        scores["range"] += 2
        scores["chop"]  += 1

    # S3: 1h bullish + strong/moderate
    if (trend_1h.direction == Direction.BULLISH
            and trend_1h.strength in (TrendStrength.STRONG, TrendStrength.MODERATE)):
        scores["trend_bull"] += 2
    # S4: 1h bearish + strong/moderate
    if (trend_1h.direction == Direction.BEARISH
            and trend_1h.strength in (TrendStrength.STRONG, TrendStrength.MODERATE)):
        scores["trend_bear"] += 2
    # S5: neutral
    if trend_1h.direction == Direction.NEUTRAL:
        scores["range"] += 2
        scores["chop"]  += 2
    # S6: weak/none strength
    if trend_1h.strength in (TrendStrength.NONE, TrendStrength.WEAK):
        scores["range"] += 1
        scores["chop"]  += 2

    # EMA stack signals
    if ind_1h:
        if ind_1h.ema_stack == "bull_aligned":
            scores["trend_bull"] += 2
        elif ind_1h.ema_stack == "bear_aligned":
            scores["trend_bear"] += 2
        elif ind_1h.ema_stack == "mixed":
            scores["range"] += 1
            scores["chop"]  += 2

        # MACD histogram
        if ind_1h.macd_hist is not None:
            if ind_1h.macd_hist > 0.001:
                scores["trend_bull"] += 1
            elif ind_1h.macd_hist < -0.001:
                scores["trend_bear"] += 1
            else:
                scores["range"] += 1
                scores["chop"]  += 1

        # ATR compression (low volatility favors range/chop)
        if ind_1h.atr and ind_1h.close and ind_1h.close > 0:
            atr_pct = ind_1h.atr / ind_1h.close
            if atr_pct < 0.002:
                scores["range"] += 1
                scores["chop"]  += 1

    # Defined S/R levels favor RANGE
    if len(state.support_levels) >= 2 and len(state.resistance_levels) >= 2:
        scores["range"] += 1

    # Clamp all to 0 minimum
    return {k: max(0, v) for k, v in scores.items()}
