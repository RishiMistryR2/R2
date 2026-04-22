"""
engine/indicator_engine.py

Reads candle history from CandleStore, computes all indicators,
assesses trend direction/strength, detects S/R levels, applies
readiness and premarket penalties, and assembles MarketState.

All math is delegated to indicators.py (pure functions).
Readiness is computed from CandleStore.get_readiness().
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional, Tuple

from engine.candle_store import CandleStore
from engine.models import (
    Direction,
    IndicatorSnapshot,
    MarketState,
    Readiness,
    SessionState,
    SessionPhase,
    SupportResistanceLevel,
    TrendState,
    TrendStrength,
)
from engine.readiness import (
    compute_readiness,
    readiness_regime_confidence_cap,
    premarket_confidence_reduction,
)
from engine import indicators as ind
from config.settings import (
    ATR_PERIOD,
    MACD_FAST,
    MACD_SIGNAL,
    MACD_SLOW,
    MIN_CANDLES_FOR_INDICATORS,
    PREMARKET_WINDOW_FRACTION,
    RSI_NEUTRAL_HIGH,
    RSI_NEUTRAL_LOW,
    RSI_OVERBOUGHT,
    RSI_OVERSOLD,
    RSI_PERIOD,
    SR_MIN_TOUCHES,
    SR_ZONE_ATR_MULT,
    SWING_LOOKBACK,
    SYMBOL,
    TIMEFRAMES,
)


class IndicatorEngine:
    """
    Computes all indicators and assembles MarketState.

    Usage:
        engine = IndicatorEngine(store)
        state  = engine.compute()   # call after each candle close
    """

    def __init__(self, store: CandleStore) -> None:
        self.store = store

    # ──────────────────────────────────────────
    # Public
    # ──────────────────────────────────────────

    def compute(self) -> MarketState:
        """
        Build a complete MarketState from current candle history.
        Readiness is computed first and governs what gets populated.
        """
        now = datetime.now(tz=timezone.utc)

        readiness, readiness_note = self.store.get_readiness()

        state = MarketState(symbol=SYMBOL, timestamp=now)

        # Attach a minimal SessionState with the readiness level so downstream
        # components (scorer, gates) have it without importing the store.
        # session_state.py will overwrite this with full phase/time info.
        state.session = SessionState(
            phase=SessionPhase.MIDDAY,   # placeholder — replaced by strategy engine
            readiness=readiness,
        )

        if readiness == Readiness.NOT_READY:
            # Not enough history to compute anything meaningful
            return state

        for tf in TIMEFRAMES:
            if self.store.candle_count(tf) < MIN_CANDLES_FOR_INDICATORS:
                continue

            snap  = self._compute_indicators(tf)
            trend = self._assess_trend(tf, snap)

            state.indicators[tf] = snap
            state.trends[tf]     = trend

        # S/R from 15m (balance of noise vs significance)
        if self.store.candle_count("15m") >= MIN_CANDLES_FOR_INDICATORS:
            supports, resistances = self._find_sr_levels("15m")
            state.support_levels    = supports
            state.resistance_levels = resistances

        # HTF bias
        state.htf_bias = self._compute_htf_bias(state)

        return state

    # ──────────────────────────────────────────
    # Indicator computation
    # ──────────────────────────────────────────

    def _compute_indicators(self, tf: str) -> IndicatorSnapshot:
        closes  = self.store.get_closes(tf)
        highs   = self.store.get_highs(tf)
        lows    = self.store.get_lows(tf)
        volumes = self.store.get_volumes(tf)

        if not closes:
            return IndicatorSnapshot(timeframe=tf, timestamp=datetime.now(tz=timezone.utc))

        now  = datetime.now(tz=timezone.utc)
        snap = IndicatorSnapshot(
            timeframe=tf,
            timestamp=now,
            close=closes[-1],
        )

        snap.ema_9   = ind.ema(closes, 9)
        snap.ema_20  = ind.ema(closes, 20)
        snap.ema_50  = ind.ema(closes, 50)
        snap.ema_200 = ind.ema(closes, 200)

        # VWAP: intraday accumulator for 1m/5m; computed over window for others
        if tf in ("1m", "5m"):
            snap.vwap = self.store.get_vwap()
        else:
            snap.vwap = ind.vwap(highs, lows, closes, volumes)

        snap.rsi = ind.rsi(closes, RSI_PERIOD)

        snap.macd_line, snap.macd_signal, snap.macd_hist = ind.macd(
            closes, MACD_FAST, MACD_SLOW, MACD_SIGNAL
        )

        snap.atr = ind.atr(highs, lows, closes, ATR_PERIOD)

        snap.ema_stack     = ind.classify_ema_stack(
            snap.ema_9, snap.ema_20, snap.ema_50, snap.ema_200, snap.close
        )
        snap.rsi_zone      = self._classify_rsi(snap.rsi)
        snap.macd_bias     = self._classify_macd(snap.macd_line, snap.macd_signal, snap.macd_hist)
        snap.price_vs_vwap = self._classify_vwap(snap.close, snap.vwap)

        # Premarket session mix flag
        pre_frac = self.store.premarket_fraction(tf)
        snap.session_mix = pre_frac > PREMARKET_WINDOW_FRACTION

        return snap

    # ──────────────────────────────────────────
    # Trend assessment
    # ──────────────────────────────────────────

    def _assess_trend(self, tf: str, snap: IndicatorSnapshot) -> TrendState:
        trend  = TrendState(timeframe=tf)
        closes = self.store.get_closes(tf)
        highs  = self.store.get_highs(tf)
        lows   = self.store.get_lows(tf)

        if len(closes) < SWING_LOOKBACK * 2 + 5:
            return trend

        # ── Direction scoring ─────────────────
        bull = bear = 0

        if snap.ema_stack == "bull_aligned":
            bull += 3
        elif snap.ema_stack == "bear_aligned":
            bear += 3

        close = snap.close
        if snap.ema_20:
            if close > snap.ema_20: bull += 1
            else:                   bear += 1
        if snap.ema_50:
            if close > snap.ema_50: bull += 1
            else:                   bear += 1

        if snap.macd_bias == "bullish":   bull += 1
        elif snap.macd_bias == "bearish": bear += 1

        if snap.price_vs_vwap == "above": bull += 1
        elif snap.price_vs_vwap == "below": bear += 1

        if bull > bear + 1:     trend.direction = Direction.BULLISH
        elif bear > bull + 1:   trend.direction = Direction.BEARISH
        else:                   trend.direction = Direction.NEUTRAL

        # ── Strength ──────────────────────────
        delta = abs(bull - bear)
        if delta >= 5:    trend.strength = TrendStrength.STRONG
        elif delta >= 3:  trend.strength = TrendStrength.MODERATE
        elif delta >= 1:  trend.strength = TrendStrength.WEAK
        else:             trend.strength = TrendStrength.NONE

        # ── Swing structure ───────────────────
        swing_highs = ind.find_swing_highs(highs, SWING_LOOKBACK)
        swing_lows  = ind.find_swing_lows(lows,  SWING_LOOKBACK)

        if len(swing_highs) >= 2:
            trend.last_swing_high = swing_highs[-1][1]
            trend.higher_highs    = swing_highs[-1][1] > swing_highs[-2][1]
        if len(swing_lows) >= 2:
            trend.last_swing_low = swing_lows[-1][1]
            trend.lower_lows     = swing_lows[-1][1] < swing_lows[-2][1]

        # ── Weakening ─────────────────────────
        trend.is_weakening = self._detect_weakening(trend, snap)

        # ── Structure break ───────────────────
        trend.structure_break = self._detect_structure_break(trend, snap, close)

        return trend

    def _detect_weakening(
        self,
        trend: TrendState,
        snap:  IndicatorSnapshot,
    ) -> bool:
        """2 of 3 signals required."""
        signals = 0

        if trend.direction == Direction.BULLISH:
            if snap.rsi and snap.rsi > RSI_OVERBOUGHT and snap.macd_hist and snap.macd_hist < 0:
                signals += 1
            if snap.macd_hist is not None and snap.macd_hist < 0:
                signals += 1
            if snap.ema_stack == "mixed" and snap.close > (snap.ema_50 or 0):
                signals += 1
        elif trend.direction == Direction.BEARISH:
            if snap.rsi and snap.rsi < RSI_OVERSOLD and snap.macd_hist and snap.macd_hist > 0:
                signals += 1
            if snap.macd_hist is not None and snap.macd_hist > 0:
                signals += 1
            if snap.ema_stack == "mixed" and snap.close < (snap.ema_50 or float("inf")):
                signals += 1

        return signals >= 2

    def _detect_structure_break(
        self,
        trend: TrendState,
        snap:  IndicatorSnapshot,
        close: float,
    ) -> bool:
        if trend.direction == Direction.BULLISH and trend.last_swing_low:
            return close < trend.last_swing_low
        if trend.direction == Direction.BEARISH and trend.last_swing_high:
            return close > trend.last_swing_high
        return False

    # ──────────────────────────────────────────
    # Support / Resistance
    # ──────────────────────────────────────────

    def _find_sr_levels(
        self,
        tf: str,
    ) -> Tuple[List[SupportResistanceLevel], List[SupportResistanceLevel]]:
        highs  = self.store.get_highs(tf)
        lows   = self.store.get_lows(tf)
        closes = self.store.get_closes(tf)

        if not highs:
            return [], []

        atr_val = ind.atr(highs, lows, closes, ATR_PERIOD) or 1.0
        zone    = atr_val * SR_ZONE_ATR_MULT

        swing_highs = ind.find_swing_highs(highs, SWING_LOOKBACK)
        swing_lows  = ind.find_swing_lows(lows,  SWING_LOOKBACK)

        resistances = self._cluster_levels([p for _, p in swing_highs], zone, "resistance")
        supports    = self._cluster_levels([p for _, p in swing_lows],  zone, "support")

        current = closes[-1]

        resistances = sorted(
            [r for r in resistances if r.touches >= SR_MIN_TOUCHES and r.price > current],
            key=lambda r: r.price,
        )[:5]
        supports = sorted(
            [s for s in supports if s.touches >= SR_MIN_TOUCHES and s.price < current],
            key=lambda s: -s.price,
        )[:5]

        return supports, resistances

    @staticmethod
    def _cluster_levels(
        prices:     List[float],
        zone:       float,
        level_type: str,
    ) -> List[SupportResistanceLevel]:
        if not prices:
            return []
        clusters: List[SupportResistanceLevel] = []
        for price in sorted(prices):
            merged = False
            for cluster in clusters:
                if abs(cluster.price - price) <= zone:
                    cluster.price   = (cluster.price * cluster.touches + price) / (cluster.touches + 1)
                    cluster.touches += 1
                    merged = True
                    break
            if not merged:
                clusters.append(SupportResistanceLevel(
                    price=price, level_type=level_type, touches=1
                ))
        for c in clusters:
            if c.touches >= 4:   c.strength = "strong"
            elif c.touches >= 2: c.strength = "moderate"
            else:                c.strength = "weak"
        return clusters

    # ──────────────────────────────────────────
    # HTF bias
    # ──────────────────────────────────────────

    def _compute_htf_bias(self, state: MarketState) -> Direction:
        bias = 0
        for tf in ("1h", "1d"):
            trend = state.trends.get(tf)
            if trend is None:
                continue
            w = 2 if trend.strength == TrendStrength.STRONG else 1
            if trend.direction == Direction.BULLISH:   bias += w
            elif trend.direction == Direction.BEARISH: bias -= w

        if bias >= 2:   return Direction.BULLISH
        if bias <= -2:  return Direction.BEARISH
        return Direction.NEUTRAL

    # ──────────────────────────────────────────
    # Label helpers
    # ──────────────────────────────────────────

    @staticmethod
    def _classify_rsi(rsi_val: Optional[float]) -> Optional[str]:
        if rsi_val is None:
            return None
        if rsi_val >= RSI_OVERBOUGHT:  return "overbought"
        if rsi_val <= RSI_OVERSOLD:    return "oversold"
        if rsi_val >= RSI_NEUTRAL_HIGH: return "bullish_neutral"
        if rsi_val <= RSI_NEUTRAL_LOW:  return "bearish_neutral"
        return "neutral"

    @staticmethod
    def _classify_macd(
        line:   Optional[float],
        signal: Optional[float],
        hist:   Optional[float],
    ) -> Optional[str]:
        if line is None or signal is None or hist is None:
            return None
        if line > signal and hist > 0:   return "bullish"
        if line < signal and hist < 0:   return "bearish"
        if hist > 0:                     return "bullish_crossing"
        return "bearish_crossing"

    @staticmethod
    def _classify_vwap(close: float, vwap: Optional[float]) -> Optional[str]:
        if vwap is None:
            return None
        return "above" if close >= vwap else "below"