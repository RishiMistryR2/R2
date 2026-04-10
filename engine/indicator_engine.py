"""
IndicatorEngine: reads candle history, computes all indicators,
assesses trend direction/strength, detects S/R levels,
and assembles a MarketState snapshot.

Called by the main engine loop whenever a candle closes.
All math is delegated to indicators.py (pure functions).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from engine.candle_store import CandleStore
from engine.models import (
    Candle,
    Direction,
    IndicatorSnapshot,
    MarketState,
    SupportResistanceLevel,
    TrendState,
    TrendStrength,
)
from engine import indicators as ind
from config.settings import (
    ATR_PERIOD,
    EMA_PERIODS,
    MACD_FAST,
    MACD_SIGNAL,
    MACD_SLOW,
    MIN_CANDLES_FOR_INDICATORS,
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
        Called after each 1m close (or whenever you want a fresh snapshot).
        """
        now = datetime.now(tz=timezone.utc)
        state = MarketState(symbol=SYMBOL, timestamp=now)

        for tf in TIMEFRAMES:
            if self.store.candle_count(tf) < MIN_CANDLES_FOR_INDICATORS:
                continue  # not enough history yet

            snap  = self._compute_indicators(tf)
            trend = self._assess_trend(tf, snap)

            state.indicators[tf] = snap
            state.trends[tf]     = trend

        # S/R levels from 15m (good balance of noise vs significance)
        if self.store.candle_count("15m") >= MIN_CANDLES_FOR_INDICATORS:
            supports, resistances = self._find_sr_levels("15m")
            state.support_levels    = supports
            state.resistance_levels = resistances

        # Higher-timeframe bias from 1h + daily
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

        snap = IndicatorSnapshot(
            timeframe=tf,
            timestamp=datetime.now(tz=timezone.utc),
            close=closes[-1],
        )

        # EMAs
        snap.ema_9   = ind.ema(closes, 9)
        snap.ema_20  = ind.ema(closes, 20)
        snap.ema_50  = ind.ema(closes, 50)
        snap.ema_200 = ind.ema(closes, 200)

        # VWAP (intraday only — meaningful on 1m/5m)
        if tf in ("1m", "5m"):
            snap.vwap = self.store.get_vwap()
        else:
            snap.vwap = ind.vwap(highs, lows, closes, volumes)

        # RSI
        snap.rsi = ind.rsi(closes, RSI_PERIOD)

        # MACD
        snap.macd_line, snap.macd_signal, snap.macd_hist = ind.macd(
            closes, MACD_FAST, MACD_SLOW, MACD_SIGNAL
        )

        # ATR
        snap.atr = ind.atr(highs, lows, closes, ATR_PERIOD)

        # Derived labels
        snap.ema_stack     = ind.classify_ema_stack(
            snap.ema_9, snap.ema_20, snap.ema_50, snap.ema_200, snap.close
        )
        snap.rsi_zone      = self._classify_rsi(snap.rsi)
        snap.macd_bias     = self._classify_macd(snap.macd_line, snap.macd_signal, snap.macd_hist)
        snap.price_vs_vwap = self._classify_vwap(snap.close, snap.vwap)

        return snap

    # ──────────────────────────────────────────
    # Trend assessment
    # ──────────────────────────────────────────

    def _assess_trend(self, tf: str, snap: IndicatorSnapshot) -> TrendState:
        trend = TrendState(timeframe=tf)
        closes = self.store.get_closes(tf)
        highs  = self.store.get_highs(tf)
        lows   = self.store.get_lows(tf)

        if len(closes) < SWING_LOOKBACK * 2 + 5:
            return trend

        # ── Direction ──────────────────────────
        bull_points = 0
        bear_points = 0

        # EMA stack
        if snap.ema_stack == "bull_aligned":
            bull_points += 3
        elif snap.ema_stack == "bear_aligned":
            bear_points += 3

        # Price vs key EMAs
        close = snap.close
        if snap.ema_20 and close > snap.ema_20:
            bull_points += 1
        elif snap.ema_20 and close < snap.ema_20:
            bear_points += 1

        if snap.ema_50 and close > snap.ema_50:
            bull_points += 1
        elif snap.ema_50 and close < snap.ema_50:
            bear_points += 1

        # MACD
        if snap.macd_bias == "bullish":
            bull_points += 1
        elif snap.macd_bias == "bearish":
            bear_points += 1

        # Price vs VWAP
        if snap.price_vs_vwap == "above":
            bull_points += 1
        elif snap.price_vs_vwap == "below":
            bear_points += 1

        if bull_points > bear_points + 1:
            trend.direction = Direction.BULLISH
        elif bear_points > bull_points + 1:
            trend.direction = Direction.BEARISH
        else:
            trend.direction = Direction.NEUTRAL

        # ── Strength ───────────────────────────
        delta = abs(bull_points - bear_points)
        if delta >= 5:
            trend.strength = TrendStrength.STRONG
        elif delta >= 3:
            trend.strength = TrendStrength.MODERATE
        elif delta >= 1:
            trend.strength = TrendStrength.WEAK
        else:
            trend.strength = TrendStrength.NONE

        # ── Swing structure (HH/LL) ────────────
        swing_highs = ind.find_swing_highs(highs, SWING_LOOKBACK)
        swing_lows  = ind.find_swing_lows(lows, SWING_LOOKBACK)

        if len(swing_highs) >= 2:
            trend.last_swing_high = swing_highs[-1][1]
            trend.higher_highs    = swing_highs[-1][1] > swing_highs[-2][1]
        if len(swing_lows) >= 2:
            trend.last_swing_low = swing_lows[-1][1]
            trend.lower_lows     = swing_lows[-1][1] < swing_lows[-2][1]

        # ── Weakening detection ─────────────────
        trend.is_weakening = self._detect_weakening(trend, snap, swing_highs, swing_lows)

        # ── Structure break ─────────────────────
        trend.structure_break = self._detect_structure_break(
            trend, snap, swing_highs, swing_lows, close
        )

        return trend

    def _detect_weakening(
        self,
        trend:        TrendState,
        snap:         IndicatorSnapshot,
        swing_highs:  list,
        swing_lows:   list,
    ) -> bool:
        """
        Trend is weakening when:
        - Bullish but RSI diverging (lower highs on RSI while price makes higher highs)
        - MACD histogram shrinking while price continues
        - Price makes new high but EMA stack is no longer fully aligned
        """
        if trend.direction == Direction.BULLISH:
            # Price making new highs but RSI in overbought and diverging
            if snap.rsi and snap.rsi > RSI_OVERBOUGHT:
                if snap.macd_hist is not None and snap.macd_hist < 0:
                    return True
            # Price above EMAs but stack mixed
            if snap.ema_stack == "mixed" and snap.close > (snap.ema_50 or 0):
                return True

        elif trend.direction == Direction.BEARISH:
            if snap.rsi and snap.rsi < RSI_OVERSOLD:
                if snap.macd_hist is not None and snap.macd_hist > 0:
                    return True
            if snap.ema_stack == "mixed" and snap.close < (snap.ema_50 or float("inf")):
                return True

        return False

    def _detect_structure_break(
        self,
        trend:       TrendState,
        snap:        IndicatorSnapshot,
        swing_highs: list,
        swing_lows:  list,
        close:       float,
    ) -> bool:
        """
        Bullish structure break: price closes below last significant swing low.
        Bearish structure break: price closes above last significant swing high.
        """
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
    ) -> tuple[List[SupportResistanceLevel], List[SupportResistanceLevel]]:
        """
        Identify S/R from swing highs/lows on the given timeframe.
        Clusters levels that are within ATR/2 of each other (same zone).
        """
        highs  = self.store.get_highs(tf)
        lows   = self.store.get_lows(tf)
        closes = self.store.get_closes(tf)

        if not highs:
            return [], []

        atr_val = ind.atr(highs, lows, closes, ATR_PERIOD) or 1.0
        zone    = atr_val * SR_ZONE_ATR_MULT

        swing_highs = ind.find_swing_highs(highs, SWING_LOOKBACK)
        swing_lows  = ind.find_swing_lows(lows, SWING_LOOKBACK)

        resistances = self._cluster_levels(
            [p for _, p in swing_highs], zone, "resistance"
        )
        supports = self._cluster_levels(
            [p for _, p in swing_lows], zone, "support"
        )

        # Filter by minimum touches and sort by proximity to current price
        current_price = closes[-1]
        resistances = [
            r for r in resistances if r.touches >= SR_MIN_TOUCHES
        ]
        supports = [
            s for s in supports if s.touches >= SR_MIN_TOUCHES
        ]

        # Sort: resistances ascending (nearest first), supports descending (nearest first)
        resistances.sort(key=lambda r: r.price)
        supports.sort(key=lambda s: -s.price)

        # Only return levels near current price
        resistances = [r for r in resistances if r.price > current_price][:5]
        supports    = [s for s in supports    if s.price < current_price][:5]

        return supports, resistances

    @staticmethod
    def _cluster_levels(
        prices:     List[float],
        zone:       float,
        level_type: str,
    ) -> List[SupportResistanceLevel]:
        """Merge prices within `zone` distance into a single level."""
        if not prices:
            return []

        clusters: List[SupportResistanceLevel] = []
        for price in sorted(prices):
            merged = False
            for cluster in clusters:
                if abs(cluster.price - price) <= zone:
                    # Update cluster to weighted average
                    cluster.price   = (cluster.price * cluster.touches + price) / (cluster.touches + 1)
                    cluster.touches += 1
                    merged = True
                    break
            if not merged:
                clusters.append(SupportResistanceLevel(
                    price=price, level_type=level_type, touches=1
                ))

        # Assign strength
        for c in clusters:
            if c.touches >= 4:
                c.strength = "strong"
            elif c.touches >= 2:
                c.strength = "moderate"
            else:
                c.strength = "weak"

        return clusters

    # ──────────────────────────────────────────
    # Higher-timeframe bias
    # ──────────────────────────────────────────

    def _compute_htf_bias(self, state: MarketState) -> Direction:
        """
        Combine 1h and daily trend to get overall directional bias.
        Used to align trade direction on lower timeframes.
        """
        bias_points = 0

        for tf in ("1h", "1d"):
            trend = state.trends.get(tf)
            if trend is None:
                continue
            if trend.direction == Direction.BULLISH:
                weight = 2 if trend.strength == TrendStrength.STRONG else 1
                bias_points += weight
            elif trend.direction == Direction.BEARISH:
                weight = 2 if trend.strength == TrendStrength.STRONG else 1
                bias_points -= weight

        if bias_points >= 2:
            return Direction.BULLISH
        if bias_points <= -2:
            return Direction.BEARISH
        return Direction.NEUTRAL

    # ──────────────────────────────────────────
    # Label helpers
    # ──────────────────────────────────────────

    @staticmethod
    def _classify_rsi(rsi_val: Optional[float]) -> Optional[str]:
        if rsi_val is None:
            return None
        if rsi_val >= RSI_OVERBOUGHT:
            return "overbought"
        if rsi_val <= RSI_OVERSOLD:
            return "oversold"
        if rsi_val >= RSI_NEUTRAL_HIGH:
            return "bullish_neutral"
        if rsi_val <= RSI_NEUTRAL_LOW:
            return "bearish_neutral"
        return "neutral"

    @staticmethod
    def _classify_macd(
        line: Optional[float],
        signal: Optional[float],
        hist: Optional[float],
    ) -> Optional[str]:
        if line is None or signal is None or hist is None:
            return None
        if line > signal and hist > 0:
            return "bullish"
        if line < signal and hist < 0:
            return "bearish"
        if hist > 0:
            return "bullish_crossing"
        return "bearish_crossing"

    @staticmethod
    def _classify_vwap(close: float, vwap: Optional[float]) -> Optional[str]:
        if vwap is None:
            return None
        return "above" if close >= vwap else "below"
