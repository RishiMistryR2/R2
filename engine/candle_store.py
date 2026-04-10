"""
CandleStore: rolling per-timeframe candle history.

Design:
- 1m candles are ingested directly from the Alpaca feed.
- Higher timeframes are built by aggregating 1m candles.
- Each timeframe is a deque capped at CANDLE_HISTORY[tf] candles.
- The currently-forming candle is tracked separately (`live_candle`).

Thread safety: the store is mutated from the WebSocket callback thread.
A simple asyncio.Lock protects writes. Reads from the strategy/AI layer
should snapshot the state before iterating.
"""

from __future__ import annotations

import asyncio
from collections import deque
from datetime import datetime, timezone
from typing import Dict, Deque, Optional, List

from engine.models import Candle
from config.settings import (
    TIMEFRAMES,
    CANDLE_HISTORY,
    TIMEFRAME_SECONDS,
)


class CandleStore:
    """
    Maintains rolling OHLCV history for every configured timeframe.

    Usage:
        store = CandleStore()
        store.ingest_trade(price, volume, timestamp)  # called per tick

        candles_5m = store.get_candles("5m")
        live_1m    = store.get_live_candle("1m")
    """

    def __init__(self) -> None:
        # Closed (complete) candles per timeframe
        self._closed: Dict[str, Deque[Candle]] = {
            tf: deque(maxlen=CANDLE_HISTORY[tf]) for tf in TIMEFRAMES
        }
        # Currently-forming candle per timeframe
        self._live: Dict[str, Optional[Candle]] = {tf: None for tf in TIMEFRAMES}

        self._lock = asyncio.Lock()

        # VWAP accumulators (reset each trading day)
        self._vwap_cum_tp_vol: float = 0.0
        self._vwap_cum_vol:    float = 0.0

    # ──────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────

    async def ingest_trade(
        self,
        price:     float,
        volume:    float,
        timestamp: datetime,
    ) -> List[str]:
        """
        Process a single trade tick.
        Returns list of timeframes that just had a candle close.
        Used by the engine to trigger indicator recalculation.
        """
        async with self._lock:
            closed_tfs = []
            for tf in TIMEFRAMES:
                closed = self._update_candle(tf, price, volume, timestamp)
                if closed:
                    closed_tfs.append(tf)
            self._update_vwap(price, volume)
            return closed_tfs

    async def ingest_bar(self, bar: dict, timeframe: str = "1m") -> None:
        """
        Ingest a pre-built OHLCV bar (e.g. from REST catch-up or Alpaca bar stream).
        `bar` expects keys: timestamp, open, high, low, close, volume
        """
        async with self._lock:
            candle = Candle(
                timestamp=bar["timestamp"],
                open=bar["open"],
                high=bar["high"],
                low=bar["low"],
                close=bar["close"],
                volume=bar["volume"],
                timeframe=timeframe,
                is_closed=True,
            )
            self._closed[timeframe].append(candle)

    def get_candles(self, timeframe: str, n: Optional[int] = None) -> List[Candle]:
        """Return closed candles for `timeframe`, most recent last."""
        candles = list(self._closed[timeframe])
        if n is not None:
            candles = candles[-n:]
        return candles

    def get_live_candle(self, timeframe: str) -> Optional[Candle]:
        """Return the currently-forming (open) candle for `timeframe`."""
        return self._live.get(timeframe)

    def get_closes(self, timeframe: str, n: Optional[int] = None) -> List[float]:
        return [c.close for c in self.get_candles(timeframe, n)]

    def get_highs(self, timeframe: str, n: Optional[int] = None) -> List[float]:
        return [c.high for c in self.get_candles(timeframe, n)]

    def get_lows(self, timeframe: str, n: Optional[int] = None) -> List[float]:
        return [c.low for c in self.get_candles(timeframe, n)]

    def get_volumes(self, timeframe: str, n: Optional[int] = None) -> List[float]:
        return [c.volume for c in self.get_candles(timeframe, n)]

    def get_vwap(self) -> Optional[float]:
        if self._vwap_cum_vol == 0:
            return None
        return self._vwap_cum_tp_vol / self._vwap_cum_vol

    def reset_vwap(self) -> None:
        """Call at market open each day to reset the intraday VWAP."""
        self._vwap_cum_tp_vol = 0.0
        self._vwap_cum_vol    = 0.0

    def candle_count(self, timeframe: str) -> int:
        return len(self._closed[timeframe])

    # ──────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────

    def _update_candle(
        self,
        tf:        str,
        price:     float,
        volume:    float,
        timestamp: datetime,
    ) -> bool:
        """
        Update the live candle for `tf`.
        Returns True if a candle just CLOSED (i.e. a new period started).
        """
        period_secs = TIMEFRAME_SECONDS[tf]

        # Bucket this tick into its candle period
        ts_epoch      = int(timestamp.timestamp())
        candle_start  = datetime.fromtimestamp(
            (ts_epoch // period_secs) * period_secs, tz=timezone.utc
        )

        live = self._live[tf]

        if live is None:
            # First tick ever for this timeframe
            self._live[tf] = self._new_candle(tf, candle_start, price, volume)
            return False

        if live.timestamp == candle_start:
            # Same period — update live candle
            live.high   = max(live.high, price)
            live.low    = min(live.low, price)
            live.close  = price
            live.volume += volume
            return False

        # New period — close the live candle and start a new one
        live.is_closed = True
        self._closed[tf].append(live)
        self._live[tf] = self._new_candle(tf, candle_start, price, volume)
        return True

    @staticmethod
    def _new_candle(tf: str, timestamp: datetime, price: float, volume: float) -> Candle:
        return Candle(
            timestamp=timestamp,
            open=price,
            high=price,
            low=price,
            close=price,
            volume=volume,
            timeframe=tf,
            is_closed=False,
        )

    def _update_vwap(self, price: float, volume: float) -> None:
        """Accumulate typical price × volume for VWAP."""
        # Note: we only have one price point per tick, so typical ≈ price
        self._vwap_cum_tp_vol += price * volume
        self._vwap_cum_vol    += volume
