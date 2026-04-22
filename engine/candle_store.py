"""
engine/candle_store.py — rolling per-timeframe candle history.

Responsibilities:
  - Maintain a deque of closed candles per timeframe (capped at CANDLE_HISTORY[tf])
  - Track the currently-forming live candle per timeframe
  - Aggregate live trades into candles across all timeframes simultaneously
  - Ingest pre-built bars from REST seeding (with dedup)
  - Accumulate intraday VWAP (reset at market open)
  - Tag candles with session_type ("regular" | "premarket" | "afterhours")
  - Provide a readiness snapshot (candle counts by timeframe)

Thread safety: asyncio.Lock protects all writes.
Reads snapshot state before iterating.
"""

from __future__ import annotations

import asyncio
from collections import deque
from datetime import datetime, time as dt_time, timezone, timedelta
from typing import Dict, Deque, List, Optional, Tuple

from engine.models import Candle, Readiness
from engine.readiness import compute_readiness
from config.settings import (
    TIMEFRAMES,
    CANDLE_HISTORY,
    TIMEFRAME_SECONDS,
    PREMARKET_WINDOW_FRACTION,
    PREMARKET_CONFIDENCE_PENALTY,
)

# Market session boundaries in ET (approximate UTC offset used as fallback)
_MARKET_OPEN_TIME  = dt_time(9, 30)
_MARKET_CLOSE_TIME = dt_time(16, 0)
_PRE_START_TIME    = dt_time(4, 0)
_AH_END_TIME       = dt_time(20, 0)

# ET offset (EDT: UTC-4, EST: UTC-5). We use a simple approximation here;
# session_state.py uses pytz for precise phase computation.
_ET_UTC_OFFSET_H = 4   # approximate EDT


class CandleStore:
    """
    Maintains rolling OHLCV history for every configured timeframe.

    Public read API (no lock required — reads snapshot):
        candles  = store.get_candles("5m")
        closes   = store.get_closes("15m", n=50)
        live     = store.get_live_candle("1m")
        ready, note = store.get_readiness()
        counts   = store.candle_counts()

    Write API (acquires lock — call from async context):
        await store.ingest_trade(price, volume, timestamp)
        await store.ingest_bar(bar_dict, timeframe)
        store.reset_vwap()
    """

    def __init__(self) -> None:
        self._closed: Dict[str, Deque[Candle]] = {
            tf: deque(maxlen=CANDLE_HISTORY[tf]) for tf in TIMEFRAMES
        }
        self._live: Dict[str, Optional[Candle]] = {tf: None for tf in TIMEFRAMES}

        # Dedup: track latest ingested timestamp per TF to prevent duplicate bars
        self._latest_ts: Dict[str, Optional[datetime]] = {tf: None for tf in TIMEFRAMES}

        self._lock = asyncio.Lock()

        # Intraday VWAP accumulators
        self._vwap_cum_tp_vol: float = 0.0
        self._vwap_cum_vol:    float = 0.0

    # ──────────────────────────────────────────
    # Write API
    # ──────────────────────────────────────────

    async def ingest_trade(
        self,
        price:     float,
        volume:    float,
        timestamp: datetime,
        session_type: str = "regular",
    ) -> List[str]:
        """
        Process a single trade tick.
        Returns list of timeframes where a candle just closed.
        """
        async with self._lock:
            closed_tfs: List[str] = []
            for tf in TIMEFRAMES:
                did_close = self._update_candle(tf, price, volume, timestamp, session_type)
                if did_close:
                    closed_tfs.append(tf)
            self._update_vwap(price, volume, timestamp)
            return closed_tfs

    async def ingest_bar(
        self,
        bar:          dict,
        timeframe:    str,
        session_type: str = "regular",
    ) -> bool:
        """
        Ingest a pre-built OHLCV bar (REST seed or bar-stream).

        Expected keys: timestamp, open, high, low, close, volume
        timestamp must be a datetime object (UTC-aware).

        Returns True if the bar was accepted, False if it was a duplicate.
        """
        ts = bar["timestamp"]
        if not isinstance(ts, datetime):
            raise ValueError(f"bar['timestamp'] must be datetime, got {type(ts)}")

        async with self._lock:
            # Dedup: skip if we already have this timestamp for this TF
            latest = self._latest_ts.get(timeframe)
            if latest is not None and ts <= latest:
                return False

            candle = Candle(
                timestamp=ts,
                open=float(bar["open"]),
                high=float(bar["high"]),
                low=float(bar["low"]),
                close=float(bar["close"]),
                volume=float(bar["volume"]),
                timeframe=timeframe,
                is_closed=True,
                session_type=session_type,
            )
            self._closed[timeframe].append(candle)
            self._latest_ts[timeframe] = ts
            return True

    def reset_vwap(self) -> None:
        """Reset intraday VWAP accumulators. Call at 9:30 ET market open."""
        self._vwap_cum_tp_vol = 0.0
        self._vwap_cum_vol    = 0.0

    # ──────────────────────────────────────────
    # Read API
    # ──────────────────────────────────────────

    def get_candles(self, timeframe: str, n: Optional[int] = None) -> List[Candle]:
        """Return closed candles, oldest first (most recent last)."""
        candles = list(self._closed[timeframe])
        if n is not None:
            candles = candles[-n:]
        return candles

    def get_live_candle(self, timeframe: str) -> Optional[Candle]:
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

    def candle_count(self, timeframe: str) -> int:
        return len(self._closed[timeframe])

    def candle_counts(self) -> Dict[str, int]:
        """Return {timeframe: count} for all timeframes."""
        return {tf: len(self._closed[tf]) for tf in TIMEFRAMES}

    def get_readiness(self) -> Tuple[Readiness, str]:
        """Compute current readiness from candle counts."""
        return compute_readiness(self.candle_counts())

    def premarket_fraction(self, timeframe: str, lookback: int = 50) -> float:
        """
        Fraction of the last `lookback` candles that are premarket.
        Used by the indicator engine to apply confidence penalties.
        """
        candles = self.get_candles(timeframe, n=lookback)
        if not candles:
            return 0.0
        pre = sum(1 for c in candles if c.session_type == "premarket")
        return pre / len(candles)

    def get_latest_close(self, timeframe: str) -> Optional[float]:
        """Latest closed candle's close price, or None."""
        candles = self.get_candles(timeframe, n=1)
        return candles[-1].close if candles else None

    # ──────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────

    def _update_candle(
        self,
        tf:           str,
        price:        float,
        volume:       float,
        timestamp:    datetime,
        session_type: str,
    ) -> bool:
        """
        Update the live candle for `tf`.
        Returns True if the live candle just closed (new period started).
        """
        period_secs  = TIMEFRAME_SECONDS[tf]
        ts_epoch     = int(timestamp.timestamp())
        candle_start = datetime.fromtimestamp(
            (ts_epoch // period_secs) * period_secs,
            tz=timezone.utc,
        )

        live = self._live[tf]

        if live is None:
            self._live[tf] = self._make_candle(tf, candle_start, price, volume, session_type)
            return False

        if live.timestamp == candle_start:
            live.high    = max(live.high, price)
            live.low     = min(live.low, price)
            live.close   = price
            live.volume += volume
            return False

        # Period boundary — close current live, start new one
        live.is_closed = True
        self._closed[tf].append(live)
        self._latest_ts[tf] = live.timestamp
        self._live[tf] = self._make_candle(tf, candle_start, price, volume, session_type)
        return True

    @staticmethod
    def _make_candle(
        tf:           str,
        timestamp:    datetime,
        price:        float,
        volume:       float,
        session_type: str = "regular",
    ) -> Candle:
        return Candle(
            timestamp=timestamp,
            open=price,
            high=price,
            low=price,
            close=price,
            volume=volume,
            timeframe=tf,
            is_closed=False,
            session_type=session_type,
        )

    def _update_vwap(self, price: float, volume: float, timestamp: datetime) -> None:
        """Accumulate VWAP. Only counts regular-session ticks."""
        et_hour = (timestamp - timedelta(hours=_ET_UTC_OFFSET_H)).hour
        # Only accumulate during regular session (9:30–16:00 ET approx)
        if 9 <= et_hour < 16:
            self._vwap_cum_tp_vol += price * volume
            self._vwap_cum_vol    += volume


# ──────────────────────────────────────────────
# Session-type classifier (standalone utility)
# ──────────────────────────────────────────────

def classify_session_type(timestamp: datetime) -> str:
    """
    Given a UTC timestamp, return the session type string.
    Uses a simple hour-based approximation (EDT = UTC-4).
    For production precision, use session_state.py with pytz.
    """
    et_dt   = timestamp - timedelta(hours=_ET_UTC_OFFSET_H)
    et_time = et_dt.time()

    if _PRE_START_TIME <= et_time < _MARKET_OPEN_TIME:
        return "premarket"
    if _MARKET_OPEN_TIME <= et_time < _MARKET_CLOSE_TIME:
        return "regular"
    if _MARKET_CLOSE_TIME <= et_time < _AH_END_TIME:
        return "afterhours"
    return "overnight"