"""
Indicator calculations.

All functions are pure:  input arrays → output float (or list).
No state, no side effects. Feed in a list of closes/highs/lows/volumes
and get a number back. This makes every indicator trivially testable.

We deliberately avoid pandas here — plain Python lists keep the engine
dependency-light and fast for small rolling windows.
"""

from __future__ import annotations
from typing import List, Optional, Tuple


# ──────────────────────────────────────────────
# EMA
# ──────────────────────────────────────────────

def ema(values: List[float], period: int) -> Optional[float]:
    """
    Exponential Moving Average — returns the most recent value.
    Uses the standard multiplier k = 2 / (period + 1).
    Returns None if not enough data.
    """
    if len(values) < period:
        return None

    k = 2.0 / (period + 1)
    # Seed with SMA of first `period` values
    current = sum(values[:period]) / period
    for price in values[period:]:
        current = price * k + current * (1 - k)
    return current


def ema_series(values: List[float], period: int) -> List[Optional[float]]:
    """
    Full EMA series — same length as input, None until we have enough data.
    Used when we need historical EMA values (e.g. for MACD).
    """
    if len(values) < period:
        return [None] * len(values)

    k = 2.0 / (period + 1)
    result: List[Optional[float]] = [None] * (period - 1)

    # Seed
    seed = sum(values[:period]) / period
    result.append(seed)
    current = seed

    for price in values[period:]:
        current = price * k + current * (1 - k)
        result.append(current)

    return result


# ──────────────────────────────────────────────
# VWAP
# ──────────────────────────────────────────────

def vwap(
    highs:   List[float],
    lows:    List[float],
    closes:  List[float],
    volumes: List[float],
) -> Optional[float]:
    """
    Volume-Weighted Average Price.
    Typical price = (H + L + C) / 3
    VWAP = cumulative(typical * volume) / cumulative(volume)
    """
    n = len(highs)
    if not n or len(lows) != n or len(closes) != n or len(volumes) != n:
        return None
    cum_tp_vol = 0.0
    cum_vol    = 0.0
    for h, l, c, v in zip(highs, lows, closes, volumes):
        tp = (h + l + c) / 3.0
        cum_tp_vol += tp * v
        cum_vol    += v
    return cum_tp_vol / cum_vol if cum_vol > 0 else None


# ──────────────────────────────────────────────
# RSI
# ──────────────────────────────────────────────

def rsi(closes: List[float], period: int = 14) -> Optional[float]:
    """
    Relative Strength Index (Wilder smoothing).
    Returns the most recent RSI value, or None if insufficient data.
    """
    if len(closes) < period + 1:
        return None

    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains  = [d if d > 0 else 0.0 for d in deltas]
    losses = [-d if d < 0 else 0.0 for d in deltas]

    # Initial averages (simple mean for seed)
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    # Wilder smoothing for the rest
    for g, l in zip(gains[period:], losses[period:]):
        avg_gain = (avg_gain * (period - 1) + g) / period
        avg_loss = (avg_loss * (period - 1) + l) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


# ──────────────────────────────────────────────
# MACD
# ──────────────────────────────────────────────

def macd(
    closes: List[float],
    fast:   int = 12,
    slow:   int = 26,
    signal: int = 9,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    MACD line, signal line, histogram.
    Returns (macd_line, signal_line, histogram) — all None if not enough data.
    """
    min_len = slow + signal
    if len(closes) < min_len:
        return None, None, None

    fast_series   = ema_series(closes, fast)
    slow_series   = ema_series(closes, slow)

    # MACD line = fast EMA - slow EMA (only where both are valid)
    macd_line_series: List[Optional[float]] = []
    for f, s in zip(fast_series, slow_series):
        if f is not None and s is not None:
            macd_line_series.append(f - s)
        else:
            macd_line_series.append(None)

    # Strip leading Nones for signal EMA
    valid_macd = [v for v in macd_line_series if v is not None]
    if len(valid_macd) < signal:
        return None, None, None

    sig = ema(valid_macd, signal)
    mac = valid_macd[-1]
    hist = (mac - sig) if sig is not None else None
    return mac, sig, hist


# ──────────────────────────────────────────────
# ATR
# ──────────────────────────────────────────────

def atr(
    highs:  List[float],
    lows:   List[float],
    closes: List[float],
    period: int = 14,
) -> Optional[float]:
    """
    Average True Range (Wilder smoothing).
    True Range = max(H-L, |H-prev_C|, |L-prev_C|)
    """
    if len(closes) < period + 1:
        return None

    trs = []
    for i in range(1, len(closes)):
        hl  = highs[i] - lows[i]
        hpc = abs(highs[i] - closes[i - 1])
        lpc = abs(lows[i]  - closes[i - 1])
        trs.append(max(hl, hpc, lpc))

    if len(trs) < period:
        return None

    # Seed with simple mean
    atr_val = sum(trs[:period]) / period
    for tr in trs[period:]:
        atr_val = (atr_val * (period - 1) + tr) / period
    return atr_val


# ──────────────────────────────────────────────
# Swing highs / lows
# ──────────────────────────────────────────────

def find_swing_highs(highs: List[float], lookback: int = 10) -> List[Tuple[int, float]]:
    """
    Returns (index, price) pairs for local swing highs.
    A swing high is a bar where high[i] > all highs within `lookback` bars on each side.
    """
    result = []
    n = len(highs)
    for i in range(lookback, n - lookback):
        window = highs[i - lookback: i] + highs[i + 1: i + lookback + 1]
        if highs[i] == max(highs[i - lookback: i + lookback + 1]):
            result.append((i, highs[i]))
    return result


def find_swing_lows(lows: List[float], lookback: int = 10) -> List[Tuple[int, float]]:
    """
    Returns (index, price) pairs for local swing lows.
    """
    result = []
    n = len(lows)
    for i in range(lookback, n - lookback):
        if lows[i] == min(lows[i - lookback: i + lookback + 1]):
            result.append((i, lows[i]))
    return result


# ──────────────────────────────────────────────
# EMA alignment helper
# ──────────────────────────────────────────────

def classify_ema_stack(
    ema_9:   Optional[float],
    ema_20:  Optional[float],
    ema_50:  Optional[float],
    ema_200: Optional[float],
    close:   float,
) -> str:
    """
    Bull aligned:  close > 9 > 20 > 50 > 200
    Bear aligned:  close < 9 < 20 < 50 < 200
    Mixed otherwise.
    """
    vals = [v for v in [ema_9, ema_20, ema_50, ema_200] if v is not None]
    if not vals:
        return "insufficient_data"

    if all(vals[i] > vals[i + 1] for i in range(len(vals) - 1)) and close > vals[0]:
        return "bull_aligned"
    if all(vals[i] < vals[i + 1] for i in range(len(vals) - 1)) and close < vals[0]:
        return "bear_aligned"
    return "mixed"
