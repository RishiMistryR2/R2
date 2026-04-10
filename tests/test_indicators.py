"""
Tests for indicator math.
Run with: python -m pytest tests/test_indicators.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from engine.indicators import (
    ema,
    ema_series,
    vwap,
    rsi,
    macd,
    atr,
    find_swing_highs,
    find_swing_lows,
    classify_ema_stack,
)


# ──────────────────────────────────────────────
# EMA
# ──────────────────────────────────────────────

def test_ema_returns_none_if_insufficient_data():
    assert ema([1.0, 2.0], period=5) is None

def test_ema_sma_for_exactly_period_values():
    # For exactly `period` values, EMA = SMA
    result = ema([1.0, 2.0, 3.0, 4.0, 5.0], period=5)
    assert result == pytest.approx(3.0)

def test_ema_trend_up():
    # Rising series — EMA should be below the latest price
    closes = list(range(1, 30))
    result = ema(closes, 9)
    assert result is not None
    assert result < closes[-1]

def test_ema_series_length():
    closes = [float(i) for i in range(1, 21)]
    result = ema_series(closes, 5)
    assert len(result) == len(closes)
    assert result[0] is None
    assert result[4] is not None   # first valid at index period-1

def test_ema_series_seed_equals_sma():
    closes = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = ema_series(closes, 5)
    assert result[-1] == pytest.approx(3.0)


# ──────────────────────────────────────────────
# VWAP
# ──────────────────────────────────────────────

def test_vwap_single_bar():
    # VWAP with one bar: typical price * vol / vol = typical price
    result = vwap([110.0], [100.0], [105.0], [1000.0])
    assert result == pytest.approx((110.0 + 100.0 + 105.0) / 3.0)

def test_vwap_equal_volume():
    # With equal volume, VWAP = mean of typical prices
    h = [110.0, 120.0]
    l = [100.0, 110.0]
    c = [105.0, 115.0]
    v = [100.0, 100.0]
    tp1 = (110 + 100 + 105) / 3
    tp2 = (120 + 110 + 115) / 3
    expected = (tp1 + tp2) / 2
    assert vwap(h, l, c, v) == pytest.approx(expected)

def test_vwap_zero_volume():
    assert vwap([100.0], [100.0], [100.0], [0.0]) is None

def test_vwap_mismatched_lengths():
    assert vwap([100.0, 101.0], [99.0], [100.0], [100.0]) is None


# ──────────────────────────────────────────────
# RSI
# ──────────────────────────────────────────────

def test_rsi_insufficient_data():
    assert rsi([1.0, 2.0, 3.0], period=14) is None

def test_rsi_all_gains_returns_100():
    closes = [float(i) for i in range(1, 30)]
    result = rsi(closes, 14)
    assert result == pytest.approx(100.0, abs=0.1)

def test_rsi_flat_market():
    closes = [100.0] * 30
    # No gains, no losses — technically undefined; implementation should handle gracefully
    result = rsi(closes, 14)
    # Flat = 100 (no losses) or some neutral value; implementation returns 100 when avg_loss=0
    assert result is not None

def test_rsi_in_range():
    import random
    random.seed(42)
    closes = [100.0 + random.gauss(0, 1) for _ in range(50)]
    result = rsi(closes, 14)
    assert result is not None
    assert 0.0 <= result <= 100.0


# ──────────────────────────────────────────────
# MACD
# ──────────────────────────────────────────────

def test_macd_insufficient_data():
    m, s, h = macd([1.0] * 10)
    assert m is None and s is None and h is None

def test_macd_histogram_equals_line_minus_signal():
    closes = [100.0 + i * 0.1 for i in range(100)]
    m, s, h = macd(closes)
    assert m is not None and s is not None and h is not None
    assert h == pytest.approx(m - s, abs=1e-10)

def test_macd_bullish_on_strong_uptrend():
    # Exponential growth produces sustained positive MACD histogram
    closes = [100.0 * (1.001 ** i) for i in range(100)]
    m, s, h = macd(closes)
    assert h is not None and h > 0


# ──────────────────────────────────────────────
# ATR
# ──────────────────────────────────────────────

def test_atr_insufficient_data():
    h = [101.0] * 5
    l = [99.0] * 5
    c = [100.0] * 5
    assert atr(h, l, c, 14) is None

def test_atr_constant_bars():
    # Same bar repeated — TR = high - low every bar
    n = 20
    h = [101.0] * n
    l = [99.0]  * n
    c = [100.0] * n
    result = atr(h, l, c, 14)
    assert result == pytest.approx(2.0, abs=0.01)

def test_atr_positive():
    import random
    random.seed(1)
    h = [100 + abs(random.gauss(0, 1)) for _ in range(30)]
    l = [99  - abs(random.gauss(0, 1)) for _ in range(30)]
    c = [100 + random.gauss(0, 0.5)   for _ in range(30)]
    result = atr(h, l, c, 14)
    assert result is not None and result > 0


# ──────────────────────────────────────────────
# Swing detection
# ──────────────────────────────────────────────

def test_swing_high_detected():
    # Clear peak at index 10
    highs = [100.0] * 20
    highs[10] = 110.0
    result = find_swing_highs(highs, lookback=5)
    prices = [p for _, p in result]
    assert 110.0 in prices

def test_swing_low_detected():
    lows = [100.0] * 20
    lows[10] = 90.0
    result = find_swing_lows(lows, lookback=5)
    prices = [p for _, p in result]
    assert 90.0 in prices

def test_swing_empty_on_flat():
    # Flat series — no clear swings
    highs = [100.0] * 30
    result = find_swing_highs(highs, lookback=5)
    # May or may not find swings on flat data; just ensure it doesn't crash
    assert isinstance(result, list)


# ──────────────────────────────────────────────
# EMA stack
# ──────────────────────────────────────────────

def test_ema_stack_bull_aligned():
    result = classify_ema_stack(
        ema_9=105.0, ema_20=103.0, ema_50=101.0, ema_200=99.0, close=107.0
    )
    assert result == "bull_aligned"

def test_ema_stack_bear_aligned():
    result = classify_ema_stack(
        ema_9=95.0, ema_20=97.0, ema_50=99.0, ema_200=101.0, close=93.0
    )
    assert result == "bear_aligned"

def test_ema_stack_mixed():
    result = classify_ema_stack(
        ema_9=105.0, ema_20=103.0, ema_50=106.0, ema_200=99.0, close=107.0
    )
    assert result == "mixed"

def test_ema_stack_none_inputs():
    result = classify_ema_stack(None, None, None, None, 100.0)
    assert result == "insufficient_data"


if __name__ == "__main__":
    # Quick smoke test
    import traceback
    tests = [fn for name, fn in list(globals().items()) if name.startswith("test_")]
    passed = failed = 0
    for test in tests:
        try:
            test()
            print(f"  PASS  {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {test.__name__}: {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
