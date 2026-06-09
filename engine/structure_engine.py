"""
engine/structure_engine.py — Market Structure Engine v2.0A

Analyses OHLCV bar data to produce a StructureState describing what the
market is actually doing, not just what the indicators read.

Outputs are designed to give the AI commentary layer five key answers:
  1. What structure exists   (trend direction + HH/HL/LH/LL sequence)
  2. What recently changed   (behavioral events: reclaim, sweep, etc.)
  3. What buyers are doing   (defending which level, building what structure)
  4. What sellers are doing  (rejecting which level, distributing where)
  5. What confirmation is still missing (next structural requirement)

Architecture:
  MarketState (with .bars)
    → detect_swing_points()
    → classify_trend_structure()
    → classify_market_state()
    → detect_key_levels()
    → detect_behavioral_events()
    → build_narrative()

Public API:
  analyze(market_state, primary_tf="5m", confirm_tf="15m") → StructureState
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("engine.structure")


# ══════════════════════════════════════════════════════════════
# Enums
# ══════════════════════════════════════════════════════════════

class MarketStateLabel(Enum):
    TREND       = "trend"
    PULLBACK    = "pullback"
    COMPRESSION = "compression"
    EXPANSION   = "expansion"
    RANGE       = "range"
    BREAKDOWN   = "breakdown"
    BREAKOUT    = "breakout"


class BehavioralEventType(Enum):
    VWAP_RECLAIM              = "vwap_reclaim"
    VWAP_REJECTION            = "vwap_rejection"
    FAILED_BREAKDOWN          = "failed_breakdown"
    FAILED_BREAKOUT           = "failed_breakout"
    LIQUIDITY_SWEEP_LOW       = "liquidity_sweep_low"
    LIQUIDITY_SWEEP_HIGH      = "liquidity_sweep_high"
    TREND_CONTINUATION        = "trend_continuation"
    TREND_EXHAUSTION          = "trend_exhaustion"
    HIGHER_LOW_FORMED         = "higher_low_formed"
    LOWER_HIGH_FORMED         = "lower_high_formed"
    SESSION_HIGH_RECLAIM      = "session_high_reclaim"
    PRIOR_DAY_HIGH_RECLAIM    = "prior_day_high_reclaim"
    PRIOR_DAY_HIGH_REJECTION  = "prior_day_high_rejection"


# ══════════════════════════════════════════════════════════════
# Data types
# ══════════════════════════════════════════════════════════════

@dataclass
class SwingPoint:
    price:      float
    bar_index:  int
    timestamp:  Optional[datetime]
    swing_type: str    # "high" | "low"


@dataclass
class TrendStructure:
    direction:   str            # "uptrend" | "downtrend" | "ranging"
    sequence:    List[str]      # e.g. ["HH", "HL", "HH", "HL"] oldest→newest
    swing_highs: List[SwingPoint]
    swing_lows:  List[SwingPoint]
    last_hh:     Optional[float] = None
    last_hl:     Optional[float] = None
    last_lh:     Optional[float] = None
    last_ll:     Optional[float] = None
    trend_intact: bool = True

    def sequence_str(self) -> str:
        return " ".join(self.sequence[-6:]) if self.sequence else "—"

    def last_swing_high_price(self) -> Optional[float]:
        return self.swing_highs[-1].price if self.swing_highs else None

    def last_swing_low_price(self) -> Optional[float]:
        return self.swing_lows[-1].price if self.swing_lows else None


@dataclass
class MarketStateResult:
    state:      MarketStateLabel
    confidence: float
    detail:     str


@dataclass
class KeyLevels:
    session_high:       Optional[float] = None
    session_low:        Optional[float] = None
    prior_day_high:     Optional[float] = None
    prior_day_low:      Optional[float] = None
    range_high:         Optional[float] = None
    range_low:          Optional[float] = None
    vwap:               Optional[float] = None
    nearest_support:    Optional[float] = None
    nearest_resistance: Optional[float] = None

    def to_dict(self) -> dict:
        return {k: round(v, 2) for k, v in self.__dict__.items() if v is not None}

    def all_levels(self) -> List[Tuple[str, float]]:
        pairs = []
        for k, v in self.__dict__.items():
            if v is not None:
                pairs.append((k.replace("_", " "), v))
        return sorted(pairs, key=lambda x: x[1])

    def nearest_to(self, price: float) -> Tuple[Optional[str], Optional[float]]:
        levels = [(k, v) for k, v in self.__dict__.items() if v is not None]
        if not levels:
            return None, None
        closest = min(levels, key=lambda kv: abs(kv[1] - price))
        return closest[0].replace("_", " "), closest[1]


@dataclass
class BehavioralEvent:
    event_type: BehavioralEventType
    price:      float
    bars_ago:   int
    detail:     str


@dataclass
class StructureState:
    symbol:      str
    timestamp:   datetime
    primary_tf:  str
    confirm_tf:  str

    trend:              TrendStructure
    market_state:       MarketStateResult
    key_levels:         KeyLevels
    behavioral_events:  List[BehavioralEvent] = field(default_factory=list)

    structure_narrative:    str = ""
    buyer_context:          str = ""
    seller_context:         str = ""
    missing_confirmation:   str = ""

    def to_dict(self) -> dict:
        events_str = [
            f"{e.event_type.value} ({e.bars_ago}b ago): {e.detail}"
            for e in self.behavioral_events[:6]
        ]
        return {
            "primary_tf":           self.primary_tf,
            "direction":            self.trend.direction,
            "sequence":             self.trend.sequence_str(),
            "trend_intact":         self.trend.trend_intact,
            "last_hh":              self.trend.last_hh,
            "last_hl":              self.trend.last_hl,
            "last_lh":              self.trend.last_lh,
            "last_ll":              self.trend.last_ll,
            "market_state":         self.market_state.state.value,
            "market_state_detail":  self.market_state.detail,
            "key_levels":           self.key_levels.to_dict(),
            "recent_events":        events_str,
            "structure_narrative":  self.structure_narrative,
            "buyer_context":        self.buyer_context,
            "seller_context":       self.seller_context,
            "missing_confirmation": self.missing_confirmation,
        }


# ══════════════════════════════════════════════════════════════
# Public entry point
# ══════════════════════════════════════════════════════════════

def analyze(
    market_state,
    primary_tf:  str = "5m",
    confirm_tf:  str = "15m",
) -> StructureState:
    """
    Analyse market structure from MarketState.
    Uses market_state.bars[tf] when available.
    Falls back to TrendState swing levels for demo/test mode.
    Never raises.
    """
    try:
        return _analyze_internal(market_state, primary_tf, confirm_tf)
    except Exception as e:
        logger.error(f"[structure] Analysis failed: {e}", exc_info=True)
        return _empty_state(
            getattr(market_state, "symbol", "SPY"),
            getattr(market_state, "timestamp", datetime.now(tz=timezone.utc)),
            primary_tf, confirm_tf,
        )


# ══════════════════════════════════════════════════════════════
# Internal pipeline
# ══════════════════════════════════════════════════════════════

def _analyze_internal(ms, primary_tf: str, confirm_tf: str) -> StructureState:
    symbol = getattr(ms, "symbol", "SPY")
    now    = getattr(ms, "timestamp", datetime.now(tz=timezone.utc))

    all_bars: dict = getattr(ms, "bars", {}) or {}
    bars_p   = all_bars.get(primary_tf, [])
    bars_1d  = all_bars.get("1d", [])

    ind_p    = (getattr(ms, "indicators", {}) or {}).get(primary_tf)
    vwap     = ind_p.vwap if ind_p else None
    atr      = ind_p.atr  if ind_p and ind_p.atr else 1.5
    current  = bars_p[-1]["close"] if bars_p else (ind_p.close if ind_p else 0.0)

    if len(bars_p) >= 10:
        swing_highs, swing_lows = detect_swing_points(bars_p)
    else:
        swing_highs, swing_lows = _swings_from_trend_state(ms, primary_tf)

    trend      = classify_trend_structure(swing_highs, swing_lows)
    key_levels = detect_key_levels(bars_1d, bars_p, vwap, now)

    if swing_highs:
        resistances = [s.price for s in swing_highs if s.price > current]
        if resistances:
            key_levels.nearest_resistance = min(resistances)
    if swing_lows:
        supports = [s.price for s in swing_lows if s.price < current]
        if supports:
            key_levels.nearest_support = max(supports)

    mkt_state = classify_market_state(bars_p, trend, atr, current, vwap, key_levels)
    events    = detect_behavioral_events(bars_p, key_levels, vwap, atr)

    narr   = _build_structure_narrative(symbol, trend, mkt_state, key_levels, events, current)
    buyer  = _build_buyer_context(trend, mkt_state, key_levels, current, vwap, events)
    seller = _build_seller_context(trend, mkt_state, key_levels, current, vwap, events)
    miss   = _build_missing_confirmation(trend, mkt_state, key_levels, current, vwap, events)

    state = StructureState(
        symbol=symbol, timestamp=now,
        primary_tf=primary_tf, confirm_tf=confirm_tf,
        trend=trend, market_state=mkt_state,
        key_levels=key_levels, behavioral_events=events,
        structure_narrative=narr, buyer_context=buyer,
        seller_context=seller, missing_confirmation=miss,
    )

    logger.debug(
        f"[structure] {symbol} {primary_tf}"
        f"  dir={trend.direction}"
        f"  state={mkt_state.state.value}"
        f"  seq={trend.sequence_str()}"
        f"  events={len(events)}"
    )
    return state


# ══════════════════════════════════════════════════════════════
# Swing detection
# ══════════════════════════════════════════════════════════════

def detect_swing_points(
    bars: List[dict],
    left:  int = 3,
    right: int = 2,
) -> Tuple[List[SwingPoint], List[SwingPoint]]:
    """
    Detect pivot swing highs and lows from a list of OHLCV bar dicts.

    A swing high: bar[i].high is strictly the highest of *left* bars before
    it AND at least as high as *right* bars after it (strict on at least one).
    Returns (swing_highs, swing_lows) sorted by bar_index ascending.
    """
    n = len(bars)
    highs: List[SwingPoint] = []
    lows:  List[SwingPoint] = []

    for i in range(left, n - right):
        h  = bars[i]["high"]
        lo = bars[i]["low"]
        ts = bars[i].get("timestamp")

        left_ok  = all(h > bars[j]["high"] for j in range(i - left, i))
        right_ok = all(h >= bars[j]["high"] for j in range(i + 1, i + right + 1))
        right_strict = any(h > bars[j]["high"] for j in range(i + 1, i + right + 1))
        if left_ok and right_ok and right_strict:
            highs.append(SwingPoint(h, i, ts, "high"))

        left_ok_l  = all(lo < bars[j]["low"] for j in range(i - left, i))
        right_ok_l = all(lo <= bars[j]["low"] for j in range(i + 1, i + right + 1))
        right_strict_l = any(lo < bars[j]["low"] for j in range(i + 1, i + right + 1))
        if left_ok_l and right_ok_l and right_strict_l:
            lows.append(SwingPoint(lo, i, ts, "low"))

    return highs, lows


# ══════════════════════════════════════════════════════════════
# Trend structure classification
# ══════════════════════════════════════════════════════════════

def classify_trend_structure(
    swing_highs: List[SwingPoint],
    swing_lows:  List[SwingPoint],
) -> TrendStructure:
    """Build the HH/HL/LH/LL label sequence and determine trend direction."""
    h_labels: List[str] = []
    for i in range(1, len(swing_highs)):
        diff = swing_highs[i].price - swing_highs[i - 1].price
        if diff > 0:
            h_labels.append("HH")
        elif diff < 0:
            h_labels.append("LH")

    l_labels: List[str] = []
    for i in range(1, len(swing_lows)):
        diff = swing_lows[i].price - swing_lows[i - 1].price
        if diff > 0:
            l_labels.append("HL")
        elif diff < 0:
            l_labels.append("LL")

    all_labeled: List[Tuple[int, str]] = []
    for i, lbl in enumerate(h_labels):
        if i + 1 < len(swing_highs):
            all_labeled.append((swing_highs[i + 1].bar_index, lbl))
    for i, lbl in enumerate(l_labels):
        if i + 1 < len(swing_lows):
            all_labeled.append((swing_lows[i + 1].bar_index, lbl))
    all_labeled.sort(key=lambda x: x[0])
    sequence = [lbl for _, lbl in all_labeled][-8:]

    recent_h = h_labels[-2:] if h_labels else []
    recent_l = l_labels[-2:] if l_labels else []

    if not recent_h and not recent_l:
        direction = "ranging"
    elif (recent_h and all(h == "HH" for h in recent_h) and
          recent_l and all(l == "HL" for l in recent_l)):
        direction = "uptrend"
    elif (recent_h and all(h == "LH" for h in recent_h) and
          recent_l and all(l == "LL" for l in recent_l)):
        direction = "downtrend"
    else:
        bull = (sum(1 for h in recent_h if h == "HH") +
                sum(1 for l in recent_l if l == "HL"))
        bear = (sum(1 for h in recent_h if h == "LH") +
                sum(1 for l in recent_l if l == "LL"))
        direction = ("uptrend" if bull > bear else
                     "downtrend" if bear > bull else "ranging")

    trend_intact = True
    if direction == "uptrend":
        if h_labels and h_labels[-1] == "LH":
            trend_intact = False
        if l_labels and l_labels[-1] == "LL":
            trend_intact = False
    elif direction == "downtrend":
        if h_labels and h_labels[-1] == "HH":
            trend_intact = False
        if l_labels and l_labels[-1] == "HL":
            trend_intact = False

    def _last_labeled(labels, points, target):
        for i in range(len(labels) - 1, -1, -1):
            if labels[i] == target and (i + 1) < len(points):
                return points[i + 1].price
        return None

    return TrendStructure(
        direction    = direction,
        sequence     = sequence,
        swing_highs  = swing_highs,
        swing_lows   = swing_lows,
        last_hh      = _last_labeled(h_labels, swing_highs, "HH"),
        last_hl      = _last_labeled(l_labels, swing_lows,  "HL"),
        last_lh      = _last_labeled(h_labels, swing_highs, "LH"),
        last_ll      = _last_labeled(l_labels, swing_lows,  "LL"),
        trend_intact = trend_intact,
    )


# ══════════════════════════════════════════════════════════════
# Market state classification
# ══════════════════════════════════════════════════════════════

def classify_market_state(
    bars:       List[dict],
    trend:      TrendStructure,
    atr:        float,
    current:    float,
    vwap:       Optional[float],
    key_levels: KeyLevels,
) -> MarketStateResult:
    if len(bars) < 6:
        return MarketStateResult(MarketStateLabel.RANGE, 0.4, "Insufficient bar data")

    recent     = bars[-15:]
    recent_atr = _rolling_atr(bars[-15:])
    base_atr   = _average_atr(bars[-40:]) if len(bars) >= 40 else recent_atr
    atr_ratio  = recent_atr / base_atr if base_atr > 0 else 1.0

    rh = key_levels.range_high
    rl = key_levels.range_low

    if rh and current > rh + atr * 0.1:
        return MarketStateResult(
            MarketStateLabel.BREAKOUT, 0.72,
            f"Price broke above range high {rh:.2f}"
        )
    if rl and current < rl - atr * 0.1:
        return MarketStateResult(
            MarketStateLabel.BREAKDOWN, 0.72,
            f"Price broke below range low {rl:.2f}"
        )
    if atr_ratio < 0.65 and len(bars) >= 40:
        return MarketStateResult(
            MarketStateLabel.COMPRESSION, 0.62,
            f"ATR contracted {(1 - atr_ratio) * 100:.0f}% below 40-bar average"
        )
    if atr_ratio > 1.45 and len(bars) >= 20:
        return MarketStateResult(
            MarketStateLabel.EXPANSION, 0.62,
            f"ATR expanded {(atr_ratio - 1) * 100:.0f}% above average"
        )

    if trend.direction == "uptrend":
        recent_closes = [b["close"] for b in bars[-6:]]
        if current < recent_closes[0] and trend.last_hl and current > trend.last_hl:
            return MarketStateResult(
                MarketStateLabel.PULLBACK, 0.65,
                f"Pullback within uptrend — above HL support {trend.last_hl:.2f}"
            )
        return MarketStateResult(
            MarketStateLabel.TREND, 0.68, "Uptrend in progress — HH/HL structure intact"
        )

    if trend.direction == "downtrend":
        recent_closes = [b["close"] for b in bars[-6:]]
        if current > recent_closes[0] and trend.last_lh and current < trend.last_lh:
            return MarketStateResult(
                MarketStateLabel.PULLBACK, 0.65,
                f"Bounce within downtrend — below LH resistance {trend.last_lh:.2f}"
            )
        return MarketStateResult(
            MarketStateLabel.TREND, 0.68, "Downtrend in progress — LH/LL structure intact"
        )

    rng_lo = min(b["low"]  for b in recent)
    rng_hi = max(b["high"] for b in recent)
    return MarketStateResult(
        MarketStateLabel.RANGE, 0.55, f"Ranging {rng_lo:.2f} — {rng_hi:.2f}"
    )


# ══════════════════════════════════════════════════════════════
# Key levels
# ══════════════════════════════════════════════════════════════

def detect_key_levels(
    bars_1d:       List[dict],
    bars_intraday: List[dict],
    vwap:          Optional[float],
    now:           datetime,
) -> KeyLevels:
    levels = KeyLevels(vwap=vwap)

    if bars_1d and len(bars_1d) >= 2:
        try:
            today     = now.astimezone(timezone.utc).date()
            completed = [
                b for b in bars_1d
                if b["timestamp"].astimezone(timezone.utc).date() < today
            ]
            if completed:
                prev = completed[-1]
                levels.prior_day_high = prev["high"]
                levels.prior_day_low  = prev["low"]
        except Exception:
            pass

    if bars_intraday:
        session_bars = _current_session_bars(bars_intraday, now)
        if session_bars:
            levels.session_high = max(b["high"] for b in session_bars)
            levels.session_low  = min(b["low"]  for b in session_bars)

    levels.range_high = levels.session_high
    levels.range_low  = levels.session_low
    return levels


def _current_session_bars(bars: List[dict], now: datetime) -> List[dict]:
    try:
        from engine.market_session import _to_et
        from datetime import time as dtime
        result = []
        for b in reversed(bars):
            et = _to_et(b["timestamp"])
            t  = et.time() if hasattr(et, "time") else et
            if hasattr(t, "__ge__") and t >= dtime(9, 30):
                result.append(b)
            elif result:
                break
        return list(reversed(result)) if result else bars[-78:]
    except Exception:
        return bars[-78:]


# ══════════════════════════════════════════════════════════════
# Behavioral events
# ══════════════════════════════════════════════════════════════

def detect_behavioral_events(
    bars:       List[dict],
    key_levels: KeyLevels,
    vwap:       Optional[float],
    atr:        float,
) -> List[BehavioralEvent]:
    if len(bars) < 5:
        return []
    events: List[BehavioralEvent] = []
    recent = bars[-20:]
    if vwap:
        _detect_vwap_events(recent, vwap, atr, events)
    _detect_level_events(recent, key_levels, atr, events)
    _detect_liquidity_sweeps(recent, key_levels, atr, events)
    _detect_structure_events(recent, events)
    return events


def _detect_vwap_events(bars, vwap, atr, events):
    n      = len(bars)
    closes = [b["close"] for b in bars]
    if closes[-1] > vwap:
        for i in range(n - 2, max(n - 12, -1), -1):
            if closes[i] < vwap:
                bars_ago = n - 1 - i
                events.append(BehavioralEvent(
                    BehavioralEventType.VWAP_RECLAIM, vwap, bars_ago,
                    f"Price reclaimed VWAP ({vwap:.2f}) "
                    f"{bars_ago} bar{'s' if bars_ago > 1 else ''} ago",
                ))
                break
    if closes[-1] < vwap:
        for i in range(max(n - 8, 0), n - 1):
            if (bars[i]["high"] >= vwap - atr * 0.2 and
                    bars[i]["high"] <= vwap + atr * 0.5 and
                    bars[i]["close"] < vwap):
                bars_ago = n - 1 - i
                events.append(BehavioralEvent(
                    BehavioralEventType.VWAP_REJECTION, vwap, bars_ago,
                    f"VWAP rejection at {vwap:.2f} — high reached but close failed {bars_ago}b ago",
                ))
                break


def _detect_level_events(bars, key, atr, events):
    n      = len(bars)
    closes = [b["close"] for b in bars]
    for level_name, level_price in [
        ("session low",   key.session_low),
        ("prior day low", key.prior_day_low),
        ("range low",     key.range_low),
    ]:
        if level_price is None:
            continue
        broke_idx = None
        for i in range(n - 10, n - 1):
            if closes[i] < level_price - atr * 0.05:
                broke_idx = i
                break
        if broke_idx is not None and closes[-1] > level_price:
            bars_ago = n - 1 - broke_idx
            if bars_ago <= 8:
                events.append(BehavioralEvent(
                    BehavioralEventType.FAILED_BREAKDOWN, level_price, bars_ago,
                    f"Failed breakdown below {level_name} ({level_price:.2f}) — recovered in {bars_ago}b",
                ))
                break
    for level_name, level_price in [
        ("session high",   key.session_high),
        ("prior day high", key.prior_day_high),
        ("range high",     key.range_high),
    ]:
        if level_price is None:
            continue
        broke_idx = None
        for i in range(n - 10, n - 1):
            if closes[i] > level_price + atr * 0.05:
                broke_idx = i
                break
        if broke_idx is not None and closes[-1] < level_price:
            bars_ago = n - 1 - broke_idx
            if bars_ago <= 8:
                events.append(BehavioralEvent(
                    BehavioralEventType.FAILED_BREAKOUT, level_price, bars_ago,
                    f"Failed breakout above {level_name} ({level_price:.2f}) — rejected in {bars_ago}b",
                ))
                break


def _detect_liquidity_sweeps(bars, key, atr, events):
    n = len(bars)
    if n < 4:
        return
    for level_name, level_price in [
        ("session low",   key.session_low),
        ("prior day low", key.prior_day_low),
    ]:
        if level_price is None:
            continue
        for i in range(max(n - 6, 1), n):
            b = bars[i]
            if (b["low"] < level_price - atr * 0.1 and
                    b["close"] > level_price and
                    bars[i - 1]["close"] > level_price):
                events.append(BehavioralEvent(
                    BehavioralEventType.LIQUIDITY_SWEEP_LOW, level_price, n - 1 - i,
                    f"Liquidity sweep below {level_name} ({level_price:.2f}) — wick only, closed above",
                ))
                break
    for level_name, level_price in [
        ("session high",   key.session_high),
        ("prior day high", key.prior_day_high),
    ]:
        if level_price is None:
            continue
        for i in range(max(n - 6, 1), n):
            b = bars[i]
            if (b["high"] > level_price + atr * 0.1 and
                    b["close"] < level_price and
                    bars[i - 1]["close"] < level_price):
                events.append(BehavioralEvent(
                    BehavioralEventType.LIQUIDITY_SWEEP_HIGH, level_price, n - 1 - i,
                    f"Liquidity sweep above {level_name} ({level_price:.2f}) — wick only, closed below",
                ))
                break


def _detect_structure_events(bars, events):
    n = len(bars)
    if n < 6:
        return
    bodies = [abs(bars[i]["close"] - bars[i]["open"]) for i in range(-3, 0)]
    if bodies[0] > bodies[1] > bodies[2] and bodies[2] < bodies[0] * 0.4:
        events.append(BehavioralEvent(
            BehavioralEventType.TREND_EXHAUSTION,
            bars[-1]["close"], 0,
            "Three consecutive shrinking candle bodies — momentum fading",
        ))


# ══════════════════════════════════════════════════════════════
# Narrative construction
# ══════════════════════════════════════════════════════════════

def _build_structure_narrative(symbol, trend, mkt_state, key, events, current):
    parts = []
    s = mkt_state.state.value

    if s == "trend":
        if trend.direction == "uptrend":
            seq = trend.sequence_str()
            hl  = f" Most recent higher low at {trend.last_hl:.2f}." if trend.last_hl else ""
            parts.append(f"{symbol} is in an uptrend with {seq} structure.{hl}")
        elif trend.direction == "downtrend":
            seq = trend.sequence_str()
            lh  = f" Most recent lower high at {trend.last_lh:.2f}." if trend.last_lh else ""
            parts.append(f"{symbol} is in a downtrend with {seq} structure.{lh}")
    elif s == "pullback":
        if trend.direction == "uptrend":
            hl = f" Holding above higher low support {trend.last_hl:.2f}." if trend.last_hl else ""
            parts.append(f"{symbol} is pulling back within an intact uptrend.{hl}")
        else:
            lh = f" Below lower high resistance {trend.last_lh:.2f}." if trend.last_lh else ""
            parts.append(f"{symbol} is bouncing within a downtrend.{lh}")
    elif s == "compression":
        parts.append(
            f"{symbol} is in a compression phase. ATR has contracted. "
            f"Price is coiling near {current:.2f}."
        )
    elif s == "expansion":
        parts.append(f"{symbol} is in an expansion phase with widening ranges.")
    elif s == "range":
        rh = f"{key.range_high:.2f}" if key.range_high else "?"
        rl = f"{key.range_low:.2f}"  if key.range_low  else "?"
        parts.append(
            f"{symbol} is ranging between {rl} and {rh}. "
            "Recent buying has not yet produced a structural breakout."
        )
    elif s == "breakout":
        rh = f"{key.range_high:.2f}" if key.range_high else "resistance"
        parts.append(
            f"{symbol} broke above {rh}. Expansion phase beginning — structure shifting bullish."
        )
    elif s == "breakdown":
        rl = f"{key.range_low:.2f}" if key.range_low else "support"
        parts.append(
            f"{symbol} broke below {rl}. Sellers in control — structure shifting bearish."
        )
    else:
        parts.append(f"{symbol} structure: {mkt_state.detail}.")

    for ev in events[:2]:
        if ev.event_type in (
            BehavioralEventType.VWAP_RECLAIM,
            BehavioralEventType.FAILED_BREAKDOWN,
            BehavioralEventType.FAILED_BREAKOUT,
            BehavioralEventType.LIQUIDITY_SWEEP_LOW,
        ):
            parts.append(ev.detail.rstrip(".") + ".")
            break

    return " ".join(parts)


def _build_buyer_context(trend, mkt_state, key, current, vwap, events):
    parts = []
    if vwap:
        if current > vwap:
            parts.append(f"Buyers holding above VWAP ({vwap:.2f}).")
        else:
            parts.append(f"Buyers unable to reclaim VWAP ({vwap:.2f}).")
    if trend.direction == "uptrend" and trend.last_hl:
        parts.append(f"Defending higher low support at {trend.last_hl:.2f}.")
    elif trend.last_swing_low_price():
        parts.append(f"Most recent swing low at {trend.last_swing_low_price():.2f}.")
    for ev in events:
        if ev.event_type == BehavioralEventType.VWAP_RECLAIM and ev.bars_ago <= 6:
            parts.append(f"Regained control after VWAP reclaim {ev.bars_ago}b ago.")
            break
        if ev.event_type == BehavioralEventType.FAILED_BREAKDOWN and ev.bars_ago <= 6:
            parts.append("Stepped in after failed breakdown — reclaimed the level.")
            break
    return " ".join(parts) if parts else "Buyer activity not well-defined."


def _build_seller_context(trend, mkt_state, key, current, vwap, events):
    parts = []
    if key.session_high:
        parts.append(f"Sellers defending session high at {key.session_high:.2f}.")
    if key.prior_day_high and current < key.prior_day_high:
        parts.append(f"Prior day high ({key.prior_day_high:.2f}) acting as resistance.")
    if trend.direction == "uptrend" and trend.last_hh:
        parts.append(f"Last HH at {trend.last_hh:.2f} — sellers defending.")
    elif trend.direction == "downtrend" and trend.last_lh:
        parts.append(f"Sellers capping rallies near lower high zone {trend.last_lh:.2f}.")
    for ev in events:
        if ev.event_type == BehavioralEventType.FAILED_BREAKOUT and ev.bars_ago <= 6:
            parts.append(f"Failed breakout {ev.bars_ago}b ago — sellers absorbed buying pressure.")
            break
        if ev.event_type == BehavioralEventType.VWAP_REJECTION and ev.bars_ago <= 6:
            vwap_str = f" at {key.vwap:.2f}" if key.vwap else ""
            parts.append(f"VWAP rejected buyers {ev.bars_ago}b ago{vwap_str}.")
            break
    return " ".join(parts) if parts else "Seller positioning not well-defined."


def _build_missing_confirmation(trend, mkt_state, key, current, vwap, events):
    s = mkt_state.state

    if s == MarketStateLabel.TREND and trend.direction == "uptrend":
        if key.session_high and current < key.session_high:
            return (f"Need break and hold above session high ({key.session_high:.2f}) "
                    "to confirm trend continuation.")
        if trend.last_hh:
            return (f"Need close above prior HH ({trend.last_hh:.2f}) to extend the uptrend.")

    if s == MarketStateLabel.PULLBACK and trend.direction == "uptrend":
        if key.nearest_resistance:
            return (f"Need reclaim of {key.nearest_resistance:.2f} "
                    "to resume uptrend. Must hold the HL or structure breaks.")
        if vwap and current < vwap:
            return (f"Need reclaim of VWAP ({vwap:.2f}) with conviction "
                    "to confirm pullback is over.")

    if s == MarketStateLabel.RANGE:
        if key.range_high:
            return (f"Need break above {key.range_high:.2f} (range high) "
                    "with follow-through to resolve range bullishly.")
        if key.range_low:
            return (f"Need hold above {key.range_low:.2f} (range low) "
                    "to avoid a bearish breakdown.")

    if s == MarketStateLabel.COMPRESSION:
        return ("Need a directional expansion break. "
                "Compression resolves with conviction when it breaks.")

    if s == MarketStateLabel.BREAKOUT:
        lvl = f"{key.range_high:.2f}" if key.range_high else "the breakout level"
        return (f"Need follow-through above {lvl} — "
                "confirmation requires a higher low to form above.")

    if s == MarketStateLabel.BREAKDOWN:
        return ("Need bearish follow-through and a lower high to form "
                "for the breakdown to be sustained.")

    if s == MarketStateLabel.TREND and trend.direction == "downtrend":
        ll_str = f" ({trend.last_ll:.2f})" if trend.last_ll else ""
        return f"Need close below prior LL{ll_str} to extend the downtrend."

    return "No clear structural confirmation requirement identified."


# ══════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════

def _rolling_atr(bars: List[dict], period: int = 14) -> float:
    if len(bars) < 2:
        return 1.0
    trs = [
        max(
            bars[i]["high"] - bars[i]["low"],
            abs(bars[i]["high"] - bars[i - 1]["close"]),
            abs(bars[i]["low"]  - bars[i - 1]["close"]),
        )
        for i in range(1, len(bars))
    ]
    recent = trs[-period:]
    return sum(recent) / len(recent) if recent else 1.0


def _average_atr(bars: List[dict]) -> float:
    """Average True Range across ALL bars in window (used for baseline comparison)."""
    if len(bars) < 2:
        return 1.0
    trs = [
        max(
            bars[i]["high"] - bars[i]["low"],
            abs(bars[i]["high"] - bars[i - 1]["close"]),
            abs(bars[i]["low"]  - bars[i - 1]["close"]),
        )
        for i in range(1, len(bars))
    ]
    return sum(trs) / len(trs)


def _swings_from_trend_state(ms, tf: str):
    trends = getattr(ms, "trends", {}) or {}
    ts     = trends.get(tf)
    if ts is None:
        return [], []
    now   = getattr(ms, "timestamp", datetime.now(tz=timezone.utc))
    highs = [SwingPoint(ts.last_swing_high, 10, now, "high")] if ts.last_swing_high else []
    lows  = [SwingPoint(ts.last_swing_low,   5, now, "low")]  if ts.last_swing_low  else []
    return highs, lows


def _empty_state(symbol, now, primary_tf, confirm_tf):
    return StructureState(
        symbol=symbol, timestamp=now,
        primary_tf=primary_tf, confirm_tf=confirm_tf,
        trend=TrendStructure("ranging", [], [], []),
        market_state=MarketStateResult(
            MarketStateLabel.RANGE, 0.0, "Structure analysis unavailable"
        ),
        key_levels=KeyLevels(),
        structure_narrative="Structure data unavailable.",
        buyer_context="", seller_context="", missing_confirmation="",
    )