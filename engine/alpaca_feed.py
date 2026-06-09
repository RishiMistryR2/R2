"""
AlpacaFeed: handles all market data ingestion.

Async WebSocket + REST seeding via AlpacaFeed class.
Synchronous REST helpers for update_cycle: validate_credentials,
get_latest_price, get_latest_trade_price, get_latest_quote, build_live_state.

Current-price policy (stamped on MarketState by _stamp_current_price):

  Priority:
    1. Latest trade  → source = "latest_trade" | "extended_hours_trade"
    2. Quote midpoint → source = "quote_midpoint" | "extended_hours_quote_midpoint"
    3. 1m bar close  → source = "1m_bar_fallback"

  The session (regular vs extended) is determined by engine.market_session so
  labeling is authoritative and consistent with the dashboard.

Credential policy:
  ALPACA_API_KEY / ALPACA_SECRET_KEY are read via os.getenv() at call time.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone, timedelta
from typing import Callable, Awaitable, Optional

try:
    import aiohttp
    _HAS_AIOHTTP = True
except ImportError:
    aiohttp = None          # type: ignore[assignment]
    _HAS_AIOHTTP = False

try:
    import websockets
    _HAS_WEBSOCKETS = True
except ImportError:
    websockets = None       # type: ignore[assignment]
    _HAS_WEBSOCKETS = False

import requests as _requests

from config.settings import (
    ALPACA_BASE_URL,
    ALPACA_DATA_URL,
    ALPACA_WS_URL,
    DATA_FEED_TIMEOUT,
    SYMBOL,
    TIMEFRAMES,
    CANDLE_HISTORY,
    ALPACA_TF_MAP,
)
from engine.market_session import get_market_session

logger = logging.getLogger(__name__)

TickCallback = Callable[[float, float, datetime], Awaitable[None]]
BarCallback  = Callable[[dict, str], Awaitable[None]]


# ──────────────────────────────────────────────────────────────
# Credential helpers
# ──────────────────────────────────────────────────────────────

def _api_key() -> str:
    return os.getenv("ALPACA_API_KEY", "")


def _secret_key() -> str:
    return os.getenv("ALPACA_SECRET_KEY", "")


def _data_headers() -> dict:
    return {
        "APCA-API-KEY-ID":     _api_key(),
        "APCA-API-SECRET-KEY": _secret_key(),
        "Accept":              "application/json",
    }


def _ws_auth_headers() -> dict:
    return {
        "APCA-API-KEY-ID":     _api_key(),
        "APCA-API-SECRET-KEY": _secret_key(),
    }


# ══════════════════════════════════════════════════════════════
# AlpacaFeed — async WebSocket + REST seeding
# ══════════════════════════════════════════════════════════════

class AlpacaFeed:
    """
    Manages real-time and historical market data from Alpaca.

    Usage:
        feed = AlpacaFeed(on_tick=..., on_bar=...)
        await feed.seed_history()
        await feed.run()
    """

    def __init__(self, on_tick: TickCallback, on_bar: BarCallback) -> None:
        self.on_tick  = on_tick
        self.on_bar   = on_bar
        self._running = False
        self._ws      = None

    async def seed_history(self) -> None:
        if not _HAS_AIOHTTP:
            logger.warning("aiohttp not installed — skipping async seed_history")
            return
        logger.info("Seeding historical candle data...")
        async with aiohttp.ClientSession(headers=_ws_auth_headers()) as session:
            for tf in TIMEFRAMES:
                alpaca_tf = ALPACA_TF_MAP.get(tf)
                if alpaca_tf:
                    await self._fetch_bars(session, alpaca_tf, CANDLE_HISTORY[tf], tf)
        logger.info("Historical seed complete.")

    async def _fetch_bars(self, session, alpaca_tf, limit, our_tf):
        url    = f"{ALPACA_DATA_URL}/v2/stocks/{SYMBOL}/bars"
        params = {"timeframe": alpaca_tf, "limit": min(limit, 1000),
                  "feed": "iex", "sort": "asc"}
        try:
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    logger.warning(f"Failed {our_tf}: {resp.status}")
                    return
                data = await resp.json()
                for bar in data.get("bars", []):
                    await self.on_bar(self._parse_bar(bar), our_tf)
                logger.info(f"  {our_tf}: {len(data.get('bars', []))} bars")
        except Exception as e:
            logger.error(f"Error fetching {our_tf}: {e}")

    async def run(self) -> None:
        if not _HAS_WEBSOCKETS:
            logger.warning("websockets not installed — live WS unavailable")
            return
        self._running = True
        while self._running:
            try:
                await self._connect_and_stream()
            except Exception as e:
                logger.error(f"WebSocket error: {e}. Reconnecting in 5s...")
                await asyncio.sleep(5)

    async def _connect_and_stream(self) -> None:
        logger.info(f"Connecting: {ALPACA_WS_URL}")
        async with websockets.connect(ALPACA_WS_URL) as ws:
            self._ws = ws
            await ws.send(json.dumps({"action": "auth",
                                       "key": _api_key(), "secret": _secret_key()}))
            logger.info(f"Auth: {json.loads(await ws.recv())}")
            await ws.send(json.dumps({"action": "subscribe", "trades": [SYMBOL]}))
            logger.info(f"Subscribed to {SYMBOL}.")
            async for raw in ws:
                for msg in json.loads(raw):
                    await self._handle_message(msg)

    async def _handle_message(self, msg):
        t = msg.get("T")
        if t == "t":
            await self.on_tick(float(msg["p"]), float(msg["s"]),
                               _parse_ts_str(msg["t"]))
        elif t == "b":
            await self.on_bar({
                "timestamp": _parse_ts_str(msg["t"]),
                "open": float(msg["o"]), "high": float(msg["h"]),
                "low":  float(msg["l"]), "close": float(msg["c"]),
                "volume": float(msg["v"]),
            }, "1m")
        elif t == "error":
            logger.error(f"Stream error: {msg}")

    async def stop(self) -> None:
        self._running = False
        if self._ws:
            await self._ws.close()

    @staticmethod
    def _parse_bar(raw):
        return {"timestamp": _parse_ts_str(raw["t"]),
                "open": float(raw["o"]), "high": float(raw["h"]),
                "low":  float(raw["l"]), "close": float(raw["c"]),
                "volume": float(raw["v"])}

    @staticmethod
    def _parse_ts(ts_str):
        return _parse_ts_str(ts_str)


# ══════════════════════════════════════════════════════════════
# Synchronous REST helpers
# ══════════════════════════════════════════════════════════════

def validate_credentials() -> bool:
    """Test Alpaca keys via a lightweight REST call.  Never raises."""
    if not _api_key() or not _secret_key():
        return False
    try:
        resp = _requests.get(
            f"{ALPACA_DATA_URL}/v2/stocks/SPY/bars/latest",
            headers=_data_headers(), params={"feed": "iex"},
            timeout=DATA_FEED_TIMEOUT,
        )
        if resp.status_code == 200:
            return bool(resp.json().get("bar", {}).get("c"))
        logger.warning(f"[alpaca] validate HTTP {resp.status_code}: {resp.text[:120]}")
        return False
    except Exception as e:
        logger.warning(f"[alpaca] validate failed: {e}")
        return False


def get_latest_price(symbol: str = "SPY") -> Optional[float]:
    """Return most recent bar close for *symbol*.  Never raises."""
    try:
        resp = _requests.get(
            f"{ALPACA_DATA_URL}/v2/stocks/{symbol}/bars/latest",
            headers=_data_headers(), params={"feed": "iex"},
            timeout=DATA_FEED_TIMEOUT,
        )
        if resp.status_code == 200:
            c = resp.json().get("bar", {}).get("c")
            return float(c) if c is not None else None
        return None
    except Exception as e:
        logger.warning(f"[alpaca] get_latest_price failed: {e}")
        return None


def get_latest_trade_price(symbol: str = "SPY") -> Optional[tuple]:
    """
    Fetch the most recent trade from Alpaca (regular + extended hours).

    Returns (price: float, timestamp: datetime) or None on any failure.
    The timestamp may be from a previous session (e.g. yesterday's close)
    when called outside trading hours — callers should check the age.
    Never raises.
    """
    try:
        resp = _requests.get(
            f"{ALPACA_DATA_URL}/v2/stocks/{symbol}/trades/latest",
            headers=_data_headers(), params={"feed": "iex"},
            timeout=DATA_FEED_TIMEOUT,
        )
        if resp.status_code == 200:
            trade = resp.json().get("trade", {})
            price = trade.get("p")
            ts    = trade.get("t")
            if price is not None:
                return float(price), (_parse_ts_str(ts) if ts else None)
        return None
    except Exception as e:
        logger.warning(f"[alpaca] get_latest_trade_price failed: {e}")
        return None


def get_latest_quote(symbol: str = "SPY") -> Optional[dict]:
    """
    Fetch the most recent NBBO quote from Alpaca.

    Returns dict {bid, ask, bid_size, ask_size, midpoint, spread, timestamp}
    or None on any failure.  Never raises.
    """
    try:
        resp = _requests.get(
            f"{ALPACA_DATA_URL}/v2/stocks/{symbol}/quotes/latest",
            headers=_data_headers(), params={"feed": "iex"},
            timeout=DATA_FEED_TIMEOUT,
        )
        if resp.status_code == 200:
            q   = resp.json().get("quote", {})
            bid = q.get("bp")
            ask = q.get("ap")
            if bid is not None and ask is not None:
                bid, ask = float(bid), float(ask)
                return {
                    "bid":       bid,
                    "ask":       ask,
                    "bid_size":  int(q.get("bs", 0)),
                    "ask_size":  int(q.get("as", 0)),
                    "midpoint":  round((bid + ask) / 2, 4),
                    "spread":    round(ask - bid, 4),
                    "timestamp": _parse_ts_str(q["t"]) if q.get("t") else None,
                }
        return None
    except Exception as e:
        logger.warning(f"[alpaca] get_latest_quote failed: {e}")
        return None


def _stamp_current_price(state, symbol: str, bar_fallback: float) -> None:
    """
    Fetch the best available current price and stamp it on *state*.

    Source priority:  latest trade  >  quote midpoint  >  1m bar close
    The source label reflects both what was used AND the current session:

      regular hours   → "latest_trade"               / "quote_midpoint"
      extended hours  → "extended_hours_trade"        / "extended_hours_quote_midpoint"
      any session     → "1m_bar_fallback"  (when trade+quote unavailable)

    Stamped attributes (dynamic — use getattr with fallback in callers):
      current_price              float
      current_price_source       str
      current_price_timestamp    datetime | None
      bid                        float | None
      ask                        float | None
      spread                     float | None
    """
    from engine.market_session import is_extended_hours

    session = get_market_session()
    ext     = is_extended_hours(session)

    # ── 1. Latest trade ────────────────────────────────────
    trade = get_latest_trade_price(symbol)
    if trade:
        price, ts = trade
        state.current_price           = price
        state.current_price_source    = "extended_hours_trade" if ext else "latest_trade"
        state.current_price_timestamp = ts
        state.bid    = None
        state.ask    = None
        state.spread = None
        logger.debug(
            f"[alpaca] current_price={price:.4f}"
            f"  source={state.current_price_source}"
        )
        return

    # ── 2. Quote midpoint ───────────────────────────────────
    quote = get_latest_quote(symbol)
    if quote:
        state.current_price           = quote["midpoint"]
        state.current_price_source    = (
            "extended_hours_quote_midpoint" if ext else "quote_midpoint"
        )
        state.current_price_timestamp = quote["timestamp"]
        state.bid    = quote["bid"]
        state.ask    = quote["ask"]
        state.spread = quote["spread"]
        logger.debug(
            f"[alpaca] current_price={quote['midpoint']:.4f}"
            f"  source={state.current_price_source}"
            f"  bid={quote['bid']}  ask={quote['ask']}"
        )
        return

    # ── 3. 1m bar close fallback ────────────────────────────
    state.current_price           = bar_fallback
    state.current_price_source    = "1m_bar_fallback"
    state.current_price_timestamp = None
    state.bid    = None
    state.ask    = None
    state.spread = None
    logger.debug(
        f"[alpaca] current_price={bar_fallback:.4f}"
        f"  source=1m_bar_fallback (trade+quote unavailable)"
    )


# ──────────────────────────────────────────────────────────────
# Internal bar-fetch helper
# ──────────────────────────────────────────────────────────────

def _fetch_bars_sync(symbol, alpaca_tf, limit, lookback_days):
    start_iso = (
        datetime.now(tz=timezone.utc) - timedelta(days=lookback_days)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    resp = _requests.get(
        f"{ALPACA_DATA_URL}/v2/stocks/{symbol}/bars",
        headers=_data_headers(),
        params={"timeframe": alpaca_tf, "start": start_iso,
                "limit": min(limit, 1000), "feed": "iex", "sort": "asc"},
        timeout=DATA_FEED_TIMEOUT,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Alpaca bars HTTP {resp.status_code}: {resp.text[:200]}")
    return [
        {
            "timestamp": _parse_ts_str(b["t"]),
            "open":      float(b["o"]), "high":  float(b["h"]),
            "low":       float(b["l"]), "close": float(b["c"]),
            "volume":    float(b.get("v", 0)),
            "vwap":      float(b["vw"]) if b.get("vw") else float(b["c"]),
        }
        for b in resp.json().get("bars", [])
    ]


def _parse_ts_str(ts_str: str) -> datetime:
    return datetime.fromisoformat(
        ts_str.replace("Z", "+00:00")
    ).replace(tzinfo=timezone.utc)


# ──────────────────────────────────────────────────────────────
# Indicator maths (pure Python)
# ──────────────────────────────────────────────────────────────

def _ema(closes, period):
    if not closes:
        return 0.0
    if len(closes) < period:
        return closes[-1]
    k   = 2.0 / (period + 1)
    val = sum(closes[:period]) / period
    for c in closes[period:]:
        val = c * k + val * (1 - k)
    return round(val, 4)


def _rsi(closes, period=14):
    if len(closes) <= period:
        return 50.0
    recent = closes[-(period + 1):]
    gains, losses = [], []
    for i in range(1, len(recent)):
        d = recent[i] - recent[i - 1]
        gains.append(max(0.0, d))
        losses.append(max(0.0, -d))
    avg_g = sum(gains) / period
    avg_l = sum(losses) / period
    if avg_l == 0:
        return 100.0
    return round(100 - (100 / (1 + avg_g / avg_l)), 2)


def _atr(bars, period=14):
    if len(bars) < 2:
        return 1.0
    trs = [
        max(bars[i]["high"] - bars[i]["low"],
            abs(bars[i]["high"] - bars[i - 1]["close"]),
            abs(bars[i]["low"]  - bars[i - 1]["close"]))
        for i in range(1, len(bars))
    ]
    recent = trs[-period:]
    return round(sum(recent) / len(recent), 4) if recent else 1.0


def _session_vwap(bars):
    vol = sum(b["volume"] for b in bars)
    if vol == 0:
        return bars[-1]["close"] if bars else 0.0
    return round(sum(b["close"] * b["volume"] for b in bars) / vol, 2)


def _macd_components(closes):
    if len(closes) < 27:
        return 0.0, 0.0, "neutral"
    line = _ema(closes, 12) - _ema(closes, 26)
    return line, round(line, 4), ("bullish" if line > 0 else "bearish" if line < 0 else "neutral")


def _swing_levels(bars, lookback=20):
    recent = bars[-lookback:] if len(bars) >= lookback else bars
    return min(b["low"] for b in recent), max(b["high"] for b in recent)


# ──────────────────────────────────────────────────────────────
# build_live_state
# ──────────────────────────────────────────────────────────────

def build_live_state(symbol: str = "SPY"):
    """
    Build a fully-populated MarketState from real Alpaca REST data.

    Bars → indicators + regime + narrative.
    Trade/quote → current_price (more recent than completed bar close).
    MarketSession → correct session phase and label.

    Raises on failure — caller catches and falls back to demo state.
    """
    from engine.models import (
        MarketState, IndicatorSnapshot, TrendState,
        Direction, TrendStrength,
        RegimeState, MarketRegime,
        NarrativeState, MarketNarrative,
        SessionState, Readiness,
        SupportResistanceLevel, SessionIntent,
    )
    from engine.market_session import (
        get_market_session, to_session_phase, is_regular_hours, MarketSession,
    )

    now     = datetime.now(tz=timezone.utc)
    session = get_market_session(now)
    phase   = to_session_phase(session)

    TF_CONFIG = [
        ("1d",  "1Day",  60,  60),
        ("1h",  "1Hour", 10,  80),
        ("15m", "15Min",  4, 100),
        ("5m",  "5Min",   2, 120),
        ("1m",  "1Min",   1, 120),
    ]

    all_bars: dict = {}
    for tf_name, alpaca_tf, lookback, limit in TF_CONFIG:
        try:
            all_bars[tf_name] = _fetch_bars_sync(symbol, alpaca_tf, limit, lookback)
            logger.debug(f"[alpaca] {tf_name}: {len(all_bars[tf_name])} bars")
        except Exception as e:
            logger.warning(f"[alpaca] {tf_name} bars failed: {e}")
            all_bars[tf_name] = []

    bars_5m = all_bars.get("5m", [])
    if not bars_5m:
        raise ValueError(f"No 5m bars for {symbol} — Alpaca unavailable?")

    latest_bar_close = bars_5m[-1]["close"]

    # ── Indicators ──────────────────────────────────────────
    indicators = {}
    for tf_name, _, _, _ in TF_CONFIG:
        bars = all_bars.get(tf_name, [])
        if not bars:
            indicators[tf_name] = IndicatorSnapshot(tf_name, now, latest_bar_close)
            continue
        closes = [b["close"] for b in bars]
        last   = bars[-1]
        close  = last["close"]
        ind               = IndicatorSnapshot(tf_name, now, close)
        ind.atr           = _atr(bars)
        ind.ema_9         = _ema(closes, 9)
        ind.ema_20        = _ema(closes, 20)
        ind.rsi           = _rsi(closes)
        _, ind.macd_hist, ind.macd_bias = _macd_components(closes)
        ind.vwap          = last.get("vwap") or _session_vwap(bars[-60:])
        ind.price_vs_vwap = ("above" if close > ind.vwap else
                              "below" if close < ind.vwap else "at")
        ind.ema_stack     = ("bull_aligned" if close > ind.ema_9 > ind.ema_20 else
                              "bear_aligned" if close < ind.ema_9 < ind.ema_20 else
                              "mixed")
        indicators[tf_name] = ind

    # ── Trends ──────────────────────────────────────────────
    trends = {}
    for tf_name, _, _, _ in TF_CONFIG:
        bars   = all_bars.get(tf_name, [])
        closes = [b["close"] for b in bars] if bars else [latest_bar_close]
        pct    = 0.0
        if len(closes) >= 10:
            ra  = sum(closes[-5:]) / 5
            pa  = sum(closes[-10:-5]) / 5
            pct = (ra - pa) / pa * 100 if pa else 0
        if pct > 0.3:
            d, s = Direction.BULLISH, (TrendStrength.STRONG if pct > 1.0 else TrendStrength.MODERATE)
        elif pct < -0.3:
            d, s = Direction.BEARISH, (TrendStrength.STRONG if pct < -1.0 else TrendStrength.MODERATE)
        else:
            d, s = Direction.NEUTRAL, TrendStrength.WEAK
        tr              = TrendState(tf_name, d, s)
        tr.higher_highs = (d == Direction.BULLISH)
        if bars:
            tr.last_swing_low, tr.last_swing_high = _swing_levels(bars)
        else:
            tr.last_swing_low  = latest_bar_close * 0.992
            tr.last_swing_high = latest_bar_close * 1.008
        trends[tf_name] = tr

    # ── Support / Resistance ─────────────────────────────────
    sl, sh = _swing_levels(bars_5m, lookback=40)
    support_levels    = [SupportResistanceLevel(round(sl, 2), "support",    3)]
    resistance_levels = [SupportResistanceLevel(round(sh, 2), "resistance", 3)]

    # ── Regime ──────────────────────────────────────────────
    dir_1h, dir_15m = trends["1h"].direction, trends["15m"].direction
    if   dir_1h == Direction.BULLISH and dir_15m == Direction.BULLISH:
        regime, rc, ok, caution = MarketRegime.TREND_BULL, 0.75, True,  ""
    elif dir_1h == Direction.BEARISH and dir_15m == Direction.BEARISH:
        regime, rc, ok, caution = MarketRegime.TREND_BEAR, 0.75, True,  ""
    elif dir_1h == Direction.NEUTRAL  and dir_15m == Direction.NEUTRAL:
        regime, rc, ok, caution = MarketRegime.RANGE,      0.55, False, "No trend — wait for breakout"
    else:
        regime, rc, ok, caution = MarketRegime.CHOP,       0.55, False, "Mixed signals — reduce size"

    # During extended/closed hours, disable new entries
    if not is_regular_hours(session):
        ok = False
        caution = caution or f"{session.value} — no new entries"

    score_mult = 1.0 if ok else 0.7

    # ── Narrative ────────────────────────────────────────────
    _NAR = {
        MarketRegime.TREND_BULL: (MarketNarrative.TRENDING_BULL, "long_bias",  0.78, "Bullish trend"),
        MarketRegime.TREND_BEAR: (MarketNarrative.TRENDING_BEAR, "short_bias", 0.78, "Bearish trend"),
        MarketRegime.RANGE:      (MarketNarrative.CHOP,       "neutral",    0.60, "Ranging"),
        MarketRegime.CHOP:       (MarketNarrative.CHOP,        "neutral",    0.55, "Choppy"),
    }
    narrative, posture, ncf, ndesc = _NAR.get(
        regime, (MarketNarrative.CHOP, "neutral", 0.55, "Unclear")
    )

    # ── Assemble ─────────────────────────────────────────────
    state = MarketState(symbol=symbol, timestamp=now)
    for tf_name in ("1d", "1h", "15m", "5m", "1m"):
        state.indicators[tf_name] = indicators[tf_name]
        state.trends[tf_name]     = trends[tf_name]
    state.support_levels    = support_levels
    state.resistance_levels = resistance_levels
    state.session = SessionState(
        phase=phase, market_open=is_regular_hours(session),
        entries_allowed=ok, score_modifier=score_mult,
        entry_filter="normal", phase_note="live data",
        readiness=Readiness.FULLY_READY,
    )
    state.regime = RegimeState(
        regime, confidence=rc, score_multiplier=score_mult,
        entry_permitted=ok, caution_note=caution,
    )
    state.narrative = NarrativeState(
        narrative=narrative, confidence=ncf,
        description=ndesc, trade_posture=posture,
    )
    state.intent             = SessionIntent(min_score=60)
    state.context_bias       = "neutral"
    state.context_risk_level = "normal"
    state.data_source        = "LIVE"
    state.market_session     = session   # authoritative MarketSession enum
    state.bars               = all_bars  # store raw bars for structure engine

    # Current price (trade > quote > 1m bar close)
    bars_1m      = all_bars.get("1m", [])
    bar_1m_close = bars_1m[-1]["close"] if bars_1m else latest_bar_close
    _stamp_current_price(state, symbol, bar_1m_close)

    logger.info(
        f"[alpaca] Live state: {symbol}"
        f"  bar={latest_bar_close:.2f}"
        f"  current={state.current_price:.2f}"
        f"  src={state.current_price_source}"
        f"  session={session.value}"
        f"  regime={regime.value}"
    )
    return state