"""
AlpacaFeed: handles all market data ingestion.

Two modes:
1. WebSocket (live) — subscribes to SPY trades for real-time tick data.
2. REST catch-up    — fetches historical bars at startup to seed candle history.

The feed calls a callback (on_tick or on_bar) whenever data arrives.
The main engine wires these callbacks to the CandleStore.

Sync REST helpers (validate_credentials, get_latest_price, build_live_state)
are defined at module level below AlpacaFeed and use `requests` so they can
be called from synchronous update_cycle code without an event loop.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Callable, Awaitable, Optional

try:
    import aiohttp
    _HAS_AIOHTTP = True
except ImportError:
    aiohttp = None         # type: ignore[assignment]
    _HAS_AIOHTTP = False

try:
    import websockets
    _HAS_WEBSOCKETS = True
except ImportError:
    websockets = None      # type: ignore[assignment]
    _HAS_WEBSOCKETS = False

from config.settings import (
    ALPACA_API_KEY,
    ALPACA_SECRET_KEY,
    ALPACA_BASE_URL,
    ALPACA_DATA_URL,
    ALPACA_WS_URL,
    SYMBOL,
    TIMEFRAMES,
    CANDLE_HISTORY,
)

logger = logging.getLogger(__name__)

# Type aliases
TickCallback = Callable[[float, float, datetime], Awaitable[None]]
BarCallback  = Callable[[dict, str], Awaitable[None]]


class AlpacaFeed:
    """
    Manages real-time and historical market data from Alpaca.

    Usage:
        feed = AlpacaFeed(on_tick=..., on_bar=...)
        await feed.seed_history()   # REST catch-up before going live
        await feed.run()            # starts WebSocket, runs forever
    """

    def __init__(
        self,
        on_tick: TickCallback,
        on_bar:  BarCallback,
    ) -> None:
        self.on_tick = on_tick
        self.on_bar  = on_bar
        self._running = False
        self._ws = None

    # ──────────────────────────────────────────
    # REST: historical catch-up
    # ──────────────────────────────────────────

    async def seed_history(self) -> None:
        """
        Fetch enough historical bars to seed all timeframe candle histories.
        Called once at startup before the WebSocket connects.
        """
        if not _HAS_AIOHTTP:
            logger.warning("aiohttp not installed — skipping async seed_history")
            return

        logger.info("Seeding historical candle data...")
        headers = self._auth_headers()

        async with aiohttp.ClientSession(headers=headers) as session:
            for tf in TIMEFRAMES:
                if tf == "1m":
                    await self._fetch_bars(session, "1Min", CANDLE_HISTORY["1m"], "1m")
                elif tf == "5m":
                    await self._fetch_bars(session, "5Min", CANDLE_HISTORY["5m"], "5m")
                elif tf == "15m":
                    await self._fetch_bars(session, "15Min", CANDLE_HISTORY["15m"], "15m")
                elif tf == "1h":
                    await self._fetch_bars(session, "1Hour", CANDLE_HISTORY["1h"], "1h")
                elif tf == "1d":
                    await self._fetch_bars(session, "1Day", CANDLE_HISTORY["1d"], "1d")

        logger.info("Historical seed complete.")

    async def _fetch_bars(
        self,
        session:   "aiohttp.ClientSession",
        alpaca_tf: str,
        limit:     int,
        our_tf:    str,
    ) -> None:
        """Fetch up to `limit` bars of `alpaca_tf` for SYMBOL."""
        url = f"{ALPACA_DATA_URL}/v2/stocks/{SYMBOL}/bars"
        params = {
            "timeframe": alpaca_tf,
            "limit":     min(limit, 1000),
            "feed":      "iex",
            "sort":      "asc",
        }

        try:
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    logger.warning(f"Failed to fetch {our_tf} bars: {resp.status} — {text}")
                    return

                data = await resp.json()
                bars = data.get("bars", [])
                logger.info(f"  {our_tf}: fetched {len(bars)} bars")

                for bar in bars:
                    await self.on_bar(self._parse_bar(bar), our_tf)

        except Exception as e:
            logger.error(f"Error fetching {our_tf} history: {e}")

    # ──────────────────────────────────────────
    # WebSocket: live feed
    # ──────────────────────────────────────────

    async def run(self) -> None:
        """
        Connect to Alpaca WebSocket and stream live trades.
        Reconnects automatically on disconnect.
        """
        if not _HAS_WEBSOCKETS:
            logger.warning("websockets not installed — live WebSocket feed unavailable")
            return

        self._running = True
        while self._running:
            try:
                await self._connect_and_stream()
            except Exception as e:
                logger.error(f"WebSocket error: {e}. Reconnecting in 5s...")
                await asyncio.sleep(5)

    async def _connect_and_stream(self) -> None:
        logger.info(f"Connecting to Alpaca WebSocket: {ALPACA_WS_URL}")

        async with websockets.connect(ALPACA_WS_URL) as ws:
            self._ws = ws
            logger.info("WebSocket connected.")

            await ws.send(json.dumps({
                "action": "auth",
                "key":    ALPACA_API_KEY,
                "secret": ALPACA_SECRET_KEY,
            }))

            resp = json.loads(await ws.recv())
            logger.info(f"Auth response: {resp}")

            await ws.send(json.dumps({
                "action": "subscribe",
                "trades": [SYMBOL],
            }))

            logger.info(f"Subscribed to {SYMBOL} trades.")

            async for raw in ws:
                messages = json.loads(raw)
                for msg in messages:
                    await self._handle_message(msg)

    async def _handle_message(self, msg: dict) -> None:
        msg_type = msg.get("T")

        if msg_type == "t":
            price     = float(msg["p"])
            volume    = float(msg["s"])
            timestamp = _parse_ts_str(msg["t"])
            await self.on_tick(price, volume, timestamp)

        elif msg_type == "b":
            bar = {
                "timestamp": _parse_ts_str(msg["t"]),
                "open":      float(msg["o"]),
                "high":      float(msg["h"]),
                "low":       float(msg["l"]),
                "close":     float(msg["c"]),
                "volume":    float(msg["v"]),
            }
            await self.on_bar(bar, "1m")

        elif msg_type == "error":
            logger.error(f"Alpaca stream error: {msg}")

    async def stop(self) -> None:
        self._running = False
        if self._ws:
            await self._ws.close()

    # ──────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────

    def _auth_headers(self) -> dict:
        return {
            "APCA-API-KEY-ID":     ALPACA_API_KEY,
            "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
        }

    @staticmethod
    def _parse_bar(raw: dict) -> dict:
        return {
            "timestamp": _parse_ts_str(raw["t"]),
            "open":      float(raw["o"]),
            "high":      float(raw["h"]),
            "low":       float(raw["l"]),
            "close":     float(raw["c"]),
            "volume":    float(raw["v"]),
        }

    @staticmethod
    def _parse_ts(ts_str: str) -> datetime:
        """Parse Alpaca ISO timestamps (with or without Z suffix)."""
        return _parse_ts_str(ts_str)


# ══════════════════════════════════════════════════════════════
# Synchronous REST helpers  (used by update_cycle at every tick)
# ══════════════════════════════════════════════════════════════
#
# These functions do NOT require broker permissions — they only
# call the market-data endpoint (data.alpaca.markets).
# They use `requests` (always available) rather than the async
# `aiohttp` used by AlpacaFeed, so they can be called from
# synchronous update_cycle code without an event loop.

import requests as _requests

from config.settings import (
    DATA_FEED_TIMEOUT,
    ALPACA_TF_MAP,
)


def _data_headers() -> dict:
    """Build auth headers for Alpaca market-data REST calls."""
    return {
        "APCA-API-KEY-ID":     ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
        "Accept":              "application/json",
    }


def validate_credentials() -> bool:
    """
    Test whether valid Alpaca API keys are configured by making a
    single lightweight REST call (latest bar for SPY).

    Returns True on success, False on any failure.
    Never raises.
    """
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        return False
    try:
        url  = f"{ALPACA_DATA_URL}/v2/stocks/SPY/bars/latest"
        resp = _requests.get(
            url,
            headers = _data_headers(),
            params  = {"feed": "iex"},
            timeout = DATA_FEED_TIMEOUT,
        )
        if resp.status_code == 200:
            bar = resp.json().get("bar", {})
            return bool(bar.get("c"))
        logger.warning(
            f"[alpaca] validate_credentials HTTP {resp.status_code}: {resp.text[:120]}"
        )
        return False
    except Exception as e:
        logger.warning(f"[alpaca] validate_credentials failed: {e}")
        return False


def get_latest_price(symbol: str = "SPY") -> Optional[float]:
    """
    Return the most recent close price for *symbol* from Alpaca IEX feed.
    Returns None on any failure.
    """
    try:
        url  = f"{ALPACA_DATA_URL}/v2/stocks/{symbol}/bars/latest"
        resp = _requests.get(
            url,
            headers = _data_headers(),
            params  = {"feed": "iex"},
            timeout = DATA_FEED_TIMEOUT,
        )
        if resp.status_code == 200:
            bar = resp.json().get("bar", {})
            c   = bar.get("c")
            return float(c) if c is not None else None
        logger.warning(f"[alpaca] get_latest_price HTTP {resp.status_code}")
        return None
    except Exception as e:
        logger.warning(f"[alpaca] get_latest_price failed: {e}")
        return None


def _fetch_bars_sync(
    symbol:        str,
    alpaca_tf:     str,
    limit:         int,
    lookback_days: int,
) -> list:
    """
    Fetch up to *limit* bars for *symbol* at *alpaca_tf* granularity.
    Returns a list of normalised bar dicts.
    Raises on HTTP error so callers can catch and fall back.
    """
    start_iso = (
        datetime.now(tz=timezone.utc) - timedelta(days=lookback_days)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")

    url    = f"{ALPACA_DATA_URL}/v2/stocks/{symbol}/bars"
    params = {
        "timeframe": alpaca_tf,
        "start":     start_iso,
        "limit":     min(limit, 1000),
        "feed":      "iex",
        "sort":      "asc",
    }
    resp = _requests.get(
        url, headers=_data_headers(), params=params, timeout=DATA_FEED_TIMEOUT
    )
    if resp.status_code != 200:
        raise RuntimeError(
            f"Alpaca bars HTTP {resp.status_code}: {resp.text[:200]}"
        )
    raw_bars = resp.json().get("bars", [])
    return [
        {
            "timestamp": _parse_ts_str(b["t"]),
            "open":      float(b["o"]),
            "high":      float(b["h"]),
            "low":       float(b["l"]),
            "close":     float(b["c"]),
            "volume":    float(b.get("v", 0)),
            "vwap":      float(b["vw"]) if b.get("vw") else float(b["c"]),
        }
        for b in raw_bars
    ]


def _parse_ts_str(ts_str: str) -> datetime:
    ts_str = ts_str.replace("Z", "+00:00")
    return datetime.fromisoformat(ts_str).replace(tzinfo=timezone.utc)


# ──────────────────────────────────────────────────────────────
# Indicator maths (pure Python, no numpy required)
# ──────────────────────────────────────────────────────────────

def _ema(closes: list, period: int) -> float:
    if not closes:
        return 0.0
    if len(closes) < period:
        return closes[-1]
    k   = 2.0 / (period + 1)
    val = sum(closes[:period]) / period
    for c in closes[period:]:
        val = c * k + val * (1 - k)
    return round(val, 4)


def _rsi(closes: list, period: int = 14) -> float:
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


def _atr(bars: list, period: int = 14) -> float:
    if len(bars) < 2:
        return 1.0
    trs = []
    for i in range(1, len(bars)):
        h, l, pc = bars[i]["high"], bars[i]["low"], bars[i - 1]["close"]
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    recent = trs[-period:]
    return round(sum(recent) / len(recent), 4) if recent else 1.0


def _session_vwap(bars: list) -> float:
    """Volume-weighted average price over the provided bars."""
    total_vol = sum(b["volume"] for b in bars)
    if total_vol == 0:
        return bars[-1]["close"] if bars else 0.0
    return round(
        sum(b["close"] * b["volume"] for b in bars) / total_vol, 2
    )


def _macd_components(closes: list) -> tuple:
    """Returns (macd_line, approx_histogram, bias_str)."""
    if len(closes) < 27:
        return 0.0, 0.0, "neutral"
    ema12 = _ema(closes, 12)
    ema26 = _ema(closes, 26)
    line  = ema12 - ema26
    bias  = "bullish" if line > 0 else "bearish" if line < 0 else "neutral"
    return line, round(line, 4), bias


def _swing_levels(bars: list, lookback: int = 20) -> tuple:
    """Return (swing_low, swing_high) from recent bars."""
    recent = bars[-lookback:] if len(bars) >= lookback else bars
    lows   = [b["low"]  for b in recent]
    highs  = [b["high"] for b in recent]
    return min(lows), max(highs)


# ──────────────────────────────────────────────────────────────
# build_live_state  — the main public entry point for live data
# ──────────────────────────────────────────────────────────────

def build_live_state(symbol: str = "SPY"):
    """
    Build a fully-populated MarketState from real Alpaca REST data.

    Fetches bars for all priority timeframes, calculates indicators,
    and infers regime / narrative from recent price action.

    Raises on failure — caller must catch and fall back to demo state.
    Requires valid ALPACA_API_KEY / ALPACA_SECRET_KEY in environment.
    """
    from engine.models import (
        MarketState, IndicatorSnapshot, TrendState,
        Direction, TrendStrength,
        RegimeState, MarketRegime,
        NarrativeState, MarketNarrative,
        SessionState, SessionPhase, Readiness,
        SupportResistanceLevel, SessionIntent,
    )

    now = datetime.now(tz=timezone.utc)

    # (tf_label, alpaca_tf_string, lookback_days, limit)
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
            bars = _fetch_bars_sync(symbol, alpaca_tf, limit, lookback)
            all_bars[tf_name] = bars
            logger.debug(f"[alpaca] {tf_name}: {len(bars)} bars")
        except Exception as e:
            logger.warning(f"[alpaca] {tf_name} bars failed: {e}")
            all_bars[tf_name] = []

    # Need 5m bars at minimum to build a useful state
    bars_5m = all_bars.get("5m", [])
    if not bars_5m:
        raise ValueError(
            f"No 5m bars returned for {symbol} — Alpaca unavailable?"
        )

    latest_close = bars_5m[-1]["close"]

    # ── Build IndicatorSnapshots ────────────────────────────
    indicators = {}
    for tf_name, _, _, _ in TF_CONFIG:
        bars = all_bars.get(tf_name, [])
        if not bars:
            ind = IndicatorSnapshot(tf_name, now, latest_close)
            indicators[tf_name] = ind
            continue

        closes = [b["close"] for b in bars]
        last   = bars[-1]
        close  = last["close"]

        ind             = IndicatorSnapshot(tf_name, now, close)
        ind.atr         = _atr(bars)
        ind.ema_9       = _ema(closes, 9)
        ind.ema_20      = _ema(closes, 20)
        ind.rsi         = _rsi(closes)
        _, ind.macd_hist, ind.macd_bias = _macd_components(closes)

        ind.vwap = last.get("vwap") or _session_vwap(bars[-60:])
        ind.price_vs_vwap = (
            "above" if close > ind.vwap else
            "below" if close < ind.vwap else "at"
        )

        if close > ind.ema_9 > ind.ema_20:
            ind.ema_stack = "bull_aligned"
        elif close < ind.ema_9 < ind.ema_20:
            ind.ema_stack = "bear_aligned"
        else:
            ind.ema_stack = "mixed"

        indicators[tf_name] = ind

    # ── TrendStates ─────────────────────────────────────────
    trends = {}
    for tf_name, _, _, _ in TF_CONFIG:
        bars   = all_bars.get(tf_name, [])
        closes = [b["close"] for b in bars] if bars else [latest_close]

        if len(closes) >= 10:
            recent_avg = sum(closes[-5:]) / 5
            prior_avg  = sum(closes[-10:-5]) / 5
            pct_chg    = (recent_avg - prior_avg) / prior_avg * 100 if prior_avg else 0
        else:
            pct_chg = 0.0

        if pct_chg > 0.3:
            direction = Direction.BULLISH
            strength  = TrendStrength.STRONG if pct_chg > 1.0 else TrendStrength.MODERATE
        elif pct_chg < -0.3:
            direction = Direction.BEARISH
            strength  = TrendStrength.STRONG if pct_chg < -1.0 else TrendStrength.MODERATE
        else:
            direction = Direction.NEUTRAL
            strength  = TrendStrength.WEAK

        tr = TrendState(tf_name, direction, strength)
        tr.higher_highs = (direction == Direction.BULLISH)

        if bars:
            swing_low, swing_high  = _swing_levels(bars)
            tr.last_swing_low      = swing_low
            tr.last_swing_high     = swing_high
        else:
            tr.last_swing_low  = latest_close * 0.992
            tr.last_swing_high = latest_close * 1.008

        trends[tf_name] = tr

    # ── Support / Resistance ─────────────────────────────────
    swing_low_5m, swing_high_5m = _swing_levels(bars_5m, lookback=40)
    support_levels    = [SupportResistanceLevel(round(swing_low_5m,  2), "support",    3)]
    resistance_levels = [SupportResistanceLevel(round(swing_high_5m, 2), "resistance", 3)]

    # ── Regime from 1h + 15m alignment ──────────────────────
    dir_1h  = trends.get("1h",  TrendState("1h",  Direction.NEUTRAL, TrendStrength.WEAK)).direction
    dir_15m = trends.get("15m", TrendState("15m", Direction.NEUTRAL, TrendStrength.WEAK)).direction

    if dir_1h == Direction.BULLISH and dir_15m == Direction.BULLISH:
        regime    = MarketRegime.TREND_BULL
        regime_cf = 0.75
        entry_ok  = True
        caution   = ""
    elif dir_1h == Direction.BEARISH and dir_15m == Direction.BEARISH:
        regime    = MarketRegime.TREND_BEAR
        regime_cf = 0.75
        entry_ok  = True
        caution   = ""
    elif dir_1h == Direction.NEUTRAL and dir_15m == Direction.NEUTRAL:
        regime    = MarketRegime.RANGE
        regime_cf = 0.55
        entry_ok  = False
        caution   = "No directional trend — wait for breakout"
    else:
        regime    = MarketRegime.CHOP
        regime_cf = 0.55
        entry_ok  = False
        caution   = "Mixed timeframe signals — reduce size"

    score_mult = 1.0 if entry_ok else 0.7

    # ── Narrative ────────────────────────────────────────────
    if regime == MarketRegime.TREND_BULL:
        narrative     = MarketNarrative.TRENDING_BULL
        trade_posture = "long_bias"
        narr_cf       = 0.78
        narr_desc     = "Bullish trend across timeframes"
    elif regime == MarketRegime.TREND_BEAR:
        narrative     = MarketNarrative.TRENDING_BEAR
        trade_posture = "short_bias"
        narr_cf       = 0.78
        narr_desc     = "Bearish trend across timeframes"
    elif regime == MarketRegime.RANGE:
        narrative     = MarketNarrative.RANGING
        trade_posture = "neutral"
        narr_cf       = 0.60
        narr_desc     = "Price ranging — mean-reversion favoured"
    else:
        narrative     = MarketNarrative.CHOPPY
        trade_posture = "neutral"
        narr_cf       = 0.55
        narr_desc     = "Choppy — wait for clarity"

    # ── Session phase ────────────────────────────────────────
    h = now.hour
    if h < 14:
        phase = SessionPhase.OPEN
    elif h < 18:
        phase = SessionPhase.MIDDAY
    else:
        phase = SessionPhase.POWER_HOUR

    # ── Assemble MarketState ─────────────────────────────────
    state = MarketState(symbol=symbol, timestamp=now)

    for tf_name in ("1d", "1h", "15m", "5m", "1m"):
        state.indicators[tf_name] = indicators[tf_name]
        state.trends[tf_name]     = trends[tf_name]

    state.support_levels    = support_levels
    state.resistance_levels = resistance_levels

    state.session = SessionState(
        phase           = phase,
        market_open     = True,
        entries_allowed = entry_ok,
        score_modifier  = score_mult,
        entry_filter    = "normal",
        phase_note      = "live data",
        readiness       = Readiness.FULLY_READY,
    )

    state.regime = RegimeState(
        regime,
        confidence       = regime_cf,
        score_multiplier = score_mult,
        entry_permitted  = entry_ok,
        caution_note     = caution,
    )

    state.narrative = NarrativeState(
        narrative     = narrative,
        confidence    = narr_cf,
        description   = narr_desc,
        trade_posture = trade_posture,
    )

    state.intent = SessionIntent(min_score=60)

    state.context_bias       = "neutral"
    state.context_risk_level = "normal"

    # Always tag explicitly — no silent fake prices
    state.data_source = "LIVE"

    logger.info(
        f"[alpaca] Live state built: {symbol} @ {latest_close:.2f}"
        f"  regime={regime.value}"
    )
    return state