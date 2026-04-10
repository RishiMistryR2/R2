"""
AlpacaFeed: handles all market data ingestion.

Two modes:
1. WebSocket (live) — subscribes to SPY trades for real-time tick data.
2. REST catch-up    — fetches historical bars at startup to seed candle history.

The feed calls a callback (on_tick or on_bar) whenever data arrives.
The main engine wires these callbacks to the CandleStore.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Callable, Awaitable, Optional

import aiohttp
import websockets

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
        self._ws: Optional[websockets.WebSocketClientProtocol] = None

    # ──────────────────────────────────────────
    # REST: historical catch-up
    # ──────────────────────────────────────────

    async def seed_history(self) -> None:
        """
        Fetch enough historical bars to seed all timeframe candle histories.
        Called once at startup before the WebSocket connects.
        """
        logger.info("Seeding historical candle data...")
        headers = self._auth_headers()

        async with aiohttp.ClientSession(headers=headers) as session:
            for tf in TIMEFRAMES:
                if tf == "1m":
                    # 1m: fetch last 2 trading days
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
        session:    aiohttp.ClientSession,
        alpaca_tf:  str,
        limit:      int,
        our_tf:     str,
    ) -> None:
        """Fetch up to `limit` bars of `alpaca_tf` for SYMBOL."""
        url = f"{ALPACA_DATA_URL}/v2/stocks/{SYMBOL}/bars"
        params = {
            "timeframe": alpaca_tf,
            "limit":     min(limit, 1000),
            "feed":      "iex",   # free tier
            "sort":      "asc",
        }

        try:
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    logger.warning(f"Failed to fetch {our_tf} bars: {resp.status} — {text}")
                    return

                data  = await resp.json()
                bars  = data.get("bars", [])
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

            # Auth
            await ws.send(json.dumps({
                "action": "auth",
                "key":    ALPACA_API_KEY,
                "secret": ALPACA_SECRET_KEY,
            }))

            # Wait for auth confirmation
            resp = json.loads(await ws.recv())
            logger.info(f"Auth response: {resp}")

            # Subscribe to trades
            await ws.send(json.dumps({
                "action":  "subscribe",
                "trades":  [SYMBOL],
                # Optionally add "bars": [SYMBOL] for server-side aggregation
            }))

            logger.info(f"Subscribed to {SYMBOL} trades.")

            async for raw in ws:
                messages = json.loads(raw)
                for msg in messages:
                    await self._handle_message(msg)

    async def _handle_message(self, msg: dict) -> None:
        msg_type = msg.get("T")

        if msg_type == "t":   # trade
            price     = float(msg["p"])
            volume    = float(msg["s"])
            timestamp = self._parse_ts(msg["t"])
            await self.on_tick(price, volume, timestamp)

        elif msg_type == "b":   # bar (if subscribed)
            bar = {
                "timestamp": self._parse_ts(msg["t"]),
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
            "timestamp": AlpacaFeed._parse_ts(raw["t"]),
            "open":      float(raw["o"]),
            "high":      float(raw["h"]),
            "low":       float(raw["l"]),
            "close":     float(raw["c"]),
            "volume":    float(raw["v"]),
        }

    @staticmethod
    def _parse_ts(ts_str: str) -> datetime:
        """Parse Alpaca ISO timestamps (with or without Z suffix)."""
        ts_str = ts_str.replace("Z", "+00:00")
        return datetime.fromisoformat(ts_str).replace(tzinfo=timezone.utc)
