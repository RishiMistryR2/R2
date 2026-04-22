"""
engine/alpaca_feed.py — market data ingestion from Alpaca.

Two modes:
  REST seed:  fetches historical bars at startup in SEED_ORDER (1d first)
              so regime is available as early as possible.
  WebSocket:  streams live trades; reconnects automatically on disconnect.

Session-type tagging: every bar/tick is classified as
"regular" | "premarket" | "afterhours" based on its timestamp.

Callbacks:
  on_tick(price, volume, timestamp, session_type) — per trade
  on_bar(bar_dict, timeframe)                      — per REST bar
  on_seed_complete(timeframe, count)               — after each TF seeds

All are async callables so they can await the CandleStore lock.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Awaitable, Callable, Optional

from engine.candle_store import classify_session_type

from config.settings import (
    ALPACA_API_KEY,
    ALPACA_DATA_URL,
    ALPACA_SECRET_KEY,
    ALPACA_WS_URL,
    ALPACA_TF_MAP,
    SEED_BAR_COUNTS,
    SEED_ORDER,
    SYMBOL,
)

logger = logging.getLogger(__name__)

# Type aliases
TickCallback       = Callable[[float, float, datetime, str], Awaitable[None]]
BarCallback        = Callable[[dict, str], Awaitable[None]]
SeedDoneCallback   = Callable[[str, int], Awaitable[None]]

# Alpaca REST fetch parameters
_FETCH_TIMEOUT_S = 15
_RECONNECT_DELAY_S = 5


class AlpacaFeed:
    """
    Manages REST seeding + live WebSocket streaming from Alpaca.

    Usage:
        feed = AlpacaFeed(on_tick=..., on_bar=..., on_seed_complete=...)
        await feed.seed_history()   # blocks until all TFs seeded
        await feed.run()            # runs WebSocket loop forever
    """

    def __init__(
        self,
        on_tick:          TickCallback,
        on_bar:           BarCallback,
        on_seed_complete: Optional[SeedDoneCallback] = None,
    ) -> None:
        self._on_tick          = on_tick
        self._on_bar           = on_bar
        self._on_seed_complete = on_seed_complete
        self._running          = False
        self._ws               = None

    # ──────────────────────────────────────────
    # REST historical seed
    # ──────────────────────────────────────────

    async def seed_history(self) -> None:
        """
        Fetch historical bars for all timeframes in SEED_ORDER.
        Largest timeframe first (1d → 1h → 15m → 5m → 1m) so regime
        is computable as early as possible.

        Progress is reported via on_seed_complete after each TF.
        """
        logger.info(f"[R2] Seeding historical bars ({' → '.join(SEED_ORDER)})...")

        try:
            import aiohttp
        except ImportError:
            logger.warning("[R2] aiohttp not installed — skipping historical seed")
            return

        headers = self._auth_headers()

        async with aiohttp.ClientSession(headers=headers) as session:
            for tf in SEED_ORDER:
                alpaca_tf = ALPACA_TF_MAP.get(tf)
                if not alpaca_tf:
                    continue
                count = await self._fetch_bars(session, tf, alpaca_tf)
                logger.info(f"[R2]   {tf}: seeded {count} bars")
                if self._on_seed_complete:
                    await self._on_seed_complete(tf, count)

        logger.info("[R2] Historical seed complete")

    async def _fetch_bars(
        self,
        session:   "aiohttp.ClientSession",
        our_tf:    str,
        alpaca_tf: str,
    ) -> int:
        """
        Fetch up to SEED_BAR_COUNTS[our_tf] bars of alpaca_tf for SYMBOL.
        Returns count of bars ingested.
        """
        import aiohttp

        limit = SEED_BAR_COUNTS.get(our_tf, 200)
        url   = f"{ALPACA_DATA_URL}/v2/stocks/{SYMBOL}/bars"
        params = {
            "timeframe": alpaca_tf,
            "limit":     min(limit, 1000),
            "feed":      "iex",
            "sort":      "asc",
        }

        try:
            timeout = aiohttp.ClientTimeout(total=_FETCH_TIMEOUT_S)
            async with session.get(url, params=params, timeout=timeout) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.warning(f"[R2] Seed {our_tf}: HTTP {resp.status} — {body[:200]}")
                    return 0

                data = await resp.json()
                bars = data.get("bars", [])
                ingested = 0

                for raw in bars:
                    bar = self._parse_bar(raw)
                    await self._on_bar(bar, our_tf)
                    ingested += 1

                # Handle pagination (next_page_token)
                while data.get("next_page_token") and ingested < limit:
                    params["page_token"] = data["next_page_token"]
                    async with session.get(url, params=params, timeout=timeout) as r2:
                        if r2.status != 200:
                            break
                        data = await r2.json()
                        for raw in data.get("bars", []):
                            bar = self._parse_bar(raw)
                            await self._on_bar(bar, our_tf)
                            ingested += 1
                        if ingested >= limit:
                            break

                return ingested

        except asyncio.TimeoutError:
            logger.warning(f"[R2] Seed {our_tf}: timeout after {_FETCH_TIMEOUT_S}s")
            return 0
        except Exception as e:
            logger.error(f"[R2] Seed {our_tf}: {type(e).__name__}: {e}")
            return 0

    # ──────────────────────────────────────────
    # WebSocket live feed
    # ──────────────────────────────────────────

    async def run(self) -> None:
        """Connect and stream. Reconnects automatically on disconnect."""
        self._running = True
        while self._running:
            try:
                await self._connect_and_stream()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"[R2] WebSocket error: {e} — reconnecting in {_RECONNECT_DELAY_S}s")
                await asyncio.sleep(_RECONNECT_DELAY_S)

    async def stop(self) -> None:
        self._running = False
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass

    async def _connect_and_stream(self) -> None:
        try:
            import websockets
        except ImportError:
            logger.error("[R2] websockets not installed — live feed unavailable")
            self._running = False
            return

        logger.info(f"[R2] Connecting to Alpaca WebSocket: {ALPACA_WS_URL}")

        async with websockets.connect(ALPACA_WS_URL) as ws:
            self._ws = ws
            logger.info("[R2] WebSocket connected")

            # Authenticate
            await ws.send(json.dumps({
                "action": "auth",
                "key":    ALPACA_API_KEY,
                "secret": ALPACA_SECRET_KEY,
            }))
            auth_resp = json.loads(await ws.recv())
            logger.debug(f"[R2] Auth response: {auth_resp}")

            # Subscribe to trades
            await ws.send(json.dumps({
                "action": "subscribe",
                "trades": [SYMBOL],
            }))
            logger.info(f"[R2] Subscribed to {SYMBOL} trades")

            async for raw in ws:
                if not self._running:
                    break
                try:
                    messages = json.loads(raw)
                    for msg in (messages if isinstance(messages, list) else [messages]):
                        await self._handle_message(msg)
                except Exception as e:
                    logger.debug(f"[R2] Message parse error: {e}")

    async def _handle_message(self, msg: dict) -> None:
        msg_type = msg.get("T")

        if msg_type == "t":   # trade
            price     = float(msg["p"])
            volume    = float(msg["s"])
            timestamp = self._parse_ts(msg["t"])
            session_t = classify_session_type(timestamp)
            await self._on_tick(price, volume, timestamp, session_t)

        elif msg_type == "b":  # bar (if subscribed to bars as well)
            bar = {
                "timestamp": self._parse_ts(msg["t"]),
                "open":      float(msg["o"]),
                "high":      float(msg["h"]),
                "low":       float(msg["l"]),
                "close":     float(msg["c"]),
                "volume":    float(msg["v"]),
            }
            await self._on_bar(bar, "1m")

        elif msg_type == "error":
            logger.error(f"[R2] Alpaca stream error: {msg}")

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
        """Parse Alpaca ISO timestamps (handles Z suffix and +00:00)."""
        ts_str = ts_str.replace("Z", "+00:00")
        return datetime.fromisoformat(ts_str).replace(tzinfo=timezone.utc)