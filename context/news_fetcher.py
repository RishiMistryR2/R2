"""
News Fetcher.

Primary source: Alpaca Markets news API (uses existing API key).
Covers Reuters, Benzinga, and major financial outlets.
Falls back gracefully to empty/cached results — never blocks.

Architecture:
  NewsFetcher.fetch() -> List[NewsItem]   (raw, unscored)
  ContextEngine then scores them with HeadlineScorer.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone, timedelta
from typing import List, Optional

import aiohttp

from context.models import NewsItem, SourceTier
from context.headline_scorer import HeadlineScorer
from config.settings import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_DATA_URL

logger = logging.getLogger(__name__)
_scorer = HeadlineScorer()


class NewsFetcher:
    """
    Fetches and caches financial news.
    Free Alpaca tier: ~200 req/day. Fetching every 5 min = ~80/day.
    """

    def __init__(self) -> None:
        self._last_fetch: Optional[datetime] = None
        self._cache: List[NewsItem] = []

    async def fetch(
        self,
        symbols:    List[str] = ["SPY"],
        hours_back: int = 4,
    ) -> List[NewsItem]:
        """
        Fetch recent news. Returns scored items sorted by importance.
        On failure returns cached result or empty list.
        """
        raw = await self._fetch_alpaca(symbols, hours_back)
        if not raw:
            return self._cache or []

        scored = _scorer.score_items(raw)
        self._cache = scored
        self._last_fetch = datetime.now(tz=timezone.utc)
        n_moving = sum(1 for i in scored if i.is_market_moving)
        logger.info(f"News: {len(scored)} items, {n_moving} market-moving")
        return scored

    async def fetch_market_moving_only(self) -> List[NewsItem]:
        items = await self.fetch()
        return [i for i in items if i.is_market_moving]

    async def _fetch_alpaca(self, symbols: List[str], hours_back: int) -> List[NewsItem]:
        if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
            logger.warning("Alpaca credentials not set — skipping news fetch")
            return []

        start = datetime.now(tz=timezone.utc) - timedelta(hours=hours_back)
        params = {
            "symbols": ",".join(symbols),
            "start":   start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "limit":   50,
            "sort":    "DESC",
        }
        headers = {
            "APCA-API-KEY-ID":     ALPACA_API_KEY,
            "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
        }
        url = f"{ALPACA_DATA_URL}/v1beta1/news"

        try:
            async with aiohttp.ClientSession(headers=headers) as session:
                async with session.get(
                    url, params=params,
                    timeout=aiohttp.ClientTimeout(total=8)
                ) as resp:
                    if resp.status != 200:
                        logger.warning(f"News fetch {resp.status}: {(await resp.text())[:200]}")
                        return []
                    data = await resp.json()
                    return [self._parse(item) for item in data.get("news", [])]
        except Exception as e:
            logger.error(f"News fetch error: {e}")
            return []

    def _parse(self, raw: dict) -> NewsItem:
        source = raw.get("author", "") or raw.get("source", "unknown")
        tier = _scorer.classify_source_tier(source)
        ts_str = raw.get("created_at", "")
        try:
            published_at = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            published_at = datetime.now(tz=timezone.utc)

        return NewsItem(
            id=raw.get("id", str(uuid.uuid4())),
            headline=raw.get("headline", ""),
            summary=raw.get("summary", ""),
            source=source,
            source_tier=tier,
            published_at=published_at,
            url=raw.get("url"),
        )
