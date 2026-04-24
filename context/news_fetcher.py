"""
context/news_fetcher.py — news retrieval for the context layer.

Two interfaces:

  fetch_news(symbol, injected_events=None) -> List[NewsEvent]
    Synchronous stub. Returns injected test data or sensible mock items.
    No external dependencies. Used by build_context() and tests.

  NewsFetcher  (class)
    Async wrapper around the Alpaca news API. Used in production.
    aiohttp is a soft dependency — absent → falls back to empty list.
    Scored items are returned as NewsItem (fully enriched).
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone, timedelta
from typing import List, Optional

from context.models import NewsEvent, NewsItem, SourceTier
from context.headline_scorer import HeadlineScorer, score_headline

try:
    import aiohttp as _aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

try:
    from config.settings import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_DATA_URL
except ImportError:
    ALPACA_API_KEY = ALPACA_SECRET_KEY = ALPACA_DATA_URL = ""

logger = logging.getLogger(__name__)
_scorer = HeadlineScorer()


# ──────────────────────────────────────────────
# Standalone stub (synchronous, no external deps)
# ──────────────────────────────────────────────

# Module-level injectable: tests (and build_context) set this to control output.
_injected_events: Optional[List[NewsEvent]] = None


def inject_news(events: Optional[List[NewsEvent]]) -> None:
    """
    Override the mock data returned by fetch_news().
    Pass None to restore the built-in mock.
    Intended for testing and dry-run scenarios.
    """
    global _injected_events
    _injected_events = events


def fetch_news(
    symbol: str = "SPY",
    injected: Optional[List[NewsEvent]] = None,
) -> List[NewsEvent]:
    """
    Return recent NewsEvent objects for the given symbol.

    Priority:
      1. `injected` argument  (per-call override, highest priority)
      2. `_injected_events`   (module-level override, set via inject_news())
      3. built-in mock data   (fallback — sensible neutral placeholder)

    The mock data is intentionally neutral so it has no bias effect on
    context unless tests explicitly inject something more directional.
    """
    if injected is not None:
        return injected
    if _injected_events is not None:
        return _injected_events
    return _mock_news(symbol)


def _mock_news(symbol: str) -> List[NewsEvent]:
    """Return neutral placeholder news for dry-run and test environments."""
    now = datetime.now(tz=timezone.utc)
    return [
        NewsEvent(
            headline=f"{symbol} trades near session highs amid steady volume",
            timestamp=now - timedelta(minutes=15),
            source="mock",
            symbols=[symbol],
            sentiment_score=score_headline(
                f"{symbol} trades near session highs amid steady volume"
            ),
        ),
        NewsEvent(
            headline="Federal Reserve officials signal data-dependent approach",
            timestamp=now - timedelta(hours=1),
            source="mock",
            symbols=["SPY", "QQQ"],
            sentiment_score=score_headline(
                "Federal Reserve officials signal data-dependent approach"
            ),
        ),
    ]


# ──────────────────────────────────────────────
# NewsFetcher — async production class
# ──────────────────────────────────────────────

class NewsFetcher:
    """
    Async news fetcher backed by the Alpaca news API.

    Falls back to fetch_news() stub when aiohttp is unavailable
    or API credentials are not configured.

    Usage (production):
        fetcher = NewsFetcher()
        items   = await fetcher.fetch_all()
    """

    def __init__(self, api_keys: dict = None) -> None:
        self._keys  = api_keys or {}
        self._cache: List[NewsItem] = []
        self._last_fetch: Optional[datetime] = None

    async def fetch_all(
        self,
        symbols:    List[str] = None,
        hours_back: int = 4,
    ) -> List[NewsItem]:
        """
        Fetch and score recent news.
        Returns scored NewsItem list sorted by importance.
        Falls back to empty list on any error.
        """
        symbols = symbols or ["SPY"]
        if not HAS_AIOHTTP:
            logger.debug("NewsFetcher: aiohttp not available, using stub")
            return self._stub_items(symbols)

        key = self._keys.get("alpaca") or ALPACA_API_KEY
        secret = self._keys.get("alpaca_secret") or ALPACA_SECRET_KEY
        if not key or not secret:
            return self._stub_items(symbols)

        raw = await self._fetch_alpaca(symbols, hours_back, key, secret)
        if not raw:
            return self._cache or []

        scored = _scorer.score_items(raw)
        self._cache = scored
        self._last_fetch = datetime.now(tz=timezone.utc)
        return scored

    def _stub_items(self, symbols: List[str]) -> List[NewsItem]:
        """Convert stub NewsEvent data to NewsItem format for consistency."""
        events = fetch_news(symbols[0] if symbols else "SPY")
        items  = []
        for e in events:
            item = NewsItem(
                id=str(uuid.uuid4()),
                headline=e.headline,
                summary="",
                source=e.source,
                source_tier=SourceTier.TIER_2_CURATED,
                published_at=e.timestamp,
            )
            _scorer._score_item(item)
            items.append(item)
        return items

    async def _fetch_alpaca(
        self,
        symbols:    List[str],
        hours_back: int,
        key:        str,
        secret:     str,
    ) -> List[NewsItem]:
        start   = datetime.now(tz=timezone.utc) - timedelta(hours=hours_back)
        params  = {
            "symbols": ",".join(symbols),
            "start":   start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "limit":   50,
            "sort":    "DESC",
        }
        headers = {
            "APCA-API-KEY-ID":     key,
            "APCA-API-SECRET-KEY": secret,
        }
        url = f"{ALPACA_DATA_URL}/v1beta1/news"
        try:
            async with _aiohttp.ClientSession(headers=headers) as session:
                async with session.get(
                    url, params=params,
                    timeout=_aiohttp.ClientTimeout(total=8),
                ) as resp:
                    if resp.status != 200:
                        logger.warning(f"News fetch HTTP {resp.status}")
                        return []
                    data  = await resp.json()
                    items = [self._parse(raw) for raw in data.get("news", [])]
                    return items
        except Exception as e:
            logger.error(f"News fetch error: {e}")
            return []

    def _parse(self, raw: dict) -> NewsItem:
        source  = raw.get("author", "") or raw.get("source", "unknown")
        tier    = _scorer.classify_source_tier(source)
        ts_str  = raw.get("created_at", "")
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