"""
Macro Calendar.

Fetches and maintains the economic event calendar:
  - Upcoming scheduled events (CPI, NFP, Fed meetings, earnings)
  - Recently released data with actual vs expected
  - Current VIX level
  - Fed stance snapshot

The system works without this data — if fetch fails, ContextState
flags elevated_risk conservatively and the AI notes data limitations.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Tuple

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

from context.context_state import (
    FedContext,
    HeadlineImpact,
    MacroEvent,
    MacroEventType,
    MarketRegime,
)

logger = logging.getLogger(__name__)
FETCH_TIMEOUT = 8


class MacroCalendar:
    """
    Maintains the macro event calendar and VIX level.

    Usage:
        cal = MacroCalendar(api_keys={"tradier": "...", "finnhub": "..."})
        events, vix, fed = await cal.fetch_all()
    """

    def __init__(self, api_keys: dict = None) -> None:
        self._keys = api_keys or {}

    async def fetch_all(self) -> Tuple[List[MacroEvent], Optional[float], Optional[FedContext]]:
        """
        Returns (events, vix_level, fed_context) — any component may be None on failure.
        """
        vix_result, events_result = await asyncio.gather(
            self._fetch_vix(),
            self._fetch_upcoming_events(),
            return_exceptions=True,
        )
        vix    = vix_result    if not isinstance(vix_result, Exception)    else None
        events = events_result if not isinstance(events_result, Exception) else []
        fed    = self._build_fed_context(events)
        return events, vix, fed

    def assess_risk(
        self,
        events:   List[MacroEvent],
        vix:      Optional[float],
        window_h: float = 2.0,
    ) -> Tuple[bool, str]:
        """Returns (elevated_risk, reason_note)."""
        now     = datetime.now(tz=timezone.utc)
        horizon = now + timedelta(hours=window_h)

        for event in events:
            if (event.status == "upcoming"
                    and event.market_impact == HeadlineImpact.HIGH
                    and now <= event.scheduled_at <= horizon):
                delta_m = int((event.scheduled_at - now).total_seconds() / 60)
                return True, f"{event.name} in ~{delta_m} min"

        if vix and vix >= 25:
            label = "elevated" if vix < 30 else "extreme"
            return True, f"VIX {vix:.1f} ({label})"

        return False, ""

    def classify_regime(
        self,
        vix:    Optional[float],
        events: List[MacroEvent],
    ) -> MarketRegime:
        if vix is None:
            return MarketRegime.NEUTRAL
        if vix > 25:
            return MarketRegime.RISK_OFF
        if vix < 14:
            recent_miss = any(
                e.status == "released"
                and e.surprise_dir == "miss"
                and e.market_impact == HeadlineImpact.HIGH
                for e in events
            )
            return MarketRegime.NEUTRAL if recent_miss else MarketRegime.RISK_ON
        return MarketRegime.NEUTRAL

    # ──────────────────────────────────────────
    # Fetchers
    # ──────────────────────────────────────────

    async def _fetch_vix(self) -> Optional[float]:
        """VIX via Yahoo Finance (no key required)."""
        if not HAS_AIOHTTP:
            return None
        url = "https://query1.finance.yahoo.com/v8/finance/chart/%5EVIX"
        try:
            async with aiohttp.ClientSession(headers={"User-Agent": "Mozilla/5.0"}) as s:
                async with s.get(url, timeout=aiohttp.ClientTimeout(total=FETCH_TIMEOUT)) as r:
                    if r.status != 200:
                        return None
                    data = await r.json()
                    price = (data.get("chart", {})
                               .get("result", [{}])[0]
                               .get("meta", {})
                               .get("regularMarketPrice"))
                    return float(price) if price else None
        except Exception as e:
            logger.debug(f"VIX fetch failed: {e}")
            return None

    async def _fetch_upcoming_events(self) -> List[MacroEvent]:
        """
        Stub — returns empty list in v1.
        Production: connect to Econoday or Trading Economics API.
        The ContextEngine handles the empty case gracefully.
        """
        return []

    def _build_fed_context(self, events: List[MacroEvent]) -> FedContext:
        fed = FedContext(last_updated=datetime.now(tz=timezone.utc))
        for event in events:
            if event.event_type == MacroEventType.FED_DECISION and event.status == "released":
                if event.actual is not None:
                    fed.current_rate = event.actual
                    break
        return fed


# ──────────────────────────────────────────────
# Standalone stub (synchronous, no external deps)
# ──────────────────────────────────────────────

# Module-level injectable for testing
_injected_events: list = []


def inject_macro_events(events: list) -> None:
    """Override events returned by get_macro_events(). Pass [] to clear."""
    global _injected_events
    _injected_events = events


def get_macro_events(
    injected: list = None,
) -> list:
    """
    Return upcoming MacroEvent objects.

    Priority:
      1. `injected` argument  (per-call override)
      2. `_injected_events`   (module-level override)
      3. empty list           (safe default — no events scheduled)

    In production this will call _fetch_upcoming_events() asynchronously.
    For Phase 4 the synchronous stub is sufficient for all logic and tests.
    """
    if injected is not None:
        return list(injected)
    if _injected_events:
        return list(_injected_events)
    return []