"""
ContextEngine: orchestrates all external context collection.

Runs on a configurable refresh interval (default: 5 minutes).
Each refresh assembles a fresh ContextState passed into the AI context.
Completely decoupled from the market engine — runs on its own schedule.
"""

from __future__ import annotations

import asyncio
import copy
import logging
from datetime import datetime, timezone
from typing import List, Optional

from context.context_state import ContextState
from context.headline_scorer import compute_vix_regime
from context.news_fetcher import NewsFetcher
from context.macro_calendar import MacroCalendar
from engine.models import TradeState

logger = logging.getLogger(__name__)


class ContextEngine:
    """
    Manages external context and maintains a current ContextState.

    Usage:
        engine = ContextEngine(api_keys={...})
        await engine.start()
        ctx = engine.get_context(active_trade=state.active_trade)
    """

    def __init__(self, api_keys: dict = None, refresh_secs: int = 300) -> None:
        self._keys    = api_keys or {}
        self._refresh = refresh_secs
        self._news    = NewsFetcher(api_keys=self._keys)
        self._macro   = MacroCalendar(api_keys=self._keys)
        self._state:  Optional[ContextState] = None
        self._running = False

    async def start(self) -> None:
        self._running = True
        await self._do_refresh()
        asyncio.create_task(self._loop())
        logger.info(f"ContextEngine started (refresh every {self._refresh}s)")

    async def stop(self) -> None:
        self._running = False

    def get_context(self, active_trade: Optional[TradeState] = None) -> ContextState:
        if self._state is None:
            return ContextState.empty()
        if active_trade:
            return self._enrich_for_trade(self._state, active_trade)
        return self._state

    async def _loop(self) -> None:
        while self._running:
            await asyncio.sleep(self._refresh)
            if self._running:
                await self._do_refresh()

    async def _do_refresh(self) -> None:
        logger.debug("ContextEngine: refreshing...")
        now = datetime.now(tz=timezone.utc)
        try:
            headlines, (events, vix, fed) = await asyncio.gather(
                self._news.fetch_all(),
                self._macro.fetch_all(),
            )
        except Exception as e:
            logger.error(f"Context refresh failed: {e}")
            if self._state is None:
                self._state = ContextState.empty()
            return

        elevated_risk, risk_note = self._macro.assess_risk(events, vix)
        market_regime = self._macro.classify_regime(vix, events)
        vix_regime    = compute_vix_regime(vix) if vix else "unknown"

        self._state = ContextState(
            timestamp=now,
            top_headlines=headlines,
            trade_relevant_headlines=[h for h in headlines if h.trade_relevant],
            upcoming_events=[e for e in events if e.status == "upcoming"],
            recent_releases=[e for e in events if e.status == "released"],
            fed=fed,
            vix_level=vix,
            vix_regime=vix_regime,
            market_regime=market_regime,
            elevated_risk=elevated_risk,
            risk_note=risk_note,
            context_valid=True,
        )
        logger.debug(f"Context refreshed: {len(headlines)} headlines, VIX={vix}")

    def _enrich_for_trade(self, state: ContextState, trade: TradeState) -> ContextState:
        enriched = copy.copy(state)
        keywords = self._extract_keywords(trade.thesis) + ["SPY", "S&P 500"]
        enriched.trade_relevant_headlines = [
            h for h in state.top_headlines
            if any(kw.lower() in h.headline.lower() for kw in keywords)
        ]
        return enriched

    @staticmethod
    def _extract_keywords(thesis: str) -> List[str]:
        if not thesis:
            return []
        stopwords = {"with","from","that","this","have","been","will",
                     "above","below","level","price","entry","setup"}
        words = [w.strip(".,;:()") for w in thesis.split() if len(w) > 4]
        return [w for w in words if w.lower() not in stopwords]
