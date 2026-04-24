"""
context/context_engine.py — orchestrates all external context collection.

Two interfaces:

  build_context(symbol, now) -> ContextState
    Synchronous, pure, no external deps. Uses stubs from news_fetcher and
    macro_calendar. Called by StrategyEngine Step 4b. Always returns a valid
    ContextState — never raises.

  ContextEngine (class)
    Async production engine. Runs on a refresh loop, calls real APIs.
    Falls back to ContextState.empty() on any failure.
"""

from __future__ import annotations

import asyncio
import copy
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Optional

from context.context_state import ContextState, HeadlineImpact, MarketRegime
from context.headline_scorer import compute_vix_regime, classify_sentiment, score_headline
from context.news_fetcher import NewsFetcher, fetch_news
from context.macro_calendar import MacroCalendar, get_macro_events
from engine.models import TradeState

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Standalone synchronous builder (Phase 4 spec)
# ──────────────────────────────────────────────

def build_context(
    symbol: str = "SPY",
    now:    Optional[datetime] = None,
    injected_news:   Optional[list] = None,
    injected_events: Optional[list] = None,
) -> ContextState:
    """
    Build a complete ContextState from stub data.

    Logic:
      1. Fetch news via fetch_news() (returns mock/injected data)
      2. Aggregate sentiment across all items
      3. Fetch macro events via get_macro_events()
      4. Identify active events (within MACRO_WINDOW_H hours)
      5. Set volatility_risk_level from event impact + proximity
      6. Set context_bias from aggregate sentiment + risk level
      7. Return complete ContextState

    Always returns a valid object — never raises.
    """
    if now is None:
        now = datetime.now(tz=timezone.utc)

    try:
        return _build(symbol, now, injected_news, injected_events)
    except Exception as e:
        logger.error(f"build_context failed: {e}")
        ctx = ContextState.empty()
        ctx.context_valid = False
        return ctx


MACRO_WINDOW_H = 2.0    # flag events within this many hours


def _build(
    symbol:          str,
    now:             datetime,
    injected_news:   Optional[list],
    injected_events: Optional[list],
) -> ContextState:
    # ── 1. News ───────────────────────────────
    news_events = fetch_news(symbol, injected=injected_news)

    # ── 2. Aggregate sentiment ────────────────
    scored_sentiments = [
        e.sentiment_score
        for e in news_events
        if e.sentiment_score is not None
    ]
    if scored_sentiments:
        news_sentiment = round(
            sum(scored_sentiments) / len(scored_sentiments), 3
        )
    else:
        news_sentiment = 0.0

    # ── 3. Macro events ───────────────────────
    macro_events = get_macro_events(injected=injected_events)

    # ── 4. Active (imminent) events ───────────
    horizon = now + timedelta(hours=MACRO_WINDOW_H)
    active_events = [
        e for e in macro_events
        if getattr(e, "status", "upcoming") == "upcoming"
        and now <= e.scheduled_at <= horizon
    ]

    # ── 5. Volatility risk level ──────────────
    volatility_risk_level = _compute_risk_level(active_events, news_sentiment)

    # ── 6. Context bias ───────────────────────
    context_bias = _compute_bias(news_sentiment, volatility_risk_level, active_events)

    # ── 7. Elevated risk flag ─────────────────
    elevated_risk = volatility_risk_level in ("elevated", "extreme")
    risk_note     = _build_risk_note(active_events, volatility_risk_level)

    return ContextState(
        timestamp=now,
        upcoming_events=active_events,
        recent_releases=[
            e for e in macro_events
            if getattr(e, "status", "upcoming") == "released"
        ],
        news_sentiment=news_sentiment,
        context_bias=context_bias,
        volatility_risk_level=volatility_risk_level,
        elevated_risk=elevated_risk,
        risk_note=risk_note,
        context_valid=True,
    )


def _compute_risk_level(
    active_events: list,
    news_sentiment: float,
) -> str:
    """
    Derive volatility risk level from macro events and news.

    HIGH-impact event imminent → elevated
    Two+ MEDIUM events imminent → elevated
    Extreme news sentiment (|s| > 0.7) → elevated
    Otherwise normal.
    """
    high_count   = sum(
        1 for e in active_events
        if getattr(e, "market_impact", HeadlineImpact.LOW) == HeadlineImpact.HIGH
    )
    medium_count = sum(
        1 for e in active_events
        if getattr(e, "market_impact", HeadlineImpact.LOW) == HeadlineImpact.MEDIUM
    )

    if high_count >= 1:
        return "elevated"
    if medium_count >= 2:
        return "elevated"
    if abs(news_sentiment) > 0.7:
        return "elevated"
    if abs(news_sentiment) > 0.4 or medium_count == 1:
        return "normal"
    return "normal"


def _compute_bias(
    news_sentiment:       float,
    volatility_risk_level: str,
    active_events:         list,
) -> str:
    """
    Derive context_bias from news + macro risk.

    High-risk environment → neutral (don't lean into risk)
    Strong positive sentiment → bullish
    Strong negative sentiment → bearish
    """
    if volatility_risk_level in ("elevated", "extreme"):
        return "neutral"

    sentiment_obj = classify_sentiment(news_sentiment)
    return sentiment_obj.value   # "bullish" | "bearish" | "neutral"


def _build_risk_note(active_events: list, risk_level: str) -> str:
    if not active_events and risk_level == "normal":
        return ""
    if active_events:
        first = active_events[0]
        return f"{getattr(first, 'name', 'macro event')} imminent"
    return f"risk level: {risk_level}"


# ──────────────────────────────────────────────
# ContextEngine — async production class
# ──────────────────────────────────────────────

class ContextEngine:
    """
    Async context engine with refresh loop.
    Used in production; not required by StrategyEngine (which uses build_context).

    Usage:
        engine = ContextEngine(api_keys={...})
        await engine.start()
        ctx = engine.get_context()
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
            items           = await self._news.fetch_all()
            events, vix, fed = await self._macro.fetch_all()
        except Exception as e:
            logger.error(f"Context refresh failed: {e}")
            if self._state is None:
                self._state = ContextState.empty()
            return

        elevated_risk, risk_note = self._macro.assess_risk(events, vix)
        market_regime  = self._macro.classify_regime(vix, events)
        vix_regime     = compute_vix_regime(vix) if vix else "unknown"
        news_sentiment = (
            round(sum(i.sentiment for i in items) / len(items), 3)
            if items else 0.0
        )
        context_bias = _compute_bias(
            news_sentiment,
            vix_regime if vix_regime in ("elevated", "extreme") else "normal",
            [],
        )

        self._state = ContextState(
            timestamp=now,
            upcoming_events=[e for e in events if e.status == "upcoming"],
            recent_releases=[e for e in events if e.status == "released"],
            fed=fed,
            vix_level=vix,
            vix_regime=vix_regime,
            market_regime=market_regime,
            elevated_risk=elevated_risk,
            risk_note=risk_note,
            news_sentiment=news_sentiment,
            context_bias=context_bias,
            volatility_risk_level=vix_regime if vix else "normal",
            context_valid=True,
        )
        logger.debug(f"Context refreshed: {len(items)} news, VIX={vix}")

    def _enrich_for_trade(
        self, state: ContextState, trade: TradeState
    ) -> ContextState:
        return copy.copy(state)

    @staticmethod
    def _extract_keywords(thesis: str) -> List[str]:
        if not thesis:
            return []
        stopwords = {
            "with","from","that","this","have","been","will",
            "above","below","level","price","entry","setup",
        }
        words = [w.strip(".,;:()") for w in thesis.split() if len(w) > 4]
        return [w for w in words if w.lower() not in stopwords]