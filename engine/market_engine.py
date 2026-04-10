"""
MarketEngine: the beating heart of the system.

Responsibilities:
- Owns the CandleStore and IndicatorEngine
- Receives ticks from AlpacaFeed
- Triggers indicator recompute on candle close
- Detects significant state changes
- Fires MarketEvents to the event bus
- Keeps a rolling snapshot of MarketState

Everything downstream (strategy, AI, UI) reads from this engine.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone, time as dt_time
from typing import Callable, Awaitable, Optional, List

from engine.candle_store     import CandleStore
from engine.indicator_engine import IndicatorEngine
from engine.alpaca_feed      import AlpacaFeed
from engine.models           import MarketEvent, MarketState, Direction, TrendStrength

logger = logging.getLogger(__name__)

EventCallback = Callable[[MarketEvent], Awaitable[None]]

# Market hours (Eastern → we'll check UTC offset at runtime)
MARKET_OPEN_ET  = dt_time(9, 30)
MARKET_CLOSE_ET = dt_time(16, 0)


class MarketEngine:
    """
    The top-level orchestrator.

    Usage:
        engine = MarketEngine(on_event=my_handler)
        await engine.start()          # seeds history, starts live feed
        state = engine.get_state()    # always returns latest MarketState
    """

    def __init__(self, on_event: Optional[EventCallback] = None) -> None:
        self.store   = CandleStore()
        self.ind_eng = IndicatorEngine(self.store)
        self._on_event = on_event

        self._state: Optional[MarketState] = None
        self._prev_state: Optional[MarketState] = None

        self._feed = AlpacaFeed(
            on_tick=self._handle_tick,
            on_bar=self._handle_bar,
        )

        # Track significant state for event detection
        self._last_htf_bias: Optional[Direction] = None
        self._structure_breaks_seen: set = set()

    # ──────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────

    async def start(self) -> None:
        """
        1. Seed historical candle data.
        2. Compute initial state.
        3. Start live WebSocket feed.
        """
        logger.info("MarketEngine starting...")

        # Seed history (REST)
        await self._feed.seed_history()

        # Initial compute
        self._state = self.ind_eng.compute()
        logger.info(f"Initial state computed. HTF bias: {self._state.htf_bias.value}")

        # Run WebSocket (blocking — wrap in task if needed)
        await self._feed.run()

    async def stop(self) -> None:
        await self._feed.stop()
        logger.info("MarketEngine stopped.")

    # ──────────────────────────────────────────
    # State access
    # ──────────────────────────────────────────

    def get_state(self) -> Optional[MarketState]:
        """Thread-safe read of latest MarketState."""
        return self._state

    def get_candle_store(self) -> CandleStore:
        return self.store

    # ──────────────────────────────────────────
    # Tick / bar handlers (called from feed)
    # ──────────────────────────────────────────

    async def _handle_tick(self, price: float, volume: float, timestamp: datetime) -> None:
        """Process a single trade tick."""
        try:
            closed_tfs = await self.store.ingest_trade(price, volume, timestamp)

            if closed_tfs:
                # One or more candles just closed — recompute everything
                await self._on_candles_closed(closed_tfs)

        except Exception as e:
            logger.error(f"Error in tick handler: {e}", exc_info=True)

    async def _handle_bar(self, bar: dict, timeframe: str) -> None:
        """Process a pre-built bar (historical seed or bar stream)."""
        try:
            await self.store.ingest_bar(bar, timeframe)
        except Exception as e:
            logger.error(f"Error in bar handler: {e}", exc_info=True)

    # ──────────────────────────────────────────
    # Post-candle-close logic
    # ──────────────────────────────────────────

    async def _on_candles_closed(self, closed_tfs: List[str]) -> None:
        """Called whenever one or more candles just closed."""
        self._prev_state = self._state
        self._state = self.ind_eng.compute()

        # Check for VWAP reset at market open
        self._maybe_reset_vwap()

        # Detect and fire events
        events = self._detect_events(closed_tfs)
        for event in events:
            logger.info(f"Event: {event.event_type} | {event.payload}")
            if self._on_event:
                await self._on_event(event)

    def _detect_events(self, closed_tfs: List[str]) -> List[MarketEvent]:
        """
        Compare current state to previous state.
        Fire events when significant changes occur.
        """
        events = []
        now = datetime.now(tz=timezone.utc)

        if not self._state or not self._prev_state:
            return events

        # ── HTF bias flip ──────────────────────
        curr_bias = self._state.htf_bias
        prev_bias = self._prev_state.htf_bias
        if curr_bias != prev_bias and curr_bias != Direction.NEUTRAL:
            events.append(MarketEvent(
                event_type="htf_bias_flip",
                timestamp=now,
                payload={
                    "from": prev_bias.value,
                    "to":   curr_bias.value,
                },
                priority=3,
            ))
            self._last_htf_bias = curr_bias

        # ── Trend flips on key timeframes ──────
        for tf in ("5m", "15m", "1h"):
            curr_trend = self._state.trends.get(tf)
            prev_trend = self._prev_state.trends.get(tf)
            if curr_trend and prev_trend:
                if curr_trend.direction != prev_trend.direction:
                    events.append(MarketEvent(
                        event_type="trend_flip",
                        timestamp=now,
                        payload={
                            "timeframe": tf,
                            "from":      prev_trend.direction.value,
                            "to":        curr_trend.direction.value,
                        },
                        priority=2 if tf in ("15m", "1h") else 1,
                    ))

                # Trend weakening
                if curr_trend.is_weakening and not prev_trend.is_weakening:
                    events.append(MarketEvent(
                        event_type="trend_weakening",
                        timestamp=now,
                        payload={"timeframe": tf, "direction": curr_trend.direction.value},
                        priority=1,
                    ))

                # Structure break (new)
                break_key = f"{tf}_{curr_trend.direction.value}"
                if curr_trend.structure_break and break_key not in self._structure_breaks_seen:
                    self._structure_breaks_seen.add(break_key)
                    events.append(MarketEvent(
                        event_type="structure_break",
                        timestamp=now,
                        payload={"timeframe": tf, "direction": curr_trend.direction.value},
                        priority=2,
                    ))
                elif not curr_trend.structure_break and break_key in self._structure_breaks_seen:
                    self._structure_breaks_seen.discard(break_key)

        # ── Candle close on important TFs ──────
        for tf in closed_tfs:
            if tf in ("5m", "15m", "1h"):
                events.append(MarketEvent(
                    event_type="candle_close",
                    timestamp=now,
                    payload={"timeframe": tf},
                    priority=1,
                ))

        return events

    # ──────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────

    def _maybe_reset_vwap(self) -> None:
        """Reset intraday VWAP at the start of each trading day."""
        now_et = datetime.now()  # simplified — in prod, use pytz for ET
        if now_et.hour == 9 and now_et.minute == 30 and now_et.second < 5:
            self.store.reset_vwap()
            logger.info("VWAP reset for new trading day.")
