"""
engine/market_engine.py — top-level market data orchestrator.

Owns:
  CandleStore      — rolling candle history
  IndicatorEngine  — computes indicators + MarketState
  AlpacaFeed       — REST seed + WebSocket (lazy-imported; aiohttp optional)

Exposes:
  start()          — seed history then start live WebSocket
  stop()
  get_state()      — latest MarketState (always safe to call)
  on_event         — caller-supplied async callback for MarketEvents

Events fired:
  seed_progress    — after each timeframe finishes seeding
  readiness_change — when readiness tier advances
  candle_close     — when a candle closes on 5m, 15m, or 1h
  trend_flip       — direction change detected on key TF
  trend_weakening  — weakening flags appeared
  structure_break  — price closed through structural level
  htf_bias_flip    — daily+1h bias changed direction
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Awaitable, Callable, List, Optional

from engine.candle_store     import CandleStore, classify_session_type
from engine.indicator_engine import IndicatorEngine
from engine.models           import (
    Direction,
    MarketEvent,
    MarketState,
    Readiness,
    TrendStrength,
)

logger = logging.getLogger(__name__)

EventCallback = Callable[[MarketEvent], Awaitable[None]]


class MarketEngine:
    """
    Top-level market data and indicator orchestrator.

    Usage:
        engine = MarketEngine(on_event=handle_event)
        await engine.start()
        state = engine.get_state()
    """

    def __init__(self, on_event: Optional[EventCallback] = None) -> None:
        self.store   = CandleStore()
        self.ind_eng = IndicatorEngine(self.store)

        self._on_event  = on_event
        self._state:    Optional[MarketState] = None
        self._prev_state: Optional[MarketState] = None
        self._readiness: Readiness = Readiness.NOT_READY
        self._running   = False
        self._feed      = None   # created lazily in start()

        # Seen structure-break keys to avoid repeat events
        self._struct_break_seen: set = set()

    # ──────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────

    async def start(self) -> None:
        """Seed history, compute initial state, then start live feed."""
        from engine.alpaca_feed import AlpacaFeed

        self._running = True

        self._feed = AlpacaFeed(
            on_tick=self._handle_tick,
            on_bar=self._handle_bar,
            on_seed_complete=self._handle_seed_complete,
        )

        # Seed historical bars (1d → 1h → 15m → 5m → 1m)
        await self._feed.seed_history()

        # Initial compute after seed
        self._state = self.ind_eng.compute()
        logger.info(
            f"[R2] Initial state — readiness: {self._readiness.value}, "
            f"htf_bias: {self._state.htf_bias.value}"
        )

        # Start live WebSocket (runs until stop() called)
        await self._feed.run()

    async def stop(self) -> None:
        self._running = False
        if self._feed:
            await self._feed.stop()
        logger.info("[R2] MarketEngine stopped")

    # ──────────────────────────────────────────
    # State access
    # ──────────────────────────────────────────

    def get_state(self) -> Optional[MarketState]:
        return self._state

    def get_store(self) -> CandleStore:
        return self.store

    def get_readiness(self) -> Readiness:
        return self._readiness

    # ──────────────────────────────────────────
    # Feed callbacks
    # ──────────────────────────────────────────

    async def _handle_tick(
        self,
        price:        float,
        volume:       float,
        timestamp:    datetime,
        session_type: str = "regular",
    ) -> None:
        try:
            closed_tfs = await self.store.ingest_trade(
                price, volume, timestamp, session_type
            )
            if closed_tfs:
                await self._on_candles_closed(closed_tfs)
        except Exception as e:
            logger.error(f"[R2] Tick handler error: {e}", exc_info=True)

    async def _handle_bar(self, bar: dict, timeframe: str) -> None:
        try:
            session_type = classify_session_type(bar["timestamp"])
            await self.store.ingest_bar(bar, timeframe, session_type)
        except Exception as e:
            logger.error(f"[R2] Bar ingest error ({timeframe}): {e}", exc_info=True)

    async def _handle_seed_complete(self, timeframe: str, count: int) -> None:
        """Called after each timeframe finishes seeding. Fires readiness events."""
        new_readiness, note = self.store.get_readiness()

        if new_readiness != self._readiness:
            prev = self._readiness
            self._readiness = new_readiness
            logger.info(f"[R2] Readiness: {prev.value} → {new_readiness.value} ({note})")
            await self._fire(MarketEvent(
                event_type="readiness_change",
                timestamp=datetime.now(tz=timezone.utc),
                payload={
                    "from":       prev.value,
                    "to":         new_readiness.value,
                    "note":       note,
                    "timeframe":  timeframe,
                    "bars_seeded": count,
                },
                priority=2,
            ))

        # Recompute state after each seeded TF so readiness is current
        self._state = self.ind_eng.compute()

        await self._fire(MarketEvent(
            event_type="seed_progress",
            timestamp=datetime.now(tz=timezone.utc),
            payload={"timeframe": timeframe, "count": count, "readiness": new_readiness.value},
            priority=1,
        ))

    # ──────────────────────────────────────────
    # Post-candle-close logic
    # ──────────────────────────────────────────

    async def _on_candles_closed(self, closed_tfs: List[str]) -> None:
        self._prev_state = self._state
        self._state = self.ind_eng.compute()

        # Update readiness
        new_readiness, _ = self.store.get_readiness()
        self._readiness = new_readiness
        if self._state.session:
            self._state.session.readiness = new_readiness

        # Check for VWAP reset
        self._maybe_reset_vwap()

        # Detect and fire events
        events = self._detect_events(closed_tfs)
        for event in events:
            await self._fire(event)

    def _detect_events(self, closed_tfs: List[str]) -> List[MarketEvent]:
        events: List[MarketEvent] = []
        now = datetime.now(tz=timezone.utc)

        if not self._state or not self._prev_state:
            return events

        # ── HTF bias flip ─────────────────────
        curr_bias = self._state.htf_bias
        prev_bias = self._prev_state.htf_bias
        if curr_bias != prev_bias and curr_bias != Direction.NEUTRAL:
            events.append(MarketEvent(
                event_type="htf_bias_flip",
                timestamp=now,
                payload={"from": prev_bias.value, "to": curr_bias.value},
                priority=3,
            ))

        # ── Trend changes on key timeframes ───
        for tf in ("5m", "15m", "1h"):
            curr_t = self._state.trends.get(tf)
            prev_t = self._prev_state.trends.get(tf)
            if not curr_t or not prev_t:
                continue

            if curr_t.direction != prev_t.direction:
                events.append(MarketEvent(
                    event_type="trend_flip",
                    timestamp=now,
                    payload={
                        "timeframe": tf,
                        "from":      prev_t.direction.value,
                        "to":        curr_t.direction.value,
                    },
                    priority=2 if tf in ("15m", "1h") else 1,
                ))

            if curr_t.is_weakening and not prev_t.is_weakening:
                events.append(MarketEvent(
                    event_type="trend_weakening",
                    timestamp=now,
                    payload={"timeframe": tf, "direction": curr_t.direction.value},
                    priority=1,
                ))

            # Structure break (new, not seen before)
            break_key = f"{tf}_{curr_t.direction.value}"
            if curr_t.structure_break and break_key not in self._struct_break_seen:
                self._struct_break_seen.add(break_key)
                events.append(MarketEvent(
                    event_type="structure_break",
                    timestamp=now,
                    payload={"timeframe": tf, "direction": curr_t.direction.value},
                    priority=2,
                ))
            elif not curr_t.structure_break and break_key in self._struct_break_seen:
                self._struct_break_seen.discard(break_key)

        # ── Candle close on important TFs ─────
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

    async def _fire(self, event: MarketEvent) -> None:
        if self._on_event:
            try:
                await self._on_event(event)
            except Exception as e:
                logger.error(f"[R2] Event callback error: {e}", exc_info=True)

    def _maybe_reset_vwap(self) -> None:
        """Reset intraday VWAP at 9:30 ET each day."""
        now_utc = datetime.now(tz=timezone.utc)
        # Approximate EDT offset
        now_et  = now_utc.replace(tzinfo=None) - __import__("datetime").timedelta(hours=4)
        if now_et.hour == 9 and now_et.minute == 30 and now_et.second < 5:
            self.store.reset_vwap()
            logger.info("[R2] VWAP reset for new trading day")