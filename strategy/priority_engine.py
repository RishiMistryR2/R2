"""
Priority Engine.

Transforms raw MarketEvents into PrioritizedEvents with:
  1. A 0–100 score and named priority level
  2. A one-line human headline (for CLI display)
  3. A detail string (for AI context)
  4. Flags: should_trigger_ai, should_alert
  5. Coalescing of related events within a short window
  6. Cooldown tracking to prevent AI spam

Priority levels:
  CRITICAL (80–100): stop hit, major flip, regime change → interrupt AI immediately
  HIGH     (60–79):  A+ setup, structure break → trigger proactive commentary
  MEDIUM   (40–59):  B setup, trend weakening → alert user, queue for update
  LOW      (20–39):  routine candle close, minor change → log only
  INFO     (0–19):   background → completely silent

Design rule: every AI trigger must be justified by a HIGH or CRITICAL event.
Medium events accumulate until the next scheduled update, not per-event.
"""

from __future__ import annotations
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

from engine.models import (
    EventPriority,
    MarketEvent,
    PrioritizedEvent,
)


# ──────────────────────────────────────────────
# Base score table
# ──────────────────────────────────────────────

EVENT_BASE_SCORES: Dict[str, int] = {
    # CRITICAL tier (80+)
    "exit_triggered":     92,
    "stop_loss_hit":      95,
    "htf_bias_flip":      85,
    "trend_flip_1h":      82,
    "trend_flip_15m":     78,   # → HIGH boundary
    "regime_change_to_chop": 80,
    "regime_change":      75,   # → HIGH

    # HIGH tier (60–79)
    "setup_formed_A+":    72,
    "setup_formed_B":     62,
    "structure_break_15m": 68,
    "structure_break_1h": 75,
    "narrative_change":   65,
    "trend_flip_5m":      58,   # → MEDIUM boundary

    # MEDIUM tier (40–59)
    "setup_formed_C":     48,
    "trend_weakening_1h": 52,
    "trend_weakening_15m": 46,
    "trend_weakening_5m": 40,
    "candle_close_1h":    35,   # → LOW
    "candle_close_15m":   25,
    "candle_close_5m":    15,
    "candle_close_1m":    5,
}

# AI trigger threshold (must be >= this)
AI_TRIGGER_MIN_SCORE = 60

# Alert threshold (CLI flash, no AI)
ALERT_MIN_SCORE = 40

# Cooldown per event type (seconds)
COOLDOWN_SECONDS: Dict[str, int] = {
    "setup_formed":        120,   # 2 min — avoid re-alerting same setup
    "trend_weakening":      90,
    "trend_flip":           60,
    "narrative_change":     60,
    "structure_break":      60,
    "candle_close":          0,   # no cooldown — always low priority anyway
    "_default":             45,
}

# Critical events bypass cooldown entirely
COOLDOWN_BYPASS = {"exit_triggered", "stop_loss_hit", "htf_bias_flip"}

# Coalesce window — events within this many seconds may be merged
COALESCE_WINDOW_SECS = 5


class PriorityEngine:
    """
    Scores, coalesces, and routes MarketEvents.

    Usage:
        engine = PriorityEngine()
        prioritized = engine.process(raw_events)
        for pe in prioritized:
            if pe.should_trigger_ai:
                await claude_client.proactive_commentary(context)
    """

    def __init__(self) -> None:
        # last trigger time per event type (for cooldown)
        self._last_trigger: Dict[str, datetime] = {}

    def process(self, events: List[MarketEvent]) -> List[PrioritizedEvent]:
        """
        Process a batch of raw events into prioritized events.
        Coalesces related events, applies cooldown, returns sorted list.
        """
        if not events:
            return []

        now = datetime.now(tz=timezone.utc)

        # Step 1: Score each event individually
        scored = [self._score_event(e, now) for e in events]

        # Step 2: Coalesce related events fired close together
        coalesced = self._coalesce(scored, now)

        # Step 3: Apply cooldown — determine which trigger AI
        result = []
        for pe in coalesced:
            can_trigger = self._check_cooldown(pe, now)
            pe.should_trigger_ai = (
                can_trigger
                and pe.score >= AI_TRIGGER_MIN_SCORE
            )
            pe.should_alert = pe.score >= ALERT_MIN_SCORE
            if pe.should_trigger_ai:
                self._last_trigger[pe.event_type] = now
            result.append(pe)

        # Sort: highest priority first
        result.sort(key=lambda pe: -pe.score)
        return result

    def reset_cooldown(self, event_type: str) -> None:
        """Manually clear cooldown for an event type."""
        self._last_trigger.pop(event_type, None)

    # ──────────────────────────────────────────
    # Scoring
    # ──────────────────────────────────────────

    def _score_event(self, event: MarketEvent, now: datetime) -> PrioritizedEvent:
        """Score a single MarketEvent."""
        score = self._lookup_score(event)
        priority = _score_to_priority(score)
        headline, detail = _build_messages(event)

        return PrioritizedEvent(
            priority=priority,
            score=score,
            event_type=event.event_type,
            headline=headline,
            detail=detail,
            should_trigger_ai=False,   # set after cooldown check
            should_alert=False,
            original_events=[event.event_type],
            timestamp=event.timestamp or now,
        )

    def _lookup_score(self, event: MarketEvent) -> int:
        """Look up base score, with payload-based adjustments."""
        base_key = event.event_type
        payload  = event.payload or {}

        # Event type + payload refinements
        if event.event_type == "setup_formed":
            score = payload.get("score", 0)
            if score >= 80:   return EVENT_BASE_SCORES["setup_formed_A+"]
            if score >= 60:   return EVENT_BASE_SCORES["setup_formed_B"]
            return EVENT_BASE_SCORES["setup_formed_C"]

        if event.event_type == "trend_flip":
            tf = payload.get("timeframe", "5m")
            key = f"trend_flip_{tf}"
            return EVENT_BASE_SCORES.get(key, EVENT_BASE_SCORES.get("trend_flip_5m", 55))

        if event.event_type == "structure_break":
            tf = payload.get("timeframe", "15m")
            key = f"structure_break_{tf}"
            return EVENT_BASE_SCORES.get(key, 65)

        if event.event_type == "trend_weakening":
            tf = payload.get("timeframe", "5m")
            key = f"trend_weakening_{tf}"
            return EVENT_BASE_SCORES.get(key, 40)

        if event.event_type == "candle_close":
            tf = payload.get("timeframe", "1m")
            key = f"candle_close_{tf}"
            return EVENT_BASE_SCORES.get(key, 10)

        if event.event_type == "regime_change":
            to_regime = payload.get("to", "")
            if "chop" in to_regime:
                return EVENT_BASE_SCORES["regime_change_to_chop"]
            return EVENT_BASE_SCORES["regime_change"]

        return EVENT_BASE_SCORES.get(base_key, 30)

    # ──────────────────────────────────────────
    # Coalescing
    # ──────────────────────────────────────────

    def _coalesce(
        self,
        events: List[PrioritizedEvent],
        now:    datetime,
    ) -> List[PrioritizedEvent]:
        """
        Merge related events fired within COALESCE_WINDOW_SECS of each other.

        Coalesce rules:
          trend_weakening + structure_break → one HIGH event
          multiple candle_close (same TF) → single LOW event
          setup_formed + narrative_change → keep both (different decisions)
        """
        if len(events) <= 1:
            return events

        result = []
        used = set()

        for i, e1 in enumerate(events):
            if i in used:
                continue
            merge_candidates = []
            for j, e2 in enumerate(events):
                if j <= i or j in used:
                    continue
                if _should_coalesce(e1, e2):
                    merge_candidates.append(j)

            if merge_candidates:
                merged = self._merge_events(e1, [events[j] for j in merge_candidates])
                result.append(merged)
                used.add(i)
                used.update(merge_candidates)
            else:
                result.append(e1)
                used.add(i)

        return result

    def _merge_events(
        self,
        primary:   PrioritizedEvent,
        others:    List[PrioritizedEvent],
    ) -> PrioritizedEvent:
        """Merge multiple events into one, keeping the highest score."""
        all_events = [primary] + others
        best = max(all_events, key=lambda e: e.score)
        all_types = [e.event_type for e in all_events]

        return PrioritizedEvent(
            priority=best.priority,
            score=min(100, best.score + 5),   # slight bonus for coalescing
            event_type=best.event_type,
            headline=f"{best.headline} (+{len(others)} related)",
            detail=best.detail + " | " + " | ".join(e.headline for e in others),
            should_trigger_ai=False,
            should_alert=False,
            original_events=all_types,
            timestamp=best.timestamp,
        )

    # ──────────────────────────────────────────
    # Cooldown
    # ──────────────────────────────────────────

    def _check_cooldown(self, pe: PrioritizedEvent, now: datetime) -> bool:
        """
        Returns True if this event is allowed to trigger AI (not in cooldown).
        CRITICAL events always bypass cooldown.
        """
        if pe.event_type in COOLDOWN_BYPASS or pe.priority == EventPriority.CRITICAL:
            return True

        cooldown_secs = COOLDOWN_SECONDS.get(
            pe.event_type,
            COOLDOWN_SECONDS["_default"]
        )
        if cooldown_secs == 0:
            return False   # no cooldown but also no trigger (e.g. candle_close)

        last = self._last_trigger.get(pe.event_type)
        if last is None:
            return True
        return (now - last).total_seconds() >= cooldown_secs


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _score_to_priority(score: int) -> EventPriority:
    if score >= 80:  return EventPriority.CRITICAL
    if score >= 60:  return EventPriority.HIGH
    if score >= 40:  return EventPriority.MEDIUM
    if score >= 20:  return EventPriority.LOW
    return EventPriority.INFO


def _build_messages(event: MarketEvent) -> Tuple[str, str]:
    """Build a (headline, detail) pair from a raw MarketEvent."""
    p = event.payload or {}
    t = event.event_type

    if t == "setup_formed":
        dir_ = p.get("direction", "unknown").upper()
        score = p.get("score", 0)
        grade = p.get("grade", "")
        stype = p.get("setup_type", "")
        headline = f"{dir_} setup — score {score:.0f} ({grade}) [{stype}]"
        detail   = f"Setup formed: {p.get('reason', '')}. Score breakdown: {score:.0f}/100."

    elif t == "trend_flip":
        tf   = p.get("timeframe", "?")
        frm  = p.get("from", "?")
        to   = p.get("to", "?")
        headline = f"{tf} trend flipped {frm} → {to}"
        detail   = f"Trend flip on {tf}: {p.get('reason', frm + ' to ' + to)}"

    elif t == "exit_triggered":
        reason = p.get("reason", "unknown")
        pnl    = p.get("pnl_pct", 0)
        urgency = p.get("urgency", "normal")
        headline = f"EXIT TRIGGERED [{urgency.upper()}] — {reason} | P&L {pnl:+.2f}%"
        detail   = p.get("detail", headline)

    elif t == "narrative_change":
        frm = p.get("from", "?")
        to  = p.get("to", "?")
        headline = f"Market narrative: {frm} → {to}"
        detail   = p.get("description", headline)

    elif t == "regime_change":
        frm = p.get("from", "?")
        to  = p.get("to", "?")
        headline = f"Regime shift: {frm} → {to}"
        detail   = f"Market regime changed from {frm} to {to}."

    elif t == "htf_bias_flip":
        frm = p.get("from", "?")
        to  = p.get("to", "?")
        headline = f"HTF bias flipped: {frm} → {to}"
        detail   = headline

    elif t == "trend_weakening":
        tf   = p.get("timeframe", "?")
        dir_ = p.get("direction", "?")
        headline = f"{tf} {dir_} trend weakening"
        detail   = f"Weakening signals on {tf}: {p.get('reason', '')}"

    elif t == "structure_break":
        tf   = p.get("timeframe", "?")
        dir_ = p.get("direction", "?")
        headline = f"{tf} structure break ({dir_})"
        detail   = headline

    elif t == "candle_close":
        tf = p.get("timeframe", "?")
        headline = f"{tf} candle closed"
        detail   = headline

    else:
        headline = t.replace("_", " ").title()
        detail   = str(p)

    return headline, detail


def _should_coalesce(e1: PrioritizedEvent, e2: PrioritizedEvent) -> bool:
    """True if two events should be merged into one."""
    # Same type on the same timeframe
    if (e1.event_type == e2.event_type
            and e1.event_type == "candle_close"):
        return True
    # Complementary signals (weakening + structure_break = same story)
    pairs = {
        ("trend_weakening", "structure_break"),
        ("narrative_change", "regime_change"),
    }
    pair = tuple(sorted([e1.event_type, e2.event_type]))
    return pair in pairs
