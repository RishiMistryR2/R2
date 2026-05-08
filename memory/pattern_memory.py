"""
memory/pattern_memory.py — pattern tracking for R2.

Two conceptually distinct pattern types live here:

  BehaviorPattern   — recurring user behaviors and setup performance observations
                      (repeated mistakes, winning / failing setup + regime combos,
                      HTF alignment tendencies)

  MarketPattern     — recurring technical conditions R2 *observes* on the chart
                      (bull flag, VWAP pullback, failed breakout, etc.)

IMPORTANT CONSTRAINTS
─────────────────────
  • Advisory context only — patterns never bypass existing entry / exit gates.
  • No auto-trading because a pattern fires.
  • Pattern recognition is a memory structure; detection logic lives in Phase 7+.
  • is_surfaced prevents a pattern from showing up in briefings until
    it meets both the minimum occurrence and confidence thresholds.

Learning loop for MarketPatterns
──────────────────────────────────
  1. Detection:  record_market_pattern(type, direction, tf, ...)
                 → creates or updates a MarketPattern with pending observation

  2. Outcome:    update_market_pattern_outcome(pattern_id, trade_id, pnl_pct, ...)
                 → attaches the trade result to the matching pending observation
                    and re-derives aggregate stats (win_rate, avg_return)

Public API (PatternMemory class):
  .record_market_pattern(...)         → MarketPatternObservation
  .update_market_pattern_outcome(...) → bool
  .record_behavior(...)               → BehaviorPattern
  .update_behavior_from_trade(trade)
  .get_surfaced_market()              → List[MarketPattern]
  .get_surfaced_behavior()            → List[BehaviorPattern]
  .get_market_pattern(pattern_id)     → Optional[MarketPattern]
  .get_mistakes()                     → List[BehaviorPattern]
  .save() / (auto-load on __init__)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from config.settings import (
    MEMORY_DIR,
    MEMORY_SCHEMA_VERSION,
    PATTERN_CONFIDENCE_THRESHOLD,
    PATTERN_MIN_OCCURRENCES,
)
from memory.persistence import load, save
from memory.trade_memory import TradeRecord, OUTCOME_WIN

PATTERN_FILE = Path(MEMORY_DIR) / "patterns.json"


# ══════════════════════════════════════════════
# MarketPattern — technical setup recognition
# ══════════════════════════════════════════════

class MarketPatternType:
    """
    Named constants for known technical market patterns.

    Detection logic will be added in Phase 7+.
    These constants serve as the stable identity keys used in PatternMemory.
    Using string constants (not an Enum) so Phase 7+ can add patterns without
    breaking any serialised data that references existing values.
    """
    # Continuation patterns
    BULL_FLAG               = "bull_flag"
    BEAR_FLAG               = "bear_flag"
    HIGHER_LOW_CONTINUATION = "higher_low_continuation"
    LOWER_HIGH_CONTINUATION = "lower_high_continuation"

    # Breakout / breakdown
    BREAKOUT         = "breakout"
    BREAKDOWN        = "breakdown"
    BREAKOUT_RETEST  = "breakout_retest"
    FAILED_BREAKOUT  = "failed_breakout"
    TRAP             = "trap"           # false breakout / bull trap / bear trap

    # Mean-reversion / pullback
    PULLBACK_TO_VWAP = "pullback_to_vwap"
    PULLBACK_TO_EMA  = "pullback_to_ema"
    RANGE_REJECTION  = "range_rejection"

    # Gap patterns
    GAP_AND_GO       = "gap_and_go"
    GAP_FILL_ATTEMPT = "gap_fill_attempt"

    # Exhaustion
    TREND_EXHAUSTION = "trend_exhaustion"

    # Generic bucket for anything not yet classified
    CUSTOM           = "custom"

    ALL: List[str] = [
        BULL_FLAG, BEAR_FLAG, HIGHER_LOW_CONTINUATION, LOWER_HIGH_CONTINUATION,
        BREAKOUT, BREAKDOWN, BREAKOUT_RETEST, FAILED_BREAKOUT, TRAP,
        PULLBACK_TO_VWAP, PULLBACK_TO_EMA, RANGE_REJECTION,
        GAP_AND_GO, GAP_FILL_ATTEMPT, TREND_EXHAUSTION, CUSTOM,
    ]


@dataclass
class MarketPatternObservation:
    """
    A single occurrence of a detected market pattern.
    Outcome is filled in later when the associated trade closes.
    """
    observation_id: str
    detected_at:    str            # ISO timestamp
    trade_id:       Optional[str]  = None
    pnl_pct:        Optional[float] = None
    outcome:        Optional[str]  = None   # "win" | "loss" | "breakeven"
    notes:          str            = ""

    def to_dict(self) -> dict:
        return {
            "observation_id": self.observation_id,
            "detected_at":    self.detected_at,
            "trade_id":       self.trade_id,
            "pnl_pct":        self.pnl_pct,
            "outcome":        self.outcome,
            "notes":          self.notes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MarketPatternObservation":
        return cls(
            observation_id = d.get("observation_id", ""),
            detected_at    = d.get("detected_at", ""),
            trade_id       = d.get("trade_id"),
            pnl_pct        = d.get("pnl_pct"),
            outcome        = d.get("outcome"),
            notes          = d.get("notes", ""),
        )


@dataclass
class MarketPattern:
    """
    Aggregated statistics for a recurring technical market pattern.

    Fields updated incrementally as observations and outcomes arrive.
    Never used to auto-trigger trades — advisory context only.
    """
    pattern_id:   str    # unique key: "type__direction__timeframe"
    pattern_type: str    # MarketPatternType constant
    direction:    str    # "bullish" | "bearish" | "neutral"
    timeframe:    str    # "1m" | "5m" | "15m" | "1h" | "any"

    description:      str   = ""
    occurrence_count: int   = 0
    resolved_count:   int   = 0      # observations with known outcome
    win_count:        int   = 0
    win_rate:         float = 0.0
    avg_return:       float = 0.0    # average pnl_pct over resolved trades
    confidence_score: float = 0.0   # mirrors win_rate once ≥ min occurrences

    first_seen:       str       = ""
    last_seen:        str       = ""
    example_trade_ids: List[str] = field(default_factory=list)
    pending_observations: List[MarketPatternObservation] = field(default_factory=list)

    notes:       str  = ""
    is_surfaced: bool = False    # True when meets min-occurrence + confidence

    def record_observation(
        self,
        trade_id: Optional[str] = None,
        notes:    str           = "",
    ) -> MarketPatternObservation:
        """Register a new sighting.  Returns the observation for later outcome update."""
        import uuid
        now = datetime.now(tz=timezone.utc).isoformat()
        obs = MarketPatternObservation(
            observation_id = str(uuid.uuid4())[:8],
            detected_at    = now,
            trade_id       = trade_id,
            notes          = notes,
        )
        self.pending_observations.append(obs)
        self.occurrence_count += 1
        if not self.first_seen:
            self.first_seen = now
        self.last_seen = now
        if trade_id and trade_id not in self.example_trade_ids:
            self.example_trade_ids = (self.example_trade_ids + [trade_id])[-10:]
        self._refresh_surfaced()
        return obs

    def record_outcome(
        self,
        observation_id: str,
        pnl_pct:        float,
        outcome:        str,
        trade_id:       Optional[str] = None,
    ) -> bool:
        """
        Attach a trade outcome to a pending observation and recompute stats.
        Matches by observation_id first; falls back to trade_id.
        Returns True if a matching observation was found and updated.
        """
        for obs in self.pending_observations:
            if obs.observation_id == observation_id or (
                trade_id and obs.trade_id == trade_id
            ):
                obs.pnl_pct  = pnl_pct
                obs.outcome  = outcome
                obs.trade_id = obs.trade_id or trade_id
                self._recompute_stats()
                return True
        return False

    def _recompute_stats(self) -> None:
        resolved = [o for o in self.pending_observations if o.outcome is not None]
        self.resolved_count = len(resolved)
        self.win_count = sum(1 for o in resolved if o.outcome == OUTCOME_WIN)
        if self.resolved_count > 0:
            self.win_rate   = round(self.win_count / self.resolved_count, 4)
            self.avg_return = round(
                sum(o.pnl_pct for o in resolved if o.pnl_pct is not None)
                / self.resolved_count, 4
            )
            self.confidence_score = self.win_rate
        self._refresh_surfaced()

    def _refresh_surfaced(self) -> None:
        self.is_surfaced = (
            self.occurrence_count >= PATTERN_MIN_OCCURRENCES
            and self.resolved_count >= max(3, PATTERN_MIN_OCCURRENCES // 2)
            and self.confidence_score >= PATTERN_CONFIDENCE_THRESHOLD
        )

    def to_dict(self) -> dict:
        return {
            "pattern_id":       self.pattern_id,
            "pattern_type":     self.pattern_type,
            "direction":        self.direction,
            "timeframe":        self.timeframe,
            "description":      self.description,
            "occurrence_count": self.occurrence_count,
            "resolved_count":   self.resolved_count,
            "win_count":        self.win_count,
            "win_rate":         self.win_rate,
            "avg_return":       self.avg_return,
            "confidence_score": self.confidence_score,
            "first_seen":       self.first_seen,
            "last_seen":        self.last_seen,
            "example_trade_ids": self.example_trade_ids,
            "pending_observations": [o.to_dict() for o in self.pending_observations],
            "notes":            self.notes,
            "is_surfaced":      self.is_surfaced,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MarketPattern":
        p = cls(
            pattern_id   = d.get("pattern_id", ""),
            pattern_type = d.get("pattern_type", ""),
            direction    = d.get("direction", "neutral"),
            timeframe    = d.get("timeframe", "any"),
        )
        p.description      = d.get("description", "")
        p.occurrence_count = d.get("occurrence_count", 0)
        p.resolved_count   = d.get("resolved_count", 0)
        p.win_count        = d.get("win_count", 0)
        p.win_rate         = d.get("win_rate", 0.0)
        p.avg_return       = d.get("avg_return", 0.0)
        p.confidence_score = d.get("confidence_score", 0.0)
        p.first_seen       = d.get("first_seen", "")
        p.last_seen        = d.get("last_seen", "")
        p.example_trade_ids = d.get("example_trade_ids", [])
        p.pending_observations = [
            MarketPatternObservation.from_dict(o)
            for o in d.get("pending_observations", [])
        ]
        p.notes       = d.get("notes", "")
        p.is_surfaced = d.get("is_surfaced", False)
        return p

    @staticmethod
    def make_id(pattern_type: str, direction: str, timeframe: str) -> str:
        return f"{pattern_type}__{direction}__{timeframe}"


# ══════════════════════════════════════════════
# BehaviorPattern — user behavior / setup perf
# ══════════════════════════════════════════════

BTYPE_WINNING   = "winning"
BTYPE_FAILING   = "failing"
BTYPE_MISTAKE   = "mistake"
BTYPE_CONDITION = "condition"


@dataclass
class BehaviorPattern:
    """Recurring behavioral or setup-performance observation."""

    pattern_id:   str
    pattern_type: str
    description:  str
    conditions:   Dict[str, str] = field(default_factory=dict)

    occurrences:       int   = 0
    positive_outcomes: int   = 0
    confidence_score:  float = 0.0
    is_surfaced:       bool  = False

    first_seen:        str       = ""
    last_seen:         str       = ""
    example_trade_ids: List[str] = field(default_factory=list)
    notes:             str       = ""

    def update(self, positive: bool, trade_id: str = "") -> None:
        now = datetime.now(tz=timezone.utc).isoformat()
        self.occurrences += 1
        if positive:
            self.positive_outcomes += 1
        if not self.first_seen:
            self.first_seen = now
        self.last_seen = now
        if trade_id and trade_id not in self.example_trade_ids:
            self.example_trade_ids = (self.example_trade_ids + [trade_id])[-10:]
        self.confidence_score = round(self.positive_outcomes / self.occurrences, 4)
        self.is_surfaced = (
            self.occurrences >= PATTERN_MIN_OCCURRENCES
            and self.confidence_score >= PATTERN_CONFIDENCE_THRESHOLD
        )

    def to_dict(self) -> dict:
        return {
            "pattern_id":        self.pattern_id,
            "pattern_type":      self.pattern_type,
            "description":       self.description,
            "conditions":        self.conditions,
            "occurrences":       self.occurrences,
            "positive_outcomes": self.positive_outcomes,
            "confidence_score":  self.confidence_score,
            "is_surfaced":       self.is_surfaced,
            "first_seen":        self.first_seen,
            "last_seen":         self.last_seen,
            "example_trade_ids": self.example_trade_ids,
            "notes":             self.notes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BehaviorPattern":
        p = cls(
            pattern_id   = d.get("pattern_id", ""),
            pattern_type = d.get("pattern_type", ""),
            description  = d.get("description", ""),
        )
        p.conditions        = d.get("conditions", {})
        p.occurrences       = d.get("occurrences", 0)
        p.positive_outcomes = d.get("positive_outcomes", 0)
        p.confidence_score  = d.get("confidence_score", 0.0)
        p.is_surfaced       = d.get("is_surfaced", False)
        p.first_seen        = d.get("first_seen", "")
        p.last_seen         = d.get("last_seen", "")
        p.example_trade_ids = d.get("example_trade_ids", [])
        p.notes             = d.get("notes", "")
        return p


# ══════════════════════════════════════════════
# PatternMemory — unified store
# ══════════════════════════════════════════════

class PatternMemory:
    """
    Unified persistent pattern store.

    Holds two independent namespaces:
      _market   — Dict[str, MarketPattern]
      _behavior — Dict[str, BehaviorPattern]

    Market pattern learning loop:
        obs = pm.record_market_pattern("bull_flag", "bullish", "5m", trade_id="T1")
        # ... trade closes ...
        pm.update_market_pattern_outcome(
            "bull_flag", "bullish", "5m",
            observation_id=obs.observation_id,
            pnl_pct=1.2, outcome="win", trade_id="T1",
        )

    Behavioral patterns:
        pm.update_behavior_from_trade(trade_record)

    Query:
        pm.get_surfaced_market()
        pm.get_surfaced_behavior()
        pm.save()
    """

    def __init__(self, path: Path = PATTERN_FILE) -> None:
        self._path:     Path                           = Path(path)
        self._market:   Dict[str, MarketPattern]   = {}
        self._behavior: Dict[str, BehaviorPattern] = {}
        self._load()

    # ──────────────────────────────────────────
    # Market pattern API
    # ──────────────────────────────────────────

    def record_market_pattern(
        self,
        pattern_type: str,
        direction:    str,
        timeframe:    str,
        trade_id:     Optional[str] = None,
        description:  str           = "",
        notes:        str           = "",
    ) -> MarketPatternObservation:
        """
        Record a new observation of a market pattern.

        Creates the MarketPattern entry if this is the first occurrence.
        Returns the MarketPatternObservation so the caller can later call
        update_market_pattern_outcome() with its observation_id.
        """
        pid = MarketPattern.make_id(pattern_type, direction, timeframe)
        if pid not in self._market:
            desc = description or f"{pattern_type} ({direction}, {timeframe})"
            self._market[pid] = MarketPattern(
                pattern_id   = pid,
                pattern_type = pattern_type,
                direction    = direction,
                timeframe    = timeframe,
                description  = desc,
            )
        return self._market[pid].record_observation(trade_id=trade_id, notes=notes)

    def update_market_pattern_outcome(
        self,
        pattern_type:   str,
        direction:      str,
        timeframe:      str,
        pnl_pct:        float,
        outcome:        str,
        observation_id: Optional[str] = None,
        trade_id:       Optional[str] = None,
    ) -> bool:
        """
        Attach a resolved trade outcome to a pending market pattern observation.

        Matches by observation_id first; falls back to trade_id.
        Recomputes win_rate, avg_return, confidence_score after update.
        Returns True if a matching observation was found and updated.
        """
        pid = MarketPattern.make_id(pattern_type, direction, timeframe)
        mp  = self._market.get(pid)
        if mp is None:
            return False
        return mp.record_outcome(
            observation_id = observation_id or "",
            pnl_pct        = pnl_pct,
            outcome        = outcome,
            trade_id       = trade_id,
        )

    def get_market_pattern(self, pattern_id: str) -> Optional[MarketPattern]:
        return self._market.get(pattern_id)

    def get_all_market_patterns(self) -> List[MarketPattern]:
        return list(self._market.values())

    def get_surfaced_market(self) -> List[MarketPattern]:
        """Return market patterns that have met the surface threshold."""
        return sorted(
            [p for p in self._market.values() if p.is_surfaced],
            key=lambda p: -p.confidence_score,
        )

    def get_market_by_type(self, pattern_type: str) -> List[MarketPattern]:
        return [p for p in self._market.values() if p.pattern_type == pattern_type]

    # ──────────────────────────────────────────
    # Behavior pattern API
    # ──────────────────────────────────────────

    def update_behavior_from_trade(self, record: TradeRecord) -> None:
        """Extract and update behavioral patterns from a completed TradeRecord."""
        # Setup + regime performance
        if record.setup_type and record.regime_at_entry:
            key   = f"setup_{record.setup_type}__regime_{record.regime_at_entry}"
            btype = BTYPE_WINNING if record.outcome == OUTCOME_WIN else BTYPE_FAILING
            self._ensure_behavior(
                key, btype,
                f"{record.setup_type} in {record.regime_at_entry}",
                {"setup_type": record.setup_type, "regime": record.regime_at_entry},
            ).update(positive=(record.outcome == OUTCOME_WIN), trade_id=record.trade_id)

        # Mistake tags
        for tag in record.mistake_tags:
            key = f"mistake__{tag.replace(' ', '_')}"
            self._ensure_behavior(
                key, BTYPE_MISTAKE,
                f"Repeated mistake: {tag}",
                {"mistake_tag": tag},
            ).update(positive=True, trade_id=record.trade_id)

        # HTF alignment
        if record.htf_bias_at_entry:
            key = f"htf_{record.htf_bias_at_entry}__dir_{record.direction}"
            self._ensure_behavior(
                key, BTYPE_CONDITION,
                f"{record.direction} trade with {record.htf_bias_at_entry} HTF bias",
                {"htf_bias": record.htf_bias_at_entry, "direction": record.direction},
            ).update(positive=(record.outcome == OUTCOME_WIN), trade_id=record.trade_id)

    def record_behavior(
        self,
        pattern_id:   str,
        description:  str,
        pattern_type: str                  = BTYPE_CONDITION,
        conditions:   Optional[Dict[str, str]] = None,
        positive:     bool                 = True,
        trade_id:     str                  = "",
        notes:        str                  = "",
    ) -> BehaviorPattern:
        """Manually register a behavioral observation."""
        p = self._ensure_behavior(
            pattern_id, pattern_type, description, conditions or {}
        )
        p.update(positive=positive, trade_id=trade_id)
        if notes:
            p.notes = notes
        return p

    def get_surfaced_behavior(self) -> List[BehaviorPattern]:
        return sorted(
            [p for p in self._behavior.values() if p.is_surfaced],
            key=lambda p: -p.confidence_score,
        )

    def get_mistakes(self) -> List[BehaviorPattern]:
        return [p for p in self._behavior.values() if p.pattern_type == BTYPE_MISTAKE]

    def get_all_behavior(self) -> List[BehaviorPattern]:
        return list(self._behavior.values())

    def filter_behavior_by_confidence(
        self, min_confidence: float = PATTERN_CONFIDENCE_THRESHOLD
    ) -> List[BehaviorPattern]:
        return [p for p in self._behavior.values()
                if p.confidence_score >= min_confidence]

    # ──────────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────────

    def save(self) -> bool:
        data = {
            "schema_version":    MEMORY_SCHEMA_VERSION,
            "market_patterns":   {k: p.to_dict() for k, p in self._market.items()},
            "behavior_patterns": {k: p.to_dict() for k, p in self._behavior.items()},
        }
        return save(self._path, data)

    def _load(self) -> None:
        data = load(self._path)
        if data is None:
            return
        for k, v in data.get("market_patterns", {}).items():
            try:
                self._market[k] = MarketPattern.from_dict(v)
            except Exception:
                pass
        for k, v in data.get("behavior_patterns", {}).items():
            try:
                self._behavior[k] = BehaviorPattern.from_dict(v)
            except Exception:
                pass

    # ──────────────────────────────────────────
    # Internal
    # ──────────────────────────────────────────

    def _ensure_behavior(
        self,
        pattern_id:   str,
        pattern_type: str,
        description:  str,
        conditions:   Dict[str, str],
    ) -> BehaviorPattern:
        if pattern_id not in self._behavior:
            self._behavior[pattern_id] = BehaviorPattern(
                pattern_id   = pattern_id,
                pattern_type = pattern_type,
                description  = description,
                conditions   = conditions,
            )
        return self._behavior[pattern_id]