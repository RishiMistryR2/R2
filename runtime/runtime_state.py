"""
runtime/runtime_state.py — central RuntimeState for the R2 live operation layer.

RuntimeState is the single object that represents everything R2 currently knows.
It is mutated by update_cycle and commentary_cycle, and read by the CLI layer.

Design rules:
  - Pure Python dataclass — no async, no I/O, no network
  - All fields have safe defaults so RuntimeState is usable before first update
  - Change-detection methods used by commentary_cycle to decide when to speak

RuntimeStatus:
  STARTING  — initialisation in progress
  RUNNING   — normal operation
  PAUSED    — deliberately suspended
  ERROR     — last cycle threw unhandled exception
  STOPPING  — graceful shutdown in progress
  STOPPED   — fully stopped

SessionMode:
  LIVE   — connected to live data feed
  DEMO   — synthetic state, no live feed
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, List, Optional


# ──────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────

class RuntimeStatus(Enum):
    STARTING = "starting"
    RUNNING  = "running"
    PAUSED   = "paused"
    ERROR    = "error"
    STOPPING = "stopping"
    STOPPED  = "stopped"


class SessionMode(Enum):
    LIVE  = "live"
    DEMO  = "demo"


# ──────────────────────────────────────────────
# Loop health
# ──────────────────────────────────────────────

@dataclass
class LoopHealth:
    """Tracks update-cycle health metrics."""
    total_cycles:       int  = 0
    error_cycles:       int  = 0
    last_cycle_ok:      bool = True
    last_error:         str  = ""
    consecutive_errors: int  = 0

    def record_success(self) -> None:
        self.total_cycles      += 1
        self.last_cycle_ok      = True
        self.consecutive_errors = 0

    def record_error(self, msg: str) -> None:
        self.total_cycles      += 1
        self.error_cycles      += 1
        self.last_cycle_ok      = False
        self.last_error         = msg
        self.consecutive_errors += 1

    def to_dict(self) -> dict:
        return {
            "total_cycles":       self.total_cycles,
            "error_cycles":       self.error_cycles,
            "last_cycle_ok":      self.last_cycle_ok,
            "last_error":         self.last_error[:120] if self.last_error else "",
            "consecutive_errors": self.consecutive_errors,
        }


# ──────────────────────────────────────────────
# RuntimeState
# ──────────────────────────────────────────────

@dataclass
class RuntimeState:
    """
    Central state container for the R2 live operation layer.

    All fields are optional / have safe defaults so the object is usable
    immediately after construction, before the first update cycle completes.
    """

    # Identity
    symbol:       str           = "SPY"
    session_mode: SessionMode   = SessionMode.DEMO
    status:       RuntimeStatus = RuntimeStatus.STARTING

    # Market state (refreshed each update cycle)
    market_state:    Any = None   # engine.models.MarketState
    setup_score:     Any = None   # engine.models.SetupScore
    setup_direction: Any = None   # engine.models.TradeDirection | None
    context_state:   Any = None   # context.context_state.ContextState

    # Trade tracking
    active_trade: Any = None   # engine.models.TradeState | None

    # Warnings (refreshed each update cycle)
    active_warnings:      List[Any]      = field(default_factory=list)
    max_warning_severity: Optional[str]  = None

    # Memory / patterns
    session_memory:   Any = None   # memory.session_memory.SessionMemory
    pattern_memory:   Any = None   # memory.pattern_memory.PatternMemory
    startup_briefing: Any = None   # memory.startup_context.StartupBriefing

    # AI commentary
    latest_commentary: str              = ""
    commentary_type:   str              = ""
    commentary_at:     Optional[datetime] = None

    # Recent events (rolling, for dashboard)
    recent_events: List[str] = field(default_factory=list)
    max_events:    int        = 12

    # Loop health
    health:          LoopHealth        = field(default_factory=LoopHealth)
    started_at:      Optional[datetime] = None
    last_update_at:  Optional[datetime] = None
    update_count:    int                = 0

    # Configuration (set at startup)
    update_interval_seconds:     float = 30.0
    commentary_cooldown_seconds: float = 120.0
    mock_ai:                     bool  = False

    # Internal change-detection state
    _shutdown_requested: bool = field(default=False, repr=False)
    _prev_regime:        str  = field(default="",    repr=False)
    _prev_narrative:     str  = field(default="",    repr=False)
    _prev_warning_count: int  = field(default=0,     repr=False)
    _prev_setup_grade:   str  = field(default="",    repr=False)

    # ──────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────

    def mark_started(self) -> None:
        self.started_at = datetime.now(tz=timezone.utc)
        self.status     = RuntimeStatus.RUNNING

    def mark_update(self) -> None:
        self.last_update_at = datetime.now(tz=timezone.utc)
        self.update_count  += 1

    def request_shutdown(self) -> None:
        self._shutdown_requested = True
        self.status = RuntimeStatus.STOPPING

    def is_shutting_down(self) -> bool:
        return self._shutdown_requested

    # ──────────────────────────────────────────
    # Event log
    # ──────────────────────────────────────────

    def add_event(self, msg: str) -> None:
        """Add a timestamped entry to the rolling event log."""
        ts    = datetime.now(tz=timezone.utc).strftime("%H:%M:%S")
        entry = f"{ts}  {msg}"
        self.recent_events.append(entry)
        if len(self.recent_events) > self.max_events:
            self.recent_events = self.recent_events[-self.max_events:]

    # ──────────────────────────────────────────
    # Change detection (used by commentary_cycle)
    # ──────────────────────────────────────────

    def regime_changed(self) -> bool:
        """True when regime has changed since last call."""
        if self.market_state is None or self.market_state.regime is None:
            return False
        current = self.market_state.regime.regime.value
        changed = current != self._prev_regime
        self._prev_regime = current
        return changed

    def narrative_changed(self) -> bool:
        """True when narrative has changed since last call."""
        if self.market_state is None or self.market_state.narrative is None:
            return False
        current = self.market_state.narrative.narrative.value
        changed = current != self._prev_narrative
        self._prev_narrative = current
        return changed

    def warning_severity_increased(self) -> bool:
        """True when a new CRITICAL or WARNING appeared since last call."""
        n = sum(
            1 for w in self.active_warnings
            if hasattr(w, "severity") and w.severity.value >= 2
        )
        changed = n > self._prev_warning_count
        self._prev_warning_count = n
        return changed

    def setup_grade_changed(self) -> bool:
        """True when setup grade changed (e.g. B → A+) since last call."""
        if self.setup_score is None:
            return False
        grade   = self.setup_score.grade()
        changed = grade != self._prev_setup_grade and self._prev_setup_grade != ""
        self._prev_setup_grade = grade
        return changed

    # ──────────────────────────────────────────
    # Summary
    # ──────────────────────────────────────────

    def status_line(self) -> str:
        """One-line status for logs / display."""
        uptime = ""
        if self.started_at:
            secs   = int((datetime.now(tz=timezone.utc) - self.started_at).total_seconds())
            uptime = f"  up {secs}s"
        cycles = f"  cycles={self.update_count}"
        errors = (
            f"  errors={self.health.error_cycles}"
            if self.health.error_cycles else ""
        )
        return (
            f"[{self.status.value.upper()}]"
            f"  {self.symbol}"
            f"  {self.session_mode.value}"
            f"{uptime}{cycles}{errors}"
        )

    def to_dict(self) -> dict:
        return {
            "symbol":       self.symbol,
            "status":       self.status.value,
            "session_mode": self.session_mode.value,
            "update_count": self.update_count,
            "health":       self.health.to_dict(),
            "started_at":   self.started_at.isoformat()   if self.started_at   else None,
            "last_update":  self.last_update_at.isoformat() if self.last_update_at else None,
            "mock_ai":      self.mock_ai,
        }