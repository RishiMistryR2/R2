"""
runtime/scheduler.py — lightweight scheduling utilities for R2.

No external dependencies.  Pure Python.

CooldownTimer  — "has enough time passed since last fire?"
PeriodicTask   — "is it time to run this recurring task?"
Debouncer      — "suppress rapid repeated triggers"
Scheduler      — named registry of tasks and cooldowns
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Optional


# ──────────────────────────────────────────────
# CooldownTimer
# ──────────────────────────────────────────────

class CooldownTimer:
    """
    Returns True from .ready() only when `cooldown_seconds` have elapsed
    since the last .reset() call.

    Usage:
        timer = CooldownTimer(120)
        if timer.ready():
            generate_commentary()
            timer.reset()
    """

    def __init__(self, cooldown_seconds: float, start_ready: bool = True) -> None:
        self.cooldown_seconds = cooldown_seconds
        self._last_fire: Optional[datetime] = (
            None if start_ready else datetime.now(tz=timezone.utc)
        )

    def ready(self) -> bool:
        if self._last_fire is None:
            return True
        elapsed = (datetime.now(tz=timezone.utc) - self._last_fire).total_seconds()
        return elapsed >= self.cooldown_seconds

    def reset(self) -> None:
        """Call after the action fires to restart the cooldown window."""
        self._last_fire = datetime.now(tz=timezone.utc)

    def seconds_remaining(self) -> float:
        """Seconds until ready.  0.0 if already ready."""
        if self._last_fire is None:
            return 0.0
        elapsed = (datetime.now(tz=timezone.utc) - self._last_fire).total_seconds()
        return max(0.0, self.cooldown_seconds - elapsed)

    def force_ready(self) -> None:
        """Bypass cooldown — next .ready() returns True immediately."""
        self._last_fire = None

    def to_dict(self) -> dict:
        return {
            "cooldown_seconds":  self.cooldown_seconds,
            "ready":             self.ready(),
            "seconds_remaining": round(self.seconds_remaining(), 1),
        }


# ──────────────────────────────────────────────
# PeriodicTask
# ──────────────────────────────────────────────

class PeriodicTask:
    """
    Tracks whether it is time to run a recurring task.

    Usage:
        task = PeriodicTask("update", interval_seconds=30)
        if task.due():
            run_update()
            task.mark_done()
    """

    def __init__(
        self,
        name:             str,
        interval_seconds: float,
        run_immediately:  bool = True,
    ) -> None:
        self.name             = name
        self.interval_seconds = interval_seconds
        self.run_count:  int  = 0
        self.error_count: int = 0
        self._last_run: Optional[datetime] = (
            None if run_immediately else datetime.now(tz=timezone.utc)
        )

    def due(self) -> bool:
        if self._last_run is None:
            return True
        elapsed = (datetime.now(tz=timezone.utc) - self._last_run).total_seconds()
        return elapsed >= self.interval_seconds

    def mark_done(self, error: bool = False) -> None:
        self._last_run = datetime.now(tz=timezone.utc)
        self.run_count += 1
        if error:
            self.error_count += 1

    def seconds_until_due(self) -> float:
        if self._last_run is None:
            return 0.0
        elapsed = (datetime.now(tz=timezone.utc) - self._last_run).total_seconds()
        return max(0.0, self.interval_seconds - elapsed)

    def force_due(self) -> None:
        """Force next due() to return True."""
        self._last_run = None

    def to_dict(self) -> dict:
        return {
            "name":              self.name,
            "interval_seconds":  self.interval_seconds,
            "run_count":         self.run_count,
            "error_count":       self.error_count,
            "due":               self.due(),
            "seconds_until_due": round(self.seconds_until_due(), 1),
        }


# ──────────────────────────────────────────────
# Debouncer
# ──────────────────────────────────────────────

class Debouncer:
    """
    Suppresses rapid repeated events.  Fires only when
    `min_interval_seconds` have passed since the previous fire.
    """

    def __init__(self, min_interval_seconds: float) -> None:
        self._interval = min_interval_seconds
        self._last:     Optional[datetime] = None

    def should_fire(self) -> bool:
        now = datetime.now(tz=timezone.utc)
        if self._last is None or (now - self._last).total_seconds() >= self._interval:
            self._last = now
            return True
        return False


# ──────────────────────────────────────────────
# Scheduler
# ──────────────────────────────────────────────

class Scheduler:
    """
    Named registry of PeriodicTasks and CooldownTimers.

    Usage:
        sched = Scheduler()
        sched.add_task("update",    interval=30)
        sched.add_task("dashboard", interval=5)
        sched.add_cooldown("ai",    cooldown=120)

        if sched.task_due("update"):
            run_update()
            sched.task_done("update")

        if sched.cooldown_ready("ai"):
            generate_commentary()
            sched.cooldown_reset("ai")
    """

    def __init__(self) -> None:
        self._tasks:     Dict[str, PeriodicTask]  = {}
        self._cooldowns: Dict[str, CooldownTimer] = {}

    # Tasks
    def add_task(
        self,
        name:            str,
        interval:        float,
        run_immediately: bool = True,
    ) -> PeriodicTask:
        t = PeriodicTask(name, interval, run_immediately)
        self._tasks[name] = t
        return t

    def task_due(self, name: str) -> bool:
        t = self._tasks.get(name)
        return t.due() if t else False

    def task_done(self, name: str, error: bool = False) -> None:
        if name in self._tasks:
            self._tasks[name].mark_done(error=error)

    def force_task_due(self, name: str) -> None:
        if name in self._tasks:
            self._tasks[name].force_due()

    # Cooldowns
    def add_cooldown(
        self,
        name:        str,
        cooldown:    float,
        start_ready: bool = True,
    ) -> CooldownTimer:
        c = CooldownTimer(cooldown, start_ready)
        self._cooldowns[name] = c
        return c

    def cooldown_ready(self, name: str) -> bool:
        c = self._cooldowns.get(name)
        return c.ready() if c else True

    def cooldown_reset(self, name: str) -> None:
        if name in self._cooldowns:
            self._cooldowns[name].reset()

    def cooldown_force_ready(self, name: str) -> None:
        if name in self._cooldowns:
            self._cooldowns[name].force_ready()

    # Introspection
    def status(self) -> dict:
        return {
            "tasks":     {k: v.to_dict() for k, v in self._tasks.items()},
            "cooldowns": {k: v.to_dict() for k, v in self._cooldowns.items()},
        }