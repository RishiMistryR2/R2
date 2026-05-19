"""
runtime/runtime_loop.py — main async orchestration loop for R2.

Responsibilities:
  - Orchestrate update_cycle, commentary_cycle, dashboard refresh
  - Check for user commands (non-blocking)
  - Handle graceful shutdown
  - Catch and survive exceptions without stopping

Architecture:
  The loop is async to allow asyncio.sleep() between iterations without
  blocking, and to make future async data feeds easy to integrate.
  Most work is synchronous.

Key design constraint:
  All external callables are imported at MODULE LEVEL so that
  unittest.mock.patch("runtime.runtime_loop.X") works correctly in tests.

Entry point:
  await run_loop(state, scheduler)
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

# Module-level imports — required for mock.patch("runtime.runtime_loop.X") to work
from runtime.update_cycle import run_update_cycle
from runtime.commentary_cycle import generate_commentary, generate_startup_briefing_commentary
from cli.dashboard import render_dashboard

from runtime.runtime_state import RuntimeState, RuntimeStatus
from runtime.scheduler import Scheduler

logger = logging.getLogger("runtime.loop")

# Minimum sleep between loop iterations (prevents CPU spin)
MIN_SLEEP_SECONDS = 0.5


# ──────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────

async def run_loop(state: RuntimeState, scheduler: Scheduler) -> None:
    """
    Main async runtime loop.
    Runs until state.is_shutting_down() returns True.

    Each iteration:
      1. Update cycle (if due)
      2. Commentary cycle (if due and triggered)
      3. Dashboard refresh (if due)
      4. Non-blocking command check
      5. Health check
      6. Sleep
    """
    logger.info(
        f"[loop] Starting  mode={state.session_mode.value}"
        f"  update_interval={state.update_interval_seconds}s"
    )

    state.mark_started()
    _try_startup_commentary(state)

    while not state.is_shutting_down():
        iter_start = datetime.now(tz=timezone.utc)

        # 1. Update
        if scheduler.task_due("update"):
            _run_update(state, scheduler)

        # 2. Commentary
        if scheduler.task_due("commentary_check"):
            _run_commentary(state, scheduler)
            scheduler.task_done("commentary_check")

        # 3. Dashboard
        if scheduler.task_due("dashboard"):
            _render_dashboard(state)
            scheduler.task_done("dashboard")

        # 4. Commands
        _check_commands(state)

        # 5. Health
        _check_health(state)

        # 6. Sleep
        elapsed = (datetime.now(tz=timezone.utc) - iter_start).total_seconds()
        sleep_s = max(MIN_SLEEP_SECONDS, 1.0 - elapsed)
        await asyncio.sleep(sleep_s)

    _on_shutdown(state)


# ──────────────────────────────────────────────
# Per-iteration helpers
# ──────────────────────────────────────────────

def _run_update(state: RuntimeState, scheduler: Scheduler) -> None:
    """Run update_cycle safely — loop always continues."""
    try:
        ok = run_update_cycle(state)
        scheduler.task_done("update", error=not ok)
        if ok:
            logger.debug(f"[loop] Update OK  {state.status_line()}")
    except Exception as e:
        logger.error(f"[loop] Update cycle exception: {e}", exc_info=True)
        state.health.record_error(str(e))
        scheduler.task_done("update", error=True)


def _run_commentary(state: RuntimeState, scheduler: Scheduler) -> None:
    """Check if commentary should fire, generate it if so."""
    try:
        from runtime.commentary_cycle import should_generate_commentary
        cooldown_ready = scheduler.cooldown_ready("ai_commentary")
        should, reason = should_generate_commentary(state, cooldown_ready)
        if should:
            text = generate_commentary(state, trigger=reason)
            if text:
                scheduler.cooldown_reset("ai_commentary")
                logger.info(f"[loop] Commentary generated  reason={reason}")
    except Exception as e:
        logger.error(f"[loop] Commentary cycle exception: {e}", exc_info=True)


def _render_dashboard(state: RuntimeState) -> None:
    """Print dashboard to terminal. Failure is non-fatal."""
    try:
        output = render_dashboard(
            state           = state.market_state,
            score           = state.setup_score,
            context_state   = state.context_state,
            session_memory  = state.session_memory,
            active_patterns = (
                state.startup_briefing.active_market_patterns
                if state.startup_briefing else []
            ),
            recent_events   = state.recent_events,
        )
        print(output, flush=True)

        # Show latest commentary if recent (< 5 minutes old)
        if state.latest_commentary and state.commentary_at:
            age = (datetime.now(tz=timezone.utc) - state.commentary_at).total_seconds()
            if age < 300:
                print(
                    f"\n  ── R2 ──\n  {state.latest_commentary[:300]}\n",
                    flush=True,
                )
    except Exception as e:
        logger.warning(f"[loop] Dashboard render error: {e}")


def _check_commands(state: RuntimeState) -> None:
    """Non-blocking stdin command check."""
    try:
        from runtime.user_input import read_command_nonblocking, handle_command
        cmd = read_command_nonblocking()
        if cmd:
            output = handle_command(cmd, state)
            if output:
                print(output, flush=True)
    except Exception as e:
        logger.warning(f"[loop] Command check error: {e}")


def _check_health(state: RuntimeState) -> None:
    """Log a warning if consecutive errors are accumulating. Never stops the loop."""
    if state.health.consecutive_errors >= 5:
        logger.warning(
            f"[loop] {state.health.consecutive_errors} consecutive update errors."
            "  Check data feed / configuration."
        )


def _try_startup_commentary(state: RuntimeState) -> None:
    """Generate startup commentary. Non-fatal if it fails."""
    try:
        text = generate_startup_briefing_commentary(state)
        if text:
            print(f"\n  ── R2 Pre-market Briefing ──\n  {text[:500]}\n", flush=True)
    except Exception as e:
        logger.info(f"[loop] Startup commentary skipped: {e}")


def _on_shutdown(state: RuntimeState) -> None:
    """Clean up on graceful exit."""
    state.status = RuntimeStatus.STOPPED
    logger.info(f"[loop] Loop stopped  {state.status_line()}")