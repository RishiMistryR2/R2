"""
runtime/run_r2.py — R2 startup sequence and main entry point.

Usage:
    python -m runtime.run_r2                 # demo mode, mock AI
    python -m runtime.run_r2 --live          # live data (requires Alpaca keys)
    python -m runtime.run_r2 --no-mock-ai    # real Claude API
    python -m runtime.run_r2 --interactive   # blocking terminal REPL
    python -m runtime.run_r2 --no-dashboard  # suppress dashboard output

Startup sequence:
  1. Configure logging
  2. Load .env (non-fatal)
  3. Initialise RuntimeState
  4. Load memory — non-fatal: SessionMemory, PatternMemory, StartupBriefing
  5. Seed initial market state — non-fatal: first update cycle
  6. Build Scheduler
  7. Show startup briefing — non-fatal
  8. Set status RUNNING and return

Safety guarantee:
  Every step after state initialisation is wrapped in try/except.
  startup() never raises — it always returns a usable (state, scheduler) pair.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

# Ensure project root is on PYTHONPATH
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime.runtime_state import RuntimeState, RuntimeStatus, SessionMode
from runtime.scheduler import Scheduler

logger = logging.getLogger("runtime.startup")


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

class StartupConfig:
    """Parsed startup configuration."""

    def __init__(
        self,
        symbol:              str   = "SPY",
        live:                bool  = False,
        mock_ai:             bool  = True,
        update_interval:     float = 30.0,
        dashboard_interval:  float = 15.0,
        commentary_cooldown: float = 120.0,
        show_dashboard:      bool  = True,
        interactive:         bool  = False,
        log_level:           str   = "INFO",
    ) -> None:
        self.symbol              = symbol
        self.live                = live
        self.mock_ai             = mock_ai
        self.update_interval     = update_interval
        self.dashboard_interval  = dashboard_interval
        self.commentary_cooldown = commentary_cooldown
        self.show_dashboard      = show_dashboard
        self.interactive         = interactive
        self.log_level           = log_level


# ──────────────────────────────────────────────
# Startup sequence
# ──────────────────────────────────────────────

def startup(config: StartupConfig) -> tuple:
    """
    Execute the full startup sequence.

    Returns (RuntimeState, Scheduler) ready for the runtime loop.

    Every step after state construction is independently non-fatal:
    a failure in memory loading or the first update cycle is logged
    and skipped — R2 continues in a degraded-but-running state.
    """
    _configure_logging(config.log_level)

    logger.info("=" * 60)
    logger.info("[startup] R2 Runtime starting")
    logger.info(
        f"[startup] symbol={config.symbol}"
        f"  live={config.live}"
        f"  mock_ai={config.mock_ai}"
    )

    _load_dotenv()

    # Initialise state (the one step that is allowed to propagate failures)
    state = RuntimeState(
        symbol                      = config.symbol,
        session_mode                = SessionMode.LIVE if config.live else SessionMode.DEMO,
        update_interval_seconds     = config.update_interval,
        commentary_cooldown_seconds = config.commentary_cooldown,
        mock_ai                     = config.mock_ai,
    )
    logger.info(f"[startup] RuntimeState created  mode={state.session_mode.value}")

    # All remaining steps are non-fatal
    try:
        _load_memory(state)
    except Exception as e:
        logger.warning(f"[startup] _load_memory failed (non-fatal): {e}")

    try:
        _seed_market_state(state)
    except Exception as e:
        logger.warning(f"[startup] _seed_market_state failed (non-fatal): {e}")

    scheduler = _build_scheduler(config)

    if config.show_dashboard:
        try:
            _show_startup_briefing(state, config)
        except Exception as e:
            logger.warning(f"[startup] Startup briefing failed (non-fatal): {e}")
            print(f"  R2 starting  [{state.session_mode.value} mode]", flush=True)

    state.status = RuntimeStatus.RUNNING
    logger.info("[startup] Startup complete — entering runtime loop")
    return state, scheduler


# ──────────────────────────────────────────────
# Startup helpers
# ──────────────────────────────────────────────

def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
        env_path = ROOT / ".env"
        if env_path.exists():
            load_dotenv(env_path)
            logger.info("[startup] .env loaded")
        else:
            logger.info("[startup] No .env found — using environment variables")
    except ImportError:
        logger.info("[startup] python-dotenv not installed — skipping .env load")
    except Exception as e:
        logger.warning(f"[startup] .env load error (non-fatal): {e}")


def _configure_logging(level: str = "INFO") -> None:
    """Configure runtime log file under logs/runtime/."""
    try:
        from config.settings import LOGS_DIR
        from datetime import datetime
        runtime_logs = Path(LOGS_DIR) / "runtime"
        runtime_logs.mkdir(parents=True, exist_ok=True)
        stamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = runtime_logs / f"r2_{stamp}.log"
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s — %(message)s")
        )
        logging.getLogger("runtime").addHandler(fh)
        logger.info(f"[startup] Runtime log: {log_file}")
    except Exception as e:
        logging.basicConfig(level=level)
        logger.warning(f"[startup] Logging setup partial: {e}")


def _load_memory(state: RuntimeState) -> None:
    """
    Load all memory modules.
    Each sub-step is independently non-fatal — a failure in one
    does not prevent the others from running.
    """
    # Session memory
    try:
        from memory.session_memory import SessionMemory
        state.session_memory = SessionMemory.today(state.symbol)
        state.session_memory.open_session()
        logger.info(
            f"[startup] Session memory opened: {state.session_memory.session_date}"
        )
    except Exception as e:
        logger.warning(f"[startup] session_memory load failed (non-fatal): {e}")

    # Pattern memory
    try:
        from memory.pattern_memory import PatternMemory
        state.pattern_memory = PatternMemory()
        n_m = len(state.pattern_memory.get_all_market_patterns())
        n_b = len(state.pattern_memory.get_all_behavior())
        logger.info(f"[startup] PatternMemory loaded: {n_m} market, {n_b} behavior")
    except Exception as e:
        logger.warning(f"[startup] pattern_memory load failed (non-fatal): {e}")

    # Startup briefing
    try:
        from memory.startup_context import build_startup_briefing
        state.startup_briefing = build_startup_briefing()
        logger.info(
            f"[startup] StartupBriefing assembled:"
            f" risks={len(state.startup_briefing.top_risks)}"
        )
    except Exception as e:
        logger.warning(f"[startup] startup_briefing failed (non-fatal): {e}")


def _seed_market_state(state: RuntimeState) -> None:
    """Run first update cycle to seed market state."""
    try:
        from runtime.update_cycle import run_update_cycle
        ok = run_update_cycle(state)
        if ok:
            logger.info("[startup] Initial market state seeded")
        else:
            logger.warning("[startup] Initial update cycle returned False")
    except Exception as e:
        logger.warning(f"[startup] Seed update failed (non-fatal): {e}")


def _build_scheduler(config: StartupConfig) -> Scheduler:
    """Build and return a configured Scheduler."""
    sched = Scheduler()
    sched.add_task("update",           interval=config.update_interval,    run_immediately=True)
    sched.add_task("dashboard",        interval=config.dashboard_interval,  run_immediately=False)
    sched.add_task("commentary_check", interval=10.0,                       run_immediately=True)
    sched.add_cooldown("ai_commentary", cooldown=config.commentary_cooldown, start_ready=True)
    logger.info(
        f"[startup] Scheduler:"
        f" update={config.update_interval}s"
        f" dashboard={config.dashboard_interval}s"
        f" ai_cooldown={config.commentary_cooldown}s"
    )
    return sched


def _show_startup_briefing(state: RuntimeState, config: StartupConfig) -> None:
    """Render and print startup briefing."""
    from cli.startup_briefing import render_startup_briefing
    if state.startup_briefing:
        text = render_startup_briefing(
            briefing      = state.startup_briefing,
            state         = state.market_state,
            context_state = state.context_state,
        )
    else:
        text = "  R2 starting in demo mode.  No prior session data."
    print(text, flush=True)


# ──────────────────────────────────────────────
# Interactive (blocking) loop
# ──────────────────────────────────────────────

def run_interactive(state: RuntimeState, scheduler: Scheduler) -> None:
    """Blocking interactive command loop for terminal use."""
    from runtime.user_input import read_command_blocking, handle_command
    from runtime.update_cycle import run_update_cycle

    print(f"\n  R2 ready.  Type 'help' for commands.  Type 'quit' to exit.\n")

    while not state.is_shutting_down():
        try:
            run_update_cycle(state)
        except Exception as e:
            logger.warning(f"[interactive] Update cycle error: {e}")

        if scheduler.task_due("update"):
            scheduler.task_done("update")

        raw = read_command_blocking("r2> ")
        if not raw:
            continue

        output = handle_command(raw, state)
        if output:
            print(output, flush=True)


# ──────────────────────────────────────────────
# Async entry point
# ──────────────────────────────────────────────

async def run_async(config: StartupConfig) -> None:
    """Async entry point — startup then loop."""
    state, scheduler = startup(config)
    if config.interactive:
        run_interactive(state, scheduler)
    else:
        from runtime.runtime_loop import run_loop
        await run_loop(state, scheduler)


# ──────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────

def parse_args() -> StartupConfig:
    p = argparse.ArgumentParser(description="R2 — SPY trading analyst runtime")
    p.add_argument("--symbol",       default="SPY",  help="Symbol to track")
    p.add_argument("--live",         action="store_true", help="Live data mode")
    p.add_argument("--mock-ai",      action="store_true", default=True)
    p.add_argument("--no-mock-ai",   dest="mock_ai", action="store_false",
                   help="Use real Claude API")
    p.add_argument("--interval",     type=float, default=30.0,
                   help="Update interval (seconds)")
    p.add_argument("--interactive",  action="store_true", help="Blocking terminal mode")
    p.add_argument("--no-dashboard", dest="show_dashboard", action="store_false")
    p.add_argument("--log-level",    default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = p.parse_args()
    return StartupConfig(
        symbol          = args.symbol,
        live            = args.live,
        mock_ai         = args.mock_ai,
        update_interval = args.interval,
        show_dashboard  = args.show_dashboard,
        interactive     = args.interactive,
        log_level       = args.log_level,
    )


if __name__ == "__main__":
    cfg = parse_args()
    asyncio.run(run_async(cfg))