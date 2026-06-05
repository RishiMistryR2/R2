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
  4. Validate Alpaca credentials (_validate_alpaca) — sets data_source + session_mode
  5. Load memory — non-fatal
  6. Seed initial market state — explicit live attempt first, then demo fallback
  7. Build Scheduler — marks update as done after successful seed
  8. Show startup briefing
  9. Set status RUNNING

Credential reading:
  _validate_alpaca() reads ALPACA_API_KEY and ALPACA_SECRET_KEY via os.getenv()
  AFTER _load_dotenv() has populated the environment.  It does NOT import from
  config.settings, which is evaluated at module-import time and would therefore
  hold empty strings if the .env file had not yet been loaded.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

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
    Returns (RuntimeState, Scheduler).  Never raises.
    """
    _configure_logging(config.log_level)

    logger.info("=" * 60)
    logger.info("[startup] R2 Runtime starting")
    logger.info(
        f"[startup] symbol={config.symbol}"
        f"  live={config.live}"
        f"  mock_ai={config.mock_ai}"
    )

    # Load .env FIRST so os.getenv() calls in _validate_alpaca see the keys.
    _load_dotenv()

    # Initialise RuntimeState
    state = RuntimeState(
        symbol                      = config.symbol,
        session_mode                = SessionMode.LIVE if config.live else SessionMode.DEMO,
        update_interval_seconds     = config.update_interval,
        commentary_cooldown_seconds = config.commentary_cooldown,
        mock_ai                     = config.mock_ai,
    )
    logger.info(f"[startup] RuntimeState created  mode={state.session_mode.value}")

    # Validate Alpaca credentials — sets session_mode + data_source.
    # Reads keys via os.getenv() so .env values are visible.
    _validate_alpaca(state, config)

    # Load memory (non-fatal)
    try:
        _load_memory(state)
    except Exception as e:
        logger.warning(f"[startup] _load_memory failed (non-fatal): {e}")

    # Seed market state (non-fatal).
    # For LIVE mode: calls build_live_state() directly before the loop starts
    # so the startup briefing shows real prices, not 545.00.
    seeded = False
    try:
        seeded = _seed_market_state(state)
    except Exception as e:
        logger.warning(f"[startup] _seed_market_state failed (non-fatal): {e}")

    # Build scheduler.
    # Mark the update task as done when state was already seeded —
    # prevents the loop's first iteration from immediately overwriting
    # the freshly seeded live state.
    scheduler = _build_scheduler(config)
    if seeded:
        scheduler.task_done("update")

    # Show startup briefing (non-fatal)
    if config.show_dashboard:
        try:
            _show_startup_briefing(state, config)
        except Exception as e:
            logger.warning(f"[startup] Startup briefing failed (non-fatal): {e}")
            print(
                f"  R2 starting  [{state.session_mode.value}]  [DATA:{state.data_source}]",
                flush=True,
            )

    state.status = RuntimeStatus.RUNNING
    logger.info("[startup] Startup complete — entering runtime loop")
    return state, scheduler


# ──────────────────────────────────────────────
# Startup helpers
# ──────────────────────────────────────────────

def _validate_alpaca(state: RuntimeState, config: StartupConfig) -> None:
    """
    Validate Alpaca credentials and set state.session_mode + state.data_source.

    Reads credentials via os.getenv() — NOT from config.settings constants —
    so that values loaded by _load_dotenv() are always visible, regardless of
    when config.settings was first imported.

    Outcomes:
      live=False                           → DEMO, data_source=MOCK
      live=True, env keys missing          → DEMO, data_source=MOCK, WARNING
      live=True, keys present, conn fails  → DEMO, data_source=MOCK, WARNING
      live=True, keys present, conn OK     → LIVE, data_source=LIVE

    Never raises.
    """
    if not config.live:
        state.data_source = "MOCK"
        logger.info(
            "[startup] Demo mode — synthetic market data (MOCK).  "
            "Use --live with Alpaca keys for real prices."
        )
        return

    # Read directly from the live environment — .env was loaded above.
    api_key    = os.getenv("ALPACA_API_KEY",    "")
    secret_key = os.getenv("ALPACA_SECRET_KEY", "")

    if not api_key or not secret_key:
        logger.warning(
            "[startup] --live requested but ALPACA_API_KEY / ALPACA_SECRET_KEY "
            "not found in environment.  Falling back to MOCK mode.\n"
            "         Add keys to .env or export them before running."
        )
        state.session_mode = SessionMode.DEMO
        state.data_source  = "MOCK"
        return

    # Keys are present — test the connection.
    try:
        from engine.alpaca_feed import validate_credentials
        if validate_credentials():
            state.session_mode = SessionMode.LIVE
            state.data_source  = "LIVE"
            logger.info("[startup] Alpaca credentials validated  [DATA:LIVE]")
        else:
            logger.warning(
                "[startup] Alpaca connection test failed (bad keys or network).  "
                "Falling back to MOCK mode."
            )
            state.session_mode = SessionMode.DEMO
            state.data_source  = "MOCK"
    except Exception as e:
        logger.warning(
            f"[startup] Alpaca validation error: {e}  Falling back to MOCK mode."
        )
        state.session_mode = SessionMode.DEMO
        state.data_source  = "MOCK"


def _load_dotenv() -> None:
    """Load .env into the process environment.  Non-fatal."""
    try:
        from dotenv import load_dotenv
        env_path = ROOT / ".env"
        if env_path.exists():
            load_dotenv(env_path, override=True)
            logger.info("[startup] .env loaded")
        else:
            logger.info("[startup] No .env found — using shell environment variables")
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
    Each sub-step is independently non-fatal.
    """
    try:
        from memory.session_memory import SessionMemory
        state.session_memory = SessionMemory.today(state.symbol)
        state.session_memory.open_session()
        logger.info(
            f"[startup] Session memory opened: {state.session_memory.session_date}"
        )
    except Exception as e:
        logger.warning(f"[startup] session_memory load failed (non-fatal): {e}")

    try:
        from memory.pattern_memory import PatternMemory
        state.pattern_memory = PatternMemory()
        n_m = len(state.pattern_memory.get_all_market_patterns())
        n_b = len(state.pattern_memory.get_all_behavior())
        logger.info(f"[startup] PatternMemory loaded: {n_m} market, {n_b} behavior")
    except Exception as e:
        logger.warning(f"[startup] pattern_memory load failed (non-fatal): {e}")

    try:
        from memory.startup_context import build_startup_briefing
        state.startup_briefing = build_startup_briefing()
        logger.info(
            f"[startup] StartupBriefing assembled:"
            f" risks={len(state.startup_briefing.top_risks)}"
        )
    except Exception as e:
        logger.warning(f"[startup] startup_briefing failed (non-fatal): {e}")


def _seed_market_state(state: RuntimeState) -> bool:
    """
    Seed the initial market state before the runtime loop starts.

    Returns True if state.market_state was populated successfully.

    LIVE mode:
      Calls build_live_state() directly (not via run_update_cycle) so the
      startup briefing shows real prices before the async loop starts.
      On failure: logs a clear WARNING, falls through to demo seed.
      data_source → "MOCK"; session_mode stays LIVE so the loop retries.

    DEMO mode:
      Calls run_update_cycle() which builds the synthetic 545.00 state.
    """
    if state.session_mode == SessionMode.LIVE:
        try:
            from engine.alpaca_feed import build_live_state as _bls
            market = _bls(state.symbol)
            market.data_source = "LIVE"
            state.data_source  = "LIVE"

            # Enrich with StrategyEngine (best-effort)
            try:
                from strategy.strategy_engine import StrategyEngine
                market = StrategyEngine().process(market)
            except Exception as se:
                logger.warning(f"[startup] StrategyEngine enrichment failed: {se}")

            state.market_state = market
            state.mark_update()
            state.health.record_success()

            ind_5m = market.indicators.get("5m")
            price  = f"@ {ind_5m.close:.2f}" if ind_5m else ""
            logger.info(
                f"[startup] Live state seeded: {state.symbol} {price}  [DATA:LIVE]"
            )
            return True

        except Exception as e:
            logger.warning(
                f"[startup] Live seed failed ({type(e).__name__}: {e}).\n"
                f"          Dashboard will show [DATA:MOCK] until Alpaca responds.\n"
                f"          The update loop will keep retrying live data."
            )
            state.data_source = "MOCK"
            # session_mode intentionally stays LIVE — loop keeps retrying

    # DEMO mode or LIVE fallback
    try:
        from runtime.update_cycle import run_update_cycle
        ok = run_update_cycle(state)
        if ok:
            price = state.market_state.indicators["5m"].close
            logger.info(f"[startup] Demo state seeded  @ {price:.2f}  [DATA:MOCK]")
        else:
            logger.warning("[startup] Demo seed cycle returned False")
        return ok
    except Exception as e:
        logger.warning(f"[startup] Demo seed failed: {e}")
        return False


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
        src  = state.data_source
        text = f"  R2 starting  [{state.session_mode.value} mode]  [DATA:{src}]"
    print(text, flush=True)


# ──────────────────────────────────────────────
# Interactive (blocking) loop
# ──────────────────────────────────────────────

def run_interactive(state: RuntimeState, scheduler: Scheduler) -> None:
    """Blocking interactive command loop for terminal use."""
    from runtime.user_input import read_command_blocking, handle_command
    from runtime.update_cycle import run_update_cycle

    print("\n  R2 ready.  Type 'help' for commands.  Type 'quit' to exit.\n")

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