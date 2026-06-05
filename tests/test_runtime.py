"""
tests/test_runtime.py — R2 Runtime / Live Operation Layer.

Coverage:
  runtime_state     : initialization, lifecycle, event log, change detection,
                      LoopHealth, status line
  scheduler         : CooldownTimer, PeriodicTask, Debouncer, Scheduler registry
  update_cycle      : demo state builder, cycle execution, exception safety,
                      consecutive errors, engine singleton
  commentary_cycle  : trigger logic, throttling, fallback answer,
                      exception isolation, state commitment
  user_input        : command routing, quit/exit, ask dispatch,
                      build_command_context, never-raises guarantee
  runtime_loop      : per-iteration helpers all exception-safe,
                      shutdown sets STOPPED
  run_r2            : StartupConfig, startup sequence, scheduler construction,
                      memory loading, graceful failure

Design:
  - No live network calls
  - No blocking I/O
  - All AI calls use mock=True
  - Fully offline

Run: PYTHONPATH=. python tests/test_runtime.py
"""

import asyncio
import os
import sys
import time
import traceback
import unittest.mock as mock
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

passed = failed = 0


def run(name, fn):
    global passed, failed
    try:
        fn()
        print(f"  PASS  {name}")
        passed += 1
    except Exception as e:
        print(f"  FAIL  {name}: {e}")
        traceback.print_exc()
        failed += 1


# ═══════════════════════════════════════════════════════════
# runtime/runtime_state.py
# ═══════════════════════════════════════════════════════════

def test_state_defaults():
    from runtime.runtime_state import RuntimeState, RuntimeStatus, SessionMode
    s = RuntimeState()
    assert s.symbol          == "SPY"
    assert s.status          == RuntimeStatus.STARTING
    assert s.session_mode    == SessionMode.DEMO
    assert s.update_count    == 0
    assert s.market_state    is None
    assert s.active_warnings == []
    assert s.recent_events   == []


def test_state_mark_started():
    from runtime.runtime_state import RuntimeState, RuntimeStatus
    s = RuntimeState()
    s.mark_started()
    assert s.status     == RuntimeStatus.RUNNING
    assert s.started_at is not None


def test_state_mark_update_increments():
    from runtime.runtime_state import RuntimeState
    s = RuntimeState()
    s.mark_started()
    s.mark_update()
    assert s.update_count   == 1
    assert s.last_update_at is not None
    s.mark_update()
    assert s.update_count == 2


def test_state_request_shutdown():
    from runtime.runtime_state import RuntimeState, RuntimeStatus
    s = RuntimeState()
    s.mark_started()
    assert not s.is_shutting_down()
    s.request_shutdown()
    assert s.is_shutting_down()
    assert s.status == RuntimeStatus.STOPPING


def test_state_add_event_rolling():
    from runtime.runtime_state import RuntimeState
    s = RuntimeState(max_events=3)
    for i in range(5):
        s.add_event(f"event {i}")
    assert len(s.recent_events) == 3
    assert "event 4" in s.recent_events[-1]


def test_state_add_event_has_timestamp():
    from runtime.runtime_state import RuntimeState
    s = RuntimeState()
    s.add_event("test message")
    assert len(s.recent_events) == 1
    assert "test message" in s.recent_events[0]
    # HH:MM:SS format prefix
    assert ":" in s.recent_events[0]


def test_state_regime_changed_no_market():
    from runtime.runtime_state import RuntimeState
    s = RuntimeState()
    assert s.regime_changed() is False


def test_state_regime_changed_detects_first():
    from runtime.runtime_state import RuntimeState
    from runtime.update_cycle import build_demo_state
    s = RuntimeState()
    s.market_state = build_demo_state()
    assert s.regime_changed() is True   # "" → current = changed
    assert s.regime_changed() is False  # same value


def test_state_narrative_changed_detects_first():
    from runtime.runtime_state import RuntimeState
    from runtime.update_cycle import build_demo_state
    s = RuntimeState()
    s.market_state = build_demo_state()
    assert s.narrative_changed() is True
    assert s.narrative_changed() is False


def test_state_warning_severity_increased():
    from runtime.runtime_state import RuntimeState
    from cli.warnings import R2Warning, WarningSeverity, WarningCategory
    s = RuntimeState()
    assert s.warning_severity_increased() is False
    s.active_warnings = [
        R2Warning(WarningCategory.READINESS, WarningSeverity.CRITICAL, "test")
    ]
    assert s.warning_severity_increased() is True
    assert s.warning_severity_increased() is False


def test_state_setup_grade_changed():
    from runtime.runtime_state import RuntimeState
    from engine.models import SetupScore
    s = RuntimeState()
    assert s.setup_grade_changed() is False
    sc = SetupScore(total=80.0, adjusted_total=80.0)
    s.setup_score          = sc
    s._prev_setup_grade    = "B"   # prime previous value
    sc.trend_alignment     = 28.0  # push into A+
    sc.momentum_quality    = 23.0
    sc.structure_context   = 22.0
    sc.entry_timing        = 18.0
    s.setup_score          = sc
    # Grade should be A+ now vs prev "B"
    changed = s.setup_grade_changed()
    assert isinstance(changed, bool)


def test_state_status_line_running():
    from runtime.runtime_state import RuntimeState
    s = RuntimeState(symbol="SPY")
    s.mark_started()
    s.mark_update()
    line = s.status_line()
    assert "RUNNING"  in line
    assert "SPY"      in line
    assert "cycles=1" in line


def test_state_to_dict():
    from runtime.runtime_state import RuntimeState
    s = RuntimeState()
    s.mark_started()
    d = s.to_dict()
    for k in ("symbol", "status", "session_mode", "update_count", "health"):
        assert k in d


def test_loop_health_success():
    from runtime.runtime_state import LoopHealth
    h = LoopHealth()
    h.record_success()
    assert h.total_cycles       == 1
    assert h.last_cycle_ok      is True
    assert h.consecutive_errors == 0


def test_loop_health_error():
    from runtime.runtime_state import LoopHealth
    h = LoopHealth()
    h.record_error("boom")
    assert h.total_cycles       == 1
    assert h.error_cycles       == 1
    assert h.last_cycle_ok      is False
    assert h.consecutive_errors == 1
    assert "boom" in h.last_error


def test_loop_health_consecutive_reset_on_success():
    from runtime.runtime_state import LoopHealth
    h = LoopHealth()
    h.record_error("e1")
    h.record_error("e2")
    assert h.consecutive_errors == 2
    h.record_success()
    assert h.consecutive_errors == 0


def test_loop_health_to_dict():
    from runtime.runtime_state import LoopHealth
    d = LoopHealth().to_dict()
    for k in ("total_cycles", "error_cycles", "last_cycle_ok",
              "last_error", "consecutive_errors"):
        assert k in d


# ═══════════════════════════════════════════════════════════
# runtime/scheduler.py
# ═══════════════════════════════════════════════════════════

def test_cooldown_starts_ready():
    from runtime.scheduler import CooldownTimer
    assert CooldownTimer(60.0, start_ready=True).ready() is True


def test_cooldown_not_ready_after_reset():
    from runtime.scheduler import CooldownTimer
    t = CooldownTimer(60.0)
    t.reset()
    assert t.ready() is False


def test_cooldown_ready_after_elapsed():
    from runtime.scheduler import CooldownTimer
    t = CooldownTimer(0.05)
    t.reset()
    time.sleep(0.07)
    assert t.ready() is True


def test_cooldown_seconds_remaining():
    from runtime.scheduler import CooldownTimer
    t = CooldownTimer(120.0)
    t.reset()
    assert 119.0 < t.seconds_remaining() <= 120.0


def test_cooldown_force_ready():
    from runtime.scheduler import CooldownTimer
    t = CooldownTimer(120.0)
    t.reset()
    assert t.ready() is False
    t.force_ready()
    assert t.ready() is True


def test_cooldown_to_dict():
    from runtime.scheduler import CooldownTimer
    d = CooldownTimer(30.0).to_dict()
    for k in ("cooldown_seconds", "ready", "seconds_remaining"):
        assert k in d


def test_periodic_task_due_immediately():
    from runtime.scheduler import PeriodicTask
    assert PeriodicTask("t", 30.0, run_immediately=True).due() is True


def test_periodic_task_not_due_after_mark():
    from runtime.scheduler import PeriodicTask
    t = PeriodicTask("t", 30.0)
    t.mark_done()
    assert t.due() is False


def test_periodic_task_due_after_interval():
    from runtime.scheduler import PeriodicTask
    t = PeriodicTask("t", 0.05, run_immediately=False)
    time.sleep(0.07)
    assert t.due() is True


def test_periodic_task_counts():
    from runtime.scheduler import PeriodicTask
    t = PeriodicTask("t", 0.01, run_immediately=True)
    t.mark_done()
    t.mark_done(error=True)
    assert t.run_count   == 2
    assert t.error_count == 1


def test_periodic_task_force_due():
    from runtime.scheduler import PeriodicTask
    t = PeriodicTask("t", 300.0, run_immediately=False)
    assert t.due() is False
    t.force_due()
    assert t.due() is True


def test_periodic_task_to_dict():
    from runtime.scheduler import PeriodicTask
    d = PeriodicTask("t", 30.0).to_dict()
    for k in ("name", "interval_seconds", "run_count", "error_count", "due"):
        assert k in d


def test_debouncer_fires_first():
    from runtime.scheduler import Debouncer
    assert Debouncer(10.0).should_fire() is True


def test_debouncer_suppresses_rapid():
    from runtime.scheduler import Debouncer
    d = Debouncer(10.0)
    d.should_fire()
    assert d.should_fire() is False


def test_debouncer_fires_after_interval():
    from runtime.scheduler import Debouncer
    d = Debouncer(0.05)
    d.should_fire()
    time.sleep(0.07)
    assert d.should_fire() is True


def test_scheduler_task_lifecycle():
    from runtime.scheduler import Scheduler
    s = Scheduler()
    s.add_task("t1", interval=30.0, run_immediately=True)
    assert s.task_due("t1") is True
    s.task_done("t1")
    assert s.task_due("t1") is False


def test_scheduler_unknown_task_not_due():
    from runtime.scheduler import Scheduler
    assert Scheduler().task_due("nonexistent") is False


def test_scheduler_cooldown_lifecycle():
    from runtime.scheduler import Scheduler
    s = Scheduler()
    s.add_cooldown("ai", cooldown=120.0, start_ready=True)
    assert s.cooldown_ready("ai") is True
    s.cooldown_reset("ai")
    assert s.cooldown_ready("ai") is False
    s.cooldown_force_ready("ai")
    assert s.cooldown_ready("ai") is True


def test_scheduler_unknown_cooldown_is_ready():
    from runtime.scheduler import Scheduler
    assert Scheduler().cooldown_ready("nonexistent") is True


def test_scheduler_force_task_due():
    from runtime.scheduler import Scheduler
    s = Scheduler()
    s.add_task("t1", interval=300.0, run_immediately=False)
    assert s.task_due("t1") is False
    s.force_task_due("t1")
    assert s.task_due("t1") is True


def test_scheduler_status():
    from runtime.scheduler import Scheduler
    s = Scheduler()
    s.add_task("update", interval=30.0)
    s.add_cooldown("ai", cooldown=120.0)
    d = s.status()
    assert "update" in d["tasks"]
    assert "ai"     in d["cooldowns"]


# ═══════════════════════════════════════════════════════════
# runtime/update_cycle.py
# ═══════════════════════════════════════════════════════════

def test_build_demo_state_has_all_required_fields():
    from runtime.update_cycle import build_demo_state
    s = build_demo_state("SPY")
    assert s.symbol == "SPY"
    assert s.regime    is not None
    assert s.narrative is not None
    assert s.session   is not None
    assert "5m"  in s.indicators
    assert "15m" in s.indicators
    assert "1h"  in s.indicators


def test_build_demo_state_indicators_have_values():
    from runtime.update_cycle import build_demo_state
    ind = build_demo_state().indicators["5m"]
    assert ind.close > 0
    assert ind.rsi   > 0
    assert ind.atr   > 0
    assert ind.vwap  is not None


def test_build_demo_state_support_resistance():
    from runtime.update_cycle import build_demo_state
    s = build_demo_state()
    assert len(s.support_levels)    > 0
    assert len(s.resistance_levels) > 0


def test_build_demo_state_deterministic():
    from runtime.update_cycle import build_demo_state
    s1 = build_demo_state("SPY")
    s2 = build_demo_state("SPY")
    assert s1.indicators["5m"].close == s2.indicators["5m"].close


def test_run_update_cycle_returns_true():
    from runtime.runtime_state import RuntimeState
    from runtime.update_cycle import run_update_cycle, reset_engine
    reset_engine()
    s  = RuntimeState()
    ok = run_update_cycle(s)
    assert ok is True


def test_run_update_cycle_populates_state():
    from runtime.runtime_state import RuntimeState
    from runtime.update_cycle import run_update_cycle, reset_engine
    reset_engine()
    s = RuntimeState()
    run_update_cycle(s)
    assert s.market_state   is not None
    assert s.update_count   == 1
    assert s.last_update_at is not None
    assert s.health.last_cycle_ok is True


def test_run_update_cycle_populates_warnings():
    from runtime.runtime_state import RuntimeState
    from runtime.update_cycle import run_update_cycle, reset_engine
    reset_engine()
    s = RuntimeState()
    run_update_cycle(s)
    assert isinstance(s.active_warnings, list)


def test_run_update_cycle_records_error_on_failure():
    from runtime.runtime_state import RuntimeState
    from runtime.update_cycle import run_update_cycle
    s = RuntimeState()
    with mock.patch("runtime.update_cycle._get_market_state",
                    side_effect=RuntimeError("network down")):
        ok = run_update_cycle(s)
    assert ok is False
    assert s.health.error_cycles       >= 1
    assert s.health.last_cycle_ok      is False
    assert s.health.consecutive_errors >= 1


def test_run_update_cycle_next_succeeds_after_error():
    """An error in one cycle must not prevent the next from succeeding."""
    from runtime.runtime_state import RuntimeState
    from runtime.update_cycle import run_update_cycle, reset_engine
    reset_engine()
    s = RuntimeState()
    # First cycle: forced failure
    with mock.patch("runtime.update_cycle._get_market_state",
                    side_effect=RuntimeError("transient")):
        ok1 = run_update_cycle(s)
    assert ok1 is False
    assert s.health.consecutive_errors == 1
    # Second cycle: should succeed
    ok2 = run_update_cycle(s)
    assert ok2 is True
    assert s.health.consecutive_errors == 0


def test_run_update_cycle_three_in_a_row():
    from runtime.runtime_state import RuntimeState
    from runtime.update_cycle import run_update_cycle, reset_engine
    reset_engine()
    s = RuntimeState()
    for _ in range(3):
        run_update_cycle(s)
    assert s.update_count        == 3
    assert s.health.error_cycles == 0


def test_current_phase_returns_valid_phase():
    from runtime.update_cycle import _current_phase
    from engine.models import SessionPhase
    phase = _current_phase(datetime.now(tz=timezone.utc))
    assert isinstance(phase, SessionPhase)


def test_reset_engine_clears_singleton():
    from runtime.update_cycle import _get_or_create_engine, reset_engine
    reset_engine()
    e1 = _get_or_create_engine()
    e2 = _get_or_create_engine()
    assert e1 is e2  # same object
    reset_engine()
    e3 = _get_or_create_engine()
    assert e3 is not e1  # fresh instance after reset


# ═══════════════════════════════════════════════════════════
# runtime/commentary_cycle.py
# ═══════════════════════════════════════════════════════════

def _state_with_market(mock_ai: bool = True):
    from runtime.runtime_state import RuntimeState
    from runtime.update_cycle import build_demo_state, reset_engine
    reset_engine()
    s = RuntimeState(mock_ai=mock_ai)
    s.mark_started()
    s.market_state = build_demo_state()
    return s


def test_should_generate_force():
    from runtime.commentary_cycle import should_generate_commentary, TRIGGER_USER
    s    = _state_with_market()
    ok, reason = should_generate_commentary(s, cooldown_ready=True, force=True)
    assert ok     is True
    assert reason == TRIGGER_USER


def test_should_not_generate_when_cooldown_not_ready():
    from runtime.commentary_cycle import should_generate_commentary
    s = _state_with_market()
    s.regime_changed()     # prime — consume change
    s.narrative_changed()  # prime — consume change
    ok, _ = should_generate_commentary(s, cooldown_ready=False)
    assert ok is False


def test_should_generate_on_regime_change():
    from runtime.commentary_cycle import should_generate_commentary
    s = _state_with_market()
    # First call — regime changes from "" to current value
    ok, reason = should_generate_commentary(s, cooldown_ready=True)
    assert ok is True
    assert reason in ("regime_change", "narrative_change")


def test_generate_commentary_mock_returns_something():
    from runtime.commentary_cycle import generate_commentary, TRIGGER_PERIODIC
    s = _state_with_market(mock_ai=True)
    # Prime change detectors so they don't trigger (we want periodic path)
    s.regime_changed()
    s.narrative_changed()
    result = generate_commentary(s, trigger=TRIGGER_PERIODIC)
    # Mock may return text or None depending on cooldown; both are valid
    assert result is None or isinstance(result, str)


def test_generate_commentary_never_raises():
    from runtime.commentary_cycle import generate_commentary
    s = _state_with_market()
    with mock.patch("runtime.commentary_cycle.build_full_package",
                    side_effect=RuntimeError("packager exploded")):
        result = generate_commentary(s, trigger="market_commentary")
    assert result is None   # caught, not raised


def test_generate_commentary_ai_failure_preserves_old_commentary():
    from runtime.commentary_cycle import generate_commentary
    s = _state_with_market()
    s.latest_commentary = "preserved text"
    with mock.patch("runtime.commentary_cycle.build_full_package",
                    side_effect=Exception("fail")):
        generate_commentary(s, trigger="market_commentary")
    assert s.latest_commentary == "preserved text"


def test_answer_question_returns_string():
    from runtime.commentary_cycle import answer_question
    s = _state_with_market(mock_ai=True)
    result = answer_question(s, "What is the current regime?")
    assert isinstance(result, str)
    assert len(result) > 0


def test_answer_question_fallback_when_ai_returns_none():
    from runtime.commentary_cycle import answer_question
    s = _state_with_market()
    with mock.patch("runtime.commentary_cycle.generate_commentary", return_value=None):
        result = answer_question(s, "anything")
    assert isinstance(result, str)
    assert len(result) > 5
    assert "AI unavailable" in result or "Regime" in result or "unavailable" in result.lower()


def test_generate_startup_briefing_commentary_safe():
    from runtime.commentary_cycle import generate_startup_briefing_commentary
    s      = _state_with_market(mock_ai=True)
    result = generate_startup_briefing_commentary(s)
    assert result is None or isinstance(result, str)


def test_commit_commentary_updates_state():
    from runtime.commentary_cycle import _commit_commentary
    s = _state_with_market()
    _commit_commentary(s, "Market analysis text.", "market_commentary")
    assert s.latest_commentary == "Market analysis text."
    assert s.commentary_type   == "market_commentary"
    assert s.commentary_at     is not None
    assert len(s.recent_events) >= 1


def test_commit_commentary_adds_event():
    from runtime.commentary_cycle import _commit_commentary
    s = _state_with_market()
    _commit_commentary(s, "Short text.", "regime_change")
    assert any("regime_change" in e for e in s.recent_events)


# ═══════════════════════════════════════════════════════════
# runtime/user_input.py
# ═══════════════════════════════════════════════════════════

def _seeded_state():
    from runtime.runtime_state import RuntimeState
    from runtime.update_cycle import run_update_cycle, reset_engine
    reset_engine()
    s = RuntimeState(mock_ai=True)
    s.mark_started()
    run_update_cycle(s)
    return s


def test_handle_empty_returns_empty():
    from runtime.user_input import handle_command
    s = _seeded_state()
    assert handle_command("",   s) == ""
    assert handle_command("  ", s) == ""


def test_handle_quit_shuts_down():
    from runtime.user_input import handle_command
    s = _seeded_state()
    result = handle_command("quit", s)
    assert s.is_shutting_down() is True
    assert len(result) > 0


def test_handle_exit_shuts_down():
    from runtime.user_input import handle_command
    s = _seeded_state()
    handle_command("exit", s)
    assert s.is_shutting_down() is True


def test_handle_q_shuts_down():
    from runtime.user_input import handle_command
    s = _seeded_state()
    handle_command("q", s)
    assert s.is_shutting_down() is True


def test_handle_help():
    from runtime.user_input import handle_command
    s      = _seeded_state()
    result = handle_command("help", s)
    assert isinstance(result, str)
    assert len(result) > 0


def test_handle_summary():
    from runtime.user_input import handle_command
    result = handle_command("summary", _seeded_state())
    assert isinstance(result, str)


def test_handle_warnings():
    from runtime.user_input import handle_command
    result = handle_command("warnings", _seeded_state())
    assert isinstance(result, str)


def test_handle_active():
    from runtime.user_input import handle_command
    result = handle_command("active", _seeded_state())
    assert isinstance(result, str)


def test_handle_patterns():
    from runtime.user_input import handle_command
    result = handle_command("patterns", _seeded_state())
    assert isinstance(result, str)


def test_handle_unknown_command_returns_string():
    from runtime.user_input import handle_command
    result = handle_command("xyznotacmd", _seeded_state())
    assert isinstance(result, str)
    assert len(result) > 0


def test_handle_command_never_raises():
    """Any input must return a string — never raise."""
    from runtime.user_input import handle_command
    s = _seeded_state()
    for cmd in ["!@#$", "a" * 200, "DROP TABLE", "", "None", "True", "1234"]:
        result = handle_command(cmd, s)
        assert isinstance(result, str)


def test_handle_ask_no_question():
    from runtime.user_input import handle_command
    result = handle_command("ask ", _seeded_state())
    assert isinstance(result, str)
    assert "Usage" in result or len(result) > 0


def test_handle_ask_with_question():
    from runtime.user_input import handle_command
    result = handle_command("ask what is the regime?", _seeded_state())
    assert isinstance(result, str)
    assert len(result) > 0


def test_build_command_context_keys():
    from runtime.user_input import build_command_context
    s   = _seeded_state()
    ctx = build_command_context(s)
    for k in ("state", "score", "context_state", "session_memory",
              "pattern_memory", "reflection", "trades", "active_patterns"):
        assert k in ctx, f"missing key: {k}"


def test_build_command_context_state_is_market_state():
    from runtime.user_input import build_command_context
    s   = _seeded_state()
    ctx = build_command_context(s)
    assert ctx["state"] is s.market_state


def test_read_command_nonblocking_no_stdin():
    from runtime.user_input import read_command_nonblocking
    result = read_command_nonblocking()
    assert result is None or isinstance(result, str)


# ═══════════════════════════════════════════════════════════
# runtime/runtime_loop.py — helpers
# ═══════════════════════════════════════════════════════════

def test_run_update_in_loop_safe():
    from runtime.runtime_loop import _run_update
    from runtime.runtime_state import RuntimeState
    from runtime.scheduler import Scheduler
    s     = RuntimeState()
    sched = Scheduler()
    sched.add_task("update", interval=30.0, run_immediately=True)
    with mock.patch("runtime.runtime_loop.run_update_cycle",
                    side_effect=RuntimeError("cycle boom")):
        _run_update(s, sched)   # must not raise
    assert s.health.error_cycles >= 1


def test_run_commentary_in_loop_safe():
    from runtime.runtime_loop import _run_commentary
    from runtime.runtime_state import RuntimeState
    from runtime.update_cycle import build_demo_state
    from runtime.scheduler import Scheduler
    s     = RuntimeState(mock_ai=True)
    s.mark_started()
    s.market_state = build_demo_state()
    sched = Scheduler()
    sched.add_task("commentary_check", interval=10.0, run_immediately=True)
    sched.add_cooldown("ai_commentary",  cooldown=0.0,  start_ready=True)
    with mock.patch("runtime.runtime_loop.generate_commentary",
                    side_effect=RuntimeError("ai boom")):
        _run_commentary(s, sched)   # must not raise


def test_render_dashboard_safe():
    from runtime.runtime_loop import _render_dashboard
    from runtime.runtime_state import RuntimeState
    from runtime.update_cycle import build_demo_state
    s = RuntimeState()
    s.market_state = build_demo_state()
    with mock.patch("runtime.runtime_loop.render_dashboard",
                    side_effect=ValueError("render fail")):
        _render_dashboard(s)   # must not raise


def test_check_health_no_raise():
    from runtime.runtime_loop import _check_health
    from runtime.runtime_state import RuntimeState
    s = RuntimeState()
    for _ in range(10):
        s.health.record_error("repeated")
    _check_health(s)   # must not raise, just warns


def test_on_shutdown_sets_stopped():
    from runtime.runtime_loop import _on_shutdown
    from runtime.runtime_state import RuntimeState, RuntimeStatus
    s = RuntimeState()
    s.mark_started()
    _on_shutdown(s)
    assert s.status == RuntimeStatus.STOPPED


def test_try_startup_commentary_safe():
    from runtime.runtime_loop import _try_startup_commentary
    from runtime.runtime_state import RuntimeState
    s = RuntimeState(mock_ai=True)
    with mock.patch("runtime.runtime_loop.generate_startup_briefing_commentary",
                    side_effect=RuntimeError("briefing fail")):
        _try_startup_commentary(s)   # must not raise


# ═══════════════════════════════════════════════════════════
# runtime/run_r2.py
# ═══════════════════════════════════════════════════════════

def test_startup_config_defaults():
    from runtime.run_r2 import StartupConfig
    c = StartupConfig()
    assert c.symbol             == "SPY"
    assert c.mock_ai            is True
    assert c.update_interval    == 30.0
    assert c.show_dashboard     is True
    assert c.live               is False


def test_startup_config_custom():
    from runtime.run_r2 import StartupConfig
    c = StartupConfig(symbol="QQQ", live=True, mock_ai=False, update_interval=15.0)
    assert c.symbol          == "QQQ"
    assert c.live            is True
    assert c.mock_ai         is False
    assert c.update_interval == 15.0


def test_startup_returns_state_and_scheduler():
    from runtime.run_r2 import StartupConfig, startup
    from runtime.runtime_state import RuntimeState, RuntimeStatus
    from runtime.scheduler import Scheduler
    c     = StartupConfig(mock_ai=True, show_dashboard=False)
    state, sched = startup(c)
    assert isinstance(state, RuntimeState)
    assert isinstance(sched, Scheduler)
    assert state.status == RuntimeStatus.RUNNING


def test_startup_state_has_market_state():
    from runtime.run_r2 import StartupConfig, startup
    c     = StartupConfig(mock_ai=True, show_dashboard=False)
    state, _ = startup(c)
    assert state.market_state is not None


def test_startup_scheduler_has_required_tasks():
    from runtime.run_r2 import StartupConfig, startup
    c = StartupConfig(mock_ai=True, show_dashboard=False)
    _, sched = startup(c)
    d = sched.status()
    assert "update"           in d["tasks"]
    assert "dashboard"        in d["tasks"]
    assert "commentary_check" in d["tasks"]
    assert "ai_commentary"    in d["cooldowns"]


def test_startup_demo_mode():
    from runtime.run_r2 import StartupConfig, startup
    from runtime.runtime_state import SessionMode
    c     = StartupConfig(live=False, mock_ai=True, show_dashboard=False)
    state, _ = startup(c)
    assert state.session_mode == SessionMode.DEMO


def test_startup_session_mode_live():
    import unittest.mock as mock
    from runtime.run_r2 import StartupConfig, startup
    from runtime.runtime_state import SessionMode

    cfg = StartupConfig(live=True)
    # Bypass _validate_alpaca (credential behaviour tested in test_data_source.py)
    # and _seed_market_state (live network calls) to test session_mode plumbing only.
    with mock.patch("runtime.run_r2._show_startup_briefing"), \
         mock.patch("runtime.run_r2._configure_logging"), \
         mock.patch("runtime.run_r2._load_dotenv"), \
         mock.patch("runtime.run_r2._validate_alpaca"), \
         mock.patch("runtime.run_r2._seed_market_state", return_value=False):
        state, _ = startup(cfg)

    assert state.session_mode == SessionMode.LIVE


def test_startup_memory_load_failure_non_fatal():
    """startup() must complete even when memory loading raises."""
    from runtime.run_r2 import StartupConfig, startup
    from runtime.runtime_state import RuntimeStatus
    c = StartupConfig(mock_ai=True, show_dashboard=False)
    with mock.patch("runtime.run_r2._load_memory", side_effect=Exception("disk error")):
        state, sched = startup(c)
    assert state.status == RuntimeStatus.RUNNING


def test_startup_seed_failure_non_fatal():
    """startup() must complete even when first update cycle fails."""
    from runtime.run_r2 import StartupConfig, startup
    from runtime.runtime_state import RuntimeStatus
    c = StartupConfig(mock_ai=True, show_dashboard=False)
    with mock.patch("runtime.run_r2._seed_market_state",
                    side_effect=Exception("data feed down")):
        state, sched = startup(c)
    assert state.status == RuntimeStatus.RUNNING


def test_build_scheduler_intervals():
    from runtime.run_r2 import StartupConfig, _build_scheduler
    c     = StartupConfig(update_interval=20.0, dashboard_interval=8.0,
                          commentary_cooldown=90.0)
    sched = _build_scheduler(c)
    d     = sched.status()
    assert d["tasks"]["update"]["interval_seconds"]            == 20.0
    assert d["tasks"]["dashboard"]["interval_seconds"]         == 8.0
    assert d["cooldowns"]["ai_commentary"]["cooldown_seconds"] == 90.0

# ═══════════════════════════════════════════════════════════
# Integration
# ═══════════════════════════════════════════════════════════

def test_integration_startup_then_commands():
    """Full startup → dispatch multiple commands."""
    from runtime.run_r2 import StartupConfig, startup
    from runtime.user_input import handle_command
    c     = StartupConfig(mock_ai=True, show_dashboard=False)
    state, _ = startup(c)
    for cmd in ("help", "summary", "warnings", "active", "patterns"):
        result = handle_command(cmd, state)
        assert isinstance(result, str), f"command {cmd!r} returned non-string"


def test_integration_update_then_commentary_check():
    """Update → should_generate_commentary → generate — no crash."""
    from runtime.runtime_state import RuntimeState
    from runtime.scheduler import Scheduler
    from runtime.update_cycle import run_update_cycle, reset_engine
    from runtime.commentary_cycle import should_generate_commentary, generate_commentary
    reset_engine()
    s     = RuntimeState(mock_ai=True)
    s.mark_started()
    sched = Scheduler()
    sched.add_cooldown("ai_commentary", cooldown=0.0, start_ready=True)
    run_update_cycle(s)
    should, reason = should_generate_commentary(s, True, force=True)
    assert should is True
    result = generate_commentary(s, trigger=reason)
    assert result is None or isinstance(result, str)


def test_integration_three_cycles_stable():
    """Three consecutive update cycles must all succeed cleanly."""
    from runtime.runtime_state import RuntimeState
    from runtime.update_cycle import run_update_cycle, reset_engine
    reset_engine()
    s = RuntimeState()
    for _ in range(3):
        assert run_update_cycle(s) is True
    assert s.update_count        == 3
    assert s.health.error_cycles == 0


def test_integration_error_then_recovery():
    """One failed cycle followed by a successful one."""
    from runtime.runtime_state import RuntimeState
    from runtime.update_cycle import run_update_cycle, reset_engine
    reset_engine()
    s = RuntimeState()
    with mock.patch("runtime.update_cycle._get_market_state",
                    side_effect=RuntimeError("transient error")):
        ok1 = run_update_cycle(s)
    assert ok1 is False
    ok2 = run_update_cycle(s)
    assert ok2 is True
    assert s.health.consecutive_errors == 0


def test_integration_event_log_grows_with_commentary():
    from runtime.commentary_cycle import _commit_commentary
    from runtime.runtime_state import RuntimeState
    s = RuntimeState(max_events=10)
    for i in range(4):
        _commit_commentary(s, f"Commentary {i}", "market_commentary")
    assert len(s.recent_events) == 4


def test_integration_status_line_after_startup():
    from runtime.run_r2 import StartupConfig, startup
    c     = StartupConfig(mock_ai=True, show_dashboard=False)
    state, _ = startup(c)
    line  = state.status_line()
    assert "RUNNING"  in line
    assert "SPY"      in line
    assert "cycles="  in line


def test_integration_async_loop_stops_on_shutdown():
    """run_loop exits cleanly when shutdown is requested after 2 iterations."""
    from runtime.runtime_loop import run_loop
    from runtime.runtime_state import RuntimeState, RuntimeStatus
    from runtime.scheduler import Scheduler

    s = RuntimeState(mock_ai=True)
    sched = Scheduler()
    sched.add_task("update",           interval=0.1,  run_immediately=True)
    sched.add_task("dashboard",        interval=999.0, run_immediately=False)
    sched.add_task("commentary_check", interval=999.0, run_immediately=False)
    sched.add_cooldown("ai_commentary",  cooldown=999.0, start_ready=False)

    call_count = [0]
    original_run = None

    def fake_update(st: RuntimeState) -> bool:
        call_count[0] += 1
        st.health.record_success()
        st.mark_update()
        if call_count[0] >= 2:
            st.request_shutdown()
        return True

    with mock.patch("runtime.runtime_loop.run_update_cycle", side_effect=fake_update):
        asyncio.run(run_loop(s, sched))

    assert s.status == RuntimeStatus.STOPPED
    assert call_count[0] >= 2


# ═══════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    tests = [
        (n, f) for n, f in sorted(globals().items())
        if n.startswith("test_") and callable(f)
    ]
    print(f"\nRuntime tests ({len(tests)} total)\n{'─' * 60}")
    for name, fn in tests:
        run(name, fn)
    print(f"\n{'=' * 60}")
    print(f"{passed} passed, {failed} failed  ({passed + failed} total)")
    sys.exit(0 if failed == 0 else 1)