"""
tests/test_session_commentary.py — session-aware briefing + commentary tests.

Verifies:
  cli/startup_briefing.py:
    - session_briefing_title() returns correct title for each MarketSession
    - render_startup_briefing() header uses actual session, not "PRE-MARKET"
    - AFTER_HOURS state produces "AFTER-HOURS BRIEFING" title
    - _current_market() shows current_price when available

  ai/state_packager.py:
    - package_market_state() includes market_session value
    - after_hours state packaged as "after_hours", not "power_hour"

  ai/prompt_builder.py:
    - build_startup_briefing_prompt() uses session-appropriate wording
    - after_hours prompt does NOT say "pre-market"
    - all prompts include structure section first
    - market_commentary prompt has REQUIRED ORDER instruction
    - build_startup_briefing_prompt uses _get_session_word correctly

  runtime/commentary_cycle.py:
    - _commit_commentary() stores full text (no 80-char truncation)
    - generate_commentary() calls _log_structure_debug without crashing
    - fallback answer leads with structure when available

Run: PYTHONPATH=. python tests/test_session_commentary.py
"""

import os
import sys
import traceback
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


def _now():
    return datetime.now(tz=timezone.utc)


# ──────────────────────────────────────────────
# Minimal fakes
# ──────────────────────────────────────────────

class _FakeSession:
    def __init__(self, v): self.value = v

class _FakeMS:
    """Fake MarketSession enum-like."""
    def __init__(self, v): self.value = v


def _demo_state_with_session(session_value: str):
    """Build demo state with a specific market_session."""
    from runtime.update_cycle import build_demo_state, reset_engine
    reset_engine()
    state = build_demo_state()
    state.market_session = _FakeMS(session_value)
    return state


def _briefing():
    from memory.startup_context import StartupBriefing
    b = StartupBriefing(
        generated_at=_now().isoformat(),
        session_date="2025-06-10",
    )
    b.prev_session_date     = "2025-06-09"
    b.prev_session_pnl_pct  = -0.5
    b.prev_session_win_rate = 0.40
    b.prev_session_trades   = 3
    b.prev_session_regime   = "trend_bull"
    b.prev_session_lesson   = "wait for structure"
    b.recent_trades_n       = 10
    b.recent_win_rate       = 0.55
    b.recent_expectancy     = 0.12
    b.recent_best_setup     = "continuation"
    b.recent_worst_setup    = "reversal"
    b.top_risks             = ["Previous loss — reduce size"]
    b.summary               = "Test briefing"
    return b


# ══════════════════════════════════════════════════════════════
# cli/startup_briefing.py — session_briefing_title
# ══════════════════════════════════════════════════════════════

def test_title_premarket():
    from cli.startup_briefing import session_briefing_title
    ms = _FakeMS("premarket")
    assert "PRE-MARKET" in session_briefing_title(ms)


def test_title_open():
    from cli.startup_briefing import session_briefing_title
    ms = _FakeMS("open")
    title = session_briefing_title(ms)
    assert "OPEN" in title or "MARKET" in title


def test_title_midday():
    from cli.startup_briefing import session_briefing_title
    assert "MIDDAY" in session_briefing_title(_FakeMS("midday"))


def test_title_power_hour():
    from cli.startup_briefing import session_briefing_title
    assert "POWER" in session_briefing_title(_FakeMS("power_hour"))


def test_title_after_hours():
    from cli.startup_briefing import session_briefing_title
    title = session_briefing_title(_FakeMS("after_hours"))
    assert "AFTER" in title
    assert "PRE" not in title


def test_title_closed():
    from cli.startup_briefing import session_briefing_title
    title = session_briefing_title(_FakeMS("closed"))
    assert "PRE-MARKET" not in title


def test_title_none_falls_back():
    from cli.startup_briefing import session_briefing_title
    title = session_briefing_title(None)
    assert len(title) > 0
    assert "BRIEFING" in title


def test_title_accepts_string():
    from cli.startup_briefing import session_briefing_title
    # Some callers may pass the string value directly
    title = session_briefing_title("after_hours")
    assert "AFTER" in title


# ══════════════════════════════════════════════════════════════
# cli/startup_briefing.py — render_startup_briefing header
# ══════════════════════════════════════════════════════════════

def test_after_hours_briefing_not_pre_market():
    """Core requirement: AFTER_HOURS state must not produce PRE-MARKET title."""
    from cli.startup_briefing import render_startup_briefing
    state = _demo_state_with_session("after_hours")
    result = render_startup_briefing(_briefing(), state=state)
    assert "AFTER-HOURS" in result or "AFTER HOURS" in result, (
        f"Expected AFTER-HOURS in header, got:\n{result[:300]}"
    )
    assert "PRE-MARKET" not in result, (
        "After-hours briefing must not say PRE-MARKET"
    )


def test_premarket_briefing_says_pre_market():
    from cli.startup_briefing import render_startup_briefing
    state  = _demo_state_with_session("premarket")
    result = render_startup_briefing(_briefing(), state=state)
    assert "PRE-MARKET" in result


def test_midday_briefing_says_midday():
    from cli.startup_briefing import render_startup_briefing
    state  = _demo_state_with_session("midday")
    result = render_startup_briefing(_briefing(), state=state)
    assert "MIDDAY" in result


def test_open_briefing_says_open():
    from cli.startup_briefing import render_startup_briefing
    state  = _demo_state_with_session("open")
    result = render_startup_briefing(_briefing(), state=state)
    assert "OPEN" in result or "MARKET" in result


def test_briefing_without_state_still_renders():
    """render_startup_briefing with no state should not crash."""
    from cli.startup_briefing import render_startup_briefing
    result = render_startup_briefing(_briefing(), state=None)
    assert len(result) > 50


def test_briefing_shows_session_label_in_header():
    """Session label line should appear in the header."""
    from cli.startup_briefing import render_startup_briefing
    state = _demo_state_with_session("after_hours")
    result = render_startup_briefing(_briefing(), state=state)
    assert "AFTER HOURS" in result or "after_hours" in result.lower()


def test_briefing_current_market_uses_current_price():
    """_current_market should show current_price if stamped on state."""
    from cli.startup_briefing import render_startup_briefing
    state = _demo_state_with_session("after_hours")
    state.current_price        = 735.50
    state.current_price_source = "extended_hours_trade"
    state.bid = state.ask = state.spread = None
    result = render_startup_briefing(_briefing(), state=state)
    assert "735.50" in result


# ══════════════════════════════════════════════════════════════
# ai/state_packager.py — market_session packaging
# ══════════════════════════════════════════════════════════════

def test_package_includes_market_session():
    from ai.state_packager import package_market_state
    state = _demo_state_with_session("after_hours")
    pkg   = package_market_state(state)
    assert "market_session" in pkg, (
        "market_session must be in the packaged state"
    )


def test_package_market_session_after_hours():
    from ai.state_packager import package_market_state
    state = _demo_state_with_session("after_hours")
    pkg   = package_market_state(state)
    assert pkg["market_session"] == "after_hours"


def test_package_market_session_premarket():
    from ai.state_packager import package_market_state
    state = _demo_state_with_session("premarket")
    pkg   = package_market_state(state)
    assert pkg["market_session"] == "premarket"


def test_package_market_session_differs_from_phase():
    """market_session should say after_hours even though session.phase says power_hour."""
    from ai.state_packager import package_market_state
    state = _demo_state_with_session("after_hours")
    pkg   = package_market_state(state)
    # The lossy mapping puts AFTER_HOURS → power_hour in session.phase
    # but market_session should be the real value
    assert pkg["market_session"] == "after_hours"
    assert pkg.get("session", {}).get("phase") in ("power_hour", "after_hours", None)


def test_package_without_market_session_safe():
    """State without market_session attr should not crash packaging."""
    from ai.state_packager import package_market_state
    from runtime.update_cycle import build_demo_state, reset_engine
    reset_engine()
    state = build_demo_state()
    if hasattr(state, "market_session"):
        del state.market_session
    pkg = package_market_state(state)   # must not raise
    assert "symbol" in pkg


def test_full_package_includes_market_session():
    """build_full_package propagates market_session through to the dict."""
    from ai.state_packager import build_full_package
    state = _demo_state_with_session("after_hours")
    pkg   = build_full_package(state=state)
    d     = pkg.to_dict()
    assert d.get("market", {}).get("market_session") == "after_hours"


# ══════════════════════════════════════════════════════════════
# ai/prompt_builder.py — session-aware wording
# ══════════════════════════════════════════════════════════════

def test_startup_prompt_after_hours_not_pre_market():
    """Core: after-hours startup prompt must NOT say 'pre-market'."""
    from ai.state_packager import build_full_package
    from ai.prompt_builder import build_startup_briefing_prompt
    state = _demo_state_with_session("after_hours")
    pkg   = build_full_package(state=state, request_type="startup_briefing")
    _, user = build_startup_briefing_prompt(pkg)
    assert "pre-market" not in user.lower(), (
        f"After-hours prompt must not say 'pre-market', got:\n{user[:400]}"
    )
    assert "after-hours" in user.lower(), (
        f"After-hours prompt should say 'after-hours', got:\n{user[:400]}"
    )


def test_startup_prompt_premarket_says_pre_market():
    from ai.state_packager import build_full_package
    from ai.prompt_builder import build_startup_briefing_prompt
    state = _demo_state_with_session("premarket")
    pkg   = build_full_package(state=state, request_type="startup_briefing")
    _, user = build_startup_briefing_prompt(pkg)
    assert "pre-market" in user.lower()


def test_startup_prompt_midday_wording():
    from ai.state_packager import build_full_package
    from ai.prompt_builder import build_startup_briefing_prompt
    state = _demo_state_with_session("midday")
    pkg   = build_full_package(state=state)
    _, user = build_startup_briefing_prompt(pkg)
    assert "midday" in user.lower()


def test_market_commentary_has_required_order():
    """Commentary prompt must include explicit REQUIRED ORDER instruction."""
    from ai.state_packager import build_full_package
    from ai.prompt_builder import build_market_commentary_prompt
    from runtime.update_cycle import build_demo_state, reset_engine
    reset_engine()
    pkg = build_full_package(state=build_demo_state())
    _, user = build_market_commentary_prompt(pkg)
    assert "REQUIRED ORDER" in user
    assert "HH/HL" in user or "sequence" in user.lower()


def test_market_commentary_forbids_vwap_lead():
    """Prompt must explicitly tell AI not to lead with VWAP/RSI."""
    from ai.state_packager import build_full_package
    from ai.prompt_builder import build_market_commentary_prompt
    from runtime.update_cycle import build_demo_state, reset_engine
    reset_engine()
    _, user = build_market_commentary_prompt(build_full_package(state=build_demo_state()))
    assert "Do NOT open with" in user or "do not lead with" in user.lower()


def test_startup_prompt_includes_structure_section():
    """Context block must contain the MARKET STRUCTURE section."""
    from ai.state_packager import build_full_package
    from ai.prompt_builder import build_startup_briefing_prompt
    from engine.structure_engine import analyze
    from runtime.update_cycle import build_demo_state, reset_engine
    reset_engine()
    state  = build_demo_state()
    state.market_session = _FakeMS("open")
    struct = analyze(state)
    pkg    = build_full_package(state=state, structure_state=struct)
    _, user = build_startup_briefing_prompt(pkg)
    assert "MARKET STRUCTURE" in user, (
        "Startup prompt must include MARKET STRUCTURE section"
    )


def test_market_commentary_includes_structure_section():
    from ai.state_packager import build_full_package
    from ai.prompt_builder import build_market_commentary_prompt
    from engine.structure_engine import analyze
    from runtime.update_cycle import build_demo_state, reset_engine
    reset_engine()
    state  = build_demo_state()
    struct = analyze(state)
    pkg    = build_full_package(state=state, structure_state=struct)
    _, user = build_market_commentary_prompt(pkg)
    assert "MARKET STRUCTURE" in user


def test_indicators_labelled_secondary_in_prompt():
    """Indicator section must be labelled as secondary in the prompt."""
    from ai.state_packager import build_full_package
    from ai.prompt_builder import build_market_commentary_prompt
    from runtime.update_cycle import build_demo_state, reset_engine
    reset_engine()
    _, user = build_market_commentary_prompt(
        build_full_package(state=build_demo_state())
    )
    assert "secondary" in user.lower() or "do not lead with these" in user.lower()


def test_after_hours_context_shows_market_session():
    """Context block should show AFTER HOURS in the regime section."""
    from ai.state_packager import build_full_package
    from ai.prompt_builder import build_market_commentary_prompt
    state = _demo_state_with_session("after_hours")
    pkg   = build_full_package(state=state)
    _, user = build_market_commentary_prompt(pkg)
    assert "after" in user.lower()


# ══════════════════════════════════════════════════════════════
# runtime/commentary_cycle.py — commit and debug
# ══════════════════════════════════════════════════════════════

def test_commit_commentary_no_truncation():
    """_commit_commentary must store full text, not truncate at 80 chars."""
    from runtime.runtime_state import RuntimeState
    from runtime.commentary_cycle import _commit_commentary
    s    = RuntimeState()
    long = "A" * 300 + " important_end_word"
    _commit_commentary(s, long, "test_trigger")
    assert "important_end_word" in s.recent_events[-1], (
        "_commit_commentary must not truncate at 80 chars"
    )


def test_commit_commentary_collapses_whitespace():
    from runtime.runtime_state import RuntimeState
    from runtime.commentary_cycle import _commit_commentary
    s = RuntimeState()
    _commit_commentary(s, "line one\nline two\n  line three", "test")
    event = s.recent_events[-1]
    assert "\n" not in event
    assert "line one line two line three" in event


def test_commit_commentary_prefix():
    from runtime.runtime_state import RuntimeState
    from runtime.commentary_cycle import _commit_commentary
    s = RuntimeState()
    _commit_commentary(s, "test text", "regime_change")
    # Commentary is now logged via add_event, so a "HH:MM:SS  " timestamp
    # precedes the "AI [trigger]:" prefix.
    import re
    assert re.match(r"^\d{2}:\d{2}:\d{2}  AI \[regime_change\]: ", s.recent_events[-1])


def test_log_structure_debug_no_crash_with_none():
    """_log_structure_debug must not crash when structure_state is None."""
    from runtime.runtime_state import RuntimeState
    from runtime.commentary_cycle import _log_structure_debug
    s = RuntimeState()
    s.structure_state = None
    _log_structure_debug(s, "test")   # must not raise


def test_log_structure_debug_no_crash_with_structure():
    from runtime.runtime_state import RuntimeState
    from runtime.update_cycle import run_update_cycle, reset_engine
    from runtime.commentary_cycle import _log_structure_debug
    reset_engine()
    s = RuntimeState()
    run_update_cycle(s)
    _log_structure_debug(s, "test")   # must not raise


def test_fallback_answer_leads_with_structure():
    """_build_fallback_answer should show structure before regime when available."""
    from runtime.runtime_state import RuntimeState
    from runtime.update_cycle import run_update_cycle, reset_engine
    from runtime.commentary_cycle import _build_fallback_answer
    reset_engine()
    s = RuntimeState()
    run_update_cycle(s)
    answer = _build_fallback_answer(s, "what is happening?")
    # Structure should appear before regime in the fallback
    struct_idx = answer.find("Structure:")
    regime_idx = answer.find("Regime:")
    if struct_idx != -1 and regime_idx != -1:
        assert struct_idx < regime_idx, "Structure should precede regime in fallback"


def test_generate_commentary_with_structure_state():
    """generate_commentary must not crash when structure_state is populated."""
    from runtime.runtime_state import RuntimeState
    from runtime.update_cycle import run_update_cycle, reset_engine
    from runtime.commentary_cycle import generate_commentary, TRIGGER_PERIODIC
    reset_engine()
    s = RuntimeState(mock_ai=True)
    s.mark_started()
    run_update_cycle(s)
    assert s.structure_state is not None
    result = generate_commentary(s, trigger=TRIGGER_PERIODIC)
    assert result is None or isinstance(result, str)


# ══════════════════════════════════════════════════════════════
# Runner
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    tests = [
        (n, f) for n, f in sorted(globals().items())
        if n.startswith("test_") and callable(f)
    ]
    print(f"\nSession + commentary tests ({len(tests)} total)\n{'─' * 60}")
    for name, fn in tests:
        run(name, fn)
    print(f"\n{'=' * 60}")
    print(f"{passed} passed, {failed} failed  ({passed + failed} total)")
    import sys; sys.exit(0 if failed == 0 else 1)