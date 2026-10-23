"""
tests/test_dashboard_rich.py — Rich dashboard (cli/dashboard.py).

Covers: render_dashboard_rich returns a renderable, color thresholds,
session/direction/trigger colors, the RECENT EVENTS panel (newest first,
cap 6, 2-row clamp), hidden/always-shown panels, content-sized layout, and
that runtime_loop no longer prints commentary a second time.

Run: PYTHONPATH=. python tests/test_dashboard_rich.py
"""

import io, os, sys, traceback
from unittest import mock

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


# ──────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────

def _cycled():
    from memory.outcome_tracker import reset_tracker
    from runtime.runtime_state import RuntimeState
    from runtime.update_cycle import run_update_cycle, reset_engine
    reset_tracker(":memory:")
    reset_engine()
    state = RuntimeState()
    run_update_cycle(state)
    return state


def _text(renderable, width=100, height=None):
    from rich.console import Console
    c = Console(record=True, color_system=None, width=width, height=height,
                file=io.StringIO())
    c.print(renderable)
    return c.export_text()


def _styles(text_obj):
    """Every style string applied anywhere in a rich Text."""
    return {str(s.style) for s in text_obj.spans} | {str(text_obj.style)}


# ──────────────────────────────────────────────────────────────
# Renderable
# ──────────────────────────────────────────────────────────────

def test_render_dashboard_rich_returns_renderable():
    from rich.protocol import is_renderable
    from cli.dashboard import render_dashboard_rich
    s = _cycled()
    r = render_dashboard_rich(s.market_state, confidence_state=s.confidence_state)
    assert is_renderable(r)
    assert not isinstance(r, str)
    assert "CONVICTION" in _text(r)


def test_render_dashboard_is_plain_export_of_rich():
    from datetime import datetime, timezone
    from cli.dashboard import render_dashboard, render_dashboard_rich
    s   = _cycled()
    now = datetime(2026, 10, 22, 18, 0, tzinfo=timezone.utc)
    kw  = dict(recent_events=["x"], now=now, confidence_state=s.confidence_state)
    assert render_dashboard(s.market_state, **kw) == _text(
        render_dashboard_rich(s.market_state, **kw))


def test_plain_export_is_100_columns():
    from cli.dashboard import render_dashboard
    s   = _cycled()
    out = render_dashboard(s.market_state, confidence_state=s.confidence_state,
                           recent_events=["AI [periodic_check]: " + "word " * 80])
    assert max(len(l) for l in out.splitlines()) <= 100


def test_same_signature_as_render_dashboard():
    import inspect
    from cli.dashboard import render_dashboard, render_dashboard_rich
    assert (list(inspect.signature(render_dashboard).parameters)
            == list(inspect.signature(render_dashboard_rich).parameters))


# ──────────────────────────────────────────────────────────────
# Color rules
# ──────────────────────────────────────────────────────────────

def test_confidence_color_thresholds():
    from cli.dashboard import confidence_style
    assert confidence_style(39)   == "red"
    assert confidence_style(39.9) == "red"
    assert confidence_style(40)   == "yellow"
    assert confidence_style(59)   == "yellow"
    assert confidence_style(60)   == "green"
    assert confidence_style(90)   == "green"


def test_conviction_number_uses_confidence_color():
    from cli.dashboard import _conviction_panel
    s = _cycled()
    cs = s.confidence_state
    for value, color in ((39.0, "red"), (40.0, "yellow"), (59.0, "yellow"), (60.0, "green")):
        cs.calibrated_confidence = value
        cs.structural_confidence = value
        big = _conviction_panel(cs).renderable.renderables[0]
        assert color in str(big.style), (value, big.style)


def test_session_colors():
    from cli.dashboard import session_style
    for key in ("open", "midday", "power_hour"):
        assert session_style(key) == "green", key
    for key in ("premarket", "after_hours"):
        assert session_style(key) == "yellow", key
    assert session_style("closed") == "dim"


def test_top_bar_session_label_colored():
    from datetime import datetime, timezone
    from engine.market_session import MarketSession
    from cli.dashboard import _top_bar
    s  = _cycled()
    ms = s.market_state
    for sess, color in ((MarketSession.OPEN, "green"),
                        (MarketSession.AFTER_HOURS, "yellow"),
                        (MarketSession.CLOSED, "dim")):
        ms.market_session = sess
        bar = _top_bar(ms, datetime.now(tz=timezone.utc))
        label = f"[{sess.value.replace('_', ' ').upper()}]"
        assert label in bar.plain
        span = next(sp for sp in bar.spans if bar.plain[sp.start:sp.end] == label)
        assert str(span.style) == color


def test_top_bar_format():
    from datetime import datetime, timezone
    from cli.dashboard import _top_bar
    s   = _cycled()
    bar = _top_bar(s.market_state, datetime(2026, 10, 22, 18, 30, tzinfo=timezone.utc)).plain
    assert bar.startswith("R2 · SPY · [DATA:MOCK] · [")
    assert bar.endswith("14:30:00 ET")


def test_direction_colors():
    from cli.dashboard import direction_style
    assert direction_style("Trend Bull") == "green"
    assert direction_style("uptrend")    == "green"
    assert direction_style("Trend Bear") == "red"
    assert direction_style("downtrend")  == "red"
    assert direction_style("Chop")       == "yellow"
    assert direction_style("ranging")    == "yellow"


def test_entries_tick_colors():
    from cli.dashboard import _tick
    yes, no = _tick(True, "allowed", "disabled"), _tick(False, "allowed", "disabled")
    assert yes.plain == "✓ allowed"  and "green" in str(yes.style)
    assert no.plain  == "✗ disabled" and "red"   in str(no.style)


def test_price_colored_against_vwap():
    from cli.dashboard import _market_panel
    s  = _cycled()
    ms = s.market_state
    vwap = ms.indicators["5m"].vwap
    for price, color in ((vwap + 1, "green"), (vwap - 1, "red")):
        ms.current_price = price
        price_row = _market_panel(ms, None).renderable.renderables[0]
        assert any(color in st for st in _styles(price_row)), (price, _styles(price_row))


# ──────────────────────────────────────────────────────────────
# Panels
# ──────────────────────────────────────────────────────────────

def test_warnings_panel_always_shown():
    from cli.dashboard import render_dashboard
    s = _cycled()
    assert "No active warnings" in render_dashboard(s.market_state)


def test_warnings_colored_by_severity():
    from cli.dashboard import _warnings_panel
    from cli.warnings import R2Warning, WarningSeverity, WarningCategory
    cat = list(WarningCategory)[0]
    ws  = [R2Warning(cat, WarningSeverity.CRITICAL, "crit"),
           R2Warning(cat, WarningSeverity.WARNING,  "warn"),
           R2Warning(cat, WarningSeverity.INFO,     "info")]
    rows = _warnings_panel(ws).renderable.renderables
    assert "red"    in str(rows[0].spans[0].style)
    assert "yellow" in str(rows[1].spans[0].style)
    assert "dim"    in str(rows[2].spans[0].style)


def test_setup_and_patterns_hidden_when_empty():
    from cli.dashboard import render_dashboard
    s   = _cycled()
    out = render_dashboard(s.market_state, score=None, active_patterns=[])
    assert "SETUP SCORE"     not in out
    assert "ACTIVE PATTERNS" not in out


def test_conviction_panel_without_state():
    from cli.dashboard import render_dashboard
    s   = _cycled()
    out = render_dashboard(s.market_state, confidence_state=None)
    assert "CONVICTION" in out
    assert "Final Conviction" not in out


def test_conviction_is_largest_number():
    from cli.dashboard import big_number
    rows = big_number("72%").plain.split("\n")
    assert len(rows) == 3                     # three rows tall
    assert all(set(r) <= set("█▀▄ ") for r in rows)


# ──────────────────────────────────────────────────────────────
# Recent events
# ──────────────────────────────────────────────────────────────

def test_events_panel_caps_at_six_newest_first():
    from cli.dashboard import render_dashboard
    s      = _cycled()
    events = [f"event number {i:02d}" for i in range(10)]
    out    = render_dashboard(s.market_state, recent_events=events)
    shown  = [i for i in range(10) if f"event number {i:02d}" in out]
    assert shown == [4, 5, 6, 7, 8, 9]
    assert out.index("event number 09") < out.index("event number 04")


def test_events_clamped_to_two_rows_with_ellipsis():
    from cli.dashboard import _events_panel
    out   = _text(_events_panel(["AI [periodic_check]: " + "word " * 100]), width=60)
    inner = [l for l in out.splitlines() if l.startswith("│")]
    assert len(inner) == 2
    assert inner[-1].rstrip(" │").endswith("…")


def test_short_event_not_ellipsized():
    from cli.dashboard import _events_panel
    assert "…" not in _text(_events_panel(["12:00:01  Regime changed"]))


def test_event_trigger_and_timestamp_parsed():
    from cli.dashboard import parse_event, trigger_style
    assert parse_event("AI [regime_change]: Text here") == ("", "regime_change", "Text here")
    assert parse_event("09:31:05  Session opened") == ("09:31:05", "", "Session opened")
    assert parse_event("plain") == ("", "", "plain")
    assert trigger_style("regime_change")    == "red"
    assert trigger_style("narrative_change") == "yellow"
    assert trigger_style("periodic_check")   == "dim"


# ──────────────────────────────────────────────────────────────
# Layout sizing
# ──────────────────────────────────────────────────────────────

def test_layout_not_clipped_by_short_terminal():
    # The layout sizes itself to content, so a short console height must
    # not cut off panels.
    from cli.dashboard import render_dashboard_rich
    s   = _cycled()
    r   = render_dashboard_rich(s.market_state, confidence_state=s.confidence_state,
                                recent_events=[f"e{i}" for i in range(6)])
    out = _text(r, height=8)
    for needle in ("MARKET", "STATUS", "WARNINGS", "CONVICTION", "RECENT EVENTS", "e0", "e5"):
        assert needle in out, needle


def test_layout_not_padded_to_tall_terminal():
    from cli.dashboard import render_dashboard_rich
    s     = _cycled()
    short = _text(render_dashboard_rich(s.market_state), height=200)
    assert len(short.splitlines()) < 40


# ──────────────────────────────────────────────────────────────
# runtime_loop
# ──────────────────────────────────────────────────────────────

def test_runtime_loop_clears_then_prints_once_no_duplicate_commentary():
    import runtime.runtime_loop as rl
    from datetime import datetime, timezone
    s = _cycled()
    s.latest_commentary = "UNIQUE-COMMENTARY-TEXT"
    s.commentary_at     = datetime.now(tz=timezone.utc)
    s.recent_events.append("AI [regime_change]: UNIQUE-COMMENTARY-TEXT")
    console = mock.MagicMock()
    with mock.patch.object(rl, "_console", console), \
         mock.patch("builtins.print") as bare_print:
        rl._render_dashboard(s)
    assert [c[0] for c in console.method_calls] == ["clear", "print"]
    bare_print.assert_not_called()
    rendered = _text(console.print.call_args[0][0])
    assert rendered.count("UNIQUE-COMMENTARY-TEXT") == 1


def test_runtime_loop_render_failure_non_fatal():
    import runtime.runtime_loop as rl
    s = _cycled()
    with mock.patch.object(rl, "render_dashboard_rich", side_effect=ValueError("x")):
        rl._render_dashboard(s)   # must not raise


# ──────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        (n, f) for n, f in sorted(globals().items())
        if n.startswith("test_") and callable(f)
    ]
    print(f"\nRich dashboard tests ({len(tests)} total)\n{'─' * 60}")
    for name, fn in tests:
        run(name, fn)
    print(f"\n{'=' * 60}")
    print(f"{passed} passed, {failed} failed  ({passed + failed} total)")
    import sys; sys.exit(0 if failed == 0 else 1)
