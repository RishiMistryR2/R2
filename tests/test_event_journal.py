"""
tests/test_event_journal.py — memory/event_journal.py and its hook in
runtime/commentary_cycle.generate_commentary.

Covers: write/load round-trip, per-day (ET) files, suppressed + approved +
rate-limited decisions recorded through the commentary cycle, non-fatal
write failures, summary on an empty day, and the CLI.

Run: PYTHONPATH=. python tests/test_event_journal.py
"""

import io
import json
import os
import sys
import tempfile
import traceback
from contextlib import redirect_stdout
from datetime import date, datetime, timezone
from pathlib import Path
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

def _tmpdir() -> Path:
    return Path(tempfile.mkdtemp(prefix="r2-journal-"))


def _result(priority="important", score=32.0, suppress=False,
            keys=("direction_change",), summary="Direction: uptrend → downtrend.",
            trigger="regime_change"):
    from engine.event_engine import ChangeItem, EventIntelligenceResult
    return EventIntelligenceResult(
        priority=priority, score=score,
        changes=[ChangeItem(key=k, priority=priority, description=k, score=10.0)
                 for k in keys],
        suppress=suppress, change_summary=summary, trigger=trigger,
    )


def _conf(structural=58.0, calibrated=63.0, regime=72.0, cycle=7):
    from engine.confidence_engine import ConfidenceState
    return ConfidenceState(
        structural_confidence=structural, calibrated_confidence=calibrated,
        regime_confidence=regime, calibration_result=None,
        computed_at=datetime.now(tz=timezone.utc), source_cycle=cycle,
    )


def _cycled_state():
    from memory.outcome_tracker import reset_tracker
    from runtime.runtime_state import RuntimeState
    from runtime.update_cycle import reset_engine, run_update_cycle
    reset_tracker(":memory:")
    reset_engine()
    state = RuntimeState(mock_ai=True)
    assert run_update_cycle(state) is True
    return state


def _commentary_with(state, result, trigger="regime_change"):
    """Run generate_commentary with evaluate_event forced to `result`."""
    import runtime.commentary_cycle as cc
    with mock.patch("engine.event_engine.evaluate_event", return_value=result):
        return cc.generate_commentary(state, trigger=trigger)


# ──────────────────────────────────────────────────────────────
# Writing / reading
# ──────────────────────────────────────────────────────────────

def test_round_trip():
    from memory import event_journal as ej
    d = _tmpdir()
    assert ej.record_decision(_result(), cycle=7, trigger="regime_change",
                              confidence_state=_conf(), commentary="x" * 500,
                              directory=d)
    (e,) = ej.load_session(directory=d)
    assert e["cycle"] == 7
    assert e["trigger"] == "regime_change"
    assert e["priority"] == "important"
    assert e["score"] == 32.0
    assert e["suppressed"] is False and e["rate_limited"] is False
    assert e["change_summary"] == "Direction: uptrend → downtrend."
    assert e["change_keys"] == ["direction_change"]
    assert e["confidence"] == {"structural": 58.0, "calibrated": 63.0,
                               "regime_confidence": 72.0, "source_cycle": 7}
    assert e["commentary_generated"] is True
    assert e["commentary"] == "x" * 200
    ts = datetime.fromisoformat(e["timestamp"])
    assert ts.utcoffset().total_seconds() == 0


def test_appends_one_line_per_decision():
    from memory import event_journal as ej
    d = _tmpdir()
    for i in range(3):
        ej.record_decision(_result(score=float(i)), cycle=i, directory=d)
    lines = ej.journal_path(directory=d).read_text().splitlines()
    assert len(lines) == 3
    assert [json.loads(l)["cycle"] for l in lines] == [0, 1, 2]


def test_no_confidence_state_records_none():
    from memory import event_journal as ej
    d = _tmpdir()
    ej.record_decision(_result(), confidence_state=None, directory=d)
    (e,) = ej.load_session(directory=d)
    assert e["confidence"] is None
    assert e["commentary_generated"] is False and e["commentary"] == ""


def test_file_is_per_et_day():
    from memory import event_journal as ej
    d = _tmpdir()
    # 2026-10-22 23:30 ET (EDT) == 2026-10-23 03:30 UTC → still the 22nd in ET.
    late = datetime(2026, 10, 23, 3, 30, tzinfo=timezone.utc)
    nxt  = datetime(2026, 10, 23, 14, 0, tzinfo=timezone.utc)
    for when in (late, nxt):
        ej.append_entry(ej.build_entry(_result(), now=when), directory=d)
    assert ej.journal_path("2026-10-22", d).name == "2026-10-22-events.jsonl"
    assert len(ej.load_session("2026-10-22", d)) == 1
    assert len(ej.load_session(date(2026, 10, 23), d)) == 1
    assert sorted(p.name for p in d.iterdir()) == [
        "2026-10-22-events.jsonl", "2026-10-23-events.jsonl"]


def test_write_failure_does_not_raise():
    from memory import event_journal as ej
    d = _tmpdir()
    blocker = d / "not-a-dir"
    blocker.write_text("")            # a file where the directory should be
    with mock.patch.object(ej.logger, "warning") as warn:
        ok = ej.record_decision(_result(), directory=blocker / "sessions")
    assert ok is False
    assert warn.called


def test_build_failure_does_not_raise():
    from memory import event_journal as ej
    with mock.patch.object(ej, "build_entry", side_effect=RuntimeError("boom")), \
         mock.patch.object(ej.logger, "warning") as warn:
        assert ej.record_decision(_result(), directory=_tmpdir()) is False
    assert warn.called


def test_load_skips_malformed_line():
    from memory import event_journal as ej
    d = _tmpdir()
    ej.record_decision(_result(), directory=d)
    with ej.journal_path(directory=d).open("a") as f:
        f.write('{"truncated": \n')
    ej.record_decision(_result(), directory=d)
    assert len(ej.load_session(directory=d)) == 2


def test_load_missing_day_is_empty():
    from memory import event_journal as ej
    assert ej.load_session("2001-01-01", _tmpdir()) == []


# ──────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────

def test_summarize_empty_day():
    from memory import event_journal as ej
    buf = io.StringIO()
    with redirect_stdout(buf):
        stats = ej.summarize_session("2001-01-01", _tmpdir())
    assert stats["total"] == 0
    assert stats["confidence_min"] is None and stats["confidence_final"] is None
    assert "No decisions recorded" in buf.getvalue()


def test_summarize_counts():
    from memory import event_journal as ej
    d = _tmpdir()
    ej.record_decision(_result("critical", 45, keys=("direction_change",)),
                       confidence_state=_conf(calibrated=70), commentary="a", directory=d)
    ej.record_decision(_result("suppressed", 2, suppress=True, keys=("price_move_small",)),
                       confidence_state=_conf(calibrated=55), directory=d)
    ej.record_decision(_result("informational", 8, keys=("pattern_new",)),
                       confidence_state=_conf(calibrated=61), rate_limited=True, directory=d)
    ej.record_decision(_result("important", 25, keys=("direction_change",)),
                       confidence_state=_conf(calibrated=64), commentary="b", directory=d)
    buf = io.StringIO()
    with redirect_stdout(buf):
        s = ej.summarize_session(directory=d)
    assert s["total"] == 4
    assert s["approved"] == 2
    assert s["approved_by_priority"] == {"critical": 1, "important": 1}
    assert s["suppressed"] == 2 and s["rate_limited"] == 1
    assert (s["confidence_min"], s["confidence_max"], s["confidence_final"]) == (55, 70, 64)
    assert s["change_keys"] == ["direction_change", "pattern_new", "price_move_small"]
    assert s["commentary_count"] == 2
    out = buf.getvalue()
    assert "Total decisions:   4" in out
    assert "critical 1, important 1" in out
    assert "incl. 1 rate-limited" in out
    assert "min 55%  max 70%  final 64%" in out


def test_cli_prints_summary_for_date():
    from memory import event_journal as ej
    d = _tmpdir()
    when = datetime(2026, 10, 20, 15, 0, tzinfo=timezone.utc)
    ej.append_entry(ej.build_entry(_result(), now=when), directory=d)
    buf = io.StringIO()
    with mock.patch.object(ej, "JOURNAL_DIR", d), redirect_stdout(buf):
        assert ej.main(["2026-10-20"]) == 0
    assert "2026-10-20" in buf.getvalue()
    assert "Total decisions:   1" in buf.getvalue()


def test_cli_rejects_bad_date():
    from memory import event_journal as ej
    with mock.patch("sys.stderr", new=io.StringIO()):
        assert ej.main(["10/20/2026"]) == 2


# ──────────────────────────────────────────────────────────────
# Hook in commentary_cycle.generate_commentary
# ──────────────────────────────────────────────────────────────

def _hooked(fn):
    """Run fn(state, dir) with the journal redirected and event state reset."""
    import runtime.commentary_cycle as cc
    from memory import event_journal as ej
    d = _tmpdir()
    state = _cycled_state()
    with mock.patch.object(ej, "JOURNAL_DIR", d), \
         mock.patch.object(cc, "_last_snapshot", None), \
         mock.patch.object(cc, "_last_informational", None):
        fn(state)
        return state, ej.load_session(directory=d)


def test_hook_records_suppressed():
    state, entries = _hooked(lambda s: _commentary_with(
        s, _result("suppressed", 2, suppress=True, keys=("price_move_small",))))
    (e,) = entries
    assert e["suppressed"] is True and e["rate_limited"] is False
    assert e["commentary_generated"] is False
    assert e["change_keys"] == ["price_move_small"]
    assert e["cycle"] == state.update_count
    assert e["confidence"]["calibrated"] == state.confidence_state.calibrated_confidence
    assert e["confidence"]["source_cycle"] == state.confidence_state.source_cycle


def test_hook_records_approved_with_commentary():
    out = {}

    def go(s):
        out["text"] = _commentary_with(s, _result("critical", 45))
    _, entries = _hooked(go)
    assert out["text"]
    (e,) = entries
    assert e["suppressed"] is False
    assert e["priority"] == "critical"
    assert e["commentary_generated"] is True
    assert e["commentary"] == out["text"][:200]


def test_hook_records_suppressed_and_approved_in_order():
    def go(s):
        _commentary_with(s, _result("suppressed", 1, suppress=True))
        _commentary_with(s, _result("important", 30))
    _, entries = _hooked(go)
    assert [e["suppressed"] for e in entries] == [True, False]
    assert [e["commentary_generated"] for e in entries] == [False, True]


def test_hook_records_rate_limited():
    import runtime.commentary_cycle as cc

    def go(s):
        with mock.patch.object(cc, "_last_informational",
                               datetime.now(tz=timezone.utc)):
            _commentary_with(s, _result("informational", 8))
    _, entries = _hooked(go)
    (e,) = entries
    assert e["suppressed"] is True and e["rate_limited"] is True
    assert e["commentary_generated"] is False


def test_hook_records_approved_when_commentary_errors():
    import runtime.commentary_cycle as cc

    def go(s):
        with mock.patch.object(cc, "build_full_package",
                               side_effect=RuntimeError("package broke")):
            assert _commentary_with(s, _result("critical", 45)) is None
    _, entries = _hooked(go)
    (e,) = entries
    assert e["suppressed"] is False and e["commentary_generated"] is False


def test_hook_skips_user_request():
    import runtime.commentary_cycle as cc

    def go(s):
        with mock.patch.object(cc, "build_full_package",
                               side_effect=RuntimeError("stop")):
            cc.generate_commentary(s, trigger="user_request")
    _, entries = _hooked(go)
    assert entries == []


def test_hook_journal_failure_does_not_break_commentary():
    out = {}

    def go(s):
        with mock.patch("memory.event_journal.record_decision",
                        side_effect=RuntimeError("disk gone")):
            out["text"] = _commentary_with(s, _result("critical", 45))
    _hooked(go)
    assert out["text"]


# ──────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        (n, f) for n, f in sorted(globals().items())
        if n.startswith("test_") and callable(f)
    ]
    print(f"\nEvent journal tests ({len(tests)} total)\n{'─' * 60}")
    for name, fn in tests:
        run(name, fn)
    print(f"\n{'=' * 60}")
    print(f"{passed} passed, {failed} failed  ({passed + failed} total)")
    sys.exit(0 if failed == 0 else 1)
