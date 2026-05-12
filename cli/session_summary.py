"""
cli/session_summary.py — end-of-session summary renderer for R2.

Pure formatting functions.  No I/O.

Consumes:
  - SessionMemory
  - DayReflection
  - PerformanceReport
  - List[TradeRecord]
"""

from __future__ import annotations

from typing import List, Optional

WIDTH = 72


def render_session_summary(
    session,
    reflection  = None,
    trades      = None,
    report      = None,
) -> str:
    """
    Render the complete end-of-session summary.

    Parameters
    ----------
    session    : SessionMemory
    reflection : DayReflection | None
    trades     : List[TradeRecord] | None
    report     : PerformanceReport | None
    """
    sections: List[str] = []
    sections.append(_header(session))
    sections.append(_overview(session))
    sections.append(_trade_list(trades or []))

    if report:
        sections.append(_performance_breakdown(report))

    if reflection:
        sections.append(_quality_section(reflection))
        sections.append(_analysis_section(reflection))
        sections.append(_lessons_section(reflection))

    sections.append(_behavioral_section(session))
    sections.append(_footer(session))

    return "\n".join(s for s in sections if s)


# ──────────────────────────────────────────────
# Section builders
# ──────────────────────────────────────────────

def _header(session) -> str:
    lines = [
        _divider("R2 SESSION SUMMARY"),
        f"  Date     : {session.session_date}",
        f"  Symbol   : {session.symbol}",
    ]
    if session.session_start and session.session_end:
        start = session.session_start[:19].replace("T", " ")
        end   = session.session_end[:19].replace("T", " ")
        lines.append(f"  Session  : {start}  →  {end} UTC")
    return "\n".join(lines)


def _overview(session) -> str:
    lines    = [_section("OVERVIEW")]
    pnl_sign = "+" if session.daily_pnl_pct >= 0 else ""
    lines.append(
        f"  Trades  : {session.total_trades}  "
        f"({session.winning_trades}W / {session.losing_trades}L)"
    )
    lines.append(
        f"  Win rate: {session.win_rate * 100:.0f}%  "
        f"P&L: {pnl_sign}{session.daily_pnl_pct:.2f}%"
    )
    if session.max_intraday_drawdown_pct:
        lines.append(
            f"  Max DD  : -{session.max_intraday_drawdown_pct:.2f}%  "
            f"Peak gain: +{session.peak_gain_pct:.2f}%"
        )
    if session.opening_regime:
        regime = session.opening_regime.replace("_", " ").title()
        if session.closing_regime and session.closing_regime != session.opening_regime:
            closing = session.closing_regime.replace("_", " ").title()
            lines.append(f"  Regime  : {regime} → {closing}")
        else:
            lines.append(f"  Regime  : {regime}")
    return "\n".join(lines)


def _trade_list(trades: list) -> str:
    if not trades:
        return _section("TRADES") + "\n  No trades recorded."

    lines = [_section("TRADES")]
    for t in trades:
        outcome_marker = {
            "win":       "✓",
            "loss":      "✗",
            "breakeven": "~",
        }.get(t.outcome, "?")
        pnl_sign  = "+" if t.pnl_pct >= 0 else ""
        direction = t.direction[:4].upper() if t.direction else "—"
        setup     = t.setup_type[:12] if t.setup_type else "—"
        grade     = t.setup_grade or "—"
        lines.append(
            f"  {outcome_marker}  {t.trade_id[:8]}  "
            f"{direction:<5}  {setup:<14}  "
            f"[{grade}]  {pnl_sign}{t.pnl_pct:.2f}%"
        )
    return "\n".join(lines)


def _performance_breakdown(report) -> str:
    lines = [_section("PERFORMANCE BY SETUP")]

    if not report.by_setup_type:
        lines.append("  Insufficient data for breakdown.")
        return "\n".join(lines)

    lines.append(f"  {'Setup':<18}  {'N':>3}  {'WR':>5}  {'Expect':>7}  {'PF':>5}")
    lines.append(f"  {'─' * 18}  {'─' * 3}  {'─' * 5}  {'─' * 7}  {'─' * 5}")

    for key, s in sorted(
        report.by_setup_type.items(),
        key=lambda kv: -kv[1].expectancy,
    ):
        wr  = f"{s.win_rate * 100:.0f}%"
        exp = f"{s.expectancy:+.3f}%"
        pf  = f"{s.profit_factor:.2f}" if s.profit_factor < 99 else "∞"
        lines.append(f"  {key:<18}  {s.total_trades:>3}  {wr:>5}  {exp:>7}  {pf:>5}")

    if report.best_setup:
        lines.append(f"\n  Best setup : {report.best_setup}")
    if report.worst_setup and report.worst_setup != report.best_setup:
        lines.append(f"  Weak setup : {report.worst_setup}")

    return "\n".join(lines)


def _quality_section(reflection) -> str:
    lines       = [_section("EXECUTION QUALITY")]
    assessments = reflection.trade_quality_assessments
    if not assessments:
        lines.append("  No assessments available.")
        return "\n".join(lines)

    counts = {"good": 0, "acceptable": 0, "poor": 0}
    for a in assessments:
        counts[a.execution_quality] = counts.get(a.execution_quality, 0) + 1

    lines.append(
        f"  Good: {counts['good']}  "
        f"Acceptable: {counts['acceptable']}  "
        f"Poor: {counts['poor']}"
    )
    if reflection.good_trade_bad_outcome_count:
        n = reflection.good_trade_bad_outcome_count
        lines.append(f"  {n} well-executed trade(s) had bad outcomes — process was correct")
    if reflection.bad_trade_good_outcome_count:
        n = reflection.bad_trade_good_outcome_count
        lines.append(f"  ⚠ {n} poor-execution trade(s) had good outcomes — do not reinforce")

    return "\n".join(lines)


def _analysis_section(reflection) -> str:
    lines = [_section("ANALYSIS")]

    if reflection.mistake_analysis:
        lines.append("  Mistakes:")
        for m in reflection.mistake_analysis[:4]:
            lines.append(f"    · {m}")

    if reflection.performance_insights:
        lines.append("  Insights:")
        for ins in reflection.performance_insights[:3]:
            lines.append(f"    · {ins}")

    if reflection.suggested_adjustments:
        lines.append("  Adjustments to consider:")
        for adj in reflection.suggested_adjustments[:3]:
            lines.append(f"    · {adj}")

    return "\n".join(lines)


def _lessons_section(reflection) -> str:
    if not reflection.lessons_learned:
        return ""
    lines = [_section("LESSONS")]
    for lesson in reflection.lessons_learned[:4]:
        lines.append(f"  · {lesson}")
    return "\n".join(lines)


def _behavioral_section(session) -> str:
    if not session.behavioral_flags and not session.mistake_tags:
        return ""
    lines = [_section("BEHAVIORAL FLAGS")]
    for flag in session.behavioral_flags[:4]:
        lines.append(f"  ⚠ {flag}")
    for tag in session.mistake_tags[:4]:
        lines.append(f"  · Mistake: {tag}")
    return "\n".join(lines)


def _footer(session) -> str:
    lines = [_divider()]
    if session.post_market_notes:
        lines.append(f"  Notes  : {session.post_market_notes}")
    if session.lesson:
        lines.append(f"  Lesson : {session.lesson}")
    lines.append(_divider())
    return "\n".join(lines)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _section(label: str) -> str:
    pad = max(0, WIDTH - len(label) - 4)
    return f"\n  ── {label} {'─' * pad}"


def _divider(label: str = "") -> str:
    if label:
        pad   = max(0, WIDTH - len(label) - 4)
        left  = pad // 2
        right = pad - left
        return f"  {'═' * left} {label} {'═' * right}"
    return f"  {'═' * WIDTH}"


def render_session_one_liner(session) -> str:
    """Compact single-line session result for logs or inline display."""
    pnl_sign = "+" if session.daily_pnl_pct >= 0 else ""
    wr        = session.win_rate * 100
    return (
        f"{session.session_date}  "
        f"{session.total_trades}T  "
        f"{wr:.0f}% WR  "
        f"P&L {pnl_sign}{session.daily_pnl_pct:.2f}%"
    )