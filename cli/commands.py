"""
cli/commands.py — command routing and handler structure for R2.

No REPL loop yet.  This module implements:
  - CommandResult dataclass (consistent handler output)
  - Handler functions for each command
  - dispatch(cmd, context) router

Commands:
  help      — list available commands
  summary   — current session overview
  active    — active trade status
  patterns  — surfaced pattern list
  warnings  — active warnings
  trade     — current setup / scored trade proposal
  context   — news and macro context
  reflection — end-of-day reflection output

Context dict keys consumed:
  state           : MarketState
  score           : SetupScore | None
  context_state   : ContextState | None
  session_memory  : SessionMemory | None
  pattern_memory  : PatternMemory | None
  reflection      : DayReflection | None
  trades          : List[TradeRecord] | None
  active_patterns : List[dict] | None
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


# ──────────────────────────────────────────────
# Result type
# ──────────────────────────────────────────────

@dataclass
class CommandResult:
    """Standard return type for all command handlers."""
    command: str
    success: bool
    output:  str
    error:   str = ""

    def __str__(self) -> str:
        return self.output if self.success else f"Error: {self.error}"


# ──────────────────────────────────────────────
# Command handlers
# ──────────────────────────────────────────────

def handle_help(context: Dict[str, Any]) -> CommandResult:
    """List all available commands with brief descriptions."""
    lines = [
        "",
        "  R2 Command Reference",
        "  " + "─" * 50,
        "  help        List available commands",
        "  summary     Current session P&L, trades, win rate",
        "  active      Active trade status and stop levels",
        "  patterns    Surfaced market and behavioral patterns",
        "  warnings    Active operational warnings",
        "  trade       Current setup score and trade proposal",
        "  context     News sentiment and macro calendar",
        "  reflection  End-of-day quality analysis",
        "",
        "  All commands are read-only.  R2 advises; you execute.",
        "",
    ]
    return CommandResult(command="help", success=True, output="\n".join(lines))


def handle_summary(context: Dict[str, Any]) -> CommandResult:
    """Current session overview."""
    from cli.session_summary import render_session_one_liner
    session = context.get("session_memory")
    if session is None:
        return CommandResult(
            command="summary", success=False,
            output="", error="No active session",
        )
    lines = [""]
    lines.append(f"  {render_session_one_liner(session)}")
    if session.opening_regime:
        lines.append(f"  Regime   : {session.opening_regime.replace('_', ' ').title()}")
    if session.dominant_narrative:
        lines.append(f"  Narrative: {session.dominant_narrative.replace('_', ' ').title()}")
    if session.behavioral_flags:
        flags = ", ".join(session.behavioral_flags[:3])
        lines.append(f"  Flags    : {flags}")
    lines.append("")
    return CommandResult(command="summary", success=True, output="\n".join(lines))


def handle_active(context: Dict[str, Any]) -> CommandResult:
    """Active trade status and stop levels."""
    from cli.setup_display import render_stop_summary
    state = context.get("state")
    if state is None:
        return CommandResult(
            command="active", success=False,
            output="", error="No market state available",
        )
    trade = state.active_trade if state.active_trade else context.get("active_trade")
    if trade is None:
        return CommandResult(
            command="active", success=True,
            output="\n  No active trade.\n",
        )
    lines     = [""]
    lifecycle  = getattr(trade.lifecycle, "value", "—") if trade.lifecycle else "—"
    direction  = trade.direction.value.upper() if trade.direction else "—"
    instrument = trade.instrument_type.value if trade.instrument_type else "—"
    lines.append(f"  {direction}  ·  {instrument}  ·  [{lifecycle.upper()}]")
    lines.append(render_stop_summary(trade))
    pnl = getattr(trade, "pnl_pct", None)
    if pnl is not None:
        sign = "+" if pnl >= 0 else ""
        lines.append(f"  P&L : {sign}{pnl:.2f}%")
    lines.append("")
    return CommandResult(command="active", success=True, output="\n".join(lines))


def handle_patterns(context: Dict[str, Any]) -> CommandResult:
    """Surfaced market and behavioral patterns."""
    pm              = context.get("pattern_memory")
    active_patterns = context.get("active_patterns") or []

    lines = [""]
    if pm is not None:
        market   = pm.get_surfaced_market()
        behavior = pm.get_surfaced_behavior()
        mistakes = pm.get_mistakes()

        if market:
            lines.append("  Market patterns (surfaced):")
            for p in market[:5]:
                lines.append(
                    f"    · {p.pattern_id:<36}  "
                    f"wr {p.win_rate:.0%}  n={p.occurrence_count}"
                )
        if behavior:
            lines.append("  Behavioral patterns (surfaced):")
            for p in behavior[:3]:
                lines.append(f"    · {p.description[:40]}  conf {p.confidence_score:.0%}")
        if mistakes:
            lines.append("  Recurring mistakes:")
            for p in [m for m in mistakes if m.is_surfaced][:3]:
                lines.append(f"    ⚠ {p.description}  ({p.occurrences}×)")

        if not market and not behavior and not mistakes:
            lines.append("  No surfaced patterns yet.")
    elif active_patterns:
        lines.append("  Active patterns (from briefing):")
        for p in active_patterns[:5]:
            pid = p.get("pattern_id", p.get("description", "—"))
            lines.append(f"    · {pid}")
    else:
        lines.append("  No pattern data available.")

    lines.append("")
    return CommandResult(command="patterns", success=True, output="\n".join(lines))


def handle_warnings(context: Dict[str, Any]) -> CommandResult:
    """Active operational warnings."""
    from cli.warnings import collect_warnings, render_warnings
    state           = context.get("state")
    context_state   = context.get("context_state")
    session_memory  = context.get("session_memory")
    score           = context.get("score")
    active_patterns = context.get("active_patterns") or []

    if state is None:
        return CommandResult(
            command="warnings", success=False,
            output="", error="No market state available",
        )
    warnings = collect_warnings(
        state           = state,
        context_state   = context_state,
        session_memory  = session_memory,
        score           = score,
        active_patterns = active_patterns,
    )
    output = "\n" + render_warnings(warnings) + "\n"
    return CommandResult(command="warnings", success=True, output=output)


def handle_trade(context: Dict[str, Any]) -> CommandResult:
    """Current setup score and trade proposal."""
    from cli.setup_display import render_setup_card, render_for_against
    state         = context.get("state")
    score         = context.get("score")
    context_state = context.get("context_state")

    if state is None:
        return CommandResult(
            command="trade", success=False,
            output="", error="No market state available",
        )

    direction = getattr(state, "setup_direction", None)
    card      = render_setup_card(state, score, direction, context_state)
    fa        = "\n" + render_for_against(score or _empty_score(), state, context_state) if score else ""
    output    = "\n" + card + fa + "\n"
    return CommandResult(command="trade", success=True, output=output)


def handle_context(context: Dict[str, Any]) -> CommandResult:
    """News sentiment and macro context."""
    cs = context.get("context_state")
    if cs is None or not cs.context_valid:
        return CommandResult(
            command="context", success=True,
            output="\n  Context data unavailable.\n",
        )
    lines = [""]
    lines.append(f"  Bias       : {cs.context_bias}")
    lines.append(f"  Sentiment  : {cs.news_sentiment:+.3f}")
    lines.append(f"  Risk level : {cs.volatility_risk_level}")
    if cs.elevated_risk:
        lines.append(f"  ⚠ {cs.risk_note or 'Elevated risk'}")
    if cs.upcoming_events:
        lines.append("  Events:")
        for e in cs.upcoming_events[:3]:
            name       = getattr(e, "name", "—")
            impact     = getattr(e, "market_impact", None)
            impact_str = f"  [{impact.value}]" if impact else ""
            lines.append(f"    · {name}{impact_str}")
    if cs.vix_level:
        lines.append(f"  VIX        : {cs.vix_level:.1f}  [{cs.vix_regime}]")
    lines.append("")
    return CommandResult(command="context", success=True, output="\n".join(lines))


def handle_reflection(context: Dict[str, Any]) -> CommandResult:
    """End-of-day quality analysis."""
    from cli.session_summary import render_session_summary
    reflection     = context.get("reflection")
    session_memory = context.get("session_memory")
    trades         = context.get("trades")
    report         = context.get("report")

    if reflection is None and session_memory is None:
        return CommandResult(
            command="reflection", success=True,
            output="\n  No reflection data available.\n",
        )

    if session_memory is not None:
        output = "\n" + render_session_summary(
            session    = session_memory,
            reflection = reflection,
            trades     = trades,
            report     = report,
        ) + "\n"
        return CommandResult(command="reflection", success=True, output=output)

    # Reflection-only fallback
    lines = [""]
    if reflection:
        if reflection.lessons_learned:
            lines.append("  Lessons:")
            for l in reflection.lessons_learned[:3]:
                lines.append(f"    · {l}")
        if reflection.suggested_adjustments:
            lines.append("  Adjustments:")
            for a in reflection.suggested_adjustments[:3]:
                lines.append(f"    · {a}")
    lines.append("")
    return CommandResult(command="reflection", success=True, output="\n".join(lines))


# ──────────────────────────────────────────────
# Dispatcher
# ──────────────────────────────────────────────

HANDLERS: Dict[str, Callable] = {
    "help":       handle_help,
    "summary":    handle_summary,
    "active":     handle_active,
    "patterns":   handle_patterns,
    "warnings":   handle_warnings,
    "trade":      handle_trade,
    "context":    handle_context,
    "reflection": handle_reflection,
}


def dispatch(command: str, context: Dict[str, Any]) -> CommandResult:
    """
    Route a command string to the appropriate handler.

    Parameters
    ----------
    command : str   — command name (case-insensitive, stripped)
    context : dict  — shared context dict (state, score, session, etc.)

    Returns CommandResult.  Unknown commands return a helpful error.
    """
    cmd     = command.strip().lower().split()[0] if command.strip() else ""
    handler = HANDLERS.get(cmd)
    if handler is None:
        available = ", ".join(sorted(HANDLERS.keys()))
        return CommandResult(
            command = cmd,
            success = False,
            output  = "",
            error   = f"Unknown command '{cmd}'.  Available: {available}",
        )
    try:
        return handler(context)
    except Exception as e:
        return CommandResult(
            command = cmd,
            success = False,
            output  = "",
            error   = f"Handler error: {e}",
        )


def available_commands() -> List[str]:
    """Return sorted list of registered command names."""
    return sorted(HANDLERS.keys())


# ──────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────

def _empty_score():
    from engine.models import SetupScore
    return SetupScore()