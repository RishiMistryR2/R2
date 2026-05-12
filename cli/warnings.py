"""
cli/warnings.py — operational warnings for the R2 trading interface.

Pure functions.  No I/O, no side effects.
All functions return Warning objects or lists of Warning objects.

Warning levels (ascending severity):
  INFO      — informational, no action required
  CAUTION   — worth noting; review before entry
  WARNING   — significant concern; proceed carefully
  CRITICAL  — high-risk condition; strong reconsideration advised

Warnings never auto-block execution.
R2 advises; the trader decides.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


# ──────────────────────────────────────────────
# Warning types and severity
# ──────────────────────────────────────────────

class WarningSeverity(Enum):
    INFO     = 0
    CAUTION  = 1
    WARNING  = 2
    CRITICAL = 3

    def label(self) -> str:
        return {
            WarningSeverity.INFO:     "INFO",
            WarningSeverity.CAUTION:  "CAUTION",
            WarningSeverity.WARNING:  "WARNING",
            WarningSeverity.CRITICAL: "CRITICAL",
        }[self]


class WarningCategory(Enum):
    READINESS       = "readiness"
    OVERTRADING     = "overtrading"
    MACRO           = "macro"
    CONFIDENCE      = "confidence"
    CONTEXT         = "context"
    VOLATILITY      = "volatility"
    PATTERN         = "pattern"
    BEHAVIOR        = "behavior"
    RISK            = "risk"
    SETUP           = "setup"


@dataclass
class R2Warning:
    """A single operational warning."""
    category:  WarningCategory
    severity:  WarningSeverity
    message:   str
    detail:    str = ""

    def to_dict(self) -> dict:
        return {
            "category": self.category.value,
            "severity": self.severity.value,
            "message":  self.message,
            "detail":   self.detail,
        }

    def render(self) -> str:
        """Single-line terminal representation."""
        prefix = {
            WarningSeverity.INFO:     "  ·",
            WarningSeverity.CAUTION:  "  ▲",
            WarningSeverity.WARNING:  "  ⚠",
            WarningSeverity.CRITICAL: "  ✖",
        }[self.severity]
        label = f"[{self.severity.label()}]"
        line  = f"{prefix} {label:<10} {self.message}"
        if self.detail:
            line += f"\n              {self.detail}"
        return line


# ──────────────────────────────────────────────
# Warning generators (pure functions)
# ──────────────────────────────────────────────

def check_readiness(state) -> List[R2Warning]:
    """Warn when market data is insufficient for confident analysis."""
    warnings: List[R2Warning] = []
    if state.session is None:
        return warnings
    from engine.models import Readiness
    r = state.session.readiness
    if r == Readiness.NOT_READY:
        warnings.append(R2Warning(
            category = WarningCategory.READINESS,
            severity = WarningSeverity.CRITICAL,
            message  = "Insufficient market data — no entries advisable",
            detail   = "Requires 30+ candles on 1h and 15m timeframes",
        ))
    elif r == Readiness.PARTIALLY_READY:
        warnings.append(R2Warning(
            category = WarningCategory.READINESS,
            severity = WarningSeverity.WARNING,
            message  = "Partial readiness — score penalised, confidence capped",
            detail   = "Not all timeframes have reached minimum candle counts",
        ))
    return warnings


def check_overtrading(session_memory, max_trades: int = 3) -> List[R2Warning]:
    """Warn when trade count suggests overtrading risk."""
    warnings: List[R2Warning] = []
    if session_memory is None:
        return warnings
    n = session_memory.total_trades
    if n >= max_trades + 2:
        warnings.append(R2Warning(
            category = WarningCategory.OVERTRADING,
            severity = WarningSeverity.WARNING,
            message  = f"High trade count for session ({n} trades)",
            detail   = "Review each setup carefully — discipline over frequency",
        ))
    elif n >= max_trades:
        warnings.append(R2Warning(
            category = WarningCategory.OVERTRADING,
            severity = WarningSeverity.CAUTION,
            message  = f"Approaching typical session trade limit ({n}/{max_trades})",
            detail   = "Only A+ setups from here",
        ))
    return warnings


def check_macro_risk(context_state) -> List[R2Warning]:
    """Warn on elevated macro / volatility conditions."""
    warnings: List[R2Warning] = []
    if context_state is None or not context_state.context_valid:
        return warnings
    if context_state.elevated_risk:
        sev = WarningSeverity.CRITICAL if context_state.volatility_risk_level == "extreme" \
              else WarningSeverity.WARNING
        warnings.append(R2Warning(
            category = WarningCategory.MACRO,
            severity = sev,
            message  = f"Elevated macro risk ({context_state.volatility_risk_level})",
            detail   = context_state.risk_note or "Reduce size or stand aside",
        ))
    # Imminent events
    if context_state.upcoming_events:
        from context.context_state import HeadlineImpact
        high = [e for e in context_state.upcoming_events
                if getattr(e, "market_impact", HeadlineImpact.LOW) == HeadlineImpact.HIGH]
        if high:
            name = getattr(high[0], "name", "macro event")
            warnings.append(R2Warning(
                category = WarningCategory.MACRO,
                severity = WarningSeverity.CRITICAL,
                message  = f"High-impact event imminent: {name}",
                detail   = "Consider standing aside until after the release",
            ))
    return warnings


def check_setup_confidence(score, min_score: float = 60.0) -> List[R2Warning]:
    """Warn when setup score is below the quality threshold."""
    warnings: List[R2Warning] = []
    if score is None:
        return warnings
    effective = score.adjusted_total if score.adjusted_total > 0 else score.total
    if effective < 40:
        warnings.append(R2Warning(
            category = WarningCategory.CONFIDENCE,
            severity = WarningSeverity.CRITICAL,
            message  = f"Low-confidence setup — grade {score.grade()} ({effective:.0f}/100)",
            detail   = "Setup does not meet minimum quality threshold",
        ))
    elif effective < min_score:
        warnings.append(R2Warning(
            category = WarningCategory.CONFIDENCE,
            severity = WarningSeverity.WARNING,
            message  = f"Below-threshold setup — grade {score.grade()} ({effective:.0f}/100)",
            detail   = f"Minimum recommended score is {min_score:.0f}",
        ))
    return warnings


def check_context_conflict(state) -> List[R2Warning]:
    """Warn when setup direction and context bias are opposed."""
    warnings: List[R2Warning] = []
    if not hasattr(state, "context_bias") or not hasattr(state, "setup_direction"):
        return warnings
    bias      = getattr(state, "context_bias", "neutral")
    direction = getattr(state, "setup_direction", None)
    if direction is None or bias == "neutral":
        return warnings
    from engine.models import TradeDirection
    is_long    = (direction == TradeDirection.CALLS)
    bias_bull  = (bias == "bullish")
    if is_long != bias_bull:
        warnings.append(R2Warning(
            category = WarningCategory.CONTEXT,
            severity = WarningSeverity.CAUTION,
            message  = f"Context conflict: {direction.value} setup vs {bias} context bias",
            detail   = "News/macro sentiment opposes trade direction — reduce size",
        ))
    return warnings


def check_behavioral_flags(session_memory) -> List[R2Warning]:
    """Surface active behavioral flags from the current session."""
    warnings: List[R2Warning] = []
    if session_memory is None:
        return warnings
    for flag in session_memory.behavioral_flags:
        warnings.append(R2Warning(
            category = WarningCategory.BEHAVIOR,
            severity = WarningSeverity.CAUTION,
            message  = f"Behavioral flag active: {flag}",
            detail   = "Flagged earlier this session — stay disciplined",
        ))
    return warnings


def check_pattern_caution(active_patterns: List[dict]) -> List[R2Warning]:
    """Surface relevant surfaced patterns as advisory context."""
    warnings: List[R2Warning] = []
    for p in active_patterns[:2]:   # surface top 2 only — avoid noise
        ptype = p.get("pattern_type", p.get("description", ""))
        conf  = p.get("confidence_score", 0.0)
        if conf > 0:
            warnings.append(R2Warning(
                category = WarningCategory.PATTERN,
                severity = WarningSeverity.INFO,
                message  = f"Pattern active: {ptype} (confidence {conf:.0%})",
                detail   = "",
            ))
    return warnings


def check_regime_entry_permitted(state) -> List[R2Warning]:
    """Warn when the current regime does not support entries."""
    warnings: List[R2Warning] = []
    if state.regime is None:
        return warnings
    if not state.regime.entry_permitted:
        warnings.append(R2Warning(
            category = WarningCategory.SETUP,
            severity = WarningSeverity.WARNING,
            message  = f"Regime does not support entries: {state.regime.regime.value}",
            detail   = state.regime.caution_note or "Stand aside until regime improves",
        ))
    return warnings


# ──────────────────────────────────────────────
# Aggregator
# ──────────────────────────────────────────────

def collect_warnings(
    state,
    context_state   = None,
    session_memory  = None,
    score           = None,
    active_patterns = None,
    max_trades:      int   = 3,
    min_score:       float = 60.0,
) -> List[R2Warning]:
    """
    Run all warning checks and return a sorted list.
    Sorted: CRITICAL first, then WARNING, CAUTION, INFO.
    """
    all_warnings: List[R2Warning] = []
    all_warnings.extend(check_readiness(state))
    all_warnings.extend(check_regime_entry_permitted(state))
    all_warnings.extend(check_macro_risk(context_state))
    all_warnings.extend(check_setup_confidence(score, min_score))
    all_warnings.extend(check_context_conflict(state))
    all_warnings.extend(check_overtrading(session_memory, max_trades))
    all_warnings.extend(check_behavioral_flags(session_memory))
    all_warnings.extend(check_pattern_caution(active_patterns or []))

    all_warnings.sort(key=lambda w: -w.severity.value)
    return all_warnings


def render_warnings(warnings: List[R2Warning], max_display: int = 6) -> str:
    """Render a warning list to a terminal string block."""
    if not warnings:
        return "  · No active warnings"
    lines = [w.render() for w in warnings[:max_display]]
    if len(warnings) > max_display:
        lines.append(f"  · ... and {len(warnings) - max_display} more")
    return "\n".join(lines)