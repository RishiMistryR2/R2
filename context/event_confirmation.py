"""
context/event_confirmation.py — context-based setup confirmation.

Pure function:
  confirm_setup(setup_direction, context) -> ConfirmationResult

Logic:
  Alignment:    setup direction matches context_bias  → confidence boost
  Mismatch:     setup direction opposes context_bias  → confidence penalty
  High risk:    elevated/extreme volatility           → confidence penalty
  Neutral:      no context or neutral bias            → no adjustment

The multiplier is applied multiplicatively to the setup score's adjusted_total
in scorer.py. A multiplier of 1.0 means no change.

All functions are pure, stateless, and independently testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from context.context_state import ContextState
from engine.models import TradeDirection


# ──────────────────────────────────────────────
# Result type
# ──────────────────────────────────────────────

@dataclass
class ConfirmationResult:
    """
    Output of confirm_setup().

    confidence_multiplier:
        1.0 = no change
        >1.0 = boost (alignment)
        <1.0 = penalty (mismatch or high risk)

    notes:  list of human-readable reasons for the adjustment
    aligned: True when setup direction and context_bias agree
    """
    confidence_multiplier: float       = 1.0
    notes:                 List[str]   = field(default_factory=list)
    aligned:               bool        = False

    def to_dict(self) -> dict:
        return {
            "confidence_multiplier": round(self.confidence_multiplier, 3),
            "notes":                 self.notes,
            "aligned":               self.aligned,
        }


# ──────────────────────────────────────────────
# Tunable parameters
# ──────────────────────────────────────────────

ALIGNMENT_BOOST    = 1.06   # +6% when setup and context agree
MISMATCH_PENALTY   = 0.92   # –8% when setup opposes context
HIGH_RISK_PENALTY  = 0.88   # –12% when volatility is elevated/extreme
STRONG_ALIGNMENT_BONUS = 1.10  # +10% when sentiment is very strong (|s| > 0.6)


# ──────────────────────────────────────────────
# Core function
# ──────────────────────────────────────────────

def confirm_setup(
    setup_direction: TradeDirection,
    context:         Optional[ContextState],
) -> ConfirmationResult:
    """
    Assess whether a trade setup is confirmed, invalidated, or neutral
    given the current external context.

    Parameters
    ----------
    setup_direction : TradeDirection
        CALLS (long) or PUTS (short)
    context : ContextState | None
        Current context snapshot. None → no adjustment (1.0 multiplier).

    Returns
    -------
    ConfirmationResult with multiplier and explanatory notes.
    """
    if context is None or not context.context_valid:
        return ConfirmationResult(
            confidence_multiplier=1.0,
            notes=["no context data — no adjustment"],
        )

    notes:      List[str] = []
    multiplier: float     = 1.0
    aligned:    bool      = False

    bias = context.context_bias      # "bullish" | "bearish" | "neutral"
    risk = context.volatility_risk_level  # "low" | "normal" | "elevated" | "extreme"
    sentiment = context.news_sentiment

    # ── 1. Alignment / mismatch check ────────
    if bias != "neutral":
        setup_is_long  = (setup_direction == TradeDirection.CALLS)
        bias_is_bull   = (bias == "bullish")

        if setup_is_long == bias_is_bull:
            # Aligned
            aligned    = True
            multiplier *= ALIGNMENT_BOOST
            notes.append(
                f"context aligned: {bias} bias confirms "
                f"{'long' if setup_is_long else 'short'} setup"
            )
            # Extra bonus for strong sentiment
            if abs(sentiment) > 0.6:
                multiplier *= (STRONG_ALIGNMENT_BONUS / ALIGNMENT_BOOST)
                notes.append(f"strong sentiment reinforcement ({sentiment:+.2f})")
        else:
            # Mismatch
            multiplier *= MISMATCH_PENALTY
            notes.append(
                f"context conflict: {bias} bias opposes "
                f"{'long' if setup_is_long else 'short'} setup"
            )

    # ── 2. Volatility risk penalty ────────────
    if risk in ("elevated", "extreme"):
        multiplier *= HIGH_RISK_PENALTY
        notes.append(f"volatility risk: {risk} — confidence reduced")

    # ── 3. Imminent macro event dampener ─────
    if context.upcoming_events:
        from context.context_state import HeadlineImpact
        high_impact = any(
            getattr(e, "market_impact", HeadlineImpact.LOW) == HeadlineImpact.HIGH
            for e in context.upcoming_events
        )
        if high_impact and not aligned:
            multiplier *= 0.95
            notes.append("high-impact macro event imminent — slight dampening")

    if not notes:
        notes.append("context neutral — no adjustment")

    return ConfirmationResult(
        confidence_multiplier=round(multiplier, 3),
        notes=notes,
        aligned=aligned,
    )


def score_adjustment_from_context(
    setup_direction: TradeDirection,
    context:         Optional[ContextState],
    base_adjustment: float = 5.0,
) -> float:
    """
    Convert a ConfirmationResult into an additive score adjustment (points).

    Used by scorer.py to apply a ±N-point adjustment to adjusted_total.

    alignment    → +base_adjustment  (default +5)
    mismatch     → –base_adjustment  (default –5)
    neutral      → 0.0
    high risk    → –base_adjustment  (regardless of alignment)
    """
    result = confirm_setup(setup_direction, context)

    if context and context.volatility_risk_level in ("elevated", "extreme"):
        return -base_adjustment

    if result.aligned:
        return +base_adjustment
    if context and context.context_bias != "neutral":
        return -base_adjustment
    return 0.0