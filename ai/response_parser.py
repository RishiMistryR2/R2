"""
ai/response_parser.py — parse and validate R2 AI responses.

Responsibilities:
  1. Wrap raw Claude text in a structured ParsedResponse
  2. Validate against R2 persona guardrails (no forbidden phrases)
  3. Extract lightweight metadata (recommendation_type, confidence_label)
  4. Provide clean fallback for API failures

ParsedResponse is the only type that should cross the AI layer boundary.
Callers never see raw API dicts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

from ai.r2_persona import (
    FORBIDDEN_PHRASES,
    RECOMMENDATION_TYPES,
    CONFIDENCE_LABELS,
    is_advisory_response,
    RISK_FLAGS,
)


# ──────────────────────────────────────────────
# Output type
# ──────────────────────────────────────────────

@dataclass
class ParsedResponse:
    """
    Structured wrapper around a raw Claude response.

    text:                 clean response text (stripped)
    recommendation_type:  category from RECOMMENDATION_TYPES
    confidence_label:     high / medium / low / none
    risk_flags:           flags from the prompt context
    guardrail_passed:     True if no forbidden phrases found
    guardrail_violations: list of forbidden phrases found (if any)
    is_fallback:          True when API was unavailable
    generated_at:         ISO timestamp
    raw_text:             original unmodified text
    """
    text:                  str
    recommendation_type:   str       = "market_commentary"
    confidence_label:      str       = "medium"
    risk_flags:            List[str] = field(default_factory=list)
    guardrail_passed:      bool      = True
    guardrail_violations:  List[str] = field(default_factory=list)
    is_fallback:           bool      = False
    generated_at:          str       = ""
    raw_text:              str       = ""

    def __post_init__(self):
        if not self.generated_at:
            self.generated_at = datetime.now(tz=timezone.utc).isoformat()
        if not self.raw_text:
            self.raw_text = self.text

    def to_dict(self) -> dict:
        return {
            "text":                 self.text,
            "recommendation_type":  self.recommendation_type,
            "confidence_label":     self.confidence_label,
            "risk_flags":           self.risk_flags,
            "guardrail_passed":     self.guardrail_passed,
            "guardrail_violations": self.guardrail_violations,
            "is_fallback":          self.is_fallback,
            "generated_at":         self.generated_at,
        }

    @property
    def has_risk_flags(self) -> bool:
        return len(self.risk_flags) > 0

    @property
    def is_usable(self) -> bool:
        """True if the response is safe to show to the user."""
        return not self.is_fallback and len(self.text) > 0


# ──────────────────────────────────────────────
# Parser
# ──────────────────────────────────────────────

def parse_response(
    raw_text:            str,
    recommendation_type: str       = "market_commentary",
    risk_flags:          List[str] = None,
) -> ParsedResponse:
    """
    Parse a raw Claude response string into a ParsedResponse.

    Steps:
      1. Strip whitespace
      2. Run guardrail check (forbidden phrases)
      3. Infer confidence label from text signals
      4. Return structured ParsedResponse
    """
    text = raw_text.strip() if raw_text else ""

    violations    = _find_violations(text)
    guardrail_ok  = len(violations) == 0
    clean_text    = _redact_violations(text, violations) if violations else text
    confidence    = _infer_confidence(clean_text)

    return ParsedResponse(
        text                 = clean_text,
        recommendation_type  = recommendation_type,
        confidence_label     = confidence,
        risk_flags           = risk_flags or [],
        guardrail_passed     = guardrail_ok,
        guardrail_violations = violations,
        is_fallback          = False,
        raw_text             = raw_text,
    )


def make_fallback_response(
    reason:              str       = "API unavailable",
    recommendation_type: str       = "fallback",
    risk_flags:          List[str] = None,
) -> ParsedResponse:
    """
    Create a safe fallback ParsedResponse when the API is unavailable.
    R2 never crashes — it degrades to a canned but honest message.
    """
    text = (
        f"Analysis unavailable ({reason}). "
        "Review the dashboard data directly. "
        "All risk rules remain in effect."
    )
    return ParsedResponse(
        text                 = text,
        recommendation_type  = "fallback",
        confidence_label     = "none",
        risk_flags           = risk_flags or [],
        guardrail_passed     = True,
        guardrail_violations = [],
        is_fallback          = True,
        raw_text             = text,
    )


def validate_response(response: ParsedResponse) -> List[str]:
    """Return a list of validation issues (empty = all good)."""
    issues: List[str] = []
    if not response.text:
        issues.append("Empty response text")
    if response.is_fallback:
        issues.append(f"Fallback response: {response.text}")
    if response.guardrail_violations:
        issues.append(f"Guardrail violations: {response.guardrail_violations}")
    if (response.recommendation_type not in RECOMMENDATION_TYPES and
            response.recommendation_type != "fallback"):
        issues.append(f"Unknown recommendation_type: {response.recommendation_type}")
    return issues


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _find_violations(text: str) -> List[str]:
    """Return list of forbidden phrases found in text (case-insensitive)."""
    lower = text.lower()
    return [p for p in FORBIDDEN_PHRASES if p in lower]


def _redact_violations(text: str, violations: List[str]) -> str:
    """Replace forbidden phrases with [redacted] and prepend a note."""
    import re
    redacted = text
    for phrase in violations:
        redacted = re.sub(re.escape(phrase), "[redacted]", redacted, flags=re.IGNORECASE)
    note = "[Note: response contained advisory language that was adjusted.]\n\n"
    return note + redacted


def _infer_confidence(text: str) -> str:
    """
    Infer a confidence label from linguistic signals in the response.

    high:   decisive language ("clearly", "strongly", "confirmed")
    low:    uncertainty language ("unclear", "uncertain", "mixed signals")
    medium: default
    """
    lower = text.lower()
    high_signals = ["clearly", "strongly", "confirmed", "well-defined",
                    "high confidence", "strong setup", "all timeframes aligned"]
    low_signals  = ["unclear", "uncertain", "mixed signals", "low confidence",
                    "no clear", "choppy", "insufficient", "limited data",
                    "not ready", "caution", "avoid", "stand aside"]

    high_count = sum(1 for s in high_signals if s in lower)
    low_count  = sum(1 for s in low_signals  if s in lower)

    if low_count >= 2:
        return "low"
    if high_count >= 2 and low_count == 0:
        return "high"
    return "medium"