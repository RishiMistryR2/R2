"""
engine/confidence_engine.py — R2 Confidence & Thesis Engine v2.2

Builds an evidence-based confidence assessment for the current structural
thesis.  Every confidence score is computed from actual data — never
hardcoded.

Architecture
------------
compute_confidence(fields, regime_name, regime_conf, session_key, entries)
    → ConfidenceResult(score, label, supporting[], contradicting[], conviction)

build_thesis(struct_or_dict, ...)   → full multi-line thesis block
build_compact_thesis(...)           → narrative prose + one-line conviction

Convenience wrappers take a package dict directly:
    build_thesis_from_package(package_dict) → str
    build_compact_thesis_from_package(package_dict) → str

Evidence rules
--------------
Supporting items are added when the data actively supports the thesis.
Contradicting items are added when the data actively opposes the thesis
or degrades signal quality.  Absence of evidence is not evidence — only
genuine conflicts and degrading factors are listed.

Used by
-------
  ai/claude_client._generate_mock_response  (startup_briefing / market_commentary)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    # Type-only import: keeps engine/ free of a runtime dependency on memory/.
    from memory.statistical_intelligence import CalibrationResult


# ──────────────────────────────────────────────────────────────
# Result types
# ──────────────────────────────────────────────────────────────

@dataclass
class ConfidenceState:
    """Authoritative confidence values computed once per update cycle."""

    structural_confidence: float
    calibrated_confidence: float
    regime_confidence: float
    calibration_result: Optional["CalibrationResult"]
    computed_at: datetime
    source_cycle: int


@dataclass
class ConfidenceResult:
    score:         float       # 0.0 – 100.0
    label:         str         # Low | Moderate | Moderate-High | High
    supporting:    list = field(default_factory=list)
    contradicting: list = field(default_factory=list)
    conviction:    str  = "Moderate"

    def __str__(self) -> str:
        return f"Confidence: {self.score:.0f}% ({self.label}) — Conviction: {self.conviction}"


# ──────────────────────────────────────────────────────────────
# Core computation
# ──────────────────────────────────────────────────────────────

def compute_confidence(
    fields:          dict,
    regime_name:     str   = "",
    regime_conf:     float = 0.5,
    session_key:     str   = "",
    entries_allowed: bool  = True,
) -> ConfidenceResult:
    """
    Build evidence-based confidence from the normalised structure fields dict,
    regime metadata, and session context.

    The score starts at 35 (neutral) and each evidence item adjusts it.
    Clamped to 15–90 to avoid false certainty.
    """
    supporting:    list[str] = []
    contradicting: list[str] = []
    score: float = 35.0

    # ── Extract fields ─────────────────────────────────────────
    direction  = fields.get("direction", "")
    state      = fields.get("market_state", "range")
    seq        = fields.get("sequence", "")
    intact     = fields.get("trend_intact", True)
    kl         = fields.get("key_levels", {}) or {}
    vwap       = kl.get("vwap")
    s_high     = kl.get("session_high")
    s_low      = kl.get("session_low")
    hl         = fields.get("last_hl")
    lh         = fields.get("last_lh")
    events_raw = fields.get("recent_events", [])
    missing    = (fields.get("missing_confirmation") or "").strip()

    regime_lower = regime_name.lower().replace("-", "_")
    is_bull_reg  = "bull" in regime_lower or "up" in regime_lower
    is_bear_reg  = "bear" in regime_lower or "down" in regime_lower
    is_chop_reg  = "chop" in regime_lower or "neutral" in regime_lower

    ev_strs = [str(e).lower() for e in events_raw]

    # ── 1. Direction presence ──────────────────────────────────
    if direction in ("uptrend", "downtrend"):
        score += 10

    # ── 2. Sequence strength ───────────────────────────────────
    if seq:
        tokens = seq.split()
        pairs  = len(tokens) // 2
        boost  = min(pairs * 3, 15)
        score += boost

        if direction == "uptrend":
            supporting.append(
                f"Higher highs and higher lows intact — {seq} on 5m."
            )
        elif direction == "downtrend":
            supporting.append(
                f"Lower highs and lower lows intact — {seq} on 5m."
            )

    # ── 3. Trend intact ────────────────────────────────────────
    if direction in ("uptrend", "downtrend"):
        if intact:
            score += 10
            supporting.append("Trend sequence is unbroken — no structural violation detected.")
        else:
            score -= 10
            contradicting.append(
                "Trend sequence is weakening — structure may be breaking down."
            )

    # ── 4. Regime alignment ────────────────────────────────────
    if direction == "uptrend":
        if is_bull_reg:
            score += 12
            supporting.append(
                f"Regime confirms upside bias ({regime_name.replace('_', ' ')}, "
                f"{regime_conf:.0%} confidence)."
            )
        elif is_chop_reg:
            score -= 15
            contradicting.append(
                "Regime classified as Chop — directional edge is uncertain."
            )
        elif is_bear_reg:
            score -= 18
            contradicting.append(
                f"Regime contradicts structure — "
                f"{regime_name.replace('_', ' ')} disagrees with the uptrend reading."
            )

    elif direction == "downtrend":
        if is_bear_reg:
            score += 12
            supporting.append(
                f"Regime confirms downside bias ({regime_name.replace('_', ' ')}, "
                f"{regime_conf:.0%} confidence)."
            )
        elif is_chop_reg:
            score -= 15
            contradicting.append(
                "Regime classified as Chop — directional edge is uncertain."
            )
        elif is_bull_reg:
            score -= 18
            contradicting.append(
                f"Regime contradicts structure — "
                f"{regime_name.replace('_', ' ')} disagrees with the downtrend reading."
            )

    else:
        # Range / no direction
        if is_chop_reg:
            score += 5
            supporting.append("Regime is consistent with range-bound structure.")
        elif is_bull_reg or is_bear_reg:
            score -= 5
            contradicting.append(
                f"Regime ({regime_name.replace('_', ' ')}) suggests direction "
                "but no clear structural sequence has formed."
            )

    # ── 5. Regime confidence quality ──────────────────────────
    if regime_conf >= 0.75:
        score += 5
    elif regime_conf < 0.45:
        score -= 5
        contradicting.append(
            f"Regime confidence is low ({regime_conf:.0%}) — edge may be unreliable."
        )

    # ── 6. Defined stop / reference level ─────────────────────
    if direction == "uptrend" and hl:
        score += 5
        supporting.append(
            f"Higher low at {hl:.2f} provides a defined structural stop reference."
        )
    elif direction == "downtrend" and lh:
        score += 5
        supporting.append(
            f"Lower high at {lh:.2f} provides a defined structural resistance reference."
        )

    # ── 7. Unbroken opposing level ─────────────────────────────
    if direction == "uptrend" and s_high:
        contradicting.append(
            f"Seller resistance at session high ({s_high:.2f}) remains unbroken."
        )
    elif direction == "downtrend" and s_low:
        contradicting.append(
            f"Buyer support at session low ({s_low:.2f}) has not yet been taken."
        )

    # ── 8. Behavioral events ───────────────────────────────────
    if direction == "uptrend":
        for ev in ev_strs:
            if "vwap_reclaim" in ev:
                score += 5
                supporting.append("VWAP reclaim confirmed — buyers regained equilibrium.")
                break
        for ev in ev_strs:
            if "failed_breakdown" in ev:
                score += 5
                supporting.append("Failed breakdown — sellers lost control of support.")
                break
        for ev in ev_strs:
            if "vwap_rejection" in ev:
                score -= 5
                contradicting.append(
                    "Recent VWAP rejection — buyers struggling at the equilibrium level."
                )
                break

    elif direction == "downtrend":
        for ev in ev_strs:
            if "vwap_rejection" in ev:
                score += 5
                supporting.append("VWAP rejection confirmed — sellers defending equilibrium.")
                break
        for ev in ev_strs:
            if "failed_breakout" in ev:
                score += 5
                supporting.append("Failed breakout — buyers lost control of resistance.")
                break
        for ev in ev_strs:
            if "vwap_reclaim" in ev:
                score -= 5
                contradicting.append(
                    "Recent VWAP reclaim by buyers — bearish edge is softening."
                )
                break

    # ── 9. Pending confirmation ────────────────────────────────
    if missing:
        contradicting.append(f"Confirmation still pending: {missing.rstrip('.')}.")

    # ── 10. Session quality ────────────────────────────────────
    _SESSION_ADJ: dict[str, tuple] = {
        "open":        (0,   None),
        "midday":      (-3,  "Midday sessions can produce lower-momentum follow-through."),
        "power_hour":  (-2,  None),
        "premarket":   (-8,  "Pre-market: lower volume reduces signal reliability."),
        "after_hours": (-12, "Extended-hours trading — thin volume degrades signal quality."),
        "closed":      (-18, "Market is closed — all levels are reference only."),
    }
    adj, sess_note = _SESSION_ADJ.get(session_key, (0, None))
    score += adj
    if sess_note:
        contradicting.append(sess_note)

    # ── 11. Entries permitted ──────────────────────────────────
    # Entry readiness is an action filter, not evidence about directional
    # thesis strength.  Whether entries are permitted or prohibited tells us
    # nothing about whether the structural thesis is correct — it only tells
    # us whether we are allowed to act on it right now.
    #
    # entries_allowed is intentionally kept as a parameter for backward
    # compatibility with existing callers, but it no longer affects the
    # confidence score in either direction.
    #
    # The dashboard displays entry readiness separately under its own label
    # ("Entry readiness: Disabled") rather than folding it into this score.

    # ── Clamp ──────────────────────────────────────────────────
    score = max(15.0, min(90.0, score))

    # ── Labels ─────────────────────────────────────────────────
    label, conviction = _label_for(score)

    return ConfidenceResult(
        score=round(score, 1),
        label=label,
        supporting=supporting,
        contradicting=contradicting,
        conviction=conviction,
    )


def _label_for(score: float) -> tuple[str, str]:
    """(label, conviction) for a confidence score."""
    if score >= 75:
        return "High", "High"
    if score >= 60:
        return "Moderate-High", "Moderate"
    if score >= 45:
        return "Moderate", "Moderate"
    return "Low", "Low"


def _displayed_confidence(
    cr:                    ConfidenceResult,
    final_confidence:      Optional[float],
    structural_confidence: Optional[float],
) -> tuple[float, str, str, str]:
    """
    v2.7 — the score shown in thesis text.

    When the cycle's stored Final Conviction is supplied, show it (with its
    own label/conviction) and explain any historical adjustment.  Otherwise
    show the structural score computed here.

    Returns (score, label, conviction, adjustment_note).
    """
    if final_confidence is None:
        return cr.score, cr.label, cr.conviction, ""
    final      = float(final_confidence)
    structural = float(structural_confidence
                       if structural_confidence is not None else cr.score)
    label, conviction = _label_for(final)
    delta = final - structural
    note  = ""
    if abs(delta) >= 0.5:
        note = (f"structural {structural:.0f}%, "
                f"historical adjustment {delta:+.0f}")
    return final, label, conviction, note


# ──────────────────────────────────────────────────────────────
# Full thesis block
# ──────────────────────────────────────────────────────────────

def build_thesis(
    struct_or_dict,
    regime_name:     str   = "",
    regime_conf:     float = 0.5,
    session                = None,
    entries_allowed: bool  = True,
    current_price:   Optional[float] = None,
    symbol:          str   = "SPY",
    final_confidence:      Optional[float] = None,
    structural_confidence: Optional[float] = None,
) -> str:
    """
    Full multi-line thesis block:
      Thesis statement
      Confidence: N% (Label)
      Supporting: · ...
      Contradicting: · ...
      Conviction: X.

    Suitable for startup_briefing where space allows structured output.

    v2.7: pass final_confidence (and structural_confidence) from the
    cycle's ConfidenceState to display the stored value instead of the
    score computed here.  Evidence lists always come from the structural
    evaluation.
    """
    from engine.narrative_engine import _to_fields, _session_key, _SESS_PREFIX

    sess_key = _session_key(session)
    fields   = _to_fields(struct_or_dict)
    direction = fields.get("direction", "")
    state     = fields.get("market_state", "range")
    seq       = fields.get("sequence", "")
    intact    = fields.get("trend_intact", True)
    kl        = fields.get("key_levels", {}) or {}
    s_high    = kl.get("session_high")
    s_low     = kl.get("session_low")
    hl        = fields.get("last_hl")
    lh        = fields.get("last_lh")
    miss      = (fields.get("missing_confirmation") or "").strip()

    sess_pfx = _SESS_PREFIX.get(sess_key, "")

    # ── Thesis statement ───────────────────────────────────────
    if direction == "uptrend" and seq:
        thesis = f"{sess_pfx}{symbol} is trending higher — {seq} structure on 5m."
    elif direction == "uptrend":
        thesis = f"{sess_pfx}{symbol} is in an uptrend."
    elif direction == "downtrend" and seq:
        thesis = f"{sess_pfx}{symbol} is trending lower — {seq} structure on 5m."
    elif direction == "downtrend":
        thesis = f"{sess_pfx}{symbol} is in a downtrend."
    elif state == "pullback" and direction:
        if "up" in direction:
            anchor = hl or s_low
            anchor_str = f" Higher low at {anchor:.2f}." if anchor else ""
            thesis = f"{sess_pfx}{symbol} is pulling back within an uptrend.{anchor_str}"
        else:
            anchor = lh or s_high
            anchor_str = f" Lower high at {anchor:.2f}." if anchor else ""
            thesis = f"{sess_pfx}{symbol} is bouncing within a downtrend.{anchor_str}"
    elif state == "range" and s_high and s_low:
        thesis = (
            f"{sess_pfx}{symbol} is range-bound between "
            f"{s_low:.2f} and {s_high:.2f}."
        )
    elif state == "breakout":
        thesis = (
            f"{sess_pfx}{symbol} broke above {s_high:.2f} — expansion phase underway."
            if s_high else f"{sess_pfx}{symbol} is in a breakout."
        )
    elif state == "breakdown":
        thesis = (
            f"{sess_pfx}{symbol} broke below {s_low:.2f} — sellers accelerating."
            if s_low else f"{sess_pfx}{symbol} is in a breakdown."
        )
    else:
        thesis = (
            f"{sess_pfx}{symbol} has no clear directional thesis at this time."
        )

    # ── Compute confidence ─────────────────────────────────────
    cr = compute_confidence(
        fields, regime_name, regime_conf, sess_key, entries_allowed
    )

    # ── Key level highlight ────────────────────────────────────
    level_line = _key_level_highlight(direction, state, kl, hl, lh)

    # ── Confirmation + invalidation ───────────────────────────
    confirm_line = _confirmation_line(fields, direction, state, kl, hl, lh)
    invalid_line = _invalidation_line(fields, direction, state, kl, hl, lh)

    # ── Format ────────────────────────────────────────────────
    lines: list[str] = [thesis]

    if level_line:
        lines.append(level_line)

    shown, shown_label, shown_conv, adj_note = _displayed_confidence(
        cr, final_confidence, structural_confidence,
    )

    lines.append("")
    lines.append(f"Confidence: {shown:.0f}% ({shown_label})")
    if adj_note:
        lines.append(f"  ({adj_note[0].upper()}{adj_note[1:]}.)")

    if cr.supporting:
        lines.append("")
        lines.append("Supporting:")
        for ev in cr.supporting:
            lines.append(f"  · {ev}")

    if cr.contradicting:
        lines.append("")
        lines.append("Contradicting:")
        for ev in cr.contradicting:
            lines.append(f"  · {ev}")

    if confirm_line or invalid_line:
        lines.append("")
    if confirm_line:
        lines.append(confirm_line)
    if invalid_line:
        lines.append(invalid_line)

    lines.append("")
    lines.append(f"Conviction: {shown_conv}.")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# Compact thesis (inline — for market commentary)
# ──────────────────────────────────────────────────────────────

def build_compact_thesis(
    struct_or_dict,
    regime_name:     str   = "",
    regime_conf:     float = 0.5,
    session                = None,
    entries_allowed: bool  = True,
    current_price:   Optional[float] = None,
    symbol:          str   = "SPY",
    final_confidence:      Optional[float] = None,
    structural_confidence: Optional[float] = None,
) -> str:
    """
    Compact form: prose narrative (from narrative_engine) with a one-line
    confidence + conviction suffix appended.  Suitable for market_commentary
    where brevity matters and dashboard wrapping is constrained.
    """
    from engine.narrative_engine import (
        build_narrative, _to_fields, _session_key
    )

    sess_key = _session_key(session)
    fields   = _to_fields(struct_or_dict)

    narrative = build_narrative(
        struct_or_dict,
        session         = session,
        regime          = regime_name,
        entries_allowed = entries_allowed,
        current_price   = current_price,
        symbol          = symbol,
    )

    cr = compute_confidence(
        fields, regime_name, regime_conf, sess_key, entries_allowed
    )

    shown, shown_label, shown_conv, adj_note = _displayed_confidence(
        cr, final_confidence, structural_confidence,
    )
    detail = f"{shown_label}; {adj_note}" if adj_note else shown_label

    return (
        f"{narrative}\n"
        f"Confidence: {shown:.0f}% ({detail}) — Conviction: {shown_conv}."
    )


# ──────────────────────────────────────────────────────────────
# Package-dict convenience wrappers
# ──────────────────────────────────────────────────────────────

def build_thesis_from_package(package_dict: dict) -> str:
    """Full thesis block from a StatePackage.to_dict() result."""
    m, struct, regime_name, regime_conf, sess, entries, price = _unpack(package_dict)
    final, structural = _stored_confidence(package_dict)
    return build_thesis(
        struct,
        regime_name     = regime_name,
        regime_conf     = regime_conf,
        session         = sess,
        entries_allowed = entries,
        current_price   = price,
        final_confidence      = final,
        structural_confidence = structural,
    )


def build_compact_thesis_from_package(package_dict: dict) -> str:
    """Compact thesis from a StatePackage.to_dict() result."""
    m, struct, regime_name, regime_conf, sess, entries, price = _unpack(package_dict)
    final, structural = _stored_confidence(package_dict)
    return build_compact_thesis(
        struct,
        regime_name     = regime_name,
        regime_conf     = regime_conf,
        session         = sess,
        entries_allowed = entries,
        current_price   = price,
        final_confidence      = final,
        structural_confidence = structural,
    )


def _stored_confidence(d: dict) -> tuple[Optional[float], Optional[float]]:
    """
    (calibrated, structural) from package["confidence"] — the v2.7 block
    written by state_packager.  (None, None) when the package has none.
    """
    blk = d.get("confidence") or {}
    try:
        final = blk.get("calibrated")
        struc = blk.get("structural")
        return (float(final) if final is not None else None,
                float(struc) if struc is not None else None)
    except (TypeError, ValueError):
        return None, None


def _unpack(d: dict):
    """Extract common fields from a package dict."""
    market      = d.get("market", {}) or {}
    struct      = d.get("structure", {}) or {}
    regime_d    = market.get("regime", {}) or {}
    regime_name = regime_d.get("name", "")
    regime_conf = float(regime_d.get("confidence", 0.5))
    sess        = market.get("market_session", "")
    entries     = (market.get("session", {}) or {}).get("entries_allowed", True)
    ind_5m      = (market.get("indicators", {}) or {}).get("5m", {}) or {}
    price       = ind_5m.get("close")
    return market, struct, regime_name, regime_conf, sess, entries, price


# ──────────────────────────────────────────────────────────────
# Thesis-block helpers
# ──────────────────────────────────────────────────────────────

def _key_level_highlight(direction, state, kl, hl, lh) -> str:
    """One-line summary of the single most important level right now."""
    s_high  = kl.get("session_high")
    s_low   = kl.get("session_low")
    vwap    = kl.get("vwap")

    if direction == "uptrend":
        anchor = hl or s_low
        resist = s_high
        parts  = []
        if anchor:
            parts.append(f"Key support: {anchor:.2f}")
        if resist:
            parts.append(f"key resistance: {resist:.2f}")
        if vwap:
            parts.append(f"VWAP: {vwap:.2f}")
        return "  ".join(parts) + "." if parts else ""

    if direction == "downtrend":
        anchor = lh or s_high
        support = s_low
        parts   = []
        if anchor:
            parts.append(f"Key resistance: {anchor:.2f}")
        if support:
            parts.append(f"key support: {support:.2f}")
        if vwap:
            parts.append(f"VWAP: {vwap:.2f}")
        return "  ".join(parts) + "." if parts else ""

    if s_high and s_low:
        base = f"Range: {s_low:.2f} – {s_high:.2f}"
        if vwap:
            base += f"  VWAP: {vwap:.2f}"
        return base + "."

    return ""


def _confirmation_line(fields, direction, state, kl, hl, lh) -> str:
    """Return the confirmation requirement sentence."""
    missing = (fields.get("missing_confirmation") or "").strip()
    if missing:
        low = missing.lower()
        if low.startswith("need"):
            return "Confirmation requires" + missing[4:].rstrip(".") + "."
        return missing.rstrip(".") + "."

    s_high = kl.get("session_high")
    s_low  = kl.get("session_low")

    if direction == "uptrend" and s_high:
        return f"Confirmation requires a break and hold above {s_high:.2f}."
    if direction == "downtrend" and s_low:
        return f"Confirmation requires a break and hold below {s_low:.2f}."
    if state in ("range", "compression") and s_high and s_low:
        return (
            f"Confirmation requires a directional close beyond "
            f"{s_high:.2f} (bullish) or {s_low:.2f} (bearish)."
        )
    return ""


def _invalidation_line(fields, direction, state, kl, hl, lh) -> str:
    """Return the invalidation condition sentence."""
    s_high = kl.get("session_high")
    s_low  = kl.get("session_low")

    if direction == "uptrend":
        anchor = hl or s_low
        if anchor:
            label = (
                f"higher low ({anchor:.2f})" if hl
                else f"session low ({anchor:.2f})"
            )
            return f"Invalidation: close below the {label}."
    elif direction == "downtrend":
        anchor = lh or s_high
        if anchor:
            label = (
                f"lower high ({anchor:.2f})" if lh
                else f"session high ({anchor:.2f})"
            )
            return f"Invalidation: reclaim of the {label}."
    elif state in ("range", "compression") and s_high and s_low:
        return (
            f"Invalidation of range: confirmed close above {s_high:.2f} "
            f"(bullish) or below {s_low:.2f} (bearish)."
        )
    elif state == "breakout" and s_high:
        return f"Invalidation: close back below {s_high:.2f}."
    elif state == "breakdown" and s_low:
        return f"Invalidation: reclaim of {s_low:.2f}."
    return ""