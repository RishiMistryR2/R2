"""
ai/prompt_builder.py — structured prompt templates for R2 AI interactions.

Pure functions.  No I/O, no state.
All functions return a (system_prompt, user_message) tuple ready for the
Claude API.

Request types:
  startup_briefing   — session-aware orientation (pre-market / after-hours / etc.)
  market_commentary  — live market check
  setup_explanation  — explain a scored setup
  trade_review       — review an active trade
  warning_note       — explain an active warning
  end_of_day         — session reflection
  user_question      — answer a specific question

Format design (v2.0A):
  Context is structured readable text, NOT raw JSON.
  Structure section leads every prompt — explicitly labelled to guide model output.
  Indicators are labelled "secondary — do not lead with these values".
  Request instructions include REQUIRED ORDER for structure-first commentary.
  Session wording in the request matches the actual current session.
"""

from __future__ import annotations

import json
from typing import Optional, Tuple

from ai.r2_persona import (
    R2_SYSTEM_PROMPT,
    R2_BRIEF_SYSTEM_PROMPT,
    RISK_FLAGS,
)
from ai.state_packager import StatePackage

PromptPair = Tuple[str, str]
_MAX_CONTEXT_CHARS = 5500


# ──────────────────────────────────────────────
# Public builders
# ──────────────────────────────────────────────

def build_startup_briefing_prompt(package: StatePackage) -> PromptPair:
    """
    Session-aware orientation prompt.
    Uses actual market_session from the package for accurate wording.
    """
    ctx          = _format_context(package)
    risk         = _risk_block(package.risk_flags)
    session_word = _get_session_word(package)     # e.g. "after-hours", "pre-market"
    watch_phrase = _get_watch_phrase(package)     # e.g. "how after-hours is developing"

    user = (
        f"<context>\n{ctx}\n</context>\n"
        f"{risk}"
        "<request>\n"
        f"Provide a concise {session_word} briefing.\n"
        "REQUIRED ORDER:\n"
        "  1. Lead with the structural picture — trend direction, HH/HL/LH/LL "
        "sequence, market state.\n"
        "  2. Describe what buyers and sellers are doing at the key structural levels.\n"
        "  3. State what confirmation is still needed.\n"
        f"  4. Note any elevated risks and {watch_phrase}.\n"
        "  5. Only then mention indicators (RSI, VWAP) if they add context beyond structure.\n"
        "Do not use any session word that contradicts the actual session shown "
        f"in the context ({session_word}).\n"
        "Be specific. Lead with what the structure shows.\n"
        "</request>"
    )
    return R2_SYSTEM_PROMPT, user


def build_market_commentary_prompt(package: StatePackage) -> PromptPair:
    """
    Live market check — called on regime/narrative changes and periodic updates.
    Structure leads. Indicators support.
    """
    ctx  = _format_context(package)
    risk = _risk_block(package.risk_flags)

    user = (
        f"<context>\n{ctx}\n</context>\n"
        f"{risk}"
        "<request>\n"
        "Describe the current market state in 2-4 sentences.\n"
        "REQUIRED ORDER:\n"
        "  1. Lead with the structural observation — what the HH/HL/LH/LL sequence "
        "and market state show right now.\n"
        "  2. Describe what buyers and sellers are doing at the key structural levels.\n"
        "  3. State what confirmation is still missing.\n"
        "  4. Only then mention indicators (RSI, VWAP) IF they add context beyond structure.\n"
        "Do NOT open with 'Price is above VWAP' or 'RSI is in a healthy zone.'\n"
        "Do NOT lead with regime or narrative labels — lead with what the bars show.\n"
        "</request>"
    )
    return R2_BRIEF_SYSTEM_PROMPT, user


def build_setup_explanation_prompt(
    package:   StatePackage,
    direction: Optional[str] = None,
) -> PromptPair:
    ctx      = _format_context(package)
    risk     = _risk_block(package.risk_flags)
    dir_note = f"Direction under consideration: {direction}\n" if direction else ""

    user = (
        f"<context>\n{ctx}\n</context>\n"
        f"{risk}"
        f"{dir_note}"
        "<request>\n"
        "Explain this setup as if briefing a disciplined trader.\n"
        "Lead with the structural context (what structure supports or opposes the setup).\n"
        "Include: score rationale, key reasons FOR and AGAINST, stop logic, R:R assessment.\n"
        "Reference specific levels (HL, LH, session H/L) rather than just RSI/VWAP values.\n"
        "Flag any structural concerns clearly. Do not recommend execution.\n"
        "</request>"
    )
    return R2_SYSTEM_PROMPT, user


def build_trade_review_prompt(package: StatePackage) -> PromptPair:
    ctx  = _format_context(package)
    risk = _risk_block(package.risk_flags)

    user = (
        f"<context>\n{ctx}\n</context>\n"
        f"{risk}"
        "<request>\n"
        "Review the active trade.\n"
        "Is the structural thesis still intact? Are key levels (HL, stop) still holding?\n"
        "Are any stops at risk? What should the trader watch structurally?\n"
        "If stop flags are triggered, say so clearly.\n"
        "Do not recommend execution. Observations only.\n"
        "</request>"
    )
    return R2_SYSTEM_PROMPT, user


def build_warning_explanation_prompt(
    package:     StatePackage,
    warning_msg: str,
) -> PromptPair:
    ctx  = _format_context(package)
    risk = _risk_block(package.risk_flags)

    user = (
        f"<context>\n{ctx}\n</context>\n"
        f"{risk}"
        f"<warning>\n{warning_msg}\n</warning>\n"
        "<request>\n"
        "Explain this warning in plain language.\n"
        "Why is it relevant given the current structure and context?\n"
        "Be concise. No more than 3 sentences.\n"
        "</request>"
    )
    return R2_BRIEF_SYSTEM_PROMPT, user


def build_end_of_day_prompt(package: StatePackage) -> PromptPair:
    ctx  = _format_context(package)
    risk = _risk_block(package.risk_flags)

    user = (
        f"<context>\n{ctx}\n</context>\n"
        f"{risk}"
        "<request>\n"
        "Provide a brief end-of-day reflection.\n"
        "Cover: session P&L, execution quality, notable mistakes, "
        "and one actionable lesson for next session.\n"
        "Distinguish between execution quality and outcome quality.\n"
        "Keep it under 150 words.\n"
        "</request>"
    )
    return R2_SYSTEM_PROMPT, user


def build_user_question_prompt(
    package:  StatePackage,
    question: str,
) -> PromptPair:
    ctx  = _format_context(package)
    risk = _risk_block(package.risk_flags)

    user = (
        f"<context>\n{ctx}\n</context>\n"
        f"{risk}"
        f"<question>\n{question.strip()}\n</question>\n"
        "<request>\n"
        "Answer the question directly. Lead with the answer, then explain.\n"
        "Where relevant, reference structural context (levels, sequence, state) "
        "rather than just indicator values.\n"
        "Be concise. Advisory only — do not recommend execution.\n"
        "</request>"
    )
    return R2_SYSTEM_PROMPT, user


# ──────────────────────────────────────────────
# Session helpers
# ──────────────────────────────────────────────

# Maps market_session.value → human word for request text
_SESSION_WORDS = {
    "premarket":   "pre-market",
    "open":        "market open",
    "midday":      "midday",
    "power_hour":  "power hour",
    "after_hours": "after-hours",
    "closed":      "end-of-session",
}

# Maps market_session.value → "what to watch" phrase
_SESSION_WATCH = {
    "premarket":   "what to watch for at the open",
    "open":        "how the open is developing",
    "midday":      "how the midday session is progressing",
    "power_hour":  "what to expect in the power hour",
    "after_hours": "how after-hours price action is developing",
    "closed":      "what to watch for next session",
}


def _get_session_word(package: StatePackage) -> str:
    """
    Get human-readable session word from the package for prompt wording.
    Uses the actual market_session value (after_hours, premarket, etc.)
    rather than the lossy SessionPhase mapping.
    """
    d   = package.to_dict()
    key = d.get("market", {}).get("market_session", "")
    return _SESSION_WORDS.get(key, "session")


def _get_watch_phrase(package: StatePackage) -> str:
    """Get the 'watch for' phrase appropriate to the current session."""
    d   = package.to_dict()
    key = d.get("market", {}).get("market_session", "")
    return _SESSION_WATCH.get(key, "what to watch for in the current session")


# ──────────────────────────────────────────────
# Context formatter  (structure-first)
# ──────────────────────────────────────────────

def _format_context(package: StatePackage, max_chars: int = _MAX_CONTEXT_CHARS) -> str:
    """
    Format StatePackage as structured readable text with structure leading.

    Section order:
      1. MARKET STRUCTURE  — always first, explicitly labelled to lead response
      2. REGIME & SESSION  — includes actual market_session label
      3. INDICATOR SUPPORT — explicitly labelled secondary
      4. SETUP
      5. ACTIVE TRADE
      6. MACRO & CONTEXT
      7. SESSION MEMORY
      8. ACTIVE WARNINGS
    """
    d = package.to_dict()
    parts: list = []

    struct = d.get("structure")
    if struct:
        parts.append(_fmt_structure(struct))

    market = d.get("market")
    if market:
        parts.append(_fmt_regime(market))
        ind_block = _fmt_indicators(market)
        if ind_block:
            parts.append(ind_block)

    setup = d.get("setup")
    if setup:
        parts.append(_fmt_setup(setup))

    trade = d.get("trade")
    if trade:
        parts.append(_fmt_trade(trade))

    ctx = d.get("context")
    if ctx:
        parts.append(_fmt_macro(ctx))

    session = d.get("session")
    if session:
        parts.append(_fmt_session_memory(session))

    warnings = d.get("warnings")
    if warnings:
        parts.append(_fmt_warnings(warnings))

    text = "\n\n".join(p for p in parts if p and p.strip())

    if not text.strip():
        text = json.dumps(d, indent=2, default=str)

    if len(text) > max_chars:
        text = text[:max_chars] + "\n... [truncated for token budget]"
    return text


def _fmt_structure(s: dict) -> str:
    """Format the structure block — always leads the context."""
    lines = ["=== MARKET STRUCTURE (lead your response with this) ==="]

    direction = s.get("direction", "unknown")
    seq       = s.get("sequence", "—")
    state     = s.get("market_state", "unknown")
    intact    = "yes" if s.get("trend_intact", True) else "NO — trend broken"
    lines.append(
        f"Direction: {direction}  |  Sequence: {seq}  "
        f"|  State: {state}  |  Intact: {intact}"
    )

    narr = s.get("structure_narrative", "")
    if narr:
        lines.append(f"\nNarrative:\n  {narr}")

    buyer = s.get("buyer_context", "")
    if buyer:
        lines.append(f"\nBuyer context:\n  {buyer}")

    seller = s.get("seller_context", "")
    if seller:
        lines.append(f"Seller context:\n  {seller}")

    missing = s.get("missing_confirmation", "")
    if missing:
        lines.append(f"Confirmation needed:\n  {missing}")

    kl = s.get("key_levels", {})
    if kl:
        kl_str = "  |  ".join(f"{k}={v}" for k, v in kl.items())
        lines.append(f"\nKey levels:  {kl_str}")

    events = s.get("recent_events", [])
    if events:
        lines.append("\nRecent structural events:")
        for ev in events[:4]:
            lines.append(f"  - {ev}")

    hh = s.get("last_hh")
    hl = s.get("last_hl")
    lh = s.get("last_lh")
    ll = s.get("last_ll")
    swing_parts = []
    if hh: swing_parts.append(f"last HH={hh:.2f}")
    if hl: swing_parts.append(f"last HL={hl:.2f}")
    if lh: swing_parts.append(f"last LH={lh:.2f}")
    if ll: swing_parts.append(f"last LL={ll:.2f}")
    if swing_parts:
        lines.append(f"\nSwing prices:  {'  |  '.join(swing_parts)}")

    return "\n".join(lines)


def _fmt_regime(market: dict) -> str:
    """Format regime, narrative, and session including the actual market_session."""
    lines = ["=== REGIME & SESSION ==="]

    # Show actual market session first (more specific than phase)
    ms = market.get("market_session", "") or "unknown"
    ms_label = ms.replace("_", " ").upper()
    lines.append(f"Market session: {ms_label}")

    regime = market.get("regime", {})
    if regime:
        name = regime.get("name", "").replace("_", " ")
        conf = regime.get("confidence", 0)
        ep   = "yes" if regime.get("entry_permitted") else "no"
        lines.append(f"Regime: {name} ({conf:.0%})  |  Entry permitted: {ep}")

    narrative = market.get("narrative", {})
    if narrative:
        cond    = narrative.get("condition", "").replace("_", " ")
        posture = narrative.get("trade_posture", "")
        ncf     = narrative.get("confidence", 0)
        lines.append(f"Narrative: {cond} ({ncf:.0%})  |  Posture: {posture}")

    session = market.get("session", {})
    if session:
        readiness = session.get("readiness", "").replace("_", " ")
        mkt_open  = "yes" if session.get("market_open") else "no"
        entries   = "yes" if session.get("entries_allowed") else "no"
        lines.append(
            f"Readiness: {readiness}  |  Market open: {mkt_open}  |  Entries: {entries}"
        )

    return "\n".join(lines)


def _fmt_indicators(market: dict) -> str:
    indicators = market.get("indicators", {})
    if not indicators:
        return ""
    lines = [
        "=== INDICATOR SUPPORT (secondary — do not lead with these values) ===",
    ]
    for tf in ("5m", "15m", "1h", "1d"):
        ind = indicators.get(tf)
        if not ind:
            continue
        c       = ind.get("close", 0)
        rsi     = ind.get("rsi", 0)
        atr     = ind.get("atr", 0)
        vwap    = ind.get("vwap", 0)
        stack   = ind.get("ema_stack", "")
        macd    = ind.get("macd_bias", "")
        vs_vwap = ind.get("vs_vwap", "")
        lines.append(
            f"  {tf:>3s}: close={c:.2f}  RSI={rsi:.0f}  ATR={atr:.2f}  "
            f"VWAP={vwap:.2f} ({vs_vwap})  EMA={stack}  MACD={macd}"
        )
    return "\n".join(lines)


def _fmt_setup(setup: dict) -> str:
    total = setup.get("total", 0)
    adj   = setup.get("adjusted_total", total)
    grade = setup.get("grade", "")
    score = adj if adj > 0 else total
    lines = [f"=== SETUP: {score:.0f}/100 ({grade}) ==="]
    for comp in ("trend_alignment", "momentum_quality", "structure_context", "entry_timing"):
        val = setup.get(comp, 0)
        lines.append(f"  {comp.replace('_', ' ').title()}: {val:.0f}")
    return "\n".join(lines)


def _fmt_trade(trade: dict) -> str:
    lines = ["=== ACTIVE TRADE ==="]
    for k in ("direction", "lifecycle", "entry_price", "stop_price",
              "target_price", "pnl_pct",
              "thesis_stop_triggered", "time_stop_triggered", "premium_stop_triggered"):
        if k in trade and trade[k] is not None:
            lines.append(f"  {k}: {trade[k]}")
    return "\n".join(lines)


def _fmt_macro(ctx: dict) -> str:
    lines = ["=== MACRO & CONTEXT ==="]
    for k in ("vix_level", "vix_regime", "context_bias",
              "volatility_risk_level", "elevated_risk", "news_sentiment", "risk_note"):
        if ctx.get(k) is not None:
            lines.append(f"  {k}: {ctx[k]}")
    return "\n".join(lines)


def _fmt_session_memory(session: dict) -> str:
    lines = ["=== SESSION MEMORY ==="]
    for k in ("total_trades", "winning_trades", "daily_pnl_pct",
              "dominant_narrative", "lesson"):
        if session.get(k) is not None:
            lines.append(f"  {k}: {session[k]}")
    return "\n".join(lines)


def _fmt_warnings(warnings: dict) -> str:
    items = warnings.get("warnings", [])
    if not items:
        return ""
    lines = ["=== ACTIVE WARNINGS ==="]
    for w in items[:4]:
        sev = w.get("severity", "").upper()
        msg = w.get("message", "")
        lines.append(f"  [{sev}] {msg}")
    return "\n".join(lines)


# ──────────────────────────────────────────────
# Legacy JSON serialiser  (kept for compat)
# ──────────────────────────────────────────────

def _serialise(
    package:   StatePackage,
    keys:      Optional[tuple] = None,
    max_chars: int = _MAX_CONTEXT_CHARS,
) -> str:
    """Legacy fallback — no longer used by the main builders."""
    d = package.to_dict()
    if keys:
        d = {k: d[k] for k in keys if k in d}
    text = json.dumps(d, indent=2, default=str)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n... [truncated for token budget]"
    return text


def _risk_block(risk_flags: list) -> str:
    if not risk_flags:
        return ""
    lines = "\n".join(
        f"  - {flag}: {RISK_FLAGS.get(flag, flag)}"
        for flag in risk_flags
    )
    return f"<risk_flags>\n{lines}\n</risk_flags>\n"


def estimate_tokens(system: str, user: str) -> int:
    return (len(system) + len(user)) // 4