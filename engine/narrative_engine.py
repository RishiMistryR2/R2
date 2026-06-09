"""
engine/narrative_engine.py — R2 Structure Narrative Engine v2.0B

Converts a StructureState (or its dict representation) into trader-quality
commentary covering all six required elements:
  1. Current structure  — direction, sequence, market state
  2. Buyer behaviour    — where buyers are defending and why
  3. Seller behaviour   — where sellers are capping and why
  4. Key levels         — specific price levels with structural context
  5. Missing confirmation — what needs to happen for the thesis to activate
  6. Trading implication — session-aware, entries-aware actionable note

Design rules:
  - SYNTHESIZE, do not list.  Output reads as a trader would speak, not as
    a data dump.
  - Always specific: use actual prices when available.
  - Session-aware: after-hours never implies regular-session entries;
    "first 30 minutes" language is banned outside the open session.
  - Regime-accurate: reads actual regime from the package, never hard-codes.
  - Accepts both a live StructureState object and a package dict fragment.
  - Never raises — all paths return a useful string.

Public API:
  build_narrative(struct_or_dict, symbol, session, regime,
                  entries_allowed, current_price) -> str
  build_narrative_from_package(package_dict)            -> str

Used by:
  - ai/claude_client._generate_mock_response       (mock mode)
  - runtime/commentary_cycle._build_fallback_answer (API unavailable)
"""

from __future__ import annotations

from typing import Optional


# ──────────────────────────────────────────────────────────────
# Session labels and implications
# ──────────────────────────────────────────────────────────────

_SESS_PREFIX: dict = {
    "premarket":   "In pre-market trading, ",
    "after_hours": "After hours, ",
    "midday":      "In the midday session, ",
    "power_hour":  "Into the power hour, ",
    "closed":      "With the market closed, ",
}

_SESS_CLOSING: dict = {
    "premarket":   (
        "Pre-market levels are directional context — "
        "watch for follow-through at the open."
    ),
    "open":        "",   # no suffix for regular open
    "midday":      "Midday liquidity is lower — structure quality matters more than speed.",
    "power_hour":  "Power hour can amplify or reverse moves — watch key levels closely.",
    "after_hours": (
        "Extended-hours context only. "
        "Use these levels for next-session planning — no active entries."
    ),
    "closed":      "Market is closed — review structure ahead of tomorrow's session.",
}


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────

def build_narrative(
    struct_or_dict,
    symbol:          str   = "SPY",
    session                = None,   # MarketSession | str | None
    regime:          str   = "",
    entries_allowed: bool  = True,
    current_price:   Optional[float] = None,
) -> str:
    """
    Build a complete trader-quality narrative from structure data.

    Parameters
    ----------
    struct_or_dict  : StructureState object, package structure dict, or None
    symbol          : ticker symbol (default "SPY")
    session         : MarketSession enum, value string, or None
    regime          : regime name string e.g. "chop", "trend_bull"
    entries_allowed : whether entries are currently permitted
    current_price   : live price if available (used for relative level context)
    """
    sess_key = _session_key(session)
    fields   = _to_fields(struct_or_dict)

    if not fields or not fields.get("direction"):
        return _minimal_fallback(symbol, regime, sess_key, entries_allowed)

    try:
        return _full_narrative(
            fields, symbol, sess_key, regime, entries_allowed, current_price
        )
    except Exception:
        return _minimal_fallback(symbol, regime, sess_key, entries_allowed)


def build_narrative_from_package(package_dict: dict) -> str:
    """
    Convenience wrapper for use in claude_client where only the package
    dict is available (not the live StructureState object).
    """
    market  = package_dict.get("market", {}) or {}
    struct  = package_dict.get("structure", {}) or {}
    session = market.get("market_session", "")
    regime  = market.get("regime", {}).get("name", "")
    entries = (market.get("session", {}) or {}).get("entries_allowed", True)
    price   = None
    ind_5m  = (market.get("indicators", {}) or {}).get("5m", {}) or {}
    if ind_5m.get("close"):
        price = ind_5m["close"]
    return build_narrative(
        struct, session=session, regime=regime,
        entries_allowed=entries, current_price=price,
    )


# ──────────────────────────────────────────────────────────────
# Core builder
# ──────────────────────────────────────────────────────────────

def _full_narrative(
    f: dict, symbol: str, sess_key: str,
    regime: str, entries_allowed: bool,
    current_price: Optional[float],
) -> str:
    parts: list[str] = []

    parts.append(_opening(f, symbol, sess_key))

    bs = _buyer_seller(f)
    if bs:
        parts.append(bs)

    lv = _key_levels(f, current_price)
    if lv:
        parts.append(lv)

    cf = _confirmation(f)
    if cf:
        parts.append(cf)

    im = _implication(f, sess_key, entries_allowed, regime)
    if im:
        parts.append(im)

    return " ".join(parts)


# ──────────────────────────────────────────────────────────────
# Part 1 — Opening
# ──────────────────────────────────────────────────────────────

def _opening(f: dict, symbol: str, sess_key: str) -> str:
    direction = f.get("direction", "")
    state     = f.get("market_state", "range")
    seq       = f.get("sequence", "")
    intact    = f.get("trend_intact", True)
    kl        = f.get("key_levels", {}) or {}
    s_high    = kl.get("session_high")
    s_low     = kl.get("session_low")

    sess_pfx = _SESS_PREFIX.get(sess_key, "")

    if state == "trend" and direction == "uptrend":
        core = (
            f"{symbol} is trending higher"
            + (f" — {seq} structure intact on 5m" if seq else "")
            + ("." if intact else " but the trend is showing signs of weakness.")
        )
    elif state == "trend" and direction == "downtrend":
        core = (
            f"{symbol} is trending lower"
            + (f" — {seq} structure on 5m" if seq else "")
            + "."
        )
    elif state == "pullback" and direction == "uptrend":
        core = (
            f"{symbol} is pulling back within an uptrend"
            + (f" ({seq} on 5m)" if seq else "")
            + "."
        )
    elif state == "pullback" and direction == "downtrend":
        core = (
            f"{symbol} is bouncing within a downtrend"
            + (f" ({seq} on 5m)" if seq else "")
            + "."
        )
    elif state == "range":
        if s_high and s_low:
            core = f"{symbol} is range-bound between {s_low:.2f} and {s_high:.2f}."
        else:
            core = f"{symbol} is trading in a range with no clear directional bias."
    elif state == "compression":
        mid = (s_high + s_low) / 2 if s_high and s_low else None
        core = (
            f"{symbol} is compressing near {mid:.2f} — "
            "ATR has contracted ahead of a potential directional move." if mid
            else f"{symbol} is compressing — range has tightened ahead of a potential breakout."
        )
    elif state == "breakout":
        core = (
            (f"{symbol} broke above {s_high:.2f}" if s_high
             else f"{symbol} is breaking out to the upside")
            + " — expansion phase underway."
        )
    elif state == "breakdown":
        core = (
            (f"{symbol} broke below {s_low:.2f}" if s_low
             else f"{symbol} is breaking down")
            + " — sellers are accelerating."
        )
    elif state == "expansion":
        core = (
            f"{symbol} is in an expansion phase"
            + (f" — {direction} momentum extending on 5m" if direction else "")
            + "."
        )
    else:
        if s_high and s_low:
            core = f"{symbol} remains range-bound between {s_low:.2f} and {s_high:.2f}."
        else:
            core = f"{symbol} is oscillating without clear directional structure."

    return sess_pfx + core


# ──────────────────────────────────────────────────────────────
# Part 2 — Buyer / seller synthesis
# ──────────────────────────────────────────────────────────────

def _buyer_seller(f: dict) -> str:
    direction = f.get("direction", "")
    state     = f.get("market_state", "")
    kl        = f.get("key_levels", {}) or {}
    vwap      = kl.get("vwap")
    support   = kl.get("nearest_support")
    s_high    = kl.get("session_high")
    s_low     = kl.get("session_low")
    hh        = f.get("last_hh")
    hl        = f.get("last_hl")
    lh        = f.get("last_lh")
    ll        = f.get("last_ll")

    if direction == "uptrend":
        b_anchor = hl or support
        b_parts  = []
        if b_anchor:
            b_parts.append(f"defending the higher low at {b_anchor:.2f}")
        if vwap:
            b_parts.append(f"holding above VWAP ({vwap:.2f})")
        buyer_str = "Buyers are " + (
            " and ".join(b_parts) if b_parts else "attempting to hold higher lows"
        )
        s_anchor = s_high or hh
        seller_str = (
            f"sellers are capping at {s_anchor:.2f}" if s_anchor
            else "sellers are defending overhead resistance"
        )
        return f"{buyer_str}, while {seller_str}."

    if direction == "downtrend":
        s_anchor = lh or s_high
        s_parts  = []
        if s_anchor:
            s_parts.append(f"capping rallies near {s_anchor:.2f}")
        if vwap:
            s_parts.append(f"defending below VWAP ({vwap:.2f})")
        seller_str = "Sellers are " + (
            " and ".join(s_parts) if s_parts else "defending lower highs"
        )
        b_anchor = ll or s_low
        buyer_str = (
            f"buyers are attempting to hold above {b_anchor:.2f}" if b_anchor
            else "buyers have not yet formed a higher low"
        )
        return f"{seller_str}, while {buyer_str}."

    if state in ("range", "compression") or direction in ("ranging", ""):
        if s_high and s_low:
            if vwap:
                return (
                    f"Buyers are defending near {s_low:.2f} while sellers "
                    f"reject advances toward {s_high:.2f}; "
                    f"VWAP ({vwap:.2f}) acts as the mid-range pivot."
                )
            return (
                f"Buyers are defending near {s_low:.2f} while sellers "
                f"continue to reject advances toward {s_high:.2f}."
            )
        return (
            "Neither buyers nor sellers have established follow-through "
            "beyond current boundaries."
        )

    if state == "breakout":
        return (
            "Buyers are driving follow-through above the breakout level"
            + (f" ({s_high:.2f})" if s_high else "")
            + " — sellers have lost control of that boundary."
        )

    if state == "breakdown":
        return (
            "Sellers are driving follow-through below the breakdown level"
            + (f" ({s_low:.2f})" if s_low else "")
            + " — buyers have lost control of that boundary."
        )

    return ""


# ──────────────────────────────────────────────────────────────
# Part 3 — Key levels
# ──────────────────────────────────────────────────────────────

def _key_levels(f: dict, current_price: Optional[float]) -> str:
    kl      = f.get("key_levels", {}) or {}
    vwap    = kl.get("vwap")
    s_high  = kl.get("session_high")
    s_low   = kl.get("session_low")
    pdh     = kl.get("prior_day_high")
    support = kl.get("nearest_support")

    candidates: list[tuple[str, float]] = []
    if s_high:   candidates.append(("session high",    s_high))
    if vwap:     candidates.append(("VWAP",            vwap))
    if s_low:    candidates.append(("session low",     s_low))
    if support and support not in (s_high, s_low):
        candidates.append(("support", support))
    if pdh and pdh not in (s_high, s_low):
        candidates.append(("prior day high", pdh))

    if not candidates:
        return ""

    if current_price:
        candidates.sort(key=lambda t: abs(t[1] - current_price))

    level_str = " | ".join(
        f"{lbl} {price:.2f}" for lbl, price in candidates[:3]
    )
    return f"Key levels: {level_str}."


# ──────────────────────────────────────────────────────────────
# Part 4 — Confirmation
# ──────────────────────────────────────────────────────────────

def _confirmation(f: dict) -> str:
    missing = (f.get("missing_confirmation") or "").strip()
    if not missing:
        return ""
    low = missing.lower()
    if low.startswith("need"):
        return "Confirmation requires" + missing[4:].rstrip(".") + "."
    if low.startswith("confirmation"):
        return missing.rstrip(".") + "."
    return f"Confirmation is still pending: {missing.rstrip('.')}."


# ──────────────────────────────────────────────────────────────
# Part 5 — Trading implication
# ──────────────────────────────────────────────────────────────

def _implication(
    f: dict, sess_key: str,
    entries_allowed: bool, regime: str,
) -> str:
    direction = f.get("direction", "")
    state     = f.get("market_state", "")

    sess_suffix = _SESS_CLOSING.get(sess_key, "")
    if sess_suffix:
        return sess_suffix

    if not entries_allowed:
        return "Entries not permitted in the current session — observe only."

    is_chop = (
        state in ("range", "compression")
        or direction in ("ranging", "")
        or "chop" in regime.lower()
    )

    if is_chop:
        return (
            "Range-bound conditions — wait for a clean break of the boundary "
            "with follow-through before committing to a direction."
        )

    if state == "breakout":
        return "Breakout confirmed — continuation trades valid above the breakout level."
    if state == "breakdown":
        return "Breakdown confirmed — continuation trades valid below the breakdown level."

    if direction == "uptrend":
        if state == "pullback":
            return (
                "Long bias — look for the pullback to hold the higher low "
                "before entering with the trend."
            )
        return "Long bias with confirmation — structure favours continuation setups."

    if direction == "downtrend":
        if state == "pullback":
            return (
                "Short bias — watch for the bounce to fail at a lower high "
                "before entering with the trend."
            )
        return "Short bias with confirmation — structure favours continuation setups."

    return ""


# ──────────────────────────────────────────────────────────────
# Minimal fallback (no structure data)
# ──────────────────────────────────────────────────────────────

def _minimal_fallback(
    symbol: str, regime: str, sess_key: str, entries_allowed: bool
) -> str:
    """
    Meaningful paragraph when structure_state is absent.
    Always better than 'Regime is chop. Narrative: chop.'
    """
    regime_clean = (regime or "").replace("_", " ").strip()
    sess_pfx     = _SESS_PREFIX.get(sess_key, "")
    sess_suffix  = _SESS_CLOSING.get(sess_key, "")

    if not regime_clean or regime_clean in ("unknown", ""):
        body = (
            f"{sess_pfx}{symbol} structure is still loading. "
            "No directional edge is identifiable yet — "
            "wait for a clear HH/HL or LH/LL sequence to develop."
        )
    elif "chop" in regime_clean or "neutral" in regime_clean:
        body = (
            f"{sess_pfx}{symbol} is in a {regime_clean} regime — "
            "no clean directional structure has formed. "
            "Buyers and sellers are in equilibrium; "
            "wait for a clear break of the current range before committing."
        )
    elif "bull" in regime_clean or "trend" in regime_clean:
        body = (
            f"{sess_pfx}{symbol} is in a {regime_clean} regime. "
            "Structural confirmation is still loading — "
            "watch for higher highs and higher lows to form on 5m "
            "before entering with the trend."
        )
    elif "bear" in regime_clean:
        body = (
            f"{sess_pfx}{symbol} is in a {regime_clean} regime. "
            "Structural confirmation is still loading — "
            "watch for lower highs and lower lows to form on 5m "
            "before entering with the trend."
        )
    else:
        body = (
            f"{sess_pfx}{symbol} is in a {regime_clean} regime. "
            "No structural detail available yet — "
            "monitor price action before entering."
        )

    if sess_suffix:
        return body + " " + sess_suffix
    if not entries_allowed:
        return body + " Entries not currently permitted."
    return body


# ──────────────────────────────────────────────────────────────
# Normalisation helpers
# ──────────────────────────────────────────────────────────────

def _session_key(session) -> str:
    if session is None:
        return ""
    if isinstance(session, str):
        return session
    try:
        return session.value
    except AttributeError:
        return str(session)


def _to_fields(struct_or_dict) -> dict:
    """
    Normalise a StructureState object or a package structure dict to a
    flat fields dict that the narrative functions can consume.
    """
    if struct_or_dict is None:
        return {}

    if isinstance(struct_or_dict, dict):
        return struct_or_dict

    # StructureState object
    try:
        kl     = struct_or_dict.key_levels
        events = getattr(struct_or_dict, "behavioral_events", [])
        return {
            "direction":            struct_or_dict.trend.direction,
            "sequence":             struct_or_dict.trend.sequence_str(),
            "trend_intact":         struct_or_dict.trend.trend_intact,
            "last_hh":              struct_or_dict.trend.last_hh,
            "last_hl":              struct_or_dict.trend.last_hl,
            "last_lh":              struct_or_dict.trend.last_lh,
            "last_ll":              struct_or_dict.trend.last_ll,
            "market_state":         struct_or_dict.market_state.state.value,
            "structure_narrative":  struct_or_dict.structure_narrative,
            "buyer_context":        struct_or_dict.buyer_context,
            "seller_context":       struct_or_dict.seller_context,
            "missing_confirmation": struct_or_dict.missing_confirmation,
            "key_levels": {
                "session_high":       kl.session_high,
                "session_low":        kl.session_low,
                "prior_day_high":     kl.prior_day_high,
                "prior_day_low":      kl.prior_day_low,
                "vwap":               kl.vwap,
                "nearest_support":    kl.nearest_support,
                "nearest_resistance": kl.nearest_resistance,
            },
            "recent_events": [
                f"{e.event_type.value} ({e.bars_ago}b ago)"
                for e in events[:4]
            ],
        }
    except Exception:
        return {}