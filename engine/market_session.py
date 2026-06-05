"""
engine/market_session.py — authoritative US equity market session classifier.

Single source of truth for session state, used by:
  - engine/alpaca_feed.py  (price source labeling)
  - runtime/update_cycle.py  (SessionPhase mapping, demo state)
  - cli/dashboard.py  (session display)

Session boundaries (NYSE, US/Eastern):
  PREMARKET   04:00 – 09:30
  OPEN        09:30 – 12:00
  MIDDAY      12:00 – 15:00
  POWER_HOUR  15:00 – 16:00
  AFTER_HOURS 16:00 – 20:00
  CLOSED      otherwise (including weekends)

DST is handled automatically via zoneinfo (Python 3.9+, stdlib).
"""

from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
from enum import Enum
from typing import Optional

try:
    from zoneinfo import ZoneInfo
    _ET: Optional[ZoneInfo] = ZoneInfo("America/New_York")
except Exception:
    _ET = None


class MarketSession(Enum):
    PREMARKET   = "premarket"
    OPEN        = "open"
    MIDDAY      = "midday"
    POWER_HOUR  = "power_hour"
    AFTER_HOURS = "after_hours"
    CLOSED      = "closed"


# ──────────────────────────────────────────────
# Core classifier
# ──────────────────────────────────────────────

def get_market_session(dt: Optional[datetime] = None) -> MarketSession:
    """
    Return the MarketSession for *dt* (UTC).  Uses current time if None.
    DST-aware via zoneinfo; falls back to UTC-5 approximation (EST only)
    when the system has no timezone data.
    """
    if dt is None:
        dt = datetime.now(tz=timezone.utc)

    et = _to_et(dt)

    # Weekend → always closed
    if et.weekday() >= 5:
        return MarketSession.CLOSED

    t = et.time()
    if   time( 4,  0) <= t < time( 9, 30):  return MarketSession.PREMARKET
    elif time( 9, 30) <= t < time(12,  0):  return MarketSession.OPEN
    elif time(12,  0) <= t < time(15,  0):  return MarketSession.MIDDAY
    elif time(15,  0) <= t < time(16,  0):  return MarketSession.POWER_HOUR
    elif time(16,  0) <= t < time(20,  0):  return MarketSession.AFTER_HOURS
    else:                                    return MarketSession.CLOSED


# ──────────────────────────────────────────────
# Predicates
# ──────────────────────────────────────────────

def is_regular_hours(session: MarketSession) -> bool:
    """True during the NYSE trading day (open, midday, power hour)."""
    return session in (
        MarketSession.OPEN,
        MarketSession.MIDDAY,
        MarketSession.POWER_HOUR,
    )


def is_extended_hours(session: MarketSession) -> bool:
    """True during pre- or after-market sessions."""
    return session in (MarketSession.PREMARKET, MarketSession.AFTER_HOURS)


def is_market_active(session: MarketSession) -> bool:
    """True whenever trading is possible (regular + extended)."""
    return session != MarketSession.CLOSED


# ──────────────────────────────────────────────
# SessionPhase bridge (backward compat)
# ──────────────────────────────────────────────

def to_session_phase(session: MarketSession):
    """
    Map MarketSession → engine.models.SessionPhase for code that
    still uses the older enum.  Does NOT modify models.py.
    """
    from engine.models import SessionPhase
    return {
        MarketSession.PREMARKET:   SessionPhase.OPEN,
        MarketSession.OPEN:        SessionPhase.OPEN,
        MarketSession.MIDDAY:      SessionPhase.MIDDAY,
        MarketSession.POWER_HOUR:  SessionPhase.POWER_HOUR,
        MarketSession.AFTER_HOURS: SessionPhase.POWER_HOUR,
        MarketSession.CLOSED:      SessionPhase.MIDDAY,
    }.get(session, SessionPhase.MIDDAY)


# ──────────────────────────────────────────────
# Display helpers
# ──────────────────────────────────────────────

def to_et_string(dt: Optional[datetime]) -> str:
    """Format a UTC datetime as 'HH:MM:SS ET'.  Returns '—' if None."""
    if dt is None:
        return "—"
    return _to_et(dt).strftime("%H:%M:%S ET")


def session_label(session: MarketSession) -> str:
    """Human-readable label for dashboard display."""
    return session.value.replace("_", " ").upper()


# ──────────────────────────────────────────────
# Internal
# ──────────────────────────────────────────────

def _to_et(dt: datetime) -> datetime:
    """Convert a UTC datetime to US/Eastern.  Handles DST via zoneinfo."""
    if _ET is not None:
        try:
            return dt.astimezone(_ET)
        except Exception:
            pass
    # Fallback: UTC-5 (EST; EDT would be UTC-4 but zoneinfo is stdlib on 3.9+)
    utc = dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    return (utc - timedelta(hours=5)).replace(tzinfo=None)