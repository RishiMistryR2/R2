"""
context/principal_alerts.py — context alert rules.

Pure functions: (ContextState) -> List[ContextAlert]

Alert types:
  NEWS_SPIKE       — one or more high-impact, market-moving headlines detected
  MACRO_IMMINENT   — a HIGH-impact macro event within 2 hours
  MACRO_APPROACHING — a MEDIUM-impact event within 1 hour
  HIGH_VOLATILITY  — VIX or risk level signals elevated volatility

All functions are stateless and independently testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import List

from context.context_state import ContextState, HeadlineImpact


# ──────────────────────────────────────────────
# Alert types
# ──────────────────────────────────────────────

class AlertType(Enum):
    NEWS_SPIKE          = "news_spike"
    MACRO_IMMINENT      = "macro_imminent"
    MACRO_APPROACHING   = "macro_approaching"
    HIGH_VOLATILITY     = "high_volatility"


class AlertSeverity(Enum):
    INFO     = "info"
    WARNING  = "warning"
    CRITICAL = "critical"


@dataclass
class ContextAlert:
    """A single context alert with type, severity, and human-readable message."""
    alert_type: AlertType
    severity:   AlertSeverity
    message:    str
    details:    dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "type":     self.alert_type.value,
            "severity": self.severity.value,
            "message":  self.message,
            "details":  self.details,
        }


# ──────────────────────────────────────────────
# Rule functions
# ──────────────────────────────────────────────

def check_news_spike(ctx: ContextState) -> List[ContextAlert]:
    """
    Emit NEWS_SPIKE alert when aggregate news sentiment is strongly directional
    or when the context_bias is not neutral due to news.

    Thresholds:
      |news_sentiment| >= 0.5  → WARNING
      |news_sentiment| >= 0.75 → CRITICAL
    """
    alerts: List[ContextAlert] = []
    s = ctx.news_sentiment

    if abs(s) < 0.5:
        return alerts

    severity = AlertSeverity.CRITICAL if abs(s) >= 0.75 else AlertSeverity.WARNING
    direction = "bullish" if s > 0 else "bearish"

    alerts.append(ContextAlert(
        alert_type=AlertType.NEWS_SPIKE,
        severity=severity,
        message=f"Strong {direction} news sentiment detected ({s:+.2f})",
        details={"sentiment": s, "context_bias": ctx.context_bias},
    ))
    return alerts


def check_macro_imminent(
    ctx:             ContextState,
    window_minutes:  int = 120,
) -> List[ContextAlert]:
    """
    Emit alerts for upcoming macro events within the given window.

    HIGH-impact event within 2h     → CRITICAL / MACRO_IMMINENT
    MEDIUM-impact event within 1h   → WARNING  / MACRO_APPROACHING
    """
    alerts: List[ContextAlert] = []
    now = datetime.now(tz=timezone.utc)

    for event in ctx.upcoming_events:
        scheduled = getattr(event, "scheduled_at", None)
        if scheduled is None:
            continue

        delta_m = (scheduled - now).total_seconds() / 60
        if delta_m < 0 or delta_m > window_minutes:
            continue

        impact = getattr(event, "market_impact", HeadlineImpact.LOW)
        name   = getattr(event, "name", "macro event")

        if impact == HeadlineImpact.HIGH:
            alerts.append(ContextAlert(
                alert_type=AlertType.MACRO_IMMINENT,
                severity=AlertSeverity.CRITICAL,
                message=f"HIGH-impact macro event in {int(delta_m)} min: {name}",
                details={"event": name, "minutes": int(delta_m), "impact": "high"},
            ))
        elif impact == HeadlineImpact.MEDIUM and delta_m <= 60:
            alerts.append(ContextAlert(
                alert_type=AlertType.MACRO_APPROACHING,
                severity=AlertSeverity.WARNING,
                message=f"MEDIUM-impact macro event in {int(delta_m)} min: {name}",
                details={"event": name, "minutes": int(delta_m), "impact": "medium"},
            ))

    return alerts


def check_high_volatility(ctx: ContextState) -> List[ContextAlert]:
    """
    Emit HIGH_VOLATILITY alert when the context signals elevated risk.
    """
    alerts: List[ContextAlert] = []

    if ctx.volatility_risk_level == "extreme":
        alerts.append(ContextAlert(
            alert_type=AlertType.HIGH_VOLATILITY,
            severity=AlertSeverity.CRITICAL,
            message=f"Extreme volatility environment — {ctx.risk_note or 'risk elevated'}",
            details={"risk_level": ctx.volatility_risk_level, "risk_note": ctx.risk_note},
        ))
    elif ctx.volatility_risk_level == "elevated":
        alerts.append(ContextAlert(
            alert_type=AlertType.HIGH_VOLATILITY,
            severity=AlertSeverity.WARNING,
            message=f"Elevated volatility — {ctx.risk_note or 'proceed with caution'}",
            details={"risk_level": ctx.volatility_risk_level, "risk_note": ctx.risk_note},
        ))

    return alerts


def get_alerts(ctx: ContextState) -> List[ContextAlert]:
    """
    Run all alert rules against a ContextState and return combined results.
    Results are sorted: CRITICAL first, then WARNING, then INFO.
    """
    alerts: List[ContextAlert] = []
    alerts.extend(check_news_spike(ctx))
    alerts.extend(check_macro_imminent(ctx))
    alerts.extend(check_high_volatility(ctx))

    severity_order = {
        AlertSeverity.CRITICAL: 0,
        AlertSeverity.WARNING:  1,
        AlertSeverity.INFO:     2,
    }
    alerts.sort(key=lambda a: severity_order.get(a.severity, 9))
    return alerts