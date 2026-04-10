"""
Context Intelligence data models.

These structures represent everything the system knows about the
external world — news, macro events, Fed stance, market regime —
independently of price action.

Source trust tiers:
  Tier 1 (AUTHORITATIVE): Official releases, price data, government stats.
           Always included. Unfiltered.
  Tier 2 (CURATED):       Quality financial media, analyst reports, my journal.
           Included with source label. Weight reduced for ambiguous claims.
  Tier 3 (CROWD):         Social sentiment, retail flow, forums.
           EXCLUDED from v1. Do not include.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional


# ──────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────

class SourceTier(Enum):
    TIER_1_AUTHORITATIVE = "tier_1"   # price data, official gov releases
    TIER_2_CURATED       = "tier_2"   # Reuters, Bloomberg, journal
    TIER_3_CROWD         = "tier_3"   # excluded from v1


class HeadlineImpact(Enum):
    HIGH    = "high"     # direct market-moving potential (rate decision, CPI print)
    MEDIUM  = "medium"   # relevant context (Fed speech, sector news)
    LOW     = "low"      # background noise, minimal expected impact


class MarketRegime(Enum):
    RISK_ON  = "risk_on"   # SPY trending, VIX low, broad participation
    RISK_OFF = "risk_off"  # defensive rotation, VIX elevated, flight to safety
    NEUTRAL  = "neutral"   # mixed signals, no clear regime


class MacroEventType(Enum):
    FED_DECISION     = "fed_decision"
    FED_SPEECH       = "fed_speech"
    CPI              = "cpi"
    PPI              = "ppi"
    NFP              = "nfp"              # Non-farm payrolls
    GDP              = "gdp"
    RETAIL_SALES     = "retail_sales"
    JOBLESS_CLAIMS   = "jobless_claims"
    EARNINGS         = "earnings"
    FOMC_MINUTES     = "fomc_minutes"
    OTHER            = "other"


# ──────────────────────────────────────────────
# Headline
# ──────────────────────────────────────────────

@dataclass
class Headline:
    """
    A single scored news headline.

    Scoring rules:
      importance (1-10):  raw market-moving potential of this type of news
      spy_relevance:      "direct" (SPY/macro) | "sector" | "background"
      recency_weight:     1.0 if < 1h old, 0.5 if 1-4h, 0.2 if 4-12h, 0 if > 12h
      trade_relevant:     True if headline directly touches our active thesis

    The final score passed to AI = importance × recency_weight
    Headlines with final score < 2.0 are filtered before prompt assembly.
    """
    headline:       str
    source:         str
    tier:           SourceTier
    published_at:   datetime

    importance:       float = 5.0          # 1-10 raw importance
    spy_relevance:    str   = "background" # "direct" | "sector" | "background"
    recency_weight:   float = 1.0          # 0.0 – 1.0
    trade_relevant:   bool  = False        # explicitly touches active trade thesis
    impact:           HeadlineImpact = HeadlineImpact.MEDIUM

    url:     Optional[str] = None
    summary: Optional[str] = None         # 1-sentence AI-readable summary

    @property
    def weighted_score(self) -> float:
        return round(self.importance * self.recency_weight, 2)

    def to_dict(self) -> dict:
        return {
            "headline":       self.headline,
            "source":         self.source,
            "tier":           self.tier.value,
            "published_at":   self.published_at.isoformat(),
            "importance":     self.importance,
            "spy_relevance":  self.spy_relevance,
            "trade_relevant": self.trade_relevant,
            "impact":         self.impact.value,
            "score":          self.weighted_score,
            "summary":        self.summary,
        }


# ──────────────────────────────────────────────
# Macro event
# ──────────────────────────────────────────────

@dataclass
class MacroEvent:
    """
    A scheduled or recently released macro event.

    'upcoming' events are on the horizon and add a risk flag.
    'released' events carry an actual vs expected comparison.
    """
    event_type:    MacroEventType
    name:          str
    scheduled_at:  datetime
    status:        str = "upcoming"   # "upcoming" | "released"

    # For released events
    actual:        Optional[float] = None
    expected:      Optional[float] = None
    prior:         Optional[float] = None
    unit:          str = ""           # "%" | "K" | "bps" | etc.

    # Derived impact assessment
    surprise:      Optional[float] = None   # actual - expected
    surprise_dir:  Optional[str]   = None   # "beat" | "miss" | "inline"
    market_impact: HeadlineImpact  = HeadlineImpact.MEDIUM

    notes: str = ""    # e.g. "first print since Fed pivot"

    def to_dict(self) -> dict:
        return {
            "type":         self.event_type.value,
            "name":         self.name,
            "scheduled_at": self.scheduled_at.isoformat(),
            "status":       self.status,
            "actual":       self.actual,
            "expected":     self.expected,
            "surprise":     self.surprise,
            "surprise_dir": self.surprise_dir,
            "impact":       self.market_impact.value,
            "notes":        self.notes,
        }


# ──────────────────────────────────────────────
# Fed context
# ──────────────────────────────────────────────

@dataclass
class FedContext:
    """
    Snapshot of the current Federal Reserve stance.
    Derived from FOMC statements, dot plot, and recent speeches.
    """
    current_rate:    Optional[float] = None   # federal funds rate upper bound (%)
    rate_bias:       str = "neutral"           # "hawkish" | "dovish" | "neutral"
    next_meeting:    Optional[datetime] = None
    hike_probability: float = 0.0             # CME FedWatch: next hike prob
    cut_probability:  float = 0.0             # CME FedWatch: next cut prob

    # Qualitative summary (from latest statement/speech)
    stance_summary:  str = ""   # e.g. "data-dependent, higher for longer signaled"
    last_updated:    Optional[datetime] = None

    def to_dict(self) -> dict:
        return {
            "current_rate":   self.current_rate,
            "rate_bias":      self.rate_bias,
            "hike_prob":      round(self.hike_probability, 3),
            "cut_prob":       round(self.cut_probability, 3),
            "stance_summary": self.stance_summary,
            "next_meeting":   self.next_meeting.isoformat() if self.next_meeting else None,
        }


# ──────────────────────────────────────────────
# Context state (assembled)
# ──────────────────────────────────────────────

@dataclass
class ContextState:
    """
    The complete external context snapshot.
    Assembled by ContextEngine and passed into the AI context package.

    This is the bridge between 'what the market data says' (MarketState)
    and 'what the world says' (news, macro, calendar).
    """
    timestamp: datetime

    # News
    top_headlines: List[Headline]        = field(default_factory=list)
    trade_relevant_headlines: List[Headline] = field(default_factory=list)

    # Macro
    upcoming_events:  List[MacroEvent]   = field(default_factory=list)
    recent_releases:  List[MacroEvent]   = field(default_factory=list)

    # Fed
    fed: Optional[FedContext]            = None

    # Regime
    vix_level:       Optional[float]     = None
    vix_regime:      str                 = "normal"   # "low" | "normal" | "elevated" | "extreme"
    market_regime:   MarketRegime        = MarketRegime.NEUTRAL

    # Composite risk flag
    elevated_risk:   bool  = False        # True when major event within 2h or VIX extreme
    risk_note:       str   = ""           # e.g. "CPI release in 45 minutes"

    # Data freshness
    news_stale:      bool  = False        # True if news > 30 min old
    context_valid:   bool  = True         # False if fetch failed

    def to_dict(self) -> dict:
        return {
            "timestamp":     self.timestamp.isoformat(),
            "market_regime": self.market_regime.value,
            "vix":           {"level": self.vix_level, "regime": self.vix_regime},
            "elevated_risk": self.elevated_risk,
            "risk_note":     self.risk_note,
            "fed":           self.fed.to_dict() if self.fed else None,
            "top_headlines": [h.to_dict() for h in self.top_headlines[:5]],
            "trade_headlines": [h.to_dict() for h in self.trade_relevant_headlines[:3]],
            "upcoming_events": [e.to_dict() for e in self.upcoming_events[:5]],
            "recent_releases": [e.to_dict() for e in self.recent_releases[:3]],
            "data_quality":  {
                "news_stale":    self.news_stale,
                "context_valid": self.context_valid,
            }
        }

    @classmethod
    def empty(cls) -> "ContextState":
        """Return a valid but empty context when fetching fails."""
        from datetime import timezone
        return cls(
            timestamp=datetime.now(tz=timezone.utc),
            context_valid=False,
        )
