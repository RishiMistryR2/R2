"""
context/models.py — shared dataclasses for the context layer.

Defines:
  Sentiment     — BULLISH / BEARISH / NEUTRAL enum
  ImpactLevel   — LOW / MEDIUM / HIGH enum (alias of HeadlineImpact for clean API)
  NewsItem      — internal scored news record (used by HeadlineScorer + NewsFetcher)
  NewsEvent     — simplified Phase 4 spec dataclass (headline, timestamp, source, symbols)

SourceTier is imported from context_state and re-exported here so other modules
only need one import point inside the context package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional

from context.context_state import HeadlineImpact, SourceTier  # re-export


# ──────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────

class Sentiment(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


# ImpactLevel mirrors HeadlineImpact so Phase 4 code can use either name.
# HIGH / MEDIUM / LOW values are identical.
ImpactLevel = HeadlineImpact


# ──────────────────────────────────────────────
# NewsItem — internal, fully scored record
# ──────────────────────────────────────────────

@dataclass
class NewsItem:
    """
    A raw news item enriched by HeadlineScorer.

    Fields set at construction:
      id, headline, summary, source, source_tier, published_at, url

    Fields set by HeadlineScorer._score_item():
      importance_score, spy_relevance, sentiment, sentiment_label,
      is_market_moving, keywords
    """
    id:          str
    headline:    str
    summary:     str
    source:      str
    source_tier: SourceTier
    published_at: datetime
    url:          Optional[str] = None

    # Scored fields (populated by HeadlineScorer)
    importance_score: float      = 0.0
    spy_relevance:    float      = 0.0
    sentiment:        float      = 0.0   # –1.0 … +1.0
    sentiment_label:  str        = "neutral"
    is_market_moving: bool       = False
    keywords:         List[str]  = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id":              self.id,
            "headline":        self.headline,
            "source":          self.source,
            "source_tier":     self.source_tier.value,
            "published_at":    self.published_at.isoformat(),
            "importance":      self.importance_score,
            "spy_relevance":   self.spy_relevance,
            "sentiment":       self.sentiment,
            "sentiment_label": self.sentiment_label,
            "market_moving":   self.is_market_moving,
            "keywords":        self.keywords,
        }


# ──────────────────────────────────────────────
# NewsEvent — simplified Phase 4 spec dataclass
# ──────────────────────────────────────────────

@dataclass
class NewsEvent:
    """
    Lightweight news event for context aggregation.
    Used by fetch_news() and build_context().

    sentiment_score: –1.0 (very bearish) … +1.0 (very bullish), None if unscored.
    """
    headline:        str
    timestamp:       datetime
    source:          str
    symbols:         List[str]       = field(default_factory=list)
    sentiment_score: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "headline":        self.headline,
            "timestamp":       self.timestamp.isoformat(),
            "source":          self.source,
            "symbols":         self.symbols,
            "sentiment_score": self.sentiment_score,
        }