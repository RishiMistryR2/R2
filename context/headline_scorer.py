"""
Headline Scorer.

Assigns importance and SPY relevance scores to news items
using keyword matching. Fast, deterministic, and testable.
No ML model required — good signal comes from knowing
which words actually move markets.

Design: keywords are weighted, not just presence/absence.
A headline with "emergency rate cut" scores higher than one
with just "Federal Reserve".

Scores are 0.0–1.0. Combined composite produces is_market_moving.
"""

from __future__ import annotations
import re
from datetime import datetime, timezone, timedelta
from typing import List, Tuple

from context.models import NewsItem, SourceTier


# ──────────────────────────────────────────────
# Keyword tables
# ──────────────────────────────────────────────

# (pattern, importance_score_contribution, spy_relevance_contribution, sentiment)
# sentiment: +1 bullish, -1 bearish, 0 neutral
KEYWORD_RULES: List[Tuple[str, float, float, float]] = [

    # ── Fed / rates (highest impact) ──────────
    (r"emergency rate cut",              0.95, 0.95, +1.0),
    (r"rate cut",                        0.80, 0.80, +0.8),
    (r"rate hike",                       0.80, 0.80, -0.8),
    (r"federal reserve|fed\b",           0.55, 0.55,  0.0),
    (r"fomc",                            0.70, 0.70,  0.0),
    (r"jerome powell|powell",            0.50, 0.50,  0.0),
    (r"quantitative tightening|qt\b",    0.45, 0.45, -0.3),
    (r"quantitative easing|qe\b",        0.45, 0.45, +0.3),
    (r"yield curve",                     0.40, 0.40,  0.0),
    (r"interest rate",                   0.45, 0.45,  0.0),
    (r"10.year treasury|10yr",           0.40, 0.40,  0.0),

    # ── Inflation / macro data ─────────────────
    (r"\bcpi\b|consumer price index",    0.85, 0.80,  0.0),
    (r"\bppi\b|producer price",          0.70, 0.65,  0.0),
    (r"inflation",                       0.60, 0.60,  0.0),
    (r"non-farm payroll|nfp\b",          0.85, 0.80,  0.0),
    (r"jobs report|employment report",   0.75, 0.70,  0.0),
    (r"unemployment rate",               0.65, 0.60,  0.0),
    (r"gdp|gross domestic product",      0.75, 0.65,  0.0),
    (r"retail sales",                    0.55, 0.50,  0.0),
    (r"jobless claims",                  0.45, 0.40,  0.0),

    # ── Market structure ──────────────────────
    (r"market crash|stock crash",        0.90, 0.90, -1.0),
    (r"circuit breaker",                 0.90, 0.90, -0.9),
    (r"bear market",                     0.70, 0.70, -0.8),
    (r"bull market",                     0.55, 0.55, +0.7),
    (r"volatility|vix",                  0.55, 0.55,  0.0),
    (r"sell.off|selloff",                0.60, 0.65, -0.7),
    (r"rally|surges?|jumps?",            0.45, 0.55, +0.6),
    (r"record high|all.time high",       0.55, 0.60, +0.7),
    (r"s&p 500|spy|spx\b",               0.50, 0.95,  0.0),  # directly SPY
    (r"nasdaq|qqq\b",                    0.40, 0.55,  0.0),
    (r"dow jones|djia",                  0.40, 0.55,  0.0),

    # ── Geopolitical / macro risk ──────────────
    (r"war|military strike|invasion",    0.75, 0.55, -0.7),
    (r"tariff|trade war|sanctions",      0.65, 0.60, -0.5),
    (r"debt ceiling|government shutdown",0.70, 0.65, -0.6),
    (r"recession",                       0.75, 0.70, -0.8),
    (r"stagflation",                     0.70, 0.65, -0.7),
    (r"banking crisis|bank failure",     0.85, 0.80, -0.9),
    (r"credit default|sovereign debt",   0.75, 0.65, -0.8),

    # ── Earnings (specific large-caps move SPY) ─
    (r"apple|aapl\b",                    0.45, 0.55,  0.0),
    (r"microsoft|msft\b",                0.45, 0.55,  0.0),
    (r"nvidia|nvda\b",                   0.50, 0.55,  0.0),
    (r"amazon|amzn\b",                   0.45, 0.50,  0.0),
    (r"meta\b|facebook",                 0.40, 0.45,  0.0),
    (r"alphabet|google|googl\b",         0.40, 0.45,  0.0),
    (r"earnings beat|beats estimates",   0.40, 0.45, +0.5),
    (r"earnings miss|misses estimates",  0.40, 0.45, -0.5),
    (r"revenue guidance|outlook",        0.35, 0.40,  0.0),

    # ── Positive macro sentiment ───────────────
    (r"soft landing",                    0.50, 0.50, +0.6),
    (r"strong economy|economic growth",  0.45, 0.45, +0.5),
    (r"consumer confidence",             0.40, 0.40,  0.0),
]


# ──────────────────────────────────────────────
# Tier source weights
# ──────────────────────────────────────────────

TIER1_SOURCES = {
    "federal reserve", "fed", "fomc", "bls", "bureau of labor statistics",
    "bea", "bureau of economic analysis", "fred", "u.s. treasury",
    "reuters", "associated press", "ap", "bloomberg", "wall street journal",
    "financial times", "cnbc", "sec", "fdic",
}

TIER2_SOURCES = {
    "user journal", "trade journal", "session notes", "ai summary",
    "barron's", "seeking alpha", "marketwatch",
}


# ──────────────────────────────────────────────
# Scorer
# ──────────────────────────────────────────────

class HeadlineScorer:
    """
    Score a list of raw NewsItems, assigning importance, relevance,
    and sentiment. Returns the items sorted by composite score.
    """

    def score_items(self, items: List[NewsItem]) -> List[NewsItem]:
        """Score and sort items. Deduplicates by headline similarity."""
        seen_headlines: set[str] = set()
        scored = []

        for item in items:
            # Dedup: skip if headline is very similar to one we've seen
            normalized = re.sub(r'\W+', ' ', item.headline.lower()).strip()
            if normalized in seen_headlines:
                continue
            seen_headlines.add(normalized)

            self._score_item(item)
            scored.append(item)

        # Sort by composite score descending
        scored.sort(key=lambda x: x.importance_score * 0.6 + x.spy_relevance * 0.4, reverse=True)
        return scored

    def _score_item(self, item: NewsItem) -> None:
        """Score a single item in-place."""
        text = (item.headline + " " + item.summary).lower()

        importance  = 0.0
        relevance   = 0.0
        sentiment   = 0.0
        keywords    = []
        matches     = 0

        for pattern, imp, rel, sent in KEYWORD_RULES:
            if re.search(pattern, text, re.IGNORECASE):
                importance  = max(importance, imp)
                relevance   = max(relevance, rel)
                sentiment  += sent
                keywords.append(pattern.split(r"|")[0].replace(r"\b", "").strip())
                matches += 1

        # Tier 1 source gets a reliability boost
        if item.source.lower() in TIER1_SOURCES:
            importance = min(1.0, importance * 1.15)
            relevance  = min(1.0, relevance  * 1.10)

        # Normalise sentiment to [-1, +1]
        if matches > 0:
            sentiment = max(-1.0, min(1.0, sentiment / matches))

        # Freshness decay: articles older than 4h get 0.7× importance
        age_hours = (datetime.now(tz=timezone.utc) - item.published_at).total_seconds() / 3600
        if age_hours > 4:
            importance *= 0.7
        if age_hours > 24:
            importance *= 0.5

        item.importance_score = round(min(1.0, importance), 3)
        item.spy_relevance    = round(min(1.0, relevance), 3)
        item.sentiment        = round(sentiment, 2)
        item.sentiment_label  = (
            "bullish" if sentiment > 0.15
            else "bearish" if sentiment < -0.15
            else "neutral"
        )
        item.is_market_moving = (
            item.importance_score >= 0.65
            and item.spy_relevance >= 0.55
        )
        item.keywords = keywords[:8]

    def classify_source_tier(self, source_name: str) -> SourceTier:
        """Classify a news source into a reliability tier."""
        lower = source_name.lower()
        if any(s in lower for s in TIER1_SOURCES):
            return SourceTier.TIER1_AUTHORITATIVE
        if any(s in lower for s in TIER2_SOURCES):
            return SourceTier.TIER2_CURATED
        # Default unknown sources to tier2 (don't exclude, but don't elevate)
        return SourceTier.TIER2_CURATED
