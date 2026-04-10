"""
AIContext: the complete package assembled locally before every
call to the hosted Claude API.

This is the most important interface in the system.
Everything that crosses the local/hosted boundary goes through here.

Design principles:
  1. The AI should never need to ask for data it wasn't given.
     If it matters, it's in the context.
  2. Every field is optional — the AI handles missing data gracefully.
  3. The dict representation is what ends up in the prompt.
     Keep it dense but human-readable (Claude reads it as structured prose).
  4. Token budget matters. We summarize, not dump.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any

from engine.models import MarketState, TradeState
from context.context_state import ContextState
from memory.memory_state import MemoryContext


@dataclass
class AIContext:
    """
    The unified context package passed to every Claude API call.

    Assembled by the ContextBuilder from:
      market_state   → MarketState.to_dict()
      context_state  → ContextState.to_dict()
      memory         → MemoryContext.to_dict()
      session_history → recent conversation turns

    The prompt_builder formats this into the actual API payload.
    """
    # Core market data (always present)
    market_state: Optional[Dict]    = None

    # Strategy outputs (present when indicators are ready)
    narrative:    Optional[Dict]    = None
    setup_score:  Optional[Dict]    = None
    active_trade: Optional[Dict]    = None

    # External context (present when fetch succeeds)
    context:      Optional[Dict]    = None

    # Memory (present when matches found)
    memory:       Optional[Dict]    = None

    # Conversation history (last N turns)
    session_history: List[Dict]    = field(default_factory=list)

    # Metadata
    assembled_at:  Optional[str]    = None
    request_type:  str              = "analysis"  # "analysis"|"question"|"review"|"update"
    user_question: Optional[str]    = None        # if user asked a specific question

    def to_prompt_dict(self) -> dict:
        """
        Produces the dict that goes into the user-turn message.
        Structured for Claude to parse naturally.
        """
        d: Dict[str, Any] = {
            "assembled_at": self.assembled_at,
            "request_type": self.request_type,
        }

        if self.market_state:
            d["market"] = self.market_state

        if self.narrative:
            d["narrative"] = self.narrative

        if self.setup_score:
            d["setup_quality"] = self.setup_score

        if self.active_trade:
            d["active_trade"] = self.active_trade

        if self.context:
            d["external_context"] = self.context

        if self.memory and not _is_empty_memory(self.memory):
            d["memory_context"] = self.memory

        if self.user_question:
            d["user_question"] = self.user_question

        return d

    def estimated_tokens(self) -> int:
        """Rough token estimate (chars / 4) for budget awareness."""
        import json
        return len(json.dumps(self.to_prompt_dict())) // 4


def build_ai_context(
    market_state:   MarketState,
    context_state:  Optional[ContextState] = None,
    memory:         Optional[MemoryContext] = None,
    session_history: List[Dict]            = None,
    user_question:  Optional[str]          = None,
    request_type:   str                    = "analysis",
) -> AIContext:
    """
    Assemble an AIContext from the current system state.

    This is the single function that crosses the local/hosted boundary.
    Call it once per AI interaction, not per indicator update.
    """
    now = datetime.utcnow().isoformat() + "Z"
    state_dict = market_state.to_dict()

    ctx = AIContext(
        market_state=_compact_market(state_dict),
        narrative=state_dict.get("narrative"),
        setup_score=state_dict.get("setup", {}).get("score"),
        active_trade=state_dict.get("active_trade"),
        context=context_state.to_dict() if context_state else None,
        memory=memory.to_dict() if memory else None,
        session_history=session_history or [],
        assembled_at=now,
        request_type=request_type,
        user_question=user_question,
    )
    return ctx


def _compact_market(state_dict: dict) -> dict:
    """
    Extract only the most important fields per TF to stay within token budget.
    Full detail is available in logs; the AI needs signal, not noise.
    """
    indicators = state_dict.get("indicators", {})

    # For the AI, include: 1h, 15m, 5m (plus daily for context)
    priority_tfs = ["1d", "1h", "15m", "5m"]
    compact_indicators = {
        tf: ind for tf, ind in indicators.items()
        if tf in priority_tfs
    }

    return {
        "symbol":      state_dict.get("symbol"),
        "timestamp":   state_dict.get("timestamp"),
        "htf_bias":    state_dict.get("htf_bias"),
        "indicators":  compact_indicators,
        "trends":      state_dict.get("trends", {}),
        "support":     state_dict.get("support", [])[:3],
        "resistance":  state_dict.get("resistance", [])[:3],
    }


def _is_empty_memory(m: dict) -> bool:
    return (not m.get("similar_setups")
            and not m.get("journal_notes")
            and not m.get("patterns"))
