"""
core/module_base.py

The extensibility foundation.

Trading is domain 1 and the current priority.
This module defines the abstract base that future domains plug into.

Future domains (not implemented):
  - Productivity: task management, calendar, focus sessions
  - Research: deep-dive stock/crypto research, document analysis
  - Media: podcast queue, reading list, local file management
  - Personal workflows: custom automation, reminders

Design rule: the PersonalAssistant orchestrator (core/assistant.py)
routes user messages to the appropriate module. Each module:
  - Registers capabilities (keywords / intents it handles)
  - Contributes context to the AI layer if relevant
  - Has its own data store and state

The AI layer (ClaudeClient) is shared across all modules.
The system prompt is module-specific — trading uses TRADING_COPILOT_SYSTEM_PROMPT,
other modules would provide their own.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class ModuleContext:
    """
    Per-module context contribution to the AI.
    Each module can inject relevant data into the AI context.
    """
    module_name:  str
    summary:      str = ""          # 1-sentence state summary
    data:         Dict[str, Any] = field(default_factory=dict)
    is_active:    bool = True       # True if this module has live data


class BaseModule(ABC):
    """
    Abstract base for all domain modules.

    Subclass this to add a new domain to the assistant.
    Minimal implementation: define `name`, `keywords`, and `get_context`.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Module identifier, e.g. 'trading', 'productivity'."""
        ...

    @property
    def keywords(self) -> List[str]:
        """
        Words/phrases that route a user message to this module.
        Override to add routing hints.
        """
        return []

    @property
    def is_active(self) -> bool:
        """False for stub modules not yet implemented."""
        return True

    def get_context(self) -> Optional[ModuleContext]:
        """
        Return a context contribution for the AI.
        Return None if this module has nothing relevant right now.
        """
        return None

    async def handle(self, message: str) -> Optional[str]:
        """
        Handle a message routed to this module.
        Return a response string, or None to pass to AI layer.
        """
        return None

    def __repr__(self) -> str:
        status = "active" if self.is_active else "stub"
        return f"<Module:{self.name} [{status}]>"


# ──────────────────────────────────────────────
# Stub modules (not active in v1)
# ──────────────────────────────────────────────

class ProductivityModule(BaseModule):
    """Stub: task management, calendar, focus sessions."""

    @property
    def name(self) -> str:
        return "productivity"

    @property
    def keywords(self) -> List[str]:
        return ["task", "todo", "calendar", "focus", "reminder", "schedule"]

    @property
    def is_active(self) -> bool:
        return False   # not implemented in v1


class ResearchModule(BaseModule):
    """Stub: deep research, document analysis, stock screening."""

    @property
    def name(self) -> str:
        return "research"

    @property
    def keywords(self) -> List[str]:
        return ["research", "analyze", "deep dive", "report", "screen"]

    @property
    def is_active(self) -> bool:
        return False   # partial: stock screening lives in trading/research/


class MediaModule(BaseModule):
    """Stub: podcasts, reading list, local file management."""

    @property
    def name(self) -> str:
        return "media"

    @property
    def keywords(self) -> List[str]:
        return ["podcast", "read", "article", "queue", "playlist"]

    @property
    def is_active(self) -> bool:
        return False
