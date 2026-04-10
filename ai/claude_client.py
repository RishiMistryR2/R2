"""
ClaudeClient: async wrapper around the Anthropic API.

Handles:
  - API authentication
  - Request construction from AIContext
  - Streaming responses (yields text chunks)
  - Session history management
  - Token budget enforcement
  - Graceful error handling

The client is the only place in the codebase that imports anthropic.
Everything else is decoupled from the specific API client.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import AsyncIterator, Dict, List, Optional

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

from ai.ai_context import AIContext
from ai.system_prompt import TRADING_COPILOT_SYSTEM_PROMPT, UPDATE_SYSTEM_PROMPT
from config.settings import (
    ANTHROPIC_API_KEY,
    CLAUDE_MODEL,
    CLAUDE_MAX_TOKENS,
)

logger = logging.getLogger(__name__)

# Max conversation turns to keep in history (controls token growth)
MAX_HISTORY_TURNS = 8


class ClaudeClient:
    """
    Async Claude API client with session history.

    Usage:
        client = ClaudeClient()
        response = await client.ask("Is this still a pullback?", context)
        async for chunk in client.stream("Calls or puts here?", context):
            print(chunk, end="", flush=True)
    """

    def __init__(self, api_key: str = None) -> None:
        self._key  = api_key or ANTHROPIC_API_KEY
        self._history: List[Dict] = []
        self._client = None
        if HAS_ANTHROPIC and self._key:
            self._client = anthropic.AsyncAnthropic(api_key=self._key)
        elif not HAS_ANTHROPIC:
            logger.warning("anthropic package not installed — AI layer unavailable")
        elif not self._key:
            logger.warning("ANTHROPIC_API_KEY not set — AI layer unavailable")

    @property
    def is_available(self) -> bool:
        return self._client is not None

    # ──────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────

    async def ask(
        self,
        question:     str,
        context:      AIContext,
        brief:        bool = False,
    ) -> str:
        """
        Send a question + context to Claude and return the full response.
        Adds the exchange to session history.
        """
        if not self.is_available:
            return "[AI unavailable — check ANTHROPIC_API_KEY]"

        context.user_question = question
        context.request_type  = "question"

        user_msg = self._build_user_message(context)
        system   = UPDATE_SYSTEM_PROMPT if brief else TRADING_COPILOT_SYSTEM_PROMPT
        messages = self._history[-MAX_HISTORY_TURNS * 2:] + [
            {"role": "user", "content": user_msg}
        ]

        try:
            response = await self._client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=CLAUDE_MAX_TOKENS,
                system=system,
                messages=messages,
            )
            reply = response.content[0].text
            self._add_to_history(user_msg, reply)
            return reply

        except Exception as e:
            logger.error(f"Claude API error: {e}")
            return f"[API error: {e}]"

    async def stream(
        self,
        question:     str,
        context:      AIContext,
    ) -> AsyncIterator[str]:
        """
        Stream a response from Claude. Yields text chunks as they arrive.
        Use this for the CLI to show tokens as they stream.
        """
        if not self.is_available:
            yield "[AI unavailable — check ANTHROPIC_API_KEY]"
            return

        context.user_question = question
        user_msg = self._build_user_message(context)
        messages = self._history[-MAX_HISTORY_TURNS * 2:] + [
            {"role": "user", "content": user_msg}
        ]

        full_reply = []
        try:
            async with self._client.messages.stream(
                model=CLAUDE_MODEL,
                max_tokens=CLAUDE_MAX_TOKENS,
                system=TRADING_COPILOT_SYSTEM_PROMPT,
                messages=messages,
            ) as stream:
                async for text in stream.text_stream:
                    full_reply.append(text)
                    yield text

            self._add_to_history(user_msg, "".join(full_reply))

        except Exception as e:
            logger.error(f"Claude stream error: {e}")
            yield f"\n[stream error: {e}]"

    async def proactive_commentary(self, context: AIContext) -> str:
        """
        Generate an unprompted market update.
        Used when a significant event fires (trend flip, setup formed, etc.)
        Uses a brief, focused prompt to avoid token waste.
        """
        if not self.is_available:
            return ""

        context.request_type = "update"
        user_msg = self._build_update_message(context)
        messages = self._history[-4:] + [
            {"role": "user", "content": user_msg}
        ]

        try:
            response = await self._client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=400,   # brief updates only
                system=UPDATE_SYSTEM_PROMPT,
                messages=messages,
            )
            reply = response.content[0].text
            self._add_to_history(user_msg, reply)
            return reply
        except Exception as e:
            logger.error(f"Proactive commentary error: {e}")
            return ""

    def clear_history(self) -> None:
        """Reset conversation history (new session)."""
        self._history = []

    def get_history(self) -> List[Dict]:
        return list(self._history)

    # ──────────────────────────────────────────
    # Message builders
    # ──────────────────────────────────────────

    @staticmethod
    def _build_user_message(context: AIContext) -> str:
        """
        Format the AIContext into a user-turn message.
        Claude receives structured JSON + the question.
        """
        payload = context.to_prompt_dict()
        question = context.user_question or ""

        # Structured data block first, question after
        msg = f"<market_data>\n{json.dumps(payload, indent=2)}\n</market_data>"
        if question:
            msg += f"\n\n{question}"
        return msg

    @staticmethod
    def _build_update_message(context: AIContext) -> str:
        """
        Compact message for proactive updates.
        Only includes the most relevant fields.
        """
        payload = {
            "event":     context.request_type,
            "narrative": context.narrative,
            "setup":     context.setup_score,
            "trade":     context.active_trade,
            "risk_note": (context.context or {}).get("risk_note", ""),
        }
        return (
            f"<update>\n{json.dumps(payload, indent=2)}\n</update>\n"
            "Briefly explain what just changed and what to watch."
        )

    def _add_to_history(self, user_msg: str, assistant_reply: str) -> None:
        self._history.append({"role": "user",      "content": user_msg})
        self._history.append({"role": "assistant", "content": assistant_reply})
        # Trim if too long
        if len(self._history) > MAX_HISTORY_TURNS * 2:
            self._history = self._history[-MAX_HISTORY_TURNS * 2:]
