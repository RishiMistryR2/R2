"""
ai/claude_client.py — R2 Claude API wrapper (Phase 8).

This is the only file in R2 that calls the Anthropic API directly.

Features:
  - Mock mode for testing (no real API calls)
  - Graceful fallback when API unavailable
  - Timeout and error handling
  - Session history management
  - ParsedResponse always returned (never raw dicts)

Public sync functions (Phase 8 integration API):
  generate_startup_commentary(package)    → ParsedResponse
  generate_market_commentary(package)     → ParsedResponse
  generate_setup_explanation(package)     → ParsedResponse
  generate_trade_review(package)          → ParsedResponse
  answer_user_question(package, q)        → ParsedResponse
  generate_end_of_day_reflection(package) → ParsedResponse

For async streaming use ClaudeClient directly.
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

from ai.r2_persona import R2_SYSTEM_PROMPT, R2_BRIEF_SYSTEM_PROMPT
from ai.state_packager import StatePackage
from ai.prompt_builder import (
    build_startup_briefing_prompt,
    build_market_commentary_prompt,
    build_setup_explanation_prompt,
    build_trade_review_prompt,
    build_warning_explanation_prompt,
    build_end_of_day_prompt,
    build_user_question_prompt,
)
from ai.response_parser import ParsedResponse, parse_response, make_fallback_response
from config.settings import (
    ANTHROPIC_API_KEY,
    CLAUDE_MODEL,
    CLAUDE_MAX_TOKENS,
    CLAUDE_UPDATE_TOKENS,
    AI_HISTORY_MAX_TURNS,
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Mock responses (used in test / mock mode)
# ──────────────────────────────────────────────

_MOCK_RESPONSES: Dict[str, str] = {
    "startup_briefing": (
        "Pre-market: SPY is in a trend-bull regime with moderate confidence. "
        "No elevated macro risk detected. "
        "The structure favors continuation setups on pullbacks to VWAP. "
        "Watch for a clean 5m setup in the first 30 minutes."
    ),
    "market_commentary": (
        "Current narrative is trending-bull with aligned timeframes. "
        "Price is holding above VWAP; RSI is in a healthy zone. "
        "Regime confidence is adequate for selective entries."
    ),
    "setup_explanation": (
        "This is a B-grade setup. Trend alignment is solid across 1h and 15m, "
        "but entry timing is marginal — the 5m candle has not yet confirmed the "
        "pullback completion. Stop sits below the 15m swing low. "
        "R:R is acceptable at 1.5. Main risk: context bias is neutral, not aligned. "
        "Worth monitoring for a cleaner entry candle."
    ),
    "trade_review": (
        "Trade thesis is intact. Price is holding above the entry level and "
        "stop has not been approached. Time stop window has not elapsed. "
        "Monitor the 15m structure for any deterioration."
    ),
    "end_of_day": (
        "Session shows adequate execution quality. "
        "Win rate and expectancy are within historical norms. "
        "Key lesson: continue waiting for A-grade setups rather than forcing B setups "
        "in choppy conditions."
    ),
    "user_answer": (
        "Based on the current data: the structure suggests the setup remains valid. "
        "The thesis is intact as long as price holds above the indicated support level."
    ),
}


# ──────────────────────────────────────────────
# Async ClaudeClient
# ──────────────────────────────────────────────

class ClaudeClient:
    """
    Async Claude API client with session history and mock mode.

    Usage:
        client = ClaudeClient()
        response = await client.ask_with_package(package, "startup_briefing")

    Mock mode (for tests — no network):
        client = ClaudeClient(mock=True)
    """

    def __init__(
        self,
        api_key:  Optional[str] = None,
        mock:     bool          = False,
        timeout:  float         = 30.0,
    ) -> None:
        self._mock    = mock
        self._timeout = timeout
        self._history: List[Dict] = []
        self._client  = None

        if mock:
            logger.info("[R2-AI] Mock mode enabled — no API calls will be made")
            return

        key = api_key or ANTHROPIC_API_KEY
        if HAS_ANTHROPIC and key:
            self._client = anthropic.AsyncAnthropic(api_key=key, timeout=timeout)
        elif not HAS_ANTHROPIC:
            logger.warning("[R2-AI] anthropic package not installed — AI unavailable")
        else:
            logger.warning("[R2-AI] ANTHROPIC_API_KEY not set — AI unavailable")

    @property
    def is_available(self) -> bool:
        return self._mock or (self._client is not None)

    # ──────────────────────────────────────────
    # Phase 8 package-based API
    # ──────────────────────────────────────────

    async def ask_with_package(
        self,
        package:       StatePackage,
        request_type:  str           = "market_commentary",
        user_question: Optional[str] = None,
        direction:     Optional[str] = None,
        warning_msg:   Optional[str] = None,
        max_tokens:    Optional[int] = None,
    ) -> ParsedResponse:
        """
        Main Phase 8 entry point.
        Builds the correct prompt, calls Claude, returns ParsedResponse.
        """
        if not self.is_available:
            return make_fallback_response(
                "API unavailable", request_type, package.risk_flags
            )

        if self._mock:
            key = {
                "startup_briefing":  "startup_briefing",
                "market_commentary": "market_commentary",
                "setup_explanation": "setup_explanation",
                "trade_review":      "trade_review",
                "end_of_day":        "end_of_day",
                "user_question":     "user_answer",
            }.get(request_type, "market_commentary")
            raw = _MOCK_RESPONSES[key]
            return parse_response(raw, request_type, package.risk_flags)

        system, user = self._build_prompt(
            package, request_type, user_question, direction, warning_msg
        )
        tokens = max_tokens or (
            CLAUDE_UPDATE_TOKENS if request_type == "market_commentary"
            else CLAUDE_MAX_TOKENS
        )
        messages = self._history[-(AI_HISTORY_MAX_TURNS * 2):] + [
            {"role": "user", "content": user}
        ]

        try:
            resp = await self._client.messages.create(
                model=CLAUDE_MODEL, max_tokens=tokens,
                system=system, messages=messages,
            )
            raw  = resp.content[0].text
            self._add_to_history(user, raw)
            return parse_response(raw, request_type, package.risk_flags)
        except Exception as e:
            logger.error(f"[R2-AI] API error ({request_type}): {e}")
            return make_fallback_response(str(e), request_type, package.risk_flags)

    async def stream_with_package(
        self,
        package:      StatePackage,
        question:     str,
        request_type: str = "user_question",
    ) -> AsyncIterator[str]:
        """Stream a response token-by-token. Yields str chunks."""
        if not self.is_available:
            yield make_fallback_response("API unavailable").text
            return

        if self._mock:
            yield _MOCK_RESPONSES.get("user_answer", "Analysis unavailable.")
            return

        system, user = self._build_prompt(
            package, request_type, user_question=question
        )
        messages = self._history[-(AI_HISTORY_MAX_TURNS * 2):] + [
            {"role": "user", "content": user}
        ]

        full_reply: List[str] = []
        try:
            async with self._client.messages.stream(
                model=CLAUDE_MODEL, max_tokens=CLAUDE_MAX_TOKENS,
                system=system, messages=messages,
            ) as stream:
                async for text in stream.text_stream:
                    full_reply.append(text)
                    yield text
            self._add_to_history(user, "".join(full_reply))
        except Exception as e:
            logger.error(f"[R2-AI] Stream error: {e}")
            yield f"\n[stream error: {e}]"

    # ──────────────────────────────────────────
    # Legacy API (Phase 1 compatibility)
    # ──────────────────────────────────────────

    async def ask(self, question: str, context, brief: bool = False) -> str:
        """Legacy single-string ask.  Preserved for Phase 1 compatibility."""
        if self._mock:
            return _MOCK_RESPONSES["user_answer"]
        if not self._client:
            return "[AI unavailable — check ANTHROPIC_API_KEY]"
        context.user_question = question
        context.request_type  = "question"
        user_msg = ClaudeClient._build_legacy_user_message(context)
        system   = R2_BRIEF_SYSTEM_PROMPT if brief else R2_SYSTEM_PROMPT
        messages = self._history[-(AI_HISTORY_MAX_TURNS * 2):] + [
            {"role": "user", "content": user_msg}
        ]
        try:
            resp  = await self._client.messages.create(
                model=CLAUDE_MODEL, max_tokens=CLAUDE_MAX_TOKENS,
                system=system, messages=messages,
            )
            reply = resp.content[0].text
            self._add_to_history(user_msg, reply)
            return reply
        except Exception as e:
            logger.error(f"[R2-AI] Legacy ask error: {e}")
            return f"[API error: {e}]"

    async def proactive_commentary(self, context) -> str:
        """Legacy proactive update.  Preserved for compatibility."""
        if self._mock:
            return _MOCK_RESPONSES["market_commentary"]
        if not self._client:
            return ""
        context.request_type = "update"
        user_msg = ClaudeClient._build_legacy_update_message(context)
        messages = self._history[-4:] + [{"role": "user", "content": user_msg}]
        try:
            resp  = await self._client.messages.create(
                model=CLAUDE_MODEL, max_tokens=CLAUDE_UPDATE_TOKENS,
                system=R2_BRIEF_SYSTEM_PROMPT, messages=messages,
            )
            reply = resp.content[0].text
            self._add_to_history(user_msg, reply)
            return reply
        except Exception as e:
            logger.error(f"[R2-AI] Proactive error: {e}")
            return ""

    def clear_history(self) -> None:
        self._history = []

    def get_history(self) -> List[Dict]:
        return list(self._history)

    # ──────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────

    @staticmethod
    def _build_prompt(
        package:       StatePackage,
        request_type:  str,
        user_question: Optional[str] = None,
        direction:     Optional[str] = None,
        warning_msg:   Optional[str] = None,
    ):
        if request_type == "startup_briefing":
            return build_startup_briefing_prompt(package)
        elif request_type == "market_commentary":
            return build_market_commentary_prompt(package)
        elif request_type == "setup_explanation":
            return build_setup_explanation_prompt(package, direction)
        elif request_type == "trade_review":
            return build_trade_review_prompt(package)
        elif request_type == "warning_note":
            return build_warning_explanation_prompt(package, warning_msg or "")
        elif request_type == "end_of_day":
            return build_end_of_day_prompt(package)
        else:
            q = user_question or request_type
            return build_user_question_prompt(package, q)

    def _add_to_history(self, user_msg: str, reply: str) -> None:
        self._history.append({"role": "user",      "content": user_msg})
        self._history.append({"role": "assistant", "content": reply})
        max_turns = AI_HISTORY_MAX_TURNS * 2
        if len(self._history) > max_turns:
            self._history = self._history[-max_turns:]

    @staticmethod
    def _build_legacy_user_message(context) -> str:
        payload  = context.to_prompt_dict()
        question = context.user_question or ""
        msg = f"<market_data>\n{json.dumps(payload, indent=2)}\n</market_data>"
        if question:
            msg += f"\n\n{question}"
        return msg

    @staticmethod
    def _build_legacy_update_message(context) -> str:
        payload = {
            "event":     context.request_type,
            "narrative": context.narrative,
            "setup":     context.setup_score,
            "trade":     context.active_trade,
        }
        return (
            f"<update>\n{json.dumps(payload, indent=2)}\n</update>\n"
            "Briefly explain what just changed and what to watch."
        )


# ──────────────────────────────────────────────
# Synchronous integration API (Phase 8 public)
# ──────────────────────────────────────────────

def _run_async(coro):
    """Run an async coroutine from sync code safely."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


def _client(mock: bool = False) -> ClaudeClient:
    return ClaudeClient(mock=mock)


def generate_startup_commentary(
    package: StatePackage, mock: bool = False
) -> ParsedResponse:
    """Pre-market orientation."""
    return _run_async(_client(mock).ask_with_package(package, "startup_briefing"))


def generate_market_commentary(
    package: StatePackage, mock: bool = False
) -> ParsedResponse:
    """Live market observation."""
    return _run_async(_client(mock).ask_with_package(package, "market_commentary"))


def generate_setup_explanation(
    package: StatePackage, direction: Optional[str] = None, mock: bool = False
) -> ParsedResponse:
    """Full setup analysis."""
    return _run_async(
        _client(mock).ask_with_package(
            package, "setup_explanation", direction=direction
        )
    )


def generate_trade_review(
    package: StatePackage, mock: bool = False
) -> ParsedResponse:
    """Active trade review."""
    return _run_async(_client(mock).ask_with_package(package, "trade_review"))


def generate_end_of_day_reflection(
    package: StatePackage, mock: bool = False
) -> ParsedResponse:
    """End-of-session reflection."""
    return _run_async(_client(mock).ask_with_package(package, "end_of_day"))


def answer_user_question(
    package: StatePackage, question: str, mock: bool = False
) -> ParsedResponse:
    """Answer a specific user question."""
    return _run_async(
        _client(mock).ask_with_package(
            package, "user_question", user_question=question
        )
    )