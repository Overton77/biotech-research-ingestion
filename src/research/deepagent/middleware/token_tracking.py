"""Token tracking middleware — tracks cumulative token usage after each model call.

Uses an @after_model hook to read usage_metadata from the last AI message
and accumulate total_tokens_used in agent state.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain.agents.middleware import AgentState, after_model
from langgraph.runtime import Runtime
from typing_extensions import NotRequired

logger = logging.getLogger(__name__)


class TokenTrackingState(AgentState):
    """Extended state that tracks cumulative token usage."""
    total_tokens_used: NotRequired[int]


@after_model(state_schema=TokenTrackingState)
def track_tokens(state: TokenTrackingState, runtime: Runtime) -> dict | None:
    """Track cumulative token usage after each model call."""
    messages = state.get("messages", [])
    if not messages:
        return None

    last_msg = messages[-1]
    usage = getattr(last_msg, "usage_metadata", None)
    tokens_this_call = 0
    if usage:
        tokens_this_call = usage.get("total_tokens", 0) if isinstance(usage, dict) else 0

    if tokens_this_call <= 0:
        return None

    current_total = state.get("total_tokens_used", 0)
    return {"total_tokens_used": current_total + tokens_this_call}
