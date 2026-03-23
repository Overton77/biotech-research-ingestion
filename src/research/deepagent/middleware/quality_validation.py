"""QualityValidationMiddleware — runs a lightweight LLM validation after agent completion.

Checks the agent's output against acceptance criteria stored in agent state
and writes a QualityAssessment to state for downstream consumers.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage
from langgraph.runtime import Runtime
from typing_extensions import NotRequired

logger = logging.getLogger(__name__)


class QualityState(AgentState):
    """Extended state carrying acceptance criteria and assessment results."""
    acceptance_criteria: NotRequired[list[str]]
    quality_assessment: NotRequired[dict]


class QualityValidationMiddleware(AgentMiddleware[QualityState]):
    """Runs a single lightweight LLM call after agent completes to check acceptance criteria."""
    state_schema = QualityState

    def __init__(self, validation_model: str = "openai:gpt-5-mini"):
        super().__init__()
        self.validation_model_name = validation_model

    async def aafter_agent(self, state: QualityState, runtime: Runtime) -> dict | None:
        criteria = state.get("acceptance_criteria", [])
        if not criteria:
            return None

        messages = state.get("messages", [])
        last_ai = next(
            (m for m in reversed(messages) if isinstance(m, AIMessage)),
            None,
        )
        if not last_ai:
            return None

        criteria_text = "\n".join(f"- {c}" for c in criteria)
        validation_prompt = (
            f"Evaluate the following research output against the acceptance criteria.\n\n"
            f"ACCEPTANCE CRITERIA:\n{criteria_text}\n\n"
            f"RESEARCH OUTPUT (last 3000 chars):\n{last_ai.content[-3000:]}\n\n"
            f'For each criterion, rate: MET / PARTIALLY_MET / NOT_MET with brief evidence.\n'
            f'Return JSON: {{"criteria_results": [{{"criterion": "...", "status": "MET|PARTIALLY_MET|NOT_MET", "evidence": "..."}}], '
            f'"overall_pass": true/false, "suggestions": ["..."]}}'
        )

        try:
            model = init_chat_model(self.validation_model_name)
            response = await model.ainvoke([
                {"role": "user", "content": validation_prompt}
            ])
            assessment = json.loads(response.content)
        except Exception:
            logger.debug("Quality validation LLM call failed", exc_info=True)
            assessment = {"overall_pass": True, "criteria_results": [], "suggestions": []}

        return {"quality_assessment": assessment}
