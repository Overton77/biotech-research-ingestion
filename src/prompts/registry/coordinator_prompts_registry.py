"""Coordinator agent prompts: defined here, pushed to LangSmith, pulled at runtime.

Flow:
  1. Prompt text and ChatPromptTemplate are defined in this file.
  2. Push to LangSmith via push_coordinator_prompt() (e.g. from a script or deploy step).
  3. Coordinator pulls via get_coordinator_system_prompt(use_langsmith=True);
     LangSmith SDK handles caching; no local cache needed.
"""

from __future__ import annotations

import logging
from typing import Union

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

from src.clients.langsmith_client import pull_prompt, push_prompt
from src.prompts.coordinator_prompt_builders import COORDINATOR_SYSTEM_PROMPT_TEXT

logger = logging.getLogger(__name__)



# ---------------------------------------------------------------------------
# ChatPromptTemplate: built from the text above.
# Use {variable} only for real template variables (e.g. {context}); escape
# literal braces as {{ and }} (e.g. {{ "url": str }} in schema descriptions).
# ---------------------------------------------------------------------------

COORDINATOR_SYSTEM_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", COORDINATOR_SYSTEM_PROMPT_TEXT),
])

COORDINATOR_PROMPTS: dict[str, ChatPromptTemplate] = {
    "coordinator_system": COORDINATOR_SYSTEM_TEMPLATE,
}

COORDINATOR_PROMPT_NAME = "coordinator_system"


def render_system_message(
    template: Union[ChatPromptTemplate, PromptTemplate],
    **variables: str,
) -> str:
    """
    Render the prompt template with the given variables and return the system message content.

    Use this when the template has placeholders (e.g. {context}). Pass them as kwargs:
        render_system_message(template, context="...")
    For templates with no variables, call with no args: render_system_message(template).
    """
    prompt_value = template.invoke(variables)
    messages = getattr(prompt_value, "messages", None)
    if messages:
        msg = messages[0]
        return getattr(msg, "content", str(msg)) or ""
    return getattr(prompt_value, "to_string", lambda: str(prompt_value))() or ""


def get_coordinator_system_prompt_from_registry(
    key: str = "coordinator_system",
    **variables: str,
) -> str:
    """Return the system prompt string from the local registry (no LangSmith)."""
    t = COORDINATOR_PROMPTS.get(key)
    if t is None:
        raise KeyError(f"Unknown coordinator prompt key: {key!r}")
    return render_system_message(t, **variables)


def get_coordinator_system_prompt(
    use_langsmith: bool = True,
    *,
    skip_cache: bool = False,
    **variables: str,
) -> str:
    """
    Return the coordinator system prompt string, optionally rendered with variables.

    If use_langsmith is True, pulls from LangSmith (SDK caches by default).
    On pull failure or use_langsmith=False, falls back to the local registry.
    Pass any template variables (e.g. context) as kwargs so they are filled in:
        get_coordinator_system_prompt(use_langsmith=True, context="...")
    """
    if use_langsmith:
        try:
            template = pull_prompt(COORDINATOR_PROMPT_NAME, skip_cache=skip_cache)
            out = render_system_message(template, **variables)
            logger.debug("Coordinator system prompt loaded from LangSmith")
            return out
        except Exception as e:
            logger.warning(
                "Failed to pull coordinator prompt from LangSmith: %s; using registry",
                e,
            )
    return get_coordinator_system_prompt_from_registry(**variables)


def push_coordinator_prompt(prompt_name: str | None = None) -> str:
    """
    Push the coordinator system prompt to LangSmith. Returns the prompt URL.

    Run from a script or deploy step to sync the in-file prompt to LangSmith:
        python -m src.prompts.registry.coordinator_prompts
    """
    name = prompt_name or COORDINATOR_PROMPT_NAME
    template = COORDINATOR_PROMPTS["coordinator_system"]
    url = push_prompt(name, template)
    logger.info("Pushed coordinator prompt %s to LangSmith: %s", name, url)
    return url


if __name__ == "__main__":
    push_coordinator_prompt()
    print(get_coordinator_system_prompt(use_langsmith=True))
