"""LangSmith client for prompt management (push/pull).

Prompts are defined in code as ChatPromptTemplate, pushed to LangSmith for
versioning and storage, and pulled at runtime. The LangSmith SDK caches pulled
prompts by default (see https://docs.langchain.com/langsmith/manage-prompts-programmatically).
"""

from __future__ import annotations

import os
from typing import Union

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langsmith import Client

load_dotenv()

_client: Client | None = None


def get_langsmith_client() -> Client:
    """Return the shared LangSmith client (uses LANGSMITH_API_KEY from env)."""
    global _client
    if _client is None:
        _client = Client(api_key=os.getenv("LANGSMITH_API_KEY"))
    return _client


def push_prompt(
    prompt_name: str,
    prompt: Union[ChatPromptTemplate, PromptTemplate],
) -> str:
    """Push a prompt template to LangSmith. Returns the prompt URL."""
    client = get_langsmith_client()
    return client.push_prompt(prompt_name, object=prompt)


def pull_prompt(
    prompt_name: str,
    *,
    skip_cache: bool = False,
) -> Union[ChatPromptTemplate, PromptTemplate]:
    """Pull a prompt from LangSmith. Returns a ChatPromptTemplate or PromptTemplate.

    The LangSmith SDK caches pulled prompts by default. Use skip_cache=True
    to force a fresh fetch (e.g. after updating the prompt in the UI).
    """
    client = get_langsmith_client()
    return client.pull_prompt(prompt_name, skip_cache=skip_cache)
