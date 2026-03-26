from __future__ import annotations

from unittest.mock import patch

import pytest

from src.research.langchain_agent.agent.subagents import build_compiled_subagents


@pytest.mark.asyncio
async def test_build_compiled_subagents_passes_store_and_checkpointer():
    captured: dict[str, object] = {}
    marker_store = object()
    marker_checkpointer = object()
    marker_backend = object()

    async def fake_builder(**kwargs):
        captured.update(kwargs)
        return {"name": "browser_control"}

    with patch.dict(
        "src.research.langchain_agent.agent.subagents._SUBAGENT_BUILDERS",
        {"browser_control": fake_builder},
        clear=True,
    ):
        result = await build_compiled_subagents(
            ["browser_control"],
            backend=marker_backend,  # type: ignore[arg-type]
            store=marker_store,  # type: ignore[arg-type]
            checkpointer=marker_checkpointer,  # type: ignore[arg-type]
        )

    assert result == [{"name": "browser_control"}]
    assert captured["backend"] is marker_backend
    assert captured["store"] is marker_store
    assert captured["checkpointer"] is marker_checkpointer


@pytest.mark.asyncio
async def test_build_compiled_subagents_supports_vercel_agent_browser():
    captured: dict[str, object] = {}
    marker_store = object()
    marker_checkpointer = object()
    marker_backend = object()

    async def fake_builder(**kwargs):
        captured.update(kwargs)
        return {"name": "vercel_agent_browser"}

    with patch.dict(
        "src.research.langchain_agent.agent.subagents._SUBAGENT_BUILDERS",
        {"vercel_agent_browser": fake_builder},
        clear=True,
    ):
        result = await build_compiled_subagents(
            ["vercel_agent_browser"],
            backend=marker_backend,  # type: ignore[arg-type]
            store=marker_store,  # type: ignore[arg-type]
            checkpointer=marker_checkpointer,  # type: ignore[arg-type]
        )

    assert result == [{"name": "vercel_agent_browser"}]
    assert captured["backend"] is marker_backend
    assert captured["store"] is marker_store
    assert captured["checkpointer"] is marker_checkpointer
