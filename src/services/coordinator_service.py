"""Coordinator service — thread message handling, streaming, persistence."""

import logging
from typing import Any, Callable, Awaitable
from typing_extensions import TypedDict

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from src.agents.coordinator import get_coordinator_graph
from src.models import Message, Thread 
from langchain_core.runnables import RunnableConfig
from beanie.odm.fields import PydanticObjectId
from langgraph.graph.state import CompiledStateGraph

logger = logging.getLogger(__name__)

_pending_interrupts: dict[str, str] = {}


def _extract_hitl_interrupt(state: Any) -> dict[str, Any] | None:
    """Pull the HITL interrupt payload out of a LangGraph state snapshot.

    HumanInTheLoopMiddleware uses interrupt(HITLRequest) where HITLRequest is:
        {"action_requests": [...], "review_configs": [...]}

    The state.tasks[i].interrupts[j].value contains the HITLRequest dict.
    """
    try:
        if state and hasattr(state, "tasks") and state.tasks:
            for task in state.tasks:
                if hasattr(task, "interrupts") and task.interrupts:
                    return {"__interrupt__": list(task.interrupts)}
    except Exception:
        pass
    return None


async def stream_coordinator_response(
    thread_id: str,
    user_content: str,
    *,
    on_token: Callable[[str], Awaitable[None]] | None = None,
    on_tool_start: Callable[[str, str], Awaitable[None]] | None = None,
    on_tool_end: Callable[[str, str], Awaitable[None]] | None = None,
) -> tuple[str, dict[str, Any] | None]:
    """Run the Coordinator with a user message, stream events, return
    (assistant_content, interrupt_payload)."""
    graph: CompiledStateGraph = await get_coordinator_graph()
    config: RunnableConfig = {
        "configurable": {"thread_id": thread_id},
        "metadata": {"thread_id": thread_id},
    }
    input_state = {"messages": [HumanMessage(content=user_content)]}
    full_content: list[str] = []
    interrupt_payload: dict[str, Any] | None = None

    async for event in graph.astream_events(input_state, config=config, version="v2"):
        kind = event.get("event")

        if kind == "on_chat_model_stream":
            chunk = event.get("data", {}).get("chunk")
            if chunk and hasattr(chunk, "content") and chunk.content:
                content = chunk.content
                token = content if isinstance(content, str) else (content[0] if content else "")
                if token and isinstance(token, str):
                    full_content.append(token)
                    if on_token:
                        await on_token(token)

        elif kind == "on_tool_start":
            name = event.get("name", "")
            args = event.get("data", {}).get("input", {})
            args_summary = str(args)[:200] if args else ""
            if on_tool_start:
                await on_tool_start(name, args_summary)

        elif kind == "on_tool_end":
            name = event.get("name", "")
            output = event.get("data", {}).get("output", "")
            out_summary = str(output)[:300] if output else ""
            if on_tool_end:
                await on_tool_end(name, out_summary)

    # After stream completes, check state for interrupt
    try:
        state = await graph.aget_state(config)
        interrupt_payload = _extract_hitl_interrupt(state)
    except Exception:
        logger.exception("Failed to check state for interrupt")

    return "".join(full_content), interrupt_payload


async def persist_messages(
    thread_id: PydanticObjectId,
    user_content: str,
    assistant_content: str,
    run_id: str | None = None,
) -> None:
    """Save user and assistant messages to MongoDB and bump thread updated_at."""
    thread = await Thread.get(thread_id)
    if not thread:
        return
    user_msg = Message(
        thread_id=thread_id,
        role="user",
        content=user_content,
        metadata={"run_id": run_id} if run_id else {},
    )
    await user_msg.insert()
    if assistant_content.strip():
        assistant_msg = Message(
            thread_id=thread_id,
            role="assistant",
            content=assistant_content,
            run_id=run_id,
            metadata={},
        )
        await assistant_msg.insert()
        thread.updated_at = assistant_msg.created_at
    else:
        thread.updated_at = user_msg.created_at
    await thread.save()
    logger.info("Persisted messages for thread_id=%s", thread_id)


def register_pending_interrupt(interrupt_id: str, thread_id: str) -> None:
    _pending_interrupts[interrupt_id] = thread_id


def get_thread_id_for_interrupt(interrupt_id: str) -> str | None:
    return _pending_interrupts.get(interrupt_id)


def clear_pending_interrupt(interrupt_id: str) -> None:
    _pending_interrupts.pop(interrupt_id, None)


async def resume_coordinator_after_approval(
    thread_id: str,
    resume_value: dict[str, Any],
    *,
    on_token: Callable[[str], Awaitable[None]] | None = None,
    on_tool_start: Callable[[str, str], Awaitable[None]] | None = None,
    on_tool_end: Callable[[str, str], Awaitable[None]] | None = None,
) -> tuple[str, dict[str, Any] | None]:
    """Resume the Coordinator after a HITL decision.

    resume_value must be in HumanInTheLoopMiddleware format:
        Approve  -> {"decisions": [{"type": "approve"}]}
        Reject   -> {"decisions": [{"type": "reject", "message": "..."}]}
        Edit     -> {"decisions": [{"type": "edit", "edited_action": {"name": ..., "args": {...}}}]}
    """
    graph: CompiledStateGraph = await get_coordinator_graph()
    config: RunnableConfig = {
        "configurable": {"thread_id": thread_id},
        "metadata": {"thread_id": thread_id},
    }
    full_content: list[str] = []
    interrupt_payload: dict[str, Any] | None = None

    async for event in graph.astream_events(
        Command(resume=resume_value),
        config=config,
        version="v2",
    ):
        kind = event.get("event")

        if kind == "on_chat_model_stream":
            chunk = event.get("data", {}).get("chunk")
            if chunk and hasattr(chunk, "content") and chunk.content:
                content = chunk.content
                token = content if isinstance(content, str) else (content[0] if content else "")
                if token and isinstance(token, str):
                    full_content.append(token)
                    if on_token:
                        await on_token(token)

        elif kind == "on_tool_start":
            name = event.get("name", "")
            args = event.get("data", {}).get("input", {})
            args_summary = str(args)[:200] if args else ""
            if on_tool_start:
                await on_tool_start(name, args_summary)

        elif kind == "on_tool_end":
            name = event.get("name", "")
            output = event.get("data", {}).get("output", "")
            out_summary = str(output)[:300] if output else ""
            if on_tool_end:
                await on_tool_end(name, out_summary)

    try:
        state = await graph.aget_state(config)
        interrupt_payload = _extract_hitl_interrupt(state)
    except Exception:
        pass

    return "".join(full_content), interrupt_payload
