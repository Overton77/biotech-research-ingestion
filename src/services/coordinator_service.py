"""Coordinator service — thread message handling, streaming, persistence."""

import logging
from typing import Any, Callable, Awaitable

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from src.agents.coordinator import get_coordinator_graph
from src.models import Message, Thread
from beanie.odm.fields import PydanticObjectId

logger = logging.getLogger(__name__)

# Primitive HITL store: interrupt_id -> thread_id (for validation). Migrate to Redis later.
_pending_interrupts: dict[str, str] = {}


async def stream_coordinator_response(
    thread_id: str,
    user_content: str,
    *,
    on_token: Callable[[str], Awaitable[None]] | None = None,
    on_tool_start: Callable[[str, str], Awaitable[None]] | None = None,
    on_tool_end: Callable[[str, str], Awaitable[None]] | None = None,
) -> tuple[str, dict[str, Any] | None]:
    """
    Run the Coordinator with the user message, stream events, and return the full assistant text plus optional interrupt payload.

    Returns:
        (assistant_content, interrupt_payload). interrupt_payload is non-None if the run paused for HITL.
    """
    graph = get_coordinator_graph()
    config = {
        "configurable": {"thread_id": thread_id},
        "metadata": {"thread_id": thread_id},
    }
    input_messages = [HumanMessage(content=user_content)]
    full_content: list[str] = []
    interrupt_payload: dict[str, Any] | None = None

    async for event in graph.astream_events(
        {"messages": input_messages},
        config=config,
        version="v2",
    ):
        kind = event.get("event")
        if kind == "on_chat_model_stream":
            chunk = event.get("data", {}).get("chunk")
            if chunk and hasattr(chunk, "content") and chunk.content:
                token = chunk.content if isinstance(chunk.content, str) else chunk.content[0] if chunk.content else ""
                if token:
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
        elif kind == "on_chain_end":
            output = event.get("data", {}).get("output")
            if isinstance(output, dict) and output.get("__interrupt__"):
                interrupt_payload = output

    # If stream ended without interrupt in events, check state (interrupt can end the stream)
    if interrupt_payload is None:
        try:
            state = await graph.aget_state(config)
            if state and hasattr(state, "tasks") and state.tasks:
                for task in state.tasks:
                    if hasattr(task, "interrupts") and task.interrupts:
                        interrupt_payload = {"__interrupt__": list(task.interrupts)}
                        break
        except Exception:
            pass

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
    assistant_msg = Message(
        thread_id=thread_id,
        role="assistant",
        content=assistant_content,
        run_id=run_id,
        metadata={},
    )
    await assistant_msg.insert()
    thread.updated_at = assistant_msg.created_at
    await thread.save()
    logger.info("Persisted user + assistant messages for thread_id=%s", thread_id)


def register_pending_interrupt(interrupt_id: str, thread_id: str) -> None:
    """Store interrupt_id -> thread_id for validation on resume (primitive; use Redis later)."""
    _pending_interrupts[interrupt_id] = thread_id


def get_thread_id_for_interrupt(interrupt_id: str) -> str | None:
    """Return thread_id for interrupt_id, or None."""
    return _pending_interrupts.get(interrupt_id)


def clear_pending_interrupt(interrupt_id: str) -> None:
    """Remove interrupt from store after resume."""
    _pending_interrupts.pop(interrupt_id, None)


async def resume_coordinator_after_approval(
    thread_id: str,
    resume_value: dict[str, Any],
    *,
    on_token: Callable[[str], Awaitable[None]] | None = None,
    on_tool_start: Callable[[str, str], Awaitable[None]] | None = None,
    on_tool_end: Callable[[str, str], Awaitable[None]] | None = None,
) -> tuple[str, dict[str, Any] | None]:
    """
    Resume the Coordinator with Command(resume=resume_value). Stream events and return (assistant_content, interrupt_payload).
    """
    graph = get_coordinator_graph()
    config = {
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
                token = chunk.content if isinstance(chunk.content, str) else chunk.content[0] if chunk.content else ""
                if token:
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
        elif kind == "on_chain_end":
            output = event.get("data", {}).get("output")
            if isinstance(output, dict) and output.get("__interrupt__"):
                interrupt_payload = output

    if interrupt_payload is None:
        try:
            state = await graph.aget_state(config)
            if state and hasattr(state, "tasks") and state.tasks:
                for task in state.tasks:
                    if hasattr(task, "interrupts") and task.interrupts:
                        interrupt_payload = {"__interrupt__": list(task.interrupts)}
                        break
        except Exception:
            pass

    return "".join(full_content), interrupt_payload
