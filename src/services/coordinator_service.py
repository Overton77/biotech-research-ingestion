"""Coordinator service — thread message handling, streaming, persistence."""

import logging
from typing import Any, Callable, Awaitable

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


def _extract_text_parts(content: Any) -> list[str]:
    if content is None:
        return []
    if isinstance(content, str):
        return [content]
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            parts.extend(_extract_text_parts(item))
        return parts
    if isinstance(content, dict):
        if isinstance(content.get("text"), str):
            return [content["text"]]
        if isinstance(content.get("content"), str):
            return [content["content"]]
        if isinstance(content.get("output_text"), str):
            return [content["output_text"]]
        if isinstance(content.get("summary"), str):
            return [content["summary"]]
        if isinstance(content.get("value"), str):
            return [content["value"]]
        return []
    if hasattr(content, "text") and isinstance(content.text, str):
        return [content.text]
    if hasattr(content, "content"):
        return _extract_text_parts(content.content)
    return []


def _extract_assistant_from_state(state: Any) -> str:
    try:
        values = getattr(state, "values", {}) or {}
        messages = values.get("messages") or []
        for message in reversed(messages):
            role = getattr(message, "type", "") or getattr(message, "role", "")
            if role in {"ai", "assistant"}:
                return "".join(_extract_text_parts(getattr(message, "content", None))).strip()
    except Exception:
        logger.debug("Failed to extract assistant content from graph state", exc_info=True)
    return ""


async def _run_and_collect_events(
    *,
    stream: Any,
    on_token: Callable[[str], Awaitable[None]] | None = None,
    on_tool_start: Callable[[str, str], Awaitable[None]] | None = None,
    on_tool_end: Callable[[str, str], Awaitable[None]] | None = None,
) -> tuple[str, str]:
    full_content: list[str] = []
    fallback_content = ""

    async for event in stream:
        kind = event.get("event")

        if kind == "on_chat_model_stream":
            chunk = event.get("data", {}).get("chunk")
            tokens = _extract_text_parts(getattr(chunk, "content", None) if chunk else None)
            for token in tokens:
                if token:
                    full_content.append(token)
                    if on_token:
                        await on_token(token)

        elif kind == "on_chat_model_end":
            output = event.get("data", {}).get("output")
            extracted = "".join(_extract_text_parts(getattr(output, "content", output))).strip()
            if extracted:
                fallback_content = extracted

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

    return "".join(full_content).strip(), fallback_content


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
    interrupt_payload: dict[str, Any] | None = None

    streamed_content, fallback_content = await _run_and_collect_events(
        stream=graph.astream_events(input_state, config=config, version="v2"),
        on_token=on_token,
        on_tool_start=on_tool_start,
        on_tool_end=on_tool_end,
    )

    # After stream completes, check state for interrupt
    assistant_content = streamed_content or fallback_content
    try:
        state = await graph.aget_state(config)
        interrupt_payload = _extract_hitl_interrupt(state)
        if not assistant_content:
            assistant_content = _extract_assistant_from_state(state)
    except Exception:
        logger.exception("Failed to check state for interrupt")

    return assistant_content, interrupt_payload


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
    interrupt_payload: dict[str, Any] | None = None

    streamed_content, fallback_content = await _run_and_collect_events(
        stream=graph.astream_events(
            Command(resume=resume_value),
            config=config,
            version="v2",
        ),
        on_token=on_token,
        on_tool_start=on_tool_start,
        on_tool_end=on_tool_end,
    )

    assistant_content = streamed_content or fallback_content
    try:
        state = await graph.aget_state(config)
        interrupt_payload = _extract_hitl_interrupt(state)
        if not assistant_content:
            assistant_content = _extract_assistant_from_state(state)
    except Exception:
        pass

    return assistant_content, interrupt_payload
