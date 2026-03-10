# src/temporal/activities/openai_research.py
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import aiofiles
from beanie import init_beanie
from pymongo import AsyncMongoClient
from temporalio import activity

from src.agents.native_openai import openai_client
from src.config import get_settings
from src.models import Message, Thread
from src.models.openai_research import OpenAIResearchPlan, OpenAIResearchRun
from src.models.plan import ResearchPlan
from src.services.openai_research_prompt_builder import build_openai_research_input
from src.utils.now import utc_now

logger = logging.getLogger(__name__)

TERMINAL_STATUSES = {"completed", "failed", "incomplete", "cancelled"}
SUPPORTED_DEEP_RESEARCH_TOOLS = {"web_search_preview", "file_search", "code_interpreter"}

_beanie_initialized = False
_mongo_client: AsyncMongoClient | None = None


async def ensure_beanie_initialized() -> None:
    global _beanie_initialized, _mongo_client

    if _beanie_initialized:
        return

    settings = get_settings()
    _mongo_client = AsyncMongoClient(settings.MONGODB_URI)
    db = _mongo_client[settings.MONGODB_DB]

    await init_beanie(
        database=db,
        document_models=[
            Thread,
            Message,
            ResearchPlan,
            OpenAIResearchPlan,
            OpenAIResearchRun,
        ],
    )
    _beanie_initialized = True
    logger.info("Temporal activities Beanie initialized")


def append_status_event(
    history: list[dict[str, Any]],
    *,
    source: str,
    status: str,
    details: dict[str, Any] | None = None,
) -> None:
    if history and history[-1].get("source") == source and history[-1].get("status") == status:
        return

    history.append(
        {
            "source": source,
            "status": status,
            "at": utc_now().isoformat(),
            "details": details or {},
        }
    )


def set_internal_status(
    run: OpenAIResearchRun,
    status: str,
    *,
    details: dict[str, Any] | None = None,
) -> None:
    if run.status != status:
        run.status = status
        append_status_event(run.status_history, source="internal", status=status, details=details)


def set_openai_status(
    run: OpenAIResearchRun,
    status: str | None,
    *,
    details: dict[str, Any] | None = None,
) -> None:
    if not status:
        return

    if run.openai_status != status:
        run.openai_status = status
        append_status_event(run.status_history, source="openai", status=status, details=details)


def serialize_exception(exc: Exception) -> dict[str, Any]:
    return {
        "type": exc.__class__.__name__,
        "message": str(exc),
    }


def dedupe_dicts(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()

    for item in items:
        key = json.dumps(item, sort_keys=True, ensure_ascii=False, default=str)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)

    return deduped


def build_tools_for_openai(plan: OpenAIResearchPlan) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    vector_store_ids = sorted(
        {
            source.value
            for source in plan.seeded_sources
            if source.type == "vector_store_id" and source.value
        }
    )

    for tool_name in plan.tools or ["web_search_preview"]:
        if tool_name not in SUPPORTED_DEEP_RESEARCH_TOOLS:
            raise ValueError(
                f"Unsupported OpenAI Deep Research tool '{tool_name}'. "
                f"Supported tools for this integration: {sorted(SUPPORTED_DEEP_RESEARCH_TOOLS)}"
            )

        if tool_name == "web_search_preview":
            tools.append({"type": "web_search_preview"})
        elif tool_name == "code_interpreter":
            tools.append({"type": "code_interpreter", "container": {"type": "auto"}})
        elif tool_name == "file_search":
            if not vector_store_ids:
                raise ValueError(
                    "file_search was requested, but no seeded_sources of type "
                    "'vector_store_id' were provided."
                )
            tools.append({"type": "file_search", "vector_store_ids": vector_store_ids})

    return tools or [{"type": "web_search_preview"}]


def build_request_metadata(plan: OpenAIResearchPlan) -> dict[str, Any]:
    return {
        "plan_id": str(plan.id),
        "thread_id": str(plan.thread_id),
        "plan_title": plan.title,
        "objective": plan.objective,
        "requested_tools": list(plan.tools),
        "seeded_sources": [source.model_dump(mode="json") for source in plan.seeded_sources],
    }


def build_request_payload(
    *,
    model: str,
    request_input: str,
    request_tools: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "model": model,
        "input": request_input,
        "background": True,
        "tools": request_tools,
    }


@activity.defn
async def launch_openai_research_run(openai_research_run_id: str) -> dict[str, Any]:
    await ensure_beanie_initialized()

    run = await OpenAIResearchRun.get(openai_research_run_id)
    if run is None:
        raise ValueError(f"OpenAIResearchRun not found: {openai_research_run_id}")

    if run.openai_response_id:
        logger.info(
            "OpenAI research run %s already launched with response_id=%s",
            run.id,
            run.openai_response_id,
        )
        return {
            "openai_research_run_id": str(run.id),
            "openai_response_id": run.openai_response_id,
            "status": run.openai_status,
        }

    plan = await OpenAIResearchPlan.get(run.openai_research_plan_id)
    if plan is None:
        raise ValueError(f"OpenAIResearchPlan not found: {run.openai_research_plan_id}")

    request_input = build_openai_research_input(plan)
    request_tools = build_tools_for_openai(plan)
    request_payload = build_request_payload(
        model=plan.model,
        request_input=request_input,
        request_tools=request_tools,
    )

    run.model = plan.model
    run.request_input = request_input
    run.request_tools = request_tools
    run.request_metadata = build_request_metadata(plan)
    run.openai_request_payload = request_payload
    run.updated_at = utc_now()

    try:
        response = await openai_client.responses.create(**request_payload)
    except Exception as exc:
        run.error_message = f"Failed to launch OpenAI Deep Research run: {exc}"
        run.error_payload = serialize_exception(exc)
        run.failed_at = utc_now()
        run.updated_at = utc_now()
        set_internal_status(run, "failed", details={"stage": "launch"})
        await run.save()

        plan.status = "failed"
        plan.updated_at = utc_now()
        await plan.save()
        raise

    response_payload = response.model_dump(mode="json")
    response_status = response_payload.get("status")

    run.openai_response_id = response.id
    run.openai_initial_response = response_payload
    run.submitted_at = utc_now()
    run.started_at = run.started_at or utc_now()
    run.updated_at = utc_now()
    set_internal_status(run, "in_progress", details={"stage": "launch"})
    set_openai_status(run, response_status, details={"response_id": response.id})
    await run.save()

    plan.status = "submitted"
    plan.updated_at = utc_now()
    await plan.save()

    logger.info("Launched OpenAI research run %s -> response_id=%s", run.id, response.id)

    return {
        "openai_research_run_id": str(run.id),
        "openai_response_id": response.id,
        "status": response_status,
    }


@activity.defn
async def poll_openai_research_run(payload: dict[str, Any]) -> dict[str, Any]:
    await ensure_beanie_initialized()

    openai_research_run_id = payload["openai_research_run_id"]
    openai_response_id = payload["openai_response_id"]

    run = await OpenAIResearchRun.get(openai_research_run_id)
    if run is None:
        raise ValueError(f"OpenAIResearchRun not found: {openai_research_run_id}")

    response = await openai_client.responses.retrieve(openai_response_id)
    response_payload = response.model_dump(mode="json")
    response_status = response_payload.get("status")

    run.last_polled_at = utc_now()
    run.updated_at = utc_now()
    set_openai_status(run, response_status, details={"response_id": openai_response_id})
    await run.save()

    logger.info("Polled response_id=%s status=%s", openai_response_id, response_status)

    return {
        "openai_research_run_id": openai_research_run_id,
        "openai_response_id": openai_response_id,
        "status": response_status,
        "is_terminal": response_status in TERMINAL_STATUSES,
        "response": response_payload if response_status in TERMINAL_STATUSES else None,
    }


@activity.defn
async def persist_openai_research_result(payload: dict[str, Any]) -> dict[str, Any]:
    await ensure_beanie_initialized()

    openai_research_run_id = payload["openai_research_run_id"]
    final_result = payload["final_result"]

    run = await OpenAIResearchRun.get(openai_research_run_id)
    if run is None:
        raise ValueError(f"OpenAIResearchRun not found: {openai_research_run_id}")

    plan = await OpenAIResearchPlan.get(run.openai_research_plan_id)
    if plan is None:
        raise ValueError(f"OpenAIResearchPlan not found: {run.openai_research_plan_id}")

    status = final_result.get("status")
    output = final_result.get("output", [])
    annotations = extract_annotations(final_result)
    citations = extract_citations(annotations)
    final_report_text = extract_output_text(final_result)

    run.openai_final_response = final_result
    run.openai_usage = final_result.get("usage")
    run.openai_incomplete_details = final_result.get("incomplete_details")
    run.output_items = output if isinstance(output, list) else []
    run.annotations = annotations
    run.citations = citations
    run.final_report_text = final_report_text
    run.last_polled_at = utc_now()
    run.updated_at = utc_now()
    set_openai_status(run, status, details={"response_id": run.openai_response_id})

    if status == "completed":
        run.completed_at = utc_now()
        run.failed_at = None
        run.cancelled_at = None
        run.error_message = None
        run.error_payload = None
        set_internal_status(run, "completed", details={"stage": "persist"})
        plan.status = "complete"
    elif status == "cancelled":
        run.cancelled_at = utc_now()
        run.error_message = "OpenAI Deep Research run was cancelled."
        run.error_payload = final_result.get("error")
        set_internal_status(run, "cancelled", details={"stage": "persist"})
        plan.status = "failed"
    elif status == "incomplete":
        run.failed_at = utc_now()
        run.error_message = "OpenAI Deep Research run finished with status 'incomplete'."
        run.error_payload = final_result.get("incomplete_details") or final_result.get("error")
        set_internal_status(run, "incomplete", details={"stage": "persist"})
        plan.status = "failed"
    else:
        run.failed_at = utc_now()
        run.error_message = final_result.get("error", {}).get("message") or "OpenAI Deep Research run failed."
        run.error_payload = final_result.get("error") or serialize_exception(RuntimeError("OpenAI research failed"))
        set_internal_status(run, "failed", details={"stage": "persist"})
        plan.status = "failed"

    await write_result_to_filesystem(run)
    await run.save()

    plan.updated_at = utc_now()
    await plan.save()

    logger.info("Persisted final OpenAI research result for run %s", run.id)

    return {
        "openai_research_run_id": str(run.id),
        "status": run.status,
        "response_id": run.openai_response_id,
        "saved_report": bool(run.final_report_text),
        "output_dir": run.output_dir,
    }


def extract_output_text(final_result: dict[str, Any]) -> str | None:
    output_text = final_result.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    chunks: list[str] = []
    for item in final_result.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text" and content.get("text"):
                chunks.append(content["text"].strip())

    joined = "\n\n".join(chunk for chunk in chunks if chunk).strip()
    return joined or None


def extract_annotations(final_result: dict[str, Any]) -> list[dict[str, Any]]:
    annotations: list[dict[str, Any]] = []

    for item in final_result.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            content_annotations = content.get("annotations", [])
            if isinstance(content_annotations, list):
                annotations.extend(annotation for annotation in content_annotations if isinstance(annotation, dict))

    return dedupe_dicts(annotations)


def extract_citations(annotations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    citations = [
        annotation
        for annotation in annotations
        if isinstance(annotation.get("type"), str) and "citation" in annotation["type"]
    ]
    return dedupe_dicts(citations)


def isoformat_or_none(value: Any) -> str | None:
    return value.isoformat() if value else None


async def write_json(path: Path, payload: Any) -> None:
    async with aiofiles.open(path, "w", encoding="utf-8") as file:
        await file.write(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


async def write_result_to_filesystem(run: OpenAIResearchRun) -> None:
    base_dir = Path(os.getenv("OPENAI_RESEARCH_OUTPUT_DIR", "./data/openai_research_runs"))
    run_dir = (base_dir / str(run.id)).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    artifacts = {
        "request": str(run_dir / "request.json"),
        "initial_response": str(run_dir / "initial_response.json"),
        "final_response": str(run_dir / "final_response.json"),
        "output_items": str(run_dir / "output_items.json"),
        "annotations": str(run_dir / "annotations.json"),
        "citations": str(run_dir / "citations.json"),
        "metadata": str(run_dir / "metadata.json"),
    }
    if run.final_report_text:
        artifacts["report_markdown"] = str(run_dir / "report.md")

    request_snapshot = {
        "run_id": str(run.id),
        "plan_id": str(run.openai_research_plan_id),
        "thread_id": str(run.thread_id),
        "model": run.model,
        "request_input": run.request_input,
        "request_tools": run.request_tools,
        "request_metadata": run.request_metadata,
        "openai_request_payload": run.openai_request_payload,
    }

    metadata = {
        "run_id": str(run.id),
        "plan_id": str(run.openai_research_plan_id),
        "thread_id": str(run.thread_id),
        "openai_response_id": run.openai_response_id,
        "status": run.status,
        "openai_status": run.openai_status,
        "status_history": run.status_history,
        "created_at": isoformat_or_none(run.created_at),
        "submitted_at": isoformat_or_none(run.submitted_at),
        "started_at": isoformat_or_none(run.started_at),
        "last_polled_at": isoformat_or_none(run.last_polled_at),
        "completed_at": isoformat_or_none(run.completed_at),
        "failed_at": isoformat_or_none(run.failed_at),
        "cancelled_at": isoformat_or_none(run.cancelled_at),
        "artifacts": artifacts,
        "usage": run.openai_usage,
        "incomplete_details": run.openai_incomplete_details,
        "error_message": run.error_message,
        "error_payload": run.error_payload,
    }

    await write_json(Path(artifacts["request"]), request_snapshot)
    await write_json(Path(artifacts["initial_response"]), run.openai_initial_response or {})
    await write_json(Path(artifacts["final_response"]), run.openai_final_response or {})
    await write_json(Path(artifacts["output_items"]), run.output_items)
    await write_json(Path(artifacts["annotations"]), run.annotations)
    await write_json(Path(artifacts["citations"]), run.citations)
    await write_json(Path(artifacts["metadata"]), metadata)

    if run.final_report_text:
        async with aiofiles.open(artifacts["report_markdown"], "w", encoding="utf-8") as file:
            await file.write(run.final_report_text)

    run.output_dir = str(run_dir)
    run.filesystem_artifacts = artifacts