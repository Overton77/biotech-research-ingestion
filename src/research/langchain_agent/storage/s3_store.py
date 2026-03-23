"""
S3 artifact persistence for the langchain research agent.

Key structure:
  {mission_id}/{task_slug}/reports/{task_slug}.md          ← final report
  {mission_id}/{task_slug}/runs/{filename}                  ← intermediate files
  {mission_id}/{task_slug}/memory/memory_report.json        ← LangMem report
  {mission_id}/{task_slug}/state/agent_state.json           ← provenance snapshot

For iterative missions, cycle prefix is inserted:
  {mission_id}/{task_slug}/cycle-{N:02d}/reports/{task_slug}.md
  ...

All uploads are fire-and-forget: exceptions are caught + logged so S3 failures
never interrupt the research pipeline. The local filesystem is always the
primary output surface.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.infrastructure.aws.async_s3 import AsyncS3Client
from src.research.langchain_agent.agent.config import ROOT_FILESYSTEM, MissionSliceInput
from src.research.langchain_agent.storage.models import (
    ArtifactRef,
    MemoryReportRecord,
    StageArtifacts,
)

logger = logging.getLogger(__name__)

# Module-level singleton — uses BIOTECH_RESEARCH_RUNS_BUCKET env var
_s3 = AsyncS3Client()


# ---------------------------------------------------------------------------
# Key + metadata builders
# ---------------------------------------------------------------------------


def _key(
    mission_id: str,
    task_slug: str,
    artifact_dir: str,
    filename: str,
    iteration: int | None,
) -> str:
    """
    Build the S3 object key.

    Stage-based:  {mission_id}/{task_slug}/{artifact_dir}/{filename}
    Iterative:    {mission_id}/{task_slug}/cycle-{N:02d}/{artifact_dir}/{filename}
    """
    parts = [mission_id, task_slug]
    if iteration is not None:
        parts.append(f"cycle-{iteration:02d}")
    parts.append(artifact_dir)
    parts.append(filename)
    return "/".join(parts)


def _meta(
    mission_id: str,
    task_slug: str,
    stage_type: str,
    artifact_type: str,
    iteration: int | None,
) -> dict[str, str]:
    """
    S3 object metadata. Keys use hyphens (S3 normalises header names).
    Values are always strings.
    """
    m: dict[str, str] = {
        "mission-id": mission_id,
        "task-slug": task_slug,
        "stage-type": stage_type,
        "artifact-type": artifact_type,
        "uploaded-at": datetime.now(timezone.utc).isoformat(),
    }
    if iteration is not None:
        m["iteration"] = str(iteration)
    return m


# ---------------------------------------------------------------------------
# Per-artifact upload functions
# ---------------------------------------------------------------------------


async def upload_final_report(
    *,
    text: str,
    run_input: MissionSliceInput,
    report_local_path: str,
    iteration: int | None = None,
) -> ArtifactRef | None:
    """Upload the final stage report markdown to S3. Returns ArtifactRef or None on failure."""
    filename = f"{run_input.task_slug}.md"
    key = _key(run_input.mission_id, run_input.task_slug, "reports", filename, iteration)
    meta = _meta(
        run_input.mission_id, run_input.task_slug,
        run_input.stage_type, "final_report", iteration,
    )
    try:
        uri = await _s3.put_text(
            key, text,
            metadata=meta,
            content_type="text/markdown; charset=utf-8",
        )
        logger.info("S3 final report uploaded: %s", uri)
        return ArtifactRef(
            s3_uri=uri,
            s3_key=key,
            local_path=report_local_path,
            artifact_type="final_report",
            filename=filename,
            size_bytes=len(text.encode()),
        )
    except Exception:
        logger.exception("S3 upload failed: final report %s", key)
        return None


async def upload_intermediate_files(
    *,
    written_file_paths: list[str],
    run_input: MissionSliceInput,
    iteration: int | None = None,
    root: Path = ROOT_FILESYSTEM,
) -> list[ArtifactRef]:
    """
    Upload every file written to runs/<task_slug>/ during the agent run.
    Files that no longer exist on disk are silently skipped.
    Returns a list of ArtifactRef for successfully uploaded files.
    """
    refs: list[ArtifactRef] = []
    for rel_path in written_file_paths:
        if not rel_path.startswith("runs/"):
            continue
        local_abs = root / rel_path
        if not local_abs.exists():
            continue
        try:
            text = local_abs.read_text(encoding="utf-8")
        except OSError:
            logger.warning("Could not read intermediate file: %s", local_abs)
            continue

        filename = Path(rel_path).name
        key = _key(run_input.mission_id, run_input.task_slug, "runs", filename, iteration)
        meta = _meta(
            run_input.mission_id, run_input.task_slug,
            run_input.stage_type, "intermediate_file", iteration,
        )
        try:
            uri = await _s3.put_text(
                key, text,
                metadata=meta,
                content_type="text/markdown; charset=utf-8",
            )
            refs.append(ArtifactRef(
                s3_uri=uri,
                s3_key=key,
                local_path=rel_path,
                artifact_type="intermediate_file",
                filename=filename,
                size_bytes=len(text.encode()),
            ))
        except Exception:
            logger.exception("S3 upload failed: intermediate file %s", key)

    logger.info(
        "S3 intermediate files: %d/%d uploaded for %s",
        len(refs), len([p for p in written_file_paths if p.startswith("runs/")]),
        run_input.task_slug,
    )
    return refs


async def upload_memory_report(
    *,
    memory_report: Any,           # ResearchTaskMemoryReport (avoid circular import)
    run_input: MissionSliceInput,
    iteration: int | None = None,
) -> tuple[ArtifactRef | None, MemoryReportRecord]:
    """
    Upload the structured LangMem memory report as JSON to S3.
    Returns (ArtifactRef | None, MemoryReportRecord).
    The MemoryReportRecord is always returned (inline copy for MongoDB embedding).
    """
    record = MemoryReportRecord(
        mission_id=memory_report.mission_id,
        summary=memory_report.summary,
        file_paths=memory_report.file_paths,
    )
    filename = "memory_report.json"
    key = _key(run_input.mission_id, run_input.task_slug, "memory", filename, iteration)
    meta = _meta(
        run_input.mission_id, run_input.task_slug,
        run_input.stage_type, "memory_report", iteration,
    )
    payload = {
        "mission_id": memory_report.mission_id,
        "summary": memory_report.summary,
        "file_paths": memory_report.file_paths,
    }
    try:
        uri = await _s3.put_json(key, payload, metadata=meta)
        artifact = ArtifactRef(
            s3_uri=uri,
            s3_key=key,
            local_path="",
            artifact_type="memory_report",
            filename=filename,
        )
        logger.info("S3 memory report uploaded: %s", uri)
        return artifact, record
    except Exception:
        logger.exception("S3 upload failed: memory report %s", key)
        return None, record


async def upload_agent_state(
    *,
    agent_state: dict[str, Any],
    run_input: MissionSliceInput,
    iteration: int | None = None,
) -> ArtifactRef | None:
    """
    Upload a sanitized provenance snapshot of agent_state as JSON.
    Full LangGraph message history is excluded — only key provenance fields are kept.
    """
    filename = "agent_state.json"
    key = _key(run_input.mission_id, run_input.task_slug, "state", filename, iteration)
    meta = _meta(
        run_input.mission_id, run_input.task_slug,
        run_input.stage_type, "agent_state", iteration,
    )
    snapshot: dict[str, Any] = {
        "task_id": agent_state.get("task_id"),
        "task_slug": agent_state.get("task_slug"),
        "mission_id": agent_state.get("mission_id"),
        "stage_type": agent_state.get("stage_type"),
        "targets": agent_state.get("targets"),
        "step_count": agent_state.get("step_count"),
        "visited_urls": (agent_state.get("visited_urls") or [])[-30:],
        "written_file_paths": agent_state.get("written_file_paths"),
        "read_file_paths": agent_state.get("read_file_paths"),
        "edited_file_paths": agent_state.get("edited_file_paths"),
    }
    try:
        uri = await _s3.put_json(key, snapshot, metadata=meta)
        logger.info("S3 agent state uploaded: %s", uri)
        return ArtifactRef(
            s3_uri=uri,
            s3_key=key,
            local_path="",
            artifact_type="agent_state",
            filename=filename,
        )
    except Exception:
        logger.exception("S3 upload failed: agent state %s", key)
        return None


# ---------------------------------------------------------------------------
# Orchestrator: upload everything for one completed slice
# ---------------------------------------------------------------------------


async def persist_slice_artifacts(
    *,
    run_input: MissionSliceInput,
    final_report_text: str,
    written_file_paths: list[str],
    agent_state: dict[str, Any],
    structured_memory_report: Any,   # ResearchTaskMemoryReport
    report_local_path: str,
    iteration: int | None = None,
    root: Path = ROOT_FILESYSTEM,
) -> tuple[StageArtifacts, MemoryReportRecord]:
    """
    Upload all artifacts for a completed slice and return:
      - StageArtifacts  (S3 refs, embedded in StageRunRecord)
      - MemoryReportRecord  (inline copy, embedded in StageRunRecord)

    All uploads run concurrently. Individual failures are logged and skipped —
    the research pipeline is never interrupted by S3 issues.
    """
    import asyncio

    # Run all uploads concurrently
    (
        final_report_ref,
        intermediate_refs,
        (memory_ref, memory_record),
        state_ref,
    ) = await asyncio.gather(
        upload_final_report(
            text=final_report_text,
            run_input=run_input,
            report_local_path=report_local_path,
            iteration=iteration,
        ),
        upload_intermediate_files(
            written_file_paths=written_file_paths,
            run_input=run_input,
            iteration=iteration,
            root=root,
        ),
        upload_memory_report(
            memory_report=structured_memory_report,
            run_input=run_input,
            iteration=iteration,
        ),
        upload_agent_state(
            agent_state=agent_state,
            run_input=run_input,
            iteration=iteration,
        ),
    )

    artifacts = StageArtifacts(
        final_report=final_report_ref,
        intermediate_files=intermediate_refs,
        memory_report_json=memory_ref,
        agent_state_json=state_ref,
    )
    return artifacts, memory_record
