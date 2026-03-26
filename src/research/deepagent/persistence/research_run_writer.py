"""Idempotent Mongo persistence for task execution results.

LangGraph may replay nodes on resume — the upsert must be safe to call twice
with identical data for the same (mission_id, task_id, attempt_number).
"""

from __future__ import annotations

import logging
from typing import Any

from beanie.odm.fields import PydanticObjectId

from src.research.deepagent.models.mission import ResearchRun, TaskResult

logger = logging.getLogger(__name__)


class ResearchRunWriter:
    """Writes TaskResult snapshots to Mongo as ResearchRun documents."""

    async def upsert_run(
        self,
        mission_id: str,
        task_result: TaskResult,
        resolved_inputs: dict[str, Any],
    ) -> ResearchRun:
        """
        Insert or update a ResearchRun document for (mission_id, task_id, attempt_number).
        Idempotent: safe to call multiple times with the same data.
        """
        mission_oid: PydanticObjectId = PydanticObjectId(mission_id)

        existing: ResearchRun | None = await ResearchRun.find_one({
            "mission_id": mission_oid,
            "task_id": task_result.task_id,
            "attempt_number": task_result.attempt_number,
        })

        if existing:
            existing.status = task_result.status
            existing.outputs_snapshot = task_result.outputs
            existing.artifacts = task_result.artifacts
            existing.error_message = task_result.error_message
            existing.completed_at = task_result.completed_at
            await existing.save()
            logger.info(
                "Updated ResearchRun for mission=%s task=%s attempt=%d",
                mission_id, task_result.task_id, task_result.attempt_number,
            )
            return existing

        doc: ResearchRun = ResearchRun(
            mission_id=mission_oid,
            task_id=task_result.task_id,
            attempt_number=task_result.attempt_number,
            status=task_result.status,
            resolved_inputs_snapshot=resolved_inputs,
            outputs_snapshot=task_result.outputs,
            artifacts=task_result.artifacts,
            error_message=task_result.error_message,
            started_at=task_result.started_at,
            completed_at=task_result.completed_at,
        )
        await doc.insert()
        logger.info(
            "Inserted ResearchRun for mission=%s task=%s attempt=%d",
            mission_id, task_result.task_id, task_result.attempt_number,
        )
        return doc
