from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from src.infrastructure.aws.async_s3 import AsyncS3Client
from src.research.models.mission import (
    ArtifactRef,
    ResearchMission,
    ResearchMissionDraft,
    ResearchRun,
    TaskDef,
    TaskResult,
)

logger = logging.getLogger(__name__)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def stringify_metadata(values: dict[str, Any]) -> dict[str, str]:
    """
    S3 metadata values must be strings.
    """
    out: dict[str, str] = {}
    for key, value in values.items():
        if value is None:
            continue
        out[key] = str(value)
    return out


def normalize_filename(value: str) -> str:
    """
    Reasonably safe filename normalization for S3 object keys.
    """
    value: str = value.strip().replace(" ", "-")
    value: str = re.sub(r"[^A-Za-z0-9._/\-]+", "-", value)
    value: str = re.sub(r"-{2,}", "-", value)
    return value.strip("-") or "artifact"


def model_to_dict(model: BaseModel | dict[str, Any]) -> dict[str, Any]:
    if isinstance(model, BaseModel):
        return model.model_dump(mode="json")
    return model


@dataclass(frozen=True)
class ResearchRunS3Paths:
    mission_id: str

    def mission_prefix(self) -> str:
        return f"missions/{self.mission_id}"

    def mission_dir(self) -> str:
        return f"{self.mission_prefix()}/mission"

    def task_attempt_prefix(self, task_id: str, attempt_number: int) -> str:
        return f"{self.mission_prefix()}/tasks/{task_id}/attempts/{attempt_number}"

    def mission_json_key(self) -> str:
        return f"{self.mission_dir()}/mission.json"

    def mission_draft_json_key(self) -> str:
        return f"{self.mission_dir()}/mission-draft.json"

    def final_report_markdown_key(self) -> str:
        return f"{self.mission_dir()}/final-report.md"

    def final_report_json_key(self) -> str:
        return f"{self.mission_dir()}/final-report.json"

    def summary_json_key(self) -> str:
        return f"{self.mission_dir()}/summary.json"

    def run_json_key(self, task_id: str, attempt_number: int) -> str:
        return f"{self.task_attempt_prefix(task_id, attempt_number)}/run.json"

    def resolved_inputs_key(self, task_id: str, attempt_number: int) -> str:
        return f"{self.task_attempt_prefix(task_id, attempt_number)}/resolved-inputs.json"

    def outputs_key(self, task_id: str, attempt_number: int) -> str:
        return f"{self.task_attempt_prefix(task_id, attempt_number)}/outputs.json"

    def events_key(self, task_id: str, attempt_number: int) -> str:
        return f"{self.task_attempt_prefix(task_id, attempt_number)}/events.json"

    def artifact_key(
        self,
        task_id: str,
        attempt_number: int,
        artifact_type: str,
        artifact_name: str,
    ) -> str:
        safe_type = normalize_filename(artifact_type)
        safe_name = normalize_filename(artifact_name)
        return (
            f"{self.task_attempt_prefix(task_id, attempt_number)}"
            f"/artifacts/{safe_type}/{safe_name}"
        )

    def task_runs_index_key(self) -> str:
        return f"{self.mission_prefix()}/indexes/task-runs.json"

    def artifacts_index_key(self) -> str:
        return f"{self.mission_prefix()}/indexes/artifacts.json"

    def manifest_json_key(self) -> str:
        return f"{self.mission_prefix()}/manifest.json"


class ResearchRunsS3Store:
    """
    Domain-aware S3 persistence layer for:
    - ResearchMissionDraft
    - ResearchMission
    - ResearchRun
    - TaskResult
    - task artifacts
    """

    def __init__(self, s3_client: AsyncS3Client | None = None) -> None:
        self.s3 = s3_client or AsyncS3Client()

    def build_common_metadata(
        self,
        *,
        mission_id: str,
        entity_type: str,
        status: str | None = None,
        task_id: str | None = None,
        attempt_number: int | None = None,
        research_plan_id: str | None = None,
        thread_id: str | None = None,
        stage_label: str | None = None,
        artifact_type: str | None = None,
        content_kind: str | None = None,
    ) -> dict[str, str]:
        return stringify_metadata(
            {
                "app": "biotech-research-ingestion",
                "domain": "research-runs",
                "entity_type": entity_type,
                "mission_id": mission_id,
                "research_plan_id": research_plan_id,
                "thread_id": thread_id,
                "task_id": task_id,
                "attempt_number": attempt_number,
                "stage_label": stage_label,
                "artifact_type": artifact_type,
                "content_kind": content_kind,
                "status": status,
                "source": "deepagents",
                "created_at": utc_now_iso(),
            }
        )

    async def write_mission_draft(
        self,
        *,
        mission_id: str,
        draft: ResearchMissionDraft,
        research_plan_id: str | None = None,
        thread_id: str | None = None,
    ) -> str:
        paths = ResearchRunS3Paths(mission_id=mission_id)
        return await self.s3.put_json(
            paths.mission_draft_json_key(),
            draft.model_dump(mode="json"),
            metadata=self.build_common_metadata(
                mission_id=mission_id,
                entity_type="mission_draft",
                research_plan_id=research_plan_id,
                thread_id=thread_id,
                content_kind="mission_draft",
            ),
        )

    async def write_mission(
        self,
        mission: ResearchMission,
    ) -> str:
        mission_id = str(mission.id)
        research_plan_id = str(mission.research_plan_id)
        thread_id = str(mission.thread_id)

        paths = ResearchRunS3Paths(mission_id=mission_id)
        return await self.s3.put_json(
            paths.mission_json_key(),
            mission.model_dump(mode="json"),
            metadata=self.build_common_metadata(
                mission_id=mission_id,
                entity_type="mission",
                status=mission.status,
                research_plan_id=research_plan_id,
                thread_id=thread_id,
                content_kind="mission",
            ),
        )

    async def write_final_report_markdown(
        self,
        *,
        mission: ResearchMission,
        markdown: str,
    ) -> str:
        mission_id = str(mission.id)
        research_plan_id = str(mission.research_plan_id)
        thread_id = str(mission.thread_id)

        paths = ResearchRunS3Paths(mission_id=mission_id)
        return await self.s3.put_text(
            paths.final_report_markdown_key(),
            markdown,
            metadata=self.build_common_metadata(
                mission_id=mission_id,
                entity_type="final_report",
                status=mission.status,
                research_plan_id=research_plan_id,
                thread_id=thread_id,
                content_kind="final_report_markdown",
            ),
            content_type="text/markdown; charset=utf-8",
        )

    async def write_final_report_json(
        self,
        *,
        mission: ResearchMission,
        report_payload: dict[str, Any],
    ) -> str:
        mission_id = str(mission.id)
        research_plan_id = str(mission.research_plan_id)
        thread_id = str(mission.thread_id)

        paths = ResearchRunS3Paths(mission_id=mission_id)
        return await self.s3.put_json(
            paths.final_report_json_key(),
            report_payload,
            metadata=self.build_common_metadata(
                mission_id=mission_id,
                entity_type="final_report",
                status=mission.status,
                research_plan_id=research_plan_id,
                thread_id=thread_id,
                content_kind="final_report_json",
            ),
        )

    async def write_task_result(
        self,
        *,
        mission: ResearchMission,
        task_result: TaskResult,
        task_def: TaskDef | None = None,
    ) -> dict[str, str]:
        """
        Persist a TaskResult plus its normalized sub-documents.
        """
        mission_id = str(mission.id)
        research_plan_id = str(mission.research_plan_id)
        thread_id = str(mission.thread_id)
        task_id = task_result.task_id
        attempt_number = task_result.attempt_number

        paths = ResearchRunS3Paths(mission_id=mission_id)

        stage_label = task_def.stage_label if task_def else None

        common_metadata = self.build_common_metadata(
            mission_id=mission_id,
            entity_type="task_result",
            status=task_result.status,
            task_id=task_id,
            attempt_number=attempt_number,
            research_plan_id=research_plan_id,
            thread_id=thread_id,
            stage_label=stage_label,
        )

        run_uri = await self.s3.put_json(
            paths.run_json_key(task_id, attempt_number),
            task_result.model_dump(mode="json"),
            metadata={
                **common_metadata,
                "content_kind": "task_result",
            },
        )

        outputs_uri = await self.s3.put_json(
            paths.outputs_key(task_id, attempt_number),
            task_result.outputs,
            metadata={
                **common_metadata,
                "content_kind": "task_outputs",
            },
        )

        events_uri = await self.s3.put_json(
            paths.events_key(task_id, attempt_number),
            {"events": [event.model_dump(mode="json") for event in task_result.events]},
            metadata={
                **common_metadata,
                "content_kind": "task_events",
            },
        )

        return {
            "run_uri": run_uri,
            "outputs_uri": outputs_uri,
            "events_uri": events_uri,
        }

    async def write_research_run(
        self,
        *,
        mission: ResearchMission,
        research_run: ResearchRun,
        task_def: TaskDef | None = None,
    ) -> dict[str, str]:
        """
        Persist a ResearchRun and its normalized snapshots.
        """
        mission_id = str(research_run.mission_id)
        research_plan_id = str(mission.research_plan_id)
        thread_id = str(mission.thread_id)
        task_id = research_run.task_id
        attempt_number = research_run.attempt_number

        paths = ResearchRunS3Paths(mission_id=mission_id)
        stage_label = task_def.stage_label if task_def else None

        common_metadata = self.build_common_metadata(
            mission_id=mission_id,
            entity_type="research_run",
            status=research_run.status,
            task_id=task_id,
            attempt_number=attempt_number,
            research_plan_id=research_plan_id,
            thread_id=thread_id,
            stage_label=stage_label,
        )

        run_uri = await self.s3.put_json(
            paths.run_json_key(task_id, attempt_number),
            research_run.model_dump(mode="json"),
            metadata={
                **common_metadata,
                "content_kind": "research_run",
            },
        )

        resolved_inputs_uri = await self.s3.put_json(
            paths.resolved_inputs_key(task_id, attempt_number),
            research_run.resolved_inputs_snapshot,
            metadata={
                **common_metadata,
                "content_kind": "resolved_inputs_snapshot",
            },
        )

        outputs_uri = await self.s3.put_json(
            paths.outputs_key(task_id, attempt_number),
            research_run.outputs_snapshot,
            metadata={
                **common_metadata,
                "content_kind": "outputs_snapshot",
            },
        )

        return {
            "run_uri": run_uri,
            "resolved_inputs_uri": resolved_inputs_uri,
            "outputs_uri": outputs_uri,
        }

    async def upload_task_artifact_text(
        self,
        *,
        mission: ResearchMission,
        task_id: str,
        attempt_number: int,
        artifact_name: str,
        artifact_type: str,
        text: str,
        content_type: str = "text/plain; charset=utf-8",
        stage_label: str | None = None,
    ) -> ArtifactRef:
        mission_id = str(mission.id)
        research_plan_id = str(mission.research_plan_id)
        thread_id = str(mission.thread_id)

        paths = ResearchRunS3Paths(mission_id=mission_id)
        key = paths.artifact_key(
            task_id=task_id,
            attempt_number=attempt_number,
            artifact_type=artifact_type,
            artifact_name=artifact_name,
        )

        s3_uri = await self.s3.put_text(
            key,
            text,
            metadata=self.build_common_metadata(
                mission_id=mission_id,
                entity_type="artifact",
                task_id=task_id,
                attempt_number=attempt_number,
                research_plan_id=research_plan_id,
                thread_id=thread_id,
                stage_label=stage_label,
                artifact_type=artifact_type,
                content_kind="artifact",
            ),
            content_type=content_type,
        )

        return ArtifactRef(
            task_id=task_id,
            name=artifact_name,
            artifact_type=artifact_type,
            storage="filesystem",
            path=s3_uri,
            content_type=content_type,
        )

    async def upload_task_artifact_json(
        self,
        *,
        mission: ResearchMission,
        task_id: str,
        attempt_number: int,
        artifact_name: str,
        artifact_type: str,
        payload: dict[str, Any],
        stage_label: str | None = None,
    ) -> ArtifactRef:
        mission_id = str(mission.id)
        research_plan_id = str(mission.research_plan_id)
        thread_id = str(mission.thread_id)

        paths = ResearchRunS3Paths(mission_id=mission_id)
        key = paths.artifact_key(
            task_id=task_id,
            attempt_number=attempt_number,
            artifact_type=artifact_type,
            artifact_name=artifact_name,
        )

        s3_uri = await self.s3.put_json(
            key,
            payload,
            metadata=self.build_common_metadata(
                mission_id=mission_id,
                entity_type="artifact",
                task_id=task_id,
                attempt_number=attempt_number,
                research_plan_id=research_plan_id,
                thread_id=thread_id,
                stage_label=stage_label,
                artifact_type=artifact_type,
                content_kind="artifact",
            ),
        )

        return ArtifactRef(
            task_id=task_id,
            name=artifact_name,
            artifact_type=artifact_type,
            storage="filesystem",
            path=s3_uri,
            content_type="application/json",
        )

    async def build_task_runs_index(
        self,
        *,
        mission: ResearchMission,
        research_runs: list[ResearchRun],
    ) -> str:
        mission_id = str(mission.id)
        research_plan_id = str(mission.research_plan_id)
        thread_id = str(mission.thread_id)

        payload = {
            "mission_id": mission_id,
            "research_plan_id": research_plan_id,
            "thread_id": thread_id,
            "task_runs": [
                {
                    "task_id": run.task_id,
                    "attempt_number": run.attempt_number,
                    "status": run.status,
                    "started_at": run.started_at,
                    "completed_at": run.completed_at,
                    "created_at": run.created_at,
                }
                for run in research_runs
            ],
            "generated_at": utc_now_iso(),
        }

        paths = ResearchRunS3Paths(mission_id=mission_id)
        return await self.s3.put_json(
            paths.task_runs_index_key(),
            payload,
            metadata=self.build_common_metadata(
                mission_id=mission_id,
                entity_type="index",
                status=mission.status,
                research_plan_id=research_plan_id,
                thread_id=thread_id,
                content_kind="task_runs_index",
            ),
        )

    async def write_manifest(
        self,
        mission: ResearchMission,
        manifest: dict[str, Any],
    ) -> str:
        """Upload manifest.json to S3. Returns the S3 key."""
        mission_id = str(mission.id)
        paths = ResearchRunS3Paths(mission_id=mission_id)
        return await self.s3.put_json(
            paths.manifest_json_key(),
            manifest,
            metadata=self.build_common_metadata(
                mission_id=mission_id,
                entity_type="manifest",
                status=mission.status,
                research_plan_id=str(mission.research_plan_id),
                thread_id=str(mission.thread_id),
                content_kind="manifest",
            ),
        )

    async def get_manifest(self, mission: ResearchMission) -> dict[str, Any]:
        """Retrieve manifest.json from S3."""
        mission_id = str(mission.id)
        paths = ResearchRunS3Paths(mission_id=mission_id)
        return await self.s3.get_json(paths.manifest_json_key())

    async def list_mission_objects(
        self,
        mission_id: str,
    ) -> list[dict[str, Any]]:
        paths = ResearchRunS3Paths(mission_id=mission_id)
        return await self.s3.list_objects(paths.mission_prefix() + "/")

    async def upload_local_artifacts_to_s3(
        self,
        *,
        mission: ResearchMission,
        artifacts: list[ArtifactRef],
        task_def: TaskDef | None = None,
    ) -> list[ArtifactRef]:
        """Upload filesystem artifacts to S3, returning updated ArtifactRefs with S3 paths."""
        updated: list[ArtifactRef] = []
        stage_label = task_def.stage_label if task_def else None

        for art in artifacts:
            if not art.path or art.path.startswith("s3://") or art.storage == "s3":
                updated.append(art)
                continue

            local_path = Path(art.path)
            if not local_path.exists() or not local_path.is_file():
                logger.warning("Artifact file not found, skipping S3 upload: %s", art.path)
                updated.append(art)
                continue

            try:
                content = local_path.read_bytes()
                is_json = art.content_type == "application/json"

                if is_json:
                    import json
                    payload = json.loads(content.decode("utf-8"))
                    new_ref = await self.upload_task_artifact_json(
                        mission=mission,
                        task_id=art.task_id,
                        attempt_number=1,
                        artifact_name=art.name,
                        artifact_type=art.artifact_type,
                        payload=payload,
                        stage_label=stage_label,
                    )
                else:
                    new_ref = await self.upload_task_artifact_text(
                        mission=mission,
                        task_id=art.task_id,
                        attempt_number=1,
                        artifact_name=art.name,
                        artifact_type=art.artifact_type,
                        text=content.decode("utf-8", errors="replace"),
                        content_type=art.content_type,
                        stage_label=stage_label,
                    )

                new_ref.storage = "s3"
                updated.append(new_ref)
            except Exception:
                logger.exception("Failed to upload artifact '%s' to S3", art.name)
                updated.append(art)

        return updated


_s3_store: ResearchRunsS3Store | None = None


def get_research_runs_s3_store() -> ResearchRunsS3Store:
    """Return a module-level singleton ResearchRunsS3Store."""
    global _s3_store
    if _s3_store is None:
        _s3_store = ResearchRunsS3Store()
    return _s3_store