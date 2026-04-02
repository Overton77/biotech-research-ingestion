"""
Run a full ResearchMission with dependency ordering and report injection.
Stages with dependencies receive the final_report of each depended-on stage.

Stages may be iterative: when ``MissionStage.iterative_config`` is set, the
runner delegates to ``run_iterative_stage()`` which executes bounded passes
in a loop.  Non-iterative stages run exactly as before via
``run_single_mission_slice()``.

Post-run: each completed stage's StageRunRecord is appended to a
MissionRunDocument (MongoDB / Beanie) so the full run is persisted.
MongoDB persistence failures are logged and never interrupt execution.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.research.langchain_agent.models.mission import MissionStage, ResearchMission

from src.infrastructure.neo4j.neo4j_client import Neo4jAuraClient, Neo4jAuraSettings 
import json 

from src.research.langchain_agent.unstructured.paths import mission_unstructured_dir
from src.research.langchain_agent.unstructured.models import CandidateDocument
from src.research.langchain_agent.unstructured.run_unstructured_ingestion import (
    run_unstructured_ingestion,
) 
from src.research.langchain_agent.workflow.utils.stage_topology import _topological_stage_order

logger = logging.getLogger(__name__) 

async def _run_staged_unstructured_ingestion(
    *,
    mission: ResearchMission,
    mission_candidate_manifest_path: str,
    root: Path,
) -> dict[str, Any]:
    manifest_abs = root / mission_candidate_manifest_path
    payload = json.loads(manifest_abs.read_text(encoding="utf-8"))
    final_candidates = payload.get("candidates", [])
    if not final_candidates:
        return {"status": "skipped", "reason": "no_candidates"}

    output_dir = mission_unstructured_dir(mission.mission_id, root=root) / "executions"
    output_dir.mkdir(parents=True, exist_ok=True)

    settings = Neo4jAuraSettings.from_env()
    results: list[dict[str, Any]] = []
    async with Neo4jAuraClient(settings) as client:
        for candidate_payload in final_candidates:
            if not candidate_payload.get("local_path"):
                results.append(
                    {
                        "candidate_id": candidate_payload.get("candidate_id", ""),
                        "status": "skipped",
                        "reason": "candidate_missing_local_path",
                    }
                )
                continue

            candidate_dir = output_dir / candidate_payload["candidate_id"]
            try:
                candidate = CandidateDocument.model_validate(candidate_payload)
                ingestion_result = await run_unstructured_ingestion(
                    candidate=candidate,
                    output_dir=candidate_dir,
                    neo4j_client=client,
                    config=mission.unstructured_ingestion,
                )
                results.append(
                    {
                        "candidate_id": candidate.candidate_id,
                        "status": "completed",
                        "document_id": ingestion_result.document.document_id,
                        "chunk_count": len(ingestion_result.chunks),
                        "relationship_count": len(ingestion_result.relationship_decisions),
                    }
                )
            except Exception as exc:
                logger.exception(
                    "Unstructured ingestion failed for candidate %s",
                    candidate_payload.get("candidate_id", ""),
                )
                results.append(
                    {
                        "candidate_id": candidate_payload.get("candidate_id", ""),
                        "status": "failed",
                        "error": str(exc),
                    }
                )

    return {"status": "completed", "results": results}

