"""Temporal workflow — durable wrapper around the deep-research mission activity."""

from __future__ import annotations

from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from src.infrastructure.temporal.activities.deep_research import (
        run_deep_research_mission,
    )


@workflow.defn
class DeepResearchMissionWorkflow:
    """One workflow instance per ResearchMission.

    Currently delegates to a single long-running activity.  In the future
    each task could become its own activity for finer-grained retries.
    """

    @workflow.run
    async def run(self, mission_id: str) -> dict:
        result = await workflow.execute_activity(
            run_deep_research_mission,
            mission_id,
            start_to_close_timeout=timedelta(hours=6),
            heartbeat_timeout=timedelta(minutes=30),
            retry_policy=RetryPolicy(
                initial_interval=timedelta(seconds=10),
                backoff_coefficient=2.0,
                maximum_interval=timedelta(minutes=5),
                maximum_attempts=1,
            ),
        )
        return result
