# src/temporal/workflows/openai_research.py
from __future__ import annotations

from datetime import timedelta

from temporalio.common import RetryPolicy
from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from src.infrastructure.temporal.activities.openai_research import (
        launch_openai_research_run,
        persist_openai_research_result,
        poll_openai_research_run,
    )


@workflow.defn
class OpenAIResearchWorkflow:
    @workflow.run
    async def run(self, openai_research_run_id: str) -> dict:
        launch_result = await workflow.execute_activity(
            launch_openai_research_run,
            openai_research_run_id,
            start_to_close_timeout=timedelta(minutes=5),
            retry_policy=RetryPolicy(maximum_attempts=1),
        )

        final_result: dict | None = None
        while final_result is None:
            poll_result = await workflow.execute_activity(
                poll_openai_research_run,
                {
                    "openai_research_run_id": openai_research_run_id,
                    "openai_response_id": launch_result["openai_response_id"],
                },
                start_to_close_timeout=timedelta(minutes=5),
                retry_policy=RetryPolicy(
                    initial_interval=timedelta(seconds=5),
                    backoff_coefficient=2.0,
                    maximum_interval=timedelta(minutes=2),
                    maximum_attempts=5,
                ),
            )
            if poll_result["is_terminal"]:
                final_result = poll_result["response"]
                break
            await workflow.sleep(timedelta(seconds=30))

        persisted = await workflow.execute_activity(
            persist_openai_research_result,
            {
                "openai_research_run_id": openai_research_run_id,
                "final_result": final_result,
            },
            start_to_close_timeout=timedelta(minutes=15),
            retry_policy=RetryPolicy(
                initial_interval=timedelta(seconds=5),
                backoff_coefficient=2.0,
                maximum_interval=timedelta(minutes=1),
                maximum_attempts=5,
            ),
        )

        return persisted