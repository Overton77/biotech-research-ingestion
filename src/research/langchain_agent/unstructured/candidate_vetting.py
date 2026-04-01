from __future__ import annotations

import json
from typing import Any

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.research.langchain_agent.unstructured.models import CandidateDocument


class CandidateReviewDecision(BaseModel):
    candidate_id: str
    review_status: str
    priority: str = "low"
    rationale: str = ""


class CandidateReviewResult(BaseModel):
    decisions: list[CandidateReviewDecision] = Field(default_factory=list)


VETTING_SYSTEM_PROMPT = """
You review candidate sources for downstream unstructured ingestion.

Rules:
- Edgar filing artifacts are already handled separately and should not appear in the review set.
- Keep only high-value primary or near-primary source material.
- Prefer reject for generic marketing pages, ordinary homepages, weakly relevant downloads, or noisy artifacts.
- Return review_status as one of: keep_high_priority, keep_if_needed, reject.
- Use web search when needed to judge whether a candidate is worth downstream ingestion.
""".strip()


def _candidate_summary(candidate: CandidateDocument) -> dict[str, Any]:
    return {
        "candidate_id": candidate.candidate_id,
        "source_type": candidate.source_type,
        "title": candidate.title,
        "uri": candidate.uri,
        "local_path": candidate.relative_path or candidate.local_path,
        "issuer_name": candidate.issuer_name,
        "issuer_ticker": candidate.issuer_ticker,
        "form_type": candidate.form_type,
        "accession_number": candidate.accession_number,
        "reasons": candidate.reasons,
        "metadata": candidate.metadata,
    }


async def vet_stage_candidates(
    *,
    mission_id: str,
    task_slug: str,
    user_objective: str,
    targets: list[str],
    candidates: list[CandidateDocument],
) -> list[CandidateDocument]:
    auto_kept: list[CandidateDocument] = []
    reviewable: list[CandidateDocument] = []

    for candidate in candidates:
        if candidate.source_type == "edgar_filing":
            candidate.review_status = "auto_kept"
            candidate.priority = "high"
            candidate.reasons.append("Edgar filing artifacts are automatically retained as high-priority candidates.")
            auto_kept.append(candidate)
            continue
        reviewable.append(candidate)

    if not reviewable:
        return auto_kept

    prompt = {
        "mission_id": mission_id,
        "task_slug": task_slug,
        "user_objective": user_objective,
        "targets": targets,
        "instructions": [
            "Review candidate URLs or downloaded document artifacts for post-research unstructured ingestion.",
            "Keep only candidates that are high-value primary or near-primary source material for the research objective.",
            "Return review_status as one of: keep_high_priority, keep_if_needed, reject.",
        ],
        "candidates": [_candidate_summary(candidate) for candidate in reviewable],
    }

    agent = create_agent(
        model=ChatOpenAI(model="gpt-4o-mini", use_responses_api=True),
        tools=[{"type": "web_search"}],
        system_prompt=VETTING_SYSTEM_PROMPT,
        response_format=CandidateReviewResult,
    )
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": json.dumps(prompt, ensure_ascii=False)}]},
        config={"recursion_limit": 8},
    )
    parsed = result.get("structured_response") or CandidateReviewResult(decisions=[])

    decision_by_id = {decision.candidate_id: decision for decision in parsed.decisions}
    vetted: list[CandidateDocument] = list(auto_kept)
    for candidate in reviewable:
        decision = decision_by_id.get(candidate.candidate_id)
        if decision is None:
            candidate.review_status = "reject"
            candidate.priority = "low"
            candidate.reasons.append("Candidate vetting produced no explicit keep decision.")
            continue
        candidate.review_status = decision.review_status  # type: ignore[assignment]
        candidate.priority = decision.priority  # type: ignore[assignment]
        if decision.rationale:
            candidate.reasons.append(decision.rationale)
        if decision.review_status in {"keep_high_priority", "keep_if_needed"}:
            vetted.append(candidate)
    return vetted
