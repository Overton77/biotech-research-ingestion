from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.research.langchain_agent.tools_for_test.test_suite.cyanheads_mcp_agents import (
    DEFAULT_MODEL_NAME,
    SERVER_SPECS,
    assess_clinicaltrials_detail_payload,
    clinicaltrials_study_url,
    default_retry_interceptors,
    extract_tool_text,
    invoke_server_tool,
    parse_json_text,
    stringify_exception,
)

TEST_SUITE_DIR = Path(__file__).resolve().parent
GENERATED_DIR = TEST_SUITE_DIR / "generated" / "isolated_runs"


class ClinicalTrialRecord(BaseModel):
    nct_id: str
    brief_title: str | None = None
    official_title: str | None = None
    overall_status: str | None = None
    study_type: str | None = None
    lead_sponsor: str | None = None
    collaborators: list[str] = Field(default_factory=list)
    conditions: list[str] = Field(default_factory=list)
    interventions: list[str] = Field(default_factory=list)
    canonical_url: str
    direct_retrieval_sufficient: bool
    missing_sections: list[str] = Field(default_factory=list)
    brief_summary: str | None = None


class ClinicalTrialsScenarioOutput(BaseModel):
    server: str = "clinicaltrials"
    company: str
    search_payload: dict[str, Any]
    search_result_preview: str
    matched_nct_ids: list[str]
    trials: list[ClinicalTrialRecord] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class PubMedStudySummary(BaseModel):
    title: str
    study_type: str | None = None
    population: str | None = None
    intervention: str | None = None
    comparator: str | None = None
    key_findings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    practical_takeaway: str


class PubMedScenarioOutput(BaseModel):
    server: str = "pubmed"
    compound_or_topic: str
    query: str
    candidate_pmids: list[str]
    selected_pmid: str | None = None
    selected_title: str | None = None
    selected_article_type: str | None = None
    pubmed_url: str | None = None
    pmc_url: str | None = None
    article_metadata_preview: str | None = None
    fulltext_preview: str | None = None
    fulltext_summary: PubMedStudySummary | None = None
    errors: list[str] = Field(default_factory=list)


class PubChemCompoundRecord(BaseModel):
    query_name: str
    cid: int | None = None
    title: str | None = None
    molecular_formula: str | None = None
    molecular_weight: str | None = None
    canonical_smiles: str | None = None
    details_preview: str | None = None
    safety_preview: str | None = None
    errors: list[str] = Field(default_factory=list)


class PubChemScenarioOutput(BaseModel):
    server: str = "pubchem"
    compounds: list[str]
    records: list[PubChemCompoundRecord]


def _ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return cleaned or "scenario"


def _extract_nct_ids(text: str) -> list[str]:
    return list(dict.fromkeys(re.findall(r"\bNCT\d{8}\b", text)))


def _extract_pmids(text: str) -> list[str]:
    collected: list[str] = []
    summary_line = re.search(r"\*\*PMIDs:\*\*\s+([0-9,\s]+)", text)
    if summary_line:
        collected.extend(re.findall(r"\d{6,9}", summary_line.group(1)))
    collected.extend(re.findall(r"\bPMID\b[\s:*`_]*?(\d{6,9})\b", text))
    return list(dict.fromkeys(collected))


def _extract_first_match(pattern: str, text: str) -> str | None:
    match = re.search(pattern, text, flags=re.MULTILINE)
    return match.group(1).strip() if match else None


def _looks_like_primary_trial(article_text: str) -> bool:
    article_type = (_extract_first_match(r"^\*\*Type:\*\*\s+(.+)$", article_text) or "").lower()
    title = (_extract_first_match(r"^###\s+(.+)$", article_text) or "").lower()
    if "randomized controlled trial" in article_type:
        return True
    disqualifiers = ("meta-analysis", "systematic review", "scoping review", "review")
    return not any(flag in article_type or flag in title for flag in disqualifiers)


def _extract_intervention_names(study_payload: dict[str, Any]) -> list[str]:
    protocol = study_payload.get("protocolSection") or {}
    arms_module = protocol.get("armsInterventionsModule") or {}
    interventions = arms_module.get("interventions") or []
    names = []
    for item in interventions:
        if isinstance(item, dict) and item.get("name"):
            names.append(str(item["name"]))
    return names


async def _summarize_pubmed_fulltext(
    *,
    query: str,
    article_text: str,
    model_name: str,
) -> PubMedStudySummary:
    agent = create_agent(
        model=ChatOpenAI(model=model_name),
        tools=[],
        system_prompt=(
            "You are summarizing biomedical full text for downstream research coordination. "
            "Return a structured summary focused on study design, intervention, comparator, key findings, "
            "limitations, and practical relevance."
        ),
        response_format=PubMedStudySummary,
    )
    result = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"Research query: {query}\n\n"
                        "Summarize this full-text article in a concise structured form.\n\n"
                        f"{article_text[:25000]}"
                    ),
                }
            ]
        },
        config={"configurable": {"thread_id": f"pubmed-summary-{_slugify(query)}"}},
    )
    return result["structured_response"]


async def run_clinicaltrials_scenario(
    *,
    company: str,
    max_results: int,
    detail_limit: int,
) -> ClinicalTrialsScenarioOutput:
    interceptors = default_retry_interceptors()
    search_payload = {"sponsorQuery": company, "pageSize": max_results}
    output = ClinicalTrialsScenarioOutput(
        company=company,
        search_payload=search_payload,
        search_result_preview="",
        matched_nct_ids=[],
        trials=[],
        errors=[],
    )

    try:
        search_result = await invoke_server_tool(
            SERVER_SPECS["clinicaltrials"],
            "clinicaltrials_search_studies",
            search_payload,
            tool_interceptors=interceptors,
        )
        search_text = search_result["text"]
        output.search_result_preview = search_text[:3000]
        output.matched_nct_ids = _extract_nct_ids(search_text)
    except Exception as exc:
        output.errors.append(f"search_failed: {stringify_exception(exc)}")
        return output

    for nct_id in output.matched_nct_ids[:detail_limit]:
        try:
            detail_result = await invoke_server_tool(
                SERVER_SPECS["clinicaltrials"],
                "clinicaltrials_get_study",
                {"nctIds": nct_id, "summaryOnly": False},
                tool_interceptors=interceptors,
            )
            detail_payload = parse_json_text(detail_result["text"]) or {}
            assessment = assess_clinicaltrials_detail_payload(detail_payload)
            protocol = detail_payload.get("protocolSection") or {}
            id_module = protocol.get("identificationModule") or {}
            status_module = protocol.get("statusModule") or {}
            design_module = protocol.get("designModule") or {}
            sponsor_module = protocol.get("sponsorCollaboratorsModule") or {}
            conditions_module = protocol.get("conditionsModule") or {}
            description_module = protocol.get("descriptionModule") or {}
            collaborators = [
                str(item.get("name"))
                for item in sponsor_module.get("collaborators", [])
                if isinstance(item, dict) and item.get("name")
            ]
            output.trials.append(
                ClinicalTrialRecord(
                    nct_id=nct_id,
                    brief_title=id_module.get("briefTitle"),
                    official_title=id_module.get("officialTitle"),
                    overall_status=status_module.get("overallStatus"),
                    study_type=design_module.get("studyType"),
                    lead_sponsor=(sponsor_module.get("leadSponsor") or {}).get("name"),
                    collaborators=collaborators,
                    conditions=list(conditions_module.get("conditions") or []),
                    interventions=_extract_intervention_names(detail_payload),
                    canonical_url=clinicaltrials_study_url(nct_id),
                    direct_retrieval_sufficient=assessment["direct_retrieval_sufficient"],
                    missing_sections=assessment["missing_sections"],
                    brief_summary=description_module.get("briefSummary"),
                )
            )
        except Exception as exc:
            output.errors.append(f"{nct_id}: {stringify_exception(exc)}")

    return output


async def run_pubmed_scenario(
    *,
    query: str,
    topic_label: str,
    max_results: int,
    model_name: str,
) -> PubMedScenarioOutput:
    interceptors = default_retry_interceptors()
    output = PubMedScenarioOutput(
        compound_or_topic=topic_label,
        query=query,
        candidate_pmids=[],
        errors=[],
    )

    try:
        search_result = await invoke_server_tool(
            SERVER_SPECS["pubmed"],
            "pubmed_search_articles",
            {
                "query": query,
                "maxResults": max_results,
                "summaryCount": min(max_results, 3),
                "sort": "relevance",
                "freeFullText": True,
            },
            tool_interceptors=interceptors,
        )
        output.candidate_pmids = _extract_pmids(search_result["text"])
    except Exception as exc:
        output.errors.append(f"search_failed: {stringify_exception(exc)}")
        return output

    selected_pmid: str | None = None
    selected_article_text: str | None = None
    fallback_pmid: str | None = None
    fallback_article_text: str | None = None
    for pmid in output.candidate_pmids:
        try:
            article_result = await invoke_server_tool(
                SERVER_SPECS["pubmed"],
                "pubmed_fetch_articles",
                {"pmids": [pmid], "includeMesh": True},
                tool_interceptors=interceptors,
            )
            article_text = article_result["text"]
            if "**PMC:**" in article_text or "\n**PMC:**" in article_text:
                if _looks_like_primary_trial(article_text):
                    selected_pmid = pmid
                    selected_article_text = article_text
                    break
                if fallback_pmid is None:
                    fallback_pmid = pmid
                    fallback_article_text = article_text
        except Exception as exc:
            output.errors.append(f"{pmid}_metadata_failed: {stringify_exception(exc)}")

    if not selected_pmid and fallback_pmid and fallback_article_text:
        selected_pmid = fallback_pmid
        selected_article_text = fallback_article_text

    if not selected_pmid or not selected_article_text:
        output.errors.append("no_open_access_candidate_found")
        return output

    output.selected_pmid = selected_pmid
    output.article_metadata_preview = selected_article_text[:4000]
    output.selected_title = _extract_first_match(r"^###\s+(.+)$", selected_article_text)
    output.selected_article_type = _extract_first_match(r"^\*\*Type:\*\*\s+(.+)$", selected_article_text)
    output.pubmed_url = _extract_first_match(r"^\*\*PubMed:\*\*\s+(.+)$", selected_article_text)
    output.pmc_url = _extract_first_match(r"^\*\*PMC:\*\*\s+(.+)$", selected_article_text)

    try:
        fulltext_result = await invoke_server_tool(
            SERVER_SPECS["pubmed"],
            "pubmed_fetch_fulltext",
            {
                "pmids": [selected_pmid],
                "maxSections": 8,
                "includeReferences": False,
            },
            tool_interceptors=interceptors,
        )
        fulltext_text = extract_tool_text(fulltext_result["raw_result"])
        output.fulltext_preview = fulltext_text[:6000]
        output.fulltext_summary = await _summarize_pubmed_fulltext(
            query=query,
            article_text=fulltext_text,
            model_name=model_name,
        )
    except Exception as exc:
        output.errors.append(f"{selected_pmid}_fulltext_failed: {stringify_exception(exc)}")

    return output


async def run_pubchem_scenario(
    *,
    compounds: list[str],
) -> PubChemScenarioOutput:
    interceptors = default_retry_interceptors()
    records: list[PubChemCompoundRecord] = []

    for compound in compounds:
        record = PubChemCompoundRecord(query_name=compound)
        try:
            search_result = await invoke_server_tool(
                SERVER_SPECS["pubchem"],
                "pubchem_search_compounds",
                {
                    "searchType": "identifier",
                    "identifierType": "name",
                    "identifiers": [compound],
                    "maxResults": 3,
                    "properties": ["Title", "MolecularFormula", "MolecularWeight", "CanonicalSMILES"],
                },
                tool_interceptors=interceptors,
            )
            search_text = search_result["text"]
            cid_match = re.search(r"\*\*CID\s+(\d+)\s+[-—]", search_text)
            record.cid = int(cid_match.group(1)) if cid_match else None
            record.title = _extract_first_match(r"^\s*Title:\s+(.+)$", search_text)
            record.molecular_formula = _extract_first_match(r"^\s*MolecularFormula:\s+(.+)$", search_text)
            record.molecular_weight = _extract_first_match(r"^\s*MolecularWeight:\s+(.+)$", search_text)
            record.canonical_smiles = _extract_first_match(r"^\s*CanonicalSMILES:\s+(.+)$", search_text)

            if record.cid:
                detail_result = await invoke_server_tool(
                    SERVER_SPECS["pubchem"],
                    "pubchem_get_compound_details",
                    {
                        "cids": [record.cid],
                        "includeDrugLikeness": True,
                        "includeDescription": False,
                        "includeClassification": False,
                    },
                    tool_interceptors=interceptors,
                )
                record.details_preview = detail_result["text"][:3000]

                safety_result = await invoke_server_tool(
                    SERVER_SPECS["pubchem"],
                    "pubchem_get_compound_safety",
                    {"cid": record.cid},
                    tool_interceptors=interceptors,
                )
                record.safety_preview = safety_result["text"][:2000]
        except Exception as exc:
            record.errors.append(stringify_exception(exc))

        records.append(record)

    return PubChemScenarioOutput(compounds=compounds, records=records)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run isolated Cyanheads MCP server scenarios and save structured JSON output."
    )
    parser.add_argument("--server", choices=("clinicaltrials", "pubmed", "pubchem"), required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--max-results", type=int, default=5)
    parser.add_argument("--detail-limit", type=int, default=2)
    parser.add_argument("--company", default="Elysium Health")
    parser.add_argument(
        "--query",
        default="creatine supplementation randomized controlled trial older adults",
    )
    parser.add_argument("--topic-label", default="creatine")
    parser.add_argument(
        "--compounds",
        nargs="+",
        default=["creatine", "rapamycin", "metformin"],
    )
    parser.add_argument("--output-path", default=None)
    return parser


async def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.server == "clinicaltrials":
        result = await run_clinicaltrials_scenario(
            company=args.company,
            max_results=args.max_results,
            detail_limit=args.detail_limit,
        )
        default_name = f"clinicaltrials_{_slugify(args.company)}.json"
    elif args.server == "pubmed":
        result = await run_pubmed_scenario(
            query=args.query,
            topic_label=args.topic_label,
            max_results=args.max_results,
            model_name=args.model,
        )
        default_name = f"pubmed_{_slugify(args.topic_label)}.json"
    else:
        result = await run_pubchem_scenario(compounds=args.compounds)
        default_name = f"pubchem_{_slugify('_'.join(args.compounds[:3]))}.json"

    output_path = Path(args.output_path) if args.output_path else _ensure_directory(GENERATED_DIR) / default_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result.model_dump(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(output_path)


if __name__ == "__main__":
    asyncio.run(main())
