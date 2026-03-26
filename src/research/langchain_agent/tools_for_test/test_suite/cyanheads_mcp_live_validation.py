from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.research.langchain_agent.tools_for_test.test_suite.cyanheads_mcp_agents import (
    DEFAULT_MODEL_NAME,
    SERVER_SPECS,
    assess_clinicaltrials_detail_payload,
    clinicaltrials_study_url,
    default_retry_interceptors,
    fetch_server_prompt,
    fetch_server_resources,
    inspect_server_capabilities,
    invoke_server_tool,
    live_test_agent,
    stringify_exception,
)
from src.research.langchain_agent.tools_for_test.test_suite.docling_test_tools import (
    convert_local_file_with_docling,
    download_file_to_local,
    shutdown_docling_process_pool,
)

load_dotenv()

TEST_SUITE_DIR = Path(__file__).resolve().parent
GENERATED_DIR = TEST_SUITE_DIR / "generated" / "cyanheads_live_validation"
DEFAULT_NCT_ID = "NCT06566677"

os.environ["NCBI_API_KEY"] = os.getenv("NCBI_API_KEY")
os.environ["NCBI_EMAIL"] = os.getenv("NCBI_EMAIL")

def _ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        if isinstance(value, dict):
            return {key: _json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [_json_safe(item) for item in value]
        return str(value)


def _is_transient_error(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(
        marker in text
        for marker in (
            "429",
            "too many requests",
            "timeout",
            "timed out",
            "503",
            "502",
            "504",
            "brokenresourceerror",
        )
    )


async def _run_with_retry(factory, *, max_attempts: int = 3, base_delay_seconds: float = 1.0):
    last_exc: BaseException | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return await factory()
        except Exception as exc:
            last_exc = exc
            if attempt >= max_attempts or not _is_transient_error(exc):
                raise
            await asyncio.sleep(base_delay_seconds * attempt)
    raise RuntimeError(f"Retry wrapper exhausted without result: {last_exc}")


async def _run_tool_case(
    *,
    server_key: str,
    tool_name: str,
    payload: dict[str, Any],
    interceptors,
) -> dict[str, Any]:
    server_spec = SERVER_SPECS[server_key]
    try:
        result = await invoke_server_tool(
            server_spec,
            tool_name,
            payload,
            tool_interceptors=interceptors,
        )
        return {
            "status": "success",
            "server": server_key,
            "tool_name": tool_name,
            "payload": payload,
            "text_preview": result["text"][:2500],
            "json_payload": _json_safe(result["json_payload"]),
        }
    except Exception as exc:
        return {
            "status": "error",
            "server": server_key,
            "tool_name": tool_name,
            "payload": payload,
            "error": stringify_exception(exc),
        }


async def _run_tool_contract_tests(interceptors) -> dict[str, Any]:
    return {
        "pubmed_search": await _run_tool_case(
            server_key="pubmed",
            tool_name="pubmed_search_articles",
            payload={
                "query": "partial cellular reprogramming mice",
                "maxResults": 3,
                "summaryCount": 2,
                "sort": "relevance",
            },
            interceptors=interceptors,
        ),
        "pubmed_fetch_article": await _run_tool_case(
            server_key="pubmed",
            tool_name="pubmed_fetch_articles",
            payload={"pmids": ["27984723"], "includeMesh": True},
            interceptors=interceptors,
        ),
        "pubmed_invalid_fetch": await _run_tool_case(
            server_key="pubmed",
            tool_name="pubmed_fetch_articles",
            payload={"pmids": ["not_a_pmid"]},
            interceptors=interceptors,
        ),
        "pubchem_search": await _run_tool_case(
            server_key="pubchem",
            tool_name="pubchem_search_compounds",
            payload={
                "searchType": "identifier",
                "identifierType": "name",
                "identifiers": ["rapamycin"],
                "maxResults": 3,
                "properties": ["Title", "MolecularFormula", "MolecularWeight", "CanonicalSMILES"],
            },
            interceptors=interceptors,
        ),
        "pubchem_invalid_details": await _run_tool_case(
            server_key="pubchem",
            tool_name="pubchem_get_compound_details",
            payload={"cids": [0]},
            interceptors=interceptors,
        ),
        "clinicaltrials_search": await _run_tool_case(
            server_key="clinicaltrials",
            tool_name="clinicaltrials_search_studies",
            payload={
                "sponsorQuery": "Pfizer",
                "pageSize": 2,
                "fields": [
                    "protocolSection.identificationModule.nctId",
                    "protocolSection.identificationModule.briefTitle",
                    "protocolSection.sponsorCollaboratorsModule.leadSponsor",
                ],
            },
            interceptors=interceptors,
        ),
        "clinicaltrials_get_study": await _run_tool_case(
            server_key="clinicaltrials",
            tool_name="clinicaltrials_get_study",
            payload={"nctIds": DEFAULT_NCT_ID, "summaryOnly": False},
            interceptors=interceptors,
        ),
        "clinicaltrials_invalid_study": await _run_tool_case(
            server_key="clinicaltrials",
            tool_name="clinicaltrials_get_study",
            payload={"nctIds": "NCT00000000", "summaryOnly": False},
            interceptors=interceptors,
        ),
    }


async def _run_pubmed_prompt_resource_tests(interceptors) -> dict[str, Any]:
    try:
        prompt_payload = await _run_with_retry(
            lambda: fetch_server_prompt(
                SERVER_SPECS["pubmed"],
                "research_plan",
                arguments={
                    "title": "Partial cellular reprogramming in mice",
                    "goal": "Assess translational evidence and intervention modalities",
                    "keywords": "partial cellular reprogramming, yamanaka factors, mice",
                    "organism": "mouse",
                    "includeAgentPrompts": "true",
                },
                tool_interceptors=interceptors,
            )
        )
    except Exception as exc:
        prompt_payload = {
            "message_count": 0,
            "messages": [],
            "error": stringify_exception(exc),
        }

    try:
        resource_payload = await _run_with_retry(
            lambda: fetch_server_resources(
                SERVER_SPECS["pubmed"],
                tool_interceptors=interceptors,
            )
        )
    except Exception as exc:
        resource_payload = {
            "resource_count": 0,
            "resources": [],
            "error": stringify_exception(exc),
        }

    return {
        "prompt": {
            "message_count": prompt_payload["message_count"],
            "messages": prompt_payload["messages"],
            "error": prompt_payload.get("error"),
        },
        "resources": {
            "resource_count": resource_payload["resource_count"],
            "first_resource_preview": resource_payload["resources"][0]["text"][:2500]
            if resource_payload["resources"]
            else None,
            "error": resource_payload.get("error"),
        },
    }


async def _run_clinicaltrials_fallback_validation(interceptors) -> dict[str, Any]:
    server_spec = SERVER_SPECS["clinicaltrials"]
    direct = await _run_tool_case(
        server_key="clinicaltrials",
        tool_name="clinicaltrials_get_study",
        payload={"nctIds": DEFAULT_NCT_ID, "summaryOnly": False},
        interceptors=interceptors,
    )
    canonical_url = clinicaltrials_study_url(DEFAULT_NCT_ID)
    fallback_result: dict[str, Any] = {
        "nct_id": DEFAULT_NCT_ID,
        "canonical_url": canonical_url,
        "direct_assessment": None,
        "download_conversion": None,
    }

    if direct["status"] == "success" and isinstance(direct.get("json_payload"), dict):
        assessment = assess_clinicaltrials_detail_payload(direct["json_payload"])
        fallback_result["direct_assessment"] = assessment
    else:
        fallback_result["direct_assessment"] = {
            "nct_id": DEFAULT_NCT_ID,
            "direct_retrieval_sufficient": False,
            "missing_sections": ["direct_payload_unavailable"],
            "fallback_actions": [
                "Tool call failed or returned non-JSON detail.",
                "Use canonical URL fallback with download plus markdown conversion.",
            ],
        }

    download_dir = _ensure_directory(GENERATED_DIR / "clinicaltrials_downloads")
    conversion_dir = _ensure_directory(GENERATED_DIR / "clinicaltrials_conversions")

    try:
        download = await download_file_to_local.ainvoke(
            {
                "url": canonical_url,
                "output_dir": str(download_dir),
                "filename": f"{DEFAULT_NCT_ID}.html",
                "timeout_seconds": 60.0,
            }
        )
        conversion = await convert_local_file_with_docling.ainvoke(
            {
                "source_path": download["file_path"],
                "output_format": "markdown",
                "output_dir": str(conversion_dir),
                "output_stem": f"{DEFAULT_NCT_ID}_study_page",
                "use_multiprocessing": False,
            }
        )
        fallback_result["download_conversion"] = {
            "status": "success",
            "download": download,
            "conversion": conversion,
        }
    except Exception as exc:
        fallback_result["download_conversion"] = {
            "status": "error",
            "error": stringify_exception(exc),
        }

    return fallback_result


async def _run_agent_validation(interceptors) -> dict[str, Any]:
    if not os.getenv("OPENAI_API_KEY"):
        return {"status": "skipped", "reason": "OPENAI_API_KEY is not available."}

    runs = {}

    async def _safe_agent_run(server_key: str, prompt: str) -> dict[str, Any]:
        try:
            return await live_test_agent(
                SERVER_SPECS[server_key],
                prompt=prompt,
                model_name=DEFAULT_MODEL_NAME,
                tool_interceptors=interceptors,
            )
        except Exception as exc:
            return {"status": "error", "error": stringify_exception(exc)}

    runs["pubmed_agent"] = await _safe_agent_run(
        "pubmed",
        (
            "Search PubMed for partial cellular reprogramming in mice. "
            "Return a structured answer with PMIDs, say whether full text was available, "
            "and list any gaps that would require a fallback."
        ),
    )
    runs["pubchem_agent"] = await _safe_agent_run(
        "pubchem",
        (
            "Resolve rapamycin in PubChem and return the key identifiers, safety signals, "
            "and bioactivity context relevant for downstream trial or company mapping."
        ),
    )
    runs["clinicaltrials_agent"] = await _safe_agent_run(
        "clinicaltrials",
        (
            f"Inspect {DEFAULT_NCT_ID}. Prefer direct clinicaltrials_get_study detail. "
            "If that is not sufficient for downstream extraction, recommend the canonical URL "
            "fallback path and explain why."
        ),
    )
    return {"status": "completed", "runs": _json_safe(runs)}


async def _run_cross_agent_workflows(interceptors) -> dict[str, Any]:
    return {
        "company_to_trial_lookup": await _run_tool_case(
            server_key="clinicaltrials",
            tool_name="clinicaltrials_search_studies",
            payload={"sponsorQuery": "Pfizer", "pageSize": 3},
            interceptors=interceptors,
        ),
        "trial_intervention_to_compound": await _run_tool_case(
            server_key="pubchem",
            tool_name="pubchem_search_compounds",
            payload={
                "searchType": "identifier",
                "identifierType": "name",
                "identifiers": ["sirolimus"],
                "maxResults": 3,
                "properties": ["Title", "CanonicalSMILES", "MolecularWeight"],
            },
            interceptors=interceptors,
        ),
        "compound_to_literature": await _run_tool_case(
            server_key="pubmed",
            tool_name="pubmed_search_articles",
            payload={
                "query": "sirolimus aging mouse",
                "maxResults": 3,
                "summaryCount": 2,
                "sort": "relevance",
            },
            interceptors=interceptors,
        ),
    }


def _build_recommendations(manifest: dict[str, Any]) -> list[str]:
    recommendations: list[str] = []
    pubmed_prompt = (
        manifest.get("pubmed_prompt_resource", {})
        .get("prompt", {})
        .get("message_count", 0)
    )
    if pubmed_prompt >= 2:
        recommendations.append(
            "Use the PubMed `research_plan` prompt resource as optional planning scaffolding before search-heavy literature tasks."
        )

    fallback = manifest.get("clinicaltrials_fallback", {}).get("direct_assessment")
    if fallback and fallback.get("direct_retrieval_sufficient"):
        recommendations.append(
            "Prefer `clinicaltrials_get_study` for full structured retrieval; use page download plus Docling only when critical sections are missing or direct retrieval fails."
        )
    else:
        recommendations.append(
            "Treat ClinicalTrials.gov page download plus Docling conversion as the reliable fallback when the MCP study payload is incomplete or unavailable."
        )

    recommendations.append(
        "Keep retry/backoff interception enabled for live PubChem and ClinicalTrials.gov calls because 429s were observed during capability inspection."
    )
    recommendations.append(
        "For PubMed, abstract and metadata retrieval are strong, but full text must remain conditional because many articles will not be available through PubMed Central."
    )
    return recommendations


async def main() -> None:
    interceptors = default_retry_interceptors()
    _ensure_directory(GENERATED_DIR)

    manifest = {
        "model_name": DEFAULT_MODEL_NAME,
        "server_capabilities": {},
        "tool_contract_tests": {},
        "pubmed_prompt_resource": {},
        "clinicaltrials_fallback": {},
        "cross_agent_workflows": {},
        "agent_validation": {},
        "recommendations": [],
    }

    for server_key, server_spec in SERVER_SPECS.items():
        try:
            manifest["server_capabilities"][server_key] = await _run_with_retry(
                lambda spec=server_spec: inspect_server_capabilities(
                    spec,
                    tool_interceptors=interceptors,
                )
            )
        except Exception as exc:
            manifest["server_capabilities"][server_key] = {
                "server": server_key,
                "error": stringify_exception(exc),
            }

    try:
        manifest["tool_contract_tests"] = await _run_tool_contract_tests(interceptors)
    except Exception as exc:
        manifest["tool_contract_tests"] = {"status": "error", "error": stringify_exception(exc)}

    try:
        manifest["pubmed_prompt_resource"] = await _run_pubmed_prompt_resource_tests(interceptors)
    except Exception as exc:
        manifest["pubmed_prompt_resource"] = {
            "status": "error",
            "error": stringify_exception(exc),
        }

    try:
        manifest["clinicaltrials_fallback"] = await _run_clinicaltrials_fallback_validation(
            interceptors
        )
    except Exception as exc:
        manifest["clinicaltrials_fallback"] = {
            "status": "error",
            "error": stringify_exception(exc),
        }

    try:
        manifest["cross_agent_workflows"] = await _run_cross_agent_workflows(interceptors)
    except Exception as exc:
        manifest["cross_agent_workflows"] = {
            "status": "error",
            "error": stringify_exception(exc),
        }

    try:
        manifest["agent_validation"] = await _run_agent_validation(interceptors)
    except Exception as exc:
        manifest["agent_validation"] = {
            "status": "error",
            "error": stringify_exception(exc),
        }

    manifest["recommendations"] = _build_recommendations(manifest)

    manifest_path = GENERATED_DIR / "cyanheads_live_validation_manifest.json"
    manifest_path.write_text(
        json.dumps(_json_safe(manifest), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Cyanheads MCP live validation completed.")
    print(f"Manifest: {manifest_path}")
    for line in manifest["recommendations"]:
        print(f"- {line}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        shutdown_docling_process_pool()
