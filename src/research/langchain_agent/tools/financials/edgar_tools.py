"""LangChain tools for SEC EDGAR filing collection."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

from langchain.tools import tool

from src.research.langchain_agent.agent.config import ROOT_FILESYSTEM
from src.research.langchain_agent.tools.financials.edgar_collection import (
    EdgarCollectionRequest,
    collect_company_filings,
    stage_edgar_dir,
    summarize_edgar_manifest_path,
)
from src.research.langchain_agent.tools.financials.edgartools_power_explore import DEFAULT_FORMS
from src.research.langchain_agent.unstructured.paths import relative_to_root


@tool
def edgar_collect_company_filings(
    ticker: Annotated[str, "Ticker symbol such as VRTX or MRNA."],
    forms: Annotated[list[str], "SEC form types to collect, for example ['10-K', '10-Q']."],
    output_dir: Annotated[
        str | None,
        "Optional absolute or project-relative output directory. Defaults to the staged unstructured edgar folder.",
    ] = None,
    latest_only: Annotated[bool, "If true, keep only the latest filing per requested form."] = True,
    per_form_limit: Annotated[int, "How many filings per form to save when latest_only is false."] = 1,
    include_quarterly: Annotated[bool, "Whether to also export quarterly financial snapshots."] = False,
    include_markdown: Annotated[bool, "Whether to save filing markdown alongside raw text and html."] = True,
    include_full_submission: Annotated[bool, "Whether to save the SEC full submission file."] = False,
    download_mode: Annotated[str, "One of 'none', 'primary', or 'all'."] = "all",
    exhibits_only: Annotated[bool, "Download exhibits only instead of the primary/all attachments mode."] = False,
    identity: Annotated[str | None, "Optional SEC identity string."] = None,
) -> dict[str, Any]:
    """Download Edgar filing artifacts and return stable local manifest paths."""
    request = EdgarCollectionRequest(
        ticker=ticker,
        forms=forms or list(DEFAULT_FORMS),
        output_dir=output_dir,
        latest_only=latest_only,
        per_form_limit=per_form_limit,
        include_quarterly=include_quarterly,
        include_markdown=include_markdown,
        include_full_submission=include_full_submission,
        download_mode=download_mode,
        exhibits_only=exhibits_only,
        identity=identity,
    )
    result = collect_company_filings(request)
    return result.model_dump(mode="json")


@tool
def edgar_read_collection_manifest(
    manifest_path: Annotated[str, "Path to a filings_manifest.json or edgar_collection_result.json file."],
) -> dict[str, Any]:
    """Read a previously downloaded Edgar manifest and return a compact summary."""
    path = Path(manifest_path).expanduser().resolve()
    return summarize_edgar_manifest_path(path)


@tool
def edgar_collect_latest_filing_for_stage(
    ticker: Annotated[str, "Ticker symbol such as VRTX or MRNA."],
    form: Annotated[str, "Single SEC form type such as 10-K or 8-K."],
    task_slug: Annotated[str, "Stage task slug used to derive the mission sandbox folder."],
    identity: Annotated[str | None, "Optional SEC identity string."] = None,
) -> dict[str, Any]:
    """Convenience wrapper that saves one latest filing form into runs/<task_slug>/unstructured/edgar."""
    output_dir = stage_edgar_dir(task_slug)
    request = EdgarCollectionRequest(
        ticker=ticker,
        forms=[form],
        latest_only=True,
        per_form_limit=1,
        output_dir=str(output_dir),
        identity=identity,
    )
    result = collect_company_filings(request)
    return {
        **result.model_dump(mode="json"),
        "relative_output_dir": relative_to_root(Path(result.output_dir), root=ROOT_FILESYSTEM),
        "relative_manifest_path": relative_to_root(Path(result.manifest_path), root=ROOT_FILESYSTEM),
        "relative_preferred_filing_artifact_path": (
            relative_to_root(Path(result.preferred_filing_artifact_path), root=ROOT_FILESYSTEM)
            if result.preferred_filing_artifact_path
            else ""
        ),
    }


EDGAR_RESEARCH_TOOLS = [
    edgar_collect_company_filings,
    edgar_read_collection_manifest,
    edgar_collect_latest_filing_for_stage,
]

__all__ = [
    "EDGAR_RESEARCH_TOOLS",
    "edgar_collect_company_filings",
    "edgar_collect_latest_filing_for_stage",
    "edgar_read_collection_manifest",
]
