from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

from langchain.tools import tool
from pydantic import BaseModel, Field

from src.research.langchain_agent.agent.config import ROOT_FILESYSTEM
from src.research.langchain_agent.tools_for_test.financials.edgartools_power_explore import (
    DEFAULT_FORMS,
    configure_identity,
    explore_company,
)
from src.research.langchain_agent.unstructured.paths import relative_to_root, stage_unstructured_dir


class EdgarCollectionRequest(BaseModel):
    ticker: str
    forms: list[str] = Field(default_factory=lambda: list(DEFAULT_FORMS))
    latest_only: bool = True
    per_form_limit: int = 1
    include_quarterly: bool = False
    include_markdown: bool = True
    include_full_submission: bool = False
    download_mode: str = "all"
    exhibits_only: bool = False
    identity: str | None = None
    output_dir: str | None = None


class EdgarCollectionResult(BaseModel):
    ticker: str
    company_name: str = ""
    cik: str = ""
    output_dir: str
    manifest_path: str
    preferred_filing_artifact_path: str = ""
    filing_count: int = 0
    filings: list[dict[str, Any]] = Field(default_factory=list)


def _stage_edgar_dir(task_slug: str | None = None) -> Path:
    if task_slug:
        return stage_unstructured_dir(task_slug, root=ROOT_FILESYSTEM) / "edgar"
    return ROOT_FILESYSTEM / "runs" / "edgar_downloads"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _preferred_filing_artifact_path(ticker_dir: Path) -> str:
    filings_manifest = ticker_dir / "filings_manifest.json"
    if not filings_manifest.exists():
        return ""
    filings = _load_json(filings_manifest)
    if not filings:
        return ""

    filing = filings[0]
    filing_dir = ticker_dir / "filings" / f"{filing['filing_date']}_{filing['form'].replace('/', '_').replace(' ', '_')}_{filing['accession_no']}"
    candidates = [
        filing_dir / "primary_document.md",
        filing_dir / "primary_document.html",
        filing_dir / "primary_document.txt",
        filing_dir / "full_submission.txt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return ""


def collect_company_filings(request: EdgarCollectionRequest) -> EdgarCollectionResult:
    if request.output_dir:
        raw_output = Path(request.output_dir).expanduser()
        if raw_output.is_absolute():
            output_dir = raw_output.resolve()
        else:
            output_dir = (ROOT_FILESYSTEM / raw_output).resolve()
    else:
        output_dir = _stage_edgar_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    configure_identity(request.identity)
    summary = explore_company(
        ticker=request.ticker,
        output_dir=output_dir,
        rows=25,
        forms=request.forms,
        latest_only=request.latest_only,
        per_form_limit=request.per_form_limit,
        include_quarterly=request.include_quarterly,
        include_markdown=request.include_markdown,
        include_full_submission=request.include_full_submission,
        download_mode=request.download_mode,
        exhibits_only=request.exhibits_only,
    )

    ticker_dir = output_dir / request.ticker.strip().upper()
    manifest_path = ticker_dir / "filings_manifest.json"
    filings = _load_json(manifest_path) if manifest_path.exists() else []
    result = EdgarCollectionResult(
        ticker=request.ticker.strip().upper(),
        company_name=str(summary.get("name", "")),
        cik=str(summary.get("cik", "")),
        output_dir=str(ticker_dir),
        manifest_path=str(manifest_path),
        preferred_filing_artifact_path=_preferred_filing_artifact_path(ticker_dir),
        filing_count=len(filings),
        filings=filings,
    )
    (ticker_dir / "edgar_collection_result.json").write_text(
        result.model_dump_json(indent=2),
        encoding="utf-8",
    )
    return result


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
    payload = _load_json(path)

    if isinstance(payload, dict) and "filings" in payload:
        filings = payload.get("filings", [])
        return {
            "manifest_path": str(path),
            "output_dir": payload.get("output_dir", ""),
            "ticker": payload.get("ticker", ""),
            "company_name": payload.get("company_name", ""),
            "filing_count": len(filings),
            "filings": filings[:20],
        }

    if isinstance(payload, list):
        return {
            "manifest_path": str(path),
            "filing_count": len(payload),
            "filings": payload[:20],
        }

    return {"manifest_path": str(path), "payload": payload}


@tool
def edgar_collect_latest_filing_for_stage(
    ticker: Annotated[str, "Ticker symbol such as VRTX or MRNA."],
    form: Annotated[str, "Single SEC form type such as 10-K or 8-K."],
    task_slug: Annotated[str, "Stage task slug used to derive the mission sandbox folder."],
    identity: Annotated[str | None, "Optional SEC identity string."] = None,
) -> dict[str, Any]:
    """Convenience wrapper that saves one latest filing form into runs/<task_slug>/unstructured/edgar."""
    output_dir = _stage_edgar_dir(task_slug)
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
