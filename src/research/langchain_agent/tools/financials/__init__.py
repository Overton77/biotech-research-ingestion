"""SEC EDGAR / company financials tooling (EdgarTools-based).

Run exploration scripts from the repo root, for example:

    uv run python -m src.research.langchain_agent.tools.financials.edgartools_explore --help

Uses [EdgarTools](https://github.com/dgunning/edgartools) (PyPI: ``edgartools``). Set ``EDGAR_IDENTITY``
to ``Your Name your.email@domain.com`` as required by the SEC for EDGAR HTTP access.
"""

from __future__ import annotations

from src.research.langchain_agent.tools.financials.edgar_collection import (
    EdgarCollectionRequest,
    EdgarCollectionResult,
    collect_company_filings,
    stage_edgar_dir,
    summarize_edgar_manifest_path,
)
from src.research.langchain_agent.tools.financials.edgar_tools import (
    EDGAR_RESEARCH_TOOLS,
    edgar_collect_company_filings,
    edgar_collect_latest_filing_for_stage,
    edgar_read_collection_manifest,
)

__all__ = [
    "EDGAR_RESEARCH_TOOLS",
    "EdgarCollectionRequest",
    "EdgarCollectionResult",
    "collect_company_filings",
    "edgar_collect_company_filings",
    "edgar_collect_latest_filing_for_stage",
    "edgar_read_collection_manifest",
    "stage_edgar_dir",
    "summarize_edgar_manifest_path",
]
