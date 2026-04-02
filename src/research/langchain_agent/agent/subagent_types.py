from __future__ import annotations

from typing import Final, Sequence


BROWSER_CONTROL_SUBAGENT: Final[str] = "browser_control"
VERCEL_AGENT_BROWSER_SUBAGENT: Final[str] = "vercel_agent_browser"
CLINICALTRIALS_RESEARCH_SUBAGENT: Final[str] = "clinicaltrials_research"
TAVILY_RESEARCH_SUBAGENT: Final[str] = "tavily_research"
DOCLING_DOCUMENT_SUBAGENT: Final[str] = "docling_document"
EDGAR_RESEARCH_SUBAGENT: Final[str] = "edgar_research"

DEFAULT_STAGE_SUBAGENT_NAMES: Final[tuple[str, ...]] = (
    VERCEL_AGENT_BROWSER_SUBAGENT,
)

ALL_SUBAGENT_NAMES: Final[tuple[str, ...]] = (
    BROWSER_CONTROL_SUBAGENT,
    VERCEL_AGENT_BROWSER_SUBAGENT,
    CLINICALTRIALS_RESEARCH_SUBAGENT,
    TAVILY_RESEARCH_SUBAGENT,
    DOCLING_DOCUMENT_SUBAGENT,
    EDGAR_RESEARCH_SUBAGENT,
)

SUBAGENT_DESCRIPTIONS: Final[dict[str, str]] = {
    BROWSER_CONTROL_SUBAGENT: (
        "Use this subagent when a webpage requires interactive browsing such as "
        "clicking, expanding, or scrolling before the needed evidence is visible."
    ),
    VERCEL_AGENT_BROWSER_SUBAGENT: (
        "Use this subagent for shell-driven browser automation with Vercel agent-browser "
        "when JS-rendered pages, accordions, dynamic navigation, or direct product pages "
        "need interactive evidence collection and file-based handoff artifacts."
    ),
    CLINICALTRIALS_RESEARCH_SUBAGENT: (
        "Use this subagent for ClinicalTrials.gov sponsor searches, NCT record retrieval, "
        "and protocol/result extraction for interventional studies."
    ),
    TAVILY_RESEARCH_SUBAGENT: (
        "Use this subagent for focused web research with Tavily search, extract, map, "
        "and crawl operations when the task needs its own retrieval loop."
    ),
    DOCLING_DOCUMENT_SUBAGENT: (
        "Use this subagent to download files or webpages, convert them with Docling, "
        "and produce markdown or JSON artifacts for downstream review."
    ),
    EDGAR_RESEARCH_SUBAGENT: (
        "Use this subagent for SEC EDGAR work: resolve issuers, list and fetch filings "
        "(10-K, 10-Q, 8-K, etc.), save artifacts under the mission sandbox, and return "
        "manifest paths for downstream document ingestion."
    ),
}


def dedupe_subagent_names(names: Sequence[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for name in names:
        if name not in seen:
            ordered.append(name)
            seen.add(name)
    return ordered
