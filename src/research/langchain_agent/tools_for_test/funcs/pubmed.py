# ! IGNORE THIS FILE FOR NOW USING MCP SERVER CUSTOM LATER 


from __future__ import annotations
import aiohttp
from typing import Any, Dict, List, Optional, Sequence
import json   
import re 
from dotenv import load_dotenv  
from langchain.tools import tool, ToolRuntime 
from pydantic import BaseModel, Field  
from langchain.agents import create_agent  
from langchain_openai import ChatOpenAI  

load_dotenv()

NCBI_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
DEFAULT_TOOL = "human-upgrade-app"

# ---------------------------------------------------------------------------
# HTTP SESSION MANAGEMENT
# ---------------------------------------------------------------------------

_http_session: Optional[aiohttp.ClientSession] = None 

pubmed_summary_model = ChatOpenAI(
    model="gpt-5-nano",
    temperature=0.0,
    output_version="responses/v1",
    max_retries=2,
)

# Both prompts expose a single `{search_results}` placeholder for .format().
# PubMed path: ESearch + ESummary + abstract text from EFetch.
PUBMED_SUMMARY_PROMPT = """You are a biomedical evidence assistant. The block below is raw retrieval from NCBI E-utilities: PubMed ESearch hit context, ESummary metadata (titles, journals, authors, IDs), and EFetch abstract text. Some articles may have no abstract in the block.

Produce structured output that matches the required schema.

Guidelines:
- Write a clear narrative summary (several short paragraphs) focused on what the retrieved literature supports, conflicts with, or leaves open. If the block contains no article content, say so explicitly.
- Key findings must be discrete, falsifiable claims grounded only in the provided text. Prefer counts, populations, interventions, and directions of effect when the abstracts state them; never invent statistics or studies.
- Separate human from non-human or in vitro evidence when the text allows.
- Citations: for each source you lean on, include title when available, PMID when present in the block, DOI when present, and a stable URL (use https://pubmed.ncbi.nlm.nih.gov/<PMID>/ when a PMID is known).

--- Retrieved PubMed material ---
{search_results}
--- End material ---"""

# PMC path: ESearch on db=pmc + stripped/truncated full-text XML as plain text.
PMC_SUMMARY_PROMPT = """You are a biomedical evidence assistant. The block below is raw retrieval from NCBI: PubMed Central (PMC) ESearch metadata plus plain text derived from full-text XML (tags removed; content may be truncated or noisy).

Produce structured output that matches the required schema.

Guidelines:
- Summarize substantive content: research question, methods when visible, main results, conclusions, and limitations if discussed.
- Key findings must be grounded only in the excerpt. If the text cuts off mid-section, note uncertainty rather than inferring missing outcomes.
- Citations: include article titles when inferable, PMID/DOI when they appear, and PMC URLs of the form https://www.ncbi.nlm.nih.gov/pmc/articles/PMCxxxxxxxx/ when a PMC ID appears in the header lines.

--- Retrieved PMC material ---
{search_results}
--- End material ---"""

# Alias for informal naming (PMEC → PMC / PubMed Central full-text path).
PMEC_SUMMARY_PROMPT = PMC_SUMMARY_PROMPT


async def close_http_session() -> None: 
    """ 
    Cleanly close the shared aiohttp session, if it exists. 
    Call this on application shutdown or at the end of a one-off script  

    """ 

    global _http_session 

    if _http_session is not None and not _http_session.closed: 
        await _http_session.close() 
    _http_session = None   




async def get_http_session() -> aiohttp.ClientSession:
    """
    Return a shared aiohttp.ClientSession, creating it if necessary.
    Call this from tools or other async functions that need HTTP.
    """
    global _http_session
    if _http_session is None or _http_session.closed: 
        timeout = aiohttp.ClientTimeout(total=20) 
        connector = aiohttp.TCPConnector( 
            limit=10,  # max concurrent connections 
            ttl_dns_cache=300, 
        )
        _http_session = aiohttp.ClientSession( 
            timeout=timeout, 
            connector=connector, 
            raise_for_status=False, # let caller handle errors
        )

    return _http_session



async def _eutils_get(
    session: aiohttp.ClientSession,
    util: str,  # "esearch", "esummary", "efetch", ...
    params: Dict[str, Any],
) -> str:
    url = f"{NCBI_BASE_URL}/{util}.fcgi"
    async with session.get(url, params=params, timeout=30) as resp:
        resp.raise_for_status()
        return await resp.text()  # caller can decide to parse JSON or XML

# ---------------------------------------------------------------------------
# RAW PUBMED/PMC HELPERS
# ---------------------------------------------------------------------------

async def pubmed_esearch(
    session: aiohttp.ClientSession,
    term: str,
    retmax: int = 50,
    api_key: Optional[str] = None,
    email: Optional[str] = None,
    use_history: bool = False,  # if True, use WebEnv/QueryKey (for huge result sets)
) -> Dict[str, Any]:
    params = {
        "db": "pubmed",
        "term": term,
        "retmode": "json",
        "retmax": retmax,
        "tool": DEFAULT_TOOL,
    }
    if email:
        params["email"] = email
    if api_key:
        params["api_key"] = api_key
    if use_history:
        params["usehistory"] = "y"

    text = await _eutils_get(session, "esearch", params)
    return json.loads(text)


def format_pubmed_esearch(
    esearch_result: Dict[str, Any],
    *,
    max_ids: Optional[int] = None,
) -> str:
    """
    Format a PubMed ESearch JSON response into a compact string.
    """
    er = esearch_result.get("esearchresult", {})
    count = er.get("count")
    retmax = er.get("retmax")
    retstart = er.get("retstart")
    idlist: List[str] = er.get("idlist") or []

    if max_ids is not None:
        idlist = idlist[:max_ids]

    lines: List[str] = []
    lines.append("=== PubMed ESearch ===")
    if count is not None:
        lines.append(f"Total hits: {count}")
    if retstart is not None and retmax is not None:
        lines.append(f"Returned range: start={retstart}, size={retmax}")
    lines.append("")

    if not idlist:
        lines.append("No PubMed IDs returned.")
    else:
        lines.append("PubMed IDs:")
        for pmid in idlist:
            lines.append(f"- {pmid}")

    webevn = er.get("webenv") or er.get("WebEnv")
    query_key = er.get("querykey") or er.get("QueryKey")
    if webevn or query_key:
        lines.append("")
        lines.append("History info (for large result sets):")
        if query_key:
            lines.append(f"QueryKey: {query_key}")
        if webevn:
            lines.append(f"WebEnv: {webevn}")

    return "\n".join(lines).strip()


async def pubmed_esummary(
    session: aiohttp.ClientSession,
    ids: Sequence[str],
    api_key: Optional[str] = None,
    email: Optional[str] = None,
) -> Dict[str, Any]:
    if not ids:
        return {}
    params = {
        "db": "pubmed",
        "id": ",".join(ids),
        "retmode": "json",
        "tool": DEFAULT_TOOL,
    }
    if email:
        params["email"] = email
    if api_key:
        params["api_key"] = api_key

    text = await _eutils_get(session, "esummary", params)
    return json.loads(text)


def format_pubmed_esummary(
    esummary_result: Dict[str, Any],
    *,
    max_articles: Optional[int] = None,
) -> str:
    """
    Format a PubMed ESummary JSON response into a human/LLM-friendly string.
    """
    result = esummary_result.get("result", {})
    uids: List[str] = result.get("uids") or []

    if max_articles is not None:
        uids = uids[:max_articles]

    lines: List[str] = []
    lines.append("=== PubMed Article Summaries (ESummary) ===")

    if not uids:
        lines.append("No articles in ESummary result.")
        return "\n".join(lines)

    for idx, uid in enumerate(uids, start=1):
        art = result.get(uid) or {}
        title = art.get("title") or "(no title)"
        pubdate = art.get("pubdate") or art.get("sortpubdate") or "(no date)"
        journal = art.get("fulljournalname") or art.get("source") or "(no journal)"
        authors_raw = art.get("authors") or []
        authors = [
            a.get("name")
            for a in authors_raw
            if isinstance(a, dict) and a.get("name")
        ]
        authors_str = ", ".join(authors[:5]) + (" et al." if len(authors) > 5 else "")

        doi = None
        for aid in art.get("articleids", []):
            if isinstance(aid, dict) and aid.get("idtype") == "doi":
                doi = aid.get("value")
                break

        lines.append(f"[Article {idx}] PMID: {uid}")
        lines.append(f"Title: {title}")
        lines.append(f"Journal: {journal}")
        lines.append(f"Publication date: {pubdate}")
        if authors_str:
            lines.append(f"Authors: {authors_str}")
        if doi:
            lines.append(f"DOI: {doi}")
        lines.append("")

    return "\n".join(lines).strip()


async def pubmed_efetch_abstracts(
    session: aiohttp.ClientSession,
    ids: Sequence[str],
    api_key: Optional[str] = None,
    email: Optional[str] = None,
) -> str:
    """
    Returns plain-text abstracts concatenated together (PubMed EFetch).
    """
    if not ids:
        return ""

    params = {
        "db": "pubmed",
        "id": ",".join(ids),
        "rettype": "abstract",
        "retmode": "text",  # text abstracts; you can also use xml
        "tool": DEFAULT_TOOL,
    }
    if email:
        params["email"] = email
    if api_key:
        params["api_key"] = api_key

    return await _eutils_get(session, "efetch", params)


def format_pubmed_efetch_abstracts(
    pmids: List[str],
    abstracts_text: str,
    *,
    max_chars: Optional[int] = None,
) -> str:
    """
    Format the plain-text result from pubmed_efetch_abstracts into a chunk
    suitable for an LLM.
    """
    text = abstracts_text.strip()
    if max_chars is not None and len(text) > max_chars:
        text = text[:max_chars] + "\n...[abstracts truncated]"

    lines: List[str] = []
    lines.append("=== PubMed Abstracts (EFetch) ===")
    if pmids:
        lines.append("PMIDs: " + ", ".join(pmids))
    lines.append("")
    lines.append(text)

    return "\n".join(lines).strip()


async def pmc_esearch(
    session: aiohttp.ClientSession,
    term: str,
    retmax: int = 50,
    api_key: Optional[str] = None,
    email: Optional[str] = None,
    use_history: bool = False,
) -> Dict[str, Any]:
    params = {
        "db": "pmc",
        "term": term,
        "retmode": "json",
        "retmax": retmax,
        "tool": DEFAULT_TOOL,
    }
    if email:
        params["email"] = email
    if api_key:
        params["api_key"] = api_key
    if use_history:
        params["usehistory"] = "y"

    text = await _eutils_get(session, "esearch", params)
    return json.loads(text)


def format_pmc_esearch(
    esearch_result: Dict[str, Any],
    *,
    max_ids: Optional[int] = None,
) -> str:
    """
    Format a PMC ESearch JSON response (db='pmc').
    """
    er = esearch_result.get("esearchresult", {})
    count = er.get("count")
    retmax = er.get("retmax")
    retstart = er.get("retstart")
    idlist: List[str] = er.get("idlist") or []

    if max_ids is not None:
        idlist = idlist[:max_ids]

    lines: List[str] = []
    lines.append("=== PMC ESearch ===")
    if count is not None:
        lines.append(f"Total hits: {count}")
    if retstart is not None and retmax is not None:
        lines.append(f"Returned range: start={retstart}, size={retmax}")
    lines.append("")

    if not idlist:
        lines.append("No PMC IDs returned.")
    else:
        lines.append("PMC IDs:")
        for pmcid in idlist:
            lines.append(f"- {pmcid}")

    webevn = er.get("webenv") or er.get("WebEnv")
    query_key = er.get("querykey") or er.get("QueryKey")
    if webevn or query_key:
        lines.append("")
        lines.append("History info (for large result sets):")
        if query_key:
            lines.append(f"QueryKey: {query_key}")
        if webevn:
            lines.append(f"WebEnv: {webevn}")

    return "\n".join(lines).strip()


async def pmc_efetch_fulltext_xml(
    session: aiohttp.ClientSession,
    ids: Sequence[str],  # e.g. ["PMC1234567", "PMC7654321"]
    api_key: Optional[str] = None,
    email: Optional[str] = None,
) -> str:
    if not ids:
        return ""

    params = {
        "db": "pmc",
        "id": ",".join(ids),
        "rettype": "full",
        "retmode": "xml",
        "tool": DEFAULT_TOOL,
    }
    if email:
        params["email"] = email
    if api_key:
        params["api_key"] = api_key

    return await _eutils_get(session, "efetch", params)


def _strip_xml_tags(xml: str) -> str:
    """
    Naive XML tag stripper: remove <...> and collapse whitespace.
    """
    no_tags = re.sub(r"<[^>]+>", " ", xml)
    return re.sub(r"\s+", " ", no_tags).strip()


def format_pmc_efetch_fulltext_xml(
    pmc_ids: List[str],
    xml_text: str,
    *,
    max_chars: Optional[int] = None,
) -> str:
    """
    Format PMC fulltext XML into a plain-text chunk for LLM summarization.
    """
    text = _strip_xml_tags(xml_text)
    if max_chars is not None and len(text) > max_chars:
        text = text[:max_chars] + " ...[full text truncated]"

    lines: List[str] = []
    lines.append("=== PMC Full Text (EFetch) ===")
    if pmc_ids:
        lines.append("PMC IDs: " + ", ".join(pmc_ids))
    lines.append("")
    lines.append(text)

    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# HIGH-LEVEL CHUNK FOR PUBMED (SEARCH + SUMMARY + ABSTRACTS)
# ---------------------------------------------------------------------------

async def pubmed_search_summarizable_chunk(
    session: aiohttp.ClientSession,
    term: str,
    *,
    max_results: int = 5,
) -> str:
    """
    End-to-end helper: PubMed search -> summaries -> abstracts,
    formatted as a single string ready for LLM summarization.
    """
    # 1) Search
    es = await pubmed_esearch(session, term, retmax=max_results)
    es_formatted = format_pubmed_esearch(es, max_ids=max_results)

    pmids = (es.get("esearchresult", {}).get("idlist") or [])[:max_results]
    if not pmids:
        return es_formatted  # nothing more to do

    # 2) Summaries (metadata)
    summary = await pubmed_esummary(session, pmids)
    summary_formatted = format_pubmed_esummary(summary, max_articles=max_results)

    # 3) Abstracts (text)
    abstracts_text = await pubmed_efetch_abstracts(session, pmids)
    abstracts_formatted = format_pubmed_efetch_abstracts(
        pmids,
        abstracts_text,
        max_chars=8000,
    )

    # Final chunk to feed to your summarization LLM
    return "\n\n".join(
        [
            es_formatted,
            "",
            summary_formatted,
            "",
            abstracts_formatted,
        ]
    )


async def pmc_fulltext_summarizable_chunk(
    session: aiohttp.ClientSession,
    term: str,
    *,
    max_results: int = 3,
    max_chars: int = 12000,
) -> str:
    """
    End-to-end helper: PMC search -> fulltext XML -> plain-text fulltext,
    formatted as a single string ready for LLM summarization.

    This is analogous to `pubmed_search_summarizable_chunk`, but uses:
      - pmc_esearch
      - format_pmc_esearch
      - pmc_efetch_fulltext_xml
      - format_pmc_efetch_fulltext_xml
    """
    # 1) Search in PMC
    es = await pmc_esearch(session, term, retmax=max_results)
    es_formatted = format_pmc_esearch(es, max_ids=max_results)

    pmc_ids = (es.get("esearchresult", {}).get("idlist") or [])[:max_results]
    if not pmc_ids:
        # No IDs: just return the search summary (still useful context)
        return es_formatted

    # 2) Fulltext XML
    xml_text = await pmc_efetch_fulltext_xml(session, pmc_ids)

    # 3) Plain-text fulltext, truncated for safety
    fulltext_formatted = format_pmc_efetch_fulltext_xml(
        pmc_ids,
        xml_text,
        max_chars=max_chars,
    )

    # Final chunk to feed into the summarization LLM
    return "\n\n".join(
        [
            es_formatted,
            "",
            fulltext_formatted,
        ]
    )

# ---------------------------------------------------------------------------
# STRUCTURED OUTPUT + SUMMARIZER MODEL (INSIDE TOOL)
# ---------------------------------------------------------------------------

# Assumes you have `summary_model` and `create_agent` available in your project,
# same as in your Tavily pipeline.
# from your_project.llm import summary_model, create_agent

class PubMedCitation(BaseModel):
    pmid: Optional[str] = None
    doi: Optional[str] = None
    title: Optional[str] = None
    url: Optional[str] = None


class PubMedResultsSummary(BaseModel):
    summary: str
    key_findings: List[str]
    citations: List[PubMedCitation]





async def summarize_pubmed_results(   
    search_results: str,
    summary_prompt: str = PUBMED_SUMMARY_PROMPT, 
    
) -> PubMedResultsSummary:
    """
    Use an LLM (summary_model) + create_agent to turn the raw PubMed block
    into a structured PubMedResultsSummary.
    """
   

    agent_instructions = summary_prompt.format(search_results=search_results)

    pubmed_summary_agent = create_agent(
        pubmed_summary_model,
        response_format=PubMedResultsSummary,
    )

    agent_response = await pubmed_summary_agent.ainvoke(
        {"messages": [{"role": "user", "content": agent_instructions}]}
    )

    # create_agent pattern: structured payload under "structured_response"
    return agent_response["structured_response"]


async def summarize_pmc_results(
    fulltext_block: str,
    prompt_template: str = PMC_SUMMARY_PROMPT,
) -> PubMedResultsSummary:
    """
    Summarize PMC fulltext block into a PubMedResultsSummary-compatible object.

    Args:
        fulltext_block: String produced by `pmc_fulltext_summarizable_chunk`.
        prompt_template: Prompt with a `{search_results}` placeholder.

    Returns:
        PubMedResultsSummary: summary, key findings, citations.
    """  

    agent_instructions = prompt_template.format(search_results=fulltext_block)
    summary_agent = create_agent(
        pubmed_summary_model,
        response_format=PubMedResultsSummary,
    )
    agent_response = await summary_agent.ainvoke(
        {"messages": [{"role": "user", "content": agent_instructions}]}
    )
    return agent_response["structured_response"]

def format_pubmed_summary_results(summary: PubMedResultsSummary) -> str:
    """
    Turn PubMedResultsSummary into a nicely formatted string for the calling model.
    """
    lines: List[str] = []

    lines.append("=== PubMed Literature Summary ===")
    lines.append(summary.summary.strip())
    lines.append("")
    lines.append("Key findings:")
    for i, finding in enumerate(summary.key_findings, start=1):
        lines.append(f"- ({i}) {finding}")

    if summary.citations:
        lines.append("")
        lines.append("Citations:")
        for c in summary.citations:
            pieces = []
            if c.title:
                pieces.append(c.title)
            if c.pmid:
                pieces.append(f"PMID: {c.pmid}")
            if c.doi:
                pieces.append(f"DOI: {c.doi}")
            if c.url:
                pieces.append(c.url)
            lines.append("- " + " | ".join(pieces))

    return "\n".join(lines).strip()


def format_pmc_summary_results(result: PubMedResultsSummary) -> str:
    """
    Convert PMC summary object into a human-readable string for the tool result.
    """
    lines: List[str] = []

    lines.append("=== PMC Full-Text Evidence Summary ===")
    lines.append("")
    lines.append(result.summary.strip())
    lines.append("")
    lines.append("Key findings:")
    if result.key_findings:
        for f in result.key_findings:
            lines.append(f"- {f}")
    else:
        lines.append("- (no key findings extracted)")
    lines.append("")
    lines.append("Citations:")
    if result.citations:
        for c in result.citations:
            parts = []
            if c.title:
                parts.append(c.title)
            if c.doi:
                parts.append(f"DOI: {c.doi}")
            if c.pmid:
                parts.append(f"PMID: {c.pmid}")
            if c.url:
                parts.append(c.url)
            if not parts:
                parts.append("(citation info missing)")
            lines.append("- " + " | ".join(parts))
    else:
        lines.append("- (no citations extracted)")

    return "\n".join(lines).strip()

# ---------------------------------------------------------------------------
# LANGCHAIN TOOL: PUBMED LITERATURE SEARCH
# ---------------------------------------------------------------------------

@tool(
    description="Search PubMed for biomedical literature on a topic and return a summarized, citation-rich report.",
    parse_docstring=False,
)
async def pubmed_literature_search_tool(
    runtime: ToolRuntime,
    query: str,
    max_results: int = 5,
) -> str: 
    """Search PubMed for biomedical literature and produce a structured summary.

    Use this for MEDICAL or CASE STUDY research when you need evidence from
    peer-reviewed journals (clinical trials, case reports, mechanistic papers,
    reviews, etc.).

    Good for topics such as:
      * Effects of a compound or intervention.
      * Mechanisms underlying a biomarker or pathway.
      * Clinical outcomes for specific populations or case types.

    Args:
        query (str): Natural-language PubMed query
            (e.g., "spermidine longevity trial",
            "NRF2 activator human study",
            "case report traumatic brain injury hyperbaric").
        max_results (int, optional): Maximum number of PubMed articles
            to consider. Defaults to 5.

    Returns:
        str: A human-readable string summarizing the main findings and
        including key citations.

    """
    # increment steps_taken safely
    steps = runtime.state.get("steps_taken", 0) + 1
    runtime.state["steps_taken"] = steps

    session = await get_http_session()

    # 1) Raw combined PubMed text block
    raw_block = await pubmed_search_summarizable_chunk(
        session,
        term=query,
        max_results=max_results,
    )

    # 2) Summarize via LLM into structured output
    pubmed_summary = await summarize_pubmed_results(raw_block, PUBMED_SUMMARY_PROMPT)

    # 3) Update citations + research_notes in runtime state
    citations_state = runtime.state.get("citations") or []
    # extend with URLs if present
    for c in pubmed_summary.citations:
        if c.url:
            citations_state.append(c.url)
        elif c.pmid:
            # fallback: construct PubMed URL
            citations_state.append(f"https://pubmed.ncbi.nlm.nih.gov/{c.pmid}/")
    runtime.state["citations"] = citations_state

    notes_state = runtime.state.get("research_notes") or []
    notes_state.append(pubmed_summary.summary)
    runtime.state["research_notes"] = notes_state

    # 4) Return formatted string for the tool result message
    return format_pubmed_summary_results(pubmed_summary)  



@tool(
    description=(
        "Search PubMed Central (PMC) for full-text biomedical articles on a topic "
        "and return a summarized, citation-rich report based on the full text."
    ),
    parse_docstring=False,
)
async def pmc_fulltext_literature_tool(
    runtime: ToolRuntime,
    query: str,
    max_results: int = 3,
    max_chars: int = 12000,
) -> str:
    """
    Search PMC for full-text biomedical literature and produce a structured summary.

    Use this when you need deeper, full-text evidence, especially for:
      * Detailed mechanistic explanations.
      * Nuanced discussion of methods, limitations, or subgroup effects.
      * Articles that are open-access and available via PubMed Central (PMC).

    Args:
        query (str): Natural-language PMC query
            (e.g., "spermidine autophagy mechanism",
            "metformin longevity randomized trial",
            "CAR T cell metabolic reprogramming").
        max_results (int, optional): Maximum number of PMC articles to consider.
            Defaults to 3 (full text can be large).
        max_chars (int, optional): Maximum number of characters of fulltext to
            keep after formatting. Defaults to 12000.

    Returns:
        str: A human-readable string summarizing the main findings and
        including key citations.
    """
    # 1) increment steps_taken safely
    steps = runtime.state.get("steps_taken", 0) + 1
    runtime.state["steps_taken"] = steps

    # 2) Get shared HTTP session
    session = await get_http_session()

    # 3) Raw combined PMC fulltext block
    raw_block = await pmc_fulltext_summarizable_chunk(
        session,
        term=query,
        max_results=max_results,
        max_chars=max_chars,
    )

    # 4) Summarize via LLM into structured output
    pmc_summary = await summarize_pmc_results(raw_block, PMC_SUMMARY_PROMPT)

    # 5) Update citations + research_notes in runtime state (reusing PubMed schema)
    citations_state = runtime.state.get("citations") or []
    for c in pmc_summary.citations:
        if c.url:
            citations_state.append(c.url)
        elif c.pmid:
            # fallback: construct PubMed URL if a PMID was inferred
            citations_state.append(f"https://pubmed.ncbi.nlm.nih.gov/{c.pmid}/")
    runtime.state["citations"] = citations_state

    notes_state = runtime.state.get("research_notes") or []
    notes_state.append(pmc_summary.summary)
    runtime.state["research_notes"] = notes_state

    # 6) Return formatted string for the tool result message
    return format_pmc_summary_results(pmc_summary)