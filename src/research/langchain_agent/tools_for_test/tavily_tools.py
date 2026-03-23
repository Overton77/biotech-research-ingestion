from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional, Literal, Union, List, Tuple, Annotated, Dict, Any

from langchain.tools import tool, ToolRuntime
from langchain.messages import ToolMessage
from langgraph.types import Command

from src.agents.tools.tavily_search_tools import dedupe_urls
from src.agents.tools.utils.tavily_functions import (
    tavily_search as tavily_search_fn,
    tavily_extract as tavily_extract_fn,
    tavily_map as tavily_map_fn,
    tavily_crawl as tavily_crawl_fn,
    format_tavily_search_response,
    format_tavily_extract_response,
    format_tavily_map_response,
    format_tavily_crawl_response,
)
from src.clients.async_tavily_client import async_tavily_client

from src.research.langchain_agent.utils import save_json_artifact, save_text_artifact

logger = logging.getLogger(__name__)


_URL_RE = re.compile(r"https?://[^\s\)\]\}<>\"']+")


def _extract_urls(text: str) -> List[str]:
    if not text:
        return []
    urls = _URL_RE.findall(text)
    out: List[str] = []
    seen = set()
    for u in urls:
        u = u.rstrip(".,;:!?)\"]}")
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _merge_visited(runtime: ToolRuntime, new_urls: List[str]) -> List[str]:
    cur = list(runtime.state.get("visited_urls") or [])
    cur_set = set(cur)
    for u in new_urls:
        if u not in cur_set:
            cur.append(u)
            cur_set.add(u)
    return cur[-2000:]


def _top_domains(urls: List[str], limit: int = 5) -> List[str]:
    domains: List[str] = []
    seen = set()
    for u in urls:
        try:
            domain = u.split("/")[2].lower()
        except Exception:
            continue
        if domain not in seen:
            seen.add(domain)
            domains.append(domain)
        if len(domains) >= limit:
            break
    return domains


def _append_state_event(runtime: ToolRuntime, field_name: str, event: Dict[str, Any]) -> List[Dict[str, Any]]:
    cur = list(runtime.state.get(field_name) or [])
    cur.append(event)
    return cur[-50:]


def _build_search_event(
    *,
    query: str,
    max_results: int,
    search_depth: str,
    topic: Optional[str],
    raw_results: Dict[str, Any],
) -> Dict[str, Any]:
    result_items = list(raw_results.get("results", []) or [])
    top_items = result_items[: min(5, len(result_items))]
    top_urls = [r.get("url", "") for r in top_items if r.get("url")]
    top_titles = [r.get("title", "") for r in top_items if r.get("title")]

    return {
        "kind": "tavily_search",
        "query": query,
        "max_results": max_results,
        "search_depth": search_depth,
        "topic": topic,
        "request_id": raw_results.get("request_id"),
        "response_time": raw_results.get("response_time"),
        "result_count": len(result_items),
        "top_urls": top_urls,
        "top_titles": top_titles,
        "top_domains": _top_domains(top_urls),
    }


def _build_extract_event(
    *,
    url_list: List[str],
    query: Optional[str],
    raw_results: Dict[str, Any],
) -> Dict[str, Any]:
    result_items = list(raw_results.get("results", []) or [])
    failed_items = list(raw_results.get("failed_results", []) or [])

    extracted_urls = [r.get("url", "") for r in result_items if r.get("url")]
    extracted_titles = [r.get("title", "") for r in result_items if r.get("title")]

    return {
        "kind": "tavily_extract",
        "input_urls": url_list[:10],
        "query": query,
        "request_id": raw_results.get("request_id"),
        "response_time": raw_results.get("response_time"),
        "result_count": len(result_items),
        "failed_count": len(failed_items),
        "extracted_urls": extracted_urls[:10],
        "extracted_titles": extracted_titles[:10],
        "top_domains": _top_domains(extracted_urls),
    }


def _build_map_event(
    *,
    url: str,
    instructions: Optional[str],
    base_url: str,
    raw_count: int,
    deduped_count: int,
    returned_count: int,
    returned_urls: List[str],
    raw_results: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "kind": "tavily_map",
        "root_url": url,
        "base_url": base_url,
        "instructions": instructions,
        "request_id": raw_results.get("request_id"),
        "response_time": raw_results.get("response_time"),
        "raw_count": raw_count,
        "deduped_count": deduped_count,
        "returned_count": returned_count,
        "returned_urls": returned_urls[:20],
        "top_domains": _top_domains(returned_urls),
    }


def _build_crawl_event(
    *,
    url: str,
    raw_results: Dict[str, Any],
) -> Dict[str, Any]:
    result_items = list(raw_results.get("results", []) or [])
    page_urls = [r.get("url", "") for r in result_items if r.get("url")]
    return {
        "kind": "tavily_crawl",
        "root_url": url,
        "base_url": raw_results.get("base_url"),
        "request_id": raw_results.get("request_id"),
        "response_time": raw_results.get("response_time"),
        "page_count": len(result_items),
        "page_urls": page_urls[:20],
        "top_domains": _top_domains(page_urls),
    }


async def _tavily_search_impl(
    query: str,
    max_results: int = 5,
    search_depth: Literal["basic", "advanced"] = "basic",
    topic: Optional[Literal["general", "news", "finance"]] = "general",
    include_images: bool = False,
    include_raw_content: bool | Literal["markdown", "text"] = False,
    include_domains: Optional[List[str]] = None,
    exclude_domains: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    logger.info(f"🌐 TAVILY SEARCH: '{query[:80]}{'...' if len(query) > 80 else ''}'")

    search_results = await tavily_search_fn(
        client=async_tavily_client,
        query=query,
        max_results=max_results,
        search_depth=search_depth,
        topic=topic,
        include_images=include_images,
        include_raw_content=include_raw_content,
        include_domains=include_domains,
        exclude_domains=exclude_domains,
        start_date=start_date,
        end_date=end_date,
    )

    slug = query[:30].replace(" ", "_")
    await save_json_artifact(
        {"query": query, "results": search_results},
        "test_run",
        "tavily_search_raw",
        suffix=slug,
    )

    formatted = format_tavily_search_response(
        search_results,
        max_results=max_results,
        max_content_chars=1200,
        query_hint=query,
    )
    await save_text_artifact(
        formatted,
        "test_run",
        "tavily_search_formatted",
        suffix=slug,
        extension="md",
    )

    return formatted, search_results


async def _tavily_extract_impl(
    urls: Union[str, List[str]],
    query: Optional[str] = None,
    chunks_per_source: int = 3,
    extract_depth: Literal["basic", "advanced"] = "basic",
    include_images: bool = False,
    include_favicon: bool = False,
    format: Literal["markdown", "text"] = "markdown",
) -> Tuple[str, Dict[str, Any], List[str]]:
    url_list = urls if isinstance(urls, list) else [urls]
    logger.info(f"🔗 TAVILY EXTRACT: {len(url_list)} URL(s)")

    extract_results = await tavily_extract_fn(
        client=async_tavily_client,
        urls=urls,
        query=query,
        chunks_per_source=chunks_per_source,
        extract_depth=extract_depth,
        include_images=include_images,
        include_favicon=include_favicon,
        format=format,
    )

    ex_slug = f"{len(url_list)}_urls"
    await save_json_artifact(
        {"urls": url_list, "query": query, "results": extract_results},
        "test_run",
        "tavily_extract_raw",
        suffix=ex_slug,
    )

    formatted = format_tavily_extract_response(
        extract_results,
        max_results=10,
        max_content_chars=5000,
    )
    await save_text_artifact(
        formatted,
        "test_run",
        "tavily_extract_formatted",
        suffix=ex_slug,
        extension="md",
    )

    return formatted, extract_results, url_list


async def _tavily_map_impl(
    url: str,
    instructions: Optional[str] = None,
    max_depth: int = 2,
    max_breadth: int = 60,
    limit: int = 120,
    output_mode: Literal["formatted", "raw"] = "formatted",
    dedupe: bool = True,
    max_return_urls: Optional[int] = 300,
    drop_fragment: bool = True,
    drop_query: bool = False,
) -> Tuple[str, Dict[str, Any], List[str], int, int, int, str]:
    logger.info(f"🗺️ TAVILY MAP: '{url[:80]}{'...' if len(url) > 80 else ''}'")

    map_results = await tavily_map_fn(
        client=async_tavily_client,
        url=url,
        instructions=instructions,
        max_depth=max_depth,
        max_breadth=max_breadth,
        limit=limit,
    )

    map_slug = url.replace("/", "_")[:30]
    await save_json_artifact(
        {"url": url, "instructions": instructions, "results": map_results},
        "test_run",
        "tavily_map_raw",
        suffix=map_slug,
    )

    urls: List[str] = list(map_results.get("results", []) or [])
    raw_count = len(urls)

    if dedupe and urls:
        urls = dedupe_urls(urls, drop_fragment=drop_fragment, drop_query=drop_query)

    deduped_count = len(urls)

    if max_return_urls is not None and deduped_count > max_return_urls:
        urls = urls[:max_return_urls]

    returned_count = len(urls)
    base = map_results.get("base_url", url)

    stats = {"raw_count": raw_count, "deduped_count": deduped_count, "returned_count": returned_count}
    if output_mode == "raw":
        header = (
            "=== RAW MAP RESULTS ===\n"
            f"Base: {base}\n"
            f"Raw URLs: {raw_count} | Deduped: {deduped_count} | Returned: {returned_count}\n"
        )
        formatted = header + "\n".join(urls)
    else:
        formatted = format_tavily_map_response(
            map_results, urls_override=urls, stats=stats
        )
    await save_text_artifact(
        formatted,
        "test_run",
        "tavily_map_formatted",
        suffix=map_slug,
        extension="md",
    )

    return formatted, map_results, urls, raw_count, deduped_count, returned_count, base


async def _tavily_crawl_impl(
    url: str,
    instructions: Optional[str] = None,
    chunks_per_source: int = 2,
    max_depth: int = 1,
    max_breadth: int = 12,
    limit: int = 6,
    extract_depth: Literal["basic", "advanced"] = "basic",
    format: Literal["markdown", "text"] = "markdown",
    include_images: bool = False,
    allow_external: bool = False,
    timeout: float = 75.0,
) -> Tuple[str, Dict[str, Any]]:
    logger.info(f"🕸️ TAVILY CRAWL: '{url[:80]}{'...' if len(url) > 80 else ''}'")

    crawl_results = await tavily_crawl_fn(
        client=async_tavily_client,
        url=url,
        instructions=instructions,
        chunks_per_source=chunks_per_source,
        max_depth=max_depth,
        max_breadth=max_breadth,
        limit=limit,
        extract_depth=extract_depth,
        format=format,
        include_images=include_images,
        allow_external=allow_external,
        timeout=timeout,
    )

    crawl_slug = url.replace("/", "_")[:30]
    await save_json_artifact(
        {"url": url, "instructions": instructions, "results": crawl_results},
        "test_run",
        "tavily_crawl_raw",
        suffix=crawl_slug,
    )

    formatted = format_tavily_crawl_response(
        crawl_results,
        max_results=5,
        max_content_chars=1600,
    )
    await save_text_artifact(
        formatted,
        "test_run",
        "tavily_crawl_formatted",
        suffix=crawl_slug,
        extension="md",
    )

    return formatted, crawl_results


# ============================================================================
# TOOLS WITH STATE UPDATES
# ============================================================================

@tool(
    description=(
        "Search the web using Tavily and return formatted search results directly. "
        "No LLM summarization is applied."
    ),
    parse_docstring=False,
)
async def search_web(
    runtime: ToolRuntime,
    query: Annotated[
        str,
        "Search query. Keep it short but specific: entity + field + evidence terms."
    ],
    max_results: Annotated[
        int,
        "Maximum number of results to return (keep ≤8 for routine research unless you need breadth)."
    ] = 5,
    search_depth: Annotated[
        Literal["basic", "advanced"],
        "basic is cheaper/faster; advanced gives deeper retrieval."
    ] = "basic",
    topic: Annotated[
        Optional[Literal["general", "news", "finance"]],
        "Use general, news, or finance depending on search intent."
    ] = "general",
    include_images: Annotated[
        bool,
        "Whether to include image URLs."
    ] = False,
    include_raw_content: Annotated[
        bool | Literal["markdown", "text"],
        "Whether to include raw content from pages."
    ] = False,
    include_domains: Annotated[
        Optional[List[str]],
        "Optional allowlist of domains."
    ] = None,
    exclude_domains: Annotated[
        Optional[List[str]],
        "Optional blocklist of domains."
    ] = None,
    start_date: Annotated[
        Optional[str],
        "Optional start date filter in YYYY-MM-DD format."
    ] = None,
    end_date: Annotated[
        Optional[str],
        "Optional end date filter in YYYY-MM-DD format."
    ] = None,
) -> Command:
    formatted, raw_results = await _tavily_search_impl(
        query=query,
        max_results=max_results,
        search_depth=search_depth,
        topic=topic,
        include_images=include_images,
        include_raw_content=include_raw_content,
        include_domains=include_domains,
        exclude_domains=exclude_domains,
        start_date=start_date,
        end_date=end_date,
    )

    event = _build_search_event(
        query=query,
        max_results=max_results,
        search_depth=search_depth,
        topic=topic,
        raw_results=raw_results,
    )

    discovered_urls = list(event.get("top_urls", []))
    visited_urls = _merge_visited(runtime, discovered_urls)
    tavily_search_events = _append_state_event(runtime, "tavily_search_events", event)

    return Command(
        update={
            "visited_urls": visited_urls,
            "tavily_search_events": tavily_search_events,
            "messages": [
                ToolMessage(
                    content=formatted,
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


@tool(
    description=(
        "Extract content from one or more URLs using Tavily and return formatted extracted content directly. "
        "No LLM summarization is applied."
    ),
    parse_docstring=False,
)
async def extract_from_urls(
    runtime: ToolRuntime,
    urls: Annotated[
        Union[str, List[str]],
        "One URL or a list of URLs to extract from."
    ],
    query: Annotated[
        Optional[str],
        "Optional extraction filter to focus on specific fields or facts."
    ] = None,
    chunks_per_source: Annotated[
        int,
        "Integer between 1 and 5 (clamped if out of range)."
    ] = 3,
    extract_depth: Annotated[
        Literal["basic", "advanced"],
        "basic is cheaper/faster; advanced is better for dense pages."
    ] = "basic",
    include_images: Annotated[
        bool,
        "Whether to include images found on the page."
    ] = False,
    include_favicon: Annotated[
        bool,
        "Whether to include favicon URLs."
    ] = False,
    format: Annotated[
        Literal["markdown", "text"],
        "Format for extracted content."
    ] = "markdown",
) -> Command:
    chunks_per_source = max(1, min(5, int(chunks_per_source)))
    formatted, raw_results, url_list = await _tavily_extract_impl(
        urls=urls,
        query=query,
        chunks_per_source=chunks_per_source,
        extract_depth=extract_depth,
        include_images=include_images,
        include_favicon=include_favicon,
        format=format,
    )

    event = _build_extract_event(
        url_list=url_list,
        query=query,
        raw_results=raw_results,
    )

    discovered_urls = list(dict.fromkeys(url_list + list(event.get("extracted_urls", []))))
    visited_urls = _merge_visited(runtime, discovered_urls)
    tavily_extract_events = _append_state_event(runtime, "tavily_extract_events", event)

    return Command(
        update={
            "visited_urls": visited_urls,
            "tavily_extract_events": tavily_extract_events,
            "messages": [
                ToolMessage(
                    content=formatted,
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


@tool(
    description=(
        "Map a website using Tavily and return formatted discovered links directly. "
        "No LLM summarization is applied."
    ),
    parse_docstring=False,
)
async def map_website(
    runtime: ToolRuntime,
    url: Annotated[
        str,
        "Root URL or domain to map."
    ],
    instructions: Annotated[
        Optional[str],
        "Optional guidance to steer discovery."
    ] = None,
    max_depth: Annotated[
        int,
        "How many link-hops from the root to explore (1–2 is usually enough)."
    ] = 1,
    max_breadth: Annotated[
        int,
        "Maximum links to follow per level (higher = more API cost)."
    ] = 40,
    limit: Annotated[
        int,
        "Hard cap on total URLs processed by Tavily map."
    ] = 60,
    output_mode: Annotated[
        Literal["formatted", "raw"],
        "formatted returns formatted map results; raw returns URL-per-line output."
    ] = "formatted",
    dedupe: Annotated[
        bool,
        "Whether to deduplicate discovered URLs."
    ] = True,
    max_return_urls: Annotated[
        Optional[int],
        "Maximum number of URLs returned to the model after dedupe."
    ] = 80,
    drop_fragment: Annotated[
        bool,
        "Whether to drop URL fragments during dedupe."
    ] = True,
    drop_query: Annotated[
        bool,
        "Whether to drop query strings during dedupe."
    ] = False,
) -> Command:
    (
        formatted,
        raw_results,
        returned_urls,
        raw_count,
        deduped_count,
        returned_count,
        base_url,
    ) = await _tavily_map_impl(
        url=url,
        instructions=instructions,
        max_depth=max_depth,
        max_breadth=max_breadth,
        limit=limit,
        output_mode=output_mode,
        dedupe=dedupe,
        max_return_urls=max_return_urls,
        drop_fragment=drop_fragment,
        drop_query=drop_query,
    )

    event = _build_map_event(
        url=url,
        instructions=instructions,
        base_url=base_url,
        raw_count=raw_count,
        deduped_count=deduped_count,
        returned_count=returned_count,
        returned_urls=returned_urls,
        raw_results=raw_results,
    )

    discovered_urls = [url] + returned_urls
    visited_urls = _merge_visited(runtime, discovered_urls)
    tavily_map_events = _append_state_event(runtime, "tavily_map_events", event)

    return Command(
        update={
            "visited_urls": visited_urls,
            "tavily_map_events": tavily_map_events,
            "messages": [
                ToolMessage(
                    content=formatted,
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


@tool(
    description=(
        "Crawl a website from a root URL using Tavily and return formatted page content. "
        "No LLM summarization is applied."
    ),
    parse_docstring=False,
)
async def crawl_website(
    runtime: ToolRuntime,
    url: Annotated[str, "Canonical site root to crawl (e.g. https://example.com/)."],
    instructions: Annotated[
        Optional[str],
        "Optional focus; when set, enables semantic chunks_per_source behavior.",
    ] = None,
    chunks_per_source: Annotated[int, "1–5 when instructions are provided."] = 2,
    max_depth: Annotated[int, "Link depth from root (keep low for cost)."] = 1,
    max_breadth: Annotated[int, "Links per level cap."] = 12,
    limit: Annotated[int, "Max pages to fetch."] = 6,
    extract_depth: Annotated[
        Literal["basic", "advanced"],
        "basic=faster; advanced for heavy/JS pages.",
    ] = "basic",
    format: Annotated[Literal["markdown", "text"], "Content format."] = "markdown",
    allow_external: Annotated[
        bool,
        "False = same-site crawl only (cheaper, typical for one-domain research). True follows external links.",
    ] = False,
) -> Command:
    cps = max(1, min(5, int(chunks_per_source))) if (instructions and instructions.strip()) else chunks_per_source
    formatted, raw_results = await _tavily_crawl_impl(
        url=url,
        instructions=instructions,
        chunks_per_source=cps,
        max_depth=max_depth,
        max_breadth=max_breadth,
        limit=limit,
        extract_depth=extract_depth,
        format=format,
        allow_external=allow_external,
    )

    event = _build_crawl_event(url=url, raw_results=raw_results)
    page_urls = list(event.get("page_urls", []))
    visited_urls = _merge_visited(runtime, [url] + page_urls)
    tavily_crawl_events = _append_state_event(runtime, "tavily_crawl_events", event)

    return Command(
        update={
            "visited_urls": visited_urls,
            "tavily_crawl_events": tavily_crawl_events,
            "messages": [
                ToolMessage(
                    content=formatted,
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


def _preview(text: str, max_chars: int = 2200) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n... [truncated for terminal preview]"


async def run_tavily_smoke_test() -> None:
    """
    Minimal live checks: search -> extract -> map -> crawl on known roots.
    Writes raw JSON + formatted `.md` under `test_runs/utils/artifacts/test_run/`.
    Requires TAVILY_API_KEY.
    """
    print("=== Tavily smoke test (search -> extract -> map -> crawl) ===\n")

    # 1) Search — short query, basic depth, few results
    search_query = "Qualia Life Sciences supplement company official site"
    formatted_search, raw_search = await _tavily_search_impl(
        query=search_query,
        max_results=5,
        search_depth="basic",
        topic="general",
        include_images=False,
        include_raw_content=False,
    )
    print("[1] SEARCH")
    print(f"    query: {search_query!r}")
    print(f"    raw result count: {len(raw_search.get('results') or [])}")
    print(_preview(formatted_search))
    print()

    results = list(raw_search.get("results") or [])
    extract_url: Optional[str] = None
    for r in results:
        u = r.get("url")
        if isinstance(u, str) and u.startswith("http"):
            extract_url = u
            break
    if not extract_url:
        extract_url = "https://www.qualialife.com/"

    # 2) Extract — one URL, targeted chunks
    extract_focus = "company products science mission"
    formatted_extract, raw_extract, _urls = await _tavily_extract_impl(
        urls=[extract_url],
        query=extract_focus,
        chunks_per_source=2,
        extract_depth="basic",
    )
    print("[2] EXTRACT")
    print(f"    url: {extract_url}")
    print(f"    ok pages: {len(raw_extract.get('results') or [])} | failed: {len(raw_extract.get('failed_results') or [])}")
    print(_preview(formatted_extract))
    print()

    # 3) Map — same host as crawl below; tight breadth/limit
    map_root = "https://www.elysiumhealth.com/"
    (
        formatted_map,
        raw_map,
        _map_urls,
        map_raw_n,
        map_dedup_n,
        map_ret_n,
        _map_base,
    ) = await _tavily_map_impl(
        url=map_root,
        instructions=None,
        max_depth=1,
        max_breadth=25,
        limit=40,
        max_return_urls=28,
    )
    print("[3] MAP")
    print(f"    root: {map_root}")
    print(f"    raw: {map_raw_n} | deduped: {map_dedup_n} | returned: {map_ret_n}")
    print(_preview(formatted_map))
    print()

    # 4) Crawl — shallow, small cap (same root as map)
    crawl_root = map_root
    formatted_crawl, raw_crawl = await _tavily_crawl_impl(
        url=crawl_root,
        instructions=None,
        max_depth=1,
        max_breadth=12,
        limit=5,
        extract_depth="basic",
    )
    print("[4] CRAWL")
    print(f"    root: {crawl_root}")
    print(f"    pages returned: {len(raw_crawl.get('results') or [])}")
    print(_preview(formatted_crawl))

    art_dir = Path(__file__).resolve().parents[1] / "utils" / "artifacts" / "test_run"
    await save_json_artifact(
        {
            "steps": ["search", "extract", "map", "crawl"],
            "search_query": search_query,
            "extract_url": extract_url,
            "map_root": map_root,
            "crawl_root": crawl_root,
            "artifact_dir": str(art_dir.resolve()),
        },
        "test_run",
        "tavily_smoke_manifest",
        suffix="latest",
    )
    print(f"\n--- Artifacts (raw JSON + formatted .md) ---\n  {art_dir.resolve()}")
    if art_dir.is_dir():
        for p in sorted(art_dir.iterdir()):
            if p.is_file():
                print(f"  - {p.name}")
    print("\n=== Done ===")


tavily_test_tools = [search_web, extract_from_urls, map_website, crawl_website]