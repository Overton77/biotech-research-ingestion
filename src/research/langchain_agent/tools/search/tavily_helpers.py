"""Pure helpers for Tavily tool state and event payloads (no I/O)."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from langchain.tools import ToolRuntime

_URL_RE = re.compile(r"https?://[^\s\)\]\}<>\"']+")


def extract_urls_from_text(text: str) -> List[str]:
    if not text:
        return []
    urls = _URL_RE.findall(text)
    out: List[str] = []
    seen: set[str] = set()
    for u in urls:
        u = u.rstrip(".,;:!?)\"]}")
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def merge_visited_urls(runtime: ToolRuntime, new_urls: List[str]) -> List[str]:
    cur = list(runtime.state.get("visited_urls") or [])
    cur_set = set(cur)
    for u in new_urls:
        if u not in cur_set:
            cur.append(u)
            cur_set.add(u)
    return cur[-2000:]


def top_domains_from_urls(urls: List[str], limit: int = 5) -> List[str]:
    domains: List[str] = []
    seen: set[str] = set()
    for u in urls:
        try:
            domain = u.split("/")[2].lower()
        except (IndexError, AttributeError):
            continue
        if domain not in seen:
            seen.add(domain)
            domains.append(domain)
        if len(domains) >= limit:
            break
    return domains


def append_tool_state_event(
    runtime: ToolRuntime, field_name: str, event: Dict[str, Any]
) -> List[Dict[str, Any]]:
    cur = list(runtime.state.get(field_name) or [])
    cur.append(event)
    return cur[-50:]


def build_search_event(
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
        "top_domains": top_domains_from_urls(top_urls),
    }


def build_extract_event(
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
        "top_domains": top_domains_from_urls(extracted_urls),
    }


def build_map_event(
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
        "top_domains": top_domains_from_urls(returned_urls),
    }


def build_crawl_event(*, url: str, raw_results: Dict[str, Any]) -> Dict[str, Any]:
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
        "top_domains": top_domains_from_urls(page_urls),
    }


def clip_label(value: str, max_chars: int = 80) -> str:
    v = (value or "").strip()
    if len(v) <= max_chars:
        return v
    return v[:max_chars] + "..."
