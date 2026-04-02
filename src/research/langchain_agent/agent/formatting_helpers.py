"""Prompt-oriented string helpers for memory ingestion and dynamic prompts (no I/O)."""

from __future__ import annotations

import json
from typing import Any, List

# -----------------------------------------------------------------------------
# Text
# -----------------------------------------------------------------------------


def _truncate_text(text: str, max_chars: int) -> str:
    if text is None:
        return ""
    s = str(text)
    if max_chars <= 0:
        return ""
    suffix = "... [truncated]"
    if len(s) <= max_chars:
        return s
    if max_chars <= len(suffix):
        return suffix[:max_chars]
    return s[: max_chars - len(suffix)] + suffix


# -----------------------------------------------------------------------------
# Tavily tool state events (see tools/search/tavily_helpers.py)
# -----------------------------------------------------------------------------


def _one_line_event(ev: dict[str, Any]) -> str:
    kind = ev.get("kind", "unknown")
    if kind == "tavily_search":
        q = ev.get("query", "")
        urls = ev.get("top_urls") or []
        preview = ", ".join(u for u in urls[:3] if u)
        return (
            f'search query={json.dumps(q, ensure_ascii=False)} '
            f"n={ev.get('result_count', '?')} "
            f"depth={ev.get('search_depth')} topic={ev.get('topic')} "
            f"urls=[{preview}]"
        )
    if kind == "tavily_extract":
        urls_in = ev.get("input_urls") or []
        urls_out = ev.get("extracted_urls") or []
        return (
            f"extract in={len(urls_in)} out={ev.get('result_count', '?')} "
            f"failed={ev.get('failed_count', 0)} "
            f"sample={[u for u in urls_out[:3] if u]}"
        )
    if kind == "tavily_map":
        ru = ev.get("returned_urls") or []
        sample = ", ".join(u for u in ru[:3] if u)
        return (
            f"map root={ev.get('root_url', '')} "
            f"raw={ev.get('raw_count')} deduped={ev.get('deduped_count')} "
            f"returned={ev.get('returned_count')} sample=[{sample}]"
        )
    if kind == "tavily_crawl":
        pu = ev.get("page_urls") or []
        sample = ", ".join(u for u in pu[:3] if u)
        return (
            f"crawl root={ev.get('root_url', '')} "
            f"pages={ev.get('page_count', '?')} sample=[{sample}]"
        )
    # Fallback: compact JSON without blowing the prompt
    try:
        return json.dumps(ev, ensure_ascii=False, default=str)[:400]
    except TypeError:
        return repr(ev)[:400]


def _format_tavily_event_block(
    title: str,
    events: list[dict[str, Any]],
    *,
    max_events: int = 4,
) -> str:
    if not events:
        return f"{title}\n(no events)"

    tail = events[-max_events:] if max_events > 0 else []
    lines = [title, ""]
    for idx, ev in enumerate(tail, start=1):
        lines.append(f"- [{idx}] {_one_line_event(ev)}")
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Filesystem paths + middleware events
# -----------------------------------------------------------------------------


def _format_file_state_block(
    *,
    written_file_paths: list[str],
    edited_file_paths: list[str],
    read_file_paths: list[str],
    filesystem_events: list[dict[str, Any]],
) -> str:
    def _tail(paths: list[str], *, limit: int = 25) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for p in paths or []:
            ps = str(p).strip()
            if not ps or ps in seen:
                continue
            seen.add(ps)
            out.append(ps)
            if len(out) >= limit:
                break
        return out

    w = _tail(written_file_paths)
    e = _tail(edited_file_paths)
    r = _tail(read_file_paths)

    lines = [
        "Filesystem snapshot",
        "",
        f"Written ({len(written_file_paths)} total, showing up to 25 unique):",
    ]
    if w:
        lines.extend(f"  - {p}" for p in w)
    else:
        lines.append("  (none)")
    lines.append("")
    lines.append(f"Edited ({len(edited_file_paths)} total, showing up to 25 unique):")
    if e:
        lines.extend(f"  - {p}" for p in e)
    else:
        lines.append("  (none)")
    lines.append("")
    lines.append(f"Read ({len(read_file_paths)} total, showing up to 25 unique):")
    if r:
        lines.extend(f"  - {p}" for p in r)
    else:
        lines.append("  (none)")

    events = list(filesystem_events or [])[-20:]
    lines.extend(["", f"Recent filesystem tool events (last {len(events)} of {len(filesystem_events or [])}):"])
    if not events:
        lines.append("  (none)")
    else:
        for idx, ev in enumerate(events, start=1):
            tool = ev.get("tool_name", "?")
            path = ev.get("path", "")
            lines.append(f"  - [{idx}] {tool} {path}")

    return "\n".join(lines) 



def _safe_json_dumps(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False, default=str)


def _extract_item_value(item: Any) -> Any:
    if hasattr(item, "value"):
        return item.value
    return item


def _format_memory_items(title: str, items: List[Any], max_items: int = 5) -> str:
    if not items:
        return f"{title}:\n- none"

    lines = [f"{title}:"]
    for idx, item in enumerate(items[:max_items], start=1):
        value = _extract_item_value(item)
        if isinstance(value, dict):
            kind = value.get("kind", "memory")
            data = value.get("data", {})
            evidence = value.get("evidence", [])
            sources = value.get("sources", [])
            lines.append(f"- [{idx}] kind={kind}")
            if data:
                lines.append(f"  data={_safe_json_dumps(data)}")
            if evidence:
                lines.append(f"  evidence={_safe_json_dumps(evidence[:3])}")
            if sources:
                lines.append(f"  sources={_safe_json_dumps(sources[:3])}")
        else:
            lines.append(f"- [{idx}] {value}")
    return "\n".join(lines)



__all__ = [
    "_format_file_state_block",
    "_format_tavily_event_block",
    "_truncate_text",
    "_safe_json_dumps",
    "_extract_item_value",
    "_format_memory_items",
]
