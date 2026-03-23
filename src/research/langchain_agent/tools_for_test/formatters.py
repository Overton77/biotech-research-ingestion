from typing import List, Dict, Any 

def _truncate_text(text: str, max_chars: int = 1200) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]"


def _format_tavily_event_block(
    title: str,
    events: List[Dict[str, Any]],
    *,
    max_events: int = 3,
) -> str:
    if not events:
        return f"{title}:\n- none"

    lines: List[str] = [f"{title}:"]
    for idx, event in enumerate(events[-max_events:], start=1):
        kind = event.get("kind", "unknown")
        request_id = event.get("request_id", "")
        response_time = event.get("response_time", "")
        lines.append(f"- [{idx}] kind={kind} request_id={request_id} response_time={response_time}")

        if kind == "tavily_search":
            lines.append(f"  query={event.get('query', '')}")
            lines.append(f"  result_count={event.get('result_count', 0)}")
            if event.get("top_domains"):
                lines.append(f"  top_domains={event.get('top_domains')}")
            if event.get("top_titles"):
                lines.append(f"  top_titles={event.get('top_titles')[:3]}")
            if event.get("top_urls"):
                lines.append(f"  top_urls={event.get('top_urls')[:5]}")

        elif kind == "tavily_extract":
            lines.append(f"  input_urls={event.get('input_urls', [])[:5]}")
            lines.append(f"  result_count={event.get('result_count', 0)} failed_count={event.get('failed_count', 0)}")
            if event.get("extracted_titles"):
                lines.append(f"  extracted_titles={event.get('extracted_titles')[:3]}")
            if event.get("extracted_urls"):
                lines.append(f"  extracted_urls={event.get('extracted_urls')[:5]}")

        elif kind == "tavily_map":
            lines.append(f"  root_url={event.get('root_url', '')}")
            lines.append(
                "  counts="
                f"raw:{event.get('raw_count', 0)} "
                f"deduped:{event.get('deduped_count', 0)} "
                f"returned:{event.get('returned_count', 0)}"
            )
            if event.get("returned_urls"):
                lines.append(f"  returned_urls={event.get('returned_urls')[:8]}")

    return "\n".join(lines)


def _format_file_state_block(
    *,
    written_file_paths: List[str],
    edited_file_paths: List[str],
    read_file_paths: List[str],
    filesystem_events: List[Dict[str, Any]],
    max_paths: int = 10,
    max_events: int = 8,
) -> str:
    lines: List[str] = ["Filesystem provenance:"]

    if written_file_paths:
        lines.append(f"- written_file_paths={written_file_paths[-max_paths:]}")
    else:
        lines.append("- written_file_paths=[]")

    if edited_file_paths:
        lines.append(f"- edited_file_paths={edited_file_paths[-max_paths:]}")
    else:
        lines.append("- edited_file_paths=[]")

    if read_file_paths:
        lines.append(f"- read_file_paths={read_file_paths[-max_paths:]}")
    else:
        lines.append("- read_file_paths=[]")

    if filesystem_events:
        lines.append("- recent_filesystem_events:")
        for event in filesystem_events[-max_events:]:
            lines.append(
                f"  - tool_name={event.get('tool_name', '')} "
                f"path={event.get('path', '')} "
                f"tool_call_id={event.get('tool_call_id', '')}"
            )
    else:
        lines.append("- recent_filesystem_events=[]")

    return "\n".join(lines)