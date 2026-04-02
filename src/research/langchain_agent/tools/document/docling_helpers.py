"""Path, filename, and MIME helpers for Docling download/convert tools (no Docling I/O)."""

from __future__ import annotations

import mimetypes
import re
from pathlib import Path
from typing import Optional
from urllib.parse import unquote, urlparse

import httpx

_CONTENT_DISPOSITION_FILENAME_RE = re.compile(
    r'filename\*?=(?:UTF-8\'\')?"?([^";]+)"?',
    flags=re.IGNORECASE,
)

CONTENT_TYPE_EXTENSION_OVERRIDES: dict[str, str] = {
    "application/pdf": ".pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
    "text/html": ".html",
    "text/markdown": ".md",
    "text/plain": ".txt",
}


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_filename(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^\w.\-]+", "_", (value or "").strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("._")
    return cleaned or fallback


def resolve_output_dir(output_dir: Optional[str], default_dir: Path) -> Path:
    return ensure_directory(Path(output_dir).expanduser().resolve() if output_dir else default_dir)


def extension_from_content_type(content_type: Optional[str]) -> str:
    if not content_type:
        return ""
    normalized = content_type.split(";")[0].strip().lower()
    override = CONTENT_TYPE_EXTENSION_OVERRIDES.get(normalized)
    if override:
        return override
    guessed = mimetypes.guess_extension(normalized)
    if guessed == ".jpe":
        return ".jpg"
    return guessed or ""


def filename_from_response(
    url: str,
    response: httpx.Response,
    override_filename: Optional[str],
) -> str:
    if override_filename:
        return safe_filename(override_filename, "downloaded_file")

    content_disposition = response.headers.get("content-disposition", "")
    match = _CONTENT_DISPOSITION_FILENAME_RE.search(content_disposition)
    if match:
        return safe_filename(unquote(match.group(1).strip()), "downloaded_file")

    parsed = urlparse(url)
    candidate = Path(unquote(parsed.path)).name
    if candidate:
        candidate = safe_filename(candidate, "downloaded_file")
    else:
        candidate = "downloaded_file"

    if Path(candidate).suffix:
        return candidate

    extension = extension_from_content_type(response.headers.get("content-type"))
    return f"{candidate}{extension}" if extension else candidate


def clip_text(value: str, max_chars: int = 120) -> str:
    """Shorten strings for log lines."""
    v = (value or "").strip()
    if len(v) <= max_chars:
        return v
    return v[:max_chars] + "..."
