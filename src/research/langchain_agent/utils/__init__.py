from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

import aiofiles


def _safe_slug(value: str, max_len: int = 80) -> str:
    value = (value or "").strip()
    value = re.sub(r"[^\w\-.]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("._")
    if not value:
        value = "artifact"
    return value[:max_len]


async def save_json_artifact(
    data: Any,
    run_name: str,
    artifact_name: str,
    suffix: Optional[str] = None,
    base_dir: Optional[Path] = None,
    indent: int = 2,
) -> Path:
    """
    Save JSON asynchronously under the current module folder by default.

    Path layout:
        <current_file_dir>/artifacts/<run_name>/<artifact_name>__<suffix>.json

    If base_dir is provided, files are saved under:
        <base_dir>/artifacts/<run_name>/<artifact_name>__<suffix>.json

    Args:
        data: Any JSON-serializable payload.
        run_name: Logical run folder name, e.g. "test_run".
        artifact_name: Base filename, e.g. "tavily_search_raw".
        suffix: Optional suffix to make filenames more specific.
        base_dir: Optional override root. Recommended usage:
            base_dir=Path(__file__).resolve().parent
        indent: JSON indentation.

    Returns:
        Path to the saved file.
    """
    root = (base_dir or Path(__file__).resolve().parent).resolve()

    artifacts_dir = root / "artifacts" / _safe_slug(run_name)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    filename = _safe_slug(artifact_name)
    if suffix:
        filename = f"{filename}__{_safe_slug(suffix)}"

    file_path = artifacts_dir / f"{filename}.json"

    payload = json.dumps(
        data,
        indent=indent,
        ensure_ascii=False,
        default=str,
    )

    async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
        await f.write(payload)

    return file_path


async def save_text_artifact(
    text: str,
    run_name: str,
    artifact_name: str,
    suffix: Optional[str] = None,
    base_dir: Optional[Path] = None,
    extension: str = "txt",
) -> Path:
    """
    Save UTF-8 text (e.g. LLM-facing formatted tool output) under artifacts/.

    Path layout matches save_json_artifact but uses .txt / .md etc.
    """
    root = (base_dir or Path(__file__).resolve().parent).resolve()
    artifacts_dir = root / "artifacts" / _safe_slug(run_name)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    ext = (extension or "txt").lstrip(".")
    filename = _safe_slug(artifact_name)
    if suffix:
        filename = f"{filename}__{_safe_slug(suffix)}"

    file_path = artifacts_dir / f"{filename}.{ext}"
    async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
        await f.write(text)

    return file_path