from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Annotated, Any, Literal, Optional

import aiofiles
import httpx
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter
from langchain.tools import tool

from src.research.langchain_agent.tools.document.docling_helpers import (
    clip_text,
    ensure_directory,
    filename_from_response,
    resolve_output_dir,
    safe_filename,
)

logger = logging.getLogger(__name__)

# Artifact roots under this package (e.g. .../tools/document/generated/...)
DOCUMENT_TOOLS_DIR = Path(__file__).resolve().parent
GENERATED_DIR = DOCUMENT_TOOLS_DIR / "generated"
DEFAULT_DOWNLOAD_DIR = GENERATED_DIR / "downloads"
DEFAULT_CONVERSION_DIR = GENERATED_DIR / "conversions"

_PROCESS_POOL: ProcessPoolExecutor | None = None
_PROCESS_POOL_WORKERS: int | None = None


async def _sha256_for_file(path: Path) -> str:
    digest = hashlib.sha256()
    async with aiofiles.open(path, "rb") as file_handle:
        while chunk := await file_handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _get_process_pool(max_workers: Optional[int] = None) -> ProcessPoolExecutor:
    global _PROCESS_POOL, _PROCESS_POOL_WORKERS

    requested_workers = max_workers or max(os.cpu_count() or 1, 1)
    if _PROCESS_POOL is None or _PROCESS_POOL_WORKERS != requested_workers:
        if _PROCESS_POOL is not None:
            _PROCESS_POOL.shutdown(wait=False, cancel_futures=True)
        _PROCESS_POOL = ProcessPoolExecutor(max_workers=requested_workers)
        _PROCESS_POOL_WORKERS = requested_workers
    return _PROCESS_POOL


def shutdown_docling_process_pool() -> None:
    global _PROCESS_POOL, _PROCESS_POOL_WORKERS
    if _PROCESS_POOL is not None:
        _PROCESS_POOL.shutdown(wait=False, cancel_futures=True)
        _PROCESS_POOL = None
        _PROCESS_POOL_WORKERS = None


async def _download_file_impl(
    *,
    url: str,
    output_dir: Optional[str] = None,
    filename: Optional[str] = None,
    overwrite: bool = True,
    timeout_seconds: float = 60.0,
    verify_ssl: bool = True,
    allow_insecure_fallback: bool = True,
) -> dict[str, Any]:
    target_dir = resolve_output_dir(output_dir, DEFAULT_DOWNLOAD_DIR)
    logger.debug(
        "docling.download start url=%r dir=%s",
        clip_text(url, 160),
        target_dir,
    )

    async def _download_once(verify_value: bool) -> tuple[Path, int, Optional[str], int]:
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=httpx.Timeout(timeout_seconds),
            verify=verify_value,
        ) as client:
            async with client.stream("GET", url) as response:
                response.raise_for_status()
                resolved_filename = filename_from_response(url, response, filename)
                file_path = (target_dir / resolved_filename).resolve()
                if file_path.exists() and not overwrite:
                    raise FileExistsError(f"Refusing to overwrite existing file: {file_path}")

                total_bytes = 0
                async with aiofiles.open(file_path, "wb") as output_file:
                    async for chunk in response.aiter_bytes():
                        total_bytes += len(chunk)
                        await output_file.write(chunk)
                return (
                    file_path,
                    total_bytes,
                    response.headers.get("content-type"),
                    response.status_code,
                )

    tls_mode = "verified" if verify_ssl else "disabled"
    try:
        file_path, total_bytes, content_type, status_code = await _download_once(verify_ssl)
    except httpx.ConnectError as exc:
        cert_error = "CERTIFICATE_VERIFY_FAILED" in str(exc)
        if verify_ssl and allow_insecure_fallback and cert_error:
            file_path, total_bytes, content_type, status_code = await _download_once(False)
            tls_mode = "disabled_after_cert_failure"
        else:
            raise

    sha256 = await _sha256_for_file(file_path)
    stat = file_path.stat()
    logger.debug(
        "docling.download done path=%s bytes=%s status=%s",
        file_path.name,
        total_bytes,
        status_code,
    )
    return {
        "url": url,
        "file_path": str(file_path),
        "filename": file_path.name,
        "size_bytes": stat.st_size,
        "downloaded_bytes": total_bytes,
        "sha256": sha256,
        "content_type": content_type,
        "status_code": status_code,
        "tls_verification": tls_mode,
    }


def _convert_local_file_sync(
    *,
    source_path: str,
    output_dir: str,
    output_format: Literal["markdown", "docling_json"],
    output_stem: Optional[str],
    page_start: int,
    page_end: int,
    max_num_pages: int,
    max_file_size_bytes: int,
) -> dict[str, Any]:
    source = Path(source_path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Source file does not exist: {source}")

    target_dir = ensure_directory(Path(output_dir).expanduser().resolve())
    converter = DocumentConverter()
    result = converter.convert(
        str(source),
        page_range=(page_start, page_end),
        max_num_pages=max_num_pages,
        max_file_size=max_file_size_bytes,
    )

    if result.status.name != "SUCCESS":
        raise RuntimeError(
            f"Docling conversion failed for {source.name}: "
            + "; ".join(str(error) for error in result.errors)
        )

    stem = safe_filename(output_stem or source.stem, "converted_document")
    if output_format == "markdown":
        rendered = result.document.export_to_markdown()
        extension = ".md"
    else:
        rendered = result.document.model_dump_json(indent=2)
        extension = ".json"

    output_path = (target_dir / f"{stem}{extension}").resolve()
    output_path.write_text(rendered, encoding="utf-8")

    output_stat = output_path.stat()
    return {
        "source_path": str(source),
        "source_filename": source.name,
        "output_path": str(output_path),
        "output_format": output_format,
        "output_size_bytes": output_stat.st_size,
        "docling_status": result.status.name,
        "docling_input_format": result.input.format.name if result.input.format else None,
        "docling_document_hash": str(result.input.document_hash),
        "page_count": len(result.pages),
        "error_count": len(result.errors),
    }


async def _convert_local_file_impl(
    *,
    source_path: str,
    output_format: Literal["markdown", "docling_json"],
    output_dir: Optional[str] = None,
    output_stem: Optional[str] = None,
    page_start: int = 1,
    page_end: int = 2**31 - 1,
    max_num_pages: int = 2**31 - 1,
    max_file_size_bytes: int = 2**31 - 1,
    use_multiprocessing: bool = True,
    max_workers: Optional[int] = None,
) -> dict[str, Any]:
    target_dir = resolve_output_dir(output_dir, DEFAULT_CONVERSION_DIR)
    logger.debug(
        "docling.convert start source=%s format=%s",
        clip_text(source_path, 200),
        output_format,
    )
    loop = asyncio.get_running_loop()
    kwargs = {
        "source_path": source_path,
        "output_dir": str(target_dir),
        "output_format": output_format,
        "output_stem": output_stem,
        "page_start": page_start,
        "page_end": page_end,
        "max_num_pages": max_num_pages,
        "max_file_size_bytes": max_file_size_bytes,
    }

    if use_multiprocessing:
        pool = _get_process_pool(max_workers=max_workers)
        job = partial(_convert_local_file_sync, **kwargs)
        out = await loop.run_in_executor(pool, job)
    else:
        out = await asyncio.to_thread(_convert_local_file_sync, **kwargs)

    logger.debug(
        "docling.convert done output=%s",
        clip_text(str(out.get("output_path", "")), 200),
    )
    return out


@tool
async def list_docling_supported_formats() -> dict[str, Any]:
    """Return the document formats exposed by the installed Docling version."""
    return {
        "docling_supported_formats": sorted(format_member.name for format_member in InputFormat),
    }


@tool
async def download_file_to_local(
    url: Annotated[str, "HTTP or HTTPS URL to download to the local filesystem."],
    output_dir: Annotated[
        Optional[str],
        "Directory for the file. Defaults to tools/document/generated/downloads.",
    ] = None,
    filename: Annotated[
        Optional[str],
        "Optional filename override for the downloaded file. If omitted, infer from the response.",
    ] = None,
    overwrite: Annotated[bool, "Whether an existing file at the target path may be replaced."] = True,
    timeout_seconds: Annotated[float, "Request timeout for the download operation in seconds."] = 60.0,
    verify_ssl: Annotated[bool, "Whether TLS certificates should be validated for HTTPS downloads."] = True,
    allow_insecure_fallback: Annotated[
        bool,
        "If certificate validation fails, retry once with TLS verification disabled and report that fallback in the result.",
    ] = True,
) -> dict[str, Any]:
    """Download a remote file and return the saved path plus metadata."""
    return await _download_file_impl(
        url=url,
        output_dir=output_dir,
        filename=filename,
        overwrite=overwrite,
        timeout_seconds=timeout_seconds,
        verify_ssl=verify_ssl,
        allow_insecure_fallback=allow_insecure_fallback,
    )


@tool
async def convert_local_file_with_docling(
    source_path: Annotated[str, "Absolute or project-relative local file path to convert with Docling."],
    output_format: Annotated[
        Literal["markdown", "docling_json"],
        "Choose whether to save a Markdown rendering or a serialized Docling document JSON file.",
    ],
    output_dir: Annotated[
        Optional[str],
        "Directory for converted output. Defaults to tools/document/generated/conversions.",
    ] = None,
    output_stem: Annotated[
        Optional[str],
        "Optional output filename stem without extension. Defaults to the source file stem.",
    ] = None,
    page_start: Annotated[int, "First 1-based page to include in the conversion."] = 1,
    page_end: Annotated[int, "Last 1-based page to include in the conversion."] = 2**31 - 1,
    max_num_pages: Annotated[int, "Maximum number of pages Docling should process from the source file."] = 2**31 - 1,
    max_file_size_bytes: Annotated[int, "Maximum source size in bytes that Docling should accept."] = 2**31 - 1,
    use_multiprocessing: Annotated[
        bool,
        "Whether to run the conversion in a separate process for CPU-heavy workloads.",
    ] = True,
    max_workers: Annotated[
        Optional[int],
        "Optional process pool size to use when multiprocessing is enabled.",
    ] = None,
) -> dict[str, Any]:
    """Convert a local file with Docling and save either Markdown or Docling JSON."""
    return await _convert_local_file_impl(
        source_path=source_path,
        output_format=output_format,
        output_dir=output_dir,
        output_stem=output_stem,
        page_start=page_start,
        page_end=page_end,
        max_num_pages=max_num_pages,
        max_file_size_bytes=max_file_size_bytes,
        use_multiprocessing=use_multiprocessing,
        max_workers=max_workers,
    )


@tool
async def batch_convert_local_files_with_docling(
    source_paths: Annotated[
        list[str],
        "List of absolute or project-relative local file paths to convert with Docling.",
    ],
    output_format: Annotated[
        Literal["markdown", "docling_json"],
        "Choose whether to save Markdown or Docling JSON outputs for each source file.",
    ],
    output_dir: Annotated[
        Optional[str],
        "Directory for converted outputs. Defaults to tools/document/generated/conversions.",
    ] = None,
    use_multiprocessing: Annotated[
        bool,
        "Whether to run conversions in a process pool for CPU-heavy workloads.",
    ] = True,
    max_workers: Annotated[
        Optional[int],
        "Optional process pool size to use when multiprocessing is enabled.",
    ] = None,
) -> dict[str, Any]:
    """Convert multiple local files with Docling and return metadata for every saved output."""
    logger.debug("docling.batch_convert count=%s format=%s", len(source_paths), output_format)
    tasks = [
        _convert_local_file_impl(
            source_path=source_path,
            output_format=output_format,
            output_dir=output_dir,
            use_multiprocessing=use_multiprocessing,
            max_workers=max_workers,
        )
        for source_path in source_paths
    ]
    conversions = await asyncio.gather(*tasks)
    return {
        "output_format": output_format,
        "converted_count": len(conversions),
        "conversions": conversions,
    }


@tool
async def download_and_convert_url_with_docling(
    url: Annotated[str, "HTTP or HTTPS URL to download and then convert with Docling."],
    output_format: Annotated[
        Literal["markdown", "docling_json"],
        "Choose whether to save Markdown or Docling JSON after conversion.",
    ],
    download_dir: Annotated[
        Optional[str],
        "Directory where the downloaded file should be saved before conversion.",
    ] = None,
    conversion_dir: Annotated[
        Optional[str],
        "Directory where the converted output should be written.",
    ] = None,
    filename: Annotated[
        Optional[str],
        "Optional filename override for the downloaded file.",
    ] = None,
    output_stem: Annotated[
        Optional[str],
        "Optional output filename stem for the converted file.",
    ] = None,
    use_multiprocessing: Annotated[
        bool,
        "Whether to run the conversion in a separate process for CPU-heavy workloads.",
    ] = True,
    timeout_seconds: Annotated[float, "Request timeout for the download operation in seconds."] = 60.0,
    verify_ssl: Annotated[bool, "Whether TLS certificates should be validated for HTTPS downloads."] = True,
    allow_insecure_fallback: Annotated[
        bool,
        "If certificate validation fails, retry once with TLS verification disabled and report that fallback in the result.",
    ] = True,
) -> dict[str, Any]:
    """Download a remote file, convert it with Docling, and return both output locations."""
    logger.debug("docling.download_convert url=%r", clip_text(url, 160))
    download_result = await _download_file_impl(
        url=url,
        output_dir=download_dir,
        filename=filename,
        overwrite=True,
        timeout_seconds=timeout_seconds,
        verify_ssl=verify_ssl,
        allow_insecure_fallback=allow_insecure_fallback,
    )
    conversion_result = await _convert_local_file_impl(
        source_path=download_result["file_path"],
        output_format=output_format,
        output_dir=conversion_dir,
        output_stem=output_stem,
        use_multiprocessing=use_multiprocessing,
    )
    return {
        "download": download_result,
        "conversion": conversion_result,
    }


docling_document_tools = [
    list_docling_supported_formats,
    download_file_to_local,
    convert_local_file_with_docling,
    batch_convert_local_files_with_docling,
    download_and_convert_url_with_docling,
]
