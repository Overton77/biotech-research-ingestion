from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent

from src.research.langchain_agent.tools_for_test.test_suite.docling_test_tools import (
    DOCLING_TEST_TOOLS,
    batch_convert_local_files_with_docling,
    convert_local_file_with_docling,
    download_and_convert_url_with_docling,
    download_file_to_local,
    list_docling_supported_formats,
    shutdown_docling_process_pool,
)

load_dotenv()

DEFAULT_MODEL = "gpt-5-mini"
TEST_SUITE_DIR = Path(__file__).resolve().parent
GENERATED_DIR = TEST_SUITE_DIR / "generated"
RUN_OUTPUT_DIR = GENERATED_DIR / "run_test_outputs"
DOWNLOAD_OUTPUT_DIR = RUN_OUTPUT_DIR / "downloads"
CONVERSION_OUTPUT_DIR = RUN_OUTPUT_DIR / "conversions"

LOCAL_HTML_FILE = TEST_SUITE_DIR / "combined_resumes_docling_pipeline.html"
LOCAL_PDF_FILE = TEST_SUITE_DIR / "Final_Current_Resume_Software_Dev_6-14-24.docx.pdf"

PUBLIC_PDF_URL = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
PUBLIC_HTML_URL = "https://example.com/"


def _ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _summarize_agent_output(result: dict[str, Any]) -> str:
    messages = result.get("messages") or []
    if not messages:
        return ""
    content = messages[-1].content
    if isinstance(content, str):
        return content
    return json.dumps(content, indent=2, ensure_ascii=False, default=str)


async def _run_tool_invocation_suite() -> dict[str, Any]:
    _ensure_directory(DOWNLOAD_OUTPUT_DIR)
    _ensure_directory(CONVERSION_OUTPUT_DIR)

    manifest: dict[str, Any] = {"direct_tool_invocations": {}, "agent_invocation": None}

    manifest["direct_tool_invocations"]["supported_formats"] = await list_docling_supported_formats.ainvoke({})

    manifest["direct_tool_invocations"]["download_public_html"] = await download_file_to_local.ainvoke(
        {
            "url": PUBLIC_HTML_URL,
            "output_dir": str(DOWNLOAD_OUTPUT_DIR),
            "filename": "example_download.html",
        }
    )

    manifest["direct_tool_invocations"]["convert_local_html_to_markdown"] = await convert_local_file_with_docling.ainvoke(
        {
            "source_path": str(LOCAL_HTML_FILE),
            "output_format": "markdown",
            "output_dir": str(CONVERSION_OUTPUT_DIR),
            "output_stem": "local_html_docling_markdown",
            "use_multiprocessing": True,
            "max_workers": 2,
        }
    )

    manifest["direct_tool_invocations"]["convert_local_pdf_to_docling_json"] = await convert_local_file_with_docling.ainvoke(
        {
            "source_path": str(LOCAL_PDF_FILE),
            "output_format": "docling_json",
            "output_dir": str(CONVERSION_OUTPUT_DIR),
            "output_stem": "local_pdf_docling_document",
            "use_multiprocessing": True,
            "max_workers": 2,
        }
    )

    manifest["direct_tool_invocations"]["download_and_convert_public_pdf"] = await download_and_convert_url_with_docling.ainvoke(
        {
            "url": PUBLIC_PDF_URL,
            "output_format": "markdown",
            "download_dir": str(DOWNLOAD_OUTPUT_DIR),
            "conversion_dir": str(CONVERSION_OUTPUT_DIR),
            "filename": "public_dummy.pdf",
            "output_stem": "public_dummy_docling_markdown",
            "use_multiprocessing": True,
        }
    )

    manifest["direct_tool_invocations"]["batch_convert_local_files"] = await batch_convert_local_files_with_docling.ainvoke(
        {
            "source_paths": [str(LOCAL_HTML_FILE), str(LOCAL_PDF_FILE)],
            "output_format": "markdown",
            "output_dir": str(CONVERSION_OUTPUT_DIR / "batch_markdown"),
            "use_multiprocessing": True,
            "max_workers": 2,
        }
    )

    return manifest


async def _run_agent_validation() -> dict[str, Any]:
    if not os.getenv("OPENAI_API_KEY"):
        return {
            "status": "skipped",
            "reason": "OPENAI_API_KEY is not set in the current environment.",
        }

    agent = create_agent(
        model=DEFAULT_MODEL,
        tools=DOCLING_TEST_TOOLS,
        system_prompt=(
            "You are validating document download and Docling conversion tools. "
            "Call tools when needed, be explicit about saved output paths, and keep the final answer concise."
        ),
    )

    user_message = f"""
Use the available tools to validate the test-suite document capabilities.

1. List the supported Docling formats.
2. Convert the local HTML file `{LOCAL_HTML_FILE}` to Markdown.
3. Download and convert the public PDF `{PUBLIC_PDF_URL}` to Markdown.
4. Return only a concise summary that includes the saved output paths and the detected Docling formats used.
""".strip()

    result = await agent.ainvoke(
        {
            "messages": [{"role": "user", "content": user_message}],
        }
    )
    return {
        "status": "completed",
        "final_response": _summarize_agent_output(result),
        "message_count": len(result.get("messages") or []),
    }


async def main() -> None:
    if not LOCAL_HTML_FILE.exists():
        raise FileNotFoundError(f"Expected local HTML fixture is missing: {LOCAL_HTML_FILE}")
    if not LOCAL_PDF_FILE.exists():
        raise FileNotFoundError(f"Expected local PDF fixture is missing: {LOCAL_PDF_FILE}")

    manifest = await _run_tool_invocation_suite()
    manifest["agent_invocation"] = await _run_agent_validation()

    manifest_path = _ensure_directory(RUN_OUTPUT_DIR) / "run_test_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Docling tooling test suite completed.")
    print(f"Manifest: {manifest_path}")
    print(
        "Local HTML Markdown output:",
        manifest["direct_tool_invocations"]["convert_local_html_to_markdown"]["output_path"],
    )
    print(
        "Remote PDF Markdown output:",
        manifest["direct_tool_invocations"]["download_and_convert_public_pdf"]["conversion"]["output_path"],
    )
    print("Agent validation:", manifest["agent_invocation"]["status"])


if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        shutdown_docling_process_pool()
