"""Command-line exercises for PubMed / PMC helpers and the summary agent.

All paths stay under ``research/test_runs`` (imports from ``tools_for_test.funcs.pubmed``).

Examples (repo root)::

    uv run python -m src.research.test_runs.run_pubmed_cli esearch --term "spermidine autophagy" --retmax 5
    uv run python -m src.research.test_runs.run_pubmed_cli pubmed-chunk --term "NRF2 activator trial" --max-results 3 --artifacts
    uv run python -m src.research.test_runs.run_pubmed_cli summarize-pubmed --term "case reports[pt] AND vitamin d" --max-results 3 --artifacts
    uv run python -m src.research.test_runs.run_pubmed_cli case-study-flow --max-results 4 --artifacts

Environment (optional but recommended for NCBI): ``NCBI_API_KEY``, ``NCBI_EMAIL``.
OpenAI credentials are required for ``summarize-pubmed``, ``summarize-pmc``, and ``case-study-flow``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

from src.research.langchain_agent.tools_for_test.funcs import pubmed as pm
from src.research.langchain_agent.utils import save_json_artifact, save_text_artifact

RUN_NAME = "pubmed_cli"
_UTILS_DIR = Path(__file__).resolve().parent / "utils"

# Sensible default for case-study-style PubMed queries (publication type filter).
DEFAULT_CASE_STUDY_QUERY = 'case reports[Publication Type] AND English[Language]'


def _stdout_safe(text: str) -> str:
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    return text.encode(enc, errors="replace").decode(enc)


def _ncbi_email(args: argparse.Namespace) -> Optional[str]:
    return args.email or os.environ.get("NCBI_EMAIL")


def _ncbi_api_key(args: argparse.Namespace) -> Optional[str]:
    return args.api_key or os.environ.get("NCBI_API_KEY")


def _parse_ids(raw: str) -> list[str]:
    return [x.strip() for x in raw.replace(";", ",").split(",") if x.strip()]


async def _maybe_save_text(
    text: str,
    artifact_name: str,
    suffix: str,
    *,
    save: bool,
    extension: str = "md",
) -> None:
    if not save:
        return
    path = await save_text_artifact(
        text,
        RUN_NAME,
        artifact_name,
        suffix=suffix,
        base_dir=_UTILS_DIR,
        extension=extension,
    )
    print(f"[artifacts] wrote {path}", file=sys.stderr)


async def _maybe_save_json(
    data: Any,
    artifact_name: str,
    suffix: str,
    *,
    save: bool,
) -> None:
    if not save:
        return
    path = await save_json_artifact(
        data,
        RUN_NAME,
        artifact_name,
        suffix=suffix,
        base_dir=_UTILS_DIR,
    )
    print(f"[artifacts] wrote {path}", file=sys.stderr)


async def cmd_esearch(args: argparse.Namespace) -> None:
    session = await pm.get_http_session()
    raw = await pm.pubmed_esearch(
        session,
        args.term,
        retmax=args.retmax,
        api_key=_ncbi_api_key(args),
        email=_ncbi_email(args),
    )
    await _maybe_save_json(raw, "pubmed_esearch_raw", _slug(args.term), save=args.artifacts)
    if args.json:
        print(_stdout_safe(json.dumps(raw, indent=2)))
    else:
        print(_stdout_safe(pm.format_pubmed_esearch(raw, max_ids=args.retmax)))


async def cmd_esummary(args: argparse.Namespace) -> None:
    ids = _parse_ids(args.ids)
    session = await pm.get_http_session()
    raw = await pm.pubmed_esummary(
        session,
        ids,
        api_key=_ncbi_api_key(args),
        email=_ncbi_email(args),
    )
    await _maybe_save_json(
        raw,
        "pubmed_esummary_raw",
        _slug(",".join(ids[:5])),
        save=args.artifacts,
    )
    if args.json:
        print(_stdout_safe(json.dumps(raw, indent=2)))
    else:
        print(_stdout_safe(pm.format_pubmed_esummary(raw)))


async def cmd_abstracts(args: argparse.Namespace) -> None:
    ids = _parse_ids(args.ids)
    session = await pm.get_http_session()
    text = await pm.pubmed_efetch_abstracts(
        session,
        ids,
        api_key=_ncbi_api_key(args),
        email=_ncbi_email(args),
    )
    formatted = pm.format_pubmed_efetch_abstracts(ids, text, max_chars=args.max_chars)
    await _maybe_save_text(
        formatted,
        "pubmed_efetch_abstracts",
        _slug(",".join(ids[:5])),
        save=args.artifacts,
    )
    print(_stdout_safe(formatted))


async def cmd_pubmed_chunk(args: argparse.Namespace) -> None:
    session = await pm.get_http_session()
    chunk = await pm.pubmed_search_summarizable_chunk(
        session,
        args.term,
        max_results=args.max_results,
    )
    await _maybe_save_text(
        chunk,
        "pubmed_summarizable_chunk",
        _slug(args.term),
        save=args.artifacts,
    )
    print(_stdout_safe(chunk))


async def cmd_pmc_esearch(args: argparse.Namespace) -> None:
    session = await pm.get_http_session()
    raw = await pm.pmc_esearch(
        session,
        args.term,
        retmax=args.retmax,
        api_key=_ncbi_api_key(args),
        email=_ncbi_email(args),
    )
    await _maybe_save_json(raw, "pmc_esearch_raw", _slug(args.term), save=args.artifacts)
    if args.json:
        print(_stdout_safe(json.dumps(raw, indent=2)))
    else:
        print(_stdout_safe(pm.format_pmc_esearch(raw, max_ids=args.retmax)))


async def cmd_pmc_chunk(args: argparse.Namespace) -> None:
    session = await pm.get_http_session()
    chunk = await pm.pmc_fulltext_summarizable_chunk(
        session,
        args.term,
        max_results=args.max_results,
        max_chars=args.max_chars,
    )
    await _maybe_save_text(
        chunk,
        "pmc_summarizable_chunk",
        _slug(args.term),
        save=args.artifacts,
    )
    print(_stdout_safe(chunk))


async def cmd_summarize_pubmed(args: argparse.Namespace) -> None:
    session = await pm.get_http_session()
    raw_block = await pm.pubmed_search_summarizable_chunk(
        session,
        args.term,
        max_results=args.max_results,
    )
    await _maybe_save_text(
        raw_block,
        "pubmed_flow_raw_block",
        _slug(args.term),
        save=args.artifacts,
    )
    summary = await pm.summarize_pubmed_results(raw_block, pm.PUBMED_SUMMARY_PROMPT)
    out = pm.format_pubmed_summary_results(summary)
    await _maybe_save_text(
        out,
        "pubmed_flow_summary",
        _slug(args.term),
        save=args.artifacts,
    )
    if args.dump_structured:
        print(_stdout_safe(summary.model_dump_json(indent=2)))
    else:
        print(_stdout_safe(out))


async def cmd_summarize_pmc(args: argparse.Namespace) -> None:
    session = await pm.get_http_session()
    raw_block = await pm.pmc_fulltext_summarizable_chunk(
        session,
        args.term,
        max_results=args.max_results,
        max_chars=args.max_chars,
    )
    await _maybe_save_text(
        raw_block,
        "pmc_flow_raw_block",
        _slug(args.term),
        save=args.artifacts,
    )
    summary = await pm.summarize_pmc_results(raw_block, pm.PMC_SUMMARY_PROMPT)
    out = pm.format_pmc_summary_results(summary)
    await _maybe_save_text(
        out,
        "pmc_flow_summary",
        _slug(args.term),
        save=args.artifacts,
    )
    if args.dump_structured:
        print(_stdout_safe(summary.model_dump_json(indent=2)))
    else:
        print(_stdout_safe(out))


async def cmd_case_study_flow(args: argparse.Namespace) -> None:
    term = args.term or DEFAULT_CASE_STUDY_QUERY
    session = await pm.get_http_session()
    raw_block = await pm.pubmed_search_summarizable_chunk(
        session,
        term,
        max_results=args.max_results,
    )
    await _maybe_save_text(
        raw_block,
        "case_study_flow_raw_block",
        _slug(term),
        save=args.artifacts,
    )
    summary = await pm.summarize_pubmed_results(raw_block, pm.PUBMED_SUMMARY_PROMPT)
    out = pm.format_pubmed_summary_results(summary)
    await _maybe_save_text(
        out,
        "case_study_flow_summary",
        _slug(term),
        save=args.artifacts,
    )
    if args.dump_structured:
        print(_stdout_safe(summary.model_dump_json(indent=2)))
    else:
        print(_stdout_safe(out))


def _slug(value: str, max_len: int = 72) -> str:
    s = (value or "").strip()
    s = re.sub(r"[^\w\-.]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("._") or "query"
    return s[:max_len]


def _add_ncbi_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--email",
        default=None,
        help="Contact email for NCBI (or set NCBI_EMAIL).",
    )
    p.add_argument(
        "--api-key",
        dest="api_key",
        default=None,
        help="NCBI API key (or set NCBI_API_KEY).",
    )
    p.add_argument(
        "--artifacts",
        action="store_true",
        help=f"Write outputs under test_runs/utils/artifacts/{RUN_NAME}/",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PubMed/PMC test_runs CLI (raw E-utilities + summary agent).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("esearch", help="PubMed esearch.fcgi (JSON) + formatted listing.")
    p.add_argument("--term", required=True)
    p.add_argument("--retmax", type=int, default=20)
    p.add_argument("--json", action="store_true", help="Print raw JSON instead of formatted text.")
    _add_ncbi_flags(p)
    p.set_defaults(func=cmd_esearch)

    p = sub.add_parser("esummary", help="PubMed esummary for comma-separated PMIDs.")
    p.add_argument("--ids", required=True, help="Comma-separated PMIDs.")
    p.add_argument("--json", action="store_true")
    _add_ncbi_flags(p)
    p.set_defaults(func=cmd_esummary)

    p = sub.add_parser("abstracts", help="PubMed efetch abstracts for comma-separated PMIDs.")
    p.add_argument("--ids", required=True)
    p.add_argument("--max-chars", type=int, default=None)
    _add_ncbi_flags(p)
    p.set_defaults(func=cmd_abstracts)

    p = sub.add_parser(
        "pubmed-chunk",
        help="Search + esummary + abstracts as one LLM-ready block (no LLM call).",
    )
    p.add_argument("--term", required=True)
    p.add_argument("--max-results", type=int, default=5)
    _add_ncbi_flags(p)
    p.set_defaults(func=cmd_pubmed_chunk)

    p = sub.add_parser("pmc-esearch", help="PMC esearch (db=pmc).")
    p.add_argument("--term", required=True)
    p.add_argument("--retmax", type=int, default=20)
    p.add_argument("--json", action="store_true")
    _add_ncbi_flags(p)
    p.set_defaults(func=cmd_pmc_esearch)

    p = sub.add_parser(
        "pmc-chunk",
        help="PMC search + fulltext efetch formatted block (no LLM call).",
    )
    p.add_argument("--term", required=True)
    p.add_argument("--max-results", type=int, default=3)
    p.add_argument("--max-chars", type=int, default=12000)
    _add_ncbi_flags(p)
    p.set_defaults(func=cmd_pmc_chunk)

    p = sub.add_parser(
        "summarize-pubmed",
        help="pubmed-chunk + structured summary agent (OpenAI).",
    )
    p.add_argument("--term", required=True)
    p.add_argument("--max-results", type=int, default=5)
    p.add_argument(
        "--dump-structured",
        action="store_true",
        help="Print PubMedResultsSummary JSON instead of formatted markdown-like text.",
    )
    _add_ncbi_flags(p)
    p.set_defaults(func=cmd_summarize_pubmed)

    p = sub.add_parser(
        "summarize-pmc",
        help="pmc-chunk + structured summary agent (OpenAI).",
    )
    p.add_argument("--term", required=True)
    p.add_argument("--max-results", type=int, default=3)
    p.add_argument("--max-chars", type=int, default=12000)
    p.add_argument("--dump-structured", action="store_true")
    _add_ncbi_flags(p)
    p.set_defaults(func=cmd_summarize_pmc)

    p = sub.add_parser(
        "case-study-flow",
        help=(
            "Same as summarize-pubmed with default case-report-oriented query "
            f"unless --term is set ({DEFAULT_CASE_STUDY_QUERY!r})."
        ),
    )
    p.add_argument(
        "--term",
        default=None,
        help="Override PubMed query (default: case reports + English).",
    )
    p.add_argument("--max-results", type=int, default=5)
    p.add_argument("--dump-structured", action="store_true")
    _add_ncbi_flags(p)
    p.set_defaults(func=cmd_case_study_flow)

    return parser


async def _async_main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        await args.func(args)
        return 0
    finally:
        await pm.close_http_session()


def main(argv: Optional[Sequence[str]] = None) -> None:
    raise SystemExit(asyncio.run(_async_main(argv)))


if __name__ == "__main__":
    main()
