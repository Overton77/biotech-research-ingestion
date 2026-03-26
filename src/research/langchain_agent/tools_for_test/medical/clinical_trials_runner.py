"""Runnable smoke tests for the local ClinicalTrials.gov v2 tool set.

Examples from repo root:

    uv run python -m src.research.langchain_agent.tools_for_test.medical.clinical_trials_runner direct-smoke --lead-sponsor Pfizer --condition melanoma
    uv run python -m src.research.langchain_agent.tools_for_test.medical.clinical_trials_runner agent-smoke --lead-sponsor Pfizer
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Optional, Sequence

from src.research.langchain_agent.tools_for_test.medical.clinical_trials import (
    DEFAULT_RUN_NAME,
    DEFAULT_STUDY_FIELDS,
    build_query_syntax_guide,
    close_clinical_trials_client,
    create_clinical_trials_agent,
    format_search_payload,
    format_study_payload,
    get_clinical_trials_client,
    save_search_payload,
    save_study_payload,
)
from src.research.langchain_agent.utils import save_json_artifact, save_text_artifact

_MODULE_DIR = Path(__file__).resolve().parent


def _stdout_safe(text: str) -> str:
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    return text.encode(encoding, errors="replace").decode(encoding)


def _slug(value: str, max_len: int = 72) -> str:
    text = re.sub(r"[^\w\-.]+", "_", (value or "").strip())
    text = re.sub(r"_+", "_", text).strip("._") or "query"
    return text[:max_len]


def _final_message_text(result: dict) -> str:
    messages = result.get("messages") or []
    if not messages:
        return ""
    final = messages[-1]
    content = getattr(final, "content", final)
    if isinstance(content, str):
        return content
    return json.dumps(content, indent=2, default=str)


async def cmd_direct_smoke(args: argparse.Namespace) -> None:
    client = await get_clinical_trials_client()
    version = await client.get_version()
    print("=== VERSION ===")
    print(_stdout_safe(json.dumps(version, indent=2)))
    print()

    guide = await build_query_syntax_guide(
        focus="sponsor" if args.lead_sponsor else "condition",
        sample_term=args.lead_sponsor or args.condition,
        include_live_examples=True,
    )
    guide_path = await save_text_artifact(
        guide,
        DEFAULT_RUN_NAME,
        "clinical_trials_query_guide",
        suffix=_slug(args.lead_sponsor or args.condition),
        base_dir=_MODULE_DIR,
        extension="md",
    )
    print("=== QUERY GUIDE ===")
    print(_stdout_safe(guide))
    print(f"\n[artifacts] wrote {guide_path}\n")

    sponsor_payload = await client.collect_studies(
        lead_sponsor=args.lead_sponsor,
        fields=DEFAULT_STUDY_FIELDS,
        page_size=args.page_size,
        max_pages=args.max_pages,
        count_total=True,
    )
    sponsor_paths = await save_search_payload(
        sponsor_payload,
        label=f"lead_{args.lead_sponsor}",
    )
    print("=== SPONSOR SEARCH ===")
    print(_stdout_safe(format_search_payload(sponsor_payload)))
    print(f"\n[artifacts] wrote {sponsor_paths['json']}")
    print(f"[artifacts] wrote {sponsor_paths['text']}\n")

    condition_payload = await client.collect_studies(
        condition=args.condition,
        fields=DEFAULT_STUDY_FIELDS,
        page_size=args.page_size,
        max_pages=1,
        count_total=True,
    )
    condition_paths = await save_search_payload(
        condition_payload,
        label=f"condition_{args.condition}",
    )
    print("=== CONDITION SEARCH ===")
    print(_stdout_safe(format_search_payload(condition_payload)))
    print(f"\n[artifacts] wrote {condition_paths['json']}")
    print(f"[artifacts] wrote {condition_paths['text']}\n")

    studies = sponsor_payload.get("studies") or condition_payload.get("studies") or []
    if not studies:
        print("No studies found to download.")
        return

    first_nct = (
        studies[0]
        .get("protocolSection", {})
        .get("identificationModule", {})
        .get("nctId")
    )
    if not first_nct:
        print("First study had no NCT ID, skipping detail download.")
        return

    detail_payload = await client.get_study(first_nct)
    detail_paths = await save_study_payload(detail_payload, nct_id=first_nct)
    print("=== DOWNLOADED STUDY DETAIL ===")
    print(_stdout_safe(format_study_payload(detail_payload)))
    print(f"\n[artifacts] wrote {detail_paths['json']}")
    print(f"[artifacts] wrote {detail_paths['text']}")


async def cmd_agent_smoke(args: argparse.Namespace) -> None:
    agent = create_clinical_trials_agent(model="gpt-5.4-mini")
    prompt = (
        "Use the ClinicalTrials.gov tools to do three things: "
        f"1) explain how to search for lead sponsor '{args.lead_sponsor}' using query.lead or AREA[LeadSponsorName], "
        f"2) find up to {args.page_size} studies for that lead sponsor, "
        "3) fetch one NCT record in full and download it to disk. "
        "Return the key NCT IDs, sponsor names, and the exact download path."
    )
    config = {"configurable": {"thread_id": f"clinical-trials-agent-{_slug(args.lead_sponsor)}"}}
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": prompt}]},
        config=config,
    )
    final_text = _final_message_text(result)
    artifact = await save_text_artifact(
        final_text,
        DEFAULT_RUN_NAME,
        "clinical_trials_agent_response",
        suffix=_slug(args.lead_sponsor),
        base_dir=_MODULE_DIR,
        extension="md",
    )
    await save_json_artifact(
        result,
        DEFAULT_RUN_NAME,
        "clinical_trials_agent_result",
        suffix=_slug(args.lead_sponsor),
        base_dir=_MODULE_DIR,
    )
    print(_stdout_safe(final_text))
    print(f"\n[artifacts] wrote {artifact}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local ClinicalTrials.gov v2 smoke tests.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    direct = subparsers.add_parser("direct-smoke", help="Run direct API smoke tests.")
    direct.add_argument("--lead-sponsor", default="Pfizer")
    direct.add_argument("--condition", default="melanoma")
    direct.add_argument("--page-size", type=int, default=3)
    direct.add_argument("--max-pages", type=int, default=1)
    direct.set_defaults(func=cmd_direct_smoke)

    agent = subparsers.add_parser("agent-smoke", help="Run a create_agent smoke test with local tools.")
    agent.add_argument("--lead-sponsor", default="Pfizer")
    agent.add_argument("--page-size", type=int, default=3)
    agent.set_defaults(func=cmd_agent_smoke)

    return parser


async def _async_main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        await args.func(args)
        return 0
    finally:
        await close_clinical_trials_client()


def main(argv: Optional[Sequence[str]] = None) -> None:
    raise SystemExit(asyncio.run(_async_main(argv)))


if __name__ == "__main__":
    main()
