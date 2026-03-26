from __future__ import annotations

from src.research.langchain_agent.tools_for_test.test_suite.cyanheads_mcp_isolated_runner import (
    _extract_nct_ids,
    _extract_pmids,
    _slugify,
    build_parser,
)


def test_extract_pmids_from_markdown_search_output():
    text = """
## PubMed Search Results
**PMIDs:** 35984306, 37206869, 35334912

### Example Study
**PMID:** 35984306
""".strip()

    assert _extract_pmids(text) == ["35984306", "37206869", "35334912"]


def test_extract_nct_ids_preserves_order():
    text = """
• NCT04990869: Example One
• NCT02678611: Example Two
• NCT04990869: Example One Again
""".strip()

    assert _extract_nct_ids(text) == ["NCT04990869", "NCT02678611"]


def test_build_parser_accepts_server_specific_args():
    parser = build_parser()
    args = parser.parse_args(
        [
            "--server",
            "clinicaltrials",
            "--company",
            "Elysium Health",
            "--detail-limit",
            "2",
        ]
    )

    assert args.server == "clinicaltrials"
    assert args.company == "Elysium Health"
    assert args.detail_limit == 2
    assert _slugify(args.company) == "elysium_health"
