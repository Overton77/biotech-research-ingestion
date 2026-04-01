"""Explore SEC EDGAR financials via EdgarTools for one or more tickers.

Usage (from ``biotech-research-ingestion/`` repo root):

    uv sync
    set EDGAR_IDENTITY=Your Name your.email@company.com
    uv run python -m src.research.langchain_agent.tools_for_test.financials.edgartools_explore
    uv run python -m src.research.langchain_agent.tools_for_test.financials.edgartools_explore --ticker VRTX --rows 25

Docs: https://edgartools.readthedocs.io/en/stable/quickstart/
Library: https://github.com/dgunning/edgartools

Default tickers are large U.S. exchange-listed biotech / pharma names that file with the SEC
and typically have rich 10-K / 10-Q XBRL (e.g. MRNA, VRTX, GILD, AMGN, BIIB, REGN).
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Iterable

# EdgarTools exposes the ``edgar`` package (not the unrelated PyPI ``edgar`` package).
from edgar import Company, set_identity


# Well-known SEC filers useful for pipeline smoke tests (NASDAQ/NYSE biotech & pharma).
DEFAULT_BIOTECH_TICKERS: tuple[str, ...] = (
    "MRNA",  # Moderna
    "VRTX",  # Vertex
    "GILD",  # Gilead
    "AMGN",  # Amgen
    "BIIB",  # Biogen
    "REGN",  # Regeneron
)


def _configure_identity() -> None:
    """Set SEC User-Agent identity from EDGAR_IDENTITY or a visible fallback."""
    raw = (os.environ.get("EDGAR_IDENTITY") or "").strip()
    if not raw:
        raw = "Biotech Research Agent (set EDGAR_IDENTITY) noreply@example.com"
        print(
            "WARNING: EDGAR_IDENTITY is not set. Using a placeholder identity; "
            "set EDGAR_IDENTITY=\"Your Name your.email@domain.com\" for SEC compliance.\n",
            file=sys.stderr,
        )
    set_identity(raw)


def _preview_df(df, rows: int, title: str) -> None:
    if df is None:
        print(f"\n=== {title} ===\n(no dataframe)\n")
        return
    try:
        n = min(rows, len(df))
        body = df.head(n).to_string() if n else "(empty)"
    except Exception as exc:  # noqa: BLE001 — demo script
        body = f"(could not render: {exc})"
    print(f"\n=== {title} (first {rows} rows) ===\n{body}\n")


def explore_ticker(ticker: str, rows: int, include_quarterly: bool) -> None:
    sym = ticker.strip().upper()
    print(f"\n{'=' * 60}\nTicker: {sym}\n{'=' * 60}")

    company = Company(sym)
    print(f"Name: {getattr(company, 'name', '')!s}")
    print(f"CIK: {getattr(company, 'cik', '')!s}")

    financials = company.get_financials()
    _preview_df(financials.income_statement().to_dataframe(), rows, f"{sym} income_statement (annual 10-K view)")
    _preview_df(financials.balance_sheet().to_dataframe(), rows, f"{sym} balance_sheet (annual 10-K view)")
    _preview_df(financials.cashflow_statement().to_dataframe(), rows, f"{sym} cashflow_statement (annual 10-K view)")

    rev = financials.get_revenue()
    ni = financials.get_net_income()
    print(f"Quick facts (annual context): revenue={rev!r}, net_income={ni!r}")

    if include_quarterly:
        q = company.get_quarterly_financials()
        _preview_df(q.income_statement().to_dataframe(), rows, f"{sym} quarterly income_statement (10-Q view)")

    filings = company.get_filings(form="10-K")
    latest_k = filings.latest() if filings is not None else None
    if latest_k is not None:
        print(f"Latest 10-K: {latest_k!s}")
    else:
        print("Latest 10-K: (none found)")


def _parse_tickers(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    for v in values:
        for part in v.split(","):
            p = part.strip().upper()
            if p:
                out.append(p)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Explore SEC financials via EdgarTools.")
    parser.add_argument(
        "--ticker",
        action="append",
        default=[],
        help="Ticker symbol (repeatable). Comma-separated also accepted. Default: built-in biotech list.",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=12,
        help="Max rows to print per statement (default 12).",
    )
    parser.add_argument(
        "--quarterly",
        action="store_true",
        help="Also fetch quarterly (10-Q) income statement sample.",
    )
    args = parser.parse_args(argv)

    tickers = _parse_tickers(args.ticker) if args.ticker else list(DEFAULT_BIOTECH_TICKERS)

    _configure_identity()

    print(
        "EdgarTools SEC EDGAR demo\n"
        f"Tickers: {', '.join(tickers)}\n"
        "Flow: Company(ticker) -> get_financials() -> income/balance/cashflow -> to_dataframe()\n",
    )

    failures = 0
    for t in tickers:
        try:
            explore_ticker(t, rows=args.rows, include_quarterly=args.quarterly)
        except Exception as exc:  # noqa: BLE001 — exploration script
            failures += 1
            print(f"\nERROR [{t}]: {exc}\n", file=sys.stderr)

    if failures:
        print(f"\nCompleted with {failures} ticker error(s).", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
