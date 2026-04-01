from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Any

from edgar import Company, set_identity

DEFAULT_BIOTECH_TICKERS: tuple[str, ...] = (
    "MRNA",
    "VRTX",
    "GILD",
    "AMGN",
    "BIIB",
    "REGN",
)

DEFAULT_FORMS: tuple[str, ...] = (
    "10-K",
    "10-Q",
    "8-K",
    "DEF 14A",
    "4",
)

DEFAULT_IDENTITY = "John Bell johnoverton743@gmail.com"


@dataclass
class FilingRecord:
    ticker: str
    company: str
    cik: Any
    form: str
    filing_date: str
    accession_no: str
    period_of_report: str | None
    acceptance_datetime: str | None
    file_number: str | None
    size: Any
    is_xbrl: Any
    is_inline_xbrl: Any
    homepage_url: str | None
    filing_url: str | None
    text_url: str | None
    primary_document: str | None
    attachment_count: int
    exhibit_count: int
    obj_type: str | None


def eprint(*args: object) -> None:
    print(*args, file=sys.stderr)


def safe_filename(text: str) -> str:
    text = text.strip().replace(" ", "_")
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("._") or "item"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_text(path: Path, content: str | None) -> None:
    if content is None:
        return
    path.write_text(content, encoding="utf-8")


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def configure_identity(explicit_identity: str | None) -> str:
    raw = (explicit_identity or os.environ.get("EDGAR_IDENTITY") or "").strip()
    if not raw:
        raw = DEFAULT_IDENTITY
        eprint(
            "WARNING: EDGAR_IDENTITY was not set; using the built-in identity fallback. "
            "Override with --identity or EDGAR_IDENTITY if needed."
        )
    set_identity(raw)
    return raw


def parse_csv_args(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if part:
                out.append(part)
    return out


def maybe_to_dataframe(obj: Any) -> Any | None:
    if obj is None:
        return None
    if hasattr(obj, "to_dataframe"):
        try:
            return obj.to_dataframe()
        except Exception:
            return None
    return obj


def save_dataframe_like(obj: Any, path_base: Path, preview_rows: int) -> None:
    df = maybe_to_dataframe(obj)
    if df is None:
        return
    try:
        df.to_csv(path_base.with_suffix(".csv"), index=True)
        preview = df.head(min(preview_rows, len(df))) if hasattr(df, "head") else df
        write_text(path_base.with_suffix(".txt"), str(preview))
    except Exception as exc:
        write_text(path_base.with_suffix(".error.txt"), f"Could not save dataframe: {exc}\n")


def save_financial_snapshots(company: Any, ticker_dir: Path, rows: int, include_quarterly: bool) -> None:
    financial_dir = ensure_dir(ticker_dir / "financials")

    try:
        annual = company.get_financials()
        save_dataframe_like(annual.income_statement(), financial_dir / "annual_income_statement", rows)
        save_dataframe_like(annual.balance_sheet(), financial_dir / "annual_balance_sheet", rows)
        save_dataframe_like(annual.cashflow_statement(), financial_dir / "annual_cashflow_statement", rows)
        facts = {
            "revenue": str(annual.get_revenue()),
            "net_income": str(annual.get_net_income()),
        }
        write_json(financial_dir / "annual_quick_facts.json", facts)
    except Exception as exc:
        write_text(financial_dir / "annual_financials.error.txt", f"{exc}\n")

    if include_quarterly:
        try:
            quarterly = company.get_quarterly_financials()
            save_dataframe_like(quarterly.income_statement(), financial_dir / "quarterly_income_statement", rows)
            save_dataframe_like(quarterly.balance_sheet(), financial_dir / "quarterly_balance_sheet", rows)
            save_dataframe_like(quarterly.cashflow_statement(), financial_dir / "quarterly_cashflow_statement", rows)
        except Exception as exc:
            write_text(financial_dir / "quarterly_financials.error.txt", f"{exc}\n")


def filing_record(ticker: str, filing: Any) -> FilingRecord:
    attachments = getattr(filing, "attachments", []) or []
    exhibits = getattr(filing, "exhibits", []) or []
    return FilingRecord(
        ticker=ticker,
        company=str(getattr(filing, "company", "")),
        cik=getattr(filing, "cik", None),
        form=str(getattr(filing, "form", "")),
        filing_date=str(getattr(filing, "filing_date", "")),
        accession_no=str(getattr(filing, "accession_no", "")),
        period_of_report=getattr(filing, "period_of_report", None),
        acceptance_datetime=str(getattr(filing, "acceptance_datetime", "")) or None,
        file_number=str(getattr(filing, "file_number", "")) or None,
        size=getattr(filing, "size", None),
        is_xbrl=getattr(filing, "is_xbrl", None),
        is_inline_xbrl=getattr(filing, "is_inline_xbrl", None),
        homepage_url=getattr(filing, "homepage_url", None),
        filing_url=getattr(filing, "filing_url", None),
        text_url=getattr(filing, "text_url", None),
        primary_document=str(getattr(filing, "primary_document", "")) or None,
        attachment_count=len(attachments),
        exhibit_count=len(exhibits),
        obj_type=str(getattr(filing, "obj_type", "")) or None,
    )


def save_attachment_inventory(filing: Any, filing_dir: Path) -> None:
    attachments = getattr(filing, "attachments", None)
    if not attachments:
        return
    rows: list[dict[str, Any]] = []
    for att in attachments:
        rows.append(
            {
                "sequence": getattr(att, "sequence", None),
                "document": getattr(att, "document", None),
                "description": getattr(att, "description", None),
                "document_type": getattr(att, "document_type", None),
                "path": getattr(att, "path", None),
                "size": getattr(att, "size", None),
                "exhibit_number": getattr(att, "exhibit_number", None),
            }
        )
    write_json(filing_dir / "attachments.json", rows)


def save_primary_content(filing: Any, filing_dir: Path, include_markdown: bool, include_full_submission: bool) -> None:
    try:
        html = filing.html()
        if html:
            write_text(filing_dir / "primary_document.html", html)
    except Exception as exc:
        write_text(filing_dir / "primary_document_html.error.txt", f"{exc}\n")

    try:
        text = filing.text()
        if text:
            write_text(filing_dir / "primary_document.txt", text)
    except Exception as exc:
        write_text(filing_dir / "primary_document_txt.error.txt", f"{exc}\n")

    if include_markdown:
        try:
            markdown = filing.markdown(include_page_breaks=True, start_page_number=1)
            if markdown:
                write_text(filing_dir / "primary_document.md", markdown)
        except Exception as exc:
            write_text(filing_dir / "primary_document_md.error.txt", f"{exc}\n")

    try:
        xml = filing.xml()
        if xml:
            write_text(filing_dir / "primary_document.xml", xml)
    except Exception:
        pass

    if include_full_submission:
        try:
            submission = filing.full_text_submission()
            if submission:
                write_text(filing_dir / "full_submission.txt", submission)
        except Exception as exc:
            write_text(filing_dir / "full_submission.error.txt", f"{exc}\n")


def save_xbrl_artifacts(filing: Any, filing_dir: Path, rows: int) -> None:
    try:
        xbrl = filing.xbrl()
    except Exception as exc:
        write_text(filing_dir / "xbrl.error.txt", f"{exc}\n")
        return

    if not xbrl:
        return

    xbrl_dir = ensure_dir(filing_dir / "xbrl")
    try:
        statements = getattr(xbrl, "statements", None)
        if statements is not None:
            save_dataframe_like(statements.income_statement(), xbrl_dir / "income_statement", rows)
            save_dataframe_like(statements.balance_sheet(), xbrl_dir / "balance_sheet", rows)
            save_dataframe_like(statements.cash_flow_statement(), xbrl_dir / "cash_flow_statement", rows)
    except Exception as exc:
        write_text(xbrl_dir / "statements.error.txt", f"{exc}\n")


def download_filing_materials(
    filing: Any,
    filing_dir: Path,
    download_mode: str,
    include_exhibits_only: bool,
) -> None:
    downloads_dir = ensure_dir(filing_dir / "downloads")

    if download_mode == "none":
        return

    try:
        if include_exhibits_only:
            exhibits = getattr(filing, "exhibits", None)
            if exhibits:
                exhibits.download(downloads_dir)
            return

        if download_mode == "primary":
            primary = getattr(filing, "document", None)
            if primary is not None:
                primary.download(downloads_dir)
            return

        if download_mode == "all":
            attachments = getattr(filing, "attachments", None)
            if attachments:
                attachments.download(downloads_dir)
            return
    except Exception as exc:
        write_text(filing_dir / "downloads.error.txt", f"{exc}\n")


def save_filing_object_preview(filing: Any, filing_dir: Path) -> None:
    obj_type = getattr(filing, "obj_type", None)
    write_json(filing_dir / "object_type.json", {"obj_type": obj_type})
    try:
        obj = filing.obj()
        if obj is not None:
            write_text(filing_dir / "object_preview.txt", repr(obj))
    except Exception as exc:
        write_text(filing_dir / "object_preview.error.txt", f"{exc}\n")


def save_filing_package(
    ticker: str,
    filing: Any,
    root_dir: Path,
    rows: int,
    include_markdown: bool,
    include_full_submission: bool,
    download_mode: str,
    exhibits_only: bool,
) -> FilingRecord:
    record = filing_record(ticker, filing)
    accession_slug = safe_filename(record.accession_no)
    form_slug = safe_filename(record.form)
    date_slug = safe_filename(record.filing_date)
    filing_dir = ensure_dir(root_dir / f"{date_slug}_{form_slug}_{accession_slug}")

    write_json(filing_dir / "metadata.json", asdict(record))
    save_attachment_inventory(filing, filing_dir)
    save_primary_content(filing, filing_dir, include_markdown, include_full_submission)
    save_xbrl_artifacts(filing, filing_dir, rows)
    save_filing_object_preview(filing, filing_dir)
    download_filing_materials(filing, filing_dir, download_mode, exhibits_only)

    return record


def choose_latest_by_form(company: Any, forms: list[str]) -> list[Any]:
    selected: list[Any] = []
    for form in forms:
        filings = company.get_filings(form=form)
        if filings is None:
            continue
        latest = filings.latest()
        if latest is not None:
            selected.append(latest)
    return selected


def choose_recent_by_form(company: Any, forms: list[str], per_form_limit: int) -> list[Any]:
    selected: list[Any] = []
    seen: set[str] = set()
    for form in forms:
        filings = company.get_filings(form=form)
        if filings is None:
            continue
        count = 0
        for filing in filings:
            accession = str(getattr(filing, "accession_no", ""))
            if accession and accession in seen:
                continue
            selected.append(filing)
            if accession:
                seen.add(accession)
            count += 1
            if count >= per_form_limit:
                break
    return selected


def explore_company(
    ticker: str,
    output_dir: Path,
    rows: int,
    forms: list[str],
    latest_only: bool,
    per_form_limit: int,
    include_quarterly: bool,
    include_markdown: bool,
    include_full_submission: bool,
    download_mode: str,
    exhibits_only: bool,
) -> dict[str, Any]:
    sym = ticker.strip().upper()
    print(f"\n{'=' * 88}\nTicker: {sym}\n{'=' * 88}")
    company = Company(sym)
    company_name = str(getattr(company, "name", ""))
    cik = getattr(company, "cik", None)
    print(f"Name: {company_name}")
    print(f"CIK: {cik}")

    ticker_dir = ensure_dir(output_dir / safe_filename(sym))
    write_json(
        ticker_dir / "company.json",
        {
            "ticker": sym,
            "name": company_name,
            "cik": cik,
            "forms_requested": forms,
        },
    )

    save_financial_snapshots(company, ticker_dir, rows, include_quarterly)

    selected = choose_latest_by_form(company, forms) if latest_only else choose_recent_by_form(company, forms, per_form_limit)
    filings_dir = ensure_dir(ticker_dir / "filings")

    manifest: list[dict[str, Any]] = []
    for filing in selected:
        try:
            record = save_filing_package(
                ticker=sym,
                filing=filing,
                root_dir=filings_dir,
                rows=rows,
                include_markdown=include_markdown,
                include_full_submission=include_full_submission,
                download_mode=download_mode,
                exhibits_only=exhibits_only,
            )
            manifest.append(asdict(record))
            print(
                f"Saved {record.form} {record.filing_date} {record.accession_no} "
                f"attachments={record.attachment_count} exhibits={record.exhibit_count}"
            )
        except Exception as exc:
            eprint(f"ERROR saving {sym} filing: {exc}")

    write_json(ticker_dir / "filings_manifest.json", manifest)

    return {
        "ticker": sym,
        "name": company_name,
        "cik": cik,
        "filings_saved": len(manifest),
        "output_dir": str(ticker_dir),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Power exploration script for EdgarTools: financial statements, filing text/markdown, "
            "XBRL snapshots, and downloaded SEC report materials."
        )
    )
    parser.add_argument(
        "--ticker",
        action="append",
        default=[],
        help="Ticker symbol. Repeatable; comma-separated values also accepted. Default: built-in biotech list.",
    )
    parser.add_argument(
        "--form",
        action="append",
        default=[],
        help='SEC form(s) to fetch. Repeatable; comma-separated allowed. Default: "10-K,10-Q,8-K,DEF 14A,4".',
    )
    parser.add_argument("--rows", type=int, default=25, help="Preview rows to save from DataFrames.")
    parser.add_argument("--per-form-limit", type=int, default=2, help="How many recent filings to save per form when not using --latest-only.")
    parser.add_argument("--latest-only", action="store_true", help="Only save the latest filing for each requested form.")
    parser.add_argument("--quarterly", action="store_true", help="Also save quarterly financial statements.")
    parser.add_argument("--output-dir", default="./edgar_downloads", help="Root directory for saved outputs.")
    parser.add_argument(
        "--download-mode",
        choices=("none", "primary", "all"),
        default="all",
        help="Download none, only the primary filing document, or all filing attachments.",
    )
    parser.add_argument(
        "--exhibits-only",
        action="store_true",
        help="Download exhibits only instead of the primary/all-attachments mode.",
    )
    parser.add_argument(
        "--no-markdown",
        action="store_true",
        help="Skip markdown export of the filing body.",
    )
    parser.add_argument(
        "--no-full-submission",
        action="store_true",
        help="Skip saving the SEC full text submission file.",
    )
    parser.add_argument(
        "--identity",
        default=None,
        help="SEC identity string. If omitted, EDGAR_IDENTITY is used, then the built-in fallback.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    tickers = [t.upper() for t in parse_csv_args(args.ticker)] if args.ticker else list(DEFAULT_BIOTECH_TICKERS)
    forms = parse_csv_args(args.form) if args.form else list(DEFAULT_FORMS)
    out_root = ensure_dir(Path(args.output_dir).expanduser().resolve())
    identity = configure_identity(args.identity)

    print("EdgarTools power exploration")
    print(f"Identity: {identity}")
    print(f"Tickers: {', '.join(tickers)}")
    print(f"Forms: {', '.join(forms)}")
    print(f"Output directory: {out_root}")
    print(f"Download mode: {'exhibits-only' if args.exhibits_only else args.download_mode}")

    manifest: list[dict[str, Any]] = []
    failures = 0
    for ticker in tickers:
        try:
            manifest.append(
                explore_company(
                    ticker=ticker,
                    output_dir=out_root,
                    rows=args.rows,
                    forms=forms,
                    latest_only=args.latest_only,
                    per_form_limit=args.per_form_limit,
                    include_quarterly=args.quarterly,
                    include_markdown=not args.no_markdown,
                    include_full_submission=not args.no_full_submission,
                    download_mode=args.download_mode,
                    exhibits_only=args.exhibits_only,
                )
            )
        except Exception as exc:
            failures += 1
            eprint(f"ERROR [{ticker}]: {exc}")

    write_json(out_root / "run_manifest.json", manifest)

    print("\nDone.")
    print(f"Saved run manifest: {out_root / 'run_manifest.json'}")
    if failures:
        eprint(f"Completed with {failures} ticker error(s).")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
