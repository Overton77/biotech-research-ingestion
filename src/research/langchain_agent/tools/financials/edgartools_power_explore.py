"""Lightweight SEC EDGAR exploration helpers used by the LangChain financial tools.

This is a small local replacement for the previously referenced helper module.
It resolves a ticker to a CIK, reads the SEC submissions JSON feed, filters the
requested forms, and writes a filings manifest plus optional primary documents.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import httpx

DEFAULT_FORMS = ["10-K", "10-Q", "8-K"]
_DEFAULT_IDENTITY = os.getenv("EDGAR_IDENTITY", "Biotech Research Agent research@example.com")


def configure_identity(identity: str | None) -> None:
    global _DEFAULT_IDENTITY
    if identity:
        _DEFAULT_IDENTITY = identity


def _headers() -> dict[str, str]:
    return {
        "User-Agent": _DEFAULT_IDENTITY,
        "Accept-Encoding": "gzip, deflate",
    }


@lru_cache(maxsize=1)
def _company_tickers() -> list[dict[str, Any]]:
    with httpx.Client(timeout=30.0, follow_redirects=True, headers=_headers()) as client:
        response = client.get("https://www.sec.gov/files/company_tickers.json")
        response.raise_for_status()
        payload = response.json()
    if isinstance(payload, dict):
        return list(payload.values())
    return payload


def _lookup_company(ticker: str) -> dict[str, Any]:
    needle = ticker.strip().upper()
    for item in _company_tickers():
        if str(item.get("ticker", "")).upper() == needle:
            return item
    raise ValueError(f"Ticker not found in SEC company_tickers feed: {ticker}")


def _fetch_submissions(cik: str) -> dict[str, Any]:
    cik_padded = str(cik).zfill(10)
    url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
    with httpx.Client(timeout=30.0, follow_redirects=True, headers=_headers()) as client:
        response = client.get(url)
        response.raise_for_status()
        return response.json()


def _iter_recent_filings(submissions: dict[str, Any]) -> list[dict[str, Any]]:
    recent = submissions.get("filings", {}).get("recent", {})
    forms = recent.get("form", []) or []
    filing_dates = recent.get("filingDate", []) or []
    accession_numbers = recent.get("accessionNumber", []) or []
    primary_documents = recent.get("primaryDocument", []) or []
    primary_doc_descriptions = recent.get("primaryDocDescription", []) or []
    items: list[dict[str, Any]] = []
    for idx, form in enumerate(forms):
        items.append(
            {
                "form": form,
                "filing_date": filing_dates[idx] if idx < len(filing_dates) else "",
                "accession_no": accession_numbers[idx] if idx < len(accession_numbers) else "",
                "primary_document": primary_documents[idx] if idx < len(primary_documents) else "",
                "primary_doc_description": primary_doc_descriptions[idx]
                if idx < len(primary_doc_descriptions)
                else "",
            }
        )
    return items


def _select_filings(
    filings: list[dict[str, Any]],
    *,
    forms: list[str],
    latest_only: bool,
    per_form_limit: int,
) -> list[dict[str, Any]]:
    allowed = {form.upper() for form in forms}
    filtered = [filing for filing in filings if str(filing.get("form", "")).upper() in allowed]
    if latest_only:
        latest_by_form: dict[str, dict[str, Any]] = {}
        for filing in filtered:
            form = str(filing["form"]).upper()
            latest = latest_by_form.get(form)
            if latest is None or filing["filing_date"] > latest["filing_date"]:
                latest_by_form[form] = filing
        return sorted(latest_by_form.values(), key=lambda item: (item["filing_date"], item["form"]), reverse=True)

    selected: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for filing in sorted(filtered, key=lambda item: item["filing_date"], reverse=True):
        form = str(filing["form"]).upper()
        counts.setdefault(form, 0)
        if counts[form] >= per_form_limit:
            continue
        selected.append(filing)
        counts[form] += 1
    return selected


def _write_primary_document(
    filing: dict[str, Any],
    *,
    cik: str,
    filing_dir: Path,
    include_markdown: bool,
) -> dict[str, str]:
    accession_nodash = filing["accession_no"].replace("-", "")
    cik_nozeros = str(int(cik))
    primary_document = filing.get("primary_document") or ""
    if not primary_document:
        return {}

    url = f"https://www.sec.gov/Archives/edgar/data/{cik_nozeros}/{accession_nodash}/{primary_document}"
    with httpx.Client(timeout=30.0, follow_redirects=True, headers=_headers()) as client:
        response = client.get(url)
        response.raise_for_status()
        content = response.text

    target = filing_dir / primary_document
    target.write_text(content, encoding="utf-8")

    result = {
        "primary_document_url": url,
        "primary_document_path": str(target),
    }

    if include_markdown:
        markdown_path = filing_dir / "primary_document.md"
        markdown_path.write_text(content, encoding="utf-8")
        result["primary_document_markdown_path"] = str(markdown_path)

    return result


def explore_company(
    *,
    ticker: str,
    output_dir: Path,
    rows: int = 25,
    forms: list[str] | None = None,
    latest_only: bool = True,
    per_form_limit: int = 1,
    include_quarterly: bool = False,
    include_markdown: bool = True,
    include_full_submission: bool = False,
    download_mode: str = "all",
    exhibits_only: bool = False,
) -> dict[str, Any]:
    del rows, include_quarterly, include_full_submission, exhibits_only

    company = _lookup_company(ticker)
    cik = str(company["cik_str"])
    submissions = _fetch_submissions(cik)
    selected = _select_filings(
        _iter_recent_filings(submissions),
        forms=forms or list(DEFAULT_FORMS),
        latest_only=latest_only,
        per_form_limit=per_form_limit,
    )

    ticker_dir = output_dir / ticker.strip().upper()
    filings_root = ticker_dir / "filings"
    filings_root.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, Any]] = []
    for filing in selected:
        filing_dir = filings_root / (
            f"{filing['filing_date']}_{filing['form'].replace('/', '_').replace(' ', '_')}_{filing['accession_no']}"
        )
        filing_dir.mkdir(parents=True, exist_ok=True)
        item = {
            **filing,
            "company_name": company.get("title", ""),
            "ticker": ticker.strip().upper(),
            "cik": str(company["cik_str"]).zfill(10),
            "filing_dir": str(filing_dir),
        }
        if download_mode != "none":
            try:
                item.update(
                    _write_primary_document(
                        filing,
                        cik=cik,
                        filing_dir=filing_dir,
                        include_markdown=include_markdown,
                    )
                )
            except Exception as exc:
                item["download_error"] = str(exc)
        manifest.append(item)

    ticker_dir.mkdir(parents=True, exist_ok=True)
    (ticker_dir / "filings_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return {
        "ticker": ticker.strip().upper(),
        "name": company.get("title", ""),
        "cik": str(company["cik_str"]).zfill(10),
        "filing_count": len(manifest),
    }


__all__ = [
    "DEFAULT_FORMS",
    "configure_identity",
    "explore_company",
]
