"""SYNTHETIC 10-K fixtures for unit tests. Not real filing text; never use outside tests.

The HTML mimics the *structure* of real inline-XBRL 10-K documents (hidden ``ix:header``,
a page-numbered table of contents, bold item headings in styled spans, running page
footers, numeric tables) for a fictional "Example Widgets Corp". All prose is invented
filler written for these tests.
"""

from __future__ import annotations

import json
from typing import Any

SYNTHETIC_LABEL = "SYNTHETIC - fictional 10-K fixture for tests, not a real filing"

FIXTURE_CIK = 9999999
FIXTURE_TICKER = "EXW"
FIXTURE_ACCESSION = "0009999999-25-000001"
FIXTURE_DOC = "exw-20250927.htm"

TOPIC_WORDS = {
    "export": "export controls licensing requirements semiconductor shipments abroad",
    "rates": "interest rate increases borrowing costs floating rate debt refinancing",
    "cyber": "cybersecurity breaches ransomware attacks unauthorized network intrusions",
    "currency": "foreign currency exchange rates translation of overseas revenue hedging",
    "supply": "supply chain disruptions single source suppliers component shortages",
}


def _para(topic: str, n: int) -> str:
    words = TOPIC_WORDS[topic]
    return (
        f"Synthetic paragraph {n} about {topic}. Our business could be affected by {words}. "
        f"This invented sentence number {n} exists only to give the fixture enough length "
        f"for the {topic} risk factor. Management of the fictional company monitors {words} "
        f"on a regular basis and may adjust its plans."
    )


def _bold(text: str) -> str:
    return f'<div><span style="font-weight:700">{text}</span></div>'


def _footer(page: int) -> str:
    return (
        f'<div style="text-align:center"><span>Example Widgets Corp | 2025 Form 10-K | '
        f"{page}</span></div><div><span>{page}</span></div>"
    )


def risk_factors_html(paras_per_topic: int = 4) -> str:
    """Item 1A body: one bold heading and several paragraphs per topic."""
    parts: list[str] = []
    page = 10
    for topic in ("export", "rates", "cyber", "supply"):
        parts.append(_bold(f"Risks related to {topic} could harm our results"))
        for i in range(paras_per_topic):
            parts.append(f"<div><span>{_para(topic, i)}</span></div>")
        parts.append(_footer(page))
        page += 1
    return "".join(parts)


def fixture_html(*, include_1a: bool = True, fiscal_year: int = 2025) -> str:
    """A full SYNTHETIC 10-K document."""
    toc_rows = "".join(
        f"<tr><td>Item {code}.</td><td>{title}</td><td>{page}</td></tr>"
        for code, title, page in (
            ("1", "Business", 3),
            ("1A", "Risk Factors", 10),
            ("1B", "Unresolved Staff Comments", 20),
            ("7", "Management's Discussion and Analysis", 25),
            ("7A", "Quantitative and Qualitative Disclosures About Market Risk", 30),
            ("8", "Financial Statements", 31),
        )
    )
    numeric_table = (
        "<table><tr><td>Net sales</td><td>1,234</td><td>5,678</td></tr>"
        "<tr><td>Cost</td><td>(901)</td><td>(234)</td></tr></table>"
    )
    item_1a = _bold("Item 1A. Risk Factors") + risk_factors_html() if include_1a else ""
    mdna = "".join(f"<p>{_para('currency', i)}</p>" for i in range(6))
    market = "".join(f"<p>{_para('rates', 10 + i)}</p>" for i in range(2))
    return f"""<?xml version="1.0" encoding="utf-8"?>
<html><head><title>exw-10k</title><style>.x{{color:red}}</style></head><body>
<div style="display:none"><ix:header><ix:hidden>
<ix:nonNumeric name="dei:DocumentFiscalYearFocus" contextRef="c1">{fiscal_year}</ix:nonNumeric>
<ix:nonNumeric name="dei:DocumentType">10-K</ix:nonNumeric>
</ix:hidden></ix:header></div>
<div><span>{SYNTHETIC_LABEL}</span></div>
<div><span>Table of Contents</span></div>
<table>{toc_rows}</table>
{_bold("Item 1. Business")}
<p>{_para("supply", 99)}</p>
{item_1a}
{_bold("Item 1B. Unresolved Staff Comments")}
<p>None.</p>
{_bold("Item 7. Management's Discussion and Analysis")}
{mdna}
{numeric_table}
{_footer(25)}
{_bold("Item 7A. Quantitative and Qualitative Disclosures About Market Risk")}
{market}
{_bold("Item 8. Financial Statements")}
<p>See the fictional financial statements.</p>
<script>var x = 1;</script>
</body></html>"""


def tickers_json() -> bytes:
    """SYNTHETIC ``company_tickers.json``."""
    data = {
        "0": {"cik_str": FIXTURE_CIK, "ticker": FIXTURE_TICKER, "title": "Example Widgets Corp"},
        "1": {"cik_str": 8888888, "ticker": "NOK", "title": "No Tenk Holdings"},
    }
    return json.dumps(data).encode()


def submissions_json(
    cik: int = FIXTURE_CIK, *, with_10k: bool = True, name: str = "EXAMPLE WIDGETS CORP"
) -> bytes:
    """SYNTHETIC ``submissions/CIK##########.json`` with a 10-K/A newer than the 10-K."""
    recent: dict[str, list[Any]] = {
        "accessionNumber": ["0009999999-25-000009", "0009999999-25-000005"],
        "filingDate": ["2025-12-01", "2025-11-15"],
        "reportDate": ["2025-09-27", "2025-11-14"],
        "form": ["10-K/A", "8-K"],
        "primaryDocument": ["exw-10ka.htm", "exw-8k.htm"],
    }
    if with_10k:
        for key, val in (
            ("accessionNumber", FIXTURE_ACCESSION),
            ("filingDate", "2025-10-31"),
            ("reportDate", "2025-09-27"),
            ("form", "10-K"),
            ("primaryDocument", FIXTURE_DOC),
        ):
            recent[key].append(val)
        # an older 10-K that must not be picked
        for key, val in (
            ("accessionNumber", "0009999999-24-000001"),
            ("filingDate", "2024-11-01"),
            ("reportDate", "2024-09-28"),
            ("form", "10-K"),
            ("primaryDocument", "exw-20240928.htm"),
        ):
            recent[key].append(val)
    return json.dumps({"cik": str(cik), "name": name, "filings": {"recent": recent}}).encode()
