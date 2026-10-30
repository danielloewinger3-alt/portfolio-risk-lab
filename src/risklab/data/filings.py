"""SEC EDGAR 10-K ingestion: fetch, parse, and split into Items 1A, 7 and 7A.

Pipeline (``docs/06-rag-llm-spec.md`` §B):

1. Ticker to CIK via ``company_tickers.json``.
2. Latest ``form == "10-K"`` from ``data.sec.gov/submissions/CIK##########.json``.
3. Download the primary document from ``www.sec.gov/Archives/edgar/data/...``.
4. HTML to text with BeautifulSoup + lxml: hidden inline-XBRL blocks, scripts and
   numeric tables are dropped; page numbers, "Table of Contents" links and short lines that
   repeat on every page (running headers/footers) are removed.
5. Split on "Item N" heading lines. The table of contents matches too, so for each item we
   take the **last** heading that yields a plausible section (>= ``MIN_OK_CHARS``).

SEC etiquette: every request sends ``SEC_USER_AGENT``, requests are spaced so there are at
most 5 per second, 429/5xx responses back off exponentially, and every response is cached
under ``data/filings/`` (gitignored). Raw documents are stored gzip-compressed.

Nothing here invents filing text: sections are exact slices of the parsed document, and a
section that cannot be found is reported as ``failed`` rather than guessed.
"""

from __future__ import annotations

import gzip
import json
import logging
import re
import threading
import time
import warnings
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Final, Literal

import requests
from bs4 import BeautifulSoup, Comment, NavigableString, Tag, XMLParsedAsHTMLWarning

from risklab.config import Settings, get_settings
from risklab.errors import DataUnavailableError, RiskLabError

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- constants
TICKERS_URL: Final = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL: Final = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVE_URL: Final = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}"

TARGET_ITEMS: Final[tuple[str, ...]] = ("1A", "7", "7A")
"""Items indexed for RAG: Risk Factors, MD&A, Quantitative and Qualitative Market Risk."""

ITEM_TITLES: Final[dict[str, str]] = {
    "1A": "Risk Factors",
    "7": "Management's Discussion and Analysis",
    "7A": "Quantitative and Qualitative Disclosures About Market Risk",
}

MIN_OK_CHARS: Final = 3_000
"""A section at least this long is ``ok`` (the spec's plausibility threshold)."""
MIN_SHORT_CHARS: Final = 300
"""Below this a section counts as ``failed``; between the two it is ``short``."""
MAX_HEADING_LINE: Final = 200
"""Item heading lines are short; longer lines that start with "Item 7" are prose."""

MIN_REQUEST_INTERVAL: Final = 0.21
"""Seconds between EDGAR requests (just under 5 requests/second)."""
MAX_ATTEMPTS: Final = 5
SUBMISSIONS_TTL_DAYS: Final = 7
PARSER_VERSION: Final = 4
"""Bump when parsing changes so cached sections are rebuilt."""

SectionQuality = Literal["ok", "short", "failed"]

# Indirection points so tests can skip real sleeps and control time.
_sleep: Callable[[float], None] = time.sleep
_monotonic: Callable[[], float] = time.monotonic


class FilingError(RiskLabError):
    """A filing was fetched but could not be parsed into usable sections."""


# --------------------------------------------------------------------------- data types
@dataclass(frozen=True)
class FilingMeta:
    """Identity of one 10-K filing (all fields come from EDGAR, none are inferred)."""

    ticker: str
    cik: int
    company: str
    form: str
    accession: str  # with dashes, e.g. "0000320193-25-000079"
    filing_date: str  # ISO date
    report_date: str  # ISO date (period of report), may be ""
    fiscal_year: int  # dei:DocumentFiscalYearFocus when present, else report_date year
    primary_document: str
    source_url: str


@dataclass(frozen=True)
class Section:
    """One extracted 10-K item: exact text slice of the parsed document."""

    item: str
    title: str
    text: str
    quality: SectionQuality
    headings: tuple[str, ...] = ()
    """Lines inside this section rendered in bold/italic in the source (sub-headings)."""

    @property
    def n_chars(self) -> int:
        """Length of the section text in characters."""
        return len(self.text)


@dataclass(frozen=True)
class ParsedFiling:
    """A filing's metadata plus its target sections (missing items are ``failed``)."""

    meta: FilingMeta
    sections: dict[str, Section] = field(default_factory=dict)

    def quality(self) -> dict[str, SectionQuality]:
        """Section quality per target item (``failed`` when absent)."""
        return {
            item: self.sections[item].quality if item in self.sections else "failed"
            for item in TARGET_ITEMS
        }


# --------------------------------------------------------------------------- HTTP client
class EdgarClient:
    """Rate-limited, retrying, User-Agent-sending EDGAR HTTP client.

    One instance is shared per process (:func:`get_client`) so the 5 requests/second budget
    holds across callers. The User-Agent is never logged.
    """

    def __init__(
        self,
        user_agent: str,
        *,
        min_interval: float = MIN_REQUEST_INTERVAL,
        max_attempts: int = MAX_ATTEMPTS,
        timeout: float = 30.0,
    ) -> None:
        if not user_agent or not user_agent.strip():
            raise DataUnavailableError(
                "SEC_USER_AGENT is not set; EDGAR requires a 'Name email' User-Agent"
            )
        self._session = requests.Session()
        self._session.headers.update(
            {"User-Agent": user_agent.strip(), "Accept-Encoding": "gzip, deflate"}
        )
        self._min_interval = min_interval
        self._max_attempts = max_attempts
        self._timeout = timeout
        self._lock = threading.Lock()
        self._last = -1e9
        self.n_requests = 0

    def _throttle(self) -> None:
        with self._lock:
            wait = self._min_interval - (_monotonic() - self._last)
            if wait > 0:
                _sleep(wait)
            self._last = _monotonic()

    def get(self, url: str) -> bytes:
        """GET ``url`` and return the body; backs off on 429/5xx and network errors."""
        last_err = ""
        for attempt in range(self._max_attempts):
            self._throttle()
            self.n_requests += 1
            try:
                resp = self._session.get(url, timeout=self._timeout)
            except requests.RequestException as exc:
                last_err = f"{type(exc).__name__}"
            else:
                if resp.status_code == 200:
                    return resp.content
                if resp.status_code == 404:
                    raise DataUnavailableError(f"EDGAR 404 for {url}")
                last_err = f"HTTP {resp.status_code}"
                if resp.status_code != 429 and resp.status_code < 500:
                    raise DataUnavailableError(f"EDGAR {last_err} for {url}")
            if attempt < self._max_attempts - 1:
                backoff = 2.0**attempt
                logger.warning("edgar: %s for %s, retrying in %.0fs", last_err, url, backoff)
                _sleep(backoff)
        raise DataUnavailableError(f"EDGAR request failed after retries ({last_err}): {url}")


_CLIENT: EdgarClient | None = None
_CLIENT_LOCK = threading.Lock()


def get_client(settings: Settings | None = None) -> EdgarClient:
    """Process-wide :class:`EdgarClient` built from ``SEC_USER_AGENT``."""
    global _CLIENT
    with _CLIENT_LOCK:
        if _CLIENT is None:
            s = settings or get_settings()
            _CLIENT = EdgarClient(s.sec_user_agent or "")
        return _CLIENT


def reset_client() -> None:
    """Drop the shared client (tests)."""
    global _CLIENT
    with _CLIENT_LOCK:
        _CLIENT = None


# --------------------------------------------------------------------------- disk cache
def _fresh(path: Path, max_age_days: float | None) -> bool:
    if not path.is_file():
        return False
    if max_age_days is None:
        return True
    age = time.time() - path.stat().st_mtime
    return age < max_age_days * 86_400


def _cached_get(
    url: str,
    path: Path,
    settings: Settings,
    *,
    max_age_days: float | None = None,
    compress: bool = False,
) -> bytes:
    """Return the cached body at ``path`` if fresh, else fetch ``url`` and cache it.

    In offline mode only the cache is used (stale is acceptable). On a network failure a
    stale cache entry is used if one exists.
    """

    def read() -> bytes:
        raw = path.read_bytes()
        return gzip.decompress(raw) if compress else raw

    if _fresh(path, max_age_days):
        return read()
    if settings.offline:
        if path.is_file():
            return read()
        raise DataUnavailableError(f"offline mode and no cached copy of {url}")
    try:
        body = get_client(settings).get(url)
    except DataUnavailableError:
        if path.is_file():
            logger.warning("edgar: using stale cache for %s", url)
            return read()
        raise
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(gzip.compress(body) if compress else body)
    tmp.replace(path)
    return body


# --------------------------------------------------------------------------- EDGAR lookups
def ticker_to_cik(ticker: str, *, settings: Settings | None = None) -> tuple[int, str]:
    """Map ``ticker`` to ``(cik, company title)`` using SEC's ``company_tickers.json``."""
    s = settings or get_settings()
    body = _cached_get(TICKERS_URL, s.filings_dir / "company_tickers.json", s, max_age_days=30)
    table: dict[str, dict[str, Any]] = json.loads(body)
    want = ticker.upper().replace(".", "-")
    for row in table.values():
        if str(row.get("ticker", "")).upper() == want:
            return int(row["cik_str"]), str(row.get("title", ""))
    raise DataUnavailableError(f"ticker {ticker} not found in SEC company_tickers.json")


PREDECESSOR_CIKS: Final[dict[str, tuple[int, ...]]] = {
    # ExxonMobil Holdings Corp (CIK 2115436) became the XOM registrant by 8-K12B on
    # 2026-07-01 and had filed no 10-K yet (checked 2026-10-30). The latest XOM 10-K
    # (FY2025, filed 2026-02-18) is under the predecessor Exxon Mobil Corp, CIK 34088.
    "XOM": (34088,),
}
"""Registrants to try, in order, when the CIK SEC maps a ticker to has no 10-K yet
(holding-company reorganizations). Each entry is verified against EDGAR, not guessed."""


def _submissions(cik: int, settings: Settings) -> dict[str, Any]:
    body = _cached_get(
        SUBMISSIONS_URL.format(cik=cik),
        settings.filings_dir / "submissions" / f"CIK{cik:010d}.json",
        settings,
        max_age_days=SUBMISSIONS_TTL_DAYS,
    )
    out: dict[str, Any] = json.loads(body)
    return out


def latest_10k(ticker: str, *, settings: Settings | None = None) -> FilingMeta:
    """Metadata of the most recent ``10-K`` (amendments excluded) for ``ticker``.

    Tries the CIK from ``company_tickers.json`` first, then any :data:`PREDECESSOR_CIKS`.
    ``fiscal_year`` here is the report-date year; :func:`parse_filing` refines it from
    the document's ``dei:DocumentFiscalYearFocus`` tag.
    """
    s = settings or get_settings()
    cik, title = ticker_to_cik(ticker, settings=s)
    tried: list[str] = []
    for c in (cik, *PREDECESSOR_CIKS.get(ticker.upper(), ())):
        sub = _submissions(c, s)
        try:
            return _pick_latest_10k(ticker.upper(), c, str(sub.get("name") or title), sub)
        except DataUnavailableError:
            tried.append(str(c))
    raise DataUnavailableError(
        f"no 10-K in recent EDGAR submissions for {ticker} (CIKs tried: {', '.join(tried)})"
    )


def _pick_latest_10k(ticker: str, cik: int, company: str, sub: dict[str, Any]) -> FilingMeta:
    recent = sub.get("filings", {}).get("recent", {})
    forms: list[str] = recent.get("form", [])
    best: int | None = None
    for i, form in enumerate(forms):
        if form != "10-K":
            continue
        if best is None or recent["filingDate"][i] > recent["filingDate"][best]:
            best = i
    if best is None:
        raise DataUnavailableError(f"no 10-K in recent EDGAR submissions for {ticker}")
    accession = str(recent["accessionNumber"][best])
    doc = str(recent["primaryDocument"][best])
    report_date = str(recent.get("reportDate", [""] * len(forms))[best] or "")
    filing_date = str(recent["filingDate"][best])
    fy = int((report_date or filing_date)[:4])
    return FilingMeta(
        ticker=ticker,
        cik=cik,
        company=company,
        form="10-K",
        accession=accession,
        filing_date=filing_date,
        report_date=report_date,
        fiscal_year=fy,
        primary_document=doc,
        source_url=ARCHIVE_URL.format(cik=cik, acc=accession.replace("-", ""), doc=doc),
    )


def fetch_filing_html(meta: FilingMeta, *, settings: Settings | None = None) -> str:
    """Primary 10-K document as text (cached gzip-compressed, never re-downloaded)."""
    s = settings or get_settings()
    path = s.filings_dir / "raw" / meta.ticker / f"{meta.accession}.htm.gz"
    body = _cached_get(meta.source_url, path, s, compress=True)
    return body.decode("utf-8", errors="replace")


# --------------------------------------------------------------------------- HTML to text
_BLOCK_TAGS: Final = frozenset(
    {
        "p", "div", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "table",
        "section", "article", "body", "center", "blockquote", "dl", "dt", "dd", "hr",
        "header", "footer", "pre", "html", "tbody", "thead",
    }
)  # fmt: skip
_SKIP_TAGS: Final = frozenset({"script", "style", "head", "title", "noscript", "ix:header"})
_CELL_TAGS: Final = frozenset({"td", "th"})
_BOLD_RE: Final = re.compile(r"font-weight\s*:\s*(bold|bolder|[6-9]00)", re.I)
_NOT_BOLD_RE: Final = re.compile(r"font-weight\s*:\s*(normal|lighter|[1-5]00)", re.I)
_ITALIC_RE: Final = re.compile(r"font-style\s*:\s*italic", re.I)
_HIDDEN_RE: Final = re.compile(r"display\s*:\s*none", re.I)
_WS_RE: Final = re.compile(r"\s+")
_ZERO_WIDTH_RE: Final = re.compile("[​‌‍﻿]")
_PAGE_NUM_RE: Final = re.compile(r"^(page\s+)?[-–]?\s*([0-9]{1,3}|[ivxlc]{1,6})\s*[-–]?$", re.I)
_TOC_RE: Final = re.compile(r"^(table of contents|index|back to contents)$", re.I)
_FY_RE: Final = re.compile(
    r'name="dei:DocumentFiscalYearFocus"[^>]*>(?:\s*<[^>]+>)*\s*(\d{4})', re.I
)
_DIGITS_RE: Final = re.compile(r"\d+")
_MIN_TABLE_ALPHA: Final = 0.85
_REPEAT_MIN: Final = 5
_REPEAT_MAX_LEN: Final = 100
_HEADING_MAX_LEN: Final = 400


@dataclass
class _Block:
    parts: list[str] = field(default_factory=list)
    emph_chars: int = 0
    total_chars: int = 0

    def add(self, text: str, emph: bool) -> None:
        self.parts.append(text)
        n = sum(1 for ch in text if not ch.isspace())
        self.total_chars += n
        if emph:
            self.emph_chars += n


def _style_flags(tag: Tag, emph: bool) -> tuple[bool, bool]:
    """Return ``(hidden, emphasized)`` for ``tag`` given the inherited emphasis."""
    style = str(tag.get("style") or "")
    if style and _HIDDEN_RE.search(style):
        return True, emph
    if tag.name in {"b", "strong"} or (style and _BOLD_RE.search(style)):
        emph = True
    elif style and _NOT_BOLD_RE.search(style) and not _ITALIC_RE.search(style):
        emph = False
    if tag.name in {"i", "em"} or (style and _ITALIC_RE.search(style)):
        emph = True
    return False, emph


def _table_is_prose(table: Tag) -> bool:
    """Keep tables whose characters are overwhelmingly letters (layout tables, headings);
    drop numeric tables (financial statements) and page-numbered tables of contents."""
    text = table.get_text(" ")
    letters = sum(ch.isalpha() for ch in text)
    digits = sum(ch.isdigit() for ch in text)
    if letters == 0:
        return False
    return letters / (letters + digits) >= _MIN_TABLE_ALPHA


def _iter_blocks(root: Tag) -> Iterator[_Block]:
    """Depth-first walk yielding text blocks (paragraph-level units) with emphasis counts.

    Iterative (explicit stack) so deeply nested inline-XBRL markup cannot hit the
    recursion limit.
    """
    current = _Block()
    # Stack entries: (node, emphasized) or a sentinel string "BREAK" / "SPACE".
    stack: list[tuple[Any, bool]] = [(root, False)]
    while stack:
        node, emph = stack.pop()
        if node == "BREAK":
            if current.parts:
                yield current
                current = _Block()
            continue
        if node == "SPACE":
            current.add(" ", False)
            continue
        if isinstance(node, Comment):
            continue
        if isinstance(node, NavigableString):
            text = str(node)
            if text:
                current.add(text, emph)
            continue
        if not isinstance(node, Tag):
            continue
        name = (node.name or "").lower()
        if name in _SKIP_TAGS or name.startswith("ix:header"):
            continue
        if name == "br":
            stack.append(("BREAK", False))
            continue
        hidden, child_emph = _style_flags(node, emph)
        if hidden:
            continue
        if name == "table" and not _table_is_prose(node):
            stack.append(("BREAK", False))
            continue
        is_block = name in _BLOCK_TAGS
        trailer = "BREAK" if is_block else ("SPACE" if name in _CELL_TAGS else None)
        if trailer:
            stack.append((trailer, False))
        for child in reversed(list(node.children)):
            stack.append((child, child_emph))
        if is_block:
            stack.append(("BREAK", False))
    if current.parts:
        yield current


def html_to_lines(html: str) -> tuple[list[str], set[str]]:
    """Convert 10-K HTML to normalized text lines plus the set of emphasized lines.

    Each line is one paragraph-level block with whitespace collapsed. Emphasized lines
    (>= 90% bold or italic characters, at most 400 characters) are candidate sub-headings.
    Page numbers, "Table of Contents" links and running headers/footers (short lines
    repeated at least five times) are removed; "Item N" heading lines are always kept.
    """
    with warnings.catch_warnings():  # inline XBRL starts with an XML declaration
        warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
        soup = BeautifulSoup(html, "lxml")
    root = soup.body or soup
    lines: list[str] = []
    emphasized: set[str] = set()
    for block in _iter_blocks(root):
        text = _WS_RE.sub(" ", _ZERO_WIDTH_RE.sub("", "".join(block.parts))).strip()
        if not text:
            continue
        lines.append(text)
        if (
            block.total_chars
            and block.emph_chars / block.total_chars >= 0.9
            and len(text) <= _HEADING_MAX_LEN
        ):
            emphasized.add(text)
    counts = Counter(_repeat_key(line) for line in lines if len(line) <= _REPEAT_MAX_LEN)
    exact = Counter(lines)
    kept: list[str] = []
    for line in lines:
        if _PAGE_NUM_RE.match(line) or _TOC_RE.match(line):
            continue
        if _ITEM_LINE_RE.match(line):
            # Item headings are kept, except a bare "Item 1A" repeated verbatim on every
            # page (a running header); real headings occur at most about three times.
            if _BARE_ITEM_RE.match(line) and exact[line] >= _REPEAT_MIN:
                continue
        elif len(line) <= _REPEAT_MAX_LEN and counts[_repeat_key(line)] >= _REPEAT_MIN:
            continue
        kept.append(line)
    return kept, emphasized


def _repeat_key(line: str) -> str:
    """Line with digit runs masked, so "Form 10-K | 27" and "Form 10-K | 28" count as one
    running footer."""
    return _DIGITS_RE.sub("#", line.lower())


def fiscal_year_from_html(html: str) -> int | None:
    """``dei:DocumentFiscalYearFocus`` from inline XBRL, if tagged."""
    m = _FY_RE.search(html)
    return int(m.group(1)) if m else None


# --------------------------------------------------------------------------- sections
_ITEM_LINE_RE: Final = re.compile(r"^\s*items?\s*(\d{1,2}[a-c]?)\b", re.I)
_BARE_ITEM_RE: Final = re.compile(r"^\s*items?\s*\d{1,2}[a-c]?\s*[.:]?\s*$", re.I)


@dataclass(frozen=True)
class _Heading:
    code: str
    start: int  # char offset of the line start in the joined text


def _item_headings(text: str) -> list[_Heading]:
    """Every short line that starts with "Item <code>", in document order."""
    out: list[_Heading] = []
    pos = 0
    for line in text.split("\n"):
        if len(line) <= MAX_HEADING_LINE:
            m = _ITEM_LINE_RE.match(line)
            if m:
                out.append(_Heading(m.group(1).upper(), pos))
        pos += len(line) + 1
    return out


def _quality(n_chars: int) -> SectionQuality:
    if n_chars >= MIN_OK_CHARS:
        return "ok"
    if n_chars >= MIN_SHORT_CHARS:
        return "short"
    return "failed"


def extract_sections(
    lines: list[str],
    emphasized: set[str] | None = None,
    items: tuple[str, ...] = TARGET_ITEMS,
) -> dict[str, Section]:
    """Split document lines into the requested items.

    For each item, every heading that starts a run (consecutive repeats of the same item
    heading, e.g. running page headers, are merged) is a candidate; the section runs to
    the next heading of a *different* item. The last candidate of at least
    ``MIN_OK_CHARS`` wins (the body comes after the table of contents); if none is that
    long, the longest candidate is kept and graded ``short`` or ``failed``.
    """
    emphasized = emphasized or set()
    text = "\n".join(lines)
    heads = _item_headings(text)
    out: dict[str, Section] = {}
    for item in items:
        candidates: list[tuple[int, int]] = []
        for i, h in enumerate(heads):
            if h.code != item or (i > 0 and heads[i - 1].code == item):
                continue
            end = len(text)
            for nxt in heads[i + 1 :]:
                if nxt.code != item:
                    end = nxt.start
                    break
            candidates.append((h.start, end))
        if not candidates:
            out[item] = Section(item, ITEM_TITLES.get(item, ""), "", "failed")
            continue
        plausible = [c for c in candidates if c[1] - c[0] >= MIN_OK_CHARS]
        start, end = plausible[-1] if plausible else max(candidates, key=lambda c: c[1] - c[0])
        body = text[start:end].strip()
        quality = _quality(len(body))
        heads_in = tuple(dict.fromkeys(ln for ln in body.split("\n") if ln in emphasized))
        out[item] = Section(
            item=item,
            title=ITEM_TITLES.get(item, ""),
            text=body if quality != "failed" else "",
            quality=quality,
            headings=heads_in,
        )
    return out


def parse_filing(html: str, meta: FilingMeta) -> ParsedFiling:
    """Parse a fetched document into a :class:`ParsedFiling` (pure, no network)."""
    fy = fiscal_year_from_html(html)
    if fy is not None and fy != meta.fiscal_year:
        meta = FilingMeta(**{**asdict(meta), "fiscal_year": fy})
    lines, emphasized = html_to_lines(html)
    if not lines:
        raise FilingError(f"{meta.ticker}: document produced no text")
    return ParsedFiling(meta=meta, sections=extract_sections(lines, emphasized))


# --------------------------------------------------------------------------- top level
def _sections_cache_path(meta: FilingMeta, settings: Settings) -> Path:
    return settings.filings_dir / "sections" / meta.ticker / f"{meta.accession}.json"


def _save_parsed(parsed: ParsedFiling, path: Path) -> None:
    payload = {
        "parser_version": PARSER_VERSION,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "meta": asdict(parsed.meta),
        "sections": {
            k: {**asdict(v), "headings": list(v.headings)} for k, v in parsed.sections.items()
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False))
    tmp.replace(path)


def _load_parsed(path: Path) -> ParsedFiling | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text())
        if payload.get("parser_version") != PARSER_VERSION:
            return None
        meta = FilingMeta(**payload["meta"])
        sections = {
            k: Section(**{**v, "headings": tuple(v.get("headings", ()))})
            for k, v in payload["sections"].items()
        }
    except (ValueError, KeyError, TypeError) as exc:
        logger.warning("ignoring unreadable sections cache %s: %s", path, exc)
        return None
    return ParsedFiling(meta=meta, sections=sections)


def load_filing(ticker: str, *, settings: Settings | None = None) -> ParsedFiling:
    """Latest 10-K for ``ticker``, parsed into sections (every step cached on disk).

    Raises :class:`DataUnavailableError` when EDGAR cannot be reached and nothing is
    cached, and :class:`FilingError` when the document has no text.
    """
    s = settings or get_settings()
    meta = latest_10k(ticker, settings=s)
    cache = _sections_cache_path(meta, s)
    cached = _load_parsed(cache)
    if cached is not None:
        return cached
    html = fetch_filing_html(meta, settings=s)
    parsed = parse_filing(html, meta)
    _save_parsed(parsed, cache)
    q = parsed.quality()
    logger.info(
        "filings: %s %s FY%s sections %s",
        meta.ticker,
        meta.accession,
        parsed.meta.fiscal_year,
        ", ".join(f"{k}={v}({parsed.sections[k].n_chars})" for k, v in q.items()),
    )
    return parsed


def filing_label(ticker: str, fiscal_year: int, item: str, form: str = "10-K") -> str:
    """Citation label, e.g. ``"NVDA 10-K FY2025, Item 1A"``."""
    return f"{ticker} {form} FY{fiscal_year}, Item {item}"


def today_iso() -> str:
    """Local calendar date as ISO string (indirection for tests)."""
    return date.today().isoformat()
