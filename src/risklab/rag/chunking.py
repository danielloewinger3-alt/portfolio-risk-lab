"""Paragraph-aware chunking of 10-K sections (``docs/06-rag-llm-spec.md`` §C).

Rules:

- Units are sentences, grouped by paragraph (one line of section text is one paragraph);
  a sentence longer than the chunk budget is split at word boundaries as a last resort.
- Units are packed greedily up to ``target_chars`` (about 1,500 characters, roughly 350
  tokens). Each new chunk starts with the trailing sentences of the previous one, up to
  ``overlap`` x ``target_chars`` characters (15%), so a fact on a boundary survives.
- A chunk never ends on a heading line; the heading moves to the next chunk.
- Chunks shorter than ``min_chars`` (200) are merged into a neighbor.
- ``text`` is always an exact slice ``section_text[char_start:char_end]``, so offsets and
  citations point back to the parsed filing.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, replace
from typing import Any, Final

from risklab.data.filings import FilingMeta, Section, filing_label

TARGET_CHARS: Final = 1_500
OVERLAP: Final = 0.15
MIN_CHARS: Final = 200
HEADING_MAX_CHARS: Final = 300
"""Stored ``heading`` metadata is truncated to this length."""

_SENT_END_RE: Final = re.compile(r"(?<=[.!?])[\"”’)]?\s+(?=[A-Z0-9“\"(•])")
_ABBREV_RE: Final = re.compile(
    r"(?:\b(?:U\.S|U\.K|Inc|Corp|Co|Ltd|No|Nos|Mr|Ms|Mrs|Dr|St|vs|etc|e\.g|i\.e|Jr|Sr|"
    r"approx|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)|\b[A-Z])\.[\"”’)]?\s*$"
)
_ITEM_HEADING_RE: Final = re.compile(r"^\s*items?\s*\d{1,2}[a-c]?\b", re.I)


@dataclass(frozen=True)
class Chunk:
    """A retrievable passage of a 10-K section with full citation metadata.

    ``score`` is filled in by retrieval (RRF score; 0.0 when not retrieved).
    """

    chunk_id: str  # TICKER-FY-ITEM-NNNN, e.g. "NVDA-2026-1A-0007"
    text: str
    ticker: str
    company: str
    form: str
    fiscal_year: int
    filing_date: str
    accession: str
    item: str
    heading: str
    source_url: str
    char_start: int
    char_end: int
    score: float = 0.0

    @property
    def citation(self) -> str:
        """Human-readable citation, e.g. ``"NVDA 10-K FY2026, Item 1A"``."""
        return filing_label(self.ticker, self.fiscal_year, self.item, self.form)

    def excerpt(self, max_chars: int = 600) -> str:
        """The passage cut at a word boundary to at most ``max_chars`` (UI preview)."""
        if len(self.text) <= max_chars:
            return self.text
        cut = self.text[:max_chars].rsplit(" ", 1)[0].rstrip()
        return cut + " …"

    def to_metadata(self) -> dict[str, str | int]:
        """Flat metadata for the vector store (everything except ``text`` and ``score``)."""
        d = asdict(self)
        d.pop("text")
        d.pop("score")
        return d

    @classmethod
    def from_metadata(cls, text: str, meta: Mapping[str, Any], score: float = 0.0) -> Chunk:
        """Rebuild a chunk from stored text plus :meth:`to_metadata` output."""
        return cls(
            chunk_id=str(meta["chunk_id"]),
            text=text,
            ticker=str(meta["ticker"]),
            company=str(meta["company"]),
            form=str(meta["form"]),
            fiscal_year=int(meta["fiscal_year"]),
            filing_date=str(meta["filing_date"]),
            accession=str(meta["accession"]),
            item=str(meta["item"]),
            heading=str(meta["heading"]),
            source_url=str(meta["source_url"]),
            char_start=int(meta["char_start"]),
            char_end=int(meta["char_end"]),
            score=score,
        )

    def with_score(self, score: float) -> Chunk:
        """Copy with a retrieval score attached."""
        return replace(self, score=score)


def make_chunk_id(ticker: str, fiscal_year: int, item: str, n: int) -> str:
    """``TICKER-FY-ITEM-NNNN`` (stable: depends only on filing identity and position)."""
    return f"{ticker.upper()}-{fiscal_year}-{item.upper()}-{n:04d}"


# --------------------------------------------------------------------------- splitting
@dataclass(frozen=True)
class _Unit:
    start: int
    end: int
    para: int  # paragraph index
    is_heading: bool


def split_sentences(text: str) -> list[tuple[int, int]]:
    """Sentence spans ``(start, end)`` in ``text`` (end excludes trailing whitespace).

    Splits after ``.``, ``!`` or ``?`` followed by whitespace and an uppercase letter,
    digit, quote, parenthesis or bullet; common abbreviations ("U.S.", "Inc.") and single
    initials do not end a sentence.
    """
    spans: list[tuple[int, int]] = []
    start = 0
    for m in _SENT_END_RE.finditer(text):
        end = m.start()
        # include a closing quote/paren that the lookbehind left in the match
        while end < m.end() and not text[end].isspace():
            end += 1
        if _ABBREV_RE.search(text[start:end]):
            continue
        if end > start:
            spans.append((start, end))
        start = m.end()
    if start < len(text):
        tail = text[start:].rstrip()
        if tail:
            spans.append((start, start + len(tail)))
    return spans


def _hard_split(start: int, end: int, text: str, limit: int) -> list[tuple[int, int]]:
    """Split ``text[start:end]`` at spaces into pieces of at most ``limit`` characters."""
    out: list[tuple[int, int]] = []
    s = start
    while end - s > limit:
        cut = text.rfind(" ", s, s + limit)
        if cut <= s:
            cut = s + limit
        out.append((s, cut))
        s = cut
        while s < end and text[s] == " ":
            s += 1
    if s < end:
        out.append((s, end))
    return out


def is_heading_line(line: str, headings: Iterable[str] = ()) -> bool:
    """A line is a heading if the source emphasized it (bold/italic), it is an "Item N"
    line, or it is short (<= 150 chars) and does not end like a sentence."""
    stripped = line.strip()
    if not stripped:
        return False
    if stripped in headings or _ITEM_HEADING_RE.match(stripped):
        return True
    return (
        len(stripped) <= 150
        and stripped[-1] not in '.;:,!?)”"'
        and stripped[0].isupper()
        and not stripped.replace(",", "").replace(".", "").isdigit()
    )


def _units(text: str, headings: frozenset[str], limit: int) -> list[_Unit]:
    units: list[_Unit] = []
    pos = 0
    for para, line in enumerate(text.split("\n")):
        if line.strip():
            heading = is_heading_line(line, headings)
            if heading and len(line) <= limit:
                units.append(_Unit(pos, pos + len(line.rstrip()), para, True))
            else:
                for s, e in split_sentences(line):
                    for hs, he in _hard_split(pos + s, pos + e, text, limit):
                        units.append(_Unit(hs, he, para, False))
        pos += len(line) + 1
    return units


def _pack(units: list[_Unit], target: int, overlap_chars: int) -> list[list[_Unit]]:
    """Greedy packing with sentence overlap; no chunk ends on a heading."""
    groups: list[list[_Unit]] = []
    current: list[_Unit] = []
    fresh = 0  # units in ``current`` that are not overlap carried from the previous chunk
    for u in units:
        if current and fresh and u.end - current[0].start > target:
            carry: list[_Unit] = []
            while len(current) > 1 and current[-1].is_heading:
                carry.insert(0, current.pop())
            groups.append(current)
            overlap: list[_Unit] = []
            if not carry:  # overlap only when the boundary is mid-content
                budget = overlap_chars
                for prev in reversed(current):
                    length = prev.end - prev.start
                    if prev.is_heading or (overlap and length > budget):
                        break
                    if not overlap and length > target // 2:
                        break
                    overlap.insert(0, prev)
                    budget -= length
            current = overlap + carry
            fresh = len(carry)
        current.append(u)
        fresh += 1
    if current and fresh:
        groups.append(current)
    return groups


def _merge_small(groups: list[list[_Unit]], min_chars: int) -> list[list[_Unit]]:
    """Merge groups spanning fewer than ``min_chars`` into a neighbor."""
    out: list[list[_Unit]] = []
    for g in groups:
        span = g[-1].end - g[0].start
        if out and span < min_chars:
            out[-1] = out[-1] + [u for u in g if u.start >= out[-1][-1].end]
        else:
            out.append(g)
    if len(out) > 1 and out[0][-1].end - out[0][0].start < min_chars:
        first = out.pop(0)
        out[0] = first + [u for u in out[0] if u.start >= first[-1].end]
    return out


def chunk_text(
    text: str,
    *,
    headings: Iterable[str] = (),
    target_chars: int = TARGET_CHARS,
    overlap: float = OVERLAP,
    min_chars: int = MIN_CHARS,
) -> list[tuple[int, int, str]]:
    """Chunk spans ``(char_start, char_end, heading)`` over ``text``.

    ``heading`` is the nearest heading line at or before the chunk start (or, for a chunk
    that precedes every heading, the first heading inside it; "" if there is none).
    """
    heads = frozenset(h.strip() for h in headings)
    limit = int(target_chars * 1.3)
    units = _units(text, heads, limit)
    if not units:
        return []
    groups = _merge_small(_pack(units, target_chars, int(target_chars * overlap)), min_chars)
    heading_units = [u for u in units if u.is_heading]
    out: list[tuple[int, int, str]] = []
    hi = -1
    for g in groups:
        while hi + 1 < len(heading_units) and heading_units[hi + 1].start <= g[0].start:
            hi += 1
        if hi >= 0:
            heading = text[heading_units[hi].start : heading_units[hi].end]
        else:
            heading = next((text[u.start : u.end] for u in g if u.is_heading), "")
        out.append((g[0].start, g[-1].end, heading[:HEADING_MAX_CHARS]))
    return out


def chunk_section(
    section: Section,
    meta: FilingMeta,
    *,
    target_chars: int = TARGET_CHARS,
    overlap: float = OVERLAP,
    min_chars: int = MIN_CHARS,
) -> list[Chunk]:
    """Chunk one extracted section into :class:`Chunk` objects with full metadata."""
    spans = chunk_text(
        section.text,
        headings=section.headings,
        target_chars=target_chars,
        overlap=overlap,
        min_chars=min_chars,
    )
    return [
        Chunk(
            chunk_id=make_chunk_id(meta.ticker, meta.fiscal_year, section.item, i),
            text=section.text[s:e],
            ticker=meta.ticker,
            company=meta.company,
            form=meta.form,
            fiscal_year=meta.fiscal_year,
            filing_date=meta.filing_date,
            accession=meta.accession,
            item=section.item,
            heading=heading,
            source_url=meta.source_url,
            char_start=s,
            char_end=e,
        )
        for i, (s, e, heading) in enumerate(spans, start=1)
    ]
