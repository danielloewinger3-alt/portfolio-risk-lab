"""Chunker tests (SYNTHETIC text only)."""

from __future__ import annotations

import itertools
from dataclasses import fields

from hypothesis import given, settings
from hypothesis import strategies as st

from risklab.data.filings import Section
from risklab.rag.chunking import (
    MIN_CHARS,
    Chunk,
    chunk_section,
    chunk_text,
    is_heading_line,
    make_chunk_id,
    split_sentences,
)
from tests.fixtures.rag import synthetic_filing


def _section_and_chunks() -> tuple[Section, list[Chunk]]:
    parsed = synthetic_filing("EXW", ["export", "rates", "cyber", "supply"])
    sec = parsed.sections["1A"]
    return sec, chunk_section(sec, parsed.meta)


def test_no_empty_chunks_and_exact_slices() -> None:
    sec, chunks = _section_and_chunks()
    assert len(chunks) > 3
    for c in chunks:
        assert len(c.text) >= MIN_CHARS
        assert c.text == sec.text[c.char_start : c.char_end]
        assert c.text == c.text.strip()


def test_overlap_present_between_consecutive_chunks() -> None:
    _, chunks = _section_and_chunks()
    overlaps = [b.char_start < a.char_end for a, b in itertools.pairwise(chunks)]
    assert any(overlaps)
    for a, b in itertools.pairwise(chunks):
        assert b.char_start > a.char_start  # progress, never stuck
        assert b.char_end > a.char_end


def test_metadata_complete_and_ids_unique_and_stable() -> None:
    _sec, chunks = _section_and_chunks()
    for c in chunks:
        for fld in fields(Chunk):
            if fld.name != "score":
                assert getattr(c, fld.name) not in ("", None), fld.name
        assert c.chunk_id.startswith("EXW-2025-1A-")
        assert c.citation == "EXW 10-K FY2025, Item 1A"
    ids = [c.chunk_id for c in chunks]
    assert len(set(ids)) == len(ids)
    _, again = _section_and_chunks()
    assert [c.chunk_id for c in again] == ids
    assert [c.text for c in again] == [c.text for c in chunks]


def test_chunks_cover_the_whole_section() -> None:
    sec, chunks = _section_and_chunks()
    assert chunks[0].char_start == 0
    assert chunks[-1].char_end == len(sec.text)
    for a, b in itertools.pairwise(chunks):
        assert b.char_start <= a.char_end + 1  # no gap larger than a newline


def test_headings_assigned_and_chunks_do_not_end_on_heading() -> None:
    sec, chunks = _section_and_chunks()
    assert {c.heading for c in chunks} >= {"Risks related to rates could harm EXW"}
    for c in chunks:
        assert c.text.split("\n")[-1] not in sec.headings
        assert c.heading in sec.headings or c.heading.startswith("Item 1A")


def test_size_target_respected() -> None:
    _, chunks = _section_and_chunks()
    assert max(len(c.text) for c in chunks) <= 1500 * 1.5


def test_metadata_roundtrip() -> None:
    _, chunks = _section_and_chunks()
    c = chunks[0]
    back = Chunk.from_metadata(c.text, c.to_metadata(), score=0.5)
    assert back == c.with_score(0.5)
    assert "text" not in c.to_metadata()


def test_excerpt_cuts_at_word_boundary() -> None:
    _, chunks = _section_and_chunks()
    c = chunks[1]
    ex = c.excerpt(100)
    assert len(ex) <= 102 and ex.endswith("…")
    assert c.text.startswith(ex[:-2])
    short = Chunk.from_metadata("short text", c.to_metadata())
    assert short.excerpt() == "short text"


def test_split_sentences_respects_abbreviations() -> None:
    text = "Sales in the U.S. grew. Example Inc. reported results! Did it? Yes."
    spans = [text[s:e] for s, e in split_sentences(text)]
    assert spans == ["Sales in the U.S. grew.", "Example Inc. reported results!", "Did it?", "Yes."]


def test_very_long_sentence_is_hard_split() -> None:
    text = " ".join(["word"] * 1200)  # 5,999 chars, no sentence end
    spans = chunk_text(text, target_chars=1500)
    assert len(spans) >= 3
    # pieces are at most 1.3 x target; a tail under MIN_CHARS is merged into the last one
    assert all(e - s <= 1950 + MIN_CHARS for s, e, _ in spans)


def test_tiny_text_merges_into_single_chunk() -> None:
    spans = chunk_text("Overview\nOne short paragraph.")
    assert len(spans) == 1 and spans[0][2] == "Overview"


def test_empty_text_gives_no_chunks() -> None:
    assert chunk_text("") == []
    assert chunk_text("\n\n  \n") == []


def test_is_heading_line() -> None:
    assert is_heading_line("Risks Related to Our Industry")
    assert is_heading_line("Item 7A. Market Risk.")
    bold = "Bold sentence heading ending with a period."
    assert is_heading_line(bold, {bold})
    assert not is_heading_line("This is a normal sentence.")
    assert not is_heading_line("1,234")
    assert not is_heading_line("   ")


def test_make_chunk_id() -> None:
    assert make_chunk_id("nvda", 2025, "1a", 7) == "NVDA-2025-1A-0007"


@settings(max_examples=40, deadline=None)
@given(
    st.lists(
        st.lists(
            st.text(alphabet="abcdefghij ", min_size=5, max_size=120).map(
                lambda s: (s.strip() or "x").capitalize() + "."
            ),
            min_size=1,
            max_size=12,
        ).map(" ".join),
        min_size=1,
        max_size=15,
    )
)
def test_property_slices_cover_and_progress(paras: list[str]) -> None:
    text = "\n".join(paras)
    spans = chunk_text(text, target_chars=400, min_chars=50)
    assert spans
    assert spans[0][0] == 0 and spans[-1][1] == len(text.rstrip())
    for (s1, e1, _), (s2, e2, _) in itertools.pairwise(spans):
        assert s2 > s1 and e2 > e1 and s2 <= e1 + 1
