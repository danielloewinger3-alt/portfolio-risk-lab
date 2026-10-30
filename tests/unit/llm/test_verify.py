"""LLM safety suite, part 1: the memo and answer verifiers (docs/07 'LLM safety')."""

from __future__ import annotations

import json

import pytest

from risklab.llm.verify import (
    as_claims,
    as_text,
    number_supported,
    numbers_in,
    parse_json_object,
    scan_prose,
    verify_answer,
    verify_memo,
)
from tests.fixtures.llm import memo_json, small_report

IDS = ["AAA-2025-1A-0001", "BBB-2025-1A-0007"]


def kinds(result: object) -> list[str]:
    return [v.kind for v in result.violations]  # type: ignore[attr-defined]


def scan(text: str, ids: list[str] | None = None) -> list[str]:
    v, _, _ = scan_prose(text, small_report(), ids or IDS)
    return [x.kind for x in v]


# --------------------------------------------------------------------------- memo: pass
def test_clean_memo_verifies_and_counts_figures() -> None:
    res = verify_memo(memo_json(), small_report(), IDS)
    assert res.ok, res.violations
    assert res.figures_traced == 9
    assert res.facts_used[:3] == ["portfolio_name", "cagr", "ann_vol"]
    assert res.citations == ["AAA-2025-1A-0001", "BBB-2025-1A-0007"]
    assert res.unsupported_numbers == 0
    assert res.badge_text() == "Verified: 0 unsupported numbers, 9 figures traced to code"


@pytest.mark.parametrize(
    "text",
    [
        "VaR at 95% and 99% confidence",
        "the 95 % level and the 99th percentile",
        "a 1-day and a 10-day horizon, a 10 day VaR, the 10-trading-day view",
        "in the 2022 rate shock and the COVID crash of 2020",
        "the Classic 60/40 portfolio, a 60/40 mix",
        "see Item 1A and Item 7A of the 10-K; COVID-19 hit the S&P 500",
        "1. first point\n2) second point",
    ],
)
def test_allowlisted_literals_pass(text: str) -> None:
    assert scan(text) == []


# --------------------------------------------------------------------------- memo: fail
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("drawdown of 12.5% outside a placeholder", "stray_numeral"),
        ("a Sharpe ratio of 0.67", "stray_numeral"),
        ("losses of $1,200", "stray_numeral"),
        ("it lasted 10 days", "stray_numeral"),
        ("a 97% level", "stray_numeral"),
        ("the 2008 crisis", "stray_numeral"),  # 2008 is not in this report's literals
        ("a 5-day horizon", "stray_numeral"),
        ("[[fact:sharpe_ratio]]", "unknown_placeholder"),
        ("[[fact:Sharpe]]", "malformed_placeholder"),
        ("[[sharpe]]", "malformed_placeholder"),
        ("[[fact: sharpe]]", "malformed_placeholder"),
        ("[[cite:ZZZ-2025-1A-0009]]", "unknown_citation"),
        ("as ZZZ-2025-1A-0009 says", "unknown_citation"),
    ],
)
def test_violations_flagged(text: str, expected: str) -> None:
    assert expected in scan(text)


def test_placeholder_digits_and_valid_bare_citation_are_not_numerals() -> None:
    assert scan("value [[fact:var_95_1d_hist]] per AAA-2025-1A-0001") == []


def test_stray_numeral_records_token_and_snippet() -> None:
    v, _, _ = scan_prose("The Sharpe ratio was 0.67 last year.", small_report(), IDS, field="x")
    assert v[0].token == "0.67" and "Sharpe ratio was 0.67" in v[0].snippet and v[0].field == "x"
    assert "stray_numeral in x" in str(v[0])


def test_memo_flags_numbers_in_every_field_and_counts_them() -> None:
    raw = memo_json(
        tail_risk="VaR is 1.01% and ES 1.64%.",
        disclosed_risks=[{"text": "Sales of 1,250 million at risk", "chunk_ids": IDS[:1]}],
    )
    res = verify_memo(raw, small_report(), IDS)
    assert not res.ok
    assert res.unsupported_numbers == 3
    assert {v.field for v in res.violations} == {"tail_risk", "disclosed_risks[0]"}
    assert res.badge_text().startswith("Failed verification: 3 violations")


def test_memo_unknown_citation_in_list_and_missing_citation() -> None:
    raw = memo_json(
        disclosed_risks=[
            {"text": "Made-up risk", "chunk_ids": ["ZZZ-2025-1A-0001"]},
            {"text": "Uncited risk", "chunk_ids": []},
        ]
    )
    assert kinds(verify_memo(raw, small_report(), IDS)) == ["unknown_citation", "missing_citation"]


def test_memo_without_passages_rejects_any_citation() -> None:
    assert "unknown_citation" in kinds(verify_memo(memo_json(), small_report(), []))


def test_memo_missing_section_and_malformed_output() -> None:
    assert kinds(verify_memo(memo_json(summary=""), small_report(), IDS)) == ["missing_section"]
    res = verify_memo("The Sharpe ratio is high.", small_report(), IDS)
    assert kinds(res) == ["malformed_output"] and not res.ok
    assert kinds(verify_memo("[1, 2]", small_report(), IDS)) == ["malformed_output"]


def test_memo_word_count() -> None:
    res = verify_memo(json.dumps({"summary": "one two [[fact:cagr]]"}), small_report(), [])
    assert res.word_count == 3


# --------------------------------------------------------------------------- parsing
def test_parse_json_object_tolerates_fences_and_prose() -> None:
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object('Here you go: {"a": 1} thanks') == {"a": 1}
    with pytest.raises(ValueError, match="not valid JSON"):
        parse_json_object("{not json}")
    with pytest.raises(ValueError, match="not a JSON object"):
        parse_json_object("no braces")


def test_as_text_and_as_claims_normalize() -> None:
    assert as_text(None) == "" and as_text(["a", "b"]) == "a\nb" and as_text(3) == "3"
    claims = as_claims(["plain", {"text": "t", "chunk_id": "X"}, {"text": "u", "chunk_ids": 5}, 7])
    assert claims == [
        {"text": "plain", "chunk_ids": []},
        {"text": "t", "chunk_ids": ["X"]},
        {"text": "u", "chunk_ids": []},
    ]
    assert as_claims("nope") == []


# --------------------------------------------------------------------------- answers
PASSAGES = {
    "AAA-2025-1A-0001": "China represented 1,250 million dollars or 17% of revenue in 2025.",
    "BBB-2025-1A-0007": "A cyber incident could disrupt operations.",
}


def answer_json(
    answer: str = "Export controls matter.", claims: object = None, ie: bool = False
) -> str:
    if claims is None:
        claims = [{"text": "China was 17% of revenue.", "chunk_ids": ["AAA-2025-1A-0001"]}]
    return json.dumps({"answer": answer, "claims": claims, "insufficient_evidence": ie})


def test_numbers_helpers() -> None:
    assert numbers_in("1,250 million, 17% and 3.5x") == ["1250", "17", "3.5"]
    assert number_supported("1250", ["about 1,250 million"])
    assert number_supported("17", ["17% of revenue"])
    assert not number_supported("25", ["in 2025"])
    assert not number_supported("1.2", ["1.25"])


def test_verify_answer_accepts_grounded_claims() -> None:
    res = verify_answer(answer_json(), PASSAGES)
    assert res.ok and res.parsed and len(res.valid_claims) == 1
    res = verify_answer(answer_json(answer="Sales of 1,250 million were exposed."), PASSAGES)
    assert res.ok


def test_verify_answer_flags_each_failure() -> None:
    claims = [
        {"text": "No source.", "chunk_ids": []},
        {"text": "Invented source.", "chunk_ids": ["ZZZ-2025-1A-0001"]},
        {"text": "China was 25% of revenue.", "chunk_ids": ["AAA-2025-1A-0001"]},
        {"text": "Cyber hit 17% of revenue.", "chunk_ids": ["BBB-2025-1A-0007"]},
        {"text": "Cyber risk per BBB-2025-1A-0007.", "chunk_ids": []},
    ]
    res = verify_answer(answer_json(answer="Up 40% per ZZZ-2025-1A-0002.", claims=claims), PASSAGES)
    assert not res.ok
    assert [c.ok for c in res.claims] == [False, False, False, False, True]
    assert [v.kind for v in res.claims[0].violations] == ["missing_citation"]
    assert [v.kind for v in res.claims[1].violations] == ["unknown_citation"]
    assert [v.kind for v in res.claims[2].violations] == ["unsupported_number"]
    assert [v.kind for v in res.claims[3].violations] == ["unsupported_number"]
    assert {v.kind for v in res.answer_violations} == {"unknown_citation", "unsupported_number"}


def test_verify_answer_malformed_and_flag() -> None:
    res = verify_answer("not json", PASSAGES)
    assert not res.ok and not res.parsed and res.violations[0].kind == "malformed_output"
    res = verify_answer(answer_json(claims=[], ie=True), PASSAGES)
    assert res.ok and res.insufficient_evidence
