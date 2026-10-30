"""Deterministic verification of LLM output (``docs/06-rag-llm-spec.md`` §E-F).

Pure functions, no network, no Streamlit.

**Memo** (:func:`verify_memo`) scans the raw model output *before* substitution. The
output is a JSON object whose prose fields may contain only:

- fact placeholders ``[[fact:<name>]]`` whose name is in the :class:`RiskReport`;
- citations ``[[cite:<chunk_id>]]`` (or a bare chunk id) from the provided passages;
- allowlisted literals: confidence levels (``95%``, ``99%``), horizons (``1-day``,
  ``10-day``), numbered list markers, filing vocabulary (``10-K``, ``Item 1A``), the
  report's ``allowed_literals`` (scenario names, scenario years, the portfolio name) and
  a few proper nouns in :data:`PROPER_NOUNS`.

Any other digit is a violation, as is an unknown placeholder, a malformed ``[[...]]``
token, a citation to a chunk id that was not provided, unparseable JSON, or a missing
required section.

**Q&A** (:func:`verify_answer`): every claim must cite at least one retrieved chunk id,
and every number in a claim must appear in the passages it cites (numbers in the answer
lead must appear in some retrieved passage). Q&A answers have no report facts, so a
figure is acceptable only when it is quoted from the source text.
"""

from __future__ import annotations

import json
import re
from collections.abc import Collection, Iterable, Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, Field

from risklab.report import RiskReport

ViolationKind = Literal[
    "malformed_output",
    "missing_section",
    "stray_numeral",
    "unknown_placeholder",
    "malformed_placeholder",
    "unknown_citation",
    "missing_citation",
    "unsupported_number",
]

MEMO_SECTIONS: tuple[str, ...] = (
    "summary",
    "return_and_risk",
    "tail_risk",
    "stress_and_regimes",
    "disclosed_risks",
    "caveats",
)
"""JSON keys of a memo draft, in display order."""

REQUIRED_PROSE_SECTIONS: tuple[str, ...] = (
    "summary",
    "return_and_risk",
    "tail_risk",
    "stress_and_regimes",
)
"""Prose sections that must be non-empty (``caveats`` is templated, ``disclosed_risks``
may be empty when no filings are indexed)."""

PLACEHOLDER_RE = re.compile(r"\[\[fact:([a-z][a-z0-9_]*)\]\]")
"""The exact placeholder grammar the renderer substitutes."""
CITE_RE = re.compile(r"\[\[cite:([A-Za-z0-9.\-]+)\]\]")
"""Inline citation grammar."""
BRACKET_TOKEN_RE = re.compile(r"\[\[[^\[\]]{0,200}\]\]")
"""Anything that looks like a placeholder; checked against the two grammars above."""
CHUNK_ID_RE = re.compile(r"\b[A-Z][A-Z0-9.]{0,9}-\d{4}-(?:1A|7A|7)-\d{4}\b")
"""Bare chunk ids (``TICKER-FY-ITEM-NNNN``), treated as citations."""
DIGIT_RUN_RE = re.compile(r"[$€£]?\d(?:[\d,.]*\d)?%?")
NUMBER_RE = re.compile(r"\d(?:[\d,]*\d)?(?:\.\d+)?")

PROPER_NOUNS: tuple[str, ...] = ("COVID-19", "S&P 500", "Form 10-K", "10-K", "10-Q", "8-K")
"""Names that contain digits but are not figures."""

_ALLOWLIST_PATTERNS: tuple[re.Pattern[str], ...] = (
    # confidence levels: 95%, 99%, 95 %, 95th percentile, 99 percent
    re.compile(r"(?<![\d.,])(?:95|99)(?:\.0+)?\s?%"),
    re.compile(r"(?<![\d.,])(?:95|99)(?:th)?[\s-]?(?:percent|per cent|percentile)\b", re.I),
    # horizons: 1-day, 10-day, 10 day, 10-trading-day (singular only: "10 days" is a count)
    re.compile(r"(?<![\d.,])(?:1|10)[\s-](?:trading[\s-])?day\b", re.I),
    # filing sections
    re.compile(r"\bItems?\s+(?:1A|7A|7)\b", re.I),
    # numbered list markers at the start of a line
    re.compile(r"(?m)^\s*\d{1,2}[.)](?=\s)"),
)


class Violation(BaseModel):
    """One reason an output failed verification."""

    kind: ViolationKind
    detail: str
    field: str = ""
    snippet: str = ""
    token: str = ""

    def __str__(self) -> str:
        where = f" in {self.field}" if self.field else ""
        snip = f": ...{self.snippet}..." if self.snippet else ""
        return f"{self.kind}{where}: {self.detail}{snip}"


class VerificationResult(BaseModel):
    """Outcome of :func:`verify_memo` (stored on the memo and shown as a UI badge).

    Attributes:
        ok: no violations.
        violations: every violation found.
        figures_traced: placeholder occurrences that resolve to report facts (each will
            be substituted by a code-computed display string).
        facts_used: distinct fact names referenced.
        citations: distinct valid chunk ids cited.
        unsupported_numbers: count of ``stray_numeral`` violations.
        word_count: words in the prose fields (placeholders count as one word).
    """

    ok: bool
    violations: list[Violation] = Field(default_factory=list)
    figures_traced: int = 0
    facts_used: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    unsupported_numbers: int = 0
    word_count: int = 0

    def badge_text(self) -> str:
        """``"Verified: 0 unsupported numbers, 14 figures traced to code"`` (or a failure)."""
        if self.ok:
            return f"Verified: 0 unsupported numbers, {self.figures_traced} figures traced to code"
        return (
            f"Failed verification: {len(self.violations)} violations "
            f"({self.unsupported_numbers} unsupported numbers)"
        )


# --------------------------------------------------------------------------- parsing
def parse_json_object(raw: str) -> dict[str, Any]:
    """Parse the first JSON object in ``raw`` (tolerates code fences and leading prose).

    Raises:
        ValueError: no JSON object could be parsed.
    """
    text = raw.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.S)
    if fence:
        text = fence.group(1)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("output is not a JSON object") from None
        try:
            obj = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError(f"output is not valid JSON ({exc.msg})") from exc
    if not isinstance(obj, dict):
        raise ValueError("output JSON is not an object")
    return obj


def as_text(value: Any) -> str:
    """Coerce a JSON value to prose (lists of strings are joined by newlines)."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(as_text(v) for v in value)
    return str(value)


def as_claims(value: Any) -> list[dict[str, Any]]:
    """Normalize a claims list: each item ``{"text": str, "chunk_ids": [str]}``.

    Accepts a plain string in place of an object (no citations) and a single string in
    place of the id list.
    """
    out: list[dict[str, Any]] = []
    if not isinstance(value, list):
        return out
    for item in value:
        if isinstance(item, str):
            out.append({"text": item, "chunk_ids": []})
        elif isinstance(item, Mapping):
            ids = item.get("chunk_ids", item.get("chunk_id", []))
            if isinstance(ids, str):
                ids = [ids]
            if not isinstance(ids, list):
                ids = []
            out.append({"text": as_text(item.get("text", "")), "chunk_ids": [str(i) for i in ids]})
    return out


# --------------------------------------------------------------------------- memo scan
def _snippet(text: str, start: int, end: int, pad: int = 30) -> str:
    return text[max(0, start - pad) : end + pad].replace("\n", " ").strip()


def _blank(text: str, start: int, end: int) -> str:
    return text[:start] + " " * (end - start) + text[end:]


def _strip_literals(text: str, literals: Iterable[str]) -> str:
    for lit in sorted({x for x in literals if x}, key=len, reverse=True):
        text = re.sub(re.escape(lit), lambda m: " " * len(m.group(0)), text, flags=re.I)
    return text


def scan_prose(
    text: str,
    report: RiskReport,
    chunk_ids: Collection[str],
    *,
    field: str = "",
) -> tuple[list[Violation], list[str], list[str]]:
    """Scan one prose field. Returns ``(violations, facts referenced, chunk ids cited)``.

    ``facts referenced`` has one entry per valid placeholder occurrence.
    """
    violations: list[Violation] = []
    facts: list[str] = []
    cites: list[str] = []
    work = text
    for m in BRACKET_TOKEN_RE.finditer(text):
        token = m.group(0)
        fact = PLACEHOLDER_RE.fullmatch(token)
        cite = CITE_RE.fullmatch(token)
        if fact:
            name = fact.group(1)
            if name in report:
                facts.append(name)
            else:
                violations.append(
                    Violation(kind="unknown_placeholder", detail=name, field=field, snippet=token)
                )
        elif cite:
            cid = cite.group(1)
            if cid in chunk_ids:
                cites.append(cid)
            else:
                violations.append(
                    Violation(kind="unknown_citation", detail=cid, field=field, snippet=token)
                )
        else:
            violations.append(
                Violation(
                    kind="malformed_placeholder",
                    detail="use [[fact:<name>]] or [[cite:<chunk_id>]]",
                    field=field,
                    snippet=token,
                )
            )
        work = _blank(work, m.start(), m.end())
    for m in CHUNK_ID_RE.finditer(work):
        cid = m.group(0)
        if cid in chunk_ids:
            cites.append(cid)
        else:
            violations.append(
                Violation(kind="unknown_citation", detail=cid, field=field, snippet=cid)
            )
        work = _blank(work, m.start(), m.end())
    work = _strip_literals(work, [*report.allowed_literals, *PROPER_NOUNS])
    for pat in _ALLOWLIST_PATTERNS:
        work = pat.sub(lambda m: " " * len(m.group(0)), work)
    for m in DIGIT_RUN_RE.finditer(work):
        violations.append(
            Violation(
                kind="stray_numeral",
                detail=f"number {m.group(0)!r} outside a placeholder",
                field=field,
                snippet=_snippet(text, m.start(), m.end()),
                token=m.group(0),
            )
        )
    return violations, facts, cites


def _word_count(text: str) -> int:
    return len(re.sub(r"\[\[[^\]]*\]\]", "X", text).split())


def verify_memo(
    raw: str,
    report: RiskReport,
    chunk_ids: Collection[str] = (),
) -> VerificationResult:
    """Verify a raw memo draft (JSON, before substitution) against ``report``.

    Args:
        raw: model output; a JSON object with the keys in :data:`MEMO_SECTIONS`, where
            ``disclosed_risks`` is a list of ``{"text", "chunk_ids"}``.
        report: the facts the memo may reference.
        chunk_ids: ids of the passages given to the model (valid citation targets).
    """
    allowed = set(chunk_ids)
    try:
        obj = parse_json_object(raw)
    except ValueError as exc:
        return VerificationResult(
            ok=False, violations=[Violation(kind="malformed_output", detail=str(exc))]
        )
    violations: list[Violation] = []
    facts: list[str] = []
    cites: list[str] = []
    words = 0
    for key in MEMO_SECTIONS:
        if key == "disclosed_risks":
            continue
        text = as_text(obj.get(key))
        if key in REQUIRED_PROSE_SECTIONS and not text.strip():
            violations.append(
                Violation(kind="missing_section", detail=f"section {key!r} is empty", field=key)
            )
        v, f, c = scan_prose(text, report, allowed, field=key)
        violations += v
        facts += f
        cites += c
        words += _word_count(text)
    for i, claim in enumerate(as_claims(obj.get("disclosed_risks"))):
        field = f"disclosed_risks[{i}]"
        v, f, c = scan_prose(claim["text"], report, allowed, field=field)
        violations += v
        facts += f
        cites += c
        words += _word_count(claim["text"])
        for cid in claim["chunk_ids"]:
            if cid in allowed:
                cites.append(cid)
            else:
                violations.append(
                    Violation(kind="unknown_citation", detail=cid, field=field, snippet=cid)
                )
        if not claim["chunk_ids"] and not c:
            violations.append(
                Violation(
                    kind="missing_citation",
                    detail="disclosed risk without a chunk id",
                    field=field,
                    snippet=claim["text"][:60],
                )
            )
    return VerificationResult(
        ok=not violations,
        violations=violations,
        figures_traced=len(facts),
        facts_used=list(dict.fromkeys(facts)),
        citations=list(dict.fromkeys(cites)),
        unsupported_numbers=sum(v.kind == "stray_numeral" for v in violations),
        word_count=words,
    )


# --------------------------------------------------------------------------- Q&A
def numbers_in(text: str) -> list[str]:
    """Numeric tokens in ``text`` with thousands separators removed (``"1,200.5"``)."""
    return [m.group(0).replace(",", "") for m in NUMBER_RE.finditer(text)]


def number_supported(number: str, sources: Sequence[str]) -> bool:
    """True when ``number`` (commas removed) appears as a whole number in some source."""
    pat = re.compile(rf"(?<![\d.]){re.escape(number)}(?!\d)(?!\.\d)")
    return any(pat.search(re.sub(r"(?<=\d),(?=\d{3})", "", s)) for s in sources)


class ClaimCheck(BaseModel):
    """Verification of one Q&A claim."""

    text: str
    chunk_ids: list[str]
    ok: bool
    violations: list[Violation] = Field(default_factory=list)


class AnswerVerification(BaseModel):
    """Outcome of :func:`verify_answer`.

    Attributes:
        ok: parsed, and no violations anywhere.
        parsed: the output was a JSON object.
        answer: the answer lead text (``""`` if missing).
        insufficient_evidence: the model's own flag.
        claims: per-claim checks, in output order.
        answer_violations: violations in the lead text (unsupported numbers, ids).
        violations: every violation (output, lead, and claims).
    """

    ok: bool
    parsed: bool
    answer: str = ""
    insufficient_evidence: bool = False
    claims: list[ClaimCheck] = Field(default_factory=list)
    answer_violations: list[Violation] = Field(default_factory=list)
    violations: list[Violation] = Field(default_factory=list)

    @property
    def valid_claims(self) -> list[ClaimCheck]:
        """Claims that passed every check."""
        return [c for c in self.claims if c.ok]


def verify_answer(raw: str, passages: Mapping[str, str]) -> AnswerVerification:
    """Verify a raw Q&A answer against the retrieved passages (``chunk_id -> text``).

    Checks per claim: at least one chunk id, every id retrieved, every number present in
    one of the cited passages. In the answer lead: every number present in some
    retrieved passage, and any bare chunk id retrieved.
    """
    try:
        obj = parse_json_object(raw)
    except ValueError as exc:
        v = Violation(kind="malformed_output", detail=str(exc))
        return AnswerVerification(ok=False, parsed=False, violations=[v])
    lead = as_text(obj.get("answer"))
    lead_violations: list[Violation] = []
    for cid in CHUNK_ID_RE.findall(lead):
        if cid not in passages:
            lead_violations.append(
                Violation(kind="unknown_citation", detail=cid, field="answer", snippet=cid)
            )
    lead_scan = CHUNK_ID_RE.sub(" ", BRACKET_TOKEN_RE.sub(" ", lead))
    for num in numbers_in(lead_scan):
        if not number_supported(num, list(passages.values())):
            lead_violations.append(
                Violation(
                    kind="unsupported_number",
                    detail=f"{num!r} is not in any retrieved passage",
                    field="answer",
                )
            )
    checks: list[ClaimCheck] = []
    for i, claim in enumerate(as_claims(obj.get("claims"))):
        field = f"claims[{i}]"
        ids = [*claim["chunk_ids"], *CHUNK_ID_RE.findall(claim["text"])]
        ids = list(dict.fromkeys(ids))
        cv: list[Violation] = []
        if not ids:
            cv.append(
                Violation(kind="missing_citation", detail="claim cites no chunk", field=field)
            )
        for cid in ids:
            if cid not in passages:
                cv.append(Violation(kind="unknown_citation", detail=cid, field=field, snippet=cid))
        cited = [passages[c] for c in ids if c in passages]
        text_scan = CHUNK_ID_RE.sub(" ", BRACKET_TOKEN_RE.sub(" ", claim["text"]))
        for num in numbers_in(text_scan):
            if not number_supported(num, cited):
                cv.append(
                    Violation(
                        kind="unsupported_number",
                        detail=f"{num!r} is not in the cited passages",
                        field=field,
                    )
                )
        checks.append(ClaimCheck(text=claim["text"], chunk_ids=ids, ok=not cv, violations=cv))
    all_v = [*lead_violations, *(v for c in checks for v in c.violations)]
    return AnswerVerification(
        ok=not all_v,
        parsed=True,
        answer=lead,
        insufficient_evidence=bool(obj.get("insufficient_evidence", False)),
        claims=checks,
        answer_violations=lead_violations,
        violations=all_v,
    )
