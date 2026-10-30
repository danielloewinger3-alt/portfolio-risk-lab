"""EDGAR ingestion tests: HTTP mocked with ``responses``, SYNTHETIC HTML fixtures only."""

from __future__ import annotations

import gzip
from collections.abc import Iterator

import pytest
import requests
import responses

from risklab.config import Settings
from risklab.data import filings as f
from risklab.errors import DataUnavailableError
from tests.fixtures import filings as fx

UA = "Test Agent test@example.com"
DOC_URL = f.ARCHIVE_URL.format(
    cik=fx.FIXTURE_CIK, acc=fx.FIXTURE_ACCESSION.replace("-", ""), doc=fx.FIXTURE_DOC
)
SUB_URL = f.SUBMISSIONS_URL.format(cik=fx.FIXTURE_CIK)


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[float]]:
    sleeps: list[float] = []
    monkeypatch.setattr(f, "_sleep", sleeps.append)
    f.reset_client()
    yield sleeps
    f.reset_client()


@pytest.fixture
def sec_settings(tmp_data_dir: Settings, monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setenv("SEC_USER_AGENT", UA)
    from risklab.config import get_settings

    get_settings.cache_clear()
    return get_settings()


def _mock_edgar(rsps: responses.RequestsMock, html: str | None = None) -> None:
    rsps.get(f.TICKERS_URL, body=fx.tickers_json())
    rsps.get(SUB_URL, body=fx.submissions_json())
    rsps.get(DOC_URL, body=(html or fx.fixture_html()).encode())


# --------------------------------------------------------------------------- HTML
def test_html_to_lines_drops_hidden_numeric_toc_and_footers() -> None:
    lines, emph = f.html_to_lines(fx.fixture_html())
    text = "\n".join(lines)
    assert "DocumentFiscalYearFocus" not in text and "dei:" not in text
    assert "var x" not in text and "color:red" not in text
    assert "1,234" not in text  # numeric table dropped
    assert "Table of Contents" not in lines
    assert not any(line.startswith("Example Widgets Corp | 2025 Form 10-K") for line in lines)
    assert not any(line.isdigit() for line in lines)  # page numbers removed
    assert "Item 1A. Risk Factors" in lines
    assert "Item 1A. Risk Factors" in emph
    assert "Risks related to export could harm our results" in emph
    assert all(line == line.strip() and "  " not in line for line in lines)


def test_html_to_lines_keeps_prose_tables_and_breaks() -> None:
    html = (
        "<html><body><table><tr><td>•</td><td>Our suppliers may fail to deliver parts on "
        "time.</td></tr></table><p>First line<br/>second line</p></body></html>"
    )
    lines, _ = f.html_to_lines(html)
    assert "• Our suppliers may fail to deliver parts on time." in lines
    assert lines[-2:] == ["First line", "second line"]


def test_emphasis_respects_normal_weight_override() -> None:
    html = (
        '<html><body><div style="font-weight:bold"><span style="font-weight:400">'
        "plain inside bold</span></div><p><i>Italic subheading</i></p></body></html>"
    )
    lines, emph = f.html_to_lines(html)
    assert "plain inside bold" in lines and "plain inside bold" not in emph
    assert "Italic subheading" in emph


def test_fiscal_year_from_html() -> None:
    assert f.fiscal_year_from_html(fx.fixture_html(fiscal_year=2026)) == 2026
    assert f.fiscal_year_from_html("<html><body>no tags</body></html>") is None


# --------------------------------------------------------------------------- sections
def test_extract_sections_skips_toc_and_bounds_items() -> None:
    lines, emph = f.html_to_lines(fx.fixture_html())
    secs = f.extract_sections(lines, emph)
    s1a = secs["1A"]
    assert s1a.quality == "ok"
    assert s1a.text.startswith("Item 1A. Risk Factors\n")
    assert "Unresolved Staff Comments" not in s1a.text  # ends before Item 1B
    assert "Risks related to supply could harm our results" in s1a.headings
    assert secs["7"].text.startswith("Item 7. Management")
    assert "Item 7A" not in secs["7"].text
    assert secs["7"].quality in {"ok", "short"}
    assert secs["7A"].quality == "short"  # two paragraphs only
    assert "Item 8" not in secs["7A"].text


def test_extract_sections_missing_item_is_failed() -> None:
    lines, emph = f.html_to_lines(fx.fixture_html(include_1a=False))
    secs = f.extract_sections(lines, emph)
    # only the table-of-contents line matches, which is far too short
    assert secs["1A"].quality == "failed"
    assert secs["1A"].text == ""
    assert f.extract_sections(["no items here"])["7"].quality == "failed"


def test_extract_sections_merges_repeated_running_headers() -> None:
    body = ["x" * 80 + f" sentence {i}." for i in range(60)]
    lines = ["Item 1A. Risk Factors", *body[:30], "Item 1A", *body[30:], "Item 1B. None"]
    sec = f.extract_sections(lines, items=("1A",))["1A"]
    assert sec.quality == "ok"
    assert sec.text.startswith("Item 1A. Risk Factors") and body[-1] in sec.text


def test_long_prose_starting_with_item_is_not_a_heading() -> None:
    long_line = "Item 7 of this report discusses " + "results " * 40
    assert f._item_headings(long_line) == []


def test_parse_filing_refines_fiscal_year_and_rejects_empty() -> None:
    meta = f._pick_latest_10k("EXW", fx.FIXTURE_CIK, "Example", _sub())
    parsed = f.parse_filing(fx.fixture_html(fiscal_year=2026), meta)
    assert parsed.meta.fiscal_year == 2026
    assert parsed.quality()["1A"] == "ok"
    with pytest.raises(f.FilingError):
        f.parse_filing("<html><body></body></html>", meta)


def _sub(**kw: object) -> dict[str, object]:
    import json

    out: dict[str, object] = json.loads(fx.submissions_json(**kw))  # type: ignore[arg-type]
    return out


def test_pick_latest_10k_ignores_amendments_and_older() -> None:
    meta = f._pick_latest_10k("EXW", fx.FIXTURE_CIK, "Example", _sub())
    assert meta.accession == fx.FIXTURE_ACCESSION
    assert meta.form == "10-K"
    assert meta.fiscal_year == 2025
    assert meta.source_url == DOC_URL
    with pytest.raises(DataUnavailableError):
        f._pick_latest_10k("EXW", 1, "x", _sub(with_10k=False))


def test_filing_label() -> None:
    assert f.filing_label("NVDA", 2025, "1A") == "NVDA 10-K FY2025, Item 1A"


# --------------------------------------------------------------------------- HTTP
@responses.activate
def test_client_sends_user_agent_and_retries_429(_no_sleep: list[float]) -> None:
    responses.get("https://example.test/a", status=429)
    responses.get("https://example.test/a", status=503)
    responses.get("https://example.test/a", body=b"ok")
    client = f.EdgarClient(UA, min_interval=0.0)
    assert client.get("https://example.test/a") == b"ok"
    assert all(c.request.headers["User-Agent"] == UA for c in responses.calls)
    assert _no_sleep == [1.0, 2.0]  # exponential backoff


@responses.activate
def test_client_404_and_403_fail_fast() -> None:
    responses.get("https://example.test/missing", status=404)
    responses.get("https://example.test/forbidden", status=403)
    client = f.EdgarClient(UA, min_interval=0.0)
    with pytest.raises(DataUnavailableError, match="404"):
        client.get("https://example.test/missing")
    with pytest.raises(DataUnavailableError, match="403"):
        client.get("https://example.test/forbidden")
    assert len(responses.calls) == 2


@responses.activate
def test_client_gives_up_after_max_attempts() -> None:
    responses.get("https://example.test/down", body=requests.ConnectionError("boom"))
    client = f.EdgarClient(UA, min_interval=0.0, max_attempts=3)
    with pytest.raises(DataUnavailableError, match="ConnectionError"):
        client.get("https://example.test/down")
    assert len(responses.calls) == 3


def test_client_throttles_to_min_interval(
    monkeypatch: pytest.MonkeyPatch, _no_sleep: list[float]
) -> None:
    clock = iter([0.0, 0.0, 0.05, 0.25])
    monkeypatch.setattr(f, "_monotonic", lambda: next(clock))
    client = f.EdgarClient(UA, min_interval=0.2)
    client._throttle()  # first call: no wait
    client._throttle()  # 0.05 s later: must wait 0.15 s
    assert _no_sleep == [pytest.approx(0.15)]


def test_client_requires_user_agent() -> None:
    with pytest.raises(DataUnavailableError, match="SEC_USER_AGENT"):
        f.EdgarClient("  ")


# --------------------------------------------------------------------------- end to end
@responses.activate
def test_load_filing_end_to_end_and_cached(sec_settings: Settings) -> None:
    _mock_edgar(responses)
    parsed = f.load_filing("exw", settings=sec_settings)
    assert parsed.meta.ticker == "EXW"
    assert parsed.meta.company == "EXAMPLE WIDGETS CORP"
    assert parsed.quality()["1A"] == "ok"
    raw = sec_settings.filings_dir / "raw" / "EXW" / f"{fx.FIXTURE_ACCESSION}.htm.gz"
    assert gzip.decompress(raw.read_bytes()).startswith(b"<?xml")
    n = len(responses.calls)
    again = f.load_filing("EXW", settings=sec_settings)  # everything from disk now
    assert len(responses.calls) == n
    assert again == parsed


@responses.activate
def test_offline_mode_uses_cache_only(
    sec_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_edgar(responses)
    f.load_filing("EXW", settings=sec_settings)
    off = sec_settings.model_copy(update={"offline": True})
    n = len(responses.calls)
    assert f.load_filing("EXW", settings=off).meta.accession == fx.FIXTURE_ACCESSION
    assert len(responses.calls) == n
    with pytest.raises(DataUnavailableError, match="offline"):
        f.ticker_to_cik("EXW", settings=off.model_copy(update={"data_dir": off.cache_dir}))


@responses.activate
def test_stale_cache_used_when_network_fails(sec_settings: Settings) -> None:
    _mock_edgar(responses)
    f.latest_10k("EXW", settings=sec_settings)
    path = sec_settings.filings_dir / "submissions" / f"CIK{fx.FIXTURE_CIK:010d}.json"
    import os

    os.utime(path, (0, 0))  # make it stale
    responses.replace(responses.GET, SUB_URL, status=500)
    meta = f.latest_10k("EXW", settings=sec_settings)
    assert meta.accession == fx.FIXTURE_ACCESSION


@responses.activate
def test_predecessor_cik_fallback(sec_settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(f.PREDECESSOR_CIKS, "NOK", (fx.FIXTURE_CIK,))
    responses.get(f.TICKERS_URL, body=fx.tickers_json())
    responses.get(
        f.SUBMISSIONS_URL.format(cik=8888888),
        body=fx.submissions_json(8888888, with_10k=False, name="NO TENK HOLDINGS"),
    )
    responses.get(SUB_URL, body=fx.submissions_json())
    meta = f.latest_10k("NOK", settings=sec_settings)
    assert meta.cik == fx.FIXTURE_CIK and meta.ticker == "NOK"
    assert meta.company == "EXAMPLE WIDGETS CORP"


@responses.activate
def test_unknown_ticker_and_no_10k(sec_settings: Settings) -> None:
    responses.get(f.TICKERS_URL, body=fx.tickers_json())
    responses.get(
        f.SUBMISSIONS_URL.format(cik=8888888), body=fx.submissions_json(8888888, with_10k=False)
    )
    with pytest.raises(DataUnavailableError, match="not found"):
        f.ticker_to_cik("ZZZZ", settings=sec_settings)
    with pytest.raises(DataUnavailableError, match="no 10-K"):
        f.latest_10k("NOK", settings=sec_settings)


@responses.activate
def test_stale_parser_version_reparses(
    sec_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_edgar(responses)
    f.load_filing("EXW", settings=sec_settings)
    monkeypatch.setattr(f, "PARSER_VERSION", f.PARSER_VERSION + 1)
    parsed = f.load_filing("EXW", settings=sec_settings)  # re-parse from cached raw HTML
    assert parsed.quality()["1A"] == "ok"


@pytest.mark.network
def test_live_edgar_latest_10k() -> None:  # pragma: no cover - live only
    meta = f.latest_10k("AAPL")
    assert meta.form == "10-K" and meta.cik == 320193
