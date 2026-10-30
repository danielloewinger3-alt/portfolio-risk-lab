# 06. RAG and LLM specification

## Purpose

1. **Filings Q&A:** let a user ask questions about the qualitative risks that their holdings disclose, with verifiable citations.
2. **Risk memo:** a readable summary of the computed risk analytics, where the LLM explains but never calculates.

Both degrade gracefully with no API key: Q&A shows retrieved passages only; the memo uses a deterministic template.

## A. Corpus

- Source: SEC EDGAR, **latest 10-K per company** in the demo universe (`11-data-spec.md`)
- Sections: **Item 1A Risk Factors**, **Item 7 MD&A**, **Item 7A Market Risk**
- ETFs and non-filers have no corpus; the UI says "no filings indexed for X" and excludes them from Q&A
- Raw HTML cached under `data/filings/` (gitignored). Parsed sections and the built index are reproducible by `scripts/build_index.py`; the prebuilt Chroma index is committed

## B. Ingestion

1. Ticker to CIK via `https://www.sec.gov/files/company_tickers.json`
2. Latest 10-K via `https://data.sec.gov/submissions/CIK{cik:010d}.json`: pick the most recent `form == "10-K"`, get `accessionNumber` and `primaryDocument`
3. Download `https://www.sec.gov/Archives/edgar/data/{cik}/{accession_no_dashes}/{primaryDocument}`
4. Parse with BeautifulSoup + lxml, normalize whitespace, drop tables of contents and page headers/footers where detectable
5. Section split by regex on headings ("Item 1A", "Item 7", "Item 7A", case-insensitive, with optional punctuation). The table of contents also matches: take the **last** match that yields a plausible section length (e.g. > 3,000 characters), not the first
6. Store `section_quality` metadata (ok / short / failed). Filings whose Item 1A fails extraction are logged and skipped, never silently indexed as garbage. Verify extraction on at least 5 different filings during development and record results in `docs/eval/`

SEC etiquette: `User-Agent` from `SEC_USER_AGENT`, at most 5 requests per second, exponential backoff on 429/503, cache every response.

## C. Chunking

- Split on paragraph boundaries; target about 1,500 characters per chunk (roughly 350 tokens), 15% overlap, never split mid-sentence if avoidable
- Metadata per chunk: `chunk_id` (`TICKER-FY-ITEM-NNNN`), `ticker`, `company`, `form`, `fiscal_year`, `filing_date`, `accession`, `item`, `heading` (nearest preceding heading), `source_url`, `char_start`, `char_end`
- No empty or near-empty chunks (< 200 characters merged into neighbors)

## D. Embedding and retrieval

- Embeddings: `fastembed` `BAAI/bge-small-en-v1.5`; queries use the bge instruction prefix `"Represent this sentence for searching relevant passages: "`, documents none
- Store: ChromaDB `PersistentClient` at `data/index/chroma`, cosine distance, metadata as above
- Retrieval (`retrieve()`):
  1. Dense top-20 filtered by `ticker in portfolio tickers`
  2. BM25 (`rank-bm25`) top-20 over the same filtered chunk set
  3. Reciprocal Rank Fusion, `score = Σ 1 / (60 + rank)`
  4. Select top `k = 6` with a **per-ticker quota** (default 2) so one holding cannot dominate, then fill remaining slots by score
- Cache embeddings of the sample questions to avoid model load on cold start

## E. Answer generation (Q&A)

System prompt rules (store in `llm/prompts.py`):
- Answer **only** from the provided passages; if they do not support an answer, say so
- Every claim must cite one or more `chunk_id`s
- Do not give investment advice; do not predict prices
- Output JSON: `{"answer": str, "claims": [{"text": str, "chunk_ids": [str]}], "insufficient_evidence": bool}`

Post-processing (deterministic): every cited `chunk_id` must exist in the retrieved set, otherwise the claim is dropped and flagged. UI renders claims with citation chips (`NVDA 10-K FY2025, Item 1A`) that expand to the source passage (show at most about 600 characters with a link to the SEC filing).

## F. Risk memo with deterministic number substitution

This is the key anti-hallucination design. **The LLM never writes a number.**

1. `build_report()` produces a `RiskReport`: a flat set of named facts such as `sharpe`, `max_drawdown`, `var_95_1d_hist`, `top_risk_contributor`, `top_risk_contributor_pct`, `regime_current`, `garch_beats_lgbm`, each with `value`, a formatted `display` string, and a unit
2. The LLM receives the fact names and display strings **as context** plus up to 6 retrieved filing passages for qualitative color, and is instructed to reference figures **only** as placeholders `[[fact:<name>]]`
3. The renderer replaces each placeholder with the fact's `display` string. Unknown placeholder names are an error
4. **Verifier** (`llm/verify.py`) scans the raw LLM output **before** substitution:
   - any digit sequence outside a placeholder that is not on the allowlist is a violation. Allowlist: confidence levels (95%, 99%), horizons (1-day, 10-day), four-digit years that appear in the report's `allowed_literals`, scenario names, and numbered list markers
   - unknown placeholders are violations
   - citations to chunk ids not in the provided set are violations
5. On violation: retry once with the violations appended to the prompt; if still failing, fall back to the **template memo** (pure string formatting of the same facts, no LLM)
6. Verification outcome is shown in the UI ("Verified: 0 unsupported numbers, 14 figures traced to code") and stored in the memo object

**Memo structure (fixed sections):** Summary, Return and risk profile, Tail risk, Stress and regimes, Disclosed risks from filings (cited), Caveats and limitations. Length cap about 450 words. The caveats section is mandatory and partly templated.

**Caching and limits:** cache memo by hash of `(report JSON, retrieved chunk ids, model)`. Pre-generate and commit memos for the 4 sample portfolios (`data/sample/memos/*.json`). Per-session LLM call cap from `RISKLAB_LLM_CALL_LIMIT`.

## G. LLM client

`llm/client.py` wraps the `anthropic` SDK behind the `LLMClient` protocol: model from `RISKLAB_LLM_MODEL`, bounded `max_tokens`, request timeout, retries with backoff on transient errors, no secrets in logs. If the API rejects an optional parameter for a given model, drop it and retry rather than failing. A `FakeLLMClient` (scripted responses) lives in `tests/` for all unit tests.

## H. Evaluation (scripts, not vibes)

- **Retrieval eval (`scripts/eval_rag.py`):** build a gold set of about 25 questions across at least 8 tickers, each with the expected `chunk_id`s. Because humans are not available overnight, gold questions may be generated from sampled chunks by an LLM (store prompts and outputs), and the result is labeled **synthetic gold set** in `VALIDATION.md`. Report hit@6 and MRR for dense-only vs hybrid
- **Citation validity:** 100% of rendered citations must resolve to retrieved chunks (enforced and measured)
- **Faithfulness sample:** write 10 sampled (question, answer, cited passage) triples to `docs/eval/rag_review.md` for the human to spot check in the morning
- **Memo verification:** across the 4 sample portfolios, report violations found on first pass, retry rate, and fallback rate

## I. P2 extensions

- **Risk theme map (F12):** fixed taxonomy of about 10 themes (China/geopolitics, rates and inflation, supply chain, regulation/antitrust, cybersecurity, customer concentration, FX, energy and commodities, litigation, AI competition). For each holding and theme, score = max cosine similarity of the theme query to that holding's chunks, displayed as a **relative** rank (not a probability). Pure embeddings, no LLM
- **NL stress scenarios (F13):** tool-call schema `{shocks: [{factor: Enum[equity, rates, oil, usd, credit], pct: float}]}`; the model only fills the schema from the user's text. Factors map to ETF proxies with betas estimated by code. The user confirms the parsed shocks before they run
