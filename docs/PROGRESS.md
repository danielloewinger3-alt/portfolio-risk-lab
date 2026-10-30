# Progress log

Agents append one entry per completed milestone (newest at the bottom). Format:

```
## M<n>: <title>  (YYYY-MM-DD HH:MM)
- Built: ...
- Verified: ... (commands run, test counts)
- Surprises / notes: ...
```

## Morning handoff (finalized at M9)

_Pending._ Must list: what is done, what is blocked (link to BLOCKERS.md), exact commands the human must run, and decisions awaiting the human (O3 in particular).

---

## M0: Scaffold  (2026-10-30)
- Built: own git repo (`main`), uv project pinned to Python 3.12 with every runtime + dev dependency from `docs/02` locked in `uv.lock` (pandas 3.1, numpy 2.5, streamlit 1.66); ruff/mypy/pytest config with `.worktrees/` excluded; Makefile (+ `app-smoke`), CI workflow, MIT LICENSE, `.streamlit/config.toml` + `secrets.toml.example`; `risklab` package skeleton (`data/ risk/ ml/ rag/ llm/ viz/`) with matching `tests/unit/<area>/`; `config.py` (st.secrets > env > `.env` > defaults, `SecretStr` keys); `errors.py` typed errors; `app/_bootstrap.py`, `app/Home.py` placeholder, `app/components/layout.py` (page config + disclaimer footer).
- Verified: `make check` green (9 tests); `make app-smoke` returned health `ok`; no server left running.
- Surprises / notes:
  - An active conda base env (x86_64 Anaconda) wins uv's interpreter discovery, and onnxruntime (needed by fastembed) has no macOS x86_64 wheel. Fix: `[tool.uv] python-preference = "only-managed"`, so uv uses its native arm64 CPython 3.12.16.
  - LightGBM's macOS wheel needs Homebrew `libomp`, which isn't installed here. `make setup` runs `scripts/dev/macos_lightgbm_libomp.sh`, which adds a relative rpath in `.venv` so LightGBM uses scikit-learn's vendored `libomp`, then ad-hoc re-signs it. It does nothing on Linux (CI, Streamlit Cloud). Run it again after any `uv sync` that reinstalls lightgbm.
