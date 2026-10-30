"""Tests for the validation tooling: coverage gates, the report renderer, and the regime
label-stability check (synthetic inputs only, labelled as such; no network)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import pytest

from risklab.config import PROJECT_ROOT

SCRIPTS = PROJECT_ROOT / "scripts"


def _load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


cov_mod = _load("check_coverage", SCRIPTS / "dev" / "check_coverage.py")
report = _load("make_validation_report", SCRIPTS / "make_validation_report.py")
regimes_script = _load("run_regimes", SCRIPTS / "run_regimes.py")


def _cov(files: dict[str, tuple[int, int]], total: float) -> dict[str, Any]:
    """A synthetic coverage.py JSON report: path -> (statements, missing)."""
    return {
        "files": {
            p: {"summary": {"num_statements": n, "missing_lines": m}} for p, (n, m) in files.items()
        },
        "totals": {"percent_covered": total},
    }


GOOD = {
    "src/risklab/risk/var.py": (100, 5),
    "src/risklab/risk/metrics.py": (100, 0),
    "src/risklab/ml/regimes.py": (50, 2),
    "src/risklab/llm/verify.py": (40, 1),
    "src/risklab/rag/chunking.py": (40, 4),
    "src/risklab/app_only.py": (10, 9),
}


def test_coverage_gates_pass_and_area_is_pooled() -> None:
    cov = _cov(GOOD, 90.0)
    assert cov_mod.area_coverage(cov, "src/risklab/risk/") == pytest.approx(97.5)
    assert cov_mod.check(cov) == []
    assert cov_mod.area_coverage(cov, "src/risklab/nothing/") is None


def test_coverage_gates_fail_on_area_and_overall() -> None:
    files = dict(GOOD)
    files["src/risklab/ml/regimes.py"] = (50, 20)  # 60%
    failures = cov_mod.check(_cov(files, 70.0))
    assert any(f.startswith("src/risklab/ml/") for f in failures)
    assert any(f.startswith("overall") for f in failures)


def test_coverage_main_exit_codes(tmp_path: Path) -> None:
    import json

    ok = tmp_path / "ok.json"
    ok.write_text(json.dumps(_cov(GOOD, 90.0)))
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(_cov(GOOD, 10.0)))
    assert cov_mod.main([str(ok)]) == 0
    assert cov_mod.main([str(bad)]) == 1


def _kupiec(reject: bool) -> dict[str, Any]:
    return {
        "stat": 5.0 if reject else 0.1,
        "pvalue": 0.02 if reject else 0.7,
        "reject_5pct": reject,
        "verdict": "x",
    }


def test_var_summary_counts_passes_by_confidence() -> None:
    bt = {
        "results": [
            {"method": "normal", "alpha": 0.95, "kupiec": _kupiec(False)},
            {"method": "normal", "alpha": 0.99, "kupiec": _kupiec(True)},
            {"method": "historical", "alpha": 0.99, "kupiec": _kupiec(False)},
        ]
    }
    s = report.var_summary(bt)
    assert s["normal"] == {"runs": 2, "pass": 1, "pass_95": 1, "pass_99": 0}
    assert s["historical"]["pass_99"] == 1


def test_render_committed_results_quotes_files() -> None:
    evals = report.load_all()
    assert evals["var"] and evals["vol"] and evals["rag"]
    tests = {
        "passed": 10,
        "failed": 0,
        "skipped": 1,
        "coverage": report.coverage_summary(_cov(GOOD, 90.0)),
    }
    md = report.render(evals, tests)
    assert md.startswith("# Validation report")
    summary = md.split("## 1.")[0]
    # Headline counts come from the committed var_backtests.json.
    s = report.var_summary(evals["var"])
    cf = s["cornish_fisher"]
    assert f"Cornish-Fisher passed {cf['pass']} of {cf['runs']} runs" in summary
    assert "GARCH-t had the lowest QLIKE" in summary
    assert "SYNTHETIC gold set" in summary
    assert "**Tests**: 10 passed, 0 failed" in summary
    for heading in (
        "## 1. VaR",
        "## 2. Volatility",
        "## 4. Filings",
        "## 5. LLM",
        "## 6. Tests",
        "## 7. Reproducibility",
    ):
        assert heading in md


def test_render_handles_missing_files() -> None:
    md = report.render({k: None for k in ("var", "vol", "regimes", "rag", "memo", "qa")})
    assert "Results file missing" in md
    assert "Not run for this report" in md


def test_formatters() -> None:
    assert report.pct(0.9615) == "96.2%"
    assert report.pct(None) == "n/a"
    assert report.num(1.23456, 2) == "1.23"
    assert report.table(["a", "b"], [[1, 2]]).splitlines()[2] == "| 1 | 2 |"


def test_label_stability_on_synthetic_two_regime_series() -> None:
    """SYNTHETIC returns: calm and turbulent blocks, so every seed finds the same states."""
    rng = np.random.default_rng(0)
    parts = [rng.normal(0, s, 150) for s in (0.004, 0.02, 0.004, 0.05, 0.004, 0.02)]
    idx = pd.bdate_range("2020-01-01", periods=900)
    r = pd.Series(np.concatenate(parts), index=idx)
    out = regimes_script.label_stability(r, seeds=(1, 2))
    assert out["reference_seed"] == 42
    assert [p["seed"] for p in out["per_seed"]] == [1, 2]
    assert 0.9 <= out["min_viterbi_agreement"] <= 1.0
    assert set(out["per_seed"][0]["ann_vol_by_state"]) == {"Calm", "Transition", "Stress"}


def test_deploy_requirements_are_in_sync() -> None:
    """``make export-reqs`` writes both files; Streamlit Cloud reads the one in app/."""
    root = (PROJECT_ROOT / "requirements.txt").read_text()
    assert root == (PROJECT_ROOT / "app" / "requirements.txt").read_text()
    pins = [line for line in root.splitlines() if line and not line.startswith(("#", " "))]
    names = {line.split("==")[0].lower() for line in pins}
    assert {"streamlit", "pandas", "chromadb", "fastembed", "anthropic"} <= names
    assert not names & {"pytest", "mypy", "ruff", "hypothesis"}  # no dev dependencies
