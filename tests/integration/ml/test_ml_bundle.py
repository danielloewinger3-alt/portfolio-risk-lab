"""Integration: regimes and the volatility bake-off run offline on the real bundled data."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from risklab.config import get_settings
from risklab.data.prices import bundle_snapshot_date, load_prices
from risklab.ml.regimes import analyze_regimes
from risklab.ml.volatility import market_features, walk_forward_vol
from risklab.portfolio import load_portfolio_data, sample_portfolio

pytestmark = pytest.mark.skipif(
    not get_settings().sample_prices_path.is_file(),
    reason="bundled snapshot not built (run `make data`)",
)

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "run_walkforward.py"


def _data():  # type: ignore[no-untyped-def]
    end = bundle_snapshot_date()
    assert end is not None
    p = sample_portfolio("Classic 60/40", start=end.replace(year=end.year - 10), end=end)
    return p, load_portfolio_data(p, offline=True)


def test_regimes_on_bundle() -> None:
    _, d = _data()
    res = analyze_regimes(
        d.portfolio_returns, d.benchmark_returns, asset_returns=d.asset_returns, n_init=3
    )
    assert res.model.labels == ("Calm", "Transition", "Stress")
    assert res.stress_vol_multiple > 1.0
    assert res.summary["share_of_time"].sum() == pytest.approx(1.0)
    assert (res.summary["es"] >= res.summary["var"]).all()


def test_walk_forward_on_bundle_is_clean() -> None:
    p, d = _data()
    vix = load_prices(["^VIX"], p.start, p.end, offline=True)["^VIX"]
    ev = walk_forward_vol(
        d.portfolio_returns, extra_features=market_features(d.benchmark_returns, vix)
    )
    assert ev.dropped_features == []
    assert ev.max_purge_violation() <= 0
    assert ev.forecasts.notna().all().all()
    assert len(ev.forecasts) > 300


@pytest.mark.slow
def test_script_writes_results(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location("run_walkforward", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main(["--out-dir", str(tmp_path)]) == 0
    payload = json.loads((tmp_path / "volatility_results.json").read_text())
    assert len(payload["portfolios"]) == 4
    for res in payload["portfolios"]:
        assert res["leakage_guard"]["clean"]
        assert res["winner"] == res["metrics"][0]["model"]
    assert "# Volatility forecast bake-off" in (tmp_path / "volatility_results.md").read_text()
