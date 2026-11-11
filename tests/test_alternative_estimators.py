"""Tests for the alternative estimators: Bradley--Terry PPI++, penalized human fits, and their wiring
into the real-data cells (on a hand-built frame, never on the shipped data)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from dial_judge.baselines import fit_human_only_btl
from dial_judge.penalized import fit_boxed_btl, fit_ridge_btl, fit_ridge_btl_fixed
from dial_judge.ppi import fit_bt_ppi, ppi_lambda
from experiments.real_data import alternative_estimators as rm
from experiments.real_data import robustness as rb
from experiments.real_data.alternative_estimators import ALTERNATIVE_METHODS


def _records(s, n, rng, proxy_noise=0.0):
    N = s.size
    tri_i, tri_j = np.triu_indices(N, k=1)
    pick = rng.integers(tri_i.size, size=n)
    i, j = tri_i[pick], tri_j[pick]
    y = rng.binomial(1, expit(s[i] - s[j])).astype(float)
    flip = rng.random(n) < proxy_noise
    f = np.where(flip, 1.0 - y, y)
    return i, j, y, f


@pytest.fixture()
def truth():
    rng = np.random.default_rng(0)
    s = rng.normal(size=6); s -= s.mean()
    return s, rng


def test_ppi_lambda_zero_is_human_only(truth):
    s, rng = truth
    i, j, y, f = _records(s, 300, rng, proxy_noise=0.3)
    ui, uj, _, uf = _records(s, 2000, rng, proxy_noise=0.3)
    labeled = np.column_stack([i, j, y, f]); unlabeled = np.column_stack([ui, uj, uf])
    ppi = fit_bt_ppi(6, labeled, unlabeled, lam=0.0)
    human = fit_human_only_btl(6, [(int(a), int(b), 1.0, float(c)) for a, b, c in zip(i, j, y)])["s_H"]
    assert np.allclose(ppi["s_H"], human, atol=1e-4)


def test_ppi_recovers_truth_with_good_proxy(truth):
    s, rng = truth
    i, j, y, f = _records(s, 200, rng, proxy_noise=0.05)
    ui, uj, _, uf = _records(s, 20000, rng, proxy_noise=0.05)
    fit = fit_bt_ppi(6, np.column_stack([i, j, y, f]), np.column_stack([ui, uj, uf]))
    assert 0.0 <= fit["lam"] <= 1.0 and fit["lam"] > 0.5
    assert fit["fit_info"]["finite"]
    human = fit_human_only_btl(6, [(int(a), int(b), 1.0, float(c)) for a, b, c in zip(i, j, y)])["s_H"]
    assert np.mean((fit["s_H"] - s) ** 2) < np.mean((human - s) ** 2)


def test_ppi_lambda_small_for_useless_proxy(truth):
    s, rng = truth
    i, j, y, _ = _records(s, 400, rng)
    f = rng.binomial(1, 0.5, size=i.size).astype(float)          # proxy unrelated to the outcome
    ui, uj, _, _ = _records(s, 3000, rng)
    lam, _ = ppi_lambda(6, np.zeros(6), i, j, y, f, ui, uj)
    assert lam < 0.2


def test_ppi_reports_nonfinite_fit():
    # separated labeled sample and no proxy-only records: the human part has no finite MLE
    labeled = np.array([[0, 1, 1.0, 1.0]] * 5 + [[1, 2, 1.0, 1.0]] * 5)
    fit = fit_bt_ppi(3, labeled, np.zeros((0, 3)), lam=0.0)
    assert not fit["fit_info"]["finite"] or fit["fit_info"]["max_abs"] > 5.0


def test_ridge_endpoints(truth):
    s, rng = truth
    i, j, y, _ = _records(s, 400, rng)
    pairs = [(int(a), int(b), 1.0, float(c)) for a, b, c in zip(i, j, y)]
    human = fit_human_only_btl(6, pairs)["s_H"]
    s0, _ = fit_ridge_btl_fixed(6, pairs, 0.0)
    assert np.allclose(s0, human, atol=1e-4)
    mu = s + rng.normal(scale=0.3, size=6); mu -= mu.mean()
    s_inf, _ = fit_ridge_btl_fixed(6, pairs, 1e6, mu=mu)
    assert np.linalg.norm(s_inf - (s_inf @ mu / (mu @ mu)) * mu) < 1e-2          # on the line span(mu)
    fit = fit_ridge_btl(6, pairs, mu=mu)
    assert fit["kappa"] > 0 and fit["fit_info"]["selected_regular"]


def test_boxed_fit_flags_separation():
    pairs = [(0, 1, 5.0, 5.0), (1, 2, 5.0, 5.0)]                      # 0 > 1 > 2 with no reversals
    fit = fit_boxed_btl(3, pairs, bound=10.0)
    assert fit["fit_info"]["box_active"]
    pairs = [(0, 1, 10.0, 7.0), (1, 2, 10.0, 6.0), (0, 2, 10.0, 4.0)]
    assert not fit_boxed_btl(3, pairs)["fit_info"]["box_active"]


def test_ppi_inputs_statuses():
    llm = pd.DataFrame(dict(record=["a", "a", "b", "b", "c"], k=[0, 1, 0, 1, 0], i=0, j=1, y=[1.0, 0.0, 1.0, 1.0, 0.0], a=1))
    cal = pd.DataFrame(dict(record=["a", "d"], i=0, j=1, y=[1.0, 0.0]))
    labeled, unlabeled, info = rm.ppi_inputs(llm, cal, "panel")
    assert info["ppi_status"] == "partial" and info["n_labeled_matched"] == 1 and info["n_proxy_only"] == 2
    assert labeled[0, 3] == 0.5                                        # split panel vote on record a
    labeled, unlabeled, info = rm.ppi_inputs(llm, cal.iloc[:1], "best")
    assert info["ppi_status"] == "ok" and info["proxy_judge"] == 0


def _frame(n_records=120, judges=("j1", "j2", "j3"), items=("m1", "m2", "m3", "m4"), seed=0):
    rng = np.random.default_rng(seed)
    quality = {m: q for m, q in zip(items, [1.0, 0.3, -0.3, -1.0])}
    rows = []
    for t in range(n_records):
        i, j = sorted(rng.choice(items, size=2, replace=False))
        p = expit(quality[i] - quality[j])
        human = i if rng.random() < p else j
        for judge in judges:
            for a in (1, -1):
                outcome = i if rng.random() < expit(quality[i] - quality[j] + 0.3 * a) else j
                rows.append(dict(dataset="mt_bench", record_id=f"r{t}", judge=judge, item_i=i, item_j=j, outcome=outcome, display_order=a,
                                 swapped=(a == -1), human_outcome=human, judge_tie=False, human_decisive=True, language="English"))
    return pd.DataFrame(rows)


@pytest.mark.slow
def test_alternative_estimators_in_budget_and_matched_cells(monkeypatch):
    cfg = rb.load_config()
    cfg["study"]["exclude_judges"] = []
    monkeypatch.setitem(rb._PANELS, "mt_bench", rb.build_panel(_frame()))
    cfg["_only_methods"] = list(ALTERNATIVE_METHODS) + ["consensus_cal", "human_only"]
    rows = rb.run_cell(("mt_bench", "budget", "all", "none", 40, None, 0, cfg))
    by = {r["method"]: r for r in rows}
    assert set(ALTERNATIVE_METHODS) <= set(by)
    for m in ALTERNATIVE_METHODS:
        assert "error" not in by[m], by[m].get("error")
    assert by["ppi_bt_panel"]["ppi_status"] == "ok" and 0.0 <= by["ppi_bt_panel"]["lam"] <= 1.0
    rows = rb.run_cell(("mt_bench", "matched", "all", "none", 30, 20, 0, cfg))
    by = {r["method"]: r for r in rows}
    assert by["ppi_bt_best"]["n_U"] == 20 and by["ppi_bt_best"]["n_labeled_matched"] == 30 and by["ppi_bt_best"]["n_proxy_only"] == 20
    rows = rb.run_cell(("mt_bench", "budget_full", "all", "none", 0, None, 0, cfg))
    assert [r["method"] for r in rows] == ["llm_only"] and np.isnan(rows[0]["excess"])
