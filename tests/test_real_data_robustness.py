"""Tests for the real-data robustness protocol (experiments/real_data/robustness.py).

All on a hand-built canonical frame, never on the shipped data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dial_judge.evaluate import heldout_log_loss
from dial_judge.gacv import btl_mle_exists
from dial_judge.inference import calibration_restriction_test
from experiments.real_data import robustness as rb


def _frame(n_records=40, judges=("j1", "j2", "tie-heavy"), items=("m1", "m2", "m3"), seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(n_records):
        i, j = sorted(rng.choice(items, size=2, replace=False))
        human = i if rng.random() < 0.6 else (j if rng.random() < 0.8 else None)
        for judge in judges:
            for a in (1, -1):
                tie = judge == "tie-heavy" and rng.random() < 0.8
                outcome = None if tie else (i if rng.random() < 0.7 else j)
                rows.append(dict(dataset="mt_bench", record_id=f"r{t}", judge=judge, item_i=i, item_j=j, outcome=outcome, display_order=a,
                                 swapped=(a == -1), human_outcome=human, judge_tie=tie, human_decisive=human is not None, language="English"))
    return pd.DataFrame(rows)


@pytest.fixture()
def panel():
    return rb.build_panel(_frame())


def test_build_panel_exclusions_and_coding(panel):
    assert panel["judges"] == ["j1", "j2"] and panel["dropped_judges"] == ["tie-heavy"]
    assert panel["N"] == 3 and panel["K"] == 2
    llm = panel["llm"]
    assert (llm["i"] < llm["j"]).all() and set(llm["y"].unique()) <= {0.0, 1.0} and set(llm["a"].unique()) == {1, -1}
    assert panel["human"]["record"].is_unique
    sub = rb.restrict_panel(panel, ["j2"])
    assert sub["K"] == 1 and set(sub["llm"]["k"]) == {0} and len(sub["llm"]) == (llm["k"] == 1).sum()


def test_split_keeps_records_disjoint_and_together(panel):
    test, train = rb.split_records(panel["records"], 0.3, np.random.default_rng(1))
    assert not (test & train) and len(test) + len(train) == len(panel["records"])
    llm = panel["llm"][panel["llm"]["record"].isin(train)]
    assert (llm.groupby(["k", "record"])["a"].nunique() == 2).all()


def test_thin_and_subsample():
    llm = pd.DataFrame(dict(record=["r"] * 4, k=[0, 0, 1, 1], i=0, j=1, y=1.0, a=[1, -1, 1, -1]))
    rng = np.random.default_rng(0)
    assert rb.thin_display_order(llm, 1.0, rng).equals(llm)
    kept = rb.thin_display_order(llm, 0.0, rng)
    assert (kept["a"] == 1).all() and len(kept) == 2
    big = pd.DataFrame(dict(record=np.arange(4000) // 2, k=0, i=0, j=1, y=1.0, a=np.tile([1, -1], 2000)))
    kept = rb.thin_display_order(big, 1 / 3, np.random.default_rng(0))
    assert (kept["a"] == 1).sum() == 2000 and abs((kept["a"] == -1).mean() - 0.25) < 0.03
    assert len(rb.subsample_rows(big, 100, rng)) == 100 and rb.subsample_rows(big, -1, rng) is big


def test_inject_noise_judges(panel):
    llm = panel["llm"]
    out, K = rb.inject_noise_judges(llm, panel["K"], 3, "random", np.random.default_rng(2))
    assert K == panel["K"] + 3 and set(out[out["k"] >= panel["K"]]["k"]) == {2, 3, 4}
    assert len(out[out["k"] >= panel["K"]]) == 3 * 2 * llm["record"].nunique()
    big = pd.DataFrame(dict(record=np.arange(3000), k=0, i=0, j=1, y=1.0, a=1, swapped=False))
    r = rb.inject_noise_judges(big, 1, 1, "random", np.random.default_rng(0))[0].query("k == 1")
    assert abs(r["y"].mean() - 0.5) < 0.03
    r = rb.inject_noise_judges(big, 1, 1, "position", np.random.default_rng(0), first_prob=0.9)[0].query("k == 1")
    assert abs(np.where(r["a"] == 1, r["y"], 1 - r["y"]).mean() - 0.9) < 0.03
    anti = rb.inject_noise_judges(llm, panel["K"], 1, "anti", np.random.default_rng(0))[0].query(f"k == {panel['K']}")
    flipped = []
    for k in range(panel["K"]):
        m = anti.merge(llm[llm["k"] == k], on=["record", "i", "j", "a"], suffixes=("_anti", ""))
        flipped.append(len(m) == len(anti) and ((m["y_anti"] + m["y"]) == 1.0).all())
    assert sum(flipped) == 1


def test_draw_budget_and_arrays(panel):
    h = panel["human"]
    rng = np.random.default_rng(0)
    assert rb.draw_budget(h, -1, rng) is h and len(rb.draw_budget(h, 5, rng)) == 5
    n_ijk, y_ijk, n_order, y_order = rb.llm_arrays(panel["llm"], panel["N"], panel["K"])
    assert n_order.sum() == len(panel["llm"]) and np.allclose(n_ijk, n_order.sum(-1))
    pairs = rb.human_pairs(h)
    assert sum(p[2] for p in pairs) == len(h)


def test_heldout_log_loss_and_existence():
    recs = [(0, 0, 1, 1.0), (0, 0, 1, 0.0), (0, 1, 2, 0.5)]
    assert heldout_log_loss(np.zeros(3), recs) == pytest.approx(np.log(2))
    assert btl_mle_exists(3, [(0, 1, 1.0, 1.0), (1, 2, 1.0, 1.0), (0, 2, 1.0, 0.0)])
    assert not btl_mle_exists(3, [(0, 1, 2.0, 2.0), (0, 2, 1.0, 1.0), (1, 2, 2.0, 1.0)])
    assert not btl_mle_exists(4, [(0, 1, 2.0, 1.0), (1, 2, 2.0, 1.0), (0, 2, 2.0, 1.0)])


def test_calibration_restriction_test():
    rng = np.random.default_rng(0)
    N = 5
    s0 = np.array([1.0, 0.5, 0.0, -0.5, -1.0])
    pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            n = 40
            y = rng.binomial(n, 1 / (1 + np.exp(-(s0[i] - s0[j]))))
            pairs.append((i, j, float(n), float(y)))
    t_true = calibration_restriction_test(N, pairs, s0)             # W spans the truth: small statistic
    t_wrong = calibration_restriction_test(N, pairs, np.array([1.0, -1.0, 1.0, -1.0, 0.0]))  # orthogonal direction: large statistic
    assert t_true["df"] == N - 2 and 0 <= t_true["pvalue"] <= 1 and t_true["human_only_exists"]
    assert t_wrong["stat"] > t_true["stat"] and t_wrong["pvalue"] < 0.01
    t_full = calibration_restriction_test(N, pairs, np.column_stack([s0, np.eye(N)[:, :3] - 1 / N]))
    assert t_full["df"] == 0
    assert t_true["s_full_method"] == "mle"


def test_calibration_restriction_test_firth_fallback():
    """Under Ford (1957) separation the unrestricted fit falls back to Firth's bias-reduced
    estimator instead of silently returning the divergent MLE."""
    rng = np.random.default_rng(3)
    N = 5
    s0 = np.array([1.0, 0.5, 0.0, -0.5, -1.0])
    pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            n = 30
            y = n if i == 0 else rng.binomial(n, 1 / (1 + np.exp(-(s0[i] - s0[j]))))   # item 0 never loses
            pairs.append((i, j, float(n), float(y)))
    t = calibration_restriction_test(N, pairs, s0)
    assert not t["human_only_exists"] and t["s_full_method"] == "firth"
    assert np.all(np.isfinite(t["s_full"])) and np.max(np.abs(t["s_full"])) < 30.0
    assert t["stat"] >= 0.0 and np.isfinite(t["stat"])
    t_off = calibration_restriction_test(N, pairs, s0, firth_fallback=False)
    assert t_off["s_full_method"] == "mle"


def test_jobs():
    cfg = rb.load_config()
    jobs = rb.jobs_for("noise", cfg, range(2))
    zero = [j for j in jobs if j[4] == 0]
    assert len(zero) == 2 * len(cfg["sweeps"]["noise"]["datasets"]) and {j[3] for j in zero} == {cfg["sweeps"]["noise"]["kinds"][0]}
    assert all(j[2] == "biased5" for j in jobs)
    lb = rb.jobs_for("llm_budget", cfg, range(1), smoke=True)
    grid = cfg["sweeps"]["llm_budget"]["n_H_grid"]
    n_expected = sum(2 * len(cfg["sweeps"]["llm_budget"]["panels"]) * len(rb.per_dataset(grid, d)) for d in cfg["sweeps"]["llm_budget"]["datasets"])
    assert len(lb) == n_expected and {j[5] for j in lb if j[0] == "arena_33k"} == set(rb.per_dataset(grid, "arena_33k"))
    assert {j[4] for j in lb if j[0] == "mt_bench"} == set(rb.per_dataset(cfg["sweeps"]["llm_budget"]["levels"], "mt_bench")[:2])
    assert rb.per_dataset(5, "pandalm") == 5 and rb.per_dataset({"pandalm": 7}, "pandalm") == 7
    sp = rb.jobs_for("spectest", cfg, range(1), smoke=True)
    assert len(sp) == 3 * 2 and len(sp[0]) == 4
    # keys are unique within a dataset's rows file
    assert len({(j[0], rb.job_key(j)) for j in jobs + lb}) == len(jobs) + len(lb)
    assert len({rb.job_key(j) for j in sp}) == len(sp)


def test_paired_design_across_levels(monkeypatch):
    frame = _frame(n_records=200, judges=("j1", "j2", "j3"), items=("m1", "m2", "m3", "m4"), seed=5)
    cfg = rb.load_config()
    cfg["mt_bench"] = dict(f_test=0.3, n_H=40)
    cfg["sweeps"]["order"]["datasets"] = ["mt_bench"]
    monkeypatch.setattr(rb, "_PANELS", {"mt_bench": rb.build_panel(frame)})
    a = rb.run_cell(("mt_bench", "order", "all", "none", 1.0, None, 7, cfg))
    b = rb.run_cell(("mt_bench", "order", "all", "none", 0.0, None, 7, cfg))
    ha = next(r for r in a if r["method"] == "human_only")
    hb = next(r for r in b if r["method"] == "human_only")
    assert ha["excess"] == pytest.approx(hb["excess"]) and ha["n_test"] == hb["n_test"]
    methods = {r["method"] for r in a}
    assert {"human_only", "pooled_cal", "consensus_cal", "dial_mu", "dial_nodeb", "staged_w", "dial_w", "dial_mle_mu", "dial_mle_w", "oracle_test", "dial_rsel"} <= methods
    dm = next(r for r in a if r["method"] == "dial_mu"); assert dm["r"] == 1
    assert all("error" not in r for r in a), [r["error"][:80] for r in a if "error" in r]


def test_drop_empty_judges():
    import pandas as pd
    llm = pd.DataFrame(dict(k=[0, 0, 2, 5, 5], i=[0] * 5, j=[1] * 5, a=[1, -1, 1, 1, -1], y=[1.0, 0.0, 1.0, 1.0, 0.0]))
    out, K, K_real, dropped = rb.drop_empty_judges(llm, K=6, K_real=4)
    assert (K, K_real, dropped) == (3, 2, 3) and sorted(out.k.unique()) == [0, 1, 2]
    same, K2, Kr2, d2 = rb.drop_empty_judges(out, K=3, K_real=2)
    assert d2 == 0 and K2 == 3 and Kr2 == 2
