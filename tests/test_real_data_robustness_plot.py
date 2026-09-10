"""Tests for experiments/real_data/robustness_plot.py's planner decision rule (item 1), the
stopping-time backtest (item 3), and the joint human/LLM allocation heuristic (item 2)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.real_data.robustness_plot import (
    decision_fractions_by_level,
    fit_llm_decay,
    joint_allocation,
    noncentrality_interval,
    planner_decision,
    predict_llm_excess,
    stopping_policy_backtest,
    switch_share_by_target,
)


def _planner_row(N=60, r=1, n_H=400, stat=5.0, df=57, human_only_exists=True, level=0.90):
    lo, hi = noncentrality_interval(stat, df, level)
    return pd.Series(dict(N=N, r=r, n_H=n_H, stat=stat, df=df, human_only_exists=human_only_exists,
                          delta_lo=lo / (2.0 * n_H), delta_hi=hi / (2.0 * n_H)))


def test_planner_decision_insufficient_pilot():
    row = _planner_row(human_only_exists=False, n_H=100, N=60)
    dec = planner_decision(row, target_n=1000)
    assert dec["action"] == "insufficient_pilot"
    assert dec["suggested_n"] >= 2 * 100


def test_planner_decision_keep_anchoring_when_delta_small():
    """A small statistic (near its df) gives Delta_lo = 0, so n_star_hi = inf and any finite
    target budget keeps anchoring as the default."""
    row = _planner_row(stat=57.0, df=57, n_H=400, N=60, r=1)          # T close to df: Delta_lo = 0
    dec = planner_decision(row, target_n=10_000)
    assert dec["action"] == "keep_anchoring" and dec["n_star_hi"] == np.inf


def test_planner_decision_switch_when_target_beyond_conservative_crossover():
    """A statistic far above its df gives a finite, small n_star_hi; a target budget beyond it
    triggers the conservative "switch" branch."""
    row = _planner_row(stat=400.0, df=57, n_H=400, N=60, r=1)
    dec = planner_decision(row, target_n=10 ** 7)
    assert dec["action"] == "switch" and np.isfinite(dec["n_star_hi"]) and dec["confident"]


def test_planner_decision_target_eps_recommends_smaller_budget():
    """T = df exactly gives delta_hat = delta_lo = 0, but delta_hi (the confidence bound the
    rule actually acts on) is still positive; a target eps comfortably above delta_hi keeps
    anchoring, and the recommended budget matches the anchored-only formula (line ~3482)."""
    row = _planner_row(stat=57.0, df=57, n_H=400, N=60, r=1)
    assert row["delta_lo"] == 0.0 and row["delta_hi"] > 0.0
    eps = row["delta_hi"] + 0.02
    dec = planner_decision(row, target_eps=eps)
    n_anchor_only = (row["r"] + 1) / (2.0 * (eps - row["delta_hi"]))
    assert dec["action"] == "keep_anchoring"
    assert dec["n_recommend"] == pytest.approx(n_anchor_only)


def test_planner_decision_target_eps_switches_when_too_tight():
    """A target below the conservative floor delta_hi can never be reached by anchoring alone,
    however many labels are bought: the rule switches to the human-only budget immediately."""
    row = _planner_row(stat=57.0, df=57, n_H=400, N=60, r=1)
    eps = row["delta_hi"] / 2.0
    dec = planner_decision(row, target_eps=eps)
    assert dec["action"] == "switch"
    assert dec["n_recommend"] == pytest.approx((row["N"] - 1) / (2.0 * eps))


def test_stopping_policy_backtest_shape_and_stops_when_switch_available():
    """Two seeds on a tiny synthetic frame: seed 0's statistic grows sharply with the pilot level
    (should stop before the max level), seed 1 stays flat (should reach the max level)."""
    levels = [50, 100, 200]
    rows = []
    for seed, stats in ((0, {50: 20.0, 100: 40.0, 200: 400.0}), (1, {50: 20.0, 100: 21.0, 200: 22.0})):
        for lvl in levels:
            rows.append(dict(dataset="toy", sweep="planner", panel="all", kind="none", level=float(lvl), n_H_level=-1, n_H=float(lvl),
                             seed=seed, N=20.0, K=3, r=1, stat=stats[lvl], df=17, human_only_exists=True, method="planner", failed=False,
                             n_test=500, delta_hat=max(0.0, (stats[lvl] - 17) / (2.0 * lvl)), delta_all=0.0))
        for lvl in levels:
            for method, excess in (("dial_mu", 0.02), ("human_only", 6.0 / lvl)):
                rows.append(dict(dataset="toy", sweep="budget", panel="all", kind="none", level=float(lvl), n_H_level=lvl, n_H=float(lvl),
                                 seed=seed, N=20.0, K=3, r=1, method=method, excess=excess, failed=False))
    df = pd.DataFrame(rows)
    out = stopping_policy_backtest(df, "toy")
    assert set(out.columns) >= {"seed", "stop_level", "action", "cost", "loss_dial_stop", "loss_human_stop", "loss_dial_full", "cost_full"}
    assert len(out) == 2
    row0 = out[out.seed == 0].iloc[0]
    row1 = out[out.seed == 1].iloc[0]
    assert row0["action"] == "switch"                                    # sharply growing statistic: the rule catches the crossover
    assert row1["stop_level"] == 200 and row1["action"] == "reached_max"  # flat statistic: never confidently crosses, buys to the max level


def test_decision_fractions_by_level_tracks_growing_and_flat_statistics():
    """Figure G6(a) helper: with the same two-seed toy pilots as the backtest test above, the
    sharply growing statistic (seed 0) should show up as a "switch" share at the largest level
    while the flat one (seed 1) never does."""
    levels = [50, 100, 200]
    rows = []
    for seed, stats in ((0, {50: 20.0, 100: 40.0, 200: 400.0}), (1, {50: 20.0, 100: 21.0, 200: 22.0})):
        for lvl in levels:
            rows.append(dict(dataset="toy", sweep="planner", panel="all", kind="none", level=float(lvl), n_H_level=-1, n_H=float(lvl),
                             seed=seed, N=20.0, K=3, r=1, stat=stats[lvl], df=17, human_only_exists=True, method="planner", failed=False,
                             n_test=500, delta_hat=max(0.0, (stats[lvl] - 17) / (2.0 * lvl)), delta_all=0.0))
    df = pd.DataFrame(rows)
    out = decision_fractions_by_level(df, "toy").set_index("level")
    assert set(out.columns) >= {"n", "keep_anchoring", "switch", "insufficient_pilot"}
    for lvl in (50.0, 100.0, 200.0):
        row = out.loc[lvl]
        assert row["keep_anchoring"] + row["switch"] + row["insufficient_pilot"] == pytest.approx(1.0)
    assert out.loc[50.0, "switch"] == 0.0 and out.loc[100.0, "switch"] == 0.0
    assert out.loc[200.0, "switch"] == pytest.approx(0.5)          # only the sharply growing seed switches at the largest level


def test_switch_share_by_target_uses_largest_pilot_and_is_nondecreasing():
    """Two seeds: seed 0's full-pool pilot shows a clear signal (should start switching at large
    enough targets), seed 1's shows none (delta_lo = 0, never switches at any target)."""
    rows = []
    for seed, (n_H, stat) in ((0, (100, 25.0)), (0, (1000, 30.0)), (1, (100, 18.0)), (1, (1000, 17.0))):
        rows.append(dict(dataset="toy", sweep="planner", panel="all", kind="none", level=float(n_H), n_H_level=-1, n_H=float(n_H),
                         seed=seed, N=20.0, K=3, r=1, stat=stat, df=17, human_only_exists=True, method="planner", failed=False,
                         n_test=500, delta_hat=max(0.0, (stat - 17) / (2.0 * n_H)), delta_all=0.0))
    df = pd.DataFrame(rows)
    targets = np.geomspace(500, 500_000, 12)
    out = switch_share_by_target(df, "toy", targets)
    assert (out.n == 2).all()
    assert out.frac_switch.iloc[0] == 0.0                                  # no split switches at a small target
    assert out.frac_switch.iloc[-1] == pytest.approx(0.5)                  # only seed 0 ever switches, at the largest target
    assert (out.frac_switch.diff().dropna() >= 0).all()                    # nondecreasing in the target


def test_fit_llm_decay_recovers_known_coefficient():
    rng = np.random.default_rng(0)
    true_c = 30.0
    rows = []
    for level in (500, 1000, 2000, 5000, 20000):
        for seed in range(5):
            excess = 0.01 + true_c / level + rng.normal(0, 1e-4)
            rows.append(dict(dataset="arena_33k", sweep="llm_budget", panel="all", method="consensus_cal", n_H_level=1000, level=level, seed=seed, excess=excess, failed=False))
    df = pd.DataFrame(rows)
    out = fit_llm_decay(df, "arena_33k")
    assert out["c"] == pytest.approx(true_c, abs=1.0)


def test_fit_llm_decay_insufficient_data_returns_nan():
    df = pd.DataFrame([dict(dataset="d", sweep="llm_budget", panel="all", method="consensus_cal", n_H_level=1, level=500, seed=0, excess=0.02, failed=False)])
    out = fit_llm_decay(df, "d")
    assert np.isnan(out["c"])


def test_fit_llm_decay_max_level_restricts_the_pilot_and_extrapolates_correctly():
    """When the decay model holds exactly (clean synthetic data), fitting on a small-n_L pilot
    (max_level) and extrapolating with predict_llm_excess should recover held-out points closely
    -- the real-data version of this (fit_llm_decay's docstring) has a systematic conservative
    bias instead, because the true curve isn't exactly a + c/n_L; this test isolates that the
    extrapolation *mechanism* itself is correct when the model is well-specified."""
    true_a, true_c = 0.01, 30.0
    rows = [dict(dataset="arena_33k", sweep="llm_budget", panel="all", method="consensus_cal", n_H_level=1000, level=level,
                seed=0, excess=true_a + true_c / level, failed=False) for level in (500, 1000, 2000, 5000, 20000)]
    df = pd.DataFrame(rows)
    pilot_fit = fit_llm_decay(df, "arena_33k", max_level=1000)
    assert pilot_fit["n"] == 2
    pred = predict_llm_excess(pilot_fit, [2000, 5000, 20000])
    true_vals = true_a + true_c / np.array([2000, 5000, 20000])
    np.testing.assert_allclose(pred, true_vals, rtol=1e-6)


def test_fit_llm_decay_metric_handles_an_increasing_curve():
    """Kendall's tau rises toward an asymptote as n_L grows (the mirror image of excess falling
    toward a floor); the same a + c/n_L regression should handle it via a negative c, with no
    separate code path, and the metric column should be selectable."""
    true_a, true_c = 0.9, -20.0        # tau -> 0.9 from below as n_L grows
    rows = [dict(dataset="arena_33k", sweep="llm_budget", panel="all", method="dial_mu", n_H_level=300, level=level,
                seed=0, ref_kendall=true_a + true_c / level, failed=False) for level in (500, 1000, 2000, 5000, 20000)]
    df = pd.DataFrame(rows)
    fit = fit_llm_decay(df, "arena_33k", method="dial_mu", metric="ref_kendall")
    assert fit["c"] < 0
    pred = predict_llm_excess(fit, [500, 1000, 2000, 5000, 20000])
    true_vals = true_a + true_c / np.array([500, 1000, 2000, 5000, 20000])
    np.testing.assert_allclose(pred, true_vals, rtol=1e-6)
    assert np.all(np.diff(pred) > 0)   # increasing in n_L, as a real tau curve should be


def test_joint_allocation_matches_cost_constraint_and_is_a_local_minimum():
    row = pd.Series(dict(r=1, delta_hat=0.001))
    out = joint_allocation(row, llm_decay_c=50.0, cost_ratio=0.02, total_budget=2000.0)
    assert out["n_human"] + 0.02 * out["n_L"] == pytest.approx(2000.0)

    def excess(n_h):
        n_l = (2000.0 - n_h) / 0.02
        return 0.001 + 1.0 / n_h + 50.0 / n_l

    e0 = excess(out["n_human"])
    assert e0 <= excess(out["n_human"] + 50.0) and e0 <= excess(out["n_human"] - 50.0)


def test_joint_allocation_target_eps_hits_target_exactly():
    row = pd.Series(dict(r=1, delta_hat=0.001))
    out = joint_allocation(row, llm_decay_c=50.0, cost_ratio=0.02, target_eps=0.01)
    realized = 0.001 + 1.0 / out["n_human"] + 50.0 / out["n_L"]
    assert realized == pytest.approx(0.01, abs=1e-9)


def test_joint_allocation_requires_a_target():
    row = pd.Series(dict(r=1, delta_hat=0.001))
    with pytest.raises(ValueError):
        joint_allocation(row, llm_decay_c=50.0, cost_ratio=0.02)
