"""Tests for the record-level bootstrap helpers and runners (group 4), on tiny problems."""
from __future__ import annotations

import numpy as np

from dial_judge.bootstrap import adjacent_separation, coverage, resample_within_strata, separated_sign_agreement, summarize_contrasts
from dial_judge.simulate import generate_matched_comparisons


def test_record_effect_is_shared_within_record():
    rng = np.random.default_rng(0)
    S = rng.normal(size=(3, 5)); S -= S.mean(1, keepdims=True)
    d = generate_matched_comparisons(S, np.zeros(3), S.mean(0), n_labeled=5, n_unlabeled=5, random_seed=1, record_sd=1.0)
    assert d["record_effects"].shape == (10,) and d["record_effects"].std() > 0.3
    d0 = generate_matched_comparisons(S, np.zeros(3), S.mean(0), n_labeled=5, n_unlabeled=5, random_seed=1, record_sd=0.0)
    assert np.all(d0["record_effects"] == 0)


def test_bootstrap_summaries():
    rng = np.random.default_rng(1)
    s_hat = np.array([1.0, 0.0, -1.0])
    S = s_hat + 0.1 * rng.normal(size=(200, 3))
    S[3] = np.nan
    bs = summarize_contrasts(s_hat, S)
    assert bs["n_boot"] == 199 and bs["se"].shape == (3,) and np.all(bs["lower_pct"] < bs["upper_pct"])
    assert 0.9 <= coverage(np.array([1.0, 2.0, 1.0]), bs["lower_wald"], bs["upper_wald"]) <= 1.0
    frac, n = adjacent_separation(s_hat, bs["lower_pct"], bs["upper_pct"])
    assert frac == 1.0 and n == 2
    agree, n_sep = separated_sign_agreement(s_hat, bs["lower_pct"], bs["upper_pct"], np.array([0.0, 1.0, -1.0]))
    assert n_sep == 3 and abs(agree - 2 / 3) < 1e-9
    strata = [np.arange(4), np.arange(4, 10)]
    draw = resample_within_strata(strata, rng)
    assert draw.size == 10 and np.all(draw[:4] < 4) and np.all(draw[4:] >= 4)


def test_synthetic_record_bootstrap_cell_runs():
    from experiments.simulation.record_bootstrap import run_cell
    rows = run_cell(("main10_record", "record", 300, 200, 0.5, 0, 3))
    by = {r["method"]: r for r in rows}
    assert "error" not in by["dial_mu"], by["dial_mu"].get("error")
    assert by["dial_mu"]["record_sd"] == 0.5 and by["dial_mu"]["n_boot"] >= 2 and 0 <= by["dial_mu"]["cov_boot_pct"] <= 1
    assert "cov_boot_pct" in by["human_only"]
