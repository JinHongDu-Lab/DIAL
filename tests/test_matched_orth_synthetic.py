"""Tests for the matched-pair synthetic design and the add-on runner."""
from __future__ import annotations

import numpy as np

from dial_judge.simulate import generate_matched_comparisons
from experiments.simulation.synthetic import ADDON_METHODS, run_addon_cell, simulate_cell


def test_matched_design_shapes_and_pairing():
    rng = np.random.default_rng(0)
    S = rng.normal(size=(3, 5)); S -= S.mean(1, keepdims=True)
    b = np.array([0.2, -0.1, 0.4]); s_H = S.mean(0)
    d = generate_matched_comparisons(S, b, s_H, n_labeled=7, n_unlabeled=11, random_seed=1, swap_fraction=0.3)
    assert len(d["llm"]) == 3 * 18 and len(d["human"]) == 7
    assert d["f_labeled"].shape == (7, 3) and d["f_unlabeled"].shape == (11, 3)
    # every judge answers every record, and the human pair of a labeled record is the judges' pair
    llm = np.asarray(d["llm"])
    assert np.array_equal(llm[:3, 1:3], np.repeat(d["pairs_labeled"][:1], 3, axis=0))
    assert set(llm[:, 4]) <= {-1, 1}


def test_addon_runner_marks_unmatched_and_runs_matched():
    rows = run_addon_cell(("main10", "nH", 2000, 200, 0.0, 0))
    by = {r["method"]: r for r in rows}
    assert set(by) == set(ADDON_METHODS)
    assert by["ppi_bt_best"]["error"] == "unmatched_design" and "error" not in by["ridge_mu"]
    rows = run_addon_cell(("main10_matched", "nH", 500, 200, 0.0, 0))
    by = {r["method"]: r for r in rows}
    assert by["ppi_bt_panel"]["ppi_status"] == "ok" and 0.0 <= by["ppi_bt_panel"]["lam"] <= 1.0
    assert by["ppi_bt_best"]["n_labeled_matched"] == 200 and by["ppi_bt_best"]["n_proxy_only"] == 500
    cell = simulate_cell("main10_matched", "nH", 500, 200, 0.0, 0)
    assert cell["base"]["matched"] and cell["base"]["n_verdicts"] == 4 * 700


def test_orth_row_scales_misalignment_with_rho():
    from experiments.simulation.synthetic import cells_for
    cells = cells_for("main10_orth", "orth")
    assert len(cells) == 14 and cells[0][2] == 0.0
    c0 = simulate_cell("main10_orth", "orth", 20000, 800, 0.0, 0)
    c1 = simulate_cell("main10_orth", "orth", 20000, 800, 0.5, 0)
    assert c0["base"]["delta_mu"] < 1e-8 and c1["base"]["delta_mu"] > 0 and c1["base"]["delta_W"] > 0
    assert c1["base"]["delta_W"] <= c1["base"]["delta_mu"] + 1e-9
    assert abs(c1["base"]["orth_norm"] - 0.5 * np.sqrt(10)) < 1e-6
    assert c1["base"]["sigma_pos"] == 0.0 and c1["base"]["rho_orth"] == 0.5
