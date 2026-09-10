"""Tests for the shared drawing helpers of experiments/style.py and for reading the
pre-rename (n_0) results files with the current (n_H) column names."""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from experiments.simulation.r1_plot import normalize_schema
from experiments.style import LABEL, PRESENTED_W, STYLE, legend_handles, method_bars, method_lines


@pytest.fixture
def frame():
    return pd.DataFrame([dict(x=x, method=m, y=0.1 * x + i, se=0.01, group=f"g{x}", flag=(x == 1 and m == "dial_mu"))
                         for i, m in enumerate(PRESENTED_W) for x in (1, 2, 3)])


def test_method_lines_styles_every_method(frame):
    ax = plt.subplots()[1]
    drawn = method_lines(ax, frame, "x", "y", PRESENTED_W, se="se", band=True)
    assert drawn == PRESENTED_W
    lines = ax.lines[:len(PRESENTED_W)]
    assert [l.get_color() for l in lines] == [STYLE[m]["color"] for m in PRESENTED_W]
    assert [l.get_linewidth() for l in lines] == [STYLE[m]["lw"] for m in PRESENTED_W]
    assert len(ax.collections) == len(PRESENTED_W)          # one confidence band each
    plt.close("all")


def test_method_lines_skips_absent_methods_and_keeps_axis_labels(frame):
    ax = plt.subplots()[1]
    ax.set_xlabel("budget"); ax.set_ylabel("risk")
    drawn = method_lines(ax, frame[frame.method != "dial_w"], "x", "y", PRESENTED_W)
    assert "dial_w" not in drawn
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("budget", "risk")
    plt.close("all")


def test_method_bars_dodges_and_hatches(frame):
    ax = plt.subplots()[1]
    methods = list(PRESENTED_W)
    method_bars(ax, frame, "group", "y", methods, se="se", hatch="flag", order=["g1", "g2", "g3"])
    assert len(ax.patches) == len(methods) * 3
    assert len([p for p in ax.patches if p.get_hatch()]) == 1
    first_group = sorted(cont[0].get_x() for cont in ax.containers[:len(methods)])
    assert first_group == sorted(set(first_group))          # bars of one group do not overlap
    plt.close("all")


def test_legend_handles_order_and_kinds():
    h, l = legend_handles(PRESENTED_W, as_patch=["dial_mu"], no_marker=["human_only"],
                          extra=[("theory", dict(color="k", ls=":"))], extra_first=True)
    assert l == ["theory"] + [LABEL[m] for m in PRESENTED_W]
    assert h[1 + PRESENTED_W.index("human_only")].get_marker() == "None"


def test_normalize_schema_maps_pre_rename_names():
    df = normalize_schema(pd.DataFrame([dict(row="n0", n_L=10, n_0=5, seed=0)]))
    assert list(df["row"]) == ["nH"] and "n_H" in df and "n_0" not in df


def test_robustness_load_maps_pre_rename_names(tmp_path):
    from experiments.real_data.robustness_plot import load
    d = tmp_path / "results" / "real_robustness" / "toy"; d.mkdir(parents=True)
    (d / "rows.jsonl").write_text(json.dumps(dict(sweep="budget", panel="all", kind="none", level=1.0,
                                                  n_0_level=300, n_0=300.0, method="dial_mu", seed=0)) + "\n")
    df = load("toy", base=tmp_path)
    assert df["n_H_level"].iloc[0] == 300 and df["n_H"].iloc[0] == 300.0
