"""Aggregate results/real_robustness/<dataset>/rows.jsonl for the Section 22 figures.

Per (sweep, panel, kind, level, n_H_level, method): mean and Monte Carlo s.e. of the
held-out metrics, the paired difference to human-only on the same split, endpoint
shares of the selected weight, selected rank, and failure counts. Figures are drawn in
notebooks/real_data_robustness.ipynb.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dial_judge.inference import noncentrality_interval  # re-exported for callers of this module

KEY = ["sweep", "panel", "kind", "level", "n_H_level"]
METRICS = ["excess", "logloss", "acc", "ref_kendall", "b_mean_abs", "b_inj_mean_abs", "gamma_inj_mean_abs", "lam_rel", "r_sel", "first_share", "n_L", "n_H"]
MAIN_METHODS = ["human_only", "pooled_cal", "consensus_cal", "dial_mu", "dial", "dial_nodeb"]


def load(dataset, base="."):
    path = Path(base) / "results" / "real_robustness" / dataset / "rows.jsonl"
    df = pd.DataFrame([json.loads(l) for l in open(path)]).rename(columns={"n_0": "n_H", "n_0_level": "n_H_level"})  # pre-rename rows.jsonl
    for c in METRICS + ["error", "converged", "lam", "human_only_exists", "n_H_level", "n_at_bound", "b_at_bound", "K_dropped"]:
        if c not in df:
            df[c] = np.nan
    df["n_H_level"] = df["n_H_level"].fillna(-1).astype(int)
    df["failed"] = df["error"].notna()
    df["lam_rel"] = df["lam_rel"].replace([np.inf, -np.inf], np.nan)
    for sw in ("noise", "noise_scarce"):
        noise0 = df[(df.sweep == sw) & (df.level == 0)]
        if not noise0.empty:
            kinds = sorted(set(df.loc[df.sweep == sw, "kind"]) - set(noise0["kind"]))
            df = pd.concat([df] + [noise0.assign(kind=k) for k in kinds], ignore_index=True)
    return df


def _se(x):
    x = x.dropna()
    return x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan


def aggregate(df):
    df = df[~df.method.isin(["spectest", "planner"])]
    ok = df[~df.failed]
    g = ok.groupby(KEY + ["method"])
    agg = g.agg(n=("seed", "size"), K=("K", "first"), n_test=("n_test", "first"), floor=("floor", "mean"),
                **{f"{m}_mean": (m, "mean") for m in METRICS}, **{f"{m}_se": (m, _se) for m in METRICS},
                lam_rel_med=("lam_rel", "median"),
                lam_inf_frac=("lam", lambda x: float(np.mean(np.isinf(x.astype(float))))),
                lam_zero_frac=("lam", lambda x: float(np.mean(x.astype(float) == 0))),
                nonconv_frac=("converged", lambda x: float(np.mean(~x.fillna(True).astype(bool)))),
                sep_frac=("n_at_bound", lambda x: float(np.mean(x.fillna(0).astype(float) > 0))),      # fit touched the finite-fit box (a separated judge)
                bsep_frac=("b_at_bound", lambda x: float(np.mean(x.fillna(0).astype(float) > 0))),    # a position effect at the bound
                k_dropped_mean=("K_dropped", lambda x: float(np.nanmean(x.astype(float))) if x.notna().any() else 0.0),
                ho_exists_frac=("human_only_exists", lambda x: float(np.mean(x.fillna(True).astype(bool))))).reset_index()
    ho = ok[ok.method == "human_only"][KEY + ["seed", "excess"]].rename(columns={"excess": "ex_ho"})
    paired = ok.merge(ho, on=KEY + ["seed"], how="inner")
    paired["d_excess"] = paired["excess"] - paired["ex_ho"]
    pd_agg = paired.groupby(KEY + ["method"]).agg(d_excess_mean=("d_excess", "mean"), d_excess_se=("d_excess", _se)).reset_index()
    agg = agg.merge(pd_agg, on=KEY + ["method"], how="left")
    failed = df[df.failed].groupby(KEY + ["method"]).size().rename("n_failed").reset_index()
    return agg.merge(failed, on=KEY + ["method"], how="left").fillna({"n_failed": 0})


def aggregate_spectest(df):
    s = df[df.method == "spectest"]
    if s.empty:
        return s
    return s.groupby(["kind", "level"]).agg(n=("seed", "size"), reject_rate=("reject05", "mean"), stat_mean=("stat", "mean"), df=("df", "first"),
                                            ho_exists=("human_only_exists", "mean"), excess_restricted=("excess_restricted", "mean"), excess_full=("excess_full", "mean"),
                                            tau_mu=("tau_mu_vs_test_human", "mean"), n_H=("n_H", "first")).reset_index()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="all")
    a = p.parse_args()
    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_rows", 800)
    datasets = ("arena_33k", "mt_bench", "pandalm") if a.dataset == "all" else (a.dataset,)
    for d in datasets:
        path = Path("results/real_robustness") / d / "rows.jsonl"
        if not path.exists():
            continue
        df = load(d)
        agg = aggregate(df)
        agg.to_csv(path.parent / "summary.csv", index=False)
        print(f"=== {d}")
        for (sw, pn, kd), g in agg.groupby(["sweep", "panel", "kind"]):
            piv = g.pivot_table(index=["level", "n_H_level"], columns="method", values="excess_mean").reindex(columns=MAIN_METHODS + ["dial_mle_r0", "oracle_test"])
            print(f"--- {sw} | panel={pn} | kind={kd}   excess held-out log loss (mean over seeds)")
            print(piv.round(4).to_string())
        st = aggregate_spectest(df)
        if not st.empty:
            print("--- spectest"); print(st.round(3).to_string(index=False))


def planner_frame(df):
    """Planner rows with the first-order predictions attached (per seed and pilot size)."""
    pl = df[df.method == "planner"].copy()
    if pl.empty:
        return pl
    pl["opt_floor"] = (pl["N"] - 1) / (2.0 * pl["n_test"])          # first-order optimism of the in-sample test-pool floor
    pl["n_star"] = np.where(pl["delta_hat"] > 0, (pl["N"] - pl["r"] - 2) / (2.0 * pl["delta_hat"]), np.inf)
    pl["n_star_all"] = np.where(pl["delta_all"] > 0, (pl["N"] - pl["r"] - 2) / (2.0 * pl["delta_all"]), np.inf)
    return pl


def predicted_curves(pl_row, grid):
    """Predicted excess (relative to the in-sample test-pool floor) at human budgets `grid`."""
    grid = np.asarray(grid, dtype=float)
    anchor = pl_row["delta_hat"] + (pl_row["r"] + 1) / (2.0 * grid) + pl_row["opt_floor"]
    human = (pl_row["N"] - 1) / (2.0 * grid) + pl_row["opt_floor"]
    return anchor, human


def planner_with_intervals(pl, level=0.90):
    """Attach per-pilot interval endpoints for Delta_W (columns delta_lo, delta_hi) to a planner frame."""
    pl = pl.copy()
    ends = np.array([noncentrality_interval(t, d, level) for t, d in zip(pl["stat"], pl["df"])])
    pl["delta_lo"] = ends[:, 0] / (2.0 * pl["n_H"])
    pl["delta_hi"] = ends[:, 1] / (2.0 * pl["n_H"])
    return pl


def planner_decision(row, target_n=None, target_eps=None):
    """Item-1 decision rule (app:subsubsec:planner algorithm box): act on one row of
    `planner_with_intervals` output (must already carry delta_lo/delta_hi at the chosen level).

    Existence gate first (Ford 1957 / `human_only_exists` on the pilot): if the unrestricted
    human-only MLE does not exist, the statistic is unusable and the action is to collect more
    labels, with a suggested next pilot size on the order of the coupon-collector connectivity
    threshold N log N already used near app:optimal-design, floored at twice the current pilot.

    Otherwise anchoring is the default (manuscript line ~3495: report the anchored option when
    in doubt): with a target total human budget `target_n`, switching to human-only/DIAL without
    the anchor requires the *conservative* (delta_hi) bound to already place the crossover below
    target_n, i.e. target_n >= n_star_hi = (N-r-2)/(2 delta_lo); `confident` flags whether target_n
    also falls outside the ambiguous [n_star_lo, n_star_hi) band. With a target excess `target_eps`
    instead, compare the two budget formulas of line ~3482 (anchored evaluated conservatively at
    delta_hi) and recommend whichever is smaller.
    """
    N, r, n_H = float(row["N"]), float(row["r"]), float(row["n_H"])
    if not bool(row.get("human_only_exists", True)):
        n_min = max(2.0 * n_H, N * np.log(max(N, 2.0)))
        return dict(action="insufficient_pilot", suggested_n=float(n_min), n_star_lo=np.nan, n_star_hi=np.nan)

    delta_lo, delta_hi = float(row["delta_lo"]), float(row["delta_hi"])
    n_star_lo = (N - r - 2) / (2.0 * delta_hi) if delta_hi > 0 else np.inf     # earliest plausible crossover (pessimistic Delta)
    n_star_hi = (N - r - 2) / (2.0 * delta_lo) if delta_lo > 0 else np.inf     # latest plausible crossover (optimistic Delta); inf when Delta=0 not excluded
    out = dict(n_star_lo=n_star_lo, n_star_hi=n_star_hi)

    if target_n is not None:
        out["action"] = "switch" if target_n >= n_star_hi else "keep_anchoring"
        out["confident"] = bool(target_n < n_star_lo or target_n >= n_star_hi)
        return out
    if target_eps is not None:
        n_human = (N - 1) / (2.0 * target_eps)
        if target_eps <= delta_hi:              # anchoring's own worst-case floor already exceeds the target: no anchored budget reaches it
            out.update(action="switch", n_recommend=n_human)
        else:
            n_anchor = (r + 1) / (2.0 * (target_eps - delta_hi))
            out.update(action="keep_anchoring" if n_anchor <= n_human else "switch", n_recommend=min(n_anchor, n_human))
        return out
    out["action"] = "report_only"
    return out


def decision_fractions_by_level(df, dataset, level=0.90):
    """Item-1 figure helper: `planner_decision`'s action, per pilot level, targeting each seed's
    own realized full training pool -- shows how the rule's verdict on real data shifts as the
    pilot grows. Returns one row per level with the action shares and n over the seeds present.
    """
    pl = planner_with_intervals(planner_frame(df[df.dataset == dataset]), level=level)
    if pl.empty:
        return pl
    levels = sorted(pl["level"].unique(), key=lambda x: (x if x >= 0 else np.inf))
    n_max_level = levels[-1]
    rows = []
    for lvl in levels:
        g = pl[pl.level == lvl]
        full = pl[(pl.level == n_max_level) & (pl.seed.isin(g.seed))].set_index("seed")["n_H"]
        actions = [planner_decision(row, target_n=float(full.loc[row.seed]))["action"] for _, row in g.iterrows() if row.seed in full.index]
        s = pd.Series(actions).value_counts(normalize=True)
        rows.append(dict(level=lvl, n=len(actions), keep_anchoring=float(s.get("keep_anchoring", 0.0)),
                         switch=float(s.get("switch", 0.0)), insufficient_pilot=float(s.get("insufficient_pilot", 0.0))))
    return pd.DataFrame(rows)


def switch_share_by_target(df, dataset, targets, level=0.90):
    """Item-1 figure helper: using the best available pilot (the full collected pool) as evidence,
    the share of the 50 splits `planner_decision` would tell to switch to human-only for each
    hypothetical future total budget in `targets`. Answers "at what budget would the rule ever
    flip", as distinct from `decision_fractions_by_level`'s "how does confidence sharpen as the
    pilot grows" (target fixed at that split's own realized pool) -- the two vary the pilot size
    and the target size separately, matching Algorithm G.1's two separate inputs.
    """
    pl = planner_with_intervals(planner_frame(df[df.dataset == dataset]), level=level)
    if pl.empty:
        return pd.DataFrame()
    full = pl[pl.level == pl["level"].max()]          # the largest collected pilot (level = -1 sorts as -1, so use idxmax on n_H instead)
    full = pl.loc[pl.groupby("seed")["n_H"].idxmax()]  # robust to whether "-1" or a large explicit level is the largest realized n_H
    rows = []
    for t in targets:
        actions = [planner_decision(row, target_n=float(t))["action"] for _, row in full.iterrows()]
        rows.append(dict(target=float(t), frac_switch=float(np.mean([a == "switch" for a in actions])), n=len(actions)))
    return pd.DataFrame(rows)


def stopping_policy_backtest(df, dataset, level=0.90):
    """Item-3 backtest: apply `planner_decision` as a stopping rule for the human-labeling budget.

    The `planner` and `budget` sweeps use identical [seed, dataset]-seeded record splits and
    identical [seed, dataset]-seeded `draw_budget` draws, so for a matching seed and level they
    are fit on the exact same human calibration sample (robustness.run_planner and run_cell's
    budget branch both key off [seed, code, 1] for the split and [seed, code, 2] for the draw).
    Levels are walked in increasing order (config aligns sweeps.planner.levels with
    sweeps.budget.levels); the target is the realized full-pool human count for that seed, and
    the policy stops at the first level whose decision is "switch", else the largest level.
    Returns a per-seed frame with the stopping level, its cost (realized n_H), and the realized
    DIAL-mu / human-only excess log loss there and at the full pool (from the budget sweep).
    """
    dsub = df[df.dataset == dataset]
    pl = planner_with_intervals(planner_frame(dsub), level=level)
    bud = dsub[(dsub.sweep == "budget") & (dsub.panel == "all") & (~dsub.failed) & (dsub.method.isin(["dial_mu", "human_only"]))]
    if pl.empty or bud.empty:
        return pd.DataFrame()
    levels = sorted(pl["level"].unique(), key=lambda x: (x if x >= 0 else np.inf))
    n_max_level = levels[-1]
    rows = []
    for seed, g in pl.groupby("seed"):
        g = g.set_index("level")
        full = g.loc[n_max_level] if n_max_level in g.index else None
        if full is None:
            continue
        target_n = float(full["n_H"])
        stop_level, action = n_max_level, "reached_max"
        for lvl in levels:
            if lvl not in g.index:
                continue
            dec = planner_decision(g.loc[lvl], target_n=target_n)
            if dec["action"] == "switch":
                stop_level, action = lvl, "switch"
                break
        b_seed = bud[bud.seed == seed]
        b_stop = b_seed[b_seed.level == stop_level]
        b_full = b_seed[b_seed.level == n_max_level]

        def _lookup(frame, method, col):
            v = frame.loc[frame.method == method, col]
            return float(v.iloc[0]) if len(v) else np.nan

        rows.append(dict(seed=seed, stop_level=stop_level, action=action, cost=_lookup(b_stop, "dial_mu", "n_H"),
                          loss_dial_stop=_lookup(b_stop, "dial_mu", "excess"), loss_human_stop=_lookup(b_stop, "human_only", "excess"),
                          loss_dial_full=_lookup(b_full, "dial_mu", "excess"), cost_full=_lookup(b_full, "dial_mu", "n_H")))
    return pd.DataFrame(rows)


def fit_llm_decay(df, dataset, panel="all", method="consensus_cal", n_H_level=None, max_level=None, metric="excess"):
    """Item-2 heuristic: fit an empirical 1/n_L decay metric(n_L) ~ a + c/n_L to the llm_budget
    sweep at fixed n_human, for any metric column the sweep records (mean over seeds).

    `metric="excess"` (excess held-out log loss, decreasing in n_L, c > 0 at a well-behaved fit)
    and `metric="ref_kendall"` (Kendall's tau against the held-out human ranking, increasing in
    n_L toward an asymptote, c < 0) both use the same linear regression in 1/n_L; only the sign
    of the fitted c differs, so no separate code path is needed for an increasing metric.

    `max_level`, when given, restricts the fit to LLM pilot levels n_L <= max_level, so the same
    call can either fit the full observed curve (max_level=None) or simulate extrapolating from a
    small LLM pilot to predict a larger n_L not yet collected (see `predict_llm_excess`).

    NOT a theorem: n_L reduces estimation noise in what-hat, not the population Delta_W (which
    the human-side planner already estimates), so `c` is a fitted finite-sample term to be added
    to the proved human-side expansion in `joint_allocation`, not a substitute for it. Checked
    against Chatbot Arena's DIAL curve (the one of the two presented methods that is monotone in
    n_L, so a decay curve is meaningful to fit at all): extrapolating excess from just the two
    smallest pilot levels systematically over-predicts it (a conservative bias) by about 10% at
    2x the fit range, growing to 25-30% at 40x; a three-point pilot roughly halves the error at
    the far end, and a four-point pilot roughly halves it again. Kendall's tau extrapolates more
    accurately in relative terms over the same range (roughly 2-4% at a two-point pilot, shrinking
    the same way as more pilot points are added), since its dynamic range is far more compressed
    than log loss's. Treat any extrapolation beyond about a 5x range as a rough, optimistic- (for
    excess) or pessimistic- (for tau) leaning bound rather than a precise prediction.
    """
    sub = df[(df.dataset == dataset) & (df.sweep == "llm_budget") & (df.panel == panel) & (df.method == method) & (~df.failed)]
    if n_H_level is not None:
        sub = sub[sub.n_H_level == n_H_level]
    if max_level is not None:
        sub = sub[sub.level <= max_level]
    g = sub.groupby("level")[metric].mean()
    if len(g) < 2:
        return dict(c=np.nan, a=np.nan, n=len(g))
    x = 1.0 / g.index.to_numpy(dtype=float)
    y = g.to_numpy(dtype=float)
    (a, c), *_ = np.linalg.lstsq(np.column_stack([np.ones_like(x), x]), y, rcond=None)
    return dict(c=float(c), a=float(a), n=int(len(g)))


def predict_llm_excess(fit, levels):
    """Evaluate a `fit_llm_decay` fit (dict with keys "a", "c") at LLM budgets `levels`; metric-
    agnostic, so works for a `fit_llm_decay(..., metric="ref_kendall")` fit just as well."""
    levels = np.asarray(levels, dtype=float)
    return fit["a"] + fit["c"] / levels


def joint_allocation(pl_row, llm_decay_c, cost_ratio, total_budget=None, target_eps=None):
    """Item-2: cost-optimal split of a human/LLM label budget.

    Predicted excess is modeled as Delta_hat + A/n_human + B/n_L with A = (r+1)/2 (the proved
    dimension-reduction term of thm:population-risk-tradeoff) and B = llm_decay_c (the empirical
    1/n_L decay of `fit_llm_decay`) -- a practical heuristic combining a proved term with a fitted
    one, not a new theorem. `cost_ratio` = c_L / c_human, the cost of one LLM judgment in
    human-label-equivalent units. Both terms are proportional to 1/n, so minimizing A/n_human +
    B/n_L subject to a fixed total cost n_human + cost_ratio * n_L = total_budget (or minimizing
    cost for a target excess) has the classical Neyman (stratified-sampling) closed form
    n_human / n_L = sqrt(A * cost_ratio / B).
    """
    r, delta_hat = float(pl_row["r"]), float(pl_row["delta_hat"])
    A = (r + 1.0) / 2.0
    B = float(llm_decay_c) if np.isfinite(llm_decay_c) and llm_decay_c > 0 else 0.0
    rho = float(cost_ratio)
    root_A = np.sqrt(A)
    root_rhoB = np.sqrt(rho * B) if B > 0 else 0.0
    denom = root_A + root_rhoB
    if total_budget is not None:
        total_budget = float(total_budget)
        n_human = total_budget * root_A / denom
        excess = delta_hat + denom ** 2 / total_budget
    elif target_eps is not None:
        eps_room = max(float(target_eps) - delta_hat, 1e-12)
        total_budget = denom ** 2 / eps_room
        n_human = total_budget * root_A / denom
        excess = float(target_eps)
    else:
        raise ValueError("supply total_budget or target_eps")
    n_L = (total_budget - n_human) / rho if rho > 0 else np.inf
    return dict(n_human=float(n_human), n_L=float(n_L), total_budget=float(total_budget), predicted_excess=float(excess))


# --------------------------------------------------------------------------- tables
