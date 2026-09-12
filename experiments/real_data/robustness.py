"""Real-data robustness study (code/plan/2026-09-06-real-data-robustness-plan.md, Section 22).

LLM judgments are the recorded verdicts of the collected panel; the human test pool is
the only ground truth. Sweeps (configs/real_robustness.toml):

  order       A2  swapped-copy share with a canonical-first default order
  noise       B1  injected anti-consensus / position-only judges, one-sided display, small panel
  budget      C1  human budget n_H with the as-collected balanced LLM data
  llm_budget  C2  LLM rows subsampled to n_L at fixed human budgets
  spectest    B2  specification test of s_0 = alpha mu on Arena human subsets

Methods in every cell (presented): human-only BTL; Pooled (one BTL over all
judgments, no judge structure, no order term, plus a human-fitted scale);
Cons-Cal (order-effect structured model at the LLM rank r, consensus mu, one
human-fitted scale: the lambda = infinity endpoint of DIAL); DIAL (`dial_mu`: joint
weighted likelihood at rank r with the human score aligned to mu only, GACV weight);
DIAL-noPos (DIAL without the order term). Diagnostics: `staged_w`, `dial_w`,
`dial_mle_mu`, `dial_mle_w` (W-calibration and fixed-weight variants), `oracle_test`
(test-loss-minimizing weight on DIAL's path), `dial_rsel` ((r, lambda) by GACV).

Rows append to results/real_robustness/<dataset>/rows.jsonl keyed by
(sweep, panel, kind, level, n_H, seed). Aggregation: robustness_plot.py; figures:
notebooks/real_data_robustness.ipynb.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11, per pyproject's tomli fallback
    import tomli as tomllib  # type: ignore[no-redef]

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from dial_judge.baselines import fit_consensus_only_calibrated, fit_human_only_btl, fit_pooled_btl, fit_staged_structured_calibration
from dial_judge.benchmarks import fit_dial
from dial_judge.dial_model import fit_human_calibration
from dial_judge.evaluate import heldout_log_loss, score_accuracy
from dial_judge.gacv import btl_mle_exists, select_lambda
from dial_judge.hja import select_rank_by_bic
from dial_judge.inference import calibration_restriction_test, joint_sandwich, llm_only_sandwich

from .prepare import load_canonical

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "real_robustness.toml"
RESULTS_ROOT = ROOT / "results" / "real_robustness"


def _display_path(path: Path) -> Path:
    """Return *path* relative to ROOT when possible, otherwise return it unchanged."""
    try:
        return path.relative_to(ROOT)
    except ValueError:
        return path

SWEEPS = ("order", "noise", "noise_scarce", "budget", "llm_budget", "spectest")
NOISE_KINDS = ("random", "position", "anti")
METHODS = ("human_only", "pooled_cal", "consensus_cal", "dial_mu", "dial_nodeb", "staged_w", "dial_w", "dial_mle_mu", "dial_mle_w", "oracle_test", "dial_rsel")
DATASET_CODE = {"arena_33k": 1, "mt_bench": 2, "pandalm": 3}


def load_config(path=None):
    with open(path or CONFIG_PATH, "rb") as handle:
        return tomllib.load(handle)


# --------------------------------------------------------------------------- panel
def build_panel(frame, exclude=(), tie_rate_exclude=0.5, min_coverage=0.5, llm_ties="drop", human_ties="drop"):
    """Normalize one dataset's canonical table into index-coded LLM and human rows.

    Judges are dropped when listed in `exclude`, when their tie rate exceeds
    `tie_rate_exclude`, or when fewer than `min_coverage` of the records were
    collected in both display orders. Items are name-sorted so `display_order`
    from the adapter is reused unchanged. Ties: "drop" removes them, "half"
    keeps them as y = 0.5; the human test pool always keeps decisive labels.
    """
    n_records = frame["record_id"].nunique()
    tie_rate = frame.groupby("judge")["judge_tie"].mean()
    both = (frame.groupby(["judge", "record_id"])["display_order"].nunique() == 2).groupby(level=0).sum() / n_records
    drop = set(exclude) | set(tie_rate.index[tie_rate > tie_rate_exclude]) | set(both.index[both < min_coverage])
    judges = sorted(set(frame["judge"].unique()) - drop)
    items = sorted(set(frame["item_i"]) | set(frame["item_j"]))
    item_index = {it: n for n, it in enumerate(items)}
    judge_index = {j: k for k, j in enumerate(judges)}

    f = frame[frame["judge"].isin(judges)]
    tie = f["judge_tie"].astype(bool).to_numpy()
    kept = f if llm_ties == "half" else f[~tie]
    y_llm = np.where(kept["judge_tie"].astype(bool).to_numpy(), 0.5, (kept["outcome"] == kept["item_i"]).to_numpy(dtype=float))
    llm = pd.DataFrame({
        "record": kept["record_id"].to_numpy(),
        "k": kept["judge"].map(judge_index).to_numpy(dtype=int),
        "i": kept["item_i"].map(item_index).to_numpy(dtype=int),
        "j": kept["item_j"].map(item_index).to_numpy(dtype=int),
        "y": y_llm,
        "a": kept["display_order"].to_numpy(dtype=int),
        "swapped": kept["swapped"].to_numpy(dtype=bool) if "swapped" in kept else (kept["display_order"].to_numpy() == -1),
    })
    rec = frame.drop_duplicates("record_id")
    dec = rec["human_decisive"].astype(bool).to_numpy()
    hum = rec if human_ties == "half" else rec[dec]
    y_hum = np.where(hum["human_decisive"].astype(bool).to_numpy(), (hum["human_outcome"] == hum["item_i"]).to_numpy(dtype=float), 0.5)
    human = pd.DataFrame({
        "record": hum["record_id"].to_numpy(),
        "i": hum["item_i"].map(item_index).to_numpy(dtype=int),
        "j": hum["item_j"].map(item_index).to_numpy(dtype=int),
        "y": y_hum,
    })
    language = dict(zip(rec["record_id"], rec["language"])) if "language" in rec else {}
    return {
        "items": items, "judges": judges, "N": len(items), "K": len(judges),
        "dropped_judges": sorted(drop & set(frame["judge"].unique())),
        "records": np.array(sorted(frame["record_id"].unique())), "record_language": language,
        "llm": llm.reset_index(drop=True), "human": human.reset_index(drop=True),
        "n_llm_rows": int(len(f)), "n_llm_ties": int(f["judge_tie"].sum()),
        "n_records": int(n_records), "n_human_decisive": int(dec.sum()),
        "llm_ties": llm_ties, "human_ties": human_ties,
    }


_PANELS = {}


def get_panel(dataset, cfg):
    """Per-process cache of the built panel."""
    if dataset not in _PANELS:
        s = cfg["study"]
        _PANELS[dataset] = build_panel(load_canonical(dataset), exclude=s.get("exclude_judges", ()), tie_rate_exclude=s.get("tie_rate_exclude", 0.5),
                                       min_coverage=s.get("min_coverage", 0.5), llm_ties=s.get("llm_ties", "drop"), human_ties=s.get("human_ties", "drop"))
    return _PANELS[dataset]


def restrict_panel(panel, judge_names):
    """Sub-panel with judges re-indexed 0..K'-1 (the original order of `judge_names` is kept)."""
    if judge_names is None:
        return panel
    keep = [panel["judges"].index(j) for j in judge_names]
    sub = panel["llm"][panel["llm"]["k"].isin(keep)].copy()
    sub["k"] = sub["k"].map({k: n for n, k in enumerate(keep)}).astype(int)
    return dict(panel, llm=sub.reset_index(drop=True), judges=list(judge_names), K=len(keep))


# --------------------------------------------------------------------------- design
def split_records(records, f_test, rng):
    perm = rng.permutation(np.asarray(records))
    n_test = int(round(f_test * len(perm)))
    return set(perm[:n_test].tolist()), set(perm[n_test:].tolist())


def thin_display_order(llm, rho_swap, rng):
    """Keep every canonical-first row (a = +1); keep swapped rows with probability rho_swap."""
    if rho_swap >= 1.0:
        return llm
    keep = (llm["a"].to_numpy() == 1) | (rng.random(len(llm)) < rho_swap)
    return llm[keep].reset_index(drop=True)


def inject_noise_judges(llm, K, m, kind, rng, first_prob=0.9):
    """Append m synthetic judges (indices K..K+m-1) answering every (record, order) in `llm`."""
    if m <= 0:
        return llm, K
    if kind not in NOISE_KINDS:
        raise ValueError(f"unknown noise kind {kind!r}")
    pairs = llm.drop_duplicates("record")[["record", "i", "j"]]
    base = pd.concat([pairs.assign(a=1, swapped=False), pairs.assign(a=-1, swapped=True)], ignore_index=True)
    parts = [llm]
    for t in range(m):
        k = K + t
        if kind == "anti":
            src = int(rng.integers(K))
            s = llm[llm["k"] == src]
            parts.append(s.assign(k=k, y=1.0 - s["y"]))
            continue
        if kind == "random":
            y = (rng.random(len(base)) < 0.5).astype(float)
        else:
            first = rng.random(len(base)) < first_prob
            y = np.where(base["a"].to_numpy() == 1, first, ~first).astype(float)
        parts.append(base.assign(k=k, y=y))
    return pd.concat(parts, ignore_index=True), K + m


def amplify_bias(llm, theta, rng):
    if theta <= 0:
        return llm
    hit = rng.random(len(llm)) < theta
    return llm.assign(y=np.where(hit, (llm["a"].to_numpy() == 1).astype(float), llm["y"].to_numpy()))


def draw_budget(human, n_H, rng):
    if n_H is None or n_H < 0 or n_H >= len(human):
        return human
    idx = rng.choice(len(human), size=int(n_H), replace=False)
    return human.iloc[np.sort(idx)].reset_index(drop=True)


def drop_empty_judges(llm, K, K_real):
    """Remove judges left with no rows (after subsampling) and renumber the rest, keeping the
    real judges before the injected ones. Returns (llm, K, K_real, n_dropped)."""
    present = np.sort(llm["k"].unique().astype(int))
    if present.size == K:
        return llm, K, K_real, 0
    remap = {int(old): new for new, old in enumerate(present)}
    llm = llm.assign(k=llm["k"].map(remap).astype(int))
    return llm, int(present.size), int(np.sum(present < K_real)), int(K - present.size)


def subsample_rows(llm, n_L, rng):
    if n_L is None or n_L < 0 or n_L >= len(llm):
        return llm
    idx = rng.choice(len(llm), size=int(n_L), replace=False)
    return llm.iloc[np.sort(idx)].reset_index(drop=True)


# --------------------------------------------------------------------------- arrays
def llm_arrays(llm, N, K):
    n_order = np.zeros((K, N, N, 2))
    y_order = np.zeros((K, N, N, 2))
    k = llm["k"].to_numpy(dtype=int)
    i = llm["i"].to_numpy(dtype=int)
    j = llm["j"].to_numpy(dtype=int)
    a_idx = (llm["a"].to_numpy() > 0).astype(int)
    np.add.at(n_order, (k, i, j, a_idx), 1.0)
    np.add.at(y_order, (k, i, j, a_idx), llm["y"].to_numpy(dtype=float))
    return n_order.sum(-1), y_order.sum(-1), n_order, y_order


def human_pairs(human):
    g = human.groupby(["i", "j"])["y"].agg(["size", "sum"])
    return [(int(i), int(j), float(n), float(s)) for (i, j), (n, s) in g.iterrows()]


def human_records(human):
    return list(zip([0] * len(human), human["i"].to_numpy(dtype=int), human["j"].to_numpy(dtype=int), human["y"].to_numpy(dtype=float)))


# --------------------------------------------------------------------------- rank / clean fit
def select_rank(dataset, cfg, rank_cap=None):
    panel = get_panel(dataset, cfg)
    N, K = panel["N"], panel["K"]
    cap = N - 3 if rank_cap is None else rank_cap
    r_max = max(0, min(K - 1, N - 2, cap))
    n_ijk, y_ijk, n_order, y_order = llm_arrays(panel["llm"], N, K)
    best, table = select_rank_by_bic(N, K, n_ijk, y_ijk, candidate_ranks=list(range(r_max + 1)), n_order=n_order, y_order=y_order)
    return best, {r: float(v["bic"]) for r, v in table.items()}


def clean_fit(dataset, cfg, cluster="pair", r=1):
    """Full-panel fits behind the judge diagnostics: the joint MLE with every decisive human label and,
    for the per-judge order effects, the LLM-only fit; both with cell-clustered sandwich intervals."""
    panel = get_panel(dataset, cfg)
    N, K = panel["N"], panel["K"]
    n_ijk, y_ijk, n_order, y_order = llm_arrays(panel["llm"], N, K)
    pairs = human_pairs(panel["human"])
    st = fit_staged_structured_calibration(N, K, r, n_ijk, y_ijk, pairs, n_order=n_order, y_order=y_order)
    jt = fit_dial(N, K, r, n_ijk, y_ijk, pairs, n_order=n_order, y_order=y_order, init_params=(st["gamma"], st["mu"], st["U"], st["V"], st["b"]), tol=1e-6, max_steps=30)
    sw = joint_sandwich(jt, N, K, r, pairs, n_ijk, y_ijk, n_order, y_order, cluster=cluster)
    # the judge diagnostic is an LLM-side quantity, so it is reported from the LLM-only fit
    # (the lambda = infinity fit, `st`) with the cell-clustered sandwich of app:clustered
    sw_llm = llm_only_sandwich(st, N, K, r, n_ijk=n_ijk, y_ijk=y_ijk, n_order=n_order, y_order=y_order, cluster=cluster)
    return dict(dataset=dataset, N=N, K=K, r=r, items=panel["items"], judges=panel["judges"], dropped_judges=panel["dropped_judges"],
                n_L=float(n_order.sum()), n_H=float(sum(p[2] for p in pairs)), n_records=panel["n_records"], n_llm_ties=panel["n_llm_ties"], n_human_decisive=panel["n_human_decisive"],
                b=[float(x) for x in jt["b"]], b_lower=[float(x) for x in sw["b"]["lower"]], b_upper=[float(x) for x in sw["b"]["upper"]],
                b_llm=[float(x) for x in sw_llm["b"]["estimate"]], b_llm_lower=[float(x) for x in sw_llm["b"]["lower"]],
                b_llm_upper=[float(x) for x in sw_llm["b"]["upper"]],
                gamma=[float(x) for x in jt["gamma"]], mu=[float(x) for x in jt["mu"]], s_H=[float(x) for x in jt["s_H"]],
                s_human_only=[float(x) for x in fit_human_only_btl(N, pairs)["s_H"]], lam=float(jt["lam"]), staged_b=[float(x) for x in st["b"]], converged=bool(jt["fit_info"]["converged"]))


# --------------------------------------------------------------------------- one cell
def _gacv_rank_select(N, K, pairs, A, use_order, lam_grid, r_max, staged_cache):
    """GACV over candidate ranks (each with its own lambda path); returns (r, selection, table)."""
    best, table = None, []
    for r_c in range(r_max + 1):
        try:
            key = (r_c, use_order)
            if key not in staged_cache:
                staged_cache[key] = fit_staged_structured_calibration(N, K, r_c, A[0], A[1], pairs, n_order=A[2] if use_order else None, y_order=A[3] if use_order else None)
            st_c = staged_cache[key]
            sel_c = select_lambda(N, K, r_c, pairs, n_ijk_llm=A[0], y_ijk_llm=A[1], n_order=A[2] if use_order else None, y_order=A[3] if use_order else None,
                                  staged_fit=st_c, lambda_grid=lam_grid, return_all=(r_c == 0))
            table.append(dict(r=r_c, gacv=float(sel_c["gacv"]), lam=float(sel_c["lam"])))
            if best is None or sel_c["gacv"] < best[1]["gacv"]:
                best = (r_c, sel_c)
        except Exception as e:  # noqa: BLE001
            table.append(dict(r=r_c, error=repr(e)[:100]))
    if best is None:
        raise RuntimeError("no rank candidate fitted")
    return best[0], best[1], table


def per_dataset(value, dataset):
    """Config entries may be a scalar/list (shared) or a table keyed by dataset."""
    return value[dataset] if isinstance(value, dict) else value


def run_cell(job):
    dataset, sweep, panel_name, kind, level, n_H_req, seed, cfg = job
    scfg, dcfg, swcfg = cfg["study"], cfg[dataset], cfg["sweeps"][sweep]
    panel = restrict_panel(get_panel(dataset, cfg), None if panel_name == "all" else cfg["panels"][panel_name])
    code = DATASET_CODE[dataset]
    rng_split = np.random.default_rng([int(seed), code, 1])
    rng_budget = np.random.default_rng([int(seed), code, 2])
    rng_design = np.random.default_rng([int(seed), code, 3])
    t0 = time.perf_counter()

    rho = float(swcfg.get("rho_swap", 1.0))
    m, n_L_req = 0, -1
    n_H = int(n_H_req) if n_H_req is not None else int(dcfg["n_H"])
    if sweep == "order":
        rho = float(level)
    elif sweep in ("noise", "noise_scarce"):
        m = int(level)
        if sweep == "noise_scarce":
            n_H = int(per_dataset(swcfg.get("n_H_fixed", n_H), dataset))
    elif sweep == "budget":
        n_H = int(level)
    elif sweep == "llm_budget":
        n_L_req = int(level)
    else:
        raise ValueError(f"unknown sweep {sweep!r}")

    test, train = split_records(panel["records"], dcfg["f_test"], rng_split)
    llm = panel["llm"][panel["llm"]["record"].isin(train)].reset_index(drop=True)
    K_real = panel["K"]
    if sweep == "noise_scarce":
        llm = subsample_rows(llm, int(per_dataset(swcfg["n_L_base"], dataset)), rng_design)      # scarce base sample before injection
    llm, K = inject_noise_judges(llm, K_real, m, kind, rng_design, first_prob=scfg.get("position_noise_first_prob", 0.9))
    llm = thin_display_order(llm, rho, rng_design)
    llm = subsample_rows(llm, n_L_req, rng_design)
    llm, K, K_real, K_dropped = drop_empty_judges(llm, K, K_real)   # a judge without rows has no estimable parameters
    hum_train = panel["human"][panel["human"]["record"].isin(train)].reset_index(drop=True)
    hum_test = panel["human"][panel["human"]["record"].isin(test) & (panel["human"]["y"] != 0.5)].reset_index(drop=True)
    cal = draw_budget(hum_train, n_H, rng_budget)

    N = panel["N"]
    A = llm_arrays(llm, N, K)
    pairs = human_pairs(cal)
    test_recs = human_records(hum_test)
    n_L, n_H_actual = int(len(llm)), int(len(cal))
    mults = scfg.get("lambda_multipliers")
    lam_grid = [float(mm) * n_L / n_H_actual for mm in mults] if mults else None
    r_max = int(min(scfg.get("rank_select_max", 2), K - 1, N - 2))

    ref = fit_human_only_btl(N, human_pairs(hum_test))["s_H"]
    floor = heldout_log_loss(ref, test_recs)
    r_llm = int(min(scfg.get("llm_rank", 1), K - 1, N - 2))          # structural rank of the LLM side
    base = dict(dataset=dataset, sweep=sweep, panel=panel_name, kind=kind, level=float(level), n_H=n_H_actual, n_H_level=int(n_H_req) if n_H_req is not None else -1, seed=int(seed),
                N=N, K_real=K_real, K=K, K_dropped=K_dropped, r=r_llm, n_L=n_L, n_test=int(len(hum_test)), n_train_human=int(len(hum_train)), rho_swap=rho, m=int(m),
                first_share=float(np.mean(llm["a"].to_numpy() == 1)), floor=floor, human_only_exists=bool(btl_mle_exists(N, pairs)))
    out = []
    only = cfg.get("_only_methods")

    def want(*names):
        return only is None or any(mm in only for mm in names)

    def rec(method, s_hat, **extra):
        s_hat = np.asarray(s_hat, dtype=float)
        ll = heldout_log_loss(s_hat, test_recs)
        row = dict(base, method=method, logloss=ll, excess=ll - floor, acc=score_accuracy(s_hat, test_recs), ref_kendall=float(kendalltau(ref, s_hat).statistic), max_abs=float(np.max(np.abs(s_hat))))
        row.update(extra)
        out.append(row)

    def fail(method, e):
        out.append(dict(base, method=method, error=repr(e) + traceback.format_exc()[-200:]))

    def inj(fit):
        b = np.asarray(fit["b"], float)
        g = np.asarray(fit["gamma"], float)
        d = dict(b_mean_abs=float(np.mean(np.abs(b[:K_real]))))
        if K > K_real:
            d.update(b_inj_mean_abs=float(np.mean(np.abs(b[K_real:]))), gamma_inj_mean=float(np.mean(g[K_real:])), gamma_inj_mean_abs=float(np.mean(np.abs(g[K_real:]))))
        return d

    def flags(fit):
        """Convergence and separation diagnostics of a fitted candidate (human-only endpoint: none)."""
        fi = fit.get("fit_info", {}) or {}
        return dict(converged=bool(fi.get("converged", True)), b_at_bound=int(fi.get("b_at_bound", 0)), n_at_bound=int(fi.get("n_at_bound", 0)), inner_limit_hits=int(fi.get("inner_limit_hits", 0)),
                    max_abs_S=float(fi.get("max_abs_S", np.nan)), max_gamma=float(fi.get("max_gamma", np.nan)), polish_grad=float(fi.get("polish_grad_norm", np.nan)))

    def lam_fields(lam):
        lam = float(lam)
        return dict(lam=lam, lam_rel=(lam * n_H_actual / n_L) if np.isfinite(lam) else float("inf"))

    # ---- baselines
    if want("human_only"):
        try:
            rec("human_only", fit_human_only_btl(N, pairs)["s_H"], lam=0.0)
        except Exception as e:  # noqa: BLE001
            fail("human_only", e)
    if want("pooled_cal"):
        try:
            pooled = fit_pooled_btl(N, A[0], A[1])["score"]
            al, _ = fit_human_calibration(pooled, np.zeros((N, 0)), pairs)
            rec("pooled_cal", al * pooled, alpha=float(al))
        except Exception as e:  # noqa: BLE001
            fail("pooled_cal", e)

    # ---- Cons-Cal (staged endpoint of DIAL) and DIAL: LLM side at rank r_llm, human score aligned to mu
    st_mu = None
    if want("consensus_cal", "dial_mu", "oracle_test", "dial_mle_mu"):
        try:
            st_mu = fit_consensus_only_calibrated(N, K, r_llm, A[0], A[1], pairs, n_order=A[2], y_order=A[3])
            if want("consensus_cal"):
                rec("consensus_cal", st_mu["s_H"], **lam_fields(np.inf), alpha=float(st_mu["alpha_H"]), **flags(st_mu), **inj(st_mu))
        except Exception as e:  # noqa: BLE001
            fail("consensus_cal", e)
    if st_mu is not None and want("dial_mu", "oracle_test"):
        try:
            sel = select_lambda(N, K, r_llm, pairs, n_ijk_llm=A[0], y_ijk_llm=A[1], n_order=A[2], y_order=A[3], staged_fit=st_mu, lambda_grid=lam_grid, return_all=True, align="mu")
            lam = float(sel["lam"])
            selected_regular = next(p["regular"] for p in sel["gacv_path"] if p["lam"] == lam)
            rec("dial_mu", sel["s_H"], **lam_fields(lam), gacv=float(sel["gacv"]), selected_regular=bool(selected_regular), n_dropped=len(sel["dropped"]), **flags(sel), **(inj(sel) if lam > 0 else {}))
            if cfg.get("_gacv_endpoint_margins"):
                from .endpoint_margin import choose_endpoint_margin
                candidates = dict(sel["candidates"])
                for margin in cfg["_gacv_endpoint_margins"]:
                    chosen, reason, gap = choose_endpoint_margin(sel["gacv_path"], n_H_actual, margin, fallback=lam)
                    fitted = candidates[chosen]
                    point = next(p for p in sel["gacv_path"] if p["lam"] == chosen)
                    rec(f"dial_margin_{margin:g}", fitted["s_H"], **lam_fields(chosen), margin_c=margin, margin_reason=reason, endpoint_gacv_gap=gap, gacv=float(point["gacv"]), selected_regular=bool(point["regular"]), n_dropped=len(sel["dropped"]), **flags(fitted))
            if want("oracle_test"):
                losses = [(l, heldout_log_loss(f["s_H"], test_recs), f) for l, f in sel["candidates"]]
                lam_o, _, f_o = min(losses, key=lambda x: x[1])
                rec("oracle_test", f_o["s_H"], **lam_fields(lam_o))
        except Exception as e:  # noqa: BLE001
            fail("dial_mu", e)
    if st_mu is not None and want("dial_mle_mu"):
        try:
            jt = fit_dial(N, K, r_llm, A[0], A[1], pairs, n_order=A[2], y_order=A[3], init_params=(st_mu["gamma"], st_mu["mu"], st_mu["U"], st_mu["V"], st_mu["b"]), tol=1e-6, max_steps=30, align="mu")
            rec("dial_mle_mu", jt["s_H"], **lam_fields(jt["lam"]), **flags(jt))
        except Exception as e:  # noqa: BLE001
            fail("dial_mle_mu", e)

    # ---- DIAL-noPos: same as DIAL without the order term
    if want("dial_nodeb"):
        try:
            st0 = fit_consensus_only_calibrated(N, K, r_llm, A[0], A[1], pairs)
            sel0 = select_lambda(N, K, r_llm, pairs, n_ijk_llm=A[0], y_ijk_llm=A[1], staged_fit=st0, lambda_grid=lam_grid, align="mu")
            rec("dial_nodeb", sel0["s_H"], **lam_fields(sel0["lam"]), n_dropped=len(sel0["dropped"]), **flags(sel0))
        except Exception as e:  # noqa: BLE001
            fail("dial_nodeb", e)

    # ---- W-calibration diagnostics at the same LLM rank
    if r_llm >= 1 and want("staged_w", "dial_w", "dial_mle_w"):
        try:
            st_w = fit_staged_structured_calibration(N, K, r_llm, A[0], A[1], pairs, n_order=A[2], y_order=A[3])
            if want("staged_w"):
                rec("staged_w", st_w["s_H"], **lam_fields(np.inf), **flags(st_w))
            if want("dial_w"):
                selw = select_lambda(N, K, r_llm, pairs, n_ijk_llm=A[0], y_ijk_llm=A[1], n_order=A[2], y_order=A[3], staged_fit=st_w, lambda_grid=lam_grid, align="W")
                rec("dial_w", selw["s_H"], **lam_fields(selw["lam"]), n_dropped=len(selw["dropped"]), **flags(selw))
            if want("dial_mle_w"):
                jtw = fit_dial(N, K, r_llm, A[0], A[1], pairs, n_order=A[2], y_order=A[3], init_params=(st_w["gamma"], st_w["mu"], st_w["U"], st_w["V"], st_w["b"]), tol=1e-6, max_steps=30, align="W")
                rec("dial_mle_w", jtw["s_H"], **lam_fields(jtw["lam"]), **flags(jtw))
        except Exception as e:  # noqa: BLE001
            fail("dial_w", e)

    # ---- rank-selected W-calibration (appendix diagnostic): (r, lambda) by GACV over r in 0..r_max
    if want("dial_rsel"):
        try:
            r_sel, selr, table = _gacv_rank_select(N, K, pairs, A, True, lam_grid, r_max, {})
            rec("dial_rsel", selr["s_H"], r_sel=r_sel, **lam_fields(selr["lam"]), n_dropped=len(selr["dropped"]), **flags(selr), rank_table=table)
        except Exception as e:  # noqa: BLE001
            fail("dial_rsel", e)

    secs = time.perf_counter() - t0
    for row in out:
        row["seconds_cell"] = secs
    return out


def run_spectest(job):
    """B2: LR test of s_0 = alpha mu on a human subset, with the consensus from global or subset LLM data."""
    setting, n_H, seed, cfg = job
    panel = get_panel("arena_33k", cfg)
    N, K = panel["N"], panel["K"]
    lang = panel["record_language"]
    is_en = {r: (lang.get(r) == "English") for r in panel["records"]}
    hum_recs = [r for r in panel["records"] if (is_en[r] if setting.startswith("english") else not is_en[r])]
    llm_recs = hum_recs if setting.endswith("subset") else list(panel["records"])
    hum = panel["human"][panel["human"]["record"].isin(set(hum_recs))]
    llm = panel["llm"][panel["llm"]["record"].isin(set(llm_recs))]
    rng = np.random.default_rng([int(seed), 11])
    perm = rng.permutation(np.array(sorted(hum["record"].unique())))
    test = set(perm[: int(round(cfg["sweeps"]["spectest"].get("f_test", 0.5) * len(perm)))].tolist())
    hum_tr = hum[~hum["record"].isin(test)]
    hum_te = hum[hum["record"].isin(test)]
    llm = llm[~llm["record"].isin(test)]                      # LLM rows on test records are held out too
    A = llm_arrays(llm, N, K)
    r_llm = int(min(cfg["study"].get("llm_rank", 1), K - 1, N - 2))
    mu = fit_staged_structured_calibration(N, K, r_llm, A[0], A[1], [(0, 1, 2.0, 1.0)], n_order=A[2], y_order=A[3])["mu"]
    cal = draw_budget(hum_tr, n_H, np.random.default_rng([int(seed), 12, int(n_H)]))
    pairs = human_pairs(cal)
    test_recs = human_records(hum_te)
    floor = heldout_log_loss(fit_human_only_btl(N, human_pairs(hum_te))["s_H"], test_recs)
    t = calibration_restriction_test(N, pairs, mu)
    return [dict(dataset="arena_33k", sweep="spectest", panel="all", kind=setting, level=float(n_H), n_H=int(len(cal)), seed=int(seed), N=N, K=K, n_L=int(len(llm)),
                 n_test=int(len(hum_te)), stat=t["stat"], df=t["df"], pvalue=t["pvalue"], reject05=bool(t["pvalue"] < 0.05), human_only_exists=t["human_only_exists"],
                 excess_restricted=heldout_log_loss(t["s_restricted"], test_recs) - floor, excess_full=heldout_log_loss(t["s_full"], test_recs) - floor,
                 tau_mu_vs_test_human=float(kendalltau(mu, fit_human_only_btl(N, human_pairs(hum_te))["s_H"]).statistic), method="spectest")]


# --------------------------------------------------------------------------- driver
def jobs_for(sweep, cfg, seeds, smoke=False):
    sw = cfg["sweeps"][sweep]
    jobs = []
    if sweep == "spectest":
        levels = sw["levels"][:2] if smoke else sw["levels"]
        return [(s, nH, seed, cfg) for s in sw["settings"] for nH in levels for seed in seeds]
    for d in sw["datasets"]:
        for p in sw["panels"]:
            if sweep == "budget":
                levels = sw["levels"][d]
                levels = levels[:2] if smoke else levels
                jobs += [(d, sweep, p, "none", lv, None, s, cfg) for lv in levels for s in seeds]
            elif sweep == "llm_budget":
                levels = per_dataset(sw["levels"], d)
                levels = levels[:2] if smoke else levels
                jobs += [(d, sweep, p, "none", lv, nH, s, cfg) for lv in levels for nH in per_dataset(sw["n_H_grid"], d) for s in seeds]
            elif sweep in ("noise", "noise_scarce"):
                kinds = sw["kinds"][:1] if smoke else sw["kinds"]
                levels = sw["levels"][:2] if smoke else sw["levels"]
                for kind in kinds:
                    jobs += [(d, sweep, p, kind, lv, None, s, cfg) for lv in levels if (lv > 0 or kind == kinds[0]) for s in seeds]
            else:
                levels = sw["levels"][:2] if smoke else sw["levels"]
                jobs += [(d, sweep, p, "none", lv, None, s, cfg) for lv in levels for s in seeds]
    return jobs


def job_key(job):
    if len(job) == 4:  # spectest
        return ("spectest", "all", job[0], float(job[1]), -1, int(job[2]))
    d, sweep, p, kind, lv, nH, s, _ = job
    return (sweep, p, kind, float(lv), int(nH) if nH is not None else -1, int(s))


def existing_keys(path, only=None):
    """Keys already stored; with `only`, a key counts as done only if every listed method has a row for it."""
    keys, seen = set(), {}
    if path.exists():
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                key = (d["sweep"], d["panel"], d["kind"], float(d["level"]), int(d.get("n_H_level", d.get("n_0_level", -1))), int(d["seed"]))
                if only is None:
                    keys.add(key)
                else:
                    seen.setdefault(key, set()).add(d["method"])
    if only is not None:
        keys = {k for k, ms in seen.items() if set(only) <= ms}
    return keys


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep", default="all", choices=list(SWEEPS) + ["all"])
    p.add_argument("--seeds", default="0:3", help="start:stop seed range (stop exclusive)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--config", default=None)
    p.add_argument("--out", default=None, help="results root (default results/real_robustness)")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--select-rank", action="store_true")
    p.add_argument("--clean-fit", action="store_true")
    p.add_argument("--ja", action="store_true", help="run the JA-Ranking reanalysis (figure A1) and exit")
    p.add_argument("--methods", default=None, help="comma-separated method keys: compute only these, for cells that lack them (spectest skipped)")
    a = p.parse_args(argv)
    cfg = load_config(a.config)
    root = Path(a.out) if a.out else RESULTS_ROOT
    if a.select_rank:
        for d in cfg["study"]["datasets"]:
            best, table = select_rank(d, cfg)
            print(f"{d}: BIC rank = {best}; " + ", ".join(f"r={r}: {v:.1f}" for r, v in table.items()), flush=True)
        return
    if a.clean_fit:
        for d in cfg["study"]["datasets"]:
            res = clean_fit(d, cfg)
            (root / d).mkdir(parents=True, exist_ok=True)
            (root / d / "clean_fit.json").write_text(json.dumps(res, indent=1) + "\n")
            print(f"{d}: clean fit written, converged={res['converged']}", flush=True)
        return
    if a.ja:
        from .ja_reanalysis import run as run_ja

        tab, judges = run_ja(cfg)
        pd.set_option("display.width", 250)
        print(tab.round(3).to_string(index=False))
        print(judges.round(3).to_string(index=False))
        return

    s0, s1 = (int(x) for x in a.seeds.split(":"))
    seeds = range(s0, s1)
    sweeps = list(SWEEPS) if a.sweep == "all" else [a.sweep]
    only = [m.strip() for m in a.methods.split(",")] if a.methods else None
    if only:
        cfg["_only_methods"] = only
        sweeps = [sw for sw in sweeps if sw != "spectest"]
    jobs = []
    for sw in sweeps:
        jobs += jobs_for(sw, cfg, seeds, smoke=a.smoke)
    files, done = {}, {}
    for d in cfg["study"]["datasets"]:
        path = root / d / "rows.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        files[d], done[d] = path, existing_keys(path, only)
    def _dataset_of(j):
        return j[0] if len(j) == 8 else "arena_33k"    # spectest jobs are Arena-only

    jobs = [j for j in jobs if job_key(j) not in done[_dataset_of(j)]]
    print(f"{len(jobs)} cells to run -> {_display_path(root)}", flush=True)
    t0 = time.perf_counter()
    handles = {d: open(path, "a") for d, path in files.items()}
    try:
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            futures = [ex.submit(run_spectest if len(j) == 4 else run_cell, j) for j in jobs]
            for n, fut in enumerate(futures):
                recs = fut.result()
                for row in recs:
                    handles[row["dataset"]].write(json.dumps(row) + "\n")
                handles[recs[0]["dataset"]].flush()
                if (n + 1) % 20 == 0 or n + 1 == len(jobs):
                    print(f"{n + 1}/{len(jobs)} cells, {time.perf_counter() - t0:.0f}s", flush=True)
    finally:
        for h in handles.values():
            h.close()
    print("done", flush=True)


def describe(dataset):
    return {"study": "robustness", "dataset": dataset, "sweeps": list(SWEEPS), "methods": list(METHODS), "evaluation": "held_out_human_comparisons",
            "status": "implemented", "plan": "code/plan/2026-09-06-real-data-robustness-plan.md (Section 22)"}


if __name__ == "__main__":
    main()
