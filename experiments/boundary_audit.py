"""Finite-box boundary audit of the fits behind the reported figures (manuscript App. E.3.2).

Every optimizer works on the box |theta_j| <= 10 of the centering-reduced chart. This audit refits,
cell by cell and with the data rebuilt exactly as the study runners build it, the DIAL fits that the
figures draw, and records for each fit whether the optimizer's final chart point has a coordinate on
the box, split by parameter block (fit_info keys set by `hja.boundary_activity`, at the optimizer's
own active-bound tolerance):

  boundary_b          an order-effect coordinate b_k
  boundary_llm_other  a loading / consensus / heterogeneity coordinate (gamma, U, mu, V)
  boundary_cal        a calibration coordinate (alpha, a); for the lambda = infinity calibration,
                      the retry box of `fit_human_calibration`, which bounds the calibrated score
  boundary_any        any of the above

Fit stages: `stage1` (LLM-only fit of the staged endpoint), `candidate` (finite-lambda GACV
candidates; `selected` marks the GACV choice, `retained` the guard), `endpoint_inf` (lambda = infinity
calibration). Every cell also writes one `selection` record (selected lambda and max |s_H|) that is
checked against the stored study rows.

    python -m experiments.boundary_audit run --workers 16
    python -m experiments.boundary_audit summarize
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from dial_judge.baselines import fit_consensus_only_calibrated, fit_staged_structured_calibration
from dial_judge.gacv import select_lambda

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "boundary_audit"
SEEDS = range(50)
FLAGS = ["boundary_any", "boundary_b", "boundary_llm_other", "boundary_cal"]

NH_ABUNDANT = {"arena_33k": 300, "mt_bench": 80, "pandalm": 60}   # notebooks/2: order and noise sweeps; G8 n_H
NH_SCARCE = {"arena_33k": 1000, "mt_bench": 80, "pandalm": 60}    # notebooks/2: biased5 llm_budget n_H level
R1_METHODS = {"main10": ["dial_mu", "dial_nodeb", "dial_w"], "app20": ["dial_mu", "dial_nodeb", "dial_w"],
              "main10_mis": ["dial_mu", "dial_nodeb", "dial_w"], "main10_pos": ["dial_mu", "dial_nodeb"]}


# --------------------------------------------------------------------------- cells
def r1_jobs():
    from experiments.simulation.r1_main import CONFIGS, cells_for

    rows = {"main10": ["nH", "nL"], "app20": ["nH", "nL"], "main10_mis": ["nH", "nL"], "main10_pos": ["pos"]}
    jobs = []
    for cfg, rs in rows.items():
        for row in rs:
            for n_L, n_H, sp in cells_for(cfg, row):
                jobs.append(dict(source="r1", key=["r1", cfg, row, n_L, n_H, float(sp)], seed_args=(cfg, row, n_L, n_H, sp), methods=R1_METHODS[cfg]))
    return [dict(j, key=j["key"] + [s], seed_args=j["seed_args"] + (s,)) for j in jobs for s in SEEDS]


def real_jobs():
    """Cells of Figures 3, 4 and G4, G6--G8, with the DIAL variants each figure draws."""
    from experiments.real_data import intermediate_budget as ib
    from experiments.real_data import robustness as rb

    cfg = rb.load_config()
    methods, by_key = {}, {}

    def add(job, ms):
        k = (job[0],) + rb.job_key(job)
        by_key[k] = job
        methods.setdefault(k, set()).update(ms)

    for sweep in ("order", "noise", "noise_scarce", "budget", "llm_budget"):
        for job in rb.jobs_for(sweep, cfg, SEEDS):
            d, _, panel, kind, level, nH, _, _ = job
            if sweep == "order" and panel == "biased5":
                if level in (1.0, 0.05):
                    add(job, ["dial_mu", "dial_w", "dial_nodeb"])        # Figure 3
                if d == "arena_33k":
                    add(job, ["dial_mu", "dial_nodeb"])                  # G4(a)
            elif sweep == "noise" and panel == "biased5" and kind == "position":
                if level == 8:
                    add(job, ["dial_mu", "dial_w", "dial_nodeb"])        # Figure 3
                if d == "arena_33k":
                    add(job, ["dial_mu", "dial_nodeb"])                  # G4(b, d)
            elif sweep == "noise_scarce" and panel == "biased5" and kind == "anti" and d == "arena_33k":
                add(job, ["dial_mu", "dial_nodeb"])                      # G4(c, f)
            elif sweep == "budget":
                add(job, ["dial_mu"] + (["dial_nodeb"] if panel == "all" else []))   # Figure 4(a), G6 top, G7
            elif sweep == "llm_budget":
                if panel == "biased5" and nH == NH_SCARCE[d]:
                    add(job, ["dial_mu", "dial_nodeb"])                  # G4(e), G6 bottom
                if panel in ("all", "small6", "large6") and nH == NH_ABUNDANT[d]:
                    add(job, ["dial_mu"])                                # G8
    jobs = [dict(source="robustness", key=["robustness"] + list(k), job=by_key[k], methods=sorted(ms)) for k, ms in methods.items()]
    icfg = ib.build_config()
    for job in rb.jobs_for("llm_budget", icfg, SEEDS):
        if job[0] == "arena_33k" and job[2] == "all":                    # Figure 4(b)
            jobs.append(dict(source="intermediate", key=["intermediate", job[0]] + list(rb.job_key(job)), job=job, methods=["dial_mu"]))
    return jobs


def clean_jobs():
    return [dict(source="clean_fit", key=["clean_fit", d], dataset=d, methods=[]) for d in ("arena_33k", "mt_bench", "pandalm")]   # G5(a)


def all_jobs():
    return clean_jobs() + r1_jobs() + real_jobs()


# --------------------------------------------------------------------------- one cell
def _flags(info, keys=FLAGS):
    info = info or {}
    return {k: (bool(info[k]) if k in info else None) for k in keys}


def _audit_selection(method, st, sel, N, K):
    recs = []
    fi = st["fit_info"]
    recs.append(dict(method=method, stage="stage1", lam=float("inf"), selected=None, retained=None, converged=bool(fi.get("converged", True)),
                     max_abs_coord=fi.get("max_abs_coord"), **_flags(fi, ["boundary_any", "boundary_b", "boundary_llm_other"]), boundary_cal=None))
    regular = {p["lam"]: bool(p["regular"]) for p in sel["gacv_path"]}
    lam_sel = float(sel["lam"])
    ci = st.get("cal_info", {})
    recs.append(dict(method=method, stage="endpoint_inf", lam=float("inf"), selected=bool(lam_sel == np.inf), retained=regular.get(np.inf), converged=True,
                     max_abs_coord=ci.get("max_abs_coord"), boundary_any=ci.get("boundary_cal"), boundary_b=None, boundary_llm_other=None,
                     boundary_cal=ci.get("boundary_cal"), cal_box_used=ci.get("cal_box_used")))
    for lam, fit in sel["candidates"]:
        if not (0 < lam < np.inf):
            continue
        fi = fit["fit_info"]
        recs.append(dict(method=method, stage="candidate", lam=float(lam), selected=bool(lam == lam_sel), retained=regular.get(lam), converged=bool(fi.get("converged", True)),
                         max_abs_coord=fi.get("max_abs_coord"), **_flags(fi)))
    recs.append(dict(method=method, stage="selection", lam=lam_sel, max_abs_sH=float(np.max(np.abs(sel["s_H"]))), n_dropped=len(sel["dropped"])))
    return recs


def audit_cell(spec):
    out = []
    try:
        if spec["source"] == "clean_fit":
            from experiments.real_data import robustness as rb

            cfg = rb.load_config()
            panel = rb.get_panel(spec["dataset"], cfg)
            N, K = panel["N"], panel["K"]
            n_ijk, y_ijk, n_order, y_order = rb.llm_arrays(panel["llm"], N, K)
            st = fit_staged_structured_calibration(N, K, 1, n_ijk, y_ijk, rb.human_pairs(panel["human"]), n_order=n_order, y_order=y_order)
            fi = st["fit_info"]
            out.append(dict(method="llm_only", stage="stage1", lam=float("inf"), converged=bool(fi.get("converged", True)), max_abs_coord=fi.get("max_abs_coord"),
                            **_flags(fi, ["boundary_any", "boundary_b", "boundary_llm_other"]), boundary_cal=None))
            return dict(key=spec["key"], rows=out)
        if spec["source"] == "r1":
            from experiments.simulation.r1_main import simulate_cell

            c = simulate_cell(*spec["seed_args"])
            N, K, r, pairs = c["N"], c["K"], c["r"], c["pairs"]
            A = (c["n_ijk"], c["y_ijk"], c["n_order"], c["y_order"])
            lam_grid = None
        else:
            from experiments.real_data import robustness as rb

            job = spec["job"]
            c = rb.build_cell(job)
            N, K, pairs, A, lam_grid = c["N"], c["K"], c["pairs"], c["A"], c["lam_grid"]
            r = int(min(c["scfg"].get("llm_rank", 1), K - 1, N - 2))
        for method in spec["methods"]:
            try:
                if method == "dial_mu":
                    st = fit_consensus_only_calibrated(N, K, r, A[0], A[1], pairs, n_order=A[2], y_order=A[3])
                    sel = select_lambda(N, K, r, pairs, n_ijk_llm=A[0], y_ijk_llm=A[1], n_order=A[2], y_order=A[3], staged_fit=st, lambda_grid=lam_grid, return_all=True, align="mu")
                elif method == "dial_nodeb":
                    st = fit_consensus_only_calibrated(N, K, r, A[0], A[1], pairs)
                    sel = select_lambda(N, K, r, pairs, n_ijk_llm=A[0], y_ijk_llm=A[1], staged_fit=st, lambda_grid=lam_grid, return_all=True, align="mu")
                elif method == "dial_w":
                    st = fit_staged_structured_calibration(N, K, r, A[0], A[1], pairs, n_order=A[2], y_order=A[3])
                    sel = select_lambda(N, K, r, pairs, n_ijk_llm=A[0], y_ijk_llm=A[1], n_order=A[2], y_order=A[3], staged_fit=st, lambda_grid=lam_grid, return_all=True, align="W")
                else:
                    raise ValueError(method)
                out.extend(_audit_selection(method, st, sel, N, K))
            except Exception as e:  # noqa: BLE001
                out.append(dict(method=method, stage="error", error=repr(e) + traceback.format_exc()[-300:]))
    except Exception as e:  # noqa: BLE001
        out.append(dict(method=None, stage="error", error=repr(e) + traceback.format_exc()[-300:]))
    return dict(key=spec["key"], rows=out)


# --------------------------------------------------------------------------- run
def run(workers, limit=None, sources=None):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import time

    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / "fits.jsonl"
    done = set()
    if dest.exists():
        done = {tuple(json.loads(l)["key"]) for l in dest.read_text().splitlines() if l.strip()}
    jobs = [j for j in all_jobs() if tuple(j["key"]) not in done and (sources is None or j["source"] in sources)]
    if limit:
        jobs = jobs[:limit]
    print(f"{len(jobs)} cells to audit ({len(done)} already done)", flush=True)
    t0 = time.monotonic()
    with dest.open("a") as fh, ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(audit_cell, j) for j in jobs]
        for n, f in enumerate(as_completed(futures), 1):
            fh.write(json.dumps(f.result()) + "\n")
            fh.flush()
            if n % 100 == 0 or n == len(jobs):
                print(f"{n}/{len(jobs)} cells; {time.monotonic() - t0:.0f}s", flush=True)


# --------------------------------------------------------------------------- summarize
KEY_COLS = {"r1": ["source", "config", "row", "n_L", "n_H", "sigma_pos", "seed"],
            "robustness": ["source", "dataset", "sweep", "panel", "kind", "level", "n_H_level", "seed"],
            "intermediate": ["source", "dataset", "sweep", "panel", "kind", "level", "n_H_level", "seed"],
            "clean_fit": ["source", "dataset"]}


def load_fits():
    recs = []
    for line in (OUT / "fits.jsonl").read_text().splitlines():
        d = json.loads(line)
        key = dict(zip(KEY_COLS[d["key"][0]], d["key"]))
        recs.extend(dict(key, **r) for r in d["rows"])
    return pd.DataFrame(recs)


def figure_tags(df):
    """The reported figures each fit belongs to (a fit can support several)."""
    tags = []
    for r in df.itertuples():
        t = []
        if r.source == "clean_fit":
            t = ["G5"]
        elif r.source == "r1":
            t = [{"main10": "Fig2", "app20": "F1", "main10_mis": "F2", "main10_pos": "F3"}[r.config]]
        elif r.source == "intermediate":
            t = ["Fig4"]
        else:
            d, sw, p, k, lv, nh, m = r.dataset, r.sweep, r.panel, r.kind, r.level, r.n_H_level, r.method
            if sw == "order" and p == "biased5":
                if lv in (1.0, 0.05): t.append("Fig3")
                if d == "arena_33k" and m != "dial_w": t.append("G4")
            if sw == "noise" and p == "biased5" and k == "position":
                if lv == 8: t.append("Fig3")
                if d == "arena_33k" and m != "dial_w": t.append("G4")
            if sw == "noise_scarce": t.append("G4")
            if sw == "budget":
                if m == "dial_mu": t.append("G7")
                if p == "all": t.append("G6")
                if p == "all" and d == "arena_33k" and m == "dial_mu": t.append("Fig4")
            if sw == "llm_budget":
                if p == "biased5":
                    t.append("G6")
                    if d == "arena_33k": t.append("G4")
                else:
                    t.append("G8")
        tags.append(t)
    return tags


def _rate(s):
    s = s.dropna().astype(bool)
    return float(100 * s.mean()) if len(s) else float("nan")


def block_summary(g):
    sel = g[(g.stage == "candidate") & (g.selected == True)]  # noqa: E712
    sel_interior = sel[~sel.boundary_any.astype(bool)]
    return pd.Series({
        "fits": len(g), "pct_boundary_any": _rate(g.boundary_any), "pct_boundary_b": _rate(g.boundary_b),
        "pct_boundary_llm_other": _rate(g.boundary_llm_other), "pct_boundary_cal": _rate(g.boundary_cal),
        "selected_finite": len(sel), "pct_selected_boundary_any": _rate(sel.boundary_any),
        "pct_selected_boundary_nonb": _rate(sel.boundary_cal.fillna(False).astype(bool) | sel.boundary_llm_other.fillna(False).astype(bool)) if len(sel) else float("nan"),
        "max_abs_coord_selected_interior": float(sel_interior.max_abs_coord.max()) if len(sel_interior) else float("nan"),
    })


def check_reproduction(df):
    """Selected lambda and max |s_H| of the audit against the stored study rows."""
    from experiments.real_data.robustness_plot import load as load_real

    sel = df[df.stage == "selection"]
    out = []
    r1 = sel[sel.source == "r1"]
    if len(r1):
        for cfg, g in r1.groupby("config"):
            from experiments.simulation.r1_plot import load as load_r1

            rows = load_r1(cfg, base=ROOT)
            m = g.merge(rows[["row", "n_L", "n_H", "sigma_pos", "seed", "method", "lam", "max_abs"]], on=["row", "n_L", "n_H", "sigma_pos", "seed", "method"], how="left", suffixes=("", "_stored"))
            out.append(m)
    rob = sel[sel.source == "robustness"]
    for d, g in rob.groupby("dataset"):
        rows = load_real(d, ROOT)
        rows = rows.drop_duplicates(["sweep", "panel", "kind", "level", "n_H_level", "seed", "method"], keep="last")
        m = g.merge(rows[["sweep", "panel", "kind", "level", "n_H_level", "seed", "method", "lam", "max_abs"]], on=["sweep", "panel", "kind", "level", "n_H_level", "seed", "method"], how="left", suffixes=("", "_stored"))
        out.append(m)
    inter = sel[sel.source == "intermediate"]
    if len(inter):
        stored = []
        for line in (ROOT / "results" / "intermediate_budget" / "jobs.jsonl").read_text().splitlines():
            stored.extend(json.loads(line)["rows"])
        rows = pd.DataFrame(stored)
        m = inter.merge(rows[["dataset", "sweep", "panel", "kind", "level", "n_H_level", "seed", "method", "lam", "max_abs"]], on=["dataset", "sweep", "panel", "kind", "level", "n_H_level", "seed", "method"], how="left", suffixes=("", "_stored"))
        out.append(m)
    m = pd.concat(out, ignore_index=True)
    has = m.lam_stored.notna()
    same_lam = np.isclose(m.lam.astype(float), m.lam_stored.astype(float), rtol=1e-6) | (np.isinf(m.lam.astype(float)) & np.isinf(m.lam_stored.astype(float)))
    same_s = np.isclose(m.max_abs_sH, m.max_abs.astype(float), rtol=1e-4, atol=1e-6)
    m["stored_found"], m["lam_match"], m["score_match"] = has, has & same_lam, has & same_s
    return m


def summarize():
    df = load_fits()
    errors = df[df.stage == "error"]
    fits = df[df.stage.isin(["stage1", "candidate", "endpoint_inf"])].copy()
    fits["figures"] = figure_tags(fits)
    fits["cell"] = np.select([fits.source == "r1", fits.source == "clean_fit"], [fits.config, fits.dataset],
                             fits.dataset.astype(str) + "/" + fits.sweep.astype(str) + "/" + fits.panel.astype(str) + "/" + fits.kind.astype(str))
    fits["category"] = np.select([fits.stage == "stage1", fits.stage == "endpoint_inf", fits.selected == True], ["(i) stage-one LLM-only", "(iv) lambda=inf calibration", "(iii) selected finite lambda"],  # noqa: E712
                                 "(ii) finite-lambda candidate")
    # (ii) is all finite-lambda candidates, selected or not
    cand = fits[fits.stage == "candidate"].assign(category="(ii) all finite-lambda candidates")
    sel = fits[(fits.stage == "candidate") & (fits.selected == True)].assign(category="(iii) GACV-selected finite lambda")  # noqa: E712
    cats = pd.concat([fits[fits.stage == "stage1"].assign(category="(i) stage-one LLM-only"), cand, sel,
                      fits[fits.stage == "endpoint_inf"].assign(category="(iv) lambda=inf calibration")], ignore_index=True)
    by_cell = cats.groupby(["source", "cell", "method", "category"]).apply(block_summary).reset_index()
    by_fig = cats.explode("figures").groupby(["figures", "method", "category"]).apply(block_summary).reset_index()
    overall = cats.groupby(["category"]).apply(block_summary).reset_index()
    by_cell.to_csv(OUT / "summary_by_configuration.csv", index=False)
    by_fig.to_csv(OUT / "summary_by_figure.csv", index=False)
    overall.to_csv(OUT / "summary_overall.csv", index=False)
    cats.to_csv(OUT / "fits_flat.csv.gz", index=False)
    rep = check_reproduction(df)
    rep.to_csv(OUT / "reproduction_check.csv", index=False)

    s = cats[cats.category.str.startswith("(iii)")]
    key = dict(
        n_cells=int(df.drop_duplicates([c for c in df.columns if c in {"source", "config", "row", "n_L", "n_H", "sigma_pos", "seed", "dataset", "sweep", "panel", "kind", "level", "n_H_level"}]).shape[0]),
        n_errors=int(len(errors)),
        selected_finite_fits=int(len(s)),
        selected_finite_boundary_rate_pct=_rate(s.boundary_any),
        selected_nonb_boundary_rate_pct=_rate(s.boundary_cal.fillna(False).astype(bool) | s.boundary_llm_other.fillna(False).astype(bool)) if len(s) else None,
        selected_retained_boundary_rate_pct=_rate(s[s.retained == True].boundary_any),  # noqa: E712
        reproduction=dict(checked=int(rep.stored_found.sum()), missing=int((~rep.stored_found).sum()), lam_match=int(rep.lam_match.sum()), score_match=int(rep.score_match.sum())),
    )
    (OUT / "key_check.json").write_text(json.dumps(key, indent=2) + "\n")
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
    print(overall.round(2).to_string(index=False)); print()
    print(by_fig.round(2).to_string(index=False)); print()
    print(json.dumps(key, indent=2))
    if len(errors):
        print(errors[["source", "method", "error"]].head(10).to_string())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["run", "summarize", "count"])
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--sources", default=None, help="comma-separated subset of clean_fit,r1,robustness,intermediate")
    a = p.parse_args()
    if a.action == "count":
        jobs = all_jobs()
        print(pd.Series([j["source"] for j in jobs]).value_counts().to_string())
        print(pd.Series([m for j in jobs for m in j["methods"]]).value_counts().to_string())
    elif a.action == "run":
        run(a.workers, a.limit, a.sources.split(",") if a.sources else None)
    else:
        summarize()


if __name__ == "__main__":
    main()
