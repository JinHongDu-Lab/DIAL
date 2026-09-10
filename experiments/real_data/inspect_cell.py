"""Rebuild the data of one robustness cell exactly as `robustness.run_cell` does, for diagnostics.

    python -m experiments.real_data.inspect_cell mt_bench noise_scarce biased5 anti 0 --seed 0

prints the staged LLM fit's convergence and separation diagnostics and DIAL's GACV path.
"""
from __future__ import annotations

import argparse

import numpy as np

from dial_judge.baselines import fit_consensus_only_calibrated, fit_human_only_btl
from dial_judge.evaluate import heldout_log_loss
from dial_judge.gacv import select_lambda

from . import robustness as rb


def build(dataset, sweep, panel_name, kind, level, n_H_req, seed, cfg=None):
    """Return the arrays of one cell: N, K, K_real, A, pairs, test_recs, floor, r_llm, lam_grid, n_L, n_H."""
    cfg = cfg or rb.load_config()
    scfg, dcfg, swcfg = cfg["study"], cfg[dataset], cfg["sweeps"][sweep]
    panel = rb.restrict_panel(rb.get_panel(dataset, cfg), None if panel_name == "all" else cfg["panels"][panel_name])
    code = rb.DATASET_CODE[dataset]
    rng_split = np.random.default_rng([int(seed), code, 1])
    rng_budget = np.random.default_rng([int(seed), code, 2])
    rng_design = np.random.default_rng([int(seed), code, 3])
    rho = float(swcfg.get("rho_swap", 1.0))
    m, n_L_req = 0, -1
    n_H = int(n_H_req) if n_H_req is not None else int(dcfg["n_H"])
    if sweep == "order":
        rho = float(level)
    elif sweep in ("noise", "noise_scarce"):
        m = int(level)
        if sweep == "noise_scarce":
            n_H = int(rb.per_dataset(swcfg.get("n_H_fixed", n_H), dataset))
    elif sweep == "budget":
        n_H = int(level)
    elif sweep == "llm_budget":
        n_L_req = int(level)
    test, train = rb.split_records(panel["records"], dcfg["f_test"], rng_split)
    llm = panel["llm"][panel["llm"]["record"].isin(train)].reset_index(drop=True)
    K_real = panel["K"]
    if sweep == "noise_scarce":
        llm = rb.subsample_rows(llm, int(rb.per_dataset(swcfg["n_L_base"], dataset)), rng_design)
    llm, K = rb.inject_noise_judges(llm, K_real, m, kind, rng_design, first_prob=scfg.get("position_noise_first_prob", 0.9))
    llm = rb.thin_display_order(llm, rho, rng_design)
    llm = rb.subsample_rows(llm, n_L_req, rng_design)
    llm, K, K_real, K_dropped = rb.drop_empty_judges(llm, K, K_real)
    hum_train = panel["human"][panel["human"]["record"].isin(train)].reset_index(drop=True)
    hum_test = panel["human"][panel["human"]["record"].isin(test) & (panel["human"]["y"] != 0.5)].reset_index(drop=True)
    cal = rb.draw_budget(hum_train, n_H, rng_budget)
    N = panel["N"]
    A = rb.llm_arrays(llm, N, K)
    pairs = rb.human_pairs(cal)
    test_recs = rb.human_records(hum_test)
    n_L, n_Ha = int(len(llm)), int(len(cal))
    mults = scfg.get("lambda_multipliers")
    lam_grid = [float(mm) * n_L / n_Ha for mm in mults] if mults else None
    ref = fit_human_only_btl(N, rb.human_pairs(hum_test))["s_H"]
    floor = heldout_log_loss(ref, test_recs)
    r_llm = int(min(scfg.get("llm_rank", 1), K - 1, N - 2))
    return dict(N=N, K=K, K_real=K_real, K_dropped=K_dropped, A=A, pairs=pairs, test_recs=test_recs, floor=floor, r_llm=r_llm, lam_grid=lam_grid, n_L=n_L, n_H=n_Ha)


def report(c):
    st = fit_consensus_only_calibrated(c["N"], c["K"], c["r_llm"], c["A"][0], c["A"][1], c["pairs"], n_order=c["A"][2], y_order=c["A"][3])
    fi = st["fit_info"]
    print(f"staged: converged={fi['converged']} b_at_bound={fi.get('b_at_bound')} inner_limit_hits={fi.get('inner_limit_hits')} polish_grad={fi.get('polish_grad_norm', float('nan')):.2e} "
          f"max|b|={np.max(np.abs(st['b'])):.2f} max|s|={np.max(np.abs(st['s_H'])):.2f} excess={heldout_log_loss(st['s_H'], c['test_recs']) - c['floor']:.4f}")
    sel = select_lambda(c["N"], c["K"], c["r_llm"], c["pairs"], n_ijk_llm=c["A"][0], y_ijk_llm=c["A"][1], n_order=c["A"][2], y_order=c["A"][3], staged_fit=st, lambda_grid=c["lam_grid"], return_all=True, align="mu")
    print(f"DIAL: selected lam={sel['lam']:.3g} (rel {sel['lam'] * c['n_H'] / c['n_L'] if np.isfinite(sel['lam']) else 'inf'}), dropped={sel['dropped']}, human_only_exists={sel['human_only_exists']}")
    cands = dict(sel["candidates"])
    for p in sel["gacv_path"]:
        f = cands[p["lam"]]
        fi = f.get("fit_info", {}) or {}
        ex = heldout_log_loss(f["s_H"], c["test_recs"]) - c["floor"]
        rel = (p["lam"] * c["n_H"] / c["n_L"]) if np.isfinite(p["lam"]) else float("inf")
        print(f"   rel={rel:>7.3g} gacv={p['gacv']:.4f} regular={p['regular']!s:5} conv={fi.get('converged', '-')!s:5} b_bound={fi.get('b_at_bound', '-')!s:2} "
              f"max|s|={np.max(np.abs(f['s_H'])):5.2f} excess={ex:.4f}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset"); p.add_argument("sweep"); p.add_argument("panel"); p.add_argument("kind"); p.add_argument("level", type=float)
    p.add_argument("--nH", type=int, default=None); p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    c = build(a.dataset, a.sweep, a.panel, a.kind, a.level, a.nH, a.seed)
    print(f"{a.dataset} {a.sweep} {a.panel} {a.kind} {a.level:g} seed {a.seed}: N={c['N']} K={c['K']} (dropped {c['K_dropped']}) n_L={c['n_L']} n_H={c['n_H']}")
    report(c)


if __name__ == "__main__":
    main()
