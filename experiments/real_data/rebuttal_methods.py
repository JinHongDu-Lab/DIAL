"""Rebuttal add-on methods for the real-data cells (ICLR 2027 reviews, groups 1 and 2).

Group 1 (PPI-type and shrinkage estimators of the human score):
  ppi_bt_best   Bradley--Terry PPI++ with the proxy of the single judge agreeing most with the
                human labels of the cell's own calibration sample
  ppi_bt_panel  the same with the panel majority verdict as proxy
  ridge_mu      centered human BTL shrunk toward span(mu_hat) of the LLM-only consensus, ALO kappa

Group 2 (regularized human baselines for the small-budget regime):
  human_box     unpenalized human BTL on DIAL's coordinate box
  human_ridge   ridge toward zero, ALO kappa
  human_firth   Firth-penalized human BTL
  atc_ridge, atc_firth  AtC with the stage-1 ordering from the corresponding penalized fit

PPI++ is run exactly as defined. The protocol of the stored experiments draws the human and LLM
samples separately (unmatched verdicts), so a calibration record whose proxy is absent from the
LLM sample is not a labeled record for PPI++; ``ppi_status`` records ``ok`` (every calibration
record matched), ``partial`` (some), or ``failed`` (none, no proxy-only records, or a non-finite
fit). Nothing is imputed.
"""
from __future__ import annotations

import numpy as np

from dial_judge.baselines import _pava
from dial_judge.hja import fit_centered_btl_firth
from dial_judge.penalized import fit_boxed_btl, fit_ridge_btl
from dial_judge.ppi import fit_bt_ppi

PPI_METHODS = ("ppi_bt_best", "ppi_bt_panel")
GROUP1_METHODS = PPI_METHODS + ("ridge_mu",)
GROUP2_METHODS = ("human_box", "human_ridge", "human_firth", "atc_ridge", "atc_firth")
REBUTTAL_METHODS = GROUP1_METHODS + GROUP2_METHODS


# --------------------------------------------------------------------------- proxies
def judge_verdicts(llm):
    """Mean verdict of each (record, judge) over its retained rows (both display orders when
    present): a frame with columns record, k, i, j, f."""
    g = llm.groupby(["record", "k"], sort=False).agg(i=("i", "first"), j=("j", "first"), f=("y", "mean")).reset_index()
    return g


def panel_proxy(verdicts):
    """Majority verdict per record over the judges that answered it (0, 1, or 0.5 on a split)."""
    g = verdicts.groupby("record", sort=False).agg(i=("i", "first"), j=("j", "first"), f=("f", "mean"), n_judges=("k", "size")).reset_index()
    g["f"] = np.where(g["f"] > 0.5, 1.0, np.where(g["f"] < 0.5, 0.0, 0.5))
    return g


def best_judge(verdicts, cal):
    """Judge whose verdicts agree most with the human labels of the calibration sample; ties go
    to the lowest index. Uses only the cell's own calibration labels, so no test or extra human
    information enters the choice."""
    merged = verdicts.merge(cal[["record", "y"]], on="record", how="inner")
    if merged.empty:
        return int(verdicts["k"].min()), float("nan")
    agree = (1.0 - (merged["f"] - merged["y"]).abs()).groupby(merged["k"]).mean()
    k = int(agree.idxmax())
    return k, float(agree.loc[k])


def ppi_inputs(llm, cal, proxy):
    """(labeled, unlabeled, info) for PPI++ from the cell's LLM sample and calibration labels.

    ``proxy`` is ``"panel"`` or ``"best"``. Labeled rows are calibration records with a proxy;
    unlabeled rows are the other records of the LLM sample with a proxy.
    """
    verdicts = judge_verdicts(llm)
    info = {}
    if proxy == "panel":
        v = panel_proxy(verdicts)
    elif proxy == "best":
        k, agree = best_judge(verdicts, cal)
        info.update(proxy_judge=k, proxy_agreement=agree)
        v = verdicts[verdicts["k"] == k]
    else:
        raise ValueError(f"unknown proxy {proxy!r}")
    cal_rec = set(cal["record"].tolist())
    lab = v[v["record"].isin(cal_rec)].merge(cal[["record", "y"]], on="record", how="inner")
    unl = v[~v["record"].isin(cal_rec)]
    labeled = lab[["i", "j", "y", "f"]].to_numpy(dtype=float)
    unlabeled = unl[["i", "j", "f"]].to_numpy(dtype=float)
    n_cal = int(len(cal))
    info.update(n_cal=n_cal, n_labeled_matched=int(len(lab)), n_proxy_only=int(len(unl)))
    info["ppi_status"] = "ok" if (len(lab) == n_cal and len(lab) > 0 and len(unl) > 0) else ("failed" if (len(lab) == 0 or len(unl) == 0) else "partial")
    return labeled, unlabeled, info


# --------------------------------------------------------------------------- fits
def _atc_from_order(mu, s_stage1):
    order = np.argsort(np.asarray(s_stage1, dtype=float), kind="stable")
    mu = np.asarray(mu, dtype=float)
    fitted = np.empty(mu.size)
    fitted[order] = _pava(mu[order])
    return fitted - fitted.mean()


def run_rebuttal_methods(cell, want, rec, fail, st_mu):
    """Fit the requested add-on methods of one real-data cell.

    ``cell`` is the dict of ``robustness.build_cell`` (needs N, pairs, llm, cal); ``rec`` and
    ``fail`` are the cell's row writers; ``st_mu`` is the staged consensus fit (None when it
    failed), whose ``mu`` is the shrinkage target of ridge_mu and the score AtC projects.
    """
    N, pairs, llm, cal = cell["N"], cell["pairs"], cell["llm"], cell["cal"]
    mu = None if st_mu is None else np.asarray(st_mu["mu"], dtype=float)

    for method, proxy in (("ppi_bt_best", "best"), ("ppi_bt_panel", "panel")):
        if not want(method):
            continue
        try:
            labeled, unlabeled, info = ppi_inputs(llm, cal, proxy)
            if info["ppi_status"] == "failed":
                fail(method, RuntimeError(f"ppi inputs: {info}"))
                continue
            fit = fit_bt_ppi(N, labeled, unlabeled)
            fi = fit["fit_info"]
            status = info["ppi_status"] if fi["finite"] else "failed"
            if not fi["finite"]:
                fail(method, RuntimeError(f"non-finite PPI++ fit (max |s| = {fi['max_abs']:.1f}); {info}"))
                continue
            rec(method, fit["s_H"], lam=fit["lam"], ppi_lam_raw=fi.get("lam_raw", np.nan), ppi_lam_at=fi.get("lam_at"), ppi_status=status,
                converged=fi["converged"], **{k: v for k, v in info.items() if k != "ppi_status"})
        except Exception as e:  # noqa: BLE001
            fail(method, e)

    if want("ridge_mu") and mu is not None:
        try:
            fit = fit_ridge_btl(N, pairs, mu=mu)
            rec("ridge_mu", fit["s_H"], kappa=fit["kappa"], kappa_rel=fit["kappa_rel"], selected_regular=fit["fit_info"]["selected_regular"])
        except Exception as e:  # noqa: BLE001
            fail("ridge_mu", e)
    elif want("ridge_mu"):
        fail("ridge_mu", RuntimeError("no staged consensus fit"))

    stage1 = {}
    if want("human_box"):
        try:
            fit = fit_boxed_btl(N, pairs)
            rec("human_box", fit["s_H"], box_active=fit["fit_info"]["box_active"], converged=fit["fit_info"]["converged"])
        except Exception as e:  # noqa: BLE001
            fail("human_box", e)
    if want("human_ridge", "atc_ridge"):
        try:
            fit = fit_ridge_btl(N, pairs, mu=None)
            stage1["ridge"] = fit["s_H"]
            if want("human_ridge"):
                rec("human_ridge", fit["s_H"], kappa=fit["kappa"], kappa_rel=fit["kappa_rel"], selected_regular=fit["fit_info"]["selected_regular"])
        except Exception as e:  # noqa: BLE001
            fail("human_ridge", e)
    if want("human_firth", "atc_firth"):
        try:
            s = fit_centered_btl_firth(N, pairs)
            stage1["firth"] = s
            if want("human_firth"):
                rec("human_firth", s)
        except Exception as e:  # noqa: BLE001
            fail("human_firth", e)
    for method, key in (("atc_ridge", "ridge"), ("atc_firth", "firth")):
        if not want(method):
            continue
        if mu is None or key not in stage1:
            fail(method, RuntimeError("missing consensus or stage-1 fit"))
            continue
        try:
            rec(method, _atc_from_order(mu, stage1[key]))
        except Exception as e:  # noqa: BLE001
            fail(method, e)
