"""Reanalysis of the JA-Ranking judge files used by HJA (plan Section 16 / figure A1).

Those files list `model_a` as the alphabetically first model in every record, so the
canonical item is displayed first throughout (a one-sided design), and the panel
contains a near-pure position judge. We score each consensus estimator against the
human BTL ranking fitted on our own human labels for the same models.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from dial_judge.baselines import fit_human_only_btl, fit_pooled_btl, fit_staged_structured_calibration
from dial_judge.evaluate import score_accuracy

from .robustness import ROOT, RESULTS_ROOT, get_panel, human_pairs, human_records, llm_arrays, load_config

JA_FILES = {"arena_33k": ROOT.parent / "HJA-Ranking-main" / "data" / "judge_results_10k_chatbot_arena.json",
            "mt_bench": ROOT.parent / "HJA-Ranking-main" / "data" / "judge_results_10k_mtbench.json"}
EXTREME = "zai-org/GLM-4.5-Air-FP8"


def ja_frame(path, item_index):
    ja = pd.DataFrame(json.load(open(path)))
    ja = ja[ja.judge_preferred_model.isin(["a", "b"])]
    i = ja.model_a.map(item_index)
    j = ja.model_b.map(item_index)
    ok = i.notna() & j.notna()
    ja, i, j = ja[ok], i[ok].astype(int).to_numpy(), j[ok].astype(int).to_numpy()
    judges = sorted(ja.judge_model.unique())
    jidx = {jn: k for k, jn in enumerate(judges)}
    pref_a = (ja.judge_preferred_model == "a").to_numpy()
    llm = pd.DataFrame(dict(record=ja.question_id.to_numpy(), k=ja.judge_model.map(jidx).to_numpy(), i=np.minimum(i, j), j=np.maximum(i, j),
                            y=np.where(i < j, pref_a, ~pref_a).astype(float), a=np.where(i < j, 1, -1), chose_first=pref_a))
    return llm, judges


def consensus_estimators(llm, N, K):
    n_ijk, y_ijk, n_order, y_order = llm_arrays(llm, N, K)
    d = [(0, 1, 2.0, 1.0)]
    st1o = fit_staged_structured_calibration(N, K, 1, n_ijk, y_ijk, d, n_order=n_order, y_order=y_order)
    return {
        "pooled BTL, no order term": fit_pooled_btl(N, n_ijk, y_ijk)["score"],
        "rank-1 consensus, no order term (HJA)": fit_staged_structured_calibration(N, K, 1, n_ijk, y_ijk, d)["mu"],
        "rank-0 consensus + order term": fit_staged_structured_calibration(N, K, 0, n_ijk, y_ijk, d, n_order=n_order, y_order=y_order)["mu"],
        "rank-1 consensus + order term": st1o["mu"],
    }, st1o


def run(cfg=None):
    cfg = cfg or load_config()
    rows, judge_rows = [], []
    for dataset, path in JA_FILES.items():
        panel = get_panel(dataset, cfg)
        N = panel["N"]
        item_index = {it: n for n, it in enumerate(panel["items"])}
        hum = panel["human"]
        s_h = fit_human_only_btl(N, human_pairs(hum))["s_H"]
        hrec = human_records(hum)
        llm, judges = ja_frame(path, item_index)
        for variant, keep in (("all judges", judges), ("without " + EXTREME.split("/")[-1], [j for j in judges if j != EXTREME])):
            kidx = [judges.index(j) for j in keep]
            sub = llm[llm.k.isin(kidx)].copy()
            sub["k"] = sub.k.map({k: n for n, k in enumerate(kidx)})
            fits, st = consensus_estimators(sub, N, len(kidx))
            for method, s in fits.items():
                rows.append(dict(dataset=dataset, variant=variant, method=method, K=len(kidx), n_L=int(len(sub)), first_share=float((sub.a == 1).mean()),
                                 tau=float(kendalltau(s_h, s).statistic), record_acc=score_accuracy(s, hrec)))
            if variant == "all judges":
                for k, j in enumerate(keep):
                    lk = sub[sub.k == k]
                    judge_rows.append(dict(dataset=dataset, judge=j, n=int(len(lk)), first_position_rate=float(lk.chose_first.mean()), b_hat=float(st["b"][k]), gamma_hat=float(st["gamma"][k])))
        # our own panel at the JA size for reference: canonical-first copy only, 10 judges, 10k rows
        rng = np.random.default_rng(0)
        ours = panel["llm"][panel["llm"].a == 1]
        ks = np.sort(rng.choice(panel["K"], min(10, panel["K"]), replace=False))
        o = ours[ours.k.isin(ks)]
        o = o.iloc[np.sort(rng.choice(len(o), min(10000, len(o)), replace=False))].copy()
        o["k"] = o.k.map({k: n for n, k in enumerate(ks)})
        fits, _ = consensus_estimators(o, N, len(ks))
        for method, s in fits.items():
            rows.append(dict(dataset=dataset, variant="our panel, 10 judges, 10k one-sided rows", method=method, K=len(ks), n_L=int(len(o)), first_share=1.0,
                             tau=float(kendalltau(s_h, s).statistic), record_acc=score_accuracy(s, hrec)))
    out = {"consensus": rows, "judges": judge_rows}
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    (RESULTS_ROOT / "ja_reanalysis.json").write_text(json.dumps(out, indent=1) + "\n")
    return pd.DataFrame(rows), pd.DataFrame(judge_rows)


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    tab, judges = run()
    print(tab.round(3).to_string(index=False))
    print(judges.round(3).to_string(index=False))
