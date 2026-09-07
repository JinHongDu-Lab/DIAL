"""Shared method panel for every figure of the simulation and real-data studies.

Five presented methods, in this order, with one name, colour, marker, and line
style each; the appendix-only diagnostics follow. Both notebooks import from
here so the studies stay consistent (plan: code/plan/2026-09-07-unified-method-panel-plan.md).
"""

PRESENTED = ["human_only", "pooled_cal", "consensus_cal", "dial_mu", "dial_nodeb"]
APPENDIX = ["dial_w", "staged_w", "dial_mle_mu", "dial_mle_w", "oracle", "dial_rsel"]

LABEL = {
    "human_only": "Human-only",
    "pooled_cal": "Pooled-LLM",
    "consensus_cal": "Consensus-cal",
    "dial_mu": "DIAL",
    "dial_nodeb": "DIAL-noDeb",
    "dial_w": "DIAL-W",
    "staged_w": "Consensus-cal-W ($\\lambda=\\infty$, $W$)",
    "dial_mle_mu": "DIAL, fixed weight $n_L/n_0$",
    "dial_mle_w": "DIAL-W, fixed weight $n_L/n_0$",
    "oracle": "oracle weight",
    "dial_rsel": "DIAL, rank by GACV",
}

STYLE = {
    "human_only": dict(color="#52514e", ls="-", marker="o", lw=1.2),
    "pooled_cal": dict(color="#8e44ad", ls="-.", marker="X", lw=1.2),
    "consensus_cal": dict(color="#1baf7a", ls="--", marker="v", lw=1.2),
    "dial_mu": dict(color="#c0392b", ls="-", marker="o", lw=2.0),
    "dial_nodeb": dict(color="#eb6834", ls=":", marker="^", lw=1.2),
    "dial_w": dict(color="#2a78d6", ls="-", marker="s", lw=1.2),
    "staged_w": dict(color="#7fb2e5", ls="--", marker="D", lw=1.0),
    "dial_mle_mu": dict(color="#e59c9c", ls="-.", marker="D", lw=0.9),
    "dial_mle_w": dict(color="#a9c8ea", ls="-.", marker="d", lw=0.9),
    "oracle": dict(color="#0b0b0b", ls=":", marker=None, lw=0.9),
    "dial_rsel": dict(color="#b07aa1", ls="-", marker="P", lw=0.9),
}

INK, INK_SOFT, GRID = "#0b0b0b", "#52514e", "#e4e3df"
DATASET_LABEL = {"arena_33k": "Chatbot Arena", "mt_bench": "MT-Bench", "pandalm": "PandaLM"}
DATASET_COLOR = {"arena_33k": "#2a78d6", "mt_bench": "#eb6834", "pandalm": "#1baf7a"}

RCPARAMS = {
    "font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 9, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "axes.edgecolor": INK_SOFT, "axes.linewidth": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.dpi": 300, "figure.dpi": 130,
}


def plot_kwargs(method, with_marker=True):
    """Matplotlib keyword arguments for one method (marker optional)."""
    st = dict(STYLE[method])
    if not with_marker:
        st.pop("marker", None)
    return st


# Aliases for the oracle variants stored by the two drivers (population-risk oracle on DIAL's and
# DIAL-W's paths in the simulation; test-loss oracle on real data).
for _k, _lab in (("oracle_mu", "oracle weight (DIAL path)"), ("oracle_w", "oracle weight (DIAL-W path)"), ("oracle_test", "test-oracle weight")):
    LABEL[_k] = _lab
    STYLE[_k] = dict(STYLE["oracle"])
STYLE["oracle_w"]["color"] = "#2a78d6"
