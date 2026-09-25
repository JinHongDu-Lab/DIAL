r"""Shared method panel for every figure of the simulation and real-data studies.

Five presented methods, in this order, with one name, colour, marker, and line
style each; the appendix-only diagnostics follow. Figures that draw both
alignments use `PRESENTED_W`, which inserts DIAL-$W$ after DIAL-$\mu$. Both
notebooks import the panel and `RCPARAMS` from here, so the two studies share
their method order, colours, and fonts.
"""

import matplotlib as mpl
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

PRESENTED = ["human_only", "pooled_cal", "consensus_cal", "dial_mu", "dial_nodeb"]
# Order used by every figure that draws both alignments: DIAL-$W$ sits next to DIAL-$\mu$,
# before DIAL-noPos, so the simulation and real-data legends read alike.
PRESENTED_W = ["human_only", "pooled_cal", "consensus_cal", "dial_mu", "dial_w", "dial_nodeb"]
APPENDIX = ["staged_w", "dial_mle_mu", "dial_mle_w", "oracle", "dial_rsel"]

LABEL = {
    "human_only": "Human",
    "pooled_cal": "Pooled",
    "consensus_cal": "DIAL-Anc",
    "dial_mu": "DIAL-Ada",
    "dial_nodeb": "DIAL-noPos",
    "atc_btl": "AtC",
    "dial_w": "DIAL-Ada ($W$)",
    "staged_w": "DIAL-Anc-$W$ ($\\lambda=\\infty$, $W$)",
    "dial_mle_mu": "DIAL-Ada, fixed weight $n_{\\mathrm{L}}/n_{\\mathrm{H}}$",
    "dial_mle_w": "DIAL-Ada ($W$), fixed weight $n_{\\mathrm{L}}/n_{\\mathrm{H}}$",
    "oracle": "oracle weight",
    "dial_rsel": "DIAL-Ada, rank by GACV",
}

STYLE = {
    "human_only": dict(color="#52514e", ls="-", marker="o", lw=1.2),
    "pooled_cal": dict(color="#8e44ad", ls="-.", marker="X", lw=1.2),
    "consensus_cal": dict(color="#1baf7a", ls="-", marker="v", lw=1.2),
    "dial_mu": dict(color="#c0392b", ls="-", marker="o", lw=2.0),
    "dial_nodeb": dict(color="#eb6834", ls=":", marker="^", lw=1.2),
    # stage-matched external comparison for the human-calibration stage (main-text calibration figure only)
    "atc_btl": dict(color="#b8860b", ls="--", marker="D", lw=1.1),
    "dial_w": dict(color="#2a78d6", ls="-", marker="s", lw=1.2),
    "staged_w": dict(color="#7fb2e5", ls="--", marker="D", lw=1.0),
    "dial_mle_mu": dict(color="#e59c9c", ls="-.", marker="D", lw=0.9),
    "dial_mle_w": dict(color="#a9c8ea", ls="-.", marker="d", lw=0.9),
    "oracle": dict(color="#0b0b0b", ls=":", marker=None, lw=0.9),
    "dial_rsel": dict(color="#b07aa1", ls="-", marker="P", lw=0.9),
}

INK, INK_SOFT, GRID = "#0b0b0b", "#52514e", "#e4e3df"
# Direction-of-merit arrows for the metric axes, in the label's own ink: the glyph already says
# which way is better, and a green or red arrow would collide with DIAL-Anc and DIAL-Ada in the
# same figure. The arrow belongs to the label, so on a rotated y label it turns with the text and
# reads left or right on the page, like the rest of that label.
BETTER = {"up": "\u2191", "down": "\u2193"}
DATASET_LABEL = {"arena_33k": "Chatbot Arena", "mt_bench": "MT-Bench", "pandalm": "PandaLM"}
DATASET_COLOR = {"arena_33k": "#2a78d6", "mt_bench": "#eb6834", "pandalm": "#1baf7a"}

RCPARAMS = {
    "font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 9, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "axes.edgecolor": INK_SOFT, "axes.linewidth": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.dpi": 300, "figure.dpi": 130, "savefig.pad_inches": 0,
}


def mark_better(ax, direction, target="ylabel", sep="\u2009"):
    """Append a direction-of-merit arrow to `ax`'s y label or title: up = higher is better, down
    = lower is better.

    The arrow is part of the label, so it sits on the label's baseline and turns with it: on a
    rotated y label it is read in the label's own frame, like the text it follows. An empty label
    is skipped, which is what shared-y panels want. Call before the layout is computed so the
    glyph is measured with the label.
    """
    label = ax.yaxis.label if target == "ylabel" else ax.title
    text = label.get_text()
    if not text:
        return
    arrow = BETTER[direction]
    label.set_text(f"{text}{sep}{arrow}")


def plot_kwargs(method, with_marker=True):
    """Matplotlib keyword arguments for one method (marker optional)."""
    st = dict(STYLE[method])
    if not with_marker:
        st.pop("marker", None)
    return st


# Aliases for the oracle variants stored by the two drivers (population-risk oracle on DIAL's and
# DIAL-W's paths in the simulation; test-loss oracle on real data).
for _k, _lab in (("oracle_mu", "oracle weight (DIAL-Ada path)"), ("oracle_w", "oracle weight (DIAL-$W$ path)"), ("oracle_test", "test-oracle weight")):
    LABEL[_k] = _lab
    STYLE[_k] = dict(STYLE["oracle"])
STYLE["oracle_w"]["color"] = "#2a78d6"


# --------------------------------------------------------------------------- drawing helpers
# One seaborn call per panel replaces the per-method loops the notebooks used to repeat: the
# per-method colour, dash pattern, and marker of `STYLE` are handed to seaborn as hue/style
# mappings (line widths are set afterwards), so a panel is `method_lines(ax, frame, x, y, methods)`
# and a grouped-bar panel is `method_bars(ax, frame, x, y, methods)`.

_DASH_RC = {"--": "lines.dashed_pattern", "-.": "lines.dashdot_pattern", ":": "lines.dotted_pattern"}


def _dash(ls, lw):
    """Matplotlib's own dash pattern for `ls` at width `lw` (seaborn wants explicit tuples)."""
    if ls in ("-", "solid", None):
        return ""
    return tuple(lw * v for v in mpl.rcParams[_DASH_RC[ls]])


def palette(methods):
    return {m: STYLE[m]["color"] for m in methods}


def dashes(methods, lw=None):
    return {m: _dash(STYLE[m]["ls"], STYLE[m]["lw"] if lw is None else lw) for m in methods}


def markers(methods):
    # seaborn refuses to mix filled and unfilled markers, so the marker-less methods get a
    # filled placeholder and `method_lines` shrinks it away.
    return {m: STYLE[m].get("marker") or "o" for m in methods}


def widths(methods, lw=None):
    return {m: STYLE[m]["lw"] if lw is None else lw for m in methods}


def method_lines(ax, data, x, y, methods, se=None, band=False, caps=False, ms=3.2, lw=None,
                 se_exclude=(), emphasize=(), cap_kw=None, **kw):
    """Draw one line per method with the shared panel style (single `sns.lineplot` call).

    `se` names the standard-error column; `band` shades +/- 1.96 se, `caps` draws capped error
    bars instead, skipping the methods in `se_exclude`. `lw` overrides the per-method widths of
    `STYLE` with one width; `emphasize` lifts those methods above the rest. Methods absent from
    `data` are dropped so the others keep their colours.
    """
    methods = [m for m in methods if (data["method"] == m).any()]
    if not methods:
        return methods
    d = data[data["method"].isin(methods)].sort_values(x)
    keep = ax.get_xlabel(), ax.get_ylabel()   # seaborn names the axes after the data columns; the callers set their own
    sns.lineplot(data=d, x=x, y=y, hue="method", style="method",
                 hue_order=methods, style_order=methods,
                 palette=palette(methods), dashes=dashes(methods, lw), markers=markers(methods),
                 markersize=ms, estimator=None, errorbar=None, legend=False, ax=ax,
                 **{"markeredgewidth": 1.0, **kw})
    ax.set_xlabel(keep[0]); ax.set_ylabel(keep[1])
    # after the fact: per-method line widths (seaborn's `size` semantic would rescale the markers)
    # and matplotlib's marker edge (seaborn outlines every marker in white, which eats a small marker)
    for m, line in zip(methods, ax.lines[-len(methods):]):
        line.set_linewidth(widths([m], lw)[m]); line.set_markeredgecolor(line.get_color())
        line.set_zorder(3 if m in emphasize else 2)
        if STYLE[m].get("marker") is None:
            line.set_markersize(0)
    if se is not None and (band or caps):
        for m, g in d[~d["method"].isin(se_exclude)].groupby("method", sort=False):
            if band:
                ax.fill_between(g[x], g[y] - 1.96 * g[se], g[y] + 1.96 * g[se], color=STYLE[m]["color"], alpha=0.10, lw=0)
            else:
                ax.errorbar(g[x], g[y], yerr=1.96 * g[se], fmt="none", ecolor=STYLE[m]["color"], zorder=2,
                            **{"elinewidth": 0.5, "capsize": 1.0, "capthick": 0.5, **(cap_kw or {})})
    return methods


def method_bars(ax, data, x, y, methods, se=None, hatch=None, order=None, gap=0.1, width=0.8, ecolor=INK):
    """Grouped bars, one group per level of `x` and one bar per method (single `sns.barplot`).

    `se` names the standard-error column (1.96 se error bars) and `hatch` a boolean column that
    hatches a bar. Seaborn does the dodging; `gap` reopens the small spacing between bars.
    """
    order = list(order) if order is not None else list(dict.fromkeys(data[x]))
    methods = [m for m in methods if (data["method"] == m).any()]
    keep = ax.get_xlabel(), ax.get_ylabel()
    sns.barplot(data=data, x=x, y=y, hue="method", order=order, hue_order=methods,
                palette=palette(methods), width=width, errorbar=None, legend=False, zorder=2, ax=ax)
    ax.set_xlabel(keep[0]); ax.set_ylabel(keep[1])
    idx = data.set_index([x, "method"])
    bars = list(ax.containers)[-len(methods):]   # snapshot: the error bars below append containers too
    for m, cont in zip(methods, bars):
        for lab, bar in zip(order, cont):
            bar.set_width(bar.get_width() * (1 - gap)); bar.set_x(bar.get_x() + bar.get_width() * gap / 2)
            if (lab, m) not in idx.index:
                continue
            row = idx.loc[(lab, m)]
            if se is not None:
                ax.errorbar(bar.get_x() + bar.get_width() / 2, bar.get_height(), yerr=1.96 * float(row[se]),
                            fmt="none", lw=0.6, capsize=1.5, ecolor=ecolor, zorder=3)
            if hatch is not None and bool(row[hatch]):
                bar.set_hatch("////"); bar.set_edgecolor("white"); bar.set_linewidth(0)
    return methods


def legend_handles(methods, labels=None, extra=(), extra_first=False, ms=3.2, lw=None, as_patch=(), no_marker=()):
    """(handles, labels) for a figure legend in the shared method order; `as_patch` lists the
    methods drawn as bars, `no_marker` those drawn as a flat reference line, and `extra` holds
    (label, kwargs) pairs for non-method line entries such as the theory lines."""
    labels = LABEL if labels is None else labels
    h = [Patch(facecolor=STYLE[m]["color"], label=labels[m]) if m in as_patch else
         Line2D([], [], color=STYLE[m]["color"], ls=STYLE[m]["ls"], lw=STYLE[m]["lw"] if lw is None else lw,
                marker=None if m in no_marker else STYLE[m].get("marker"), ms=ms, label=labels[m]) for m in methods]
    ex = [Line2D([], [], label=lab, **kw) for lab, kw in extra]
    h = ex + h if extra_first else h + ex
    return h, [a.get_label() for a in h]
