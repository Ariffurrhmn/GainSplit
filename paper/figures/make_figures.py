"""Paper figures, built from the analysis outputs in results/analysis/.

Run from the repo root:  python paper/figures/make_figures.py
Writes vector PDFs (used by the paper) and PNG previews next to this file.

Every number comes from a CSV written by experiments/03_analysis.ipynb; nothing is
re-estimated here except the Holm adjustment, which follows the vault notes
(38 research contrasts for the main table, 16 for the label-quality table).

Colour rule (checked for colour-blind separation): blue, orange and aqua carry
identity; grey is used only for reference marks (baselines, chance). Every colour
is backed by a marker shape or a direct label, so the figures survive grey print.
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
AN = ROOT / "results" / "analysis"
OUT = Path(__file__).resolve().parent

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
COL_W, FULL_W = 3.5, 7.16  # IEEE column and page width, inches

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
    "legend.frameon": False,
})

NAMES = {"A0": "TF-IDF + LR", "A1": "BERT", "A2": "BERT + helpers (L)", "A3": "BERT + helpers (L+U)",
         "A4": "GAN-BERT", "A5": "GAN-BERT + helpers", "A6": "Self-training", "A7": "FGM",
         "A8": "A3, shuffled helpers", "A9": "A2, shuffled helpers"}


def holm(p):
    """Holm-adjusted p-values (p = 0 from the bootstrap is treated as 1/10,001)."""
    p = np.maximum(np.asarray(p, float), 1 / 10001)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def zero_line(ax, x=0.0):
    ax.axvline(x, color=INK2, lw=0.6, zorder=0)


def light_grid(ax, axis="x"):
    ax.grid(axis=axis, color=GRID, lw=0.4, zorder=0)
    ax.set_axisbelow(True)


def ci_point(ax, x, y, lo, hi, color, marker, filled, size=4.2, label=None):
    ax.plot([lo, hi], [y, y], color=color, lw=1.1, solid_capstyle="round", zorder=2)
    ax.plot(x, y, marker=marker, ms=size, mec=color, mew=1.0,
            mfc=color if filled else "white", ls="none", zorder=3, label=label)


# ---------------------------------------------------------------- data
con = pd.read_csv(AN / "contrasts_test_macro_f1.csv")
research = con[con["contrast"] != "A1-A0"].copy()
research["holm"] = holm(research["p"])
con = con.merge(research[["contrast", "budget", "holm"]], on=["contrast", "budget"], how="left")


def c(name, budget):
    return con[(con["contrast"] == name) & (con["budget"] == budget)].iloc[0]


# ---------------------------------------------------------------- Fig. 2: RQ1
def fig_rq1():
    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL_W, 2.05), gridspec_kw={"width_ratios": [1, 1.15], "wspace": 0.55})

    # (a) the 2x2 arms against BERT across budgets
    budgets = [1, 5, 10, 30]
    xs = np.arange(len(budgets))
    series = [("A3-A1", "Helpers (L+U)", BLUE, "o", -0.14),
              ("A4-A1", "GAN", ORANGE, "s", 0.0),
              ("A5-A1", "GAN + helpers", AQUA, "^", 0.14)]
    for name, lab, col, mk, dx in series:
        rows = [c(name, bud) for bud in budgets]
        y = [r["mean_diff"] for r in rows]
        a.plot(xs + dx, y, color=col, lw=1.0, zorder=2)
        for x, r in zip(xs + dx, rows):
            a.plot([x, x], [r["ci_low"], r["ci_high"]], color=col, lw=1.0, zorder=2)
            a.plot(x, r["mean_diff"], marker=mk, ms=4.2, mec=col, mew=1.0,
                   mfc=col if r["holm"] < 0.05 else "white", zorder=3)
        a.annotate(lab, (xs[-1] + dx, y[-1]), xytext=(6, {"A3-A1": 6, "A4-A1": -6, "A5-A1": 0}[name]),
                   textcoords="offset points", va="center", fontsize=7, color=INK)
    a.axhline(0, color=INK2, lw=0.6, zorder=0)
    a.set_xticks(xs, [f"{b}%" for b in budgets])
    a.set_xlim(-0.4, len(budgets) - 0.4 + 0.9)
    a.set_xlabel("Labelled share of the training set")
    a.set_ylabel(r"$\Delta$ macro-F1 vs. BERT (points)")
    a.set_title("(a) GAN and helper heads across budgets", loc="left")
    light_grid(a, "y")

    # (b) every arm against BERT at 1% and 10%
    arms = ["A7", "A6", "A2", "A3", "A4", "A5"]
    labels = ["FGM (supervised adv.)", "Self-training", "Helpers on L only", "Helpers on L+U",
              "GAN-BERT", "GAN-BERT + helpers"]
    for i, arm in enumerate(arms):
        y = len(arms) - 1 - i
        for bud, col, mk, dy in ((1, BLUE, "o", 0.17), (10, ORANGE, "s", -0.17)):
            r = c(f"{arm}-A1", bud)
            ci_point(b, r["mean_diff"], y + dy, r["ci_low"], r["ci_high"], col, mk, r["holm"] < 0.05)
    b.set_yticks(range(len(arms)), labels[::-1])
    b.tick_params(axis="y", length=0)
    zero_line(b)
    light_grid(b, "x")
    b.set_xlabel(r"$\Delta$ macro-F1 vs. BERT (points, 95% CI)")
    b.set_title("(b) All arms at 1% and 10% labels", loc="left")
    handles = [Line2D([], [], marker="o", color=BLUE, mfc=BLUE, ms=4, lw=1, label="1% labels"),
               Line2D([], [], marker="s", color=ORANGE, mfc=ORANGE, ms=4, lw=1, label="10% labels"),
               Line2D([], [], marker="o", color=INK2, mfc="white", ms=4, lw=0, label="not Holm-robust")]
    b.legend(handles=handles, loc="upper left", handletextpad=0.4, borderaxespad=0.1)
    save(fig, "fig_rq1")


# ---------------------------------------------------------------- Fig. 3: RQ2
def fig_rq2():
    lq = pd.read_csv(AN / "label_quality_contrasts.csv")
    lq["holm"] = holm(lq["p"])
    # A8 vs A3 at 10% comes from the main table (sign flipped: shuffled minus real)
    a8 = c("A3-A8", 10)

    rows = [("soft", "Soft probabilities"), ("flip25", "25% labels flipped"), ("flip50", "50% labels flipped"),
            ("vader", r"VADER lexicon ($\kappa$=0.26)"), ("teacher2", r"Second teacher ($\kappa$=0.14)"),
            ("sentonly", "Sentiment head only"), ("emoonly", "Emotion head only"), ("shuf", "Shuffled labels")]
    fig, ax = plt.subplots(figsize=(COL_W, 2.45))
    for i, (tag, lab) in enumerate(rows):
        y = len(rows) - 1 - i
        for bud, base, col, mk, dy in ((1, "A2", BLUE, "o", 0.17), (10, "A3", ORANGE, "s", -0.17)):
            if tag == "shuf" and bud == 10:
                d, lo, hi, h = -a8["mean_diff"], -a8["ci_high"], -a8["ci_low"], a8["holm"]
            else:
                name = f"A9[shuf]-A2" if tag == "shuf" else f"{base}[{tag}]-{base}"
                r = lq[(lq["contrast"] == name) & (lq["budget"] == bud)].iloc[0]
                d, lo, hi, h = r["mean_diff"], r["ci_low"], r["ci_high"], r["holm"]
            ci_point(ax, d, y + dy, lo, hi, col, mk, h < 0.05)
    ax.set_yticks(range(len(rows)), [r[1] for r in rows][::-1])
    ax.tick_params(axis="y", length=0)
    zero_line(ax)
    light_grid(ax, "x")
    ax.set_xlabel(r"$\Delta$ macro-F1 vs. teacher labels (points, 95% CI)")
    handles = [Line2D([], [], marker="o", color=BLUE, mfc=BLUE, ms=4, lw=1, label="1%: vs. helpers on L"),
               Line2D([], [], marker="s", color=ORANGE, mfc=ORANGE, ms=4, lw=1, label="10%: vs. helpers on L+U"),
               Line2D([], [], marker="o", color=INK2, mfc="white", ms=4, lw=0, label="not Holm-robust")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.42, 1.2), ncol=2,
              handletextpad=0.4, columnspacing=0.9)
    save(fig, "fig_rq2")


# ---------------------------------------------------------------- Fig. 4: RQ3
def fig_rq3():
    mk = pd.read_csv(AN / "masked_test_macro_f1.csv")
    tr = pd.read_csv(AN / "transfer_isarc_auc.csv")
    tr["tag"] = tr["tag"].fillna("")
    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL_W, 2.15), gridspec_kw={"width_ratios": [1.25, 1], "wspace": 0.75})

    # (a) outlet-cue masking vs random masking, 10% budget
    arms = ["A1", "A3", "A4", "A6", "A7"]
    for i, arm in enumerate(arms):
        r = mk[(mk["arm"] == arm) & (mk["budget"] == 10) & (mk["tag"] == "mask")].iloc[0]
        y = len(arms) - 1 - i
        a.plot([r["cuemask"], r["test"]], [y, y], color=GRID, lw=2.2, solid_capstyle="round", zorder=1)
        a.plot(r["test"], y, "o", ms=4.6, color=INK2, zorder=3)
        a.plot(r["randmask"], y, "D", ms=4.2, color=BLUE, mec="white", mew=0.5, zorder=3)
        a.plot(r["cuemask"], y, "s", ms=4.4, color=ORANGE, mec="white", mew=0.5, zorder=3)
        a.text(90.35, y, f"{r['shortcut_gap']:.1f} [{r['gap_ci_low']:.1f}, {r['gap_ci_high']:.1f}]",
               va="center", fontsize=6.5, color=INK)
    a.text(90.35, len(arms) - 0.45, "gap [95% CI]", fontsize=6.5, color=INK2, va="bottom")
    a.set_yticks(range(len(arms)), [NAMES[x] if x != "A3" else "Helpers (L+U)" for x in arms][::-1])
    a.tick_params(axis="y", length=0)
    a.set_xlim(81.5, 93.4)
    a.spines["bottom"].set_bounds(82, 90)
    a.set_xticks([82, 84, 86, 88, 90])
    a.set_ylim(-0.6, len(arms) - 0.1)
    a.set_xlabel("Test macro-F1, 10% labels")
    a.set_title("(a) Masking outlet cues vs. random words", loc="left")
    light_grid(a, "x")
    handles = [Line2D([], [], marker="o", color=INK2, ls="none", ms=4, label="original"),
               Line2D([], [], marker="D", color=BLUE, ls="none", ms=4, label="random mask"),
               Line2D([], [], marker="s", color=ORANGE, ls="none", ms=4, label="cue mask")]
    a.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.42, -0.27), ncol=3, handletextpad=0.2)

    # (b) transfer to iSarcasmEval (ROC-AUC with bootstrap CI)
    sel = [("A0", 10, ""), ("A1", 10, ""), ("A2", 10, ""), ("A3", 10, ""), ("A4", 10, ""), ("A5", 10, ""),
           ("A6", 10, ""), ("A7", 10, ""), ("A3", 30, ""), ("A5", 30, ""), ("A1", 100, ""),
           ("A1", 100, "isarctrain")]
    labs = []
    for i, (arm, bud, tag) in enumerate(sel):
        r = tr[(tr["arm"] == arm) & (tr["budget"] == bud) & (tr["tag"] == tag)].iloc[0]
        y = len(sel) - 1 - i
        ref = tag == "isarctrain"
        col, m = (ORANGE, "D") if ref else (BLUE, "o")
        ci_point(b, r["auc"], y, r["ci_low"], r["ci_high"], col, m, r["ci_low"] > 50, size=3.8)
        short = {"A0": "TF-IDF + LR", "A1": "BERT", "A2": "Helpers (L)", "A3": "Helpers (L+U)", "A4": "GAN-BERT",
                 "A5": "GAN + helpers", "A6": "Self-training", "A7": "FGM"}[arm]
        labs.append("In-domain BERT" if ref else f"{short}, {bud}%")
    b.set_yticks(range(len(sel)), labs[::-1])
    b.tick_params(axis="y", length=0)
    zero_line(b, 50)
    b.text(50.3, len(sel) - 0.45, "chance", fontsize=6.5, color=INK2, va="bottom")
    light_grid(b, "x")
    b.set_ylim(-0.6, len(sel) - 0.1)
    b.set_xlabel("iSarcasmEval ROC-AUC (95% CI)")
    b.set_title("(b) Transfer to iSarcasmEval", loc="left")
    handles = [Line2D([], [], marker="o", color=BLUE, ms=4, lw=1, label="trained on Headlines"),
               Line2D([], [], marker="D", color=ORANGE, ms=4, lw=1, label="trained on iSarcasmEval"),
               Line2D([], [], marker="o", color=INK2, mfc="white", ms=4, lw=0, label="CI includes 50")]
    b.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.42, -0.27), ncol=3, handletextpad=0.2,
             columnspacing=0.8)
    save(fig, "fig_rq3")


if __name__ == "__main__":
    fig_rq1()
    fig_rq2()
    fig_rq3()
    print("written:", sorted(p.name for p in OUT.glob("fig_*.pdf")))
