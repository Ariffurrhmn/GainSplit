"""Figures for the second draft (paper/v2), built only from committed result tables.

Run from the repo root:  python paper/v2/figures/make_figures.py
Writes vector PDFs (used by the paper) and PNG previews next to this file.

fig_methods: test macro-F1 gain over plain BERT (A1) at 1% and 10% labels for every route to the
             unlabelled headlines, paired bootstrap with 95% intervals (each from its own pass's table).
fig_curve:   test macro-F1 against the label budget for BERT (A1) and BERT after MLM task-adaptive
             pretraining (A20), mean +- s.d. over 5 seeds, plus the two extra pretraining seeds.
Nothing is re-estimated here.
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RES = ROOT / "results"
OUT = Path(__file__).resolve().parent

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
COL_W, FULL_W = 3.5, 7.16

mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "pdf.fonttype": 42, "savefig.dpi": 300,
})

# (label, family, csv, contrast name in that csv)
ROWS = [
    ("GAN-BERT", "unlabelled text, adversarial", "analysis/contrasts_test_macro_f1.csv", "A4-A1"),
    ("Diffusion fakes", "unlabelled text, adversarial", "pass8_diff/analysis/contrasts_f1.csv", "A24-A1"),
    ("Diffusion consistency", "unlabelled text, consistency", "pass8_diff/analysis/contrasts_f1.csv", "A25-A1"),
    ("LLM-written pairs", "generated labelled text", "analysis_pass4/contrasts.csv", "A10-A1"),
    ("Helper heads on $L$", "labelled rows only", "analysis/contrasts_test_macro_f1.csv", "A2-A1"),
    ("FGM", "labelled rows only", "analysis/contrasts_test_macro_f1.csv", "A7-A1"),
    ("Self-training", "unlabelled text, pseudo-labels", "analysis/contrasts_test_macro_f1.csv", "A6-A1"),
    ("Helpers + FGM + self-training", "stack", "pass9/analysis/contrasts_f1.csv",
     "A30-A1 (whole stack vs BERT)"),
    ("JEPA pretraining", "unlabelled text, self-supervised", "pass8_jepa/analysis/contrasts_f1.csv", "A22-A1"),
    ("MLM pretraining", "unlabelled text, self-supervised", "pass8_mlm/analysis/contrasts_f1.csv", "A20-A1"),
]


def _get(csv, name, budget):
    t = pd.read_csv(RES / csv)
    r = t[(t["contrast"] == name) & (t["budget"] == budget)]
    if len(r) != 1:
        raise ValueError(f"{csv}: {name} at {budget}% not found exactly once")
    return r.iloc[0]


def fig_methods():
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.35), sharey=True)
    xlim = (-6.5, 10.0)
    for ax, budget in zip(axes, (1, 10)):
        for i, (label, fam, csv, name) in enumerate(ROWS):
            r = _get(csv, name, budget)
            y = len(ROWS) - 1 - i
            colour = AQUA if "self-supervised" in fam else (ORANGE if r.mean_diff < 0 else BLUE)
            if r.mean_diff < xlim[0]:   # off scale (LLM pairs): arrow and the value
                ax.annotate("", xy=(xlim[0] + 0.1, y), xytext=(xlim[0] + 1.6, y),
                            arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=0.9))
                ax.text(xlim[0] + 1.8, y, f"{r.mean_diff:+.1f}", va="center", ha="left", fontsize=7, color=ORANGE)
                continue
            ax.plot([r.ci_low, r.ci_high], [y, y], color=colour, lw=1.4, solid_capstyle="butt")
            ax.plot(r.mean_diff, y, "o", ms=4, color=colour, mec="white", mew=0.5)
            ax.text(r.ci_high + 0.3, y, f"{r.mean_diff:+.1f}", va="center", fontsize=6.5,
                    color=INK2)
        ax.axvline(0, color=INK2, lw=0.6)
        ax.set_xlim(*xlim)
        ax.grid(axis="x", color=GRID, lw=0.4)
        ax.set_xlabel(f"Macro-F1 gain over BERT (points), {budget}% labels")
        ax.text(0.0, 1.02, f"({'ab'[budget == 10]}) {budget}% labels", transform=ax.transAxes, fontsize=8)
    axes[0].set_yticks(range(len(ROWS)))
    axes[0].set_yticklabels([r[0] for r in ROWS][::-1])
    fig.tight_layout(w_pad=1.0)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_methods.{ext}", bbox_inches="tight")
    plt.close(fig)


def fig_curve():
    s = pd.read_csv(RES / "pass10_seeds/analysis/summary.csv").fillna({"tag": ""})
    fig, ax = plt.subplots(figsize=(COL_W, 2.2))
    for arm, colour, label, marker in (("A1", BLUE, "BERT", "s"), ("A20", AQUA, "BERT + MLM pretraining", "o")):
        t = s[(s["arm"] == arm) & (s["tag"] == "")].sort_values("budget")
        ax.errorbar(t["budget"], t["test_macro_f1"], yerr=t["test_macro_f1_sd"], color=colour, marker=marker,
                    ms=3.5, lw=1.2, capsize=2, label=label)
    extra = s[(s["arm"] == "A20") & (s["tag"].isin(["pt1", "pt2"]))]
    ax.plot(extra["budget"], extra["test_macro_f1"], "x", color=AQUA, ms=4.5, mew=1.0,
            label="other pretraining seeds")
    ax.set_xscale("log")
    ax.set_xticks([1, 5, 10, 30, 100])
    ax.set_xticklabels(["1", "5", "10", "30", "100"])
    ax.set_xlabel("Labelled share of the training set (%)")
    ax.set_ylabel("Test macro-F1 (%)")
    ax.grid(color=GRID, lw=0.4)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_curve.{ext}", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_methods()
    fig_curve()
    print("wrote", sorted(p.name for p in OUT.glob("fig_*")))
