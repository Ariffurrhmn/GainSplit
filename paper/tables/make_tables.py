"""LaTeX tables for the paper, built from results/analysis/ (run from the repo root:
python paper/tables/make_tables.py). Writes .tex fragments next to this file."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
AN = ROOT / "results" / "analysis"
OUT = Path(__file__).resolve().parent

U = r"$^{\dagger}$"  # marks arms that train on the unlabelled pool
ARMS = [("A0", "TF-IDF + logistic regression"), ("A1", "BERT (labelled only)"),
        ("A2", "BERT + helper heads on $L$"), ("A3", "BERT + helper heads on $L{+}U$" + U),
        ("A4", "GAN-BERT" + U), ("A5", "GAN-BERT + helper heads" + U), ("A6", "Self-training" + U),
        ("A7", "BERT + FGM"), ("A8", "A3, shuffled helper labels" + U)]
BUDGETS = [1, 5, 10, 30, 100]


def main_results():
    df = pd.read_csv(AN / "summary_test_macro_f1_long.csv")
    best = df[df["arm"] != "A0"].groupby("budget")["mean"].max()
    lines = []
    for arm, name in ARMS:
        cells = []
        for b in BUDGETS:
            r = df[(df["arm"] == arm) & (df["budget"] == b)]
            if r.empty:
                cells.append("--")
                continue
            m, sd = r["mean"].iloc[0], r["sd"].iloc[0]
            txt = f"{m:.1f}" + (r"{\scriptsize$\pm${" + f"{sd:.1f}" + "}}" if b != 100 or arm != "A0" else "")
            if abs(m - best[b]) < 1e-9:
                txt = r"\textbf{" + f"{m:.1f}" + "}" + r"{\scriptsize$\pm${" + f"{sd:.1f}" + "}}"
            cells.append(txt)
        lines.append(f"{arm} & {name} & " + " & ".join(cells) + r" \\")
    head = "\n".join([r"\begin{tabular}{llccccc}", r"\toprule",
                      r" & Arm & 1\% (228) & 5\% (1,140) & 10\% (2,280) & 30\% (6,840) & 100\% (22,800) \\",
                      r"\midrule"])
    tail = "\n".join([r"\bottomrule", r"\end{tabular}"])
    (OUT / "main_results.tex").write_text("\n".join([head, *lines, tail]) + "\n", encoding="utf-8")


def gold():
    g = pd.read_csv(AN / "gold_sentiment_agreement.csv")
    names = {"sent_label": "Teacher 1 (used in training)", "vader_label": "VADER lexicon",
             "teacher2_label": "Teacher 2 (RoBERTa-large)"}
    lines = []
    for src, name in names.items():
        a = g[(g["source"] == src) & (g["subset"] == "all")].iloc[0]
        s = g[(g["source"] == src) & (g["subset"] != "all")].iloc[0]
        lines.append(f"{name} & {a['accuracy']:.1f} & {a['macro_f1']:.1f} & {a['kappa']:.2f} & {s['kappa']:.2f}" + r" \\")
    head = "\n".join([r"\begin{tabular}{lcccc}", r"\toprule",
                      r"Source & Acc. & Macro-F1 & $\kappa$ & $\kappa$ (sure) \\", r"\midrule"])
    tail = "\n".join([r"\bottomrule", r"\end{tabular}"])
    (OUT / "gold.tex").write_text("\n".join([head, *lines, tail]) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main_results()
    gold()
    print((OUT / "main_results.tex").read_text(encoding="utf-8"))
    print((OUT / "gold.tex").read_text(encoding="utf-8"))
