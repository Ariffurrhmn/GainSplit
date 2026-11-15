"""Pass 9 analysis (CPU): do helper heads on L, FGM and self-training stack? Fixed before running.

Same method as Passes 4-8 (vault: Experiment Plan, "Pass 9 plan"): paired bootstrap over test
headlines, 10,000 resamples shared by every seed, statistic = mean over seeds 0-4, Holm within a table.
Scores are in points (macro-F1 0.882 -> 88.2).

The design is a 2x2x2 over three ingredients: H = helper heads on L, F = FGM, S = self-training.
  A1 = none   A2 = H   A7 = F   A6 = S   A27 = H+F   A28 = H+S   A29 = F+S   A30 = H+F+S
Two tables, each with its own Holm correction:
  contrasts_f1   what each addition gains, at 1% and 10% (plus A27 - A7 at 100%)
  interactions   whether gains add up: (A_xy - A_x) - (A_y - A1) = A_xy - A_x - A_y + A1 is 0 when
                 two ingredients add exactly, negative when they overlap, positive when they reinforce;
                 and A30 against the sum of the three single gains (A30 - A2 - A6 - A7 + 2 A1)
"""
import json
from pathlib import Path

import pandas as pd

from . import analysis as an
from . import analysis_pass4 as a4
from . import analysis_pass5_7 as a57

BUDGETS = (1, 10)
NEW_ARMS = ("A27", "A28", "A29", "A30")
CELL_ARMS = ("A1", "A2", "A6", "A7") + NEW_ARMS
NAMES = {"A1": "BERT", "A2": "helpers", "A7": "FGM", "A6": "self-training", "A27": "helpers+FGM",
         "A28": "helpers+self-training", "A29": "FGM+self-training", "A30": "all three"}


def _w(**arms):
    return {(a, ""): v for a, v in arms.items()}


def _diff(a, b, budget, what):
    return (f"{a}-{b} ({what})", budget, _w(**{a: 1, b: -1}))


F1_SPECS = [s for b in BUDGETS for s in (
    _diff("A27", "A7", b, "helpers on top of FGM"),
    _diff("A27", "A2", b, "FGM on top of helpers"),
    _diff("A28", "A2", b, "self-training on top of helpers"),
    _diff("A29", "A7", b, "self-training on top of FGM"),
    _diff("A30", "A27", b, "self-training on top of helpers+FGM"),
    _diff("A30", "A1", b, "whole stack vs BERT"),
    _diff("A30", "A7", b, "whole stack vs FGM alone"),
)] + [_diff("A27", "A7", 100, "helpers on top of FGM")]

INTERACTION_SPECS = [s for b in BUDGETS for s in (
    ("helpers x FGM", b, _w(A27=1, A2=-1, A7=-1, A1=1)),
    ("helpers x self-training", b, _w(A28=1, A2=-1, A6=-1, A1=1)),
    ("FGM x self-training", b, _w(A29=1, A7=-1, A6=-1, A1=1)),
    ("all three vs sum of single gains", b, _w(A30=1, A2=-1, A6=-1, A7=-1, A1=2)),
)]

# Predictions fixed before running (vault: Experiment Plan, "Pass 9 plan").
# direction: "+" positive and significant, "-" negative and significant, "0" no significant difference,
# None: no direction predicted (only significance is reported)
HYPOTHESES = [
    ("H9.1", "contrasts_f1", "A30-A1 (whole stack vs BERT)", 1, "+", "the stack beats plain BERT at 1%"),
    ("H9.1", "contrasts_f1", "A30-A1 (whole stack vs BERT)", 10, "+", "... and at 10%"),
    ("H9.2", "contrasts_f1", "A30-A7 (whole stack vs FGM alone)", 1, "+",
     "the stack beats the best single ingredient at 1% (the 'more gain' claim)"),
    ("H9.2", "contrasts_f1", "A30-A7 (whole stack vs FGM alone)", 10, "0", "no difference at 10%"),
    ("H9.3", "interactions", "helpers x FGM", 1, "-", "helpers and FGM overlap at 1% (both regularisers)"),
    ("H9.3", "interactions", "all three vs sum of single gains", 1, "-",
     "the stack is below the sum of the three single gains at 1%"),
    ("H9.4", "interactions", "helpers x self-training", 1, None, "helpers and self-training (no direction)"),
    ("H9.4", "interactions", "FGM x self-training", 1, None, "FGM and self-training (no direction)"),
    ("H9.5", "contrasts_f1", "A27-A7 (helpers on top of FGM)", 100, "0", "no difference with every label"),
]


def summary(runs) -> pd.DataFrame:
    """Mean and sd over seeds (points) for the eight cells, untagged runs only."""
    keep = runs[runs["arm"].isin(CELL_ARMS) & (runs["tag"] == "")]
    cols = [c for c in ("test_accuracy", "test_macro_f1", "dev_macro_f1", "isarc_test_roc_auc") if c in keep]
    g = keep.groupby(["budget", "arm"])[cols]
    sd = (g.std(ddof=1) * 100).round(2)
    sd.columns = [f"{c}_sd" for c in sd.columns]
    out = (g.mean() * 100).round(2).join(sd)
    out["n_seeds"] = g.size()
    out = out.reset_index()
    out.insert(2, "ingredients", out["arm"].map(NAMES))
    return out


def additivity(runs) -> pd.DataFrame:
    """A reading aid (no test): each cell's gain over BERT next to the gain the single ingredients
    would predict if they added exactly. Mean test macro-F1 over seeds, points."""
    keep = runs[runs["arm"].isin(CELL_ARMS) & (runs["tag"] == "")]
    m = keep.groupby(["budget", "arm"])["test_macro_f1"].mean() * 100
    parts = {"A27": ("A2", "A7"), "A28": ("A2", "A6"), "A29": ("A7", "A6"), "A30": ("A2", "A7", "A6")}
    rows = []
    for b in sorted(keep["budget"].unique()):
        if (b, "A1") not in m:
            continue
        base = m[(b, "A1")]
        for arm, singles in parts.items():
            if (b, arm) not in m or any((b, s) not in m for s in singles):
                continue
            expected = sum(m[(b, s)] - base for s in singles)
            rows.append({"budget": b, "arm": arm, "ingredients": NAMES[arm],
                         "gain_over_bert": round(m[(b, arm)] - base, 2),
                         "sum_of_single_gains": round(expected, 2),
                         "share_of_sum": round((m[(b, arm)] - base) / expected, 2) if expected else None})
    return pd.DataFrame(rows, columns=["budget", "arm", "ingredients", "gain_over_bert", "sum_of_single_gains",
                                       "share_of_sum"])


def seed_wins(runs, specs=F1_SPECS) -> pd.DataFrame:
    """For each simple difference: on how many shared seeds the first arm is ahead (no test)."""
    rows = []
    for name, budget, w in specs:
        if len(w) != 2:
            continue
        (a, _), (b, _) = [k for k, v in w.items() if v == 1][0], [k for k, v in w.items() if v == -1][0]
        ga = runs[(runs["arm"] == a) & (runs["tag"] == "") & (runs["budget"] == budget)].set_index("seed")
        gb = runs[(runs["arm"] == b) & (runs["tag"] == "") & (runs["budget"] == budget)].set_index("seed")
        seeds = sorted(set(ga.index) & set(gb.index))
        if not seeds:
            continue
        d = (ga.loc[seeds, "test_macro_f1"] - gb.loc[seeds, "test_macro_f1"]) * 100
        rows.append({"contrast": name, "budget": budget, "seeds_ahead": int((d > 0).sum()), "n_seeds": len(seeds),
                     "per_seed_diff": " ".join(f"{v:+.2f}" for v in d)})
    return pd.DataFrame(rows, columns=["contrast", "budget", "seeds_ahead", "n_seeds", "per_seed_diff"])


def pseudo_health(runs) -> pd.DataFrame:
    """Self-training arms: the teacher's recipe and dev macro-F1, and how many pseudo-labels it kept."""
    rows = []
    for _, r in runs[runs["arm"].isin(["A6", "A28", "A29", "A30"]) & (runs["tag"] == "")].iterrows():
        p = json.loads((Path(r["run_dir"]) / "result.json").read_text()).get("pseudo") or {}
        if p:
            rows.append({"arm": r["arm"], "budget": r["budget"], "teacher_recipe": p.get("teacher_recipe", "A1"),
                         "teacher_dev_macro_f1": p["teacher_dev_macro_f1"] * 100, "n_kept": p["n_kept"]})
    if not rows:
        return pd.DataFrame()
    t = pd.DataFrame(rows)
    return (t.groupby(["budget", "arm", "teacher_recipe"]).mean().round(2).reset_index()
            .sort_values(["budget", "arm"]).reset_index(drop=True))


def verdict(diff, p_holm, direction) -> str:
    sig = p_holm < 0.05
    if direction == "+":
        return "SUPPORTED" if sig and diff > 0 else ("opposite, significant" if sig else "not supported (no significant difference)")
    if direction == "-":
        return "SUPPORTED" if sig and diff < 0 else ("opposite, significant" if sig else "not supported (no significant difference)")
    if direction == "0":
        return "SUPPORTED (no significant difference)" if not sig else "not supported (significant difference)"
    return "significant difference" if sig else "no significant difference"


def verdicts(tables) -> pd.DataFrame:
    """One row per prediction: the difference, its interval, Holm p and the verdict."""
    rows = []
    for h, table, name, b, direction, text in HYPOTHESES:
        t = tables.get(table)
        hit = None if t is None or not len(t) else t[(t["contrast"] == name) & (t["budget"] == b)]
        if hit is None or not len(hit):
            rows.append({"hypothesis": h, "contrast": name, "budget": b, "verdict": "no runs (skipped)", "claim": text})
            continue
        r = hit.iloc[0]
        rows.append({"hypothesis": h, "contrast": name, "budget": b, "mean_diff": round(r["mean_diff"], 2),
                     "ci_low": round(r["ci_low"], 2), "ci_high": round(r["ci_high"], 2),
                     "p_holm": round(r["p_holm"], 4), "verdict": verdict(r["mean_diff"], r["p_holm"], direction),
                     "claim": text})
    return pd.DataFrame(rows, columns=["hypothesis", "contrast", "budget", "mean_diff", "ci_low", "ci_high",
                                       "p_holm", "verdict", "claim"])


def run_all(results_dirs, out_dir="results/analysis_pass9", n=10000, rng_seed=0) -> dict:
    """Write every Pass 9 table to out_dir (CSV + Markdown) and return them."""
    runs = an.load_runs(*results_dirs)
    tables = {
        "summary": summary(runs),
        "additivity": additivity(runs),
        "contrasts_f1": a57.contrast_table(runs, "macro_f1", n, rng_seed, specs=F1_SPECS),
        "interactions": a57.contrast_table(runs, "macro_f1", n, rng_seed, specs=INTERACTION_SPECS),
        "seed_wins": seed_wins(runs),
        "contrasts_accuracy_100": a57.contrast_table(runs, "accuracy", n, rng_seed,
                                                     specs=[s for s in F1_SPECS if s[1] == 100]),
        "pseudo_health": pseudo_health(runs),
    }
    if (runs["tag"] == "rerun").any():
        tables["rerun_check"] = a4.check_reruns(runs, tags=("rerun",))
    tables["verdicts"] = verdicts(tables)
    for name, t in tables.items():
        if len(t):
            an.save_table(t, out_dir, name, default_decimals=3)
    return tables
