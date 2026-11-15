"""Pass 10 analysis (CPU): the MLM task-adaptive pretraining study, fixed before running.

Same method as Passes 4-9 (vault: Experiment Plan, "Pass 10 plan"): paired bootstrap over test
headlines, 10,000 resamples shared by every seed, statistic = mean over seeds 0-4, Holm within a table.
Scores are in points (macro-F1 0.882 -> 88.2). A key is (arm, tag); tag "" is the main run.

Three parts, one per notebook:
  stack        (notebook 19, Q1) the Pass 9 2x2x2 on the MLM encoder:
               A20 = none  A31 = H  A21 = F  A32 = S  A33 = H+F  A34 = H+S  A35 = F+S  A36 = H+F+S
               tables stack_f1 (gains on top of MLM) and stack_interactions (do they add up; and does
               pretraining shrink each ingredient's gain: (A_x on MLM - A20) - (A_x on BERT - A1))
  seeds_curve  (notebook 20, Q2 + Q3) A20 from three pretraining seeds (tags "", "pt1", "pt2"), and
               A20 at 5/30/100% next to A1, incl. cross-budget rows such as A20 at 1% vs A1 at 10%
  shortcuts    (notebook 21, Q4) cue-mask shortcut gap and iSarcasmEval AUC minus word count
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import analysis as an
from . import analysis_pass4 as a4
from . import analysis_pass5_7 as a57
from . import analysis_pass8 as a8
from .analysis_pass9 import verdict

BUDGETS = (1, 10)
STACK_ARMS = ("A31", "A32", "A33", "A34", "A35", "A36")
CELLS = {"A20": "MLM", "A31": "MLM+helpers", "A21": "MLM+FGM", "A32": "MLM+self-training",
         "A33": "MLM+helpers+FGM", "A34": "MLM+helpers+self-training", "A35": "MLM+FGM+self-training",
         "A36": "MLM+all three"}
# the same cell on plain BERT (Passes 1, 2, 9)
ON_BERT = {"A20": "A1", "A31": "A2", "A21": "A7", "A32": "A6", "A33": "A27", "A34": "A28", "A35": "A29", "A36": "A30"}
SINGLES = {"A33": ("A31", "A21"), "A34": ("A31", "A32"), "A35": ("A21", "A32"), "A36": ("A31", "A21", "A32")}
SEED_TAGS = ("", "pt1", "pt2")      # the three MLM encoders: Pass 8a's seed 0, then seeds 1 and 2
CURVE_BUDGETS = (1, 5, 10, 30, 100)


def _k(arm, tag=""):
    return (arm, tag)


def _diff(a, b, budget, what):
    return (f"{a}-{b} ({what})", budget, {_k(a): 1, _k(b): -1})


# ---------------------------------------------------------------- Q1: the stack on the MLM encoder

STACK_F1_SPECS = [s for b in BUDGETS for s in (
    _diff("A31", "A20", b, "helpers on top of MLM"),
    _diff("A21", "A20", b, "FGM on top of MLM"),
    _diff("A32", "A20", b, "self-training on top of MLM"),
    _diff("A33", "A20", b, "helpers+FGM on top of MLM"),
    _diff("A34", "A20", b, "helpers+self-training on top of MLM"),
    _diff("A35", "A20", b, "FGM+self-training on top of MLM"),
    _diff("A36", "A20", b, "all three on top of MLM"),
    _diff("A36", "A30", b, "MLM under the full stack"),
)]

STACK_INTERACTION_SPECS = [s for b in BUDGETS for s in (
    ("helpers x FGM (on MLM)", b, {_k("A33"): 1, _k("A31"): -1, _k("A21"): -1, _k("A20"): 1}),
    ("helpers x self-training (on MLM)", b, {_k("A34"): 1, _k("A31"): -1, _k("A32"): -1, _k("A20"): 1}),
    ("FGM x self-training (on MLM)", b, {_k("A35"): 1, _k("A21"): -1, _k("A32"): -1, _k("A20"): 1}),
    ("all three vs sum of single gains (on MLM)", b,
     {_k("A36"): 1, _k("A31"): -1, _k("A21"): -1, _k("A32"): -1, _k("A20"): 2}),
    ("pretraining x helpers", b, {_k("A31"): 1, _k("A20"): -1, _k("A2"): -1, _k("A1"): 1}),
    ("pretraining x FGM", b, {_k("A21"): 1, _k("A20"): -1, _k("A7"): -1, _k("A1"): 1}),
    ("pretraining x self-training", b, {_k("A32"): 1, _k("A20"): -1, _k("A6"): -1, _k("A1"): 1}),
)]

# ---------------------------------------------------------------- Q2 + Q3: pretraining seeds, label curve

SEEDS_SPECS = [s for b in BUDGETS for s in (
    ("A20[pt1]-A1 (pretraining seed 1)", b, {_k("A20", "pt1"): 1, _k("A1"): -1}),
    ("A20[pt2]-A1 (pretraining seed 2)", b, {_k("A20", "pt2"): 1, _k("A1"): -1}),
    ("mean of 3 encoders - A1", b, {_k("A20"): 1 / 3, _k("A20", "pt1"): 1 / 3, _k("A20", "pt2"): 1 / 3,
                                    _k("A1"): -1}),
)]


def _at(arm, budget):
    """Alias of an arm's runs at another budget, so a cross-budget contrast can use the paired bootstrap."""
    return f"{arm}@{budget}"


# cross-budget rows: (name, budget the alias is placed at, weights, [(arm, from_budget)] to alias)
CROSS_BUDGET = [
    ("A20@1% - A1@10%", 1, "A1", 10),
    ("A20@5% - A1@30%", 5, "A1", 30),
    ("A20@10% - A1@30%", 10, "A1", 30),
    ("A20@30% - A1@100%", 30, "A1", 100),
]
CURVE_SPECS = ([_diff("A20", "A1", b, "MLM vs BERT") for b in (5, 30, 100)]
               + [(name, b, {_k("A20"): 1, _k(_at(ref, rb)): -1}) for name, b, ref, rb in CROSS_BUDGET])
LONG_SPECS = [s for b in BUDGETS for s in (
    ("A20[pt20k]-A20 (20,000 vs 10,000 pretraining steps)", b, {_k("A20", "pt20k"): 1, _k("A20"): -1}),)]

# ---------------------------------------------------------------- Q4: shortcuts and transfer

GAP_SPECS = [(key, b, ref) for b in BUDGETS for key, ref in (
    (_k("A20"), "A1"), (_k("A21"), "A7"), (_k("A22"), "A1"), (_k("A23"), "A7"),
    (_k("A20", "pt1"), "A1"), (_k("A20", "pt2"), "A1"), (_k("A36"), "A1"))]
LENGTH_SPECS = [(_k(arm), b) for b in BUDGETS for arm in ("A20", "A21", "A22", "A23", "A36")]

# ---------------------------------------------------------------- predictions (fixed before running)
# (hypothesis, table, contrast name, budget, direction, claim); direction as in analysis_pass9.verdict
HYPOTHESES = [
    ("H10.1", "stack_f1", "A32-A20 (self-training on top of MLM)", 1, "+", "self-training adds on top of MLM at 1%"),
    ("H10.1", "stack_f1", "A32-A20 (self-training on top of MLM)", 10, None, "... at 10% (no direction)"),
    ("H10.2", "stack_f1", "A31-A20 (helpers on top of MLM)", 1, "0", "helpers add nothing on top of MLM at 1%"),
    ("H10.2", "stack_f1", "A31-A20 (helpers on top of MLM)", 10, "0", "... nor at 10%"),
    ("H10.2", "stack_f1", "A21-A20 (FGM on top of MLM)", 1, "0", "FGM adds nothing on top of MLM at 1%"),
    ("H10.2", "stack_f1", "A21-A20 (FGM on top of MLM)", 10, "0", "... nor at 10%"),
    ("H10.3", "stack_f1", "A36-A20 (all three on top of MLM)", 1, "+", "the full stack beats MLM alone at 1%"),
    ("H10.3", "stack_f1", "A36-A20 (all three on top of MLM)", 10, None, "... at 10% (no direction)"),
    ("H10.4", "stack_interactions", "pretraining x helpers", 1, "-", "pretraining shrinks the helpers' gain at 1%"),
    ("H10.4", "stack_interactions", "pretraining x FGM", 1, "-", "pretraining shrinks FGM's gain at 1%"),
    ("H10.4", "stack_interactions", "pretraining x self-training", 1, None, "pretraining and self-training (no direction)"),
    ("H10.5", "stack_f1", "A36-A30 (MLM under the full stack)", 1, "+", "MLM still adds under the full stack at 1%"),
    ("H10.5", "stack_f1", "A36-A30 (MLM under the full stack)", 10, "+", "... and at 10%"),
    ("H10.6", "seeds_f1", "A20[pt1]-A1 (pretraining seed 1)", 1, "+", "pretraining seed 1 helps at 1%"),
    ("H10.6", "seeds_f1", "A20[pt1]-A1 (pretraining seed 1)", 10, "+", "... and at 10%"),
    ("H10.6", "seeds_f1", "A20[pt2]-A1 (pretraining seed 2)", 1, "+", "pretraining seed 2 helps at 1%"),
    ("H10.6", "seeds_f1", "A20[pt2]-A1 (pretraining seed 2)", 10, "+", "... and at 10%"),
    ("H10.7", "curve_f1", "A20-A1 (MLM vs BERT)", 5, "+", "MLM helps at 5%"),
    ("H10.7", "curve_f1", "A20-A1 (MLM vs BERT)", 30, "+", "MLM helps at 30%"),
    ("H10.7", "curve_f1", "A20-A1 (MLM vs BERT)", 100, "0", "no difference with every label"),
    ("H10.8", "curve_f1", "A20@1% - A1@10%", 1, "0", "MLM with 1% labels matches BERT with 10%"),
    ("H10.8", "curve_f1", "A20@10% - A1@30%", 10, "0", "MLM with 10% labels matches BERT with 30%"),
    ("H10.9", "gap_contrasts", "gap(A20) - gap(A1)", 1, "-", "MLM leans less on outlet cue words at 1%"),
    ("H10.9", "gap_contrasts", "gap(A20) - gap(A1)", 10, "-", "... and at 10%"),
    *[("H10.10", "auc_vs_length", f"AUC {arm} - word count", b, "-",
       f"{arm} stays below word count alone on iSarcasmEval at {b}%")
      for b in BUDGETS for arm in ("A20", "A21", "A22", "A23", "A36")],
]
PART_TABLES = {"stack": ("stack_f1", "stack_interactions"),
               "seeds_curve": ("seeds_f1", "curve_f1", "long_f1"),
               "shortcuts": ("gap_contrasts", "auc_vs_length")}
PARTS = tuple(PART_TABLES)


# ---------------------------------------------------------------- reading aids

def _mean_sd(keep, cols, by):
    g = keep.groupby(by)[cols]
    sd = (g.std(ddof=1) * 100).round(2)
    sd.columns = [f"{c}_sd" for c in sd.columns]
    out = (g.mean() * 100).round(2).join(sd)
    out["n_seeds"] = g.size()
    return out.reset_index()


def summary(runs, arms, tags=("",)) -> pd.DataFrame:
    """Mean and sd over seeds (points) per budget, arm and tag."""
    keep = runs[runs["arm"].isin(arms) & runs["tag"].isin(tags)]
    cols = [c for c in ("test_accuracy", "test_macro_f1", "dev_macro_f1", "test_cuemask_macro_f1",
                        "isarc_test_roc_auc") if c in keep]
    out = _mean_sd(keep, cols, ["budget", "arm", "tag"])
    out.insert(3, "cell", out["arm"].map({**CELLS, "A1": "BERT", "A2": "helpers", "A7": "FGM", "A6": "self-training",
                                          "A27": "helpers+FGM", "A28": "helpers+self-training",
                                          "A29": "FGM+self-training", "A30": "all three"}).fillna(""))
    return out


def additivity(runs) -> pd.DataFrame:
    """Within the MLM encoder: each cell's gain over A20 next to the sum of its single gains (no test)."""
    keep = runs[runs["arm"].isin(tuple(CELLS)) & (runs["tag"] == "")]
    m = keep.groupby(["budget", "arm"])["test_macro_f1"].mean() * 100
    rows = []
    for b in sorted(keep["budget"].unique()):
        if (b, "A20") not in m:
            continue
        base = m[(b, "A20")]
        for arm, singles in SINGLES.items():
            if (b, arm) not in m or any((b, s) not in m for s in singles):
                continue
            expected = sum(m[(b, s)] - base for s in singles)
            rows.append({"budget": b, "arm": arm, "cell": CELLS[arm], "gain_over_mlm": round(m[(b, arm)] - base, 2),
                         "sum_of_single_gains": round(expected, 2),
                         "share_of_sum": round((m[(b, arm)] - base) / expected, 2) if expected else None})
    return pd.DataFrame(rows, columns=["budget", "arm", "cell", "gain_over_mlm", "sum_of_single_gains", "share_of_sum"])


def on_bert_vs_mlm(runs) -> pd.DataFrame:
    """Each 2x2x2 cell on plain BERT and on the MLM encoder, side by side (mean test macro-F1, points)."""
    m = runs[runs["tag"] == ""].groupby(["budget", "arm"])["test_macro_f1"].mean() * 100
    rows = []
    for b in BUDGETS:
        for mlm_arm, bert_arm in ON_BERT.items():
            if (b, mlm_arm) in m and (b, bert_arm) in m:
                rows.append({"budget": b, "cell": CELLS[mlm_arm].replace("MLM+", "").replace("MLM", "none"),
                             "bert_arm": bert_arm, "on_bert": round(m[(b, bert_arm)], 2),
                             "mlm_arm": mlm_arm, "on_mlm": round(m[(b, mlm_arm)], 2),
                             "mlm_gain": round(m[(b, mlm_arm)] - m[(b, bert_arm)], 2)})
    return pd.DataFrame(rows, columns=["budget", "cell", "bert_arm", "on_bert", "mlm_arm", "on_mlm", "mlm_gain"])


def pseudo_health(runs, train_labels: pd.DataFrame | None = None) -> pd.DataFrame:
    """Self-training arms: teacher recipe and dev F1, pseudo-labels kept and, when `train_labels`
    (id, label of headlines_train.csv) is given and the run saved pseudo_labels.csv, their accuracy."""
    truth = None if train_labels is None else train_labels.set_index("id")["label"]
    rows = []
    for _, r in runs[runs["arm"].isin(["A6", "A28", "A29", "A30", "A32", "A34", "A35", "A36"]) & (runs["tag"] == "")].iterrows():
        p = json.loads((Path(r["run_dir"]) / "result.json").read_text()).get("pseudo") or {}
        if not p:
            continue
        row = {"arm": r["arm"], "budget": r["budget"], "teacher_recipe": p.get("teacher_recipe", "A1"),
               "teacher_dev_macro_f1": p["teacher_dev_macro_f1"] * 100, "n_kept": p["n_kept"],
               "pseudo_accuracy": np.nan}
        f = Path(r["run_dir"]) / "pseudo_labels.csv"
        if truth is not None and f.exists():
            k = pd.read_csv(f)
            if len(k):
                row["pseudo_accuracy"] = float((k["pseudo_label"].to_numpy() == truth.loc[k["id"]].to_numpy()).mean() * 100)
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    t = pd.DataFrame(rows)
    return (t.groupby(["budget", "arm", "teacher_recipe"]).mean().round(2).reset_index()
            .sort_values(["budget", "arm"]).reset_index(drop=True))


def seed_spread(runs) -> pd.DataFrame:
    """Q2: mean test macro-F1 of A20 per pretraining encoder, and the spread across encoders (points)."""
    keep = runs[(runs["arm"] == "A20") & runs["tag"].isin(SEED_TAGS) & runs["budget"].isin(BUDGETS)]
    rows = []
    for b in BUDGETS:
        m = keep[keep["budget"] == b].groupby("tag")["test_macro_f1"].mean() * 100
        if not len(m):
            continue
        row = {"budget": b, **{f"encoder_{t or 'pt0'}": round(m.get(t, np.nan), 2) for t in SEED_TAGS}}
        row.update(n_encoders=int(m.notna().sum()), mean=round(m.mean(), 2), range=round(m.max() - m.min(), 2),
                   sd=round(m.std(ddof=1), 2) if len(m) > 1 else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def _matched_budget(a1_curve: dict, f1: float):
    """The A1 budget (%) whose interpolated F1 equals f1, linear in log10(budget); None if outside the curve."""
    bs = sorted(a1_curve)
    xs, ys = np.log10(bs), np.array([a1_curve[b] for b in bs])
    if f1 < ys[0]:
        return None
    if f1 > ys[-1]:
        return float("inf")
    for i in range(len(bs) - 1):
        lo, hi = ys[i], ys[i + 1]
        if lo <= f1 <= hi and hi > lo:
            return float(10 ** (xs[i] + (f1 - lo) / (hi - lo) * (xs[i + 1] - xs[i])))
    return None


def curve(runs) -> pd.DataFrame:
    """Q3: mean test macro-F1 of A1 and A20 per budget, and the A1 budget that A20's score matches
    (log-linear interpolation of A1's curve; 'more than 100%' if A20 is above A1 with every label)."""
    m = runs[runs["tag"] == ""].groupby(["arm", "budget"])["test_macro_f1"].mean() * 100
    a1 = {b: m[("A1", b)] for b in CURVE_BUDGETS if ("A1", b) in m}
    rows = []
    for b in CURVE_BUDGETS:
        if ("A20", b) not in m:
            continue
        f = m[("A20", b)]
        match = _matched_budget(a1, f) if len(a1) >= 2 else None
        rows.append({"budget": b, "A1": round(a1.get(b, np.nan), 2), "A20": round(f, 2),
                     "gain": round(f - a1.get(b, np.nan), 2),
                     "A1_budget_matched": ("more than 100" if match == float("inf") else
                                           None if match is None else round(match, 1)),
                     "label_factor": (None if match in (None, float("inf")) else round(match / b, 1))})
    return pd.DataFrame(rows, columns=["budget", "A1", "A20", "gain", "A1_budget_matched", "label_factor"])


def with_budget_aliases(runs) -> pd.DataFrame:
    """Add alias rows (arm 'A1@10' at budget 1, ...) for the cross-budget contrasts in CURVE_SPECS.

    The alias is A1's runs at the other budget, relabelled; the paired bootstrap then pairs seed s of
    A20 with seed s of the alias over the same test items. Different seeds draw different labeled sets,
    which the bootstrap does not model (it resamples test items only)."""
    extra = []
    for _, b, ref, rb in CROSS_BUDGET:
        rows = runs[(runs["arm"] == ref) & (runs["tag"] == "") & (runs["budget"] == rb)].copy()
        rows["arm"], rows["budget"] = _at(ref, rb), b
        extra.append(rows)
    return pd.concat([runs, *extra], ignore_index=True) if extra else runs


def verdicts(tables, part=None) -> pd.DataFrame:
    """One row per prediction whose table belongs to `part` (all parts if None)."""
    wanted = None if part is None else set(PART_TABLES[part])
    rows = []
    for h, table, name, b, direction, text in HYPOTHESES:
        if wanted is not None and table not in wanted:
            continue
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


def run_all(results_dirs, part, out_dir, n=10000, rng_seed=0, pretrain_infos=(), train_labels=None,
            isarc_test=None) -> dict:
    """Write one part's tables to out_dir (CSV + Markdown) and return them. Contrasts without runs are skipped.

    `train_labels` (id, label of headlines_train.csv): scores the saved pseudo-labels (stack part).
    `isarc_test` (id, text, label): needed for the word-count table (shortcuts part)."""
    if part not in PARTS:
        raise ValueError(f"part must be one of {PARTS}")
    runs = an.load_runs(*results_dirs)
    if part == "stack":
        tables = {
            "summary": summary(runs, tuple(CELLS) + tuple(ON_BERT.values())),
            "on_bert_vs_mlm": on_bert_vs_mlm(runs),
            "additivity": additivity(runs),
            "stack_f1": a57.contrast_table(runs, "macro_f1", n, rng_seed, specs=STACK_F1_SPECS),
            "stack_interactions": a57.contrast_table(runs, "macro_f1", n, rng_seed, specs=STACK_INTERACTION_SPECS),
            "seed_wins": _seed_wins(runs, STACK_F1_SPECS),
            "pseudo_health": pseudo_health(runs, train_labels),
        }
    elif part == "seeds_curve":
        aliased = with_budget_aliases(runs)
        tables = {
            "summary": summary(runs, ("A1", "A20"), tags=SEED_TAGS + ("pt20k",)),
            "seed_spread": seed_spread(runs),
            "curve": curve(runs),
            "seeds_f1": a57.contrast_table(runs, "macro_f1", n, rng_seed, specs=SEEDS_SPECS),
            "curve_f1": a57.contrast_table(aliased, "macro_f1", n, rng_seed, specs=CURVE_SPECS),
            "curve_accuracy_100": a57.contrast_table(runs, "accuracy", n, rng_seed,
                                                     specs=[s for s in CURVE_SPECS if s[1] == 100]),
            "long_f1": a57.contrast_table(runs, "macro_f1", n, rng_seed, specs=LONG_SPECS),
            "seed_wins": _seed_wins(runs, SEEDS_SPECS + LONG_SPECS + [s for s in CURVE_SPECS if "@" not in s[0]]),
        }
    else:
        tables = {
            "summary": summary(runs, ("A1", "A7", "A20", "A21", "A22", "A23", "A36"), tags=("", "mask", "pt1", "pt2")),
            "gap_contrasts": a57.gap_table(runs, n, rng_seed, specs=GAP_SPECS),
        }
        if isarc_test is not None:
            tables["auc_vs_length"] = a57.length_table(runs, isarc_test, n, rng_seed, specs=LENGTH_SPECS)
    if (runs["tag"] == "rerun").any() and part == "stack":
        tables["rerun_check"] = a4.check_reruns(runs, tags=("rerun",))
    if pretrain_infos:
        tables["pretrain"] = pretrain_table(pretrain_infos)
    tables["verdicts"] = verdicts(tables, part)
    for name, t in tables.items():
        if t is not None and len(t):
            an.save_table(t, out_dir, name, default_decimals=3)
    return tables


def pretrain_table(info_paths) -> pd.DataFrame:
    """Pass 8's receipt table plus each encoder's pretraining seed."""
    t = a8.pretrain_table(info_paths)
    seeds = []
    for p in info_paths:
        p = Path(p)
        if p.exists():
            seeds.append(json.loads(p.read_text())["config"]["seed"])
    if len(t):
        t.insert(1, "seed", seeds)
    return t


def _seed_wins(runs, specs) -> pd.DataFrame:
    """For each simple difference: on how many shared seeds the first key is ahead (no test)."""
    rows = []
    for name, budget, w in specs:
        if len(w) != 2:
            continue
        a = [k for k, v in w.items() if v > 0][0]
        b = [k for k, v in w.items() if v < 0][0]
        ga = runs[(runs["arm"] == a[0]) & (runs["tag"] == a[1]) & (runs["budget"] == budget)].set_index("seed")
        gb = runs[(runs["arm"] == b[0]) & (runs["tag"] == b[1]) & (runs["budget"] == budget)].set_index("seed")
        seeds = sorted(set(ga.index) & set(gb.index))
        if not seeds:
            continue
        d = (ga.loc[seeds, "test_macro_f1"] - gb.loc[seeds, "test_macro_f1"]) * 100
        rows.append({"contrast": name, "budget": budget, "seeds_ahead": int((d > 0).sum()), "n_seeds": len(seeds),
                     "per_seed_diff": " ".join(f"{v:+.2f}" for v in d)})
    return pd.DataFrame(rows, columns=["contrast", "budget", "seeds_ahead", "n_seeds", "per_seed_diff"])
