"""Pass 11 analysis (CPU only): the seed recipe against our recipe across label budgets, the seed paper's 70% regime,
GAN-BERT with many passes over U, the MLM dose curve, and the seed recipe on the MLM encoder.

Paired bootstrap over test headlines, 10,000 resamples shared by every seed, statistic = mean over seeds 0-4,
Holm within each table. Predictions: vault/Pass 11 Predictions.md (fixed before running). A directional prediction
counts as supported only if the difference goes the predicted way and p_holm < 0.05; "0" predictions are supported
when p_holm >= 0.05 (not proof of equality: read the interval); "not+" is supported unless the difference is
significantly positive.

Tables (each its own Holm family):
  ours_vs_seed   A36 - A5 at 1, 5, 10, 30%                                  (H11.18, H11.19)
  seed_parts     A5 - A1 at 1, 5, 10, 30, 70% and A4 - A1 at 1, 5, 10, 30%  (H11.21)
  high_f1        pairwise among A1, A5, A7, A20 at 70%, macro-F1            (H11.20)
  high_accuracy  the same contrasts on accuracy (the seed paper's metric)
  gan_long       A4 - A1 at 35x steps (1%), plus the two own-baseline rows  (H11.16)
  dose           A20 from each short pretraining - A1 at 1%                  (descriptive; H11.5-7 in dose_summary)
  a37            A37 - A20 and A37 - A5 at 1, 10%                            (H11.22)
Reading aids without tests: summary, dose_summary, high_summary.
"""
import numpy as np
import pandas as pd

from . import analysis as an
from . import analysis_pass4 as a4
from . import analysis_pass5_7 as a57

BUDGETS_CURVE = (1, 5, 10, 30)
HIGH = 70
DOSE_TAGS = (("dose100", 100), ("dose350", 350), ("dose1000", 1000), ("dose3000", 3000))


def _k(arm, tag=""):
    return (arm, tag)


def _diff(a, b, budget, what, ta="", tb=""):
    return (f"{a}-{b} ({what})" if not (ta or tb) else f"{a}[{ta}]-{b}[{tb}] ({what})" if tb else f"{a}[{ta}]-{b} ({what})",
            budget, {_k(a, ta): 1, _k(b, tb): -1})


OURS_SPECS = [_diff("A36", "A5", b, "our recipe vs the seed recipe") for b in BUDGETS_CURVE]
SEED_PARTS_SPECS = ([_diff("A5", "A1", b, "seed recipe vs BERT") for b in BUDGETS_CURVE + (HIGH,)]
                    + [_diff("A4", "A1", b, "GAN vs BERT") for b in BUDGETS_CURVE])
HIGH_SPECS = [_diff(a, b, HIGH, "at 70% labels") for a, b in
              (("A5", "A1"), ("A7", "A1"), ("A20", "A1"), ("A5", "A7"), ("A5", "A20"), ("A7", "A20"))]
GANLONG_SPECS = [
    _diff("A4", "A1", 1, "GAN vs BERT at 35x steps", "long35", "long35"),
    _diff("A1", "A1", 1, "BERT at 35x steps vs normal steps", "long35", ""),
    _diff("A4", "A4", 1, "GAN at 35x steps vs normal steps", "long35", ""),
]
DOSE_SPECS = [(f"A20[{tag}]-A1 ({steps} pretraining steps)", 1, {_k("A20", tag): 1, _k("A1"): -1})
              for tag, steps in DOSE_TAGS] + [("A20-A1 (10,000 pretraining steps)", 1, {_k("A20"): 1, _k("A1"): -1})]
A37_SPECS = [s for b in (1, 10) for s in (
    _diff("A37", "A20", b, "seed recipe on the MLM encoder vs MLM alone"),
    _diff("A37", "A5", b, "seed recipe on the MLM encoder vs on plain BERT"))]

# (hypothesis, table, contrast name, budget, direction, claim); directions: "+", "-", "0", "not+", None
HYPOTHESES = [
    *[("H11.18", "ours_vs_seed", "A36-A5 (our recipe vs the seed recipe)", b, "+", f"our recipe beats the seed recipe at {b}%")
      for b in (1, 10)],
    *[("H11.19", "ours_vs_seed", "A36-A5 (our recipe vs the seed recipe)", b, "+", f"... and at {b}%") for b in (5, 30)],
    *[("H11.20", "high_f1", f"{a}-{b} (at 70% labels)", HIGH, "not+", f"the seed recipe is not better than {b} at 70%")
      for a, b in (("A5", "A1"), ("A5", "A7"), ("A5", "A20"))],
    *[("H11.21", "seed_parts", "A5-A1 (seed recipe vs BERT)", b, "not+", f"the seed recipe does not beat BERT at {b}%")
      for b in BUDGETS_CURVE + (HIGH,)],
    ("H11.16", "gan_long", "A4[long35]-A1[long35] (GAN vs BERT at 35x steps)", 1, "0",
     "the GAN does not beat BERT at 35x steps (about 10 passes over U)"),
    *[("H11.22", "a37", "A37-A20 (seed recipe on the MLM encoder vs MLM alone)", b, "not+",
       f"the seed recipe adds nothing on the MLM encoder at {b}%") for b in (1, 10)],
]


def verdict11(diff, p_holm, direction) -> str:
    sig = p_holm < 0.05
    if direction == "+":
        return "SUPPORTED" if sig and diff > 0 else ("opposite, significant" if sig else "not supported (no significant difference)")
    if direction == "-":
        return "SUPPORTED" if sig and diff < 0 else ("opposite, significant" if sig else "not supported (no significant difference)")
    if direction == "0":
        return "SUPPORTED (no significant difference)" if not sig else "not supported (significant difference)"
    if direction == "not+":
        return "not supported (significantly better)" if sig and diff > 0 else "SUPPORTED (not better)"
    return "significant difference" if sig else "no significant difference"


def summary(runs, arms=("A1", "A4", "A5", "A7", "A20", "A36", "A37")) -> pd.DataFrame:
    """Mean and s.d. over seeds of test macro-F1 and accuracy (points), per arm, tag and budget."""
    keep = runs[runs["arm"].isin(arms)].copy()
    cols = [c for c in ("test_macro_f1", "test_accuracy") if c in keep]
    g = keep.groupby(["arm", "tag", "budget"])[cols]
    out = (g.mean() * 100).round(2).join((g.std(ddof=1) * 100).round(2), rsuffix="_sd").join(g.size().rename("n_seeds"))
    return out.reset_index().sort_values(["budget", "arm", "tag"]).reset_index(drop=True)


def dose_summary(runs) -> pd.DataFrame:
    """Gain of A20 over A1 at 1% for each pretraining length (descriptive), the share of the 10,000-step gain, and the
    three checks of H11.5-H11.7 (rises with steps within 0.5; 100 steps >= +1.5 and <= 60% of the full gain; 3,000 steps
    >= 85% of it)."""
    m = runs[(runs["budget"] == 1) & (runs["arm"].isin(["A1", "A20"]))].groupby(["arm", "tag"])["test_macro_f1"].mean() * 100
    if ("A1", "") not in m.index or ("A20", "") not in m.index:
        return pd.DataFrame()
    base, full = m[("A1", "")], m[("A20", "")] - m[("A1", "")]
    rows = []
    for tag, steps in DOSE_TAGS:
        if ("A20", tag) in m.index:
            g = m[("A20", tag)] - base
            rows.append({"pretraining_steps": steps, "gain_over_A1": round(g, 2), "share_of_full_gain": round(g / full, 2)})
    rows.append({"pretraining_steps": 10000, "gain_over_A1": round(full, 2), "share_of_full_gain": 1.0})
    t = pd.DataFrame(rows).sort_values("pretraining_steps").reset_index(drop=True)
    gains = t["gain_over_A1"].to_numpy()
    t["H11.5_rises_within_0.5"] = bool(all(gains[i + 1] >= gains[i] - 0.5 for i in range(len(gains) - 1)))
    if (t["pretraining_steps"] == 100).any():
        r = t[t["pretraining_steps"] == 100].iloc[0]
        t["H11.6_100_steps_1.5_to_60pct"] = bool(r["gain_over_A1"] >= 1.5 and r["share_of_full_gain"] <= 0.60)
    if (t["pretraining_steps"] == 3000).any():
        t["H11.7_3000_steps_at_least_85pct"] = bool(t[t["pretraining_steps"] == 3000].iloc[0]["share_of_full_gain"] >= 0.85)
    return t


def high_summary(runs) -> pd.DataFrame:
    """The 70% row for A1, A5, A7, A20 with the H11.20 spread checks (macro-F1 within 0.7; accuracy 93.0-94.2)."""
    k = runs[(runs["budget"] == HIGH) & (runs["tag"] == "") & (runs["arm"].isin(["A1", "A5", "A7", "A20"]))]
    if k.empty:
        return pd.DataFrame()
    g = k.groupby("arm")[["test_macro_f1", "test_accuracy"]].mean() * 100
    g = g.round(2).reset_index()
    g["H11.20_spread_within_0.7"] = bool(g["test_macro_f1"].max() - g["test_macro_f1"].min() <= 0.7)
    g["H11.20_accuracy_in_93.0_94.2"] = g["test_accuracy"].between(93.0, 94.2)
    return g


def verdicts(tables) -> pd.DataFrame:
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
                     "p_holm": round(r["p_holm"], 4), "verdict": verdict11(r["mean_diff"], r["p_holm"], direction),
                     "claim": text})
    return pd.DataFrame(rows, columns=["hypothesis", "contrast", "budget", "mean_diff", "ci_low", "ci_high",
                                       "p_holm", "verdict", "claim"])


def run_all(results_dirs, out_dir, n=10000, rng_seed=0) -> dict:
    """Write every table to out_dir (CSV + Markdown) and return them. Contrasts without runs are skipped."""
    runs = an.load_runs(*results_dirs)
    tables = {
        "summary": summary(runs),
        "ours_vs_seed": a57.contrast_table(runs, "macro_f1", n, rng_seed, specs=OURS_SPECS),
        "seed_parts": a57.contrast_table(runs, "macro_f1", n, rng_seed, specs=SEED_PARTS_SPECS),
        "high_f1": a57.contrast_table(runs, "macro_f1", n, rng_seed, specs=HIGH_SPECS),
        "high_accuracy": a57.contrast_table(runs, "accuracy", n, rng_seed, specs=HIGH_SPECS),
        "gan_long": a57.contrast_table(runs, "macro_f1", n, rng_seed, specs=GANLONG_SPECS),
        "dose": a57.contrast_table(runs, "macro_f1", n, rng_seed, specs=DOSE_SPECS),
        "a37": a57.contrast_table(runs, "macro_f1", n, rng_seed, specs=A37_SPECS),
        "dose_summary": dose_summary(runs),
        "high_summary": high_summary(runs),
    }
    tables["verdicts"] = verdicts(tables)
    for name, t in tables.items():
        if t is not None and len(t):
            an.save_table(t, out_dir, name, default_decimals=3)
    return tables


# ---------------------------------------------------------------- whole-dataset scoring (notebook 23)
# The seed paper's Table II has support 28,619 for every model (the whole dataset). This part scores the same trained
# models on the labeled rows L, the unlabeled rows U, the dev and test rows, and on all rows together, to show how much
# of an "all headlines" accuracy is training data. Accuracy at the 0.5 threshold, as everywhere in this repo.
SEED_PAPER_ACCURACY = {"GAN-BERT + MTL": 93.88, "BERT": 92.04}   # Table II of the seed paper (all 28,619 headlines)


def run_scores(train: pd.DataFrame, dev: pd.DataFrame, test: pd.DataFrame, labeled_ids) -> dict:
    """Accuracy (points) of one run on L, U, all training rows, dev, test and all rows together.

    Each frame has columns id, label and p_sarcastic (probability of sarcastic); `train` has every training row and
    `labeled_ids` says which of them kept their label (L); the rest are U (their labels were never used in training)."""
    def correct(df):
        return (df["p_sarcastic"].to_numpy() >= 0.5).astype(int) == df["label"].to_numpy().astype(int)
    in_l = train["id"].isin(set(labeled_ids)).to_numpy()
    c_tr, c_dev, c_te = correct(train), correct(dev), correct(test)
    return {"acc_L": 100 * c_tr[in_l].mean(), "acc_U": 100 * c_tr[~in_l].mean() if (~in_l).any() else np.nan,
            "acc_train": 100 * c_tr.mean(), "acc_dev": 100 * c_dev.mean(), "acc_test": 100 * c_te.mean(),
            "acc_all": 100 * (c_tr.sum() + c_dev.sum() + c_te.sum()) / (len(c_tr) + len(c_dev) + len(c_te)),
            "n_all": int(len(c_tr) + len(c_dev) + len(c_te))}


def full_table(runs, data_dir) -> pd.DataFrame:
    """One row per run tagged "full": the scores of `run_scores` (needs preds_train/dev/test.csv in the run folder)."""
    from pathlib import Path
    from .data import budget_ids
    rows = []
    for _, r in runs[runs["tag"] == "full"].iterrows():
        d = Path(r["run_dir"])
        s = run_scores(pd.read_csv(d / "preds_train.csv"), pd.read_csv(d / "preds_dev.csv"), pd.read_csv(d / "preds_test.csv"),
                       budget_ids(data_dir, int(r["budget"]), int(r["seed"])))
        rows.append({"arm": r["arm"], "budget": int(r["budget"]), "seed": int(r["seed"]), **s})
    return pd.DataFrame(rows)


def full_summary(per_run: pd.DataFrame) -> pd.DataFrame:
    """Mean and s.d. over seeds, per arm and budget, plus the all-rows minus test gap."""
    if per_run.empty:
        return pd.DataFrame()
    cols = ["acc_L", "acc_U", "acc_train", "acc_dev", "acc_test", "acc_all"]
    g = per_run.groupby(["arm", "budget"])[cols]
    out = g.mean().round(2).add_suffix("_mean").join(g.std(ddof=1).round(2).add_suffix("_sd")).join(g.size().rename("n_seeds"))
    out["all_minus_test"] = (out["acc_all_mean"] - out["acc_test_mean"]).round(2)
    return out.reset_index()


def rerun_check(runs, tags=("full", "rerun11"), tol=1e-6) -> pd.DataFrame:
    """Every run with one of these tags must give the same test probabilities as the untagged original (same arm,
    budget and seed): training is deterministic and the extra scoring changes nothing."""
    from pathlib import Path
    rows = []
    for _, r in runs[runs["tag"].isin(tags)].iterrows():
        orig = runs[(runs["tag"] == "") & (runs["arm"] == r["arm"]) & (runs["budget"] == r["budget"]) & (runs["seed"] == r["seed"])]
        if orig.empty:
            rows.append({"run_id": r["run_id"], "status": "no original found", "max_abs_diff": np.nan})
            continue
        a = pd.read_csv(Path(r["run_dir"]) / "preds_test.csv").sort_values("id").reset_index(drop=True)
        b = pd.read_csv(Path(orig.iloc[0]["run_dir"]) / "preds_test.csv").sort_values("id").reset_index(drop=True)
        same_rows = a["id"].equals(b["id"]) and a["label"].equals(b["label"])
        diff = float(np.abs(a["p_sarcastic"] - b["p_sarcastic"]).max()) if same_rows else float("inf")
        rows.append({"run_id": r["run_id"], "status": "PASS" if diff <= tol else "FAIL", "max_abs_diff": diff})
    return pd.DataFrame(rows, columns=["run_id", "status", "max_abs_diff"])


def full_verdicts(summary: pd.DataFrame, check: pd.DataFrame) -> pd.DataFrame:
    """H11.24-H11.28 (vault/Pass 11 Predictions.md, addendum): descriptive checks, no bootstrap."""
    def row(arm, b):
        hit = summary[(summary["arm"] == arm) & (summary["budget"] == b)] if len(summary) else summary
        return hit.iloc[0] if len(hit) else None
    rows = []
    a5, a1_70, a1_100 = row("A5", HIGH), row("A1", HIGH), row("A1", 100)
    if a5 is not None:
        rows.append({"hypothesis": "H11.24", "check": "A5@70: accuracy on all rows minus test accuracy >= 2.0",
                     "value": a5["all_minus_test"], "verdict": "SUPPORTED" if a5["all_minus_test"] >= 2.0 else "not supported"})
        rows.append({"hypothesis": "H11.25", "check": "A5@70: accuracy on all rows >= 93.88 (the seed paper's figure)",
                     "value": a5["acc_all_mean"], "verdict": "SUPPORTED" if a5["acc_all_mean"] >= 93.88 else "not supported"})
    if a1_100 is not None:
        rows.append({"hypothesis": "H11.26", "check": "A1@100 (their BERT setting): accuracy on all rows (their BERT: 92.04), descriptive",
                     "value": a1_100["acc_all_mean"], "verdict": "descriptive"})
    for name, r in (("A5@70", a5), ("A1@70", a1_70)):
        if r is not None:
            ok = abs(r["acc_U_mean"] - r["acc_test_mean"]) <= 0.7 and r["acc_L_mean"] - r["acc_test_mean"] >= 2.0
            rows.append({"hypothesis": "H11.28", "check": f"{name}: U accuracy within 0.7 of test, L accuracy at least 2.0 above test",
                         "value": f"U {r['acc_U_mean']:.2f}, test {r['acc_test_mean']:.2f}, L {r['acc_L_mean']:.2f}",
                         "verdict": "SUPPORTED" if ok else "not supported"})
    if len(check):
        n_pass = int((check["status"] == "PASS").sum())
        rows.append({"hypothesis": "H11.27", "check": f"all {len(check)} re-runs reproduce the originals' test probabilities exactly",
                     "value": f"{n_pass} of {len(check)} PASS", "verdict": "SUPPORTED" if n_pass == len(check) else "not supported"})
    return pd.DataFrame(rows, columns=["hypothesis", "check", "value", "verdict"])


def run_full(results_dirs, data_dir, out_dir) -> dict:
    """Write the whole-dataset tables to out_dir and return them."""
    runs = an.load_runs(*results_dirs)
    per_run = full_table(runs, data_dir)
    summ = full_summary(per_run)
    check = rerun_check(runs)
    tables = {"full_per_run": per_run, "full_summary": summ, "rerun_check": check, "full_verdicts": full_verdicts(summ, check)}
    for name, t in tables.items():
        if t is not None and len(t):
            an.save_table(t, out_dir, name, default_decimals=2)
    return tables
