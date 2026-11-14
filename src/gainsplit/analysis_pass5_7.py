"""Passes 5-7 analysis (CPU only): paired bootstrap with Holm correction for the planned contrasts.

Reads the committed results folders (Passes 1-7) and reuses the paired bootstrap of analysis.py
and analysis_pass4.py. Every contrast below is fixed before the analysis is run (vault: Experiment
Plan, "Bootstrap plan for Passes 5-7"). A key is (arm, tag); tag "" is the main run.

Families (Holm is applied within each table, as in Pass 4):
- F1_SPECS: test macro-F1, the Pass 5, 6 and 7 contrasts.
- ACC_SPECS: test accuracy at 100% labels (Pass 7; accuracy is the number the seed paper reports).
- AUC_SPECS: iSarcasmEval ROC-AUC of a pair arm minus its baseline.
- LENGTH_SPECS: iSarcasmEval ROC-AUC of a pair arm minus word count alone, on the same items
  (is the transfer above the length shortcut?).
- GAP_SPECS: cue-mask shortcut gap of a pair arm minus its baseline (Passes 5-6; Pass 7 has no
  masked-test baseline at 100%).
Scores in the tables are in points (macro-F1 0.882 -> 88.2).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import analysis as an
from . import analysis_pass4 as a4

W = ("A10", "weighted")   # Pass 5: A10 with pair weight 0.1, unfiltered pairs

# (name, budget, {(arm, tag): weight})
F1_SPECS = [
    # Pass 5: pair weight 0.1 and cleaning
    *[("A10[weighted]-A1", b, {W: 1, ("A1", ""): -1}) for b in (1, 10)],
    *[("A10[weighted]-A10 (weight 0.1 vs 1.0)", b, {W: 1, ("A10", ""): -1}) for b in (1, 10)],
    *[("A16-A10[weighted] (cleaning)", b, {("A16", ""): 1, W: -1}) for b in (1, 10)],
    *[("A17-A16 (FGM on top)", b, {("A17", ""): 1, ("A16", ""): -1}) for b in (1, 10)],
    *[("A17-A7", b, {("A17", ""): 1, ("A7", ""): -1}) for b in (1, 10)],
    # Pass 6: two-stage training
    *[("A18-A1", b, {("A18", ""): 1, ("A1", ""): -1}) for b in (1, 10)],
    *[("A19-A7", b, {("A19", ""): 1, ("A7", ""): -1}) for b in (1, 10)],
    *[("A19-A18 (FGM in stage 2)", b, {("A19", ""): 1, ("A18", ""): -1}) for b in (1, 10)],
    # Pass 7: all labels
    ("A7-A1", 100, {("A7", ""): 1, ("A1", ""): -1}),
    ("A19-A1", 100, {("A19", ""): 1, ("A1", ""): -1}),
    ("A19-A7", 100, {("A19", ""): 1, ("A7", ""): -1}),
]
ACC_SPECS = [s for s in F1_SPECS if s[1] == 100]

# (a, b, budget): AUC(a) - AUC(b)
AUC_SPECS = [
    *[(W, ("A10", ""), b) for b in (1, 10)],          # H5.3: does down-weighting shrink the transfer?
    *[(W, ("A1", ""), b) for b in (1, 10)],
    *[(("A17", ""), ("A7", ""), b) for b in (1, 10)],
    *[(("A18", ""), ("A1", ""), b) for b in (1, 10)],
    *[(("A19", ""), ("A7", ""), b) for b in (1, 10)],
    (("A7", ""), ("A1", ""), 100),
    (("A19", ""), ("A7", ""), 100),
]

# (key, budget): AUC(key) - AUC(word count), every pair arm of Passes 4-7
LENGTH_SPECS = [
    *[((arm, ""), b) for arm in ("A10", "A11", "A12", "A13", "A14") for b in (1, 10)],
    (("A15", ""), 10),
    *[(W, b) for b in (1, 10)],
    *[((arm, ""), b) for arm in ("A16", "A17", "A18", "A19") for b in (1, 10)],
    (("A19", ""), 100),
]

# (key, budget, baseline arm): gap(key) - gap(baseline), baseline masked runs from a4.MASK_REF
GAP_SPECS = [(key, b, ref) for b in (1, 10) for key, ref in
             ((W, "A1"), (("A16", ""), "A1"), (("A17", ""), "A7"), (("A18", ""), "A1"), (("A19", ""), "A7"))]


def _name(key):
    arm, tag = key
    return f"{arm}[{tag}]" if tag else arm


def _with_holm(rows, cols=("contrast", "budget", "mean_diff", "ci_low", "ci_high", "p", "n_seeds")):
    t = pd.DataFrame(rows, columns=list(cols))
    t["p_holm"] = a4.holm(t["p"]) if len(t) else []
    return t


def contrast_table(runs, metric="macro_f1", n=10000, rng_seed=0, specs=None) -> pd.DataFrame:
    """Paired bootstrap over test headlines (mean over shared seeds), points, plus Holm over the table."""
    specs = (F1_SPECS if metric == "macro_f1" else ACC_SPECS) if specs is None else specs
    t = an.label_quality_table(runs, "test", metric, n=n, rng_seed=rng_seed, specs=specs)
    t["p_holm"] = a4.holm(t["p"]) if len(t) else []
    return t


def auc_table(runs, n=10000, rng_seed=0, specs=AUC_SPECS) -> pd.DataFrame:
    """iSarcasmEval AUC(a) - AUC(b), paired over items, points, plus Holm."""
    rows = []
    for a, b, budget in specs:
        res = a4.auc_contrast(runs, a, b, budget, n=n, rng_seed=rng_seed)
        if res is None:
            continue
        rows.append({"contrast": f"AUC {_name(a)} - {_name(b)}", "budget": budget,
                     "mean_diff": res["mean_diff"] * 100, "ci_low": res["ci_low"] * 100,
                     "ci_high": res["ci_high"] * 100, "p": res["p"], "n_seeds": res["n_seeds"]})
    return _with_holm(rows)


def _length_scores(isarc_test: pd.DataFrame, ids, y) -> np.ndarray:
    """Word count of each iSarcasmEval test item, in the order of `ids`; labels must match `y`."""
    ref = isarc_test.set_index("id").loc[ids]
    if not np.array_equal(ref["label"].to_numpy().astype(int), y):
        raise ValueError("isarc_test labels do not match the run's preds_isarc_test.csv")
    return ref["text"].str.split().str.len().to_numpy(float)


def auc_vs_length(runs, key, budget, isarc_test, n=10000, rng_seed=0, chunk=500):
    """Paired bootstrap of AUC(model) - AUC(word count) on the iSarcasmEval test, mean over seeds.

    Word count is a fixed score (same for every seed). Each resample of the items is applied to
    the model's predictions and to the word counts together. Returns a dict in plain units, or None.
    """
    paths = a4._group(runs, key, budget, "preds_isarc_test")
    if not paths:
        return None
    data, ref = [], None
    for s in sorted(paths):
        ids, y, p = an._read_probs(paths[s])
        if ref is None:
            ref = (ids, y)
            length = _length_scores(isarc_test, ids, y)
        elif not (np.array_equal(ids, ref[0]) and np.array_equal(y, ref[1])):
            raise ValueError(f"{paths[s]}: items or labels differ")
        data.append(p)
    y = ref[1]
    n_items = len(y)

    def stat(counts):
        model = np.mean([an._boot_auc(counts, y, p) for p in data], axis=0)
        return model - an._boot_auc(counts, y, length)

    observed = float(stat(np.ones((1, n_items)))[0])
    length_auc = float(an._boot_auc(np.ones((1, n_items)), y, length)[0])
    rng = np.random.default_rng(rng_seed)
    out, done = [], 0
    while done < n:
        k = min(chunk, n - done)
        idx = rng.integers(0, n_items, size=(k, n_items))
        counts = np.bincount((idx + n_items * np.arange(k)[:, None]).ravel(),
                             minlength=k * n_items).reshape(k, n_items).astype(float)
        out.append(stat(counts))
        done += k
    d = np.concatenate(out)
    d = d[np.isfinite(d)]
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0)))
    return {"model_auc": observed + length_auc, "length_auc": length_auc, "mean_diff": observed,
            "ci_low": float(lo), "ci_high": float(hi), "p": float(p), "n_seeds": len(data)}


def length_table(runs, isarc_test, n=10000, rng_seed=0, specs=LENGTH_SPECS) -> pd.DataFrame:
    """AUC of every pair arm minus word count alone (points), plus Holm. Positive = above the length shortcut."""
    rows = []
    for key, budget in specs:
        res = auc_vs_length(runs, key, budget, isarc_test, n, rng_seed)
        if res is None:
            continue
        rows.append({"contrast": f"AUC {_name(key)} - word count", "budget": budget,
                     "model_auc": res["model_auc"] * 100, "length_auc": res["length_auc"] * 100,
                     **{k: res[k] * 100 for k in ("mean_diff", "ci_low", "ci_high")},
                     "p": res["p"], "n_seeds": res["n_seeds"]})
    return _with_holm(rows, ("contrast", "budget", "model_auc", "length_auc", "mean_diff", "ci_low", "ci_high",
                             "p", "n_seeds"))


def gap_table(runs, n=10000, rng_seed=0, specs=GAP_SPECS) -> pd.DataFrame:
    """Shortcut gap (randmask - cuemask macro-F1) of a pair arm minus its baseline's, points, plus Holm.

    Negative = the pair arm leans less on outlet cue words."""
    rows = []
    for (arm, tag), budget, ref in specs:
        ref_arm, ref_tag = a4.MASK_REF[(ref, budget)]
        terms = [(arm, tag, "test_randmask", 1), (arm, tag, "test_cuemask", -1),
                 (ref_arm, ref_tag, "test_randmask", -1), (ref_arm, ref_tag, "test_cuemask", 1)]
        res = an._bootstrap_terms(runs, terms, budget, "macro_f1", n, rng_seed)
        if res is None:
            continue
        rows.append({"contrast": f"gap({_name((arm, tag))}) - gap({ref})", "budget": budget,
                     "mean_diff": res["mean_diff"] * 100, "ci_low": res["ci_low"] * 100,
                     "ci_high": res["ci_high"] * 100, "p": res["p"], "n_seeds": res["n_seeds"]})
    return _with_holm(rows)


def summary(runs) -> pd.DataFrame:
    """Mean and sd over seeds (points) for every arm in the specs, every tag kept apart."""
    arms = ("A1", "A7", "A10", "A11", "A12", "A13", "A14", "A15", "A16", "A17", "A18", "A19")
    keep = runs[runs["arm"].isin(arms) & runs["tag"].isin(["", "weighted"])]
    cols = [c for c in ("test_accuracy", "test_macro_f1", "test_cuemask_macro_f1", "isarc_test_roc_auc") if c in keep]
    g = keep.groupby(["budget", "arm", "tag"])[cols]
    sd = (g.std(ddof=1) * 100).round(2)
    sd.columns = [f"{c}_sd" for c in sd.columns]
    out = (g.mean() * 100).round(2).join(sd)
    out["n_seeds"] = g.size()
    return out.reset_index()


def check_stage1(runs, ref=("A15", 10)) -> pd.DataFrame:
    """Every A18/A19 run's stage-1 dev macro-F1 must equal Pass 4's A15 at 10% for the same seed."""
    base = runs[(runs["arm"] == ref[0]) & (runs["budget"] == ref[1]) & (runs["tag"] == "")]
    rows = []
    for _, r in runs[runs["arm"].isin(["A18", "A19"])].iterrows():
        pre = json.loads((Path(r["run_dir"]) / "result.json").read_text()).get("pretrain") or {}
        s1 = (pre.get("dev") or {}).get("macro_f1", np.nan)
        o = base[base["seed"] == r["seed"]]
        a15 = float(o["dev_macro_f1"].iloc[0]) if len(o) else np.nan
        rows.append({"run_id": r["run_id"], "stage1_steps": pre.get("steps"), "stage1_dev_f1": s1,
                     "a15_dev_f1": a15, "ok": bool(s1 == a15)})
    return pd.DataFrame(rows, columns=["run_id", "stage1_steps", "stage1_dev_f1", "a15_dev_f1", "ok"])


def run_all(results_dirs, out_dir="results/analysis_pass5_7", n=10000, rng_seed=0, isarc_test=None) -> dict:
    """Write every Passes 5-7 table to out_dir (CSV + Markdown) and return them.

    `isarc_test` (id, text, label) is needed for the length table; skipped if not given."""
    runs = an.load_runs(*results_dirs)
    tables = {
        "summary": summary(runs),
        "rerun_check": a4.check_reruns(runs, tags=("mask", "explain", "rerun")),
        "stage1_check": check_stage1(runs),
        "contrasts_f1": contrast_table(runs, "macro_f1", n, rng_seed),
        "contrasts_accuracy_100": contrast_table(runs, "accuracy", n, rng_seed),
        "auc_contrasts": auc_table(runs, n, rng_seed),
        "gap_contrasts": gap_table(runs, n, rng_seed),
    }
    if isarc_test is not None:
        tables["auc_vs_length"] = length_table(runs, isarc_test, n, rng_seed)
    for name, t in tables.items():
        if len(t):
            an.save_table(t, out_dir, name, default_decimals=3)
    return tables
