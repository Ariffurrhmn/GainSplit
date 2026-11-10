"""Results analysis: tables, paired bootstrap tests and figures.

Uses only pandas, numpy and matplotlib (no torch), so it runs anywhere. Reads the folders written by the runner: each run has a folder
`<results>/runs/<run_id>/` with result.json, preds_test.csv, preds_dev.csv and
preds_isarc_test.csv.

Scores in the output tables are in percentage points (macro-F1 0.882 -> 88.2).
The bootstrap function itself works in plain metric units (0.882).
"""
import json
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

SPLITS = ("dev", "test", "isarc_test", "test_cuemask", "test_randmask")
METRICS = ("accuracy", "macro_f1", "f1_sarcastic", "roc_auc")

# A contrast is a dict {arm: weight}. "A3-A1" means mean(A3) - mean(A1).
CONTRASTS = {
    # Pass 1
    "A3-A1": {"A3": 1, "A1": -1},
    "A4-A1": {"A4": 1, "A1": -1},
    "A5-A1": {"A5": 1, "A1": -1},
    "A5-A3": {"A5": 1, "A3": -1},
    "A5-A4": {"A5": 1, "A4": -1},
    # Pass 2
    "A2-A1": {"A2": 1, "A1": -1},
    "A3-A2": {"A3": 1, "A2": -1},
    "A3-A8": {"A3": 1, "A8": -1},
    "A6-A1": {"A6": 1, "A1": -1},
    "A4-A6": {"A4": 1, "A6": -1},
    "A7-A1": {"A7": 1, "A1": -1},
    "A4-A7": {"A4": 1, "A7": -1},
    "A1-A0": {"A1": 1, "A0": -1},
    # Interaction: is GAN + helpers more (or less) than the sum of the two parts?
    "A5-A4-A3+A1": {"A5": 1, "A4": -1, "A3": -1, "A1": 1},
}


# ---------------------------------------------------------------- loading

def load_runs(*folders, test_only: bool = True) -> pd.DataFrame:
    """One row per finished run, from one or more results folders.

    Columns: run_id, arm, budget, seed, tag, source, run_dir, then one column per
    score named `<split>_<metric>` (e.g. test_macro_f1), then the paths of the
    three prediction files (empty string if a file is missing).
    If the same (arm, budget, seed, tag) appears in several folders, the last folder wins.
    Runs with a tag (Pass 3 variants) sit next to the untagged ones; the main tables and
    contrasts use only tag == "" (see `untagged`).
    Raises an error naming the folders if no run is found at all.
    With test_only (default), tuning runs that were scored on dev only are left out,
    so a tuning grid (several runs per arm, budget and seed) never mixes into the tables.
    """
    rows = []
    for folder in folders:
        for f in sorted(Path(folder).glob("runs/*/result.json")):
            r = json.loads(f.read_text())
            c = r["config"]
            row = {"run_id": r["run_id"], "arm": c["arm"], "budget": int(c["budget"]),
                   "seed": int(c["seed"]), "tag": c.get("tag", ""),
                   "source": str(folder), "run_dir": str(f.parent)}
            for split, s in r["scores"].items():
                for k in METRICS:
                    row[f"{split}_{k}"] = s.get(k, np.nan)
            for split in SPLITS:
                p = f.parent / f"preds_{split}.csv"
                row[f"preds_{split}"] = str(p) if p.exists() else ""
            rows.append(row)
    runs = pd.DataFrame(rows)
    if runs.empty:
        raise FileNotFoundError("No runs found (looked for runs/*/result.json in: "
                                + ", ".join(str(f) for f in folders) + "). Check the folder names.")
    if test_only and "test_macro_f1" in runs:
        # tuning runs (lr, w, FGM epsilon) score dev only; they are not results
        runs = runs[runs["test_macro_f1"].notna()]
    key = ["arm", "budget", "seed", "tag"]
    if runs.duplicated(key, keep="last").any():
        warnings.warn("Some (arm, budget, seed, tag) runs appear more than once; keeping the last one loaded.")
        runs = runs.drop_duplicates(key, keep="last")
    return runs.sort_values(key).reset_index(drop=True)


def untagged(runs: pd.DataFrame) -> pd.DataFrame:
    """The main-design runs only (tag == ""), as used by the Pass 1-2 tables and contrasts."""
    if "tag" not in runs:
        return runs
    return runs[runs["tag"] == ""]


# ---------------------------------------------------------------- summary tables

def summary_table(runs: pd.DataFrame, split: str = "test", metric: str = "macro_f1") -> pd.DataFrame:
    """Mean and standard deviation over seeds for every arm x budget (in points).

    Long format: arm, budget, mean, sd, n_seeds. sd uses n-1 (NaN with one seed).
    """
    col = f"{split}_{metric}"
    runs = untagged(runs)
    g = runs.groupby(["arm", "budget"])[col]
    out = pd.DataFrame({"mean": g.mean() * 100, "sd": g.std(ddof=1) * 100, "n_seeds": g.count()})
    return out.reset_index()


def summary_wide(summary: pd.DataFrame, decimals: int = 2) -> pd.DataFrame:
    """Reshape a summary_table into rows = arm, columns = budget, cells 'mean ± sd'."""
    def cell(r):
        sd = "n/a" if pd.isna(r["sd"]) else f"{r['sd']:.{decimals}f}"
        return f"{r['mean']:.{decimals}f} ± {sd}"
    s = summary.copy()
    s["cell"] = s.apply(cell, axis=1)
    wide = s.pivot(index="arm", columns="budget", values="cell")
    wide.columns = [f"{b}%" for b in wide.columns]
    return wide.reset_index()


def isarc_table(runs: pd.DataFrame, decimals: int = 2) -> pd.DataFrame:
    """iSarcasmEval test scores: macro-F1, sarcastic-class F1 and ROC-AUC, mean ± sd over seeds.

    One row per arm x budget, one column per metric.
    """
    rows = []
    for (arm, budget), g in untagged(runs).groupby(["arm", "budget"]):
        row = {"arm": arm, "budget": budget, "n_seeds": len(g)}
        for m in ("macro_f1", "f1_sarcastic", "roc_auc"):
            v = g[f"isarc_test_{m}"].dropna() * 100
            if v.empty:
                row[m] = "n/a"
            else:
                sd = v.std(ddof=1)
                sd_txt = "n/a" if pd.isna(sd) else f"{sd:.{decimals}f}"
                row[m] = f"{v.mean():.{decimals}f} ± {sd_txt}"
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- paired bootstrap

@lru_cache(maxsize=None)
def _read_preds(path: str):
    """Read a preds csv, sorted by id. Returns (ids, labels, predicted classes)."""
    df = pd.read_csv(path).sort_values("id")
    ids = df["id"].to_numpy()
    y = df["label"].to_numpy().astype(int)
    pred = (df["p_sarcastic"].to_numpy() >= 0.5).astype(int)  # same threshold as evaluate.scores
    return ids, y, pred


def _metric_from_counts(tn, fp, fn, tp, metric):
    """Metric computed from confusion counts. Works on arrays (one value per resample)."""
    def f1(tp_, fp_, fn_):
        denom = 2 * tp_ + fp_ + fn_
        return np.where(denom > 0, 2 * tp_ / np.where(denom > 0, denom, 1), 0.0)
    if metric == "macro_f1":
        return (f1(tp, fp, fn) + f1(tn, fn, fp)) / 2   # sarcastic class + not-sarcastic class
    if metric == "f1_sarcastic":
        return f1(tp, fp, fn)
    if metric == "accuracy":
        return (tn + tp) / (tn + fp + fn + tp)
    raise ValueError(f"metric '{metric}' is not supported by the bootstrap (use macro_f1, f1_sarcastic or accuracy)")


def _bootstrap_terms(runs, terms, budget, metric, n, rng_seed, chunk=1000):
    """Bootstrap of a weighted sum of scores, one term per (arm, tag, split, weight).

    The statistic is sum_t weight_t * metric(term t), averaged over the seeds that every
    term has. Every term and seed uses the same resample of the items, so the terms must
    share their item ids (checked). Returns a dict, or None if the terms share no seed.
    """
    sub = []                                   # one frame per term: its runs at this budget
    for arm, tag, split, _ in terms:
        col = f"preds_{split}"
        sub.append(runs[(runs["budget"] == budget) & (runs["arm"] == arm) & (runs["tag"] == tag)
                        & (runs[col] != "")])
    seeds_per_term = [set(t["seed"]) for t in sub]
    seeds = sorted(set.intersection(*seeds_per_term))   # only seeds that every term has
    if not seeds:
        return None

    # Per (term, seed): one column block of four 0/1 columns, one per confusion cell
    # (true class x predicted class). Summing the selected rows gives the counts.
    ref_ids = ref_y = None
    blocks = []                        # list of one-hot matrices of shape n x 4
    for (arm, tag, split, _), t in zip(terms, sub):
        col = f"preds_{split}"
        for s in seeds:
            path = t[t["seed"] == s][col].iloc[0]
            ids, y, pred = _read_preds(path)
            if ref_ids is None:
                ref_ids, ref_y = ids, y
            elif not (np.array_equal(ids, ref_ids) and np.array_equal(y, ref_y)):
                raise ValueError(f"{path}: test items or labels differ from the other runs")
            cell = y * 2 + pred        # 0 = tn, 1 = fp, 2 = fn, 3 = tp
            blocks.append(np.eye(4)[cell])
    n_items = len(ref_ids)
    onehot = np.concatenate(blocks, axis=1)           # n_items x (4 * terms * seeds)
    n_seeds = len(seeds)
    sign = np.array([t[3] for t in terms], dtype=float)
    n_terms = len(terms)

    def diff_from_counts(counts):
        """counts: resamples x (4*terms*seeds) -> mean over seeds of the weighted metric sum."""
        c = counts.reshape(counts.shape[0], n_terms, n_seeds, 4)
        m = _metric_from_counts(c[..., 0], c[..., 1], c[..., 2], c[..., 3], metric)  # resamples x terms x seeds
        return (m * sign[None, :, None]).sum(axis=1).mean(axis=1)

    observed = float(diff_from_counts(onehot.sum(axis=0, keepdims=True))[0])

    rng = np.random.default_rng(rng_seed)
    diffs = []
    done = 0
    while done < n:
        k = min(chunk, n - done)
        # One resample = n_items draws with replacement. Every term and seed uses the same draw.
        idx = rng.integers(0, n_items, size=(k, n_items))
        # How many times each item was drawn in each resample (k x n_items).
        draw_counts = np.bincount((idx + n_items * np.arange(k)[:, None]).ravel(),
                                  minlength=k * n_items).reshape(k, n_items)
        diffs.append(diff_from_counts(draw_counts @ onehot))
        done += k
    d = np.concatenate(diffs)

    lo, hi = np.percentile(d, [2.5, 97.5])
    p = min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0)))
    return {"mean_diff": observed, "boot_mean": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "p": float(p), "n_seeds": n_seeds}


def _combo_bootstrap(runs, weights, budget, split, metric, n, rng_seed, chunk=1000):
    """Bootstrap of a weighted sum of arm differences, e.g. {'A3': 1, 'A1': -1}.

    A key is an arm (untagged runs) or an (arm, tag) pair, e.g. {('A3', 'soft'): 1, 'A3': -1}.
    Returns a dict, or None if the arms share no seed with prediction files.
    """
    terms = []
    for key, w in weights.items():
        arm, tag = key if isinstance(key, tuple) else (key, "")
        terms.append((arm, tag, split, w))
    return _bootstrap_terms(runs, terms, budget, metric, n, rng_seed, chunk)


def paired_bootstrap(runs, arm_a, arm_b, budget, split="test", metric="macro_f1", n=10000, rng_seed=0):
    """Paired bootstrap of metric(arm_a) - metric(arm_b) over test items.

    Each iteration draws one resample of the test items and applies it to every seed.
    The difference is computed per seed, then averaged over seeds. Only seeds that both
    arms have are used.

    Returns a dict: mean_diff (on the full test set), boot_mean, ci_low, ci_high (95%
    percentile interval), p (two-sided) and n_seeds. Returns None if no shared seed.
    """
    return _combo_bootstrap(runs, {arm_a: 1, arm_b: -1}, budget, split, metric, n, rng_seed)


def contrast_table(runs, split="test", metric="macro_f1", n=10000, rng_seed=0, contrasts=None) -> pd.DataFrame:
    """Paired bootstrap for every contrast at every budget where all its arms exist.

    Columns: contrast, budget, mean_diff, ci_low, ci_high (in points), p, n_seeds.
    """
    contrasts = CONTRASTS if contrasts is None else contrasts
    rows = []
    for name, weights in contrasts.items():
        for budget in sorted(untagged(runs)["budget"].unique()):
            res = _combo_bootstrap(runs, weights, budget, split, metric, n, rng_seed)
            if res is None:
                continue
            rows.append({"contrast": name, "budget": budget,
                         "mean_diff": res["mean_diff"] * 100, "ci_low": res["ci_low"] * 100,
                         "ci_high": res["ci_high"] * 100, "p": res["p"], "n_seeds": res["n_seeds"]})
    cols = ["contrast", "budget", "mean_diff", "ci_low", "ci_high", "p", "n_seeds"]
    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------- Pass 3: masked test sets

def check_mask_reruns(runs: pd.DataFrame, tol: float = 1e-6) -> pd.DataFrame:
    """Compare every tag == "mask" re-run with the untagged run of the same arm, budget and seed.

    The mask re-runs repeat the original training (deterministic) and only add the masked test
    sets, so their test macro-F1 and test probabilities must equal the original's. Columns:
    arm, budget, seed, f1_diff, pred_diff (largest absolute difference of p_sarcastic),
    ok. A run with no untagged original, or a difference above tol, has ok == False.
    """
    rows = []
    base = untagged(runs)
    for _, r in runs[runs["tag"] == "mask"].iterrows():
        o = base[(base["arm"] == r["arm"]) & (base["budget"] == r["budget"]) & (base["seed"] == r["seed"])]
        row = {"arm": r["arm"], "budget": r["budget"], "seed": r["seed"],
               "f1_diff": np.nan, "pred_diff": np.nan, "ok": False}
        if len(o):
            o = o.iloc[0]
            row["f1_diff"] = abs(float(r["test_macro_f1"]) - float(o["test_macro_f1"]))
            if r["preds_test"] and o["preds_test"]:
                a = pd.read_csv(r["preds_test"]).sort_values("id")
                b = pd.read_csv(o["preds_test"]).sort_values("id")
                if len(a) == len(b) and np.array_equal(a["id"].to_numpy(), b["id"].to_numpy()):
                    row["pred_diff"] = float(np.abs(a["p_sarcastic"].to_numpy() - b["p_sarcastic"].to_numpy()).max())
            row["ok"] = bool(row["f1_diff"] <= tol and row["pred_diff"] <= tol)   # NaN compares False
        rows.append(row)
    out = pd.DataFrame(rows, columns=["arm", "budget", "seed", "f1_diff", "pred_diff", "ok"])
    bad = out[~out["ok"]]
    if len(bad):
        warnings.warn(f"{len(bad)} mask re-run(s) differ from (or have no) untagged original; see the 'ok' column.")
    return out


def masked_table(runs: pd.DataFrame, n=10000, rng_seed=0) -> pd.DataFrame:
    """Macro-F1 on test, cue-masked test and randomly masked test, per (arm, budget, tag).

    drop_cue = test - cuemask and drop_rand = test - randmask (points, mean over seeds);
    shortcut_gap = drop_cue - drop_rand: how much more the model loses when the outlet-indicative
    words are hidden than when the same number of random words are hidden. A positive gap
    means the model leans on outlet cues. The gap (and drop_cue) get a paired bootstrap interval
    over test items; every seed uses the same resample. Only runs that have all three splits.
    """
    need = ["test_macro_f1", "test_cuemask_macro_f1", "test_randmask_macro_f1"]
    if not all(c in runs for c in need):
        return pd.DataFrame()
    r = runs[runs[need].notna().all(axis=1)]
    rows = []
    for (arm, budget, tag), g in r.groupby(["arm", "budget", "tag"]):
        t, c, m = (g[k].mean() * 100 for k in need)
        row = {"arm": arm, "budget": budget, "tag": tag, "n_seeds": len(g),
               "test": t, "cuemask": c, "randmask": m,
               "drop_cue": t - c, "drop_rand": t - m, "shortcut_gap": (t - c) - (t - m)}
        # gap = (test - cue) - (test - rand) = rand - cue; the test terms cancel
        gap = _bootstrap_terms(runs, [(arm, tag, "test_randmask", 1), (arm, tag, "test_cuemask", -1)],
                               budget, "macro_f1", n, rng_seed)
        dcue = _bootstrap_terms(runs, [(arm, tag, "test", 1), (arm, tag, "test_cuemask", -1)],
                                budget, "macro_f1", n, rng_seed)
        for name, res in (("gap", gap), ("drop_cue", dcue)):
            for k in ("ci_low", "ci_high", "p"):
                row[f"{name}_{k}"] = np.nan if res is None else res[k] * (1 if k == "p" else 100)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- Pass 3: label-quality contrasts

LABEL_TAGS = ("soft", "flip25", "flip50", "vader", "teacher2", "sentonly", "emoonly")

# the variants are run on A3 at 10% and on A2 at 1%; each is compared with the untagged run
LABEL_REFERENCES = (("A3", 10), ("A2", 1))


def label_quality_specs(tags=LABEL_TAGS):
    """List of (name, budget, weights) for the label-quality contrasts (RQ2)."""
    specs = []
    for arm, budget in LABEL_REFERENCES:
        for tag in tags:
            specs.append((f"{arm}[{tag}]-{arm}", budget, {(arm, tag): 1, (arm, ""): -1}))
    specs.append(("A9[shuf]-A2", 1, {("A9", "shuf"): 1, ("A2", ""): -1}))
    specs.append(("A9[shuf]-A1", 1, {("A9", "shuf"): 1, ("A1", ""): -1}))
    return specs


def label_quality_table(runs, split="test", metric="macro_f1", n=10000, rng_seed=0, specs=None) -> pd.DataFrame:
    """Paired bootstrap for each label-quality contrast (variant minus reference); specs without runs are skipped.

    Columns: contrast, budget, mean_diff, ci_low, ci_high (points), p, n_seeds.
    """
    specs = label_quality_specs() if specs is None else specs
    rows = []
    for name, budget, weights in specs:
        res = _combo_bootstrap(runs, weights, budget, split, metric, n, rng_seed)
        if res is None:
            continue
        rows.append({"contrast": name, "budget": budget, "mean_diff": res["mean_diff"] * 100,
                     "ci_low": res["ci_low"] * 100, "ci_high": res["ci_high"] * 100,
                     "p": res["p"], "n_seeds": res["n_seeds"]})
    return pd.DataFrame(rows, columns=["contrast", "budget", "mean_diff", "ci_low", "ci_high", "p", "n_seeds"])


# ---------------------------------------------------------------- Pass 3: transfer intervals (ROC-AUC)

@lru_cache(maxsize=None)
def _read_probs(path: str):
    """Read a preds csv, sorted by id. Returns (ids, labels, p_sarcastic)."""
    df = pd.read_csv(path).sort_values("id")
    return df["id"].to_numpy(), df["label"].to_numpy().astype(int), df["p_sarcastic"].to_numpy(float)


def _boot_auc(counts, y, p):
    """ROC-AUC of every resample at once.

    counts: resamples x items, how often each item was drawn. The AUC is the share of
    (positive, negative) pairs with the positive scored higher (ties count half), with each
    item weighted by its draw count. Items are sorted by score once; a running sum of the
    negatives' weights then gives, for each positive, the weight of the negatives below it.
    """
    order = np.argsort(p, kind="stable")
    ps, ys, c = p[order], y[order], counts[:, order]
    n = len(ps)
    new = np.r_[True, ps[1:] != ps[:-1]]                       # first item of each tie group
    start = np.maximum.accumulate(np.where(new, np.arange(n), 0))
    last = np.r_[new[1:], True]                                # last item of each tie group
    end = np.minimum.accumulate(np.where(last, np.arange(n), n)[::-1])[::-1]
    cum = np.zeros((c.shape[0], n + 1))
    cum[:, 1:] = np.cumsum(c * (ys == 0), axis=1)              # negative weight up to each item
    pos = c * (ys == 1)
    num = (pos * 0.5 * (cum[:, start] + cum[:, end + 1])).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / (pos.sum(axis=1) * cum[:, -1])


def auc_interval(runs, split="isarc_test", n=10000, rng_seed=0, chunk=500):
    """Bootstrap 95% interval of the ROC-AUC (mean over seeds) for one group of runs.

    `runs` holds the seeds of one (arm, budget, tag). Each resample of the items is applied to
    every seed. Returns a dict: auc, ci_low, ci_high, n_seeds, n_items (plain units, 0.5 = chance).
    """
    col = f"preds_{split}"
    paths = [x for x in runs[col] if x]
    if not paths:
        return None
    ref = None
    data = []
    for path in paths:
        ids, y, p = _read_probs(path)
        if ref is None:
            ref = (ids, y)
        elif not (np.array_equal(ids, ref[0]) and np.array_equal(y, ref[1])):
            raise ValueError(f"{path}: items or labels differ from the other seeds")
        data.append((y, p))
    n_items = len(ref[0])
    observed = float(np.mean([_boot_auc(np.ones((1, n_items)), y, p)[0] for y, p in data]))
    rng = np.random.default_rng(rng_seed)
    out = []
    done = 0
    while done < n:
        k = min(chunk, n - done)
        idx = rng.integers(0, n_items, size=(k, n_items))
        counts = np.bincount((idx + n_items * np.arange(k)[:, None]).ravel(),
                             minlength=k * n_items).reshape(k, n_items).astype(float)
        out.append(np.mean([_boot_auc(counts, y, p) for y, p in data], axis=0))
        done += k
    d = np.concatenate(out)
    lo, hi = np.nanpercentile(d, [2.5, 97.5])
    return {"auc": observed, "ci_low": float(lo), "ci_high": float(hi), "n_seeds": len(paths), "n_items": n_items}


def transfer_table(runs, n=10000, rng_seed=0, split="isarc_test") -> pd.DataFrame:
    """iSarcasmEval ROC-AUC with a bootstrap interval for every (arm, budget, tag) (RQ3).

    Values in points (50 = chance); excludes_50 says whether the interval excludes chance.
    Runs trained on iSarcasmEval itself (tag isarctrain) are the in-domain reference rows.
    """
    col = f"preds_{split}"
    rows = []
    for (arm, budget, tag), g in runs[runs[col] != ""].groupby(["arm", "budget", "tag"]):
        res = auc_interval(g, split, n, rng_seed)
        rows.append({"arm": arm, "budget": budget, "tag": tag, "n_seeds": res["n_seeds"],
                     "auc": res["auc"] * 100, "ci_low": res["ci_low"] * 100, "ci_high": res["ci_high"] * 100,
                     "excludes_50": bool(res["ci_low"] > 0.5 or res["ci_high"] < 0.5),
                     "in_domain_reference": tag == "isarctrain"})
    cols = ["arm", "budget", "tag", "n_seeds", "auc", "ci_low", "ci_high", "excludes_50", "in_domain_reference"]
    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------- Pass 3: helper labels vs sarcasm

def _entropy(counts):
    q = counts[counts > 0] / counts.sum()
    return float(-(q * np.log(q)).sum())


def association(x, y):
    """Association of two categorical variables: (normalised mutual information, Cramer's V).

    NMI = I(X;Y) / mean(H(X), H(Y)): 0 for independent variables, 1 for identical partitions.
    Cramer's V = sqrt(chi2 / (n * (min(rows, cols) - 1))), from the uncorrected chi-square.
    """
    tab = pd.crosstab(pd.Series(np.asarray(x)), pd.Series(np.asarray(y))).to_numpy().astype(float)
    n = tab.sum()
    hx, hy = _entropy(tab.sum(axis=1)), _entropy(tab.sum(axis=0))
    pxy = tab / n
    px, py = pxy.sum(axis=1, keepdims=True), pxy.sum(axis=0, keepdims=True)
    nz = pxy > 0
    mi = float((pxy[nz] * np.log(pxy[nz] / (px @ py)[nz])).sum())
    nmi = 0.0 if hx + hy == 0 else max(0.0, 2 * mi / (hx + hy))
    expected = n * px @ py
    chi2 = float(((tab - expected) ** 2 / np.where(expected > 0, expected, 1)).sum())
    k = min(tab.shape) - 1
    v = 0.0 if k <= 0 else float(np.sqrt(chi2 / (n * k)))
    return nmi, v


def _cv_auc(X, y, seed=0, folds=5):
    """5-fold stratified cross-validated ROC-AUC of a logistic regression (out-of-fold probabilities)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    p = cross_val_predict(LogisticRegression(max_iter=1000), X, y, cv=cv, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p))


def _train_part(df):
    """Rows of a helper csv that belong to the training set (if it has a split column)."""
    if "split" in df and (df["split"] == "train").any():
        return df[df["split"] == "train"]
    return df


def helper_shortcut_table(helper: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    """Does a teacher's output predict sarcasm by itself, on the training set? (CPU only.)

    helper: helper_labels.csv (id, sent_label, emo_label, sent_p_*, emo_p_*); train: headlines_train.csv
    (id, text, label). Rows: association (NMI, Cramer's V) of sent_label and of emo_label with the
    sarcasm label, and a probe: logistic regression on the teacher probabilities (all sent_p_* and
    emo_p_*), 5-fold stratified cross-validated ROC-AUC (random_state 0). A probe AUC near 0.5 means
    the helper labels carry no sarcasm signal of their own.
    """
    d = _train_part(helper).merge(train[["id", "label"]], on="id", how="inner")
    rows = []
    for c in ("sent_label", "emo_label"):
        nmi, v = association(d[c], d["label"])
        rows.append({"check": f"{c} vs sarcasm", "n": len(d), "nmi": nmi, "cramers_v": v, "auc": np.nan})
    pcols = [c for c in d.columns if c.startswith(("sent_p_", "emo_p_"))]
    if pcols:
        rows.append({"check": f"probe: logistic regression on {len(pcols)} teacher probabilities, 5-fold CV",
                     "n": len(d), "nmi": np.nan, "cramers_v": np.nan,
                     "auc": _cv_auc(d[pcols].to_numpy(float), d["label"].to_numpy().astype(int))})
    return pd.DataFrame(rows)


def load_gold(processed_dir, gold_path, train=None):
    """Gold audit merged with every sentiment source found in processed_dir.

    Returns (gold, notes): one row per audited headline with gold_sentiment, unsure, the sarcasm label
    (if `train` is given) and one column per available source (sent_label, vader_label, teacher2_label).
    Missing files are skipped and named in `notes`; gold is None if the audit file itself is missing.
    """
    processed_dir, notes = Path(processed_dir), []
    gold_path = Path(gold_path)
    if not gold_path.exists():
        return None, [f"gold audit not found: {gold_path}"]
    g = pd.read_csv(gold_path)
    if train is not None:
        g = g.merge(train[["id", "label"]], on="id", how="left")
    for fname, cols in (("helper_labels.csv", ["sent_label"]),
                        ("helper_sent_alt.csv", ["vader_label", "teacher2_label"])):
        f = processed_dir / fname
        if not f.exists():
            notes.append(f"{fname} not found, skipping {', '.join(cols)}")
            continue
        h = _train_part(pd.read_csv(f))
        g = g.merge(h[["id"] + [c for c in cols if c in h]], on="id", how="left")
    return g, notes


def _as_flag(s):
    """Unsure column to 0/1 (accepts 0/1, True/False, 'yes'/'no')."""
    if s.dtype == object:
        names = {"1": 1, "true": 1, "yes": 1, "y": 1, "0": 0, "false": 0, "no": 0, "n": 0}
        s = s.astype(str).str.strip().str.lower().map(names)
    return s.fillna(0).astype(int)


def _norm_labels(s):
    return s.astype(str).str.strip().str.lower()


def gold_table(gold: pd.DataFrame, sources=("sent_label", "vader_label", "teacher2_label")) -> pd.DataFrame:
    """Accuracy, macro-F1 and Cohen's kappa of each sentiment source against gold_sentiment (RQ2 label quality).

    Reported on all audited headlines and on the confident ones only (unsure == 0). Sources
    that are not in the frame are skipped (with a printed note).
    """
    from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
    flag = _as_flag(gold["unsure"]) if "unsure" in gold else pd.Series(0, index=gold.index)
    rows = []
    for src in sources:
        if src not in gold or gold[src].isna().all():
            print(f"gold_table: no column '{src}', skipped")
            continue
        for name, mask in (("all", np.ones(len(gold), bool)), ("unsure == 0", (flag == 0).to_numpy())):
            sub = gold[mask & gold[src].notna().to_numpy()]
            a, b = _norm_labels(sub["gold_sentiment"]), _norm_labels(sub[src])
            rows.append({"source": src, "subset": name, "n": len(sub),
                         "accuracy": accuracy_score(a, b) * 100,
                         "macro_f1": f1_score(a, b, average="macro", labels=sorted(set(a)), zero_division=0) * 100,
                         "kappa": cohen_kappa_score(a, b)})
    return pd.DataFrame(rows, columns=["source", "subset", "n", "accuracy", "macro_f1", "kappa"])


def gold_shortcut_table(gold: pd.DataFrame, n_perm: int = 500, seed: int = 0) -> pd.DataFrame:
    """Does the owner's sentiment label predict sarcasm better than the teacher's, on the audited headlines?

    On the same ids: association (NMI, Cramer's V) of gold_sentiment and of the teacher's sent_label
    with the sarcasm label, and the 5-fold CV ROC-AUC of a logistic regression on the one-hot label.
    The last row is the AUC difference (gold - teacher) with a permutation p-value: in each
    permutation, every headline randomly swaps which of its two sentiment labels counts as gold.
    Needs the `label` column (pass train to load_gold).
    """
    g = gold.dropna(subset=["label", "sent_label"]).reset_index(drop=True)
    y = g["label"].to_numpy().astype(int)
    a, b = _norm_labels(g["gold_sentiment"]).to_numpy(), _norm_labels(g["sent_label"]).to_numpy()
    cats = np.array(sorted(set(a) | set(b)))
    onehot = lambda v: (v[:, None] == cats[None, :]).astype(float)
    auc_a, auc_b = _cv_auc(onehot(a), y), _cv_auc(onehot(b), y)
    rows = []
    for name, v, auc in (("gold_sentiment (owner)", a, auc_a), ("sent_label (teacher)", b, auc_b)):
        nmi, cv = association(v, y)
        rows.append({"check": f"{name} vs sarcasm", "n": len(g), "nmi": nmi, "cramers_v": cv, "auc": auc, "p_perm": np.nan})
    rng = np.random.default_rng(seed)
    obs = auc_a - auc_b
    diffs = []
    for _ in range(n_perm):
        sw = rng.random(len(g)) < 0.5
        pa, pb = np.where(sw, b, a), np.where(sw, a, b)
        diffs.append(_cv_auc(onehot(pa), y) - _cv_auc(onehot(pb), y))
    p = (1 + np.sum(np.abs(diffs) >= abs(obs) - 1e-12)) / (1 + n_perm)
    rows.append({"check": "AUC gold - teacher", "n": len(g), "nmi": np.nan, "cramers_v": np.nan, "auc": obs, "p_perm": float(p)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- writing tables

def _to_markdown(df: pd.DataFrame, decimals: dict, default_decimals: int) -> str:
    def fmt(col, v):
        if isinstance(v, (float, np.floating)):
            return "n/a" if np.isnan(v) else f"{v:.{decimals.get(col, default_decimals)}f}"
        return str(v)
    lines = ["| " + " | ".join(map(str, df.columns)) + " |",
             "|" + "|".join("---" for _ in df.columns) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(c, row[c]) for c in df.columns) + " |")
    return "\n".join(lines) + "\n"


def save_table(df: pd.DataFrame, out_dir, name: str, decimals=None, default_decimals: int = 2):
    """Write `<name>.csv` (full precision) and `<name>.md` (rounded) into out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{name}.csv", index=False)
    decimals = {"p": 4, **(decimals or {})}
    (out_dir / f"{name}.md").write_text(_to_markdown(df, decimals, default_decimals), encoding="utf-8")


# ---------------------------------------------------------------- figures

def _save_fig(fig, out_dir, name):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.png", dpi=200, bbox_inches="tight")


def plot_f1_vs_budget(runs, out_dir, split="test", metric="macro_f1", name="fig_f1_vs_budget"):
    """Mean score against label budget (log x axis); one line per arm, error bars = sd over seeds."""
    import matplotlib.pyplot as plt
    summ = summary_table(runs, split, metric)
    fig, ax = plt.subplots(figsize=(6, 4))
    for arm, g in summ.groupby("arm"):
        g = g.sort_values("budget")
        ax.errorbar(g["budget"], g["mean"], yerr=g["sd"].fillna(0), marker="o", capsize=3, label=arm)
    ax.set_xscale("log")
    budgets = sorted(summ["budget"].unique())
    ax.set_xticks(budgets)
    ax.set_xticklabels([f"{b}%" for b in budgets])
    ax.set_xlabel("Labeled data (% of training set)")
    ax.set_ylabel(f"{split} {metric} (points)")
    ax.grid(alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    _save_fig(fig, out_dir, name)
    return fig


def _read_train_log(run_dir):
    return json.loads((Path(run_dir) / "result.json").read_text()).get("train_log", [])


def plot_gan_curves(runs, out_dir, arms=("A4", "A5"), name="fig_gan_curves"):
    """Discriminator's probability of 'fake' for real and for generated texts, against step.

    One panel per arm x budget. Lines are the mean over seeds; the band is the min-max.
    """
    import matplotlib.pyplot as plt
    runs = untagged(runs)
    budgets = sorted(runs[runs["arm"].isin(arms)]["budget"].unique())
    if not budgets:
        return None
    fig, axes = plt.subplots(len(arms), len(budgets), figsize=(3.2 * len(budgets), 2.8 * len(arms)),
                             sharey=True, squeeze=False)
    for i, arm in enumerate(arms):
        for j, budget in enumerate(budgets):
            ax = axes[i][j]
            sel = runs[(runs["arm"] == arm) & (runs["budget"] == budget)]
            frames = []
            for _, r in sel.iterrows():
                log = pd.DataFrame(_read_train_log(r["run_dir"]))
                if "p_fake_real" in log:
                    frames.append(log[["step", "p_fake_real", "p_fake_fake"]].assign(seed=r["seed"]))
            if frames:
                allv = pd.concat(frames)
                for col, label in (("p_fake_real", "real texts"), ("p_fake_fake", "generated texts")):
                    g = allv.groupby("step")[col]
                    ax.plot(g.mean().index, g.mean().values, label=label)
                    ax.fill_between(g.min().index, g.min().values, g.max().values, alpha=0.2)
            ax.set_title(f"{arm}, {budget}%", fontsize=9)
            ax.grid(alpha=0.3)
            if i == len(arms) - 1:
                ax.set_xlabel("step")
            if j == 0:
                ax.set_ylabel("P(fake)")
    axes[0][0].legend(fontsize=7)
    fig.tight_layout()
    _save_fig(fig, out_dir, name)
    return fig


# ---------------------------------------------------------------- everything at once

def run_all(results_dirs, out_dir="results/analysis", n=10000, rng_seed=0):
    """Load the runs, write every table and figure, return a dict of the tables."""
    runs = load_runs(*results_dirs)
    out = {"runs": runs}
    for split in ("test", "dev"):
        s = summary_table(runs, split, "macro_f1")
        out[f"summary_{split}"] = s
        save_table(s, out_dir, f"summary_{split}_macro_f1_long")
        save_table(summary_wide(s), out_dir, f"summary_{split}_macro_f1")
    out["isarc"] = isarc_table(runs)
    save_table(out["isarc"], out_dir, "isarc_test")
    out["contrasts"] = contrast_table(runs, "test", "macro_f1", n=n, rng_seed=rng_seed)
    save_table(out["contrasts"], out_dir, "contrasts_test_macro_f1")
    plot_f1_vs_budget(runs, out_dir)
    plot_gan_curves(runs, out_dir)
    return out
