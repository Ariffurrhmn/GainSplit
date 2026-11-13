"""Pass 4 analysis (CPU only): contrasts, shortcut gaps, transfer and explanations for the pair arms.

Reads the same results folders as analysis.py and reuses its paired bootstrap. Everything is
fixed before the runs (vault: Experiment Plan, "Pass 4 plan"):
- main contrasts (test macro-F1, paired bootstrap over test headlines, mean over seeds 0-4),
  with a Holm column over these contrasts as a robustness check;
- cue-mask shortcut gap of a pair arm minus that of its baseline;
- iSarcasmEval ROC-AUC of a pair arm minus its baseline;
- share of Integrated Gradients attribution on outlet cue words, A10 vs A1 at 10%;
- re-run check: the A1 / A7 re-runs of Pass 4 must equal their Pass 1 / 2b originals.

A key is (arm, tag); tag "" is the main run. Baselines: A1, A6, A7 untagged from Passes 1-2;
masked-test baselines are the tag "mask" re-runs (10%: Pass 3; 1%: Pass 4).
"""
from pathlib import Path

import numpy as np
import pandas as pd

from . import analysis as an

PAIR_ARMS = ("A10", "A11", "A12", "A13", "A14", "A15")

# (name, budget, {(arm, tag): weight})
CONTRAST_SPECS = [
    *[(f"A10-A1", b, {("A10", ""): 1, ("A1", ""): -1}) for b in (1, 10)],
    *[(f"A11-A7", b, {("A11", ""): 1, ("A7", ""): -1}) for b in (1, 10)],
    *[(f"A11-A1", b, {("A11", ""): 1, ("A1", ""): -1}) for b in (1, 10)],
    ("A11-A6", 1, {("A11", ""): 1, ("A6", ""): -1}),
    *[(f"A10-A12 pairing", b, {("A10", ""): 1, ("A12", ""): -1}) for b in (1, 10)],
    *[(f"A13-A14 judge filter", b, {("A13", ""): 1, ("A14", ""): -1}) for b in (1, 10)],
    *[(f"A14-A10 subset and model mix", b, {("A14", ""): 1, ("A10", ""): -1}) for b in (1, 10)],
    ("A15-A1 pairs only vs 10% labels", 10, {("A15", ""): 1, ("A1", ""): -1}),
    ("A10 3x steps - A10", 1, {("A10", "steps3x"): 1, ("A10", ""): -1}),
    ("A10-A1 (100%)", 100, {("A10", ""): 1, ("A1", ""): -1}),   # only if pass4_full100 was run
]

# masked-test baseline per budget: (arm, tag) of the run that has the masked test sets
MASK_REF = {("A1", 10): ("A1", "mask"), ("A1", 1): ("A1", "mask"),
            ("A7", 10): ("A7", "mask"), ("A7", 1): ("A7", "mask")}
GAP_SPECS = [(arm, b, ref) for b in (1, 10) for arm, ref in (("A10", "A1"), ("A11", "A7"), ("A11", "A1"),
                                                             ("A12", "A1"), ("A13", "A1"), ("A14", "A1"))]
AUC_SPECS = [(("A10", ""), ("A1", ""), b) for b in (1, 10)] + [(("A11", ""), ("A7", ""), b) for b in (1, 10)] \
    + [(("A15", ""), ("A1", ""), 10), (("A12", ""), ("A1", ""), 10)]
EXPLAIN_SPECS = [(("A10", ""), ("A1", "explain"), 10)]


def holm(p) -> np.ndarray:
    """Holm-adjusted p-values (step-down), same order as the input."""
    p = np.asarray(p, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj


def contrast_table(runs, n=10000, rng_seed=0, specs=CONTRAST_SPECS) -> pd.DataFrame:
    """Paired bootstrap of every Pass 4 contrast that has runs (test macro-F1, points) + Holm."""
    t = an.label_quality_table(runs, "test", "macro_f1", n=n, rng_seed=rng_seed, specs=specs)
    t["p_holm"] = holm(t["p"]) if len(t) else []
    return t


def summary(runs, budgets=(1, 10, 100)) -> pd.DataFrame:
    """Mean and sd over seeds (points) for the pair arms and their baselines, every tag kept apart."""
    keep = runs[runs["arm"].isin(PAIR_ARMS + ("A1", "A6", "A7")) & runs["budget"].isin(budgets)]
    cols = [c for c in ("test_macro_f1", "test_cuemask_macro_f1", "isarc_test_roc_auc") if c in keep]
    g = keep.groupby(["budget", "arm", "tag"])[cols]
    sd = (g.std(ddof=1) * 100).round(2)
    sd.columns = [f"{c}_sd" for c in sd.columns]
    out = (g.mean() * 100).round(2).join(sd)
    out["n_seeds"] = g.size()
    return out.reset_index()


def gap_contrasts(runs, n=10000, rng_seed=0, specs=GAP_SPECS) -> pd.DataFrame:
    """Shortcut gap (randmask - cuemask macro-F1) of a pair arm minus that of its baseline.

    Negative = the pair arm loses less when outlet cue words are hidden, i.e. leans less on them."""
    rows = []
    for arm, budget, ref in specs:
        ref_arm, ref_tag = MASK_REF[(ref, budget)]
        terms = [(arm, "", "test_randmask", 1), (arm, "", "test_cuemask", -1),
                 (ref_arm, ref_tag, "test_randmask", -1), (ref_arm, ref_tag, "test_cuemask", 1)]
        res = an._bootstrap_terms(runs, terms, budget, "macro_f1", n, rng_seed)
        if res is None:
            continue
        rows.append({"contrast": f"gap({arm}) - gap({ref})", "budget": budget, "mean_diff": res["mean_diff"] * 100,
                     "ci_low": res["ci_low"] * 100, "ci_high": res["ci_high"] * 100, "p": res["p"],
                     "n_seeds": res["n_seeds"]})
    return pd.DataFrame(rows, columns=["contrast", "budget", "mean_diff", "ci_low", "ci_high", "p", "n_seeds"])


def _group(runs, key, budget, col):
    arm, tag = key
    g = runs[(runs["arm"] == arm) & (runs["tag"] == tag) & (runs["budget"] == budget) & (runs[col] != "")]
    return {int(s): p for s, p in zip(g["seed"], g[col])}


def auc_contrast(runs, a, b, budget, split="isarc_test", n=10000, rng_seed=0, chunk=500):
    """Paired bootstrap of ROC-AUC(a) - ROC-AUC(b) over the items of `split`, mean over shared seeds."""
    col = f"preds_{split}"
    pa, pb = _group(runs, a, budget, col), _group(runs, b, budget, col)
    seeds = sorted(set(pa) & set(pb))
    if not seeds:
        return None
    data, ref = [], None
    for s in seeds:
        for path in (pa[s], pb[s]):
            ids, y, p = an._read_probs(path)
            if ref is None:
                ref = (ids, y)
            elif not (np.array_equal(ids, ref[0]) and np.array_equal(y, ref[1])):
                raise ValueError(f"{path}: items or labels differ")
            data.append((y, p))
    n_items = len(ref[0])
    w = np.array([1.0, -1.0] * len(seeds))

    def stat(counts):
        vals = np.stack([an._boot_auc(counts, y, p) for y, p in data], axis=1)   # resamples x (2*seeds)
        return (vals * w).reshape(counts.shape[0], len(seeds), 2).sum(-1).mean(-1)

    observed = float(stat(np.ones((1, n_items)))[0])
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
    return {"mean_diff": observed, "ci_low": float(lo), "ci_high": float(hi), "p": float(p), "n_seeds": len(seeds)}


def auc_table(runs, n=10000, rng_seed=0, specs=AUC_SPECS) -> pd.DataFrame:
    rows = []
    for a, b, budget in specs:
        res = auc_contrast(runs, a, b, budget, n=n, rng_seed=rng_seed)
        if res is None:
            continue
        name = f"AUC {a[0]}{'[' + a[1] + ']' if a[1] else ''} - {b[0]}{'[' + b[1] + ']' if b[1] else ''}"
        rows.append({"contrast": name, "budget": budget, "mean_diff": res["mean_diff"] * 100,
                     "ci_low": res["ci_low"] * 100, "ci_high": res["ci_high"] * 100, "p": res["p"],
                     "n_seeds": res["n_seeds"]})
    return pd.DataFrame(rows, columns=["contrast", "budget", "mean_diff", "ci_low", "ci_high", "p", "n_seeds"])


# ---------------------------------------------------------------- explanations

def load_explain(runs) -> pd.DataFrame:
    """All explain.csv files: one row per (run, headline), with arm, budget, seed, tag."""
    parts = []
    for _, r in runs.iterrows():
        f = Path(r["run_dir"]) / "explain.csv"
        if f.exists():
            e = pd.read_csv(f, keep_default_na=False)
            parts.append(e.assign(arm=r["arm"], budget=r["budget"], seed=r["seed"], tag=r["tag"]))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def explain_table(expl: pd.DataFrame) -> pd.DataFrame:
    """Mean cue share (all sampled headlines, and those with a cue word), per model, averaged over seeds."""
    if expl.empty:
        return pd.DataFrame()
    expl = expl.assign(has_cue=expl["n_cue_words"] > 0)
    g = expl.groupby(["arm", "budget", "tag"])
    out = pd.DataFrame({"cue_share": g["cue_share"].mean(),
                        "cue_share_with_cue": expl[expl["has_cue"]].groupby(["arm", "budget", "tag"])["cue_share"].mean(),
                        "share_with_cue_word": g["has_cue"].mean(),
                        "median_rel_completeness_gap": g.apply(
                            lambda d: (d["completeness_gap"].abs() / d["score_diff"].abs().clip(lower=1e-6)).median()),
                        "n_seeds": g["seed"].nunique()})
    return out.reset_index()


def explain_contrast(expl: pd.DataFrame, a, b, budget, n=10000, rng_seed=0):
    """Bootstrap over the explained headlines of mean cue share (a) - (b), seeds averaged.

    Every seed of both models explains the same fixed headlines, so one resample of headline ids
    applies to all of them."""
    def mat(key):
        arm, tag = key
        d = expl[(expl["arm"] == arm) & (expl["tag"] == tag) & (expl["budget"] == budget)]
        return d.pivot(index="id", columns="seed", values="cue_share")
    ma, mb = mat(a), mat(b)
    seeds = sorted(set(ma.columns) & set(mb.columns))
    if not seeds:
        return None
    if not ma.index.equals(mb.index):
        raise ValueError("explained headlines differ between the two models")
    diff = (ma[seeds].to_numpy() - mb[seeds].to_numpy()).mean(axis=1)   # per headline, mean over seeds
    rng = np.random.default_rng(rng_seed)
    boots = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = min(1.0, 2 * min(np.mean(boots <= 0), np.mean(boots >= 0)))
    return {"mean_diff": float(diff.mean()), "ci_low": float(lo), "ci_high": float(hi), "p": float(p),
            "n_seeds": len(seeds), "n_items": len(diff)}


def explain_contrast_table(expl, n=10000, rng_seed=0, specs=EXPLAIN_SPECS) -> pd.DataFrame:
    rows = []
    for a, b, budget in specs:
        res = None if expl.empty else explain_contrast(expl, a, b, budget, n, rng_seed)
        if res is None:
            continue
        rows.append({"contrast": f"cue share {a[0]} - {b[0]}", "budget": budget, **res})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- re-run check

def check_reruns(runs, tags=("mask", "explain"), tol=1e-6) -> pd.DataFrame:
    """Every re-run (tag in `tags`) must give the same test scores and probabilities as the untagged original."""
    out = []
    for tag in tags:
        sub = runs[runs["tag"].isin(["", tag])].copy()
        sub["tag"] = sub["tag"].replace({tag: "mask"})   # check_mask_reruns looks for tag "mask"
        t = an.check_mask_reruns(sub, tol)
        out.append(t.assign(tag=tag))
    return pd.concat(out, ignore_index=True)


def run_all(results_dirs, out_dir="results/analysis_pass4", n=10000, rng_seed=0) -> dict:
    """Write every Pass 4 table to out_dir (CSV + Markdown) and return them."""
    runs = an.load_runs(*results_dirs)
    expl = load_explain(runs)
    tables = {
        "summary": summary(runs),
        "rerun_check": check_reruns(runs[runs["budget"].isin([1, 10])]),
        "contrasts": contrast_table(runs, n, rng_seed),
        "gap_contrasts": gap_contrasts(runs, n, rng_seed),
        "masked": an.masked_table(runs[runs["arm"].isin(PAIR_ARMS + ("A1", "A7"))], n, rng_seed),
        "transfer": an.transfer_table(runs[runs["arm"].isin(PAIR_ARMS + ("A1", "A7"))], n, rng_seed),
        "auc_contrasts": auc_table(runs, n, rng_seed),
        "explain": explain_table(expl),
        "explain_contrasts": explain_contrast_table(expl, n, rng_seed),
    }
    for name, t in tables.items():
        if len(t):
            an.save_table(t, out_dir, name, default_decimals=3)
    return tables
