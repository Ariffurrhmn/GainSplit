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

SPLITS = ("dev", "test", "isarc_test")
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

def load_runs(*folders) -> pd.DataFrame:
    """One row per finished run, from one or more results folders.

    Columns: run_id, arm, budget, seed, tag, source, run_dir, then one column per
    score named `<split>_<metric>` (e.g. test_macro_f1), then the paths of the
    three prediction files (empty string if a file is missing).
    If the same (arm, budget, seed) appears in several folders, the last folder wins.
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
        return runs
    key = ["arm", "budget", "seed"]
    if runs.duplicated(key, keep="last").any():
        warnings.warn("Some (arm, budget, seed) runs appear more than once; keeping the last one loaded.")
        runs = runs.drop_duplicates(key, keep="last")
    return runs.sort_values(key).reset_index(drop=True)


# ---------------------------------------------------------------- summary tables

def summary_table(runs: pd.DataFrame, split: str = "test", metric: str = "macro_f1") -> pd.DataFrame:
    """Mean and standard deviation over seeds for every arm x budget (in points).

    Long format: arm, budget, mean, sd, n_seeds. sd uses n-1 (NaN with one seed).
    """
    col = f"{split}_{metric}"
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
    for (arm, budget), g in runs.groupby(["arm", "budget"]):
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


def _combo_bootstrap(runs, weights, budget, split, metric, n, rng_seed, chunk=1000):
    """Bootstrap of a weighted sum of arm differences, e.g. {'A3': 1, 'A1': -1}.

    Returns a dict, or None if the arms share no seed with prediction files.
    """
    arms = list(weights)
    col = f"preds_{split}"
    sub = runs[(runs["budget"] == budget) & (runs["arm"].isin(arms)) & (runs[col] != "")]
    seeds_per_arm = [set(sub[sub["arm"] == a]["seed"]) for a in arms]
    seeds = sorted(set.intersection(*seeds_per_arm))   # only seeds that every arm has
    if not seeds:
        return None

    # Per (arm, seed): one column block of four 0/1 columns, one per confusion cell
    # (true class x predicted class). Summing the selected rows gives the counts.
    ref_ids = ref_y = None
    blocks = []                        # list of (arm, one-hot matrix of shape n x 4)
    for a in arms:
        for s in seeds:
            path = sub[(sub["arm"] == a) & (sub["seed"] == s)][col].iloc[0]
            ids, y, pred = _read_preds(path)
            if ref_ids is None:
                ref_ids, ref_y = ids, y
            elif not (np.array_equal(ids, ref_ids) and np.array_equal(y, ref_y)):
                raise ValueError(f"{path}: test items or labels differ from the other runs")
            cell = y * 2 + pred        # 0 = tn, 1 = fp, 2 = fn, 3 = tp
            blocks.append(np.eye(4)[cell])
    n_items = len(ref_ids)
    onehot = np.concatenate(blocks, axis=1)           # n_items x (4 * arms * seeds)
    n_seeds = len(seeds)
    sign = np.array([weights[a] for a in arms], dtype=float)

    def diff_from_counts(counts):
        """counts: resamples x (4*arms*seeds) -> mean over seeds of the weighted metric sum."""
        c = counts.reshape(counts.shape[0], len(arms), n_seeds, 4)
        m = _metric_from_counts(c[..., 0], c[..., 1], c[..., 2], c[..., 3], metric)  # resamples x arms x seeds
        return (m * sign[None, :, None]).sum(axis=1).mean(axis=1)

    observed = float(diff_from_counts(onehot.sum(axis=0, keepdims=True))[0])

    rng = np.random.default_rng(rng_seed)
    diffs = []
    done = 0
    while done < n:
        k = min(chunk, n - done)
        # One resample = n_items draws with replacement. Every arm and seed uses the same draw.
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
        for budget in sorted(runs["budget"].unique()):
            res = _combo_bootstrap(runs, weights, budget, split, metric, n, rng_seed)
            if res is None:
                continue
            rows.append({"contrast": name, "budget": budget,
                         "mean_diff": res["mean_diff"] * 100, "ci_low": res["ci_low"] * 100,
                         "ci_high": res["ci_high"] * 100, "p": res["p"], "n_seeds": res["n_seeds"]})
    cols = ["contrast", "budget", "mean_diff", "ci_low", "ci_high", "p", "n_seeds"]
    return pd.DataFrame(rows, columns=cols)


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
