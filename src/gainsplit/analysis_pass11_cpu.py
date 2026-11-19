"""Pass 11 CPU analyses on existing predictions (no training): the polarity-flip difference-in-differences (E0a) and the
seed-aware bootstrap (E0b). Predictions: vault/Pass 11 Predictions.md (H11.1-H11.4, H11.23), frozen before this code ran.

E0a  Do the helper heads hurt sarcastic headlines whose teacher sentiment is positive (surface sentiment points the wrong
     way) more than other sarcastic headlines? Statistic: [err(helper, positive) - err(base, positive)] minus
     [err(helper, other) - err(base, other)], error rates in points, averaged over seeds, over the sarcastic test
     headlines only; bootstrap over headlines inside each group (group sizes fixed), the same resample for both arms and
     every seed. Positive = sentiment-teacher label "positive" (107 of 1,355 sarcastic test headlines). Exploratory: n = 107.
E0b  Two-level bootstrap: each resample draws the training seeds with replacement (5 of 5) and then the test headlines
     (one resample shared by every seed and both arms). Reported next to the test-only interval (seeds held fixed, as in
     every earlier table), so the extra uncertainty from the five labeled subsets is visible. Macro-F1 at the 0.5 threshold.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from . import analysis as an
from . import analysis_pass4 as a4

THRESHOLD = 0.5


def _read(path):
    df = pd.read_csv(path).sort_values("id").reset_index(drop=True)
    return df["id"].to_numpy(), df["label"].to_numpy().astype(int), df["p_sarcastic"].to_numpy(float)


def _stack(runs, arm, budget, tag="", seeds=None):
    """Seeds, test ids, labels and the (seeds x items) matrix of predicted classes of one arm at one budget."""
    sub = runs[(runs["arm"] == arm) & (runs["budget"] == budget) & (runs["tag"] == tag)].sort_values("seed")
    if seeds is not None:
        sub = sub[sub["seed"].isin(seeds)]
    if sub.empty:
        return None
    ids, y, preds = None, None, []
    for _, r in sub.iterrows():
        i, yy, p = _read(r["preds_test"])
        if ids is None:
            ids, y = i, yy
        elif not (np.array_equal(ids, i) and np.array_equal(y, yy)):
            raise ValueError(f"{r['run_id']}: test ids or labels differ from the other seeds")
        preds.append((p >= THRESHOLD).astype(float))
    return sub["seed"].to_list(), ids, y, np.vstack(preds)


def _two_sided_p(boot):
    b = len(boot)
    return float(min(1.0, 2 * min((np.sum(boot >= 0) + 1) / (b + 1), (np.sum(boot <= 0) + 1) / (b + 1))))


# ---------------------------------------------------------------- E0a

def polarity_did(runs, sentiment: pd.DataFrame, base, helper, budget, n=10000, rng_seed=0):
    """DiD of error rates (points) of `helper` vs `base` arms, positive-sentiment sarcastic vs other sarcastic headlines.

    `sentiment` has columns id and sent_label (the teacher's label). Returns a dict, or None if an arm has no runs."""
    sb = _stack(runs, base[0], budget, base[1])
    sh = _stack(runs, helper[0], budget, helper[1])
    if sb is None or sh is None:
        return None
    seeds = sorted(set(sb[0]) & set(sh[0]))
    sb = _stack(runs, base[0], budget, base[1], seeds)
    sh = _stack(runs, helper[0], budget, helper[1], seeds)
    ids, y = sb[1], sb[2]
    lab = sentiment.set_index("id").loc[ids, "sent_label"].to_numpy()
    sarc = y == 1
    pos = sarc & (lab == "positive")
    rest = sarc & (lab != "positive")
    err_b = (sb[3] != y).astype(float)[:, :]
    err_h = (sh[3] != y).astype(float)[:, :]
    d = (err_h - err_b).mean(axis=0)                       # per item, mean over seeds
    point = 100 * (d[pos].mean() - d[rest].mean())
    rng = np.random.default_rng(rng_seed)
    ip, ir = np.where(pos)[0], np.where(rest)[0]
    boot = np.empty(n)
    for k in range(0, n, 1000):
        c = min(1000, n - k)
        bp = rng.integers(0, len(ip), (c, len(ip)))
        br = rng.integers(0, len(ir), (c, len(ir)))
        boot[k:k + c] = 100 * (d[ip][bp].mean(axis=1) - d[ir][br].mean(axis=1))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"mean_diff": float(point), "ci_low": float(lo), "ci_high": float(hi), "p": _two_sided_p(boot),
            "n_seeds": len(seeds), "n_pos": int(pos.sum()), "n_rest": int(rest.sum()),
            "err_pos_base": 100 * err_b[:, pos].mean(), "err_pos_helper": 100 * err_h[:, pos].mean(),
            "err_rest_base": 100 * err_b[:, rest].mean(), "err_rest_helper": 100 * err_h[:, rest].mean()}


POLARITY_SPECS = [(("A2", ""), ("A1", ""), b) for b in (1, 10)] + [(("A31", ""), ("A20", ""), b) for b in (1, 10)]


def polarity_table(runs, sentiment, n=10000, rng_seed=0) -> pd.DataFrame:
    rows = []
    for helper, base, b in POLARITY_SPECS:
        res = polarity_did(runs, sentiment, base, helper, b, n, rng_seed)
        if res is None:
            continue
        rows.append({"contrast": f"DiD {helper[0]} vs {base[0]}", "budget": b, **{k: round(v, 3) if isinstance(v, float) else v
                                                                               for k, v in res.items()}})
    t = pd.DataFrame(rows)
    if len(t):
        t["p_holm"] = a4.holm(t["p"])
    return t


# ---------------------------------------------------------------- E0b

def _macro_f1_diff(w, y, pa, pb):
    """(resamples x seeds) macro-F1 difference A - B, for item weights w (resamples x items)."""
    def macro(p):
        tp1 = w @ (p * y).T
        fp1 = w @ (p * (1 - y)).T
        fn1 = w @ ((1 - p) * y).T
        tp0 = w @ ((1 - p) * (1 - y)).T
        f1 = 2 * tp1 / np.maximum(2 * tp1 + fp1 + fn1, 1e-12)
        f0 = 2 * tp0 / np.maximum(2 * tp0 + fp1 + fn1, 1e-12)      # class 0: FP = fn1, FN = fp1
        return 100 * (f1 + f0) / 2
    return macro(pa) - macro(pb)


def cluster_bootstrap(runs, a, b, n=10000, rng_seed=0, chunk=500):
    """Macro-F1(a) - macro-F1(b) in points; a and b are (arm, budget, tag). Seeds present in both are paired by seed."""
    sa = _stack(runs, a[0], a[1], a[2])
    sb = _stack(runs, b[0], b[1], b[2])
    if sa is None or sb is None:
        return None
    seeds = sorted(set(sa[0]) & set(sb[0]))
    sa, sb = _stack(runs, a[0], a[1], a[2], seeds), _stack(runs, b[0], b[1], b[2], seeds)
    if not np.array_equal(sa[1], sb[1]):
        raise ValueError("the two arms have different test items")
    y, pa, pb = sa[2].astype(float), sa[3], sb[3]
    s, m = len(seeds), len(y)
    full = _macro_f1_diff(np.ones((1, m)), y, pa, pb)[0]           # per-seed difference on the full test set
    rng = np.random.default_rng(rng_seed)
    test_only, cluster = np.empty(n), np.empty(n)
    for k in range(0, n, chunk):
        c = min(chunk, n - k)
        w = rng.multinomial(m, np.full(m, 1.0 / m), size=c).astype(float)
        d = _macro_f1_diff(w, y, pa, pb)                            # c x s
        test_only[k:k + c] = d.mean(axis=1)
        sel = rng.integers(0, s, (c, s))
        cluster[k:k + c] = d[np.arange(c)[:, None], sel].mean(axis=1)
    out = {"mean_diff": float(full.mean()), "n_seeds": s, "seed_diffs": [round(float(v), 2) for v in full],
           "seeds_ahead": int((full > 0).sum())}
    for name, boot in (("test_only", test_only), ("cluster", cluster)):
        lo, hi = np.percentile(boot, [2.5, 97.5])
        out[f"{name}_low"], out[f"{name}_high"] = float(lo), float(hi)
        out[f"{name}_p"] = _two_sided_p(boot)
    out["width_ratio"] = (out["cluster_high"] - out["cluster_low"]) / max(out["test_only_high"] - out["test_only_low"], 1e-9)
    return out


# (name, a, b) with a, b = (arm, budget, tag)
CLUSTER_SPECS = [
    ("A20-A1", ("A20", 1, ""), ("A1", 1, "")), ("A20-A1", ("A20", 10, ""), ("A1", 10, "")),
    ("A2-A1", ("A2", 1, ""), ("A1", 1, "")), ("A6-A1", ("A6", 1, ""), ("A1", 1, "")), ("A7-A1", ("A7", 1, ""), ("A1", 1, "")),
    ("A30-A1", ("A30", 1, ""), ("A1", 1, "")),
    ("A22-A20", ("A22", 1, ""), ("A20", 1, "")), ("A22-A20", ("A22", 10, ""), ("A20", 10, "")),
    *[("A36-A5", ("A36", b, ""), ("A5", b, "")) for b in (1, 5, 10, 30)],
    ("A20@1%-A1@10%", ("A20", 1, ""), ("A1", 10, "")),
]


def cluster_table(runs, n=10000, rng_seed=0) -> pd.DataFrame:
    rows = []
    for name, a, b in CLUSTER_SPECS:
        res = cluster_bootstrap(runs, a, b, n, rng_seed)
        if res is None:
            continue
        rows.append({"contrast": name, "budget": a[1], "budget_b": b[1], "mean_diff": round(res["mean_diff"], 2),
                     "test_only_low": round(res["test_only_low"], 2), "test_only_high": round(res["test_only_high"], 2),
                     "cluster_low": round(res["cluster_low"], 2), "cluster_high": round(res["cluster_high"], 2),
                     "cluster_p": round(res["cluster_p"], 4), "width_ratio": round(res["width_ratio"], 2),
                     "seeds_ahead": f"{res['seeds_ahead']}/{res['n_seeds']}", "seed_diffs": res["seed_diffs"]})
    t = pd.DataFrame(rows)
    if len(t):
        t["cluster_p_holm"] = a4.holm(t["cluster_p"])
    return t


# ---------------------------------------------------------------- verdicts (H11.1-4, H11.23)

def verdicts(pol: pd.DataFrame, clu: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for b in (1, 10):
        for h, helper, base, direction in (("H11.1", "A2", "A1", "+"), ("H11.2", "A31", "A20", "0")):
            hit = pol[(pol["contrast"] == f"DiD {helper} vs {base}") & (pol["budget"] == b)] if len(pol) else pol
            if not len(hit):
                rows.append({"hypothesis": h, "check": f"DiD {helper} vs {base} at {b}%", "value": "no runs", "verdict": "no runs (skipped)"})
                continue
            r = hit.iloc[0]
            sig = r["p_holm"] < 0.05
            if direction == "+":
                v = "SUPPORTED" if sig and r["mean_diff"] > 0 else ("opposite, significant" if sig else "not supported (interval includes 0)")
            else:
                v = "not supported (significant)" if sig else "SUPPORTED (no significant DiD)"
            rows.append({"hypothesis": h, "check": f"DiD {helper} vs {base} at {b}% (positive-surface sarcastic minus other sarcastic, error points)",
                         "value": f"{r['mean_diff']:+.2f} [{r['ci_low']:+.2f}, {r['ci_high']:+.2f}], p_holm {r['p_holm']:.3f}", "verdict": v})
    def get(name, b):
        hit = clu[(clu["contrast"] == name) & (clu["budget"] == b)] if len(clu) else clu
        return hit.iloc[0] if len(hit) else None
    for b in (1, 10):
        r = get("A20-A1", b)
        if r is not None:
            rows.append({"hypothesis": "H11.3", "check": f"A20-A1 at {b}%: cluster interval above 0",
                         "value": f"{r['mean_diff']:+.2f} [{r['cluster_low']:+.2f}, {r['cluster_high']:+.2f}]",
                         "verdict": "SUPPORTED" if r["cluster_low"] > 0 else "not supported"})
    trio = [get(f"{a}-A1", 1) for a in ("A2", "A6", "A7")]
    if all(t is not None for t in trio):
        ratios = [t["width_ratio"] for t in trio]
        includes0 = [bool(t["cluster_low"] <= 0 <= t["cluster_high"]) for t in trio]
        rows.append({"hypothesis": "H11.4", "check": "A2, A6, A7 vs A1 at 1%: cluster interval at least 2x the test-only width, and at least one includes 0",
                     "value": f"width ratios {ratios}; cluster intervals include 0: {includes0}",
                     "verdict": "SUPPORTED" if min(ratios) >= 2 and any(includes0) else "not supported"})
    for b in (1, 10):
        r = get("A36-A5", b)
        if r is not None:
            rows.append({"hypothesis": "H11.23", "check": f"A36-A5 at {b}%: cluster interval excludes 0",
                         "value": f"{r['mean_diff']:+.2f} [{r['cluster_low']:+.2f}, {r['cluster_high']:+.2f}]",
                         "verdict": "SUPPORTED" if r["cluster_low"] > 0 or r["cluster_high"] < 0 else "not supported"})
    return pd.DataFrame(rows, columns=["hypothesis", "check", "value", "verdict"])


def run_all_cpu(results_dirs, data_dir, out_dir, n=10000, rng_seed=0) -> dict:
    """Write the tables to out_dir (CSV + Markdown) and return them."""
    runs = an.load_runs(*results_dirs)
    helper = pd.read_csv(Path(data_dir) / "helper_labels.csv", keep_default_na=False)[["id", "sent_label"]]
    pol = polarity_table(runs, helper, n, rng_seed)
    clu = cluster_table(runs, n, rng_seed)
    tables = {"polarity_did": pol, "cluster_bootstrap": clu, "verdicts": verdicts(pol, clu)}
    for name, t in tables.items():
        if t is not None and len(t):
            an.save_table(t, out_dir, name, default_decimals=2)
    return tables
