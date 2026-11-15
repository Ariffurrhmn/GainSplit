"""Pass 8 analysis (CPU): paired bootstrap with Holm, one family per sub-pass, fixed before running.

Same method as Passes 4-7 (vault: Experiment Plan, "Pass 8 plan"): 10,000 resamples of the test
headlines, one resample shared by every seed, statistic = mean over seeds 0-4; Holm within a table.
Each notebook calls run_all with its own sub-pass; a contrast whose runs are missing is skipped
(e.g. the JEPA-vs-MLM rows when notebook 15's results are not committed yet).

Sub-passes and contrasts:
  mlm   A20-A1, A21-A7, A21-A1 at 1% and 10%; A21-A7 at 100% (pass8_mlm_100)
  jepa  A22-A1, A23-A7 at 1% and 10%; A22-A20, A23-A21 (JEPA vs MLM) at 1% and 10%;
        A23-A7 and A23-A21 at 100%
  diff  A24-A4, A24-A1, A25-A1, A25-A26, A26-A1 at 1% and 10%
Plus, per sub-pass, iSarcasmEval AUC of each new arm minus its baseline at 10%, and the A1 re-run check.
Scores are in points (macro-F1 0.882 -> 88.2).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import analysis as an
from . import analysis_pass4 as a4
from . import analysis_pass5_7 as a57


def _c(a, b, budget, name=None):
    return (name or f"{a}-{b}", budget, {(a, ""): 1, (b, ""): -1})


F1_SPECS = {
    "mlm": [*[_c("A20", "A1", b) for b in (1, 10)], *[_c("A21", "A7", b) for b in (1, 10)],
            *[_c("A21", "A1", b) for b in (1, 10)], _c("A21", "A7", 100)],
    "jepa": [*[_c("A22", "A1", b) for b in (1, 10)], *[_c("A23", "A7", b) for b in (1, 10)],
             *[_c("A22", "A20", b, "A22-A20 (JEPA vs MLM)") for b in (1, 10)],
             *[_c("A23", "A21", b, "A23-A21 (JEPA vs MLM)") for b in (1, 10)],
             _c("A23", "A7", 100), _c("A23", "A21", 100, "A23-A21 (JEPA vs MLM)")],
    "diff": [*[_c("A24", "A4", b, "A24-A4 (diffusion vs GAN fakes)") for b in (1, 10)],
             *[_c("A24", "A1", b) for b in (1, 10)], *[_c("A25", "A1", b) for b in (1, 10)],
             *[_c("A25", "A26", b, "A25-A26 (denoising vs plain noise)") for b in (1, 10)],
             *[_c("A26", "A1", b) for b in (1, 10)]],
}
AUC_SPECS = {
    "mlm": [(("A20", ""), ("A1", ""), 10), (("A21", ""), ("A7", ""), 10)],
    "jepa": [(("A22", ""), ("A1", ""), 10), (("A23", ""), ("A7", ""), 10), (("A22", ""), ("A20", ""), 10)],
    "diff": [(("A24", ""), ("A4", ""), 10), (("A25", ""), ("A1", ""), 10), (("A26", ""), ("A1", ""), 10)],
}
NEW_ARMS = {"mlm": ("A20", "A21"), "jepa": ("A22", "A23"), "diff": ("A24", "A25", "A26")}
BASE_ARMS = ("A1", "A4", "A7")
SUBPASSES = tuple(F1_SPECS)


def accuracy_specs(sub):
    return [s for s in F1_SPECS[sub] if s[1] == 100]


def summary(runs, sub) -> pd.DataFrame:
    """Mean and sd over seeds (points) for the sub-pass's new arms and the baselines, untagged runs only."""
    arms = NEW_ARMS[sub] + BASE_ARMS + (("A20", "A21") if sub == "jepa" else ())
    keep = runs[runs["arm"].isin(arms) & (runs["tag"] == "")]
    cols = [c for c in ("test_accuracy", "test_macro_f1", "test_cuemask_macro_f1", "isarc_test_roc_auc") if c in keep]
    g = keep.groupby(["budget", "arm"])[cols]
    sd = (g.std(ddof=1) * 100).round(2)
    sd.columns = [f"{c}_sd" for c in sd.columns]
    out = (g.mean() * 100).round(2).join(sd)
    out["n_seeds"] = g.size()
    return out.reset_index()


def seed_wins(runs, sub) -> pd.DataFrame:
    """For each F1 contrast: on how many shared seeds the new arm is ahead (a plain, test-free reading aid)."""
    rows = []
    for name, budget, w in F1_SPECS[sub]:
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


def pretrain_table(info_paths) -> pd.DataFrame:
    """One row per TAPT encoder: held-out loss and frozen linear-probe dev F1, before and after."""
    rows = []
    for p in info_paths:
        p = Path(p)
        if not p.exists():
            continue
        i = json.loads(p.read_text())
        row = {"objective": i["objective"], "steps": i["config"]["steps"], "minutes": i["minutes"],
               "dev_loss_before": i["dev_before"]["loss"], "dev_loss_after": i["dev_after"]["loss"],
               "collapsed": i.get("collapsed")}
        for when in ("before", "after"):
            for b, v in (i.get(f"probe_{when}") or {}).items():
                row[f"probe_{b}_{when}"] = v["dev_macro_f1_mean"]
        if i["objective"] == "jepa":
            row["target_std_after"] = i["dev_after"].get("target_std")
        rows.append(row)
    return pd.DataFrame(rows)


def diffusion_health(runs) -> pd.DataFrame:
    """Diffusion arms: last training-curve point (denoising loss, p_fake, flip rate ...), mean over seeds."""
    rows = []
    for _, r in runs[runs["arm"].isin(["A24", "A25", "A26"]) & (runs["tag"] == "")].iterrows():
        log = json.loads((Path(r["run_dir"]) / "result.json").read_text())["train_log"]
        if log:
            rows.append({"arm": r["arm"], "budget": r["budget"], "seed": r["seed"],
                         **{k: v for k, v in log[-1].items() if k != "step"}})
    if not rows:
        return pd.DataFrame()
    t = pd.DataFrame(rows)
    return t.drop(columns="seed").groupby(["arm", "budget"]).mean().round(4).reset_index()


def run_all(results_dirs, sub, out_dir, n=10000, rng_seed=0, pretrain_infos=()) -> dict:
    """Write the sub-pass's tables to out_dir (CSV + Markdown) and return them."""
    if sub not in SUBPASSES:
        raise ValueError(f"sub must be one of {SUBPASSES}")
    runs = an.load_runs(*results_dirs)
    tables = {
        "summary": summary(runs, sub),
        "contrasts_f1": a57.contrast_table(runs, "macro_f1", n, rng_seed, specs=F1_SPECS[sub]),
        "seed_wins": seed_wins(runs, sub),
        "auc_contrasts": a57.auc_table(runs, n, rng_seed, specs=AUC_SPECS[sub]),
    }
    acc = accuracy_specs(sub)
    if acc:
        tables["contrasts_accuracy_100"] = a57.contrast_table(runs, "accuracy", n, rng_seed, specs=acc)
    if (runs["tag"] == "rerun").any():
        tables["rerun_check"] = a4.check_reruns(runs, tags=("rerun",))
    if pretrain_infos:
        tables["pretrain"] = pretrain_table(pretrain_infos)
    if sub == "diff":
        tables["diffusion_health"] = diffusion_health(runs)
    for name, t in tables.items():
        if len(t):
            an.save_table(t, out_dir, name, default_decimals=3)
    return tables
