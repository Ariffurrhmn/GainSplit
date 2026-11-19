"""Pass 11 E1 analysis (CPU): does MLM pretraining's low-label gain replicate when the label is not the outlet?

iSarcasmEval (author-labeled), label counts N = 228 / 780 / 3,120, seeds 0-9, scored on the iSarcasmEval test set
(1,400 tweets, 200 sarcastic). Primary metric ROC-AUC; secondary macro-F1 at the 0.5 threshold. Paired bootstrap over test
items (10,000 resamples, one resample for every seed, statistic = mean over seeds), Holm within each table.
Predictions: vault/Pass 11 Predictions.md, H11.8-H11.12 (frozen before this code ran).

Arms (tags): A1 [isc] BERT; A20 [isc28] MLM on the iSarcasmEval training tweets minus dev, 1,400 steps (about 28.7 passes);
A20 [isc100] the same with 4,900 steps; A20 [hs] the Headlines MLM encoder (cross-domain control). Size-matched control on
Headlines: A20 [hsub], MLM on a random 3,120-headline subset, compared with A1 and A20 at Headlines 1% and 10%.

Tables (each its own Holm family): isarc_auc, isarc_f1, size_control; reading aids: summary, replication.
"""
import numpy as np
import pandas as pd

from . import analysis as an
from . import analysis_pass4 as a4
from . import analysis_pass5_7 as a57

NS = (228, 780, 3120)


def _k(arm, tag=""):
    return (arm, tag)


A1, ISC28, ISC100, HS = _k("A1", "isc"), _k("A20", "isc28"), _k("A20", "isc100"), _k("A20", "hs")

AUC_SPECS = ([(ISC28, A1, n) for n in NS] + [(HS, A1, n) for n in NS] + [(ISC28, HS, n) for n in NS]
             + [(ISC100, ISC28, n) for n in (228, 780)])


def _f1_specs():
    nm = a57._name
    return ([(f"{nm(ISC28)}-{nm(A1)} (macro-F1, iSarcasmEval test)", n, {ISC28: 1, A1: -1}) for n in NS]
            + [(f"{nm(HS)}-{nm(A1)} (macro-F1, iSarcasmEval test)", n, {HS: 1, A1: -1}) for n in NS])


F1_SPECS = _f1_specs()
# Headlines, size-matched control: MLM on a random 3,120-headline subset vs BERT and vs the full-pool MLM encoder
CONTROL_SPECS = [s for b in (1, 10) for s in (
    ("A20[hsub]-A1 (3,120-headline pool vs BERT)", b, {_k("A20", "hsub"): 1, _k("A1"): -1}),
    ("A20[hsub]-A20 (3,120-headline pool vs full pool)", b, {_k("A20", "hsub"): 1, _k("A20"): -1}))]


def summary(runs) -> pd.DataFrame:
    """Mean and s.d. over seeds of iSarcasmEval test AUC, macro-F1 and F1-sarcastic (points), per arm, tag and budget."""
    keep = runs[((runs["tag"].isin(["isc", "isc28", "isc100", "hs"])) & runs["budget"].isin(NS) & (runs["isarc_test_roc_auc"].notna()))]
    cols = ["isarc_test_roc_auc", "isarc_test_macro_f1", "isarc_test_f1_sarcastic"]
    g = keep.groupby(["arm", "tag", "budget"])[cols]
    out = (g.mean() * 100).round(2).join((g.std(ddof=1) * 100).round(2), rsuffix="_sd").join(g.size().rename("n_seeds"))
    return out.reset_index().sort_values(["budget", "tag"]).reset_index(drop=True)


def _row(t, name, budget):
    hit = t[(t["contrast"] == name) & (t["budget"] == budget)] if len(t) else t
    return hit.iloc[0] if len(hit) else None


def verdicts(tables) -> pd.DataFrame:
    """H11.8-H11.12 plus the pre-written reading rule ("replicates" / "collapses")."""
    auc, f1, ctl = tables["isarc_auc"], tables["isarc_f1"], tables["size_control"]
    nm = a57._name
    rows = []

    def add(h, check, r, ok, extra=""):
        if r is None:
            rows.append({"hypothesis": h, "check": check, "value": "no runs", "verdict": "no runs (skipped)"})
        else:
            rows.append({"hypothesis": h, "check": check,
                         "value": f"{r['mean_diff']:+.2f} [{r['ci_low']:+.2f}, {r['ci_high']:+.2f}], p_holm {r['p_holm']:.3f}{extra}",
                         "verdict": "SUPPORTED" if ok(r) else "not supported"})
    gain = f"AUC {nm(ISC28)} - {nm(A1)}"
    for n in (228, 780):
        add("H11.8", f"AUC gain of A20[isc28] over A1 at N={n} is between +1 and +5 and significant", _row(auc, gain, n),
            lambda r: r["p_holm"] < 0.05 and 1 <= r["mean_diff"] <= 5)
    add("H11.8", "AUC gain at N=3120 is at most +2", _row(auc, gain, 3120), lambda r: r["mean_diff"] <= 2)
    add("H11.9", "macro-F1 gain of A20[isc28] over A1 at N=228 is between 0 and +4",
        _row(f1, f"{nm(ISC28)}-{nm(A1)} (macro-F1, iSarcasmEval test)", 228), lambda r: 0 <= r["mean_diff"] <= 4)
    for n in NS:
        add("H11.10", f"A20[hs] vs A1 at N={n}: AUC difference within +-2 and p_holm >= 0.05",
            _row(auc, f"AUC {nm(HS)} - {nm(A1)}", n), lambda r: abs(r["mean_diff"]) <= 2 and r["p_holm"] >= 0.05)
    add("H11.11", "size-matched control: A20[hsub] gains at least +4 over A1 at Headlines 1%",
        _row(ctl, "A20[hsub]-A1 (3,120-headline pool vs BERT)", 1), lambda r: r["mean_diff"] >= 4)
    for n in (228, 780):
        add("H11.12", f"100-pass encoder within 1 AUC point of the 28.7-pass encoder at N={n}",
            _row(auc, f"AUC {nm(ISC100)} - {nm(ISC28)}", n), lambda r: abs(r["mean_diff"]) <= 1)
    # reading rule written before the run: "replicates" = gain >= +3 and robust at N = 228 and 780;
    # "collapses" = gain < +1 or interval includes 0 at both
    g = [_row(auc, gain, n) for n in (228, 780)]
    if all(x is not None for x in g):
        rep = all(x["mean_diff"] >= 3 and x["p_holm"] < 0.05 for x in g)
        col = all(x["mean_diff"] < 1 or (x["ci_low"] <= 0 <= x["ci_high"]) for x in g)
        rows.append({"hypothesis": "reading rule", "check": "in-domain gain replicates (>= +3 AUC, robust, N=228 and 780) or collapses (< +1 or interval includes 0 at both)",
                     "value": f"N=228 {g[0]['mean_diff']:+.2f}, N=780 {g[1]['mean_diff']:+.2f}",
                     "verdict": "REPLICATES" if rep else ("COLLAPSES" if col else "in between (smaller and/or noisier)")})
    return pd.DataFrame(rows, columns=["hypothesis", "check", "value", "verdict"])


def run_all(results_dirs, out_dir, n=10000, rng_seed=0) -> dict:
    """Write the E1 tables to out_dir (CSV + Markdown) and return them. Contrasts without runs are skipped."""
    runs = an.load_runs(*results_dirs)
    tables = {
        "summary": summary(runs),
        "isarc_auc": a57.auc_table(runs, n, rng_seed, specs=AUC_SPECS),
        "isarc_f1": _isarc_f1(runs, n, rng_seed),
        "size_control": a57.contrast_table(runs, "macro_f1", n, rng_seed, specs=CONTROL_SPECS),
    }
    tables["verdicts"] = verdicts(tables)
    for name, t in tables.items():
        if t is not None and len(t):
            an.save_table(t, out_dir, name, default_decimals=2)
    return tables


def _isarc_f1(runs, n, rng_seed) -> pd.DataFrame:
    """Macro-F1 contrasts on the iSarcasmEval test set (paired bootstrap over its items) with Holm."""
    t = an.label_quality_table(runs, "isarc_test", "macro_f1", n=n, rng_seed=rng_seed, specs=F1_SPECS)
    t["p_holm"] = a4.holm(t["p"]) if len(t) else []
    return t
