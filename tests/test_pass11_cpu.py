"""Pass 11 CPU analyses (polarity DiD, seed-aware bootstrap) on made-up predictions. No torch needed."""
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from gainsplit import analysis as an
from gainsplit import analysis_pass11_cpu as cpu

N = 600


def _labels():
    return np.tile([0, 1], N // 2)


def _write(root, arm, budget, seed, tag, pred, y):
    rid = f"{arm}_p{budget}_s{seed}" + (f"_{tag}" if tag else "")
    d = root / "runs" / rid
    d.mkdir(parents=True)
    p = np.where(pred == 1, 0.8, 0.2)
    pd.DataFrame({"id": np.arange(N), "label": y, "p_sarcastic": p}).to_csv(d / "preds_test.csv", index=False)
    f1 = f1_score(y, pred, average="macro")
    (d / "result.json").write_text(json.dumps({"run_id": rid, "config": {"arm": arm, "budget": budget, "seed": seed, "tag": tag},
                                               "scores": {"test": {"macro_f1": f1, "accuracy": float((pred == y).mean()),
                                                                   "f1_sarcastic": f1, "roc_auc": 0.9}}}))


def _predict(rng, y, err, extra_err_pos=None, pos=None):
    """Predicted classes with per-item error probability `err` (plus `extra_err_pos` on the items in `pos`)."""
    e = np.full(N, err)
    if pos is not None:
        e[pos] += extra_err_pos
    wrong = rng.random(N) < e
    return np.where(wrong, 1 - y, y)


@pytest.fixture
def stand_in(tmp_path):
    """A1/A2/A20/A31 at 1% and 10% (3 seeds). A2 errs 25 points more than A1 on positive-surface sarcastic items."""
    rng = np.random.default_rng(0)
    y = _labels()
    sarc = np.where(y == 1)[0]
    pos = sarc[:60]                                            # 60 of 300 sarcastic items are positive-surface
    sent = pd.DataFrame({"id": np.arange(N), "sent_label": np.where(np.isin(np.arange(N), pos), "positive", "negative")})
    for b in (1, 10):
        for s in range(3):
            base_err = 0.20 if b == 1 else 0.10
            for arm, err, extra in (("A1", base_err + 0.05 * s, 0.0), ("A2", base_err + 0.05 * s, 0.25)):
                _write(tmp_path, arm, b, s, "", _predict(rng, y, max(err, 0.02), extra, pos), y)
            a20 = _predict(rng, y, base_err - 0.07)            # A31 = A20 with six random flips: no polarity effect
            _write(tmp_path, "A20", b, s, "", a20, y)
            a31 = a20.copy()
            flip = rng.choice(N, 6, replace=False)
            a31[flip] = 1 - a31[flip]
            _write(tmp_path, "A31", b, s, "", a31, y)
    return tmp_path, sent, y


def test_cluster_bootstrap_matches_sklearn_and_is_wider_than_test_only(stand_in):
    root, _, y = stand_in
    runs = an.load_runs(root)
    r = cpu.cluster_bootstrap(runs, ("A20", 1, ""), ("A1", 1, ""), n=600, rng_seed=1)
    diffs = []
    for s in range(3):
        a = pd.read_csv(root / "runs" / f"A20_p1_s{s}" / "preds_test.csv")["p_sarcastic"].to_numpy() >= 0.5
        b = pd.read_csv(root / "runs" / f"A1_p1_s{s}" / "preds_test.csv")["p_sarcastic"].to_numpy() >= 0.5
        diffs.append(100 * (f1_score(y, a.astype(int), average="macro") - f1_score(y, b.astype(int), average="macro")))
    assert r["mean_diff"] == pytest.approx(np.mean(diffs), abs=1e-6)
    assert r["seeds_ahead"] == 3 and r["n_seeds"] == 3
    assert r["cluster_high"] - r["cluster_low"] > r["test_only_high"] - r["test_only_low"]    # seed spread adds uncertainty
    assert r["width_ratio"] > 1 and r["cluster_low"] > 0                                       # A20 is far ahead of A1
    again = cpu.cluster_bootstrap(runs, ("A20", 1, ""), ("A1", 1, ""), n=600, rng_seed=1)
    assert again["cluster_low"] == r["cluster_low"]                                            # deterministic


def test_polarity_did_finds_a_planted_effect_and_not_a_null(stand_in):
    root, sent, _ = stand_in
    runs = an.load_runs(root)
    hit = cpu.polarity_did(runs, sent, ("A1", ""), ("A2", ""), 1, n=2000)
    assert hit["n_pos"] == 60 and hit["n_rest"] == 240
    assert hit["mean_diff"] > 10 and hit["ci_low"] > 0 and hit["p"] < 0.01
    assert hit["err_pos_helper"] > hit["err_pos_base"]
    null = cpu.polarity_did(runs, sent, ("A20", ""), ("A31", ""), 1, n=2000)
    assert abs(null["mean_diff"]) < 5 and abs(null["mean_diff"]) < 0.3 * hit["mean_diff"]   # no planted effect


def test_run_all_cpu_writes_tables_and_verdicts(stand_in, tmp_path_factory):
    root, sent, _ = stand_in
    data = tmp_path_factory.mktemp("data")
    sent.assign(split="test").to_csv(data / "helper_labels.csv", index=False)
    out = tmp_path_factory.mktemp("out")
    t = cpu.run_all_cpu([root], data, out, n=500)
    assert set(t) == {"polarity_did", "cluster_bootstrap", "verdicts"}
    assert len(t["polarity_did"]) == 4 and "p_holm" in t["polarity_did"]
    v = t["verdicts"]
    assert {"H11.1", "H11.2", "H11.3"} <= set(v["hypothesis"])
    assert v[(v["hypothesis"] == "H11.1") & v["check"].str.contains("at 1%")].iloc[0]["verdict"] == "SUPPORTED"
    assert (out / "verdicts.csv").exists()
