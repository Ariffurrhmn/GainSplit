"""Tests for the results analysis, using made-up run folders (same file format as real runs)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gainsplit import analysis as an  # noqa: E402

N_ITEMS = 400


def labels(seed=123):
    return np.random.default_rng(seed).integers(0, 2, N_ITEMS)


def probs_with_accuracy(y, acc, seed):
    """Fake probabilities that are right on about `acc` of the items."""
    rng = np.random.default_rng(seed)
    right = rng.random(len(y)) < acc
    pred = np.where(right, y, 1 - y)
    return np.where(pred == 1, 0.9, 0.1)


def write_run(root, arm, budget, seed, p_test, with_gan_log=False):
    """Write one run folder: result.json plus the three preds files."""
    run_id = f"{arm}_p{budget}_s{seed}"
    d = Path(root) / "runs" / run_id
    d.mkdir(parents=True)
    y = labels()
    for split in ("test", "dev", "isarc_test"):
        pd.DataFrame({"id": np.arange(N_ITEMS), "label": y, "p_sarcastic": p_test}).to_csv(
            d / f"preds_{split}.csv", index=False)
    score = {"accuracy": 0.8, "macro_f1": 0.8, "f1_sarcastic": 0.8, "roc_auc": 0.9, "n": N_ITEMS}
    log = [{"step": s, "loss": 1.0} for s in (100, 200)]
    if with_gan_log:
        log = [{"step": s, "p_fake_real": 0.1, "p_fake_fake": 0.5} for s in (100, 200)]
    result = {"run_id": run_id,
              "config": {"arm": arm, "budget": budget, "seed": seed, "tag": ""},
              "scores": {"dev": score, "test": score, "isarc_test": score},
              "train_log": log}
    (d / "result.json").write_text(json.dumps(result))


@pytest.fixture
def runs_two_arms(tmp_path):
    """A1 (about 70% right) and A3 (about 90% right), 3 seeds, budget 10."""
    y = labels()
    for s in range(3):
        write_run(tmp_path, "A1", 10, s, probs_with_accuracy(y, 0.70, 10 + s))
        write_run(tmp_path, "A3", 10, s, probs_with_accuracy(y, 0.90, 20 + s))
    return an.load_runs(tmp_path)


def test_load_runs(runs_two_arms):
    assert len(runs_two_arms) == 6
    assert {"arm", "budget", "seed", "test_macro_f1", "preds_test"} <= set(runs_two_arms.columns)
    assert (runs_two_arms["preds_test"] != "").all()


def test_identical_predictions_give_zero_and_p_one(tmp_path):
    y = labels()
    for s in range(3):
        p = probs_with_accuracy(y, 0.8, s)
        write_run(tmp_path, "A1", 10, s, p)
        write_run(tmp_path, "A3", 10, s, p)
    r = an.paired_bootstrap(an.load_runs(tmp_path), "A3", "A1", 10, n=500)
    assert r["mean_diff"] == 0 and r["ci_low"] == 0 and r["ci_high"] == 0
    assert r["p"] == 1.0
    assert r["n_seeds"] == 3


def test_better_arm_has_positive_interval(runs_two_arms):
    r = an.paired_bootstrap(runs_two_arms, "A3", "A1", 10, n=1000)
    assert r["mean_diff"] > 0.1
    assert r["ci_low"] > 0
    assert r["p"] < 0.01
    # swapping the arms flips the sign
    r2 = an.paired_bootstrap(runs_two_arms, "A1", "A3", 10, n=1000)
    assert r2["ci_high"] < 0


def test_bootstrap_matches_sklearn_on_full_data(runs_two_arms):
    from sklearn.metrics import f1_score
    r = an.paired_bootstrap(runs_two_arms, "A3", "A1", 10, n=50)
    y = labels()
    diffs = []
    for s in range(3):
        a = pd.read_csv(runs_two_arms.query("arm == 'A3' and seed == @s")["preds_test"].iloc[0])
        b = pd.read_csv(runs_two_arms.query("arm == 'A1' and seed == @s")["preds_test"].iloc[0])
        fa = f1_score(y, (a["p_sarcastic"] >= 0.5).astype(int), average="macro")
        fb = f1_score(y, (b["p_sarcastic"] >= 0.5).astype(int), average="macro")
        diffs.append(fa - fb)
    assert r["mean_diff"] == pytest.approx(np.mean(diffs))


def test_missing_seeds_use_shared_seeds_only(tmp_path):
    y = labels()
    for s in range(3):
        write_run(tmp_path, "A1", 10, s, probs_with_accuracy(y, 0.7, s))
    for s in (1, 2, 3):
        write_run(tmp_path, "A3", 10, s, probs_with_accuracy(y, 0.9, 50 + s))
    runs = an.load_runs(tmp_path)
    assert an.paired_bootstrap(runs, "A3", "A1", 10, n=200)["n_seeds"] == 2
    # an arm that does not exist gives None, not a crash
    assert an.paired_bootstrap(runs, "A3", "A9", 10, n=200) is None


def test_summary_table_shape_and_values(runs_two_arms):
    s = an.summary_table(runs_two_arms, "test", "macro_f1")
    assert list(s.columns) == ["arm", "budget", "mean", "sd", "n_seeds"]
    assert len(s) == 2
    assert s["mean"].iloc[0] == pytest.approx(80.0)
    wide = an.summary_wide(s)
    assert list(wide.columns) == ["arm", "10%"]
    assert "±" in wide.iloc[0, 1]


def test_contrast_table_shape(tmp_path):
    y = labels()
    for arm, acc in (("A1", 0.7), ("A3", 0.75), ("A4", 0.8), ("A5", 0.9)):
        for budget in (1, 10):
            for s in range(2):
                write_run(tmp_path, arm, budget, s, probs_with_accuracy(y, acc, ord(arm[1]) * 10 + budget + s))
    runs = an.load_runs(tmp_path)
    t = an.contrast_table(runs, n=200)
    assert list(t.columns) == ["contrast", "budget", "mean_diff", "ci_low", "ci_high", "p", "n_seeds"]
    # contrasts whose arms all exist: A3-A1, A4-A1, A5-A1, A5-A3, A5-A4 and the interaction; 2 budgets each
    assert len(t) == 6 * 2
    assert set(t["contrast"]) == {"A3-A1", "A4-A1", "A5-A1", "A5-A3", "A5-A4", "A5-A4-A3+A1"}
    assert (t["ci_low"] <= t["ci_high"]).all()


def test_interaction_sign(tmp_path):
    """When A5 is better than A3 + A4 gains combined, the interaction is positive."""
    y = labels()
    accs = {"A1": 0.60, "A3": 0.65, "A4": 0.65, "A5": 0.90}
    for arm, acc in accs.items():
        for s in range(2):
            write_run(tmp_path, arm, 10, s, probs_with_accuracy(y, acc, 7 + s + len(arm) * ord(arm[1])))
    t = an.contrast_table(an.load_runs(tmp_path), n=300)
    row = t[t["contrast"] == "A5-A4-A3+A1"].iloc[0]
    assert row["mean_diff"] > 0 and row["ci_low"] > 0


def test_save_tables_and_figures(tmp_path):
    res = tmp_path / "res"
    y = labels()
    for arm in ("A1", "A4"):
        for s in range(2):
            write_run(res, arm, 10, s, probs_with_accuracy(y, 0.8, s), with_gan_log=(arm == "A4"))
    out = tmp_path / "out"
    tables = an.run_all([res], out_dir=out, n=100)
    for f in ("summary_test_macro_f1.csv", "summary_test_macro_f1.md", "isarc_test.csv",
              "contrasts_test_macro_f1.md", "fig_f1_vs_budget.pdf", "fig_f1_vs_budget.png",
              "fig_gan_curves.pdf", "fig_gan_curves.png"):
        assert (out / f).exists(), f
    assert len(tables["isarc"]) == 2


def test_dev_only_tuning_runs_are_left_out(tmp_path):
    y = labels()
    write_run(tmp_path, "A1", 10, 0, probs_with_accuracy(y, 0.8, 1))
    d = tmp_path / "runs" / "A7_p10_s0_eps1"
    d.mkdir(parents=True)
    (d / "result.json").write_text(json.dumps({
        "run_id": "A7_p10_s0_eps1", "config": {"arm": "A7", "budget": 10, "seed": 0, "tag": "eps1"},
        "scores": {"dev": {"accuracy": 0.8, "macro_f1": 0.8, "f1_sarcastic": 0.8, "roc_auc": 0.9}},
        "train_log": []}))
    assert list(an.load_runs(tmp_path)["arm"]) == ["A1"]
    assert len(an.load_runs(tmp_path, test_only=False)) == 2


# ---------------------------------------------------------------- Pass 3

def write_run3(root, arm, budget, seed, tag, p_by_split):
    """Run folder with a tag and any set of splits (dict split -> probabilities, same items)."""
    from sklearn.metrics import f1_score
    run_id = f"{arm}_p{budget}_s{seed}" + (f"_{tag}" if tag else "")
    d = Path(root) / "runs" / run_id
    d.mkdir(parents=True)
    y = labels()
    sc = {}
    for split, p in p_by_split.items():
        pd.DataFrame({"id": np.arange(N_ITEMS), "label": y, "p_sarcastic": p}).to_csv(d / f"preds_{split}.csv", index=False)
        pred = (p >= 0.5).astype(int)
        sc[split] = {"accuracy": 0.8, "macro_f1": float(f1_score(y, pred, average="macro")),
                     "f1_sarcastic": 0.8, "roc_auc": 0.9}
    (d / "result.json").write_text(json.dumps({
        "run_id": run_id, "config": {"arm": arm, "budget": budget, "seed": seed, "tag": tag},
        "scores": sc, "train_log": []}))


def test_tagged_and_untagged_do_not_collide(tmp_path):
    y = labels()
    write_run3(tmp_path, "A3", 10, 0, "", {"test": probs_with_accuracy(y, 0.8, 1)})
    write_run3(tmp_path, "A3", 10, 0, "soft", {"test": probs_with_accuracy(y, 0.9, 2)})
    runs = an.load_runs(tmp_path)
    assert len(runs) == 2 and set(runs["tag"]) == {"", "soft"}
    assert len(an.untagged(runs)) == 1
    # main tables ignore the tagged run
    assert an.summary_table(runs)["n_seeds"].tolist() == [1]


def test_untagged_outputs_unchanged_by_tagged_runs(tmp_path):
    y = labels()
    for arm, acc in (("A1", 0.7), ("A3", 0.8)):
        for s in range(2):
            write_run3(tmp_path, arm, 10, s, "", {"test": probs_with_accuracy(y, acc, s + ord(arm[1]))})
    before = an.contrast_table(an.load_runs(tmp_path), n=300)
    for s in range(2):
        write_run3(tmp_path, "A3", 10, s, "soft", {"test": probs_with_accuracy(y, 0.95, 90 + s)})
    after_runs = an.load_runs(tmp_path)
    after = an.contrast_table(after_runs, n=300)
    pd.testing.assert_frame_equal(before, after)
    assert an.summary_table(after_runs)["n_seeds"].tolist() == [2, 2]


def test_load_runs_error_names_folders(tmp_path):
    with pytest.raises(FileNotFoundError, match="nothing_here"):
        an.load_runs(tmp_path / "nothing_here")


def test_mask_rerun_check_flags_planted_mismatch(tmp_path):
    y = labels()
    for s in range(2):
        p = probs_with_accuracy(y, 0.8, s)
        write_run3(tmp_path, "A1", 10, s, "", {"test": p})
        write_run3(tmp_path, "A1", 10, s, "mask", {"test": p, "test_cuemask": p, "test_randmask": p})
    chk = an.check_mask_reruns(an.load_runs(tmp_path))
    assert chk["ok"].all() and (chk["pred_diff"] == 0).all()
    # plant a difference in one mask run
    f = tmp_path / "runs" / "A1_p10_s1_mask" / "preds_test.csv"
    df = pd.read_csv(f)
    df.loc[0, "p_sarcastic"] = 1 - df.loc[0, "p_sarcastic"]
    df.to_csv(f, index=False)
    with pytest.warns(UserWarning):
        chk = an.check_mask_reruns(an.load_runs(tmp_path))
    assert chk["ok"].tolist() == [True, False]
    assert chk["pred_diff"].max() > 0.5


def test_shortcut_gap_sign(tmp_path):
    """Cue masking hurts a lot more than random masking -> positive gap with an interval above 0."""
    y = labels()
    for s in range(3):
        write_run3(tmp_path, "A1", 10, s, "mask", {
            "test": probs_with_accuracy(y, 0.90, s),
            "test_cuemask": probs_with_accuracy(y, 0.70, 10 + s),
            "test_randmask": probs_with_accuracy(y, 0.88, 20 + s)})
    t = an.masked_table(an.load_runs(tmp_path), n=500)
    r = t.iloc[0]
    assert r["drop_cue"] > r["drop_rand"] and r["shortcut_gap"] > 5
    assert r["gap_ci_low"] > 0 and r["drop_cue_ci_low"] > 0
    assert r["gap_p"] < 0.01


def test_auc_interval_contains_true_auc(tmp_path):
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0)
    n = 500
    y = rng.integers(0, 2, n)
    p = np.round(rng.normal(y * 1.0, 1.0), 1)        # rounding creates ties
    # on the full data the rank-based AUC equals sklearn (ties included)
    assert an._boot_auc(np.ones((1, n)), y, p)[0] == pytest.approx(roc_auc_score(y, p))
    # a resample with known draw counts equals sklearn on the expanded sample
    idx = rng.integers(0, n, n)
    counts = np.bincount(idx, minlength=n)[None, :].astype(float)
    assert an._boot_auc(counts, y, p)[0] == pytest.approx(roc_auc_score(y[idx], p[idx]))
    f = tmp_path / "preds_isarc_test.csv"
    pd.DataFrame({"id": np.arange(n), "label": y, "p_sarcastic": p}).to_csv(f, index=False)
    res = an.auc_interval(pd.DataFrame({"preds_isarc_test": [str(f)]}), n=500)
    true_auc = roc_auc_score(y, p)
    assert res["ci_low"] < true_auc < res["ci_high"]
    assert res["auc"] == pytest.approx(true_auc)


def test_transfer_table_excludes_chance(tmp_path):
    y = labels()
    rng = np.random.default_rng(5)
    good = np.clip(0.5 + (y - 0.5) * 0.4 + rng.normal(0, 0.2, N_ITEMS), 0, 1)
    chance = rng.random(N_ITEMS)
    for s in range(2):
        write_run3(tmp_path, "A1", 10, s, "", {"test": good, "isarc_test": chance})
        write_run3(tmp_path, "A1", 100, s, "isarctrain", {"test": good, "isarc_test": good})
    t = an.transfer_table(an.load_runs(tmp_path), n=300)
    chance_row = t[t["tag"] == ""].iloc[0]
    ref = t[t["tag"] == "isarctrain"].iloc[0]
    assert not chance_row["excludes_50"] and ref["excludes_50"] and ref["in_domain_reference"]


def test_label_quality_table_names_tag_pairs(tmp_path):
    y = labels()
    for s in range(2):
        write_run3(tmp_path, "A2", 1, s, "", {"test": probs_with_accuracy(y, 0.85, s)})
        write_run3(tmp_path, "A2", 1, s, "flip50", {"test": probs_with_accuracy(y, 0.65, 5 + s)})
        write_run3(tmp_path, "A9", 1, s, "shuf", {"test": probs_with_accuracy(y, 0.65, 8 + s)})
        write_run3(tmp_path, "A1", 1, s, "", {"test": probs_with_accuracy(y, 0.75, 11 + s)})
    t = an.label_quality_table(an.load_runs(tmp_path), n=300)
    assert set(t["contrast"]) == {"A2[flip50]-A2", "A9[shuf]-A2", "A9[shuf]-A1"}
    assert (t["ci_high"] < 0).all()


def test_mutual_information():
    rng = np.random.default_rng(0)
    x = rng.integers(0, 3, 5000)
    y_ind = rng.integers(0, 2, 5000)
    nmi, v = an.association(x, y_ind)
    assert nmi < 0.01 and v < 0.05
    nmi, v = an.association(x, (x == 0).astype(int))
    assert nmi > 0.3 and v > 0.9
    assert an.association(x, x)[0] == pytest.approx(1.0)


def test_helper_and_gold_tables():
    rng = np.random.default_rng(1)
    n = 200
    lab = rng.integers(0, 2, n)
    names = ["negative", "neutral", "positive"]
    gold_s = rng.choice(names, n)
    teacher = np.where(rng.random(n) < 0.7, gold_s, rng.choice(names, n))
    train = pd.DataFrame({"id": np.arange(n), "text": "x", "label": lab})
    probs = rng.dirichlet(np.ones(3), n)
    helper = pd.DataFrame({"id": np.arange(n), "split": "train", "sent_label": teacher,
                           "emo_label": rng.integers(0, 5, n),
                           **{f"sent_p_{i}": probs[:, i] for i in range(3)},
                           **{f"emo_p_{i}": rng.random(n) for i in range(5)}})
    t = an.helper_shortcut_table(helper, train)
    assert len(t) == 3 and t["auc"].iloc[-1] < 0.65     # independent labels -> probe near chance
    gold = pd.DataFrame({"id": np.arange(n), "gold_sentiment": gold_s, "unsure": rng.integers(0, 2, n),
                         "label": lab, "sent_label": teacher})
    g = an.gold_table(gold)
    assert set(g["subset"]) == {"all", "unsure == 0"} and (g["accuracy"] > 60).all()
    s = an.gold_shortcut_table(gold, n_perm=20)
    assert len(s) == 3 and 0 < s["p_perm"].iloc[-1] <= 1
