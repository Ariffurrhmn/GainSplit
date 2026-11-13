"""Pass 4 without torch: config, plan, pair loading and controls, explanation helpers, analysis."""
import json
import os

import numpy as np
import pandas as pd
import pytest

from gainsplit import plans
from gainsplit.config import ARMS, RunConfig
from gainsplit.data import (check_pairs, find_input_file, load_cue_words, load_pairs, matched_subset, preflight,
                            unpaired_sides)
from gainsplit.explain import cue_share, pick_items, word_scores


@pytest.fixture(autouse=True)
def no_kaggle_search(monkeypatch, tmp_path):
    """Never search the real /kaggle/input in tests; only the fixture folder counts."""
    monkeypatch.delenv("GAINSPLIT_PAIRS", raising=False)
    monkeypatch.setenv("GAINSPLIT_INPUT", str(tmp_path / "nothing_here"))


# ---- config ----------------------------------------------------------------------------
def test_old_run_ids_and_arms_unchanged():
    assert RunConfig("A1", 10, 0).run_id == "A1_p10_s0"
    assert ARMS["A7"] == {"gan": False, "helpers": False, "fgm": True}
    c = RunConfig("A1", 10, 0)
    assert (c.pair_batch, c.explain) == (32, False)


@pytest.mark.parametrize("kw", [
    dict(arm="A11"),                                   # FGM arm without epsilon
    dict(arm="A10", fgm_epsilon=0.5),                  # epsilon on a non-FGM arm
    dict(arm="A10", pair_batch=15),                    # odd pair batch would split pairs
    dict(arm="A10", explain=True),                     # explain needs the test set
    dict(arm="A4", explain=True, eval_test=True),      # explain needs a plain head
    dict(arm="A10", train_data="isarcasm", budget=100),
    dict(arm="A1", pair_weight=0.3),                   # pair_weight on a non-pair arm
    dict(arm="A10", pair_weight=-0.1),                 # negative weight
    dict(arm="A1", cleaned_drop_pct=20),                # cleaned_drop_pct on a non-pair arm
    dict(arm="A10", cleaned_drop_pct=20),               # cleaned_drop_pct without pairs="cleaned"
    dict(arm="A12", cleaned_drop_pct=20),                # A12 uses pairs="unpaired", not "cleaned"
    dict(arm="A16"),                                    # A16 is pairs="cleaned": needs cleaned_drop_pct
    dict(arm="A16", cleaned_drop_pct=0),                # must be > 0
    dict(arm="A16", cleaned_drop_pct=100),               # ... and < 100
])
def test_bad_pass4_combinations_are_rejected(kw):
    with pytest.raises(ValueError):
        RunConfig(**{"budget": 10, "seed": 0, **kw})


def test_good_pass4_combinations():
    RunConfig("A11", 10, 0, fgm_epsilon=0.5)
    RunConfig("A10", 10, 0, eval_test=True, explain=True)
    RunConfig("A1", 10, 0, eval_test=True, explain=True, tag="explain")
    RunConfig("A10", 10, 0, pair_weight=0.3)
    RunConfig("A10", 10, 0, pair_weight=0.0)  # allowed: pairs join every batch but get no loss weight
    RunConfig("A16", 10, 0, cleaned_drop_pct=20)
    RunConfig("A17", 10, 0, fgm_epsilon=0.5, cleaned_drop_pct=20, pair_weight=0.5)


# ---- plan -------------------------------------------------------------------------------
def test_pass4_plan():
    runs = plans.pass4()
    assert len(runs) == 75
    assert len({r.run_id for r in runs}) == 75
    assert all(r.eval_test and r.eval_masked and r.pair_batch == 32 for r in runs)
    count = pd.Series([(r.arm, r.budget, r.tag) for r in runs]).value_counts()
    assert (count == 5).all() and len(count) == 15
    assert {r.arm for r in runs if r.explain} == {"A1", "A10"} and all(r.budget == 10 for r in runs if r.explain)
    assert all(r.fgm_epsilon == plans.PASS2_FGM_EPSILON for r in runs if r.arm in ("A7", "A11"))
    assert all(r.step_mult == 3 for r in runs if r.tag == "steps3x")
    # re-runs carry a tag, so they never overwrite (or get skipped as) the original runs
    assert all(r.tag in ("mask", "explain") for r in runs if r.arm in ("A1", "A7"))
    assert [r.budget for r in runs[:35]] == [10] * 35 and [r.budget for r in runs[35:]] == [1] * 40


def test_pass4_registered_and_older_plans_unchanged():
    assert plans.PLANS["pass4"] is plans.pass4 and len(plans.PLANS["pass4_full100"]()) == 5
    expected = {"pass1": 45, "pass2a": 104, "pass2b": 10, "pass3": 105, "gan_check": 12}
    for name, n in expected.items():
        runs = plans.PLANS[name]()
        assert len(runs) == n and all(r.pair_batch == 32 and not r.explain for r in runs)


def test_pair_weight_tuning():
    runs = plans.pair_weight_tuning()
    assert len(runs) == 9 and len({r.run_id for r in runs}) == 9
    assert all(r.arm == "A10" and r.budget == 10 and not r.eval_test for r in runs)
    assert sorted({r.pair_weight for r in runs}) == sorted(plans.PAIR_WEIGHT_GRID)
    assert {r.tag for r in runs} == {f"pw{w:g}" for w in plans.PAIR_WEIGHT_GRID}


def test_cleaned_drop_tuning_needs_pair_weight_set(monkeypatch):
    monkeypatch.setattr(plans, "PASS5_PAIR_WEIGHT", None)
    with pytest.raises(ValueError, match="PASS5_PAIR_WEIGHT"):
        plans.cleaned_drop_tuning()


def test_cleaned_drop_tuning(monkeypatch):
    monkeypatch.setattr(plans, "PASS5_PAIR_WEIGHT", 0.3)
    runs = plans.cleaned_drop_tuning()
    assert len(runs) == 9 and len({r.run_id for r in runs}) == 9
    assert all(r.arm == "A16" and r.budget == 10 and r.pair_weight == 0.3 and not r.eval_test for r in runs)
    assert sorted({r.cleaned_drop_pct for r in runs}) == sorted(plans.CLEANED_DROP_GRID)


@pytest.mark.parametrize("missing", ["PASS5_PAIR_WEIGHT", "PASS5_CLEANED_DROP_PCT"])
def test_pass5_needs_both_tuned_values_set(monkeypatch, missing):
    monkeypatch.setattr(plans, "PASS5_PAIR_WEIGHT", 0.3)
    monkeypatch.setattr(plans, "PASS5_CLEANED_DROP_PCT", 20.0)
    monkeypatch.setattr(plans, missing, None)
    with pytest.raises(ValueError, match=missing):
        plans.pass5()


def test_pass5_plan(monkeypatch):
    monkeypatch.setattr(plans, "PASS5_PAIR_WEIGHT", 0.3)
    monkeypatch.setattr(plans, "PASS5_CLEANED_DROP_PCT", 20.0)
    runs = plans.pass5()
    assert len(runs) == 30 and len({r.run_id for r in runs}) == 30
    assert all(r.eval_test and r.eval_masked and r.pair_batch == 32 for r in runs)
    count = pd.Series([(r.arm, r.budget, r.tag) for r in runs]).value_counts()
    assert (count == 5).all() and len(count) == 6
    assert all(r.pair_weight == 0.3 for r in runs)
    assert all(r.cleaned_drop_pct == 20.0 for r in runs if r.arm in ("A16", "A17"))
    assert all(r.cleaned_drop_pct == 0.0 for r in runs if r.arm == "A10")  # unfiltered, weight-only comparison
    assert all(r.fgm_epsilon == plans.PASS2_FGM_EPSILON for r in runs if r.arm == "A17")
    assert all(r.tag == "weighted" for r in runs if r.arm == "A10")       # never collides with Pass 4's A10 runs
    assert {r.budget for r in runs} == {1, 10}


# ---- pair data -----------------------------------------------------------------------------
def test_load_unfiltered_sorted_and_checked(pass4_data):
    rows, info = load_pairs("unfiltered", pass4_data, check_against=pass4_data)
    assert len(rows) == 240 and info["n_seeds"] == 120
    assert rows["seed_id"].is_monotonic_increasing
    assert (rows["label"].to_numpy()[::2] == 0).all() and (rows["label"].to_numpy()[1::2] == 1).all()
    assert info["seeds_per_model"] == {"gemma": 40, "mistral": 40, "qwen": 40}


def test_filtered_and_matched_have_same_size_and_model_mix(pass4_data):
    fil, fi = load_pairs("filtered", pass4_data)
    mat, mi = load_pairs("matched", pass4_data)
    assert fi["seeds_per_model"] == mi["seeds_per_model"] == {"gemma": 40, "mistral": 25, "qwen": 20}
    assert len(fil) == len(mat) == 170
    check_pairs(mat)                                     # whole pairs only
    again, _ = load_pairs("matched", pass4_data)
    pd.testing.assert_frame_equal(mat, again)            # same subset every time
    assert set(mat["seed_id"]) != set(fil["seed_id"])    # a random draw, not the judge's choice


def test_unpaired_keeps_one_side_per_seed(pass4_data):
    rows, info = load_pairs("unpaired", pass4_data)
    assert len(rows) == 120 and rows["seed_id"].is_unique
    assert 0.3 < rows["label"].mean() < 0.7
    unf, _ = load_pairs("unfiltered", pass4_data)
    assert set(rows["id"]) <= set(unf["id"])
    pd.testing.assert_frame_equal(rows, unpaired_sides(unf).sort_values(["seed_id", "label"]).reset_index(drop=True))


def test_changed_pair_file_is_caught(pass4_data):
    p = pass4_data / "pairs_train.csv"
    p.write_text(p.read_text() + "999_s,999,qwen,extra row,1\n")
    with pytest.raises(ValueError, match="manifest_pairs"):
        load_pairs("unfiltered", pass4_data)


def test_shortcut_scores_and_cleaned_subset(pass4_data):
    from gainsplit import pair_checks as pc
    from gainsplit.data import cleaned_subset, load_shortcut_scores
    unf, _ = load_pairs("unfiltered", pass4_data)
    sc = pc.shortcut_scores(unf)
    assert set(sc.columns) >= {"id", "seed_id", "label", "row_score", "seed_shortcut_score"}
    assert len(sc) == len(unf) and 0 <= sc.attrs["cv_auc"] <= 1
    # a seed's score is the max of its two rows, same for both rows of that seed
    g = sc.groupby("seed_id")["seed_shortcut_score"].nunique()
    assert (g == 1).all()

    scores = load_shortcut_scores(pass4_data)
    assert set(scores.index) == set(unf["seed_id"])
    cleaned = cleaned_subset(unf, scores, drop_pct=20)
    check_pairs(cleaned)                                 # whole pairs only, nothing split
    assert cleaned["seed_id"].nunique() == round(unf["seed_id"].nunique() * 0.8)
    kept_max_score = scores.loc[cleaned["seed_id"].unique()].max()
    dropped = set(unf["seed_id"]) - set(cleaned["seed_id"])
    assert scores.loc[list(dropped)].min() >= kept_max_score  # dropped seeds score no lower than kept ones
    again = cleaned_subset(unf, scores, drop_pct=20)
    pd.testing.assert_frame_equal(cleaned, again)        # deterministic

    with pytest.raises(ValueError, match="between 0 and 100"):
        cleaned_subset(unf, scores, drop_pct=0)
    with pytest.raises(ValueError, match="no shortcut score"):
        cleaned_subset(unf, scores.drop(scores.index[0]), drop_pct=20)


def test_load_pairs_cleaned(pass4_data):
    rows, info = load_pairs("cleaned", pass4_data, cleaned_drop_pct=25)
    check_pairs(rows)
    unf, _ = load_pairs("unfiltered", pass4_data)
    assert rows["seed_id"].nunique() == round(unf["seed_id"].nunique() * 0.75)
    assert info["cleaned_drop_pct"] == 25
    assert set(rows["id"]) <= set(unf["id"])


def test_changed_shortcut_scores_file_is_caught(pass4_data):
    p = pass4_data / "pair_shortcut_scores.csv"
    p.write_text(p.read_text() + "99999,0.0\n")
    with pytest.raises(ValueError, match="manifest_shortcut_scores"):
        load_pairs("cleaned", pass4_data, cleaned_drop_pct=20)


def test_filtered_from_another_pairs_file_is_caught(pass4_data):
    m = json.loads((pass4_data / "manifest_judge_v2.json").read_text())
    m["input_pairs_train_sha256"] = "0" * 64
    (pass4_data / "manifest_judge_v2.json").write_text(json.dumps(m))
    with pytest.raises(ValueError, match="different pairs_train"):
        load_pairs("filtered", pass4_data)


def test_leak_and_structure_checks():
    ok = pd.DataFrame({"id": ["1_s", "1_n"], "seed_id": [1, 1], "model": ["q", "q"],
                       "text": ["a b", "c d"], "label": [1, 0]})
    check_pairs(ok, train_ids=[1, 2], heldout_texts=["x y"])
    with pytest.raises(ValueError, match="dev or test"):
        check_pairs(ok, heldout_texts=["c d"])
    with pytest.raises(ValueError, match="not in headlines_train"):
        check_pairs(ok, train_ids=[2])
    with pytest.raises(ValueError, match="exactly one"):
        check_pairs(ok.assign(label=[1, 1]))


def test_matched_subset_fails_if_not_enough_pairs():
    unf = pd.DataFrame({"id": ["1_s", "1_n"], "seed_id": [1, 1], "model": ["q", "q"], "text": ["a", "b"], "label": [1, 0]})
    fil = pd.DataFrame({"seed_id": [1, 2], "model": ["q", "q"]})
    with pytest.raises(ValueError, match="wanted"):
        matched_subset(unf, fil)


def test_find_input_file_search_and_copies(pass4_data, tmp_path, monkeypatch):
    root = tmp_path / "inputs"
    for sub in ("a", "b"):
        (root / sub).mkdir(parents=True)
        (root / sub / "pairs_train.csv").write_bytes((pass4_data / "pairs_train.csv").read_bytes())
    assert find_input_file("pairs_train.csv", None, root).name == "pairs_train.csv"
    (root / "b" / "pairs_train.csv").write_text("changed")
    with pytest.raises(ValueError, match="different copies"):
        find_input_file("pairs_train.csv", None, root)
    with pytest.raises(FileNotFoundError):
        find_input_file("nope.csv", None, root)


def test_preflight_lists_every_pair_set(pass4_data):
    t = preflight(plans.pass4(), pass4_data)
    assert list(t["pair_set"]) == ["filtered", "matched", "unfiltered", "unpaired"]
    assert list(t["rows"]) == [170, 170, 240, 120]
    assert load_cue_words(pass4_data) == ["area", "man", "nation", "report", "study", "finds"]


# ---- explanation helpers (no torch) ----------------------------------------------------------
def test_pick_items_is_balanced_and_fixed():
    test = pd.DataFrame({"id": range(100), "text": ["x"] * 100, "label": [0] * 60 + [1] * 40})
    a, b = pick_items(test, n=40), pick_items(test, n=40)
    assert a.equals(b) and (a["label"].value_counts() == 20).all()


def test_word_scores_merge_pieces_and_skip_specials():
    text = "playing area man"
    offsets = [(0, 0), (0, 4), (4, 7), (8, 12), (13, 16), (0, 0), (0, 0)]
    word_ids = [None, 0, 0, 1, 2, None, None]
    words, scores = word_scores(text, offsets, word_ids, [9, 0.5, 0.25, -1.0, 2.0, 5, 0])
    assert words == ["playing", "area", "man"] and scores == [0.75, -1.0, 2.0]
    assert cue_share(words, scores, {"area", "man"}) == pytest.approx(3.0 / 3.75)
    assert cue_share(words, [0, 0, 0], {"area"}) == 0.0


# ---- CPU pair checks -------------------------------------------------------------------------
def test_pair_checks_run_on_fixture(pass4_data):
    from gainsplit import pair_checks as pc
    train = pd.read_csv(pass4_data / "headlines_train.csv", keep_default_na=False)
    dev = pd.read_csv(pass4_data / "headlines_dev.csv", keep_default_na=False)
    pairs, _ = load_pairs("unfiltered", pass4_data)
    style = pc.style_probe(train, pairs)
    assert list(style["generated_by"]) == ["all", "gemma", "mistral", "qwen"]
    assert style["auc_generated_vs_real"].between(0, 1).all()
    reader, wide = pc.real_reader(train, dev, pairs)
    assert len(reader) == 5 and len(wide) == 120
    cues = pc.cue_predictiveness(train, pairs, load_cue_words(pass4_data))
    assert cues["with_cue_sarcastic"].between(0, 1).all()
    lengths = pc.length_table(train, pairs)
    assert lengths["auc_length_only"].between(0, 1).all()


def test_shortcut_word_table_runs_on_fixture(pass4_data):
    from gainsplit import pair_checks as pc
    train = pd.read_csv(pass4_data / "headlines_train.csv", keep_default_na=False)
    pairs, _ = load_pairs("unfiltered", pass4_data)
    t = pc.shortcut_word_table(pairs, train, top_k=5)
    assert len(t) <= 10 and set(t["leans"]) <= {"sarcastic", "serious"}
    assert 0 <= t.attrs["cv_auc"] <= 1
    assert t["real_rate_sarcastic"].between(0, 1).all() and t["real_rate_serious"].between(0, 1).all()
    assert (t["real_n_sarcastic"] > 0).all() and (t["real_n_serious"] > 0).all()
    # without train, no real_rate columns, same word/coef columns otherwise
    t2 = pc.shortcut_word_table(pairs, top_k=5)
    assert "real_rate_sarcastic" not in t2.columns and len(t2) == len(t)


def test_shortcut_feature_table_runs_on_fixture(pass4_data):
    from gainsplit import pair_checks as pc
    train = pd.read_csv(pass4_data / "headlines_train.csv", keep_default_na=False)
    pairs, _ = load_pairs("unfiltered", pass4_data)
    t = pc.shortcut_feature_table(pairs, train)
    assert set(t["feature"]) == {"n_words", "n_chars", "n_exclaim", "n_question", "n_allcaps_words",
                                 "has_thus", "has_therefore", "has_clearly", "has_obviously",
                                 "has_literally", "has_honestly", "has_indeed", "has_surely"}
    assert t["generated_auc"].dropna().between(0, 1).all() and t["real_auc"].dropna().between(0, 1).all()
    # without train, only the generated_* columns
    t2 = pc.shortcut_feature_table(pairs)
    assert not any(c.startswith("real_") for c in t2.columns)


def test_isarc_length_shortcut_runs_on_fixture(pass4_data):
    from gainsplit import pair_checks as pc
    train = pd.read_csv(pass4_data / "headlines_train.csv", keep_default_na=False)
    isarc_test = pd.read_csv(pass4_data / "isarcasm_test.csv", keep_default_na=False)
    pairs, _ = load_pairs("unfiltered", pass4_data)
    t = pc.isarc_length_shortcut(isarc_test, pairs, train)
    assert list(t["set"]) == ["isarc test", "real train (reference)", "generated pairs (reference)"]
    assert t["auc_length_only"].between(0, 1).all()
    # without the optional reference frames, only the isarc row is returned
    assert list(pc.isarc_length_shortcut(isarc_test)["set"]) == ["isarc test"]


# ---- analysis -------------------------------------------------------------------------------
def test_holm():
    from gainsplit.analysis_pass4 import holm
    np.testing.assert_allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def _write(root, arm, budget, seed, tag, y, p, explain=None):
    run_id = f"{arm}_p{budget}_s{seed}" + (f"_{tag}" if tag else "")
    d = root / "runs" / run_id
    d.mkdir(parents=True)
    for split in ("test", "dev", "isarc_test", "test_cuemask", "test_randmask"):
        pd.DataFrame({"id": np.arange(len(y)), "label": y, "p_sarcastic": p}).to_csv(d / f"preds_{split}.csv", index=False)
    s = {"accuracy": 0.8, "macro_f1": 0.8, "f1_sarcastic": 0.8, "roc_auc": 0.8}
    scores = {k: s for k in ("dev", "test", "isarc_test", "test_cuemask", "test_randmask")}
    (d / "result.json").write_text(json.dumps({"run_id": run_id, "scores": scores,
                                               "config": {"arm": arm, "budget": budget, "seed": seed, "tag": tag}}))
    if explain is not None:
        explain.to_csv(d / "explain.csv", index=False)


def test_isarc_length_auc_matches_run_items_and_rejects_mismatch(tmp_path):
    from gainsplit import analysis as an
    from gainsplit import analysis_pass4 as a4
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    p = rng.random(200)
    _write(tmp_path, "A1", 10, 0, "", y, p)
    runs = an.load_runs(tmp_path)
    ids = pd.read_csv(runs.iloc[0]["preds_isarc_test"])["id"].to_numpy()
    texts = pd.Series(["word " * (n % 5 + 1) for n in range(len(y))])
    isarc_test = pd.DataFrame({"id": ids, "text": texts.to_numpy(), "label": y})
    res = a4.isarc_length_auc(isarc_test, runs, n=200)
    assert 0 <= res["auc"] <= 100 and res["n_items"] == len(y)
    assert res["ci_low"] <= res["auc"] <= res["ci_high"]

    bad = isarc_test.copy()
    bad["label"] = 1 - bad["label"]
    with pytest.raises(ValueError):
        a4.isarc_length_auc(bad, runs, n=50)

    # with no isarc predictions available, falls back to the given frame's own labels
    empty_runs = runs.assign(preds_isarc_test="")
    res2 = a4.isarc_length_auc(isarc_test, empty_runs, n=50)
    assert res2["n_items"] == len(y)


def test_pass4_analysis_end_to_end(tmp_path):
    from gainsplit import analysis as an
    from gainsplit import analysis_pass4 as a4
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    good = np.where(rng.random(300) < 0.9, y, 1 - y) * 0.8 + 0.1
    weak = np.where(rng.random(300) < 0.7, y, 1 - y) * 0.8 + 0.1
    ex = lambda share: pd.DataFrame({"id": range(50), "cue_share": share, "n_cue_words": 1,
                                     "completeness_gap": 0.01, "score_diff": 1.0})
    for s in range(3):
        for b in (1, 10):
            _write(tmp_path, "A1", b, s, "", y, weak)
            _write(tmp_path, "A1", b, s, "mask", y, weak)
            _write(tmp_path, "A10", b, s, "", y, good, ex(0.2) if b == 10 else None)
        _write(tmp_path, "A1", 10, s, "explain", y, weak, ex(0.4))
    runs = an.load_runs(tmp_path)
    con = a4.contrast_table(runs, n=500)
    row = con[(con["contrast"] == "A10-A1") & (con["budget"] == 10)].iloc[0]
    assert row["mean_diff"] > 5 and row["p"] < 0.01 and "p_holm" in con
    assert a4.check_reruns(runs)["ok"].all()
    assert len(a4.gap_contrasts(runs, n=200)) == 2
    auc = a4.auc_table(runs, n=200)
    assert (auc.loc[auc["contrast"] == "AUC A10 - A1", "mean_diff"] > 0).all()
    e = a4.explain_contrast_table(a4.load_explain(runs), n=200)
    assert e.iloc[0]["mean_diff"] == pytest.approx(-0.2)
    isarc_test = pd.DataFrame({"id": np.arange(len(y)), "text": ["word " * (i % 5 + 1) for i in range(len(y))],
                              "label": y})
    tables = a4.run_all([tmp_path], tmp_path / "out", n=200, isarc_test=isarc_test)
    assert (tmp_path / "out" / "contrasts.csv").exists() and len(tables["summary"])
    assert (tmp_path / "out" / "isarc_length_shortcut.csv").exists()
    assert tables["isarc_length_shortcut"]["auc"].between(0, 100).all()


# ---------------------------------------------------------------------------- Pass 6

def test_pass6_plan():
    runs = plans.pass6()
    assert len(runs) == 20 and len({r.run_id for r in runs}) == 20
    assert all(r.eval_test and r.eval_masked and r.pretrain_steps == plans.PASS6_PRETRAIN_STEPS == 720 for r in runs)
    assert all(r.pair_weight == 1.0 and r.cleaned_drop_pct == 0 for r in runs)
    count = pd.Series([(r.arm, r.budget) for r in runs]).value_counts()
    assert (count == 5).all() and set(count.index) == {("A18", 10), ("A18", 1), ("A19", 10), ("A19", 1)}
    assert all((r.fgm_epsilon > 0) == (r.arm == "A19") for r in runs)


def test_pass6_stage1_matches_a15_at_10_percent():
    """Stage 1 is meant to be exactly Pass 4's A15_p10 model: same step count as A15 at 10%."""
    a15 = RunConfig("A15", 10, 0)
    assert a15.steps(2280) == plans.PASS6_PRETRAIN_STEPS   # 10% of the 22,800 training headlines


@pytest.mark.parametrize("kw,msg", [
    ({"arm": "A18"}, "pretrain_steps > 0"),
    ({"arm": "A1", "pretrain_steps": 10}, "only used by the two-stage arms"),
    ({"arm": "A18", "pretrain_steps": 10, "pair_weight": 0.5}, "pair_weight must stay 1.0"),
    ({"arm": "A19", "pretrain_steps": 10}, "fgm_epsilon > 0"),
])
def test_pass6_config_checks(kw, msg):
    with pytest.raises(ValueError, match=msg):
        RunConfig(**{"budget": 10, "seed": 0, **kw})


def test_pretrain_arms_switches():
    assert ARMS["A18"]["pretrain"] and ARMS["A18"]["pairs"] == "unfiltered" and not ARMS["A18"].get("fgm")
    assert ARMS["A19"]["pretrain"] and ARMS["A19"]["fgm"]
