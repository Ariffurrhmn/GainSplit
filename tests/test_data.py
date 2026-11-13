"""Checks on data loading and the step rule (pandas only, no torch needed)."""
import pandas as pd
import pytest

from gainsplit.config import RunConfig
from gainsplit.data import find_data_dir, load_helper_labels, load_splits, verify


def test_finds_dataset_under_root(fake_data):
    assert find_data_dir(fake_data.parent) == fake_data


def test_verify_passes_then_catches_a_changed_file(fake_data):
    verify(fake_data)
    with open(fake_data / "headlines_dev.csv", "a") as f:
        f.write("999,extra row,1\n")
    with pytest.raises(ValueError):
        verify(fake_data)


def test_labeled_and_unlabeled_split_train_exactly(fake_data):
    s = load_splits(fake_data, budget=10, seed=0)
    assert len(s.labeled) == 20 and len(s.unlabeled) == 180
    assert not set(s.labeled["id"]) & set(s.unlabeled["id"])


def test_unlabeled_pool_has_no_sarcasm_label(fake_data):
    s = load_splits(fake_data, budget=10, seed=0)
    assert "label" not in s.unlabeled.columns
    assert "label" in s.labeled.columns


def test_helper_labels_are_integer_class_indices(fake_data):
    s = load_splits(fake_data, budget=10, seed=0)
    labeled, unlabeled = load_helper_labels(fake_data, s.labeled, s.unlabeled)
    for df in (labeled, unlabeled):
        assert df["sent"].between(0, 2).all() and df["emo"].between(0, 4).all()
    raw = pd.read_csv(fake_data / "helper_labels.csv").set_index("id")
    row = labeled.iloc[0]
    assert ["negative", "neutral", "positive"][row["sent"]] == raw.loc[row["id"], "sent_label"]
    assert ["joy", "sadness", "anger", "fear", "neutral_other"][row["emo"]] == raw.loc[row["id"], "emo_label"]
    assert "label" not in unlabeled.columns


def test_full_budget_leaves_no_unlabeled_pool(fake_data):
    s = load_splits(fake_data, budget=100, seed=1)
    assert len(s.labeled) == 200 and len(s.unlabeled) == 0


def test_different_seeds_pick_different_labeled_sets(fake_data):
    a = set(load_splits(fake_data, 10, 0).labeled["id"])
    b = set(load_splits(fake_data, 10, 1).labeled["id"])
    assert a != b


def test_step_rule_is_the_same_for_every_arm():
    # 10 passes over L, at least 200 steps (Decisions 2026-11-10)
    for arm in ("A1", "A3", "A4", "A5"):
        assert RunConfig(arm, 10, 0).steps(2280) == 720     # 10% budget: 72 batches x 10
        assert RunConfig(arm, 1, 0).steps(228) == 200       # 1% budget: 80 -> floor of 200
        assert RunConfig(arm, 100, 0).steps(22800) == 7130  # 100%: 713 batches x 10


def test_unknown_arm_is_rejected():
    with pytest.raises(ValueError):
        RunConfig("A99", 10, 0)


def test_fixture_has_the_budgets_the_plans_use(fake_data):
    # budgets 1/5/10/30/100 for seeds 0-4, nested inside each other (as in the real budgets.json)
    import json
    b = json.loads((fake_data / "budgets.json").read_text())
    assert sorted(b) == ["0", "1", "2", "3", "4"]
    for seed in b:
        assert sorted(b[seed], key=int) == ["1", "5", "10", "30", "100"]
        sets = [set(b[seed][k]) for k in ("1", "5", "10", "30", "100")]
        assert all(a < c for a, c in zip(sets, sets[1:]))


def _helper_frames(n_l=20, n_u=80):
    # sentiment and emotion start equal (id % 3), so one shared permutation would keep them equal
    ids = list(range(n_l + n_u))
    df = pd.DataFrame({"id": ids, "text": ["x"] * len(ids), "sent": [i % 3 for i in ids], "emo": [i % 3 for i in ids]})
    return df.iloc[:n_l].reset_index(drop=True), df.iloc[n_l:].reset_index(drop=True)


def test_permute_helper_labels_keeps_proportions_but_breaks_the_link():
    from gainsplit.data import permute_helper_labels
    lab, unl = _helper_frames()
    plab, punl = permute_helper_labels(lab, unl, seed=0)
    for col in ("sent", "emo"):
        before = list(lab[col]) + list(unl[col])
        after = list(plab[col]) + list(punl[col])
        assert sorted(before) == sorted(after)   # same class counts overall
        assert before != after                   # but reordered
    assert len(plab) == len(lab) and len(punl) == len(unl)
    assert (plab["id"] == lab["id"]).all() and (punl["text"] == unl["text"]).all()  # headlines untouched


def test_permute_helper_labels_is_seeded_and_independent_per_head():
    from gainsplit.data import permute_helper_labels
    lab, unl = _helper_frames()
    a = permute_helper_labels(lab, unl, 0)
    b = permute_helper_labels(lab, unl, 0)
    c = permute_helper_labels(lab, unl, 1)
    assert a[0].equals(b[0]) and a[1].equals(b[1])
    assert not a[0].equals(c[0])
    assert (a[0]["sent"] != a[0]["emo"]).any() or (a[1]["sent"] != a[1]["emo"]).any()


def test_permute_helper_labels_works_when_u_is_empty():
    from gainsplit.data import permute_helper_labels
    lab, unl = _helper_frames(n_l=30, n_u=0)
    plab, punl = permute_helper_labels(lab, unl, 0)
    assert len(punl) == 0 and sorted(plab["sent"]) == sorted(lab["sent"])


def test_fgm_epsilon_rules():
    with pytest.raises(ValueError):
        RunConfig("A7", 10, 0)                       # A7 needs epsilon > 0
    with pytest.raises(ValueError):
        RunConfig("A7", 10, 0, fgm_epsilon=-1.0)
    with pytest.raises(ValueError):
        RunConfig("A3", 10, 0, fgm_epsilon=1.0)      # other arms must not carry one
    assert RunConfig("A7", 10, 0, fgm_epsilon=1.0, tag="eps1").run_id == "A7_p10_s0_eps1"


def test_step_rule_is_the_same_for_the_new_arms():
    for arm in ("A2", "A6", "A8"):
        assert RunConfig(arm, 10, 0).steps(2280) == 720
    assert RunConfig("A7", 10, 0, fgm_epsilon=1.0).steps(228) == 200
    assert RunConfig("A0", 30, 0).steps(6840) == 2140
