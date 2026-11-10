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
        RunConfig("A9", 10, 0)
