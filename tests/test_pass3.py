"""Pass 3 options: config validation, plan, helper-label variants, iSarcasmEval split, masked test sets.

Pandas only (no torch needed)."""
import numpy as np
import pandas as pd
import pytest

from gainsplit import plans
from gainsplit.config import ARMS, RunConfig
from gainsplit.data import (EMO_CLASSES, SENT_CLASSES, SENT_P, EMO_P, add_soft_helper_targets, apply_helper_variant,
                            flip_helper_labels, helper_label_counts, load_helper_labels, load_isarcasm_splits,
                            load_masked_tests, load_splits, permute_helper_labels, verify, verify_pass3)


# ---- config ---------------------------------------------------------------------
def test_new_fields_default_so_old_run_ids_do_not_change():
    c = RunConfig("A3", 10, 0, helper_weight=0.3)
    assert (c.helper_variant, c.eval_masked, c.train_data) == ("", False, "headlines")
    assert c.run_id == "A3_p10_s0"


def test_a9_is_a2_with_shuffled_labels():
    assert ARMS["A9"] == {"gan": False, "helpers": True, "l_only": True, "shuffle": True}
    assert RunConfig("A9", 1, 0, helper_weight=0.3, tag="shuf").run_id == "A9_p1_s0_shuf"


@pytest.mark.parametrize("kw", [
    dict(arm="A1", helper_variant="soft"),                       # variant needs a helper arm
    dict(arm="A0", helper_variant="flip25"),
    dict(arm="A4", helper_variant="vader"),
    dict(arm="A3", helper_variant="nonsense"),
    dict(arm="A3", train_data="somewhere"),
    dict(arm="A3", train_data="isarcasm", budget=100),           # isarcasm is A1 only
    dict(arm="A1", train_data="isarcasm", budget=10),            # ... and budget 100 only
])
def test_bad_combinations_are_rejected(kw):
    with pytest.raises(ValueError):
        RunConfig(**{"budget": 10, "seed": 0, **kw})


def test_good_combinations_are_accepted():
    RunConfig("A3", 10, 0, helper_weight=0.3, helper_variant="emoonly")
    RunConfig("A2", 1, 0, helper_weight=0.3, helper_variant="flip50")
    RunConfig("A1", 100, 0, train_data="isarcasm")


# ---- plan -----------------------------------------------------------------------
def test_pass3_plan_size_and_content():
    runs = plans.pass3()
    ids = [r.run_id for r in runs]
    assert len(runs) == 105 and len(set(ids)) == 105
    assert all(r.eval_test and r.eval_masked and r.tag for r in runs)
    assert {r.seed for r in runs} == {0, 1, 2, 3, 4}
    old = {r.run_id for build in (plans.pass1, plans.pass2a, plans.pass2b) for r in build()}
    assert not old & set(ids)
    for v in plans.PASS3_VARIANTS:
        a3 = [r for r in runs if r.arm == "A3" and r.helper_variant == v]
        a2 = [r for r in runs if r.arm == "A2" and r.helper_variant == v]
        assert len(a3) == len(a2) == 5 and {r.budget for r in a3} == {10} and {r.budget for r in a2} == {1}
        assert {r.tag for r in a3 + a2} == {v}
    assert sum(r.arm == "A9" for r in runs) == 5 and {r.budget for r in runs if r.arm == "A9"} == {1}
    mask = [r for r in runs if r.tag == "mask"]
    assert len(mask) == 25 and {r.arm for r in mask} == {"A1", "A3", "A4", "A6", "A7"} and {r.budget for r in mask} == {10}
    assert all(r.helper_variant == "" and r.train_data == "headlines" for r in mask)
    assert all(r.fgm_epsilon == plans.PASS2_FGM_EPSILON for r in mask if r.arm == "A7")
    iso = [r for r in runs if r.train_data == "isarcasm"]
    assert len(iso) == 5 and all(r.arm == "A1" and r.budget == 100 and r.tag == "isarctrain" for r in iso)
    assert all(r.helper_weight == plans.PASS1_HELPER_WEIGHT for r in runs if ARMS[r.arm]["helpers"])


def test_mask_runs_repeat_the_earlier_runs_except_for_tag_and_masked_scoring():
    earlier = {(r.arm, r.budget, r.seed): r.to_dict() for build in (plans.pass1, plans.pass2a, plans.pass2b) for r in build()
               if r.eval_test}
    for r in plans.pass3():
        if r.tag != "mask":
            continue
        d, e = r.to_dict(), earlier[(r.arm, r.budget, r.seed)]
        for k in ("tag", "eval_masked"):
            d.pop(k), e.pop(k)
        assert d == e


def test_pass3_longest_runs_first():
    runs = plans.pass3()
    assert runs[0].arm == "A6"
    assert all(r.budget == 1 for r in runs[-40:])  # the 1% runs (A2 variants, A9) are the shortest, so they go last


def test_pass3_is_registered_and_older_plans_are_unchanged(monkeypatch):
    assert plans.PLANS["pass3"] is plans.pass3
    expected = {"lr_tuning": 9, "w_tuning": 9, "gan_pilot": 3, "session2": 12, "pass1": 45,
                "fgm_tuning": 9, "pass2a": 104, "pass2b": 10}
    for name, n in expected.items():
        runs = plans.PLANS[name]()
        assert len(runs) == n
        assert all(r.helper_variant == "" and not r.eval_masked and r.train_data == "headlines" for r in runs)
    assert plans.pass1()[0].run_id == "A1_p1_s0" and plans.pass2b()[0].run_id == "A7_p1_s0"


# ---- data: helper-label variants ---------------------------------------------------
def _helpers(fake_data, budget=10, seed=0):
    s = load_splits(fake_data, budget, seed)
    return load_helper_labels(fake_data, s.labeled, s.unlabeled)


@pytest.mark.parametrize("variant,share", [("flip25", 0.25), ("flip50", 0.5)])
def test_flip_changes_exactly_the_share_and_never_keeps_the_class(fake_data, variant, share):
    lab, unl = _helpers(fake_data)
    new_l, new_u = apply_helper_variant(variant, fake_data, lab, unl, seed=0)
    for col in ("sent", "emo"):
        before = np.concatenate([lab[col], unl[col]])
        after = np.concatenate([new_l[col], new_u[col]])
        assert (before != after).sum() == round(share * len(before))   # exactly the stated share, no more
        assert after.min() >= 0 and after.max() < (3 if col == "sent" else 5)
    assert (new_l["id"] == lab["id"]).all() and len(new_l) == len(lab) and len(new_u) == len(unl)


def test_flip_is_reproducible_depends_on_seed_and_leaves_inputs_alone(fake_data):
    lab, unl = _helpers(fake_data)
    keep = lab.copy()
    a = flip_helper_labels(lab, unl, 0.25, seed=0)
    b = flip_helper_labels(lab, unl, 0.25, seed=0)
    c = flip_helper_labels(lab, unl, 0.25, seed=1)
    assert a[0].equals(b[0]) and a[1].equals(b[1])
    assert not a[0].equals(c[0]) or not a[1].equals(c[1])
    assert lab.equals(keep)
    # sentiment and emotion are flipped independently: the changed rows differ between the two columns
    hit_sent = np.concatenate([a[0]["sent"] != lab["sent"], a[1]["sent"] != unl["sent"]])
    hit_emo = np.concatenate([a[0]["emo"] != lab["emo"], a[1]["emo"] != unl["emo"]])
    assert not (hit_sent == hit_emo).all()


def test_flip_spreads_over_the_other_classes_uniformly(fake_data):
    lab, unl = _helpers(fake_data, budget=30)
    new_l, new_u = flip_helper_labels(lab, unl, 0.5, seed=0)
    before, after = np.concatenate([lab["emo"], unl["emo"]]), np.concatenate([new_l["emo"], new_u["emo"]])
    shift = (after - before)[after != before] % 5
    counts = np.bincount(shift, minlength=5)[1:]
    assert counts.min() > 0.5 * counts.mean()   # every "other class" offset is used, roughly evenly


@pytest.mark.parametrize("name", ["vader", "teacher2"])
def test_alternative_sentiment_replaces_sentiment_only(pass3_data, name):
    lab, unl = _helpers(pass3_data)
    new_l, new_u = apply_helper_variant(name, pass3_data, lab, unl, seed=0)
    alt = pd.read_csv(pass3_data / "helper_sent_alt.csv").set_index("id")[f"{name}_label"]
    for old, new in ((lab, new_l), (unl, new_u)):
        assert [SENT_CLASSES[i] for i in new["sent"]] == list(new["id"].map(alt))
        assert new["sent"].dtype.kind == "i"
        assert new["emo"].equals(old["emo"])            # emotion keeps the original teacher label
    assert not new_l["sent"].equals(lab["sent"])


def test_alt_sentiment_fails_on_missing_ids(pass3_data):
    alt = pd.read_csv(pass3_data / "helper_sent_alt.csv")
    alt.iloc[1:].to_csv(pass3_data / "helper_sent_alt.csv", index=False)
    lab, unl = _helpers(pass3_data)
    with pytest.raises(ValueError, match="missing"):
        apply_helper_variant("vader", pass3_data, lab, unl, seed=0)


def test_soft_adds_the_teacher_probabilities_and_keeps_hard_labels(fake_data):
    lab, unl = _helpers(fake_data)
    new_l, new_u = apply_helper_variant("soft", fake_data, lab, unl, seed=0)
    assert np.allclose(new_l[SENT_P].sum(axis=1), 1) and np.allclose(new_u[EMO_P].sum(axis=1), 1)
    assert new_l["sent"].equals(lab["sent"]) and new_u["emo"].equals(unl["emo"])
    raw = pd.read_csv(fake_data / "helper_labels.csv").set_index("id")
    row = new_l.iloc[0]
    assert row[SENT_P[0]] == pytest.approx(raw.loc[row["id"], SENT_P[0]])


def test_sentonly_and_emoonly_leave_the_labels_alone(fake_data):
    lab, unl = _helpers(fake_data)
    for v in ("sentonly", "emoonly", ""):
        out = apply_helper_variant(v, fake_data, lab, unl, seed=0)
        assert out[0].equals(lab) and out[1].equals(unl)


def test_helper_label_counts_add_up(fake_data):
    lab, unl = _helpers(fake_data)
    counts = helper_label_counts(lab, unl)
    assert sum(counts["sent"].values()) == sum(counts["emo"].values()) == 200
    assert set(counts["sent"]) == set(SENT_CLASSES) and set(counts["emo"]) == set(EMO_CLASSES)
    new_l, new_u = permute_helper_labels(lab, unl, 0)   # shuffling keeps the class proportions
    assert helper_label_counts(new_l, new_u) == counts


# ---- data: iSarcasmEval training set ---------------------------------------------------
def test_isarcasm_split_is_stratified_fixed_and_disjoint(pass3_data):
    s = load_isarcasm_splits(pass3_data)
    assert len(s.dev) == 10 and len(s.labeled) == 90 and len(s.unlabeled) == 0
    assert s.dev["label"].value_counts().to_dict() == {0: 7, 1: 3}              # 10% of each class
    assert s.labeled["label"].value_counts().to_dict() == {0: 63, 1: 27}
    assert not set(s.dev["id"]) & set(s.labeled["id"])
    assert set(s.dev["id"]) | set(s.labeled["id"]) == set(pd.read_csv(pass3_data / "isarcasm_train.csv")["id"])
    again = load_isarcasm_splits(pass3_data)
    assert again.dev.equals(s.dev) and again.labeled.equals(s.labeled)           # no run seed involved
    assert "label" not in s.unlabeled.columns
    assert s.test.equals(pd.read_csv(pass3_data / "headlines_test.csv"))          # reverse transfer set
    assert s.isarc_test.equals(pd.read_csv(pass3_data / "isarcasm_test.csv"))


# ---- data: masked test sets and the Pass 3 manifest --------------------------------------
def test_masked_tests_keep_ids_and_labels(pass3_data):
    m = load_masked_tests(pass3_data)
    assert set(m) == {"test_cuemask", "test_randmask"}
    test = pd.read_csv(pass3_data / "headlines_test.csv")
    for df in m.values():
        assert df["id"].tolist() == test["id"].tolist() and "[MASK]" in df["text"].iloc[0]


def test_masked_tests_must_match_the_test_rows(pass3_data):
    df = pd.read_csv(pass3_data / "headlines_test_cuemask.csv")
    df.iloc[::-1].to_csv(pass3_data / "headlines_test_cuemask.csv", index=False)
    with pytest.raises(ValueError):
        load_masked_tests(pass3_data)


def test_verify_pass3_passes_then_catches_a_changed_file(pass3_data):
    verify(pass3_data)                    # the old manifest is untouched by the new files
    assert set(verify_pass3(pass3_data)["outputs"]) >= {"helper_sent_alt.csv", "headlines_test_cuemask.csv"}
    with open(pass3_data / "helper_sent_alt.csv", "a") as f:
        f.write("9999,train,positive,0.5,positive,0.1,0.1,0.8\n")
    with pytest.raises(ValueError, match="manifest_pass3"):
        verify_pass3(pass3_data)


def test_old_data_dir_still_verifies_without_pass3_files(fake_data):
    verify(fake_data)                     # no manifest_pass3.json needed by older plans
    with pytest.raises(FileNotFoundError):
        verify_pass3(fake_data)
