"""Checks for the paired-generation helpers (no GPU, no network, made-up text only)."""
import pandas as pd
import pytest

from gainsplit import pairgen as pg


def _train(n_per_class=20):
    rows = [{"id": i, "text": f"headline number {i}", "label": i % 2} for i in range(2 * n_per_class)]
    return pd.DataFrame(rows)


def test_clean_text_matches_processed_style():
    assert pg.clean_text("Area Man’s \"Big\" Day!  ") == "area man s big day"
    assert pg.clean_text(None) == ""


def test_pick_seeds_is_balanced_and_repeatable():
    a = pg.pick_seeds(_train(), 10, seed=1)
    b = pg.pick_seeds(_train(), 10, seed=1)
    assert list(a.columns) == ["seed_id", "seed_text", "seed_label"]
    assert a["seed_label"].value_counts().to_dict() == {0: 5, 1: 5}
    assert a.equals(b)
    with pytest.raises(ValueError):
        pg.pick_seeds(_train(), 9, seed=1)


def test_assign_models_equal_shares():
    out = pg.assign_models(10, ["a", "b", "c"], seed=0)
    counts = pd.Series(out).value_counts()
    assert len(out) == 10 and counts.max() - counts.min() <= 1
    assert out == pg.assign_models(10, ["a", "b", "c"], seed=0)


def test_prompt_contains_seed_and_json_instruction():
    p = pg.build_prompt("city opens new library")
    assert p.rstrip().endswith("Headline: city opens new library")
    assert '"sarcastic"' in p and '"serious"' in p


def test_parse_pair_handles_fences_and_noise():
    raw = 'Sure!\n```json\n{"sarcastic": "a b c", "serious": "d e f"}\n```'
    assert pg.parse_pair(raw) == ("a b c", "d e f")
    assert pg.parse_pair('{"Sarcastic": "x", "Serious": "y"}') == ("x", "y")
    assert pg.parse_pair("no json here") is None
    assert pg.parse_pair('{"sarcastic": "only one"}') is None
    assert pg.parse_pair(None) is None


def test_refusal_detection():
    assert pg.is_refusal("I'm sorry, but I can't help with that.")
    assert pg.is_refusal("As an AI, I must decline")
    assert not pg.is_refusal('{"sarcastic": "nation sorry state of roads", "serious": "roads in poor state"}')


def test_check_pair_flags():
    ok = pg.check_pair("city opens library", '{"sarcastic": "City opens library, finally!", "serious": "city opens new library"}')
    assert ok["ok"] and ok["sarcastic"] == "city opens library finally"
    copy = pg.check_pair("city opens library", '{"sarcastic": "funny one", "serious": "City opens library"}')
    assert copy["copies_seed"] and not copy["ok"]
    same = pg.check_pair("x", '{"sarcastic": "same text", "serious": "same text"}')
    assert same["same_pair"] and not same["ok"]
    long = pg.check_pair("x", '{"sarcastic": "' + "w " * 40 + '", "serious": "short"}')
    assert long["too_long"] and not long["ok"]
    bad = pg.check_pair("x", "I can't do that.")
    assert bad["refusal"] and not bad["parse_ok"] and not bad["ok"]


def test_cue_rate():
    assert pg.cue_rate(["area man wins", "council meets"], ["area"]) == 0.5
    assert pg.cue_rate([], ["area"]) == 0.0


def test_parse_judge():
    assert pg.parse_judge("SARCASTIC") == 1
    assert pg.parse_judge("serious.") == 0
    assert pg.parse_judge("sarcastic or serious") is None
    assert pg.parse_judge("") is None


def test_blind_audit_hides_labels():
    pairs = pd.DataFrame({"seed_id": [1, 2], "model": ["a", "b"], "sarcastic": ["s1", "s2"], "serious": ["n1", "n2"]})
    items = pg.long_items(pairs)
    assert len(items) == 4 and set(items["intended"]) == {0, 1}
    sheet, key = pg.blind_audit(items, n=3, seed=0)
    assert list(sheet.columns) == ["audit_no", "text", "your_label"]
    assert "intended" in key.columns and len(key) == 3
    assert list(sheet["audit_no"]) == list(key["audit_no"])


def test_plan_seeds_whole_split_and_subset():
    tr = _train(30)
    full = pg.plan_seeds(tr, ["a", "b"], seed=3)
    assert len(full) == 60 and set(full["seed_id"]) == set(tr["id"])
    assert full.equals(pg.plan_seeds(tr, ["a", "b"], seed=3))
    sub = pg.plan_seeds(tr, ["a", "b"], seed=3, n=10)
    assert len(sub) == 10 and sub["model"].value_counts().to_dict() == {"a": 5, "b": 5}


def test_model_chunks_cover_each_seed_once():
    plan = pg.plan_seeds(_train(30), ["a", "b", "c"], seed=0)
    for m in "abc":
        parts = pg.model_chunks(plan, m, size=7)
        ids = pd.concat(parts)["seed_id"]
        assert ids.is_unique and len(ids) == (plan["model"] == m).sum()
        assert all(len(p) <= 7 for p in parts)


def test_find_and_collect_chunks(tmp_path):
    d = tmp_path / "x" / "raw_chunks" / "a"
    d.mkdir(parents=True)
    pd.DataFrame({"seed_id": [1, 2], "seed_text": ["t1", "t2"], "seed_label": [0, 1], "model": "a", "raw": ["r1", "r2"]}).to_csv(d / pg.chunk_name(0), index=False)
    pd.DataFrame({"seed_id": [2, 3], "seed_text": ["t2", "t3"], "seed_label": [1, 0], "model": "a", "raw": ["r2b", "r3"]}).to_csv(d / pg.chunk_name(1), index=False)
    (tmp_path / "other.csv").write_text("x")
    paths = pg.find_chunks(tmp_path)
    assert [p.name for p in paths] == ["chunk_0000.csv", "chunk_0001.csv"]
    raw = pg.collect_raw(paths)
    assert list(raw["seed_id"]) == [1, 2, 3] and raw.loc[raw["seed_id"] == 2, "raw"].item() == "r2"
    assert pg.collect_raw([]).empty


def test_training_rows_only_kept_pairs():
    pairs = pd.DataFrame({"seed_id": [1, 2], "model": ["a", "b"], "sarcastic": ["s1", "s2"],
                          "serious": ["n1", "n2"], "kept": [True, False]})
    rows = pg.training_rows(pairs)
    assert list(rows.columns) == ["id", "seed_id", "model", "text", "label"]
    assert set(rows["id"]) == {"1_s", "1_n"} and set(rows["label"]) == {0, 1}


def test_prompt_version_and_examples():
    assert pg.PROMPT_VERSION == 2
    assert len(pg.FEW_SHOT) == 4 and "obvious" in pg.build_prompt("x")


def test_balanced_accuracy_counts_unclear_as_wrong():
    assert pg.balanced_accuracy([1, 1, 0, 0], [1, 1, 0, 0]) == 1.0
    assert pg.balanced_accuracy([1, 1, 0, 0], [1, None, 0, float("nan")]) == 0.5
    assert pg.balanced_accuracy([1, 0, 0, 0], [1, 1, 1, 1]) == 0.5


def test_cross_verdict_needs_agreement():
    assert pg.cross_verdict([1, 1]) == 1
    assert pg.cross_verdict([0, 0.0]) == 0
    assert pg.cross_verdict([1, 0]) is None
    assert pg.cross_verdict([1, None]) is None
    assert pg.cross_verdict([float("nan"), 1]) is None


def test_pick_judge_threshold():
    assert pg.pick_judge({"a": 0.7, "b": 0.8}, 0.75) == "b"
    assert pg.pick_judge({"a": 0.7}, 0.75) is None
    assert pg.pick_judge({}, 0.75) is None
