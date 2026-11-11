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
