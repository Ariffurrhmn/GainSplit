"""Checks for the judge-contest v2 helpers (no GPU, no network, made-up text only)."""
import numpy as np
import pandas as pd
import pytest

from gainsplit import judge as jd


def _train(n_per_class=60):
    return pd.DataFrame([{"id": i, "text": f"headline {i}", "label": i % 2} for i in range(2 * n_per_class)])


def test_real_sets_disjoint_balanced_repeatable():
    shots, tune, rep = jd.pick_real_sets(_train(), 40, 40, 3, seed=7)
    assert shots["label"].value_counts().to_dict() == {0: 3, 1: 3}
    assert tune["label"].value_counts().to_dict() == {0: 20, 1: 20}
    assert rep["label"].value_counts().to_dict() == {0: 20, 1: 20}
    ids = [set(d["id"]) for d in (shots, tune, rep)]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
    again = jd.pick_real_sets(_train(), 40, 40, 3, seed=7)
    assert all(a.equals(b) for a, b in zip((shots, tune, rep), again))
    with pytest.raises(ValueError):
        jd.pick_real_sets(_train(10), 40, 40, 3, seed=7)


def test_prompts():
    shots, _, _ = jd.pick_real_sets(_train(), 4, 4, 2, seed=1)
    v1 = jd.build_prompt("area man wins", "v1")
    v2 = jd.build_prompt("area man wins", "v2", shots)
    assert "area man wins" in v1 and "area man wins" in v2
    assert "lowercase" in v2 and "Answer: SARCASTIC" in v2 and "Answer: SERIOUS" in v2
    with pytest.raises(ValueError):
        jd.build_prompt("x", "v2")


def test_threshold_fixes_a_shifted_but_well_ranked_judge():
    y = np.array([0] * 50 + [1] * 50)
    s = np.concatenate([np.linspace(2, 4, 50), np.linspace(5, 7, 50)])   # all "sarcastic" at 0, but ranked right
    assert jd.balanced_accuracy_at(y, s, 0.0) == 0.5
    thr, bacc = jd.best_threshold(y, s)
    assert 4 < thr < 5 and bacc == 1.0
    assert jd.auc(y, s) == 1.0
    assert jd.auc(y, np.zeros(100)) == 0.5


def test_evaluate_uses_tune_threshold_on_report():
    y = np.array([0, 0, 1, 1])
    res = jd.evaluate(y, [0, 1, 2, 3], y, [0, 0, 0, 0])
    assert 1 < res["threshold"] < 2 and res["tune_bacc"] == 1.0
    assert res["report_bacc"] == 0.5 and res["report_sarcastic_rate"] == 0.0


def test_choose_judges_never_self_and_respects_bar():
    results = {"qwen|v2": {"tune_bacc": 0.9, "report_bacc": 0.8, "base": "qwen"},
               "smollm3|v2": {"tune_bacc": 0.8, "report_bacc": 0.78, "base": "smollm3"},
               "gemma|v1": {"tune_bacc": 0.95, "report_bacc": 0.6, "base": "gemma"}}
    j = jd.choose_judges(results, ["qwen", "gemma", "mistral"], 0.75)
    assert j == {"qwen": "smollm3|v2", "gemma": "qwen|v2", "mistral": "qwen|v2"}
    assert jd.choose_judges(results, ["qwen"], 0.99) == {"qwen": None}


def test_filter_pairs():
    rows = pd.DataFrame({"id": ["1_s", "1_n", "2_s", "2_n", "3_s", "3_n"], "seed_id": [1, 1, 2, 2, 3, 3],
                         "model": ["qwen", "qwen", "qwen", "qwen", "gemma", "gemma"],
                         "text": list("abcdef"), "label": [1, 0, 1, 0, 1, 0]})
    scores = pd.DataFrame({"id": ["1_s", "1_n", "2_s", "2_n"], "candidate": "j", "score": [2.0, -1.0, -3.0, -1.0]})
    filt, pairs = jd.filter_pairs(rows, scores, {"qwen": "j", "gemma": None}, {"j": {"threshold": 0.0}})
    assert sorted(filt["seed_id"].unique()) == [1, 3]          # pair 2 fails; gemma has no judge, kept
    assert list(filt.columns) == ["id", "seed_id", "model", "text", "label"]
    p = pairs.set_index("seed_id")
    assert not p.at[3, "judged"] and p.at[1, "judged"] and not p.at[2, "kept"]
    with pytest.raises(ValueError):
        jd.filter_pairs(rows, scores.iloc[:2], {"qwen": "j", "gemma": None}, {"j": {"threshold": 0.0}})
