"""Passes 5-7 analysis (no torch): contrasts, AUC vs word count, stage-1 check, end to end."""
import json

import numpy as np
import pandas as pd
import pytest

from gainsplit import analysis as an
from gainsplit import analysis_pass5_7 as a57
from test_pass4 import _write


def _isarc(y):
    return pd.DataFrame({"id": np.arange(len(y)), "text": ["word " * (i % 7 + 1) for i in range(len(y))],
                         "label": y})


def test_auc_vs_length_is_zero_when_model_is_word_count(tmp_path):
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    isarc = _isarc(y)
    length = isarc["text"].str.split().str.len().to_numpy(float)
    for s in range(2):
        _write(tmp_path, "A15", 10, s, "", y, length / 10)        # same ranking as word count
        _write(tmp_path, "A10", 10, s, "", y, y * 0.6 + rng.random(300) * 0.4)   # much better
    runs = an.load_runs(tmp_path)
    same = a57.auc_vs_length(runs, ("A15", ""), 10, isarc, n=200)
    assert same["mean_diff"] == pytest.approx(0.0, abs=1e-12)
    assert same["model_auc"] == pytest.approx(same["length_auc"])
    better = a57.auc_vs_length(runs, ("A10", ""), 10, isarc, n=200)
    assert better["mean_diff"] > 0.2 and better["ci_low"] > 0 and better["n_seeds"] == 2
    assert a57.auc_vs_length(runs, ("A19", ""), 10, isarc, n=50) is None

    bad = isarc.assign(label=1 - isarc["label"])
    with pytest.raises(ValueError):
        a57.auc_vs_length(runs, ("A10", ""), 10, bad, n=50)


def _add_pretrain(root, run_id, dev_f1, steps=720):
    f = root / "runs" / run_id / "result.json"
    r = json.loads(f.read_text())
    r["pretrain"] = {"steps": steps, "dev": {"macro_f1": dev_f1}}
    f.write_text(json.dumps(r))


def test_check_stage1(tmp_path):
    y = np.array([0, 1] * 20)
    p = np.linspace(0.1, 0.9, 40)
    _write(tmp_path, "A15", 10, 0, "", y, p)        # its dev macro-F1 is 0.8 in the fixture
    _write(tmp_path, "A18", 10, 0, "", y, p)
    _write(tmp_path, "A19", 100, 0, "", y, p)
    _add_pretrain(tmp_path, "A18_p10_s0", 0.8)
    _add_pretrain(tmp_path, "A19_p100_s0", 0.7)
    t = a57.check_stage1(an.load_runs(tmp_path)).set_index("run_id")
    assert bool(t.loc["A18_p10_s0", "ok"]) and not bool(t.loc["A19_p100_s0", "ok"])


def test_pass5_7_analysis_end_to_end(tmp_path):
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    good = np.where(rng.random(300) < 0.9, y, 1 - y) * 0.8 + 0.1
    weak = np.where(rng.random(300) < 0.7, y, 1 - y) * 0.8 + 0.1
    for s in range(3):
        for b in (1, 10):
            _write(tmp_path, "A1", b, s, "", y, weak)
            _write(tmp_path, "A1", b, s, "mask", y, weak)
            _write(tmp_path, "A7", b, s, "", y, weak)
            _write(tmp_path, "A7", b, s, "mask", y, weak)
            _write(tmp_path, "A10", b, s, "weighted", y, good)
            _write(tmp_path, "A19", b, s, "", y, good)
        _write(tmp_path, "A1", 100, s, "", y, weak)
        _write(tmp_path, "A7", 100, s, "", y, good)
    _write(tmp_path, "A1", 100, 0, "rerun", y, weak)
    runs = an.load_runs(tmp_path)

    f1 = a57.contrast_table(runs, n=300)
    row = f1[(f1["contrast"] == "A10[weighted]-A1") & (f1["budget"] == 10)].iloc[0]
    assert row["mean_diff"] > 5 and row["p_holm"] >= row["p"]
    assert set(f1["contrast"]) == {"A10[weighted]-A1", "A19-A7", "A7-A1"}   # specs without runs skipped
    acc = a57.contrast_table(runs, "accuracy", n=300)
    assert list(acc["contrast"]) == ["A7-A1"] and acc.iloc[0]["mean_diff"] > 0

    tables = a57.run_all([tmp_path], tmp_path / "out", n=200, isarc_test=_isarc(y))
    assert tables["rerun_check"]["ok"].all() and len(tables["rerun_check"]) == 13   # 12 mask + 1 rerun
    assert len(tables["gap_contrasts"]) == 4                     # A10[weighted]-A1 and A19-A7 at 1 and 10
    assert len(tables["auc_vs_length"]) == 4                     # A10[weighted] and A19 at 1 and 10
    for name in ("contrasts_f1", "contrasts_accuracy_100", "auc_contrasts", "auc_vs_length", "summary"):
        assert (tmp_path / "out" / f"{name}.csv").exists(), name
