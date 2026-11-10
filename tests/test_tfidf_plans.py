"""A0 (TF-IDF) end to end, the runner's CPU path, and the size of each plan. No torch needed."""
import json
from collections import Counter

import pandas as pd
import pytest

from gainsplit import plans
from gainsplit.config import RunConfig
from gainsplit.runner import collect, run_all
from gainsplit.tfidf import run_tfidf


def test_a0_writes_the_same_files_as_other_arms(fake_data, tmp_path):
    out = tmp_path / "r"
    rec = run_tfidf(RunConfig("A0", 10, 0, eval_test=True), fake_data, out)
    run_dir = out / "runs" / "A0_p10_s0"
    assert set(rec["scores"]) == {"dev", "test", "isarc_test"}
    assert rec["steps"] == 0 and rec["n_labeled"] == 20
    for split in rec["scores"]:
        preds = pd.read_csv(run_dir / f"preds_{split}.csv")
        assert list(preds.columns) == ["id", "label", "p_sarcastic"]
        assert preds["p_sarcastic"].between(0, 1).all()
    assert json.loads((run_dir / "result.json").read_text())["run_id"] == "A0_p10_s0"


def test_a0_is_deterministic_and_skips_finished_runs(fake_data, tmp_path, capsys):
    for name in ("a", "b"):
        run_tfidf(RunConfig("A0", 10, 0), fake_data, tmp_path / name)
    pa, pb = (pd.read_csv(tmp_path / n / "runs" / "A0_p10_s0" / "preds_dev.csv") for n in ("a", "b"))
    assert pa.equals(pb)
    run_tfidf(RunConfig("A0", 10, 0), fake_data, tmp_path / "a")
    assert "already done" in capsys.readouterr().out


def test_a0_without_eval_test_never_touches_test(fake_data, tmp_path):
    rec = run_tfidf(RunConfig("A0", 10, 0, eval_test=False), fake_data, tmp_path / "r")
    assert set(rec["scores"]) == {"dev"}


def test_runner_routes_a0_without_a_gpu_and_collect_reads_it(fake_data, tmp_path):
    cfgs = [RunConfig("A0", b, s, eval_test=True) for b in (10, 30) for s in (0, 1)]
    table = run_all(cfgs, fake_data, tmp_path / "r", n_gpus=0)  # no subprocess, no torch
    assert sorted(table["run_id"]) == ["A0_p10_s0", "A0_p10_s1", "A0_p30_s0", "A0_p30_s1"]
    assert {"test_macro_f1", "fgm_epsilon", "minutes", "steps"} <= set(table.columns)
    run_all(cfgs, fake_data, tmp_path / "r", n_gpus=0)  # second call: everything already done
    assert len(collect(tmp_path / "r")) == 4


# ---- plans ----------------------------------------------------------------------
def _ids(cfgs):
    ids = [c.run_id for c in cfgs]
    assert len(ids) == len(set(ids)), "two runs share a run_id"
    return ids


def test_fgm_tuning_plan():
    runs = plans.fgm_tuning()
    _ids(runs)
    assert len(runs) == 9
    assert {r.arm for r in runs} == {"A7"} and {r.budget for r in runs} == {10}
    assert {r.fgm_epsilon for r in runs} == {0.5, 1.0, 2.0} and {r.seed for r in runs} == {0, 1, 2}
    assert not any(r.eval_test for r in runs)


def test_pass2a_plan_size_and_content():
    runs = plans.pass2a()
    _ids(runs)
    assert len(runs) == 104
    by = Counter((r.arm, r.budget) for r in runs if r.eval_test)
    for arm in ("A1", "A3", "A4", "A5"):
        assert by[(arm, 5)] == by[(arm, 30)] == 5
        assert (arm, 1) not in by and (arm, 10) not in by  # 1% and 10% came from Pass 1
    for arm in ("A2", "A6", "A8"):
        assert by[(arm, 1)] == by[(arm, 10)] == 5 and (arm, 5) not in by and (arm, 30) not in by
    for b in (1, 5, 10, 30, 100):
        assert by[("A0", b)] == 5
    a7 = [r for r in runs if r.arm == "A7"]
    assert len(a7) == 9 and not any(r.eval_test for r in a7)
    assert sum(r.arm != "A0" for r in runs) == 79      # GPU runs
    assert {r.seed for r in runs} == {0, 1, 2, 3, 4}


def test_pass2a_helper_weight_and_long_runs_first():
    runs = plans.pass2a()
    for r in runs:
        if r.arm in ("A2", "A3", "A5", "A8"):
            assert r.helper_weight == plans.PASS1_HELPER_WEIGHT
        else:
            assert r.helper_weight == 0.0
    assert all(r.budget == 30 for r in runs[:20])      # 30% runs start first
    assert all(r.arm == "A0" for r in runs[-25:])      # A0 last (the runner puts it on the CPU)


def test_pass2b_needs_epsilon_first(monkeypatch):
    monkeypatch.setattr(plans, "PASS2_FGM_EPSILON", None)
    with pytest.raises(ValueError, match="PASS2_FGM_EPSILON"):
        plans.pass2b()
    monkeypatch.setattr(plans, "PASS2_FGM_EPSILON", 1.0)
    runs = plans.pass2b()
    _ids(runs)
    assert len(runs) == 10 and {r.budget for r in runs} == {1, 10} and all(r.eval_test for r in runs)
    assert {r.fgm_epsilon for r in runs} == {1.0}


def test_every_registered_plan_builds_with_unique_run_ids(monkeypatch):
    monkeypatch.setattr(plans, "PASS2_FGM_EPSILON", 1.0)
    for build in plans.PLANS.values():
        _ids(build())
