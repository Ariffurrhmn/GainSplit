"""Pass 11: the seed recipe on the MLM encoder (A37), the pass11 plans, the extra 70% budgets and the analysis.
Torch tests use a tiny random BERT and made-up data; they prove the pieces fit together, not real accuracy."""
import json

import numpy as np
import pandas as pd
import pytest

from gainsplit import plans
from gainsplit.config import ARMS, RunConfig


# ---------------------------------------------------------------- no torch needed

def test_a37_is_the_seed_recipe_on_the_mlm_encoder():
    assert ARMS["A37"] == {"gan": True, "helpers": True, "init": "mlm"}
    assert ARMS["A5"] == {"gan": True, "helpers": True}                 # A5 is the same on plain BERT
    with pytest.raises(ValueError, match="init_dir"):
        RunConfig("A37", 1, 0, helper_weight=0.3)
    RunConfig("A37", 1, 0, helper_weight=0.3, init_dir="/x")


def test_pass11_plans():
    allr = plans.pass11_all("/enc")
    assert len(allr) == 70
    ids = [r.run_id for r in allr]
    assert len(ids) == len(set(ids))
    assert all(r.eval_test and not r.eval_masked for r in allr)
    counts = pd.Series([(r.arm, r.budget, r.tag) for r in allr]).value_counts()
    assert (counts == 5).all() or set(counts) == {5}                    # 5 seeds per cell
    cells = set(counts.index)
    assert cells == {("A5", 70, ""), ("A7", 70, ""), ("A1", 70, ""), ("A20", 70, ""), ("A36", 30, ""), ("A36", 5, ""),
                     ("A4", 1, "long35"), ("A1", 1, "long35"), ("A37", 10, ""), ("A37", 1, ""),
                     *[("A20", 1, t) for t in plans.PASS11_DOSE_STEPS]}
    long_ = [r for r in allr if r.tag == "long35"]
    assert all(r.step_mult == 35 and r.steps(228) == 7000 for r in long_)
    for r in allr:
        assert r.helper_weight == (plans.PASS1_HELPER_WEIGHT if ARMS[r.arm]["helpers"] else 0.0)
        assert r.fgm_epsilon == (plans.PASS2_FGM_EPSILON if ARMS[r.arm].get("fgm") else 0.0)
        assert bool(r.init_dir) == bool(ARMS[r.arm].get("init"))
    dose = [r for r in allr if r.tag in plans.PASS11_DOSE_STEPS]
    assert all(r.init_dir == plans.PASS11_CKPT[r.tag] for r in dose)
    older = {r.run_id for n in plans.PLANS if not n.startswith("pass11") for r in plans.PLANS[n]()}
    assert not (set(ids) & older)                                      # no run id repeats an earlier pass
    for n in ("pass11_high", "pass11_ours", "pass11_ganlong", "pass11_dose", "pass11_a37"):
        assert {r.run_id for r in plans.PLANS[n]()} <= set(ids)
    assert [r.arm for r in allr[:2]] == ["A5", "A5"]                   # longest runs first


# ---------------------------------------------------------------- extra budgets

def _stratified_budgets(data_dir):
    """Rewrite the fake dataset's budgets.json with the real procedure (the conftest one is not stratified)."""
    from gainsplit.data import nested_budget_ids
    train = pd.read_csv(data_dir / "headlines_train.csv", keep_default_na=False)
    b = {str(s): nested_budget_ids(train, s, [1, 5, 10, 30, 100]) for s in range(5)}
    (data_dir / "budgets.json").write_text(json.dumps(b))
    return train, b


def test_extra_budgets_extend_the_frozen_ones(fake_data, monkeypatch, tmp_path):
    from gainsplit.data import EXTRA_BUDGETS_ENV, budget_ids, load_splits, make_extra_budgets
    train, frozen = _stratified_budgets(fake_data)
    extra = make_extra_budgets(fake_data, (70,))
    for s in range(5):
        ids = set(extra[str(s)]["70"])
        assert set(frozen[str(s)]["30"]) < ids < set(frozen[str(s)]["100"])
        assert abs(len(ids) - 0.7 * len(train)) <= 2
    path = tmp_path / "extra.json"
    path.write_text(json.dumps(extra))
    with pytest.raises(KeyError, match=EXTRA_BUDGETS_ENV):
        load_splits(fake_data, 70, 0)                                  # no file set: refuse, do not guess
    monkeypatch.setenv(EXTRA_BUDGETS_ENV, str(path))
    sp = load_splits(fake_data, 70, 0)
    assert set(sp.labeled["id"]) == set(extra["0"]["70"])
    assert len(sp.labeled) + len(sp.unlabeled) == len(train) and "label" not in sp.unlabeled
    assert budget_ids(fake_data, 30, 0) == frozen["0"]["30"]            # frozen budgets are read from budgets.json


def test_extra_budgets_refuse_a_procedure_that_does_not_reproduce_the_frozen_file(fake_data):
    from gainsplit.data import make_extra_budgets
    # the conftest budgets.json is one global permutation, not per-class: the 30% check must catch it
    with pytest.raises(ValueError, match="30% set"):
        make_extra_budgets(fake_data, (70,))
    _stratified_budgets(fake_data)
    with pytest.raises(ValueError, match="already in budgets.json"):
        make_extra_budgets(fake_data, (30,))


# ---------------------------------------------------------------- analysis

def _write_run(root, arm, budget, seed, tag, f1, acc, rng):
    rid = f"{arm}_p{budget}_s{seed}" + (f"_{tag}" if tag else "")
    d = root / "runs" / rid
    d.mkdir(parents=True)
    n = 400
    y = np.tile([0, 1], n // 2)
    p = np.clip(np.where(y == 1, 0.5 + f1 - 0.5, 0.5 - (f1 - 0.5)) + rng.normal(0, 0.2, n), 0.01, 0.99)
    pd.DataFrame({"id": np.arange(n), "label": y, "p_sarcastic": p}).to_csv(d / "preds_test.csv", index=False)
    (d / "result.json").write_text(json.dumps({
        "run_id": rid, "config": {"arm": arm, "budget": budget, "seed": seed, "tag": tag},
        "scores": {"test": {"macro_f1": f1, "accuracy": acc, "f1_sarcastic": f1, "roc_auc": 0.9}}}))


def test_analysis_runs_on_stand_in_results(tmp_path):
    from gainsplit import analysis_pass11 as a11
    rng = np.random.default_rng(0)
    cells = {("A1", 1, ""): .80, ("A5", 1, ""): .78, ("A36", 1, ""): .89, ("A4", 1, ""): .80,
             ("A1", 10, ""): .88, ("A5", 10, ""): .88, ("A36", 10, ""): .92,
             ("A1", 5, ""): .86, ("A5", 5, ""): .86, ("A36", 5, ""): .91, ("A4", 5, ""): .86,
             ("A1", 30, ""): .90, ("A5", 30, ""): .90, ("A36", 30, ""): .92, ("A4", 30, ""): .90,
             ("A1", 70, ""): .934, ("A5", 70, ""): .935, ("A7", 70, ""): .938, ("A20", 70, ""): .935,
             ("A4", 1, "long35"): .80, ("A1", 1, "long35"): .80,
             ("A20", 1, ""): .88, ("A20", 1, "dose100"): .83, ("A20", 1, "dose350"): .85,
             ("A20", 1, "dose1000"): .87, ("A20", 1, "dose3000"): .88,
             ("A37", 1, ""): .88, ("A37", 10, ""): .91, ("A20", 10, ""): .91}
    for (arm, b, tag), f1 in cells.items():
        for s in range(3):
            _write_run(tmp_path, arm, b, s, tag, f1 + rng.normal(0, .002), f1 + .001, rng)
    t = a11.run_all([tmp_path], tmp_path / "out", n=300)
    v = t["verdicts"]
    assert len(v) == len(a11.HYPOTHESES)
    assert not v["verdict"].eq("no runs (skipped)").any()               # every predicted contrast was computed
    assert t["dose_summary"]["H11.5_rises_within_0.5"].all()
    assert (tmp_path / "out" / "verdicts.csv").exists()
    row = v[(v["hypothesis"] == "H11.18") & (v["budget"] == 1)].iloc[0]
    assert row["mean_diff"] > 5 and row["verdict"] == "SUPPORTED"


def test_specs_are_balanced_and_name_real_arms():
    from gainsplit import analysis_pass11 as a11
    for specs in (a11.OURS_SPECS, a11.SEED_PARTS_SPECS, a11.HIGH_SPECS, a11.GANLONG_SPECS, a11.DOSE_SPECS, a11.A37_SPECS):
        for _, _, w in specs:
            assert sum(w.values()) == pytest.approx(0)
            assert all(arm in ARMS for arm, _ in w)
    names = {(n, b) for specs in (a11.OURS_SPECS, a11.SEED_PARTS_SPECS, a11.HIGH_SPECS, a11.GANLONG_SPECS,
                                  a11.DOSE_SPECS, a11.A37_SPECS) for n, b, _ in specs}
    for _, _, name, b, _, _ in a11.HYPOTHESES:
        assert (name, b) in names, name


def test_verdict_directions():
    from gainsplit.analysis_pass11 import verdict11
    assert verdict11(2.0, 0.001, "+") == "SUPPORTED"
    assert verdict11(-2.3, 0.001, "not+").startswith("SUPPORTED")       # significantly worse still "not better"
    assert verdict11(1.0, 0.001, "not+").startswith("not supported")
    assert verdict11(0.1, 0.5, "0").startswith("SUPPORTED")


# ---------------------------------------------------------------- torch

try:
    import torch  # noqa: F401
    import transformers  # noqa: F401
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False
torch_only = pytest.mark.skipif(not HAVE_TORCH, reason="needs torch and transformers (run on Kaggle)")

if HAVE_TORCH:
    from test_train_smoke import tiny_bert  # noqa: F401,E402  (fixture)
    from test_pass10 import tiny_mlm  # noqa: F401,E402  (fixture)


@torch_only
def test_a37_trains_a_gan_from_the_mlm_encoder(fake_data, tiny_bert, tiny_mlm, tmp_path):
    from gainsplit.train import train_run
    cfg = RunConfig("A37", 10, 0, model_name=tiny_bert, max_len=16, batch_size=8, passes=1, min_steps=5, fp16=False,
                    eval_test=True, helper_weight=0.3, init_dir=tiny_mlm)
    rec = train_run(cfg, fake_data, tmp_path / "r")
    assert rec["init"]["objective"] == "mlm" and rec["uses_u"]
    last = rec["train_log"][-1]
    assert {"loss_sarc", "loss_unsup_fake", "loss_sent", "loss_emo", "loss_g"} <= set(last)
    assert all(np.isfinite(v) for v in last.values())


@torch_only
def test_step_mult_makes_a_long_run_with_the_same_settings(fake_data, tiny_bert, tmp_path):
    from gainsplit.train import train_run
    base = dict(model_name=tiny_bert, max_len=16, batch_size=8, passes=1, min_steps=5, fp16=False, eval_test=True)
    a = train_run(RunConfig("A1", 10, 0, **base), fake_data, tmp_path / "r")
    b = train_run(RunConfig("A1", 10, 0, tag="long", step_mult=3, **base), fake_data, tmp_path / "r")
    assert b["steps"] == 3 * a["steps"]


# ---------------------------------------------------------------- whole-dataset scoring (notebook 23)

def test_pass11_full_plan_and_guard():
    full = plans.pass11_full()
    assert len(full) == 16
    assert [(r.arm, r.budget, r.tag) for r in full[::5][:3]] == [("A5", 70, "full"), ("A1", 100, "full"), ("A1", 70, "full")]
    assert all(r.eval_train and r.eval_test for r in full if r.tag == "full")
    rerun = [r for r in full if r.tag == "rerun11"]
    assert len(rerun) == 1 and (rerun[0].arm, rerun[0].budget, rerun[0].seed) == ("A1", 1, 0) and not rerun[0].eval_train
    ids = [r.run_id for r in full]
    older = {r.run_id for n in plans.PLANS if n != "pass11_full" for r in plans.PLANS[n]()}
    assert len(ids) == len(set(ids)) and not (set(ids) & older)
    with pytest.raises(ValueError, match="eval_test"):
        RunConfig("A1", 10, 0, eval_train=True)
    with pytest.raises(ValueError, match="Headlines"):
        RunConfig("A1", 100, 0, eval_train=True, eval_test=True, train_data="isarcasm")


def test_run_scores_splits_l_u_test_and_all():
    from gainsplit.analysis_pass11 import run_scores
    # train rows 0-9 (0-5 labeled); the model is right on every labeled row, wrong on half of U; dev/test 4 rows each
    train = pd.DataFrame({"id": range(10), "label": [1, 0] * 5, "p_sarcastic": [0.9, 0.1] * 3 + [0.9, 0.9, 0.1, 0.1]})
    dev = pd.DataFrame({"id": range(10, 14), "label": [1, 1, 0, 0], "p_sarcastic": [0.9, 0.1, 0.1, 0.9]})
    test = pd.DataFrame({"id": range(14, 18), "label": [1, 1, 0, 0], "p_sarcastic": [0.9, 0.9, 0.9, 0.1]})
    s = run_scores(train, dev, test, [0, 1, 2, 3, 4, 5])
    assert s["acc_L"] == pytest.approx(100.0) and s["acc_U"] == pytest.approx(50.0)
    assert s["acc_train"] == pytest.approx(80.0) and s["acc_dev"] == pytest.approx(50.0) and s["acc_test"] == pytest.approx(75.0)
    assert s["acc_all"] == pytest.approx(100 * (8 + 2 + 3) / 18) and s["n_all"] == 18


def test_rerun_check_flags_a_difference(tmp_path):
    from gainsplit.analysis_pass11 import rerun_check
    rows = []
    for name, tag, shift in (("orig", "", 0.0), ("same", "full", 0.0), ("other", "rerun11", 0.01)):
        d = tmp_path / name
        d.mkdir()
        pd.DataFrame({"id": [1, 2, 3], "label": [0, 1, 1], "p_sarcastic": [0.2 + shift, 0.7, 0.9]}).to_csv(d / "preds_test.csv", index=False)
        rows.append({"run_id": name, "arm": "A1", "budget": 1, "seed": 0, "tag": tag, "run_dir": str(d)})
    t = rerun_check(pd.DataFrame(rows)).set_index("run_id")
    assert t.loc["same", "status"] == "PASS" and t.loc["other", "status"] == "FAIL"


@torch_only
def test_eval_train_saves_every_training_row_and_changes_nothing_else(fake_data, tiny_bert, tmp_path):
    from gainsplit.train import train_run
    base = dict(model_name=tiny_bert, max_len=16, batch_size=8, passes=1, min_steps=5, fp16=False, eval_test=True)
    a = train_run(RunConfig("A1", 10, 0, **base), fake_data, tmp_path / "r")
    b = train_run(RunConfig("A1", 10, 0, tag="full", eval_train=True, **base), fake_data, tmp_path / "r")
    n_train = len(pd.read_csv(fake_data / "headlines_train.csv"))
    ptr = pd.read_csv(tmp_path / "r" / "runs" / b["run_id"] / "preds_train.csv")
    assert len(ptr) == n_train and "train" in b["scores"] and "train" not in a["scores"]
    pa = pd.read_csv(tmp_path / "r" / "runs" / a["run_id"] / "preds_test.csv")
    pb = pd.read_csv(tmp_path / "r" / "runs" / b["run_id"] / "preds_test.csv")
    assert pa.equals(pb)                                               # same training, same test probabilities
