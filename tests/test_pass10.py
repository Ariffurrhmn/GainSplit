"""Pass 10: the 2x2x2 stack on the MLM encoder (A31-A36), the pass10 plans, the encoder finder, the saved
pseudo-labels and the analysis. Torch tests use a tiny random BERT and made-up data; they prove the pieces
fit together and say nothing about real accuracy."""
import json

import numpy as np
import pandas as pd
import pytest

from gainsplit import plans
from gainsplit.config import ARMS, RunConfig


def _cell(arm):
    sw = ARMS[arm]
    return (bool(sw["helpers"]), bool(sw.get("fgm")), bool(sw.get("pseudo")))


# ---------------------------------------------------------------- no torch needed

def test_pass10_arms_are_the_2x2x2_on_the_mlm_encoder():
    from gainsplit.analysis_pass10 import ON_BERT
    mlm = ("A20", "A31", "A21", "A32", "A33", "A34", "A35", "A36")
    assert len({_cell(a) for a in mlm}) == 8
    for mlm_arm, bert_arm in ON_BERT.items():
        assert _cell(mlm_arm) == _cell(bert_arm), (mlm_arm, bert_arm)    # same ingredients, other encoder
        assert ARMS[mlm_arm]["init"] == "mlm" and not ARMS[bert_arm].get("init")
        assert ARMS[mlm_arm].get("l_only", False) == ARMS[bert_arm].get("l_only", False)
    with pytest.raises(ValueError, match="init_dir"):
        RunConfig("A32", 1, 0)
    with pytest.raises(ValueError, match="fgm_epsilon"):
        RunConfig("A35", 1, 0, init_dir="/x")
    RunConfig("A36", 1, 0, init_dir="/x", helper_weight=0.3, fgm_epsilon=0.5)


def test_pass10_plans():
    stack = plans.pass10_stack("/enc")
    main = [r for r in stack if not r.tag]
    assert len(stack) == 62 and len(main) == 60
    assert {(r.arm, r.budget) for r in main} == {(a, b) for a in plans.PASS10_STACK_ARMS for b in (1, 10)}
    assert all(r.init_dir == "/enc" and r.eval_test and r.eval_masked for r in stack)
    for r in stack:
        assert r.helper_weight == (plans.PASS1_HELPER_WEIGHT if ARMS[r.arm]["helpers"] else 0.0)
        assert r.fgm_epsilon == (plans.PASS2_FGM_EPSILON if ARMS[r.arm].get("fgm") else 0.0)
    assert main[0].arm == "A36" and main[0].budget == 10        # longest first
    seeds = plans.pass10_seeds()
    assert len(seeds) == 20 and {r.tag for r in seeds} == {"pt1", "pt2"}
    assert all(r.arm == "A20" and r.init_dir == plans.PASS10_CKPT[r.tag] for r in seeds)
    assert {r.budget for r in plans.pass10_curve()} == {5, 30} and len(plans.pass10_curve()) == 10
    assert [r.run_id for r in plans.pass10_curve_100()] == [f"A20_p100_s{s}" for s in range(5)]
    assert {r.tag for r in plans.pass10_long()} == {"pt20k"}
    ids = [r.run_id for n in plans.PLANS if n.startswith("pass10") for r in plans.PLANS[n]()]
    older = {r.run_id for n in plans.PLANS if not n.startswith("pass10") for r in plans.PLANS[n]()}
    assert len(ids) == len(set(ids)) and not (set(ids) & older)


def test_reruns_match_pass8a_settings_except_the_encoder_path():
    orig = {r.run_id: r for r in plans.pass8_mlm()}
    reruns = [r for r in plans.pass10_stack("/enc") if r.tag == "rerun"]
    assert [r.run_id for r in reruns] == ["A20_p1_s0_rerun", "A21_p1_s0_rerun"]
    for r in reruns:
        a, b = r.to_dict(), orig[r.run_id.replace("_rerun", "")].to_dict()
        for d in (a, b):
            d.pop("tag"), d.pop("init_dir")
        assert a == b


def test_analysis_specs_are_balanced_and_name_real_arms():
    from gainsplit import analysis_pass10 as a10
    for specs in (a10.STACK_F1_SPECS, a10.STACK_INTERACTION_SPECS, a10.SEEDS_SPECS, a10.CURVE_SPECS, a10.LONG_SPECS):
        for _, _, w in specs:
            assert sum(w.values()) == pytest.approx(0)
            assert all(arm.split("@")[0] in ARMS for arm, _ in w)
    names = {(n, b) for specs in (a10.STACK_F1_SPECS, a10.STACK_INTERACTION_SPECS, a10.SEEDS_SPECS, a10.CURVE_SPECS)
             for n, b, _ in specs}
    names |= {(f"gap({k[0]}{'[' + k[1] + ']' if k[1] else ''}) - gap({ref})", b) for k, b, ref in a10.GAP_SPECS}
    names |= {(f"AUC {k[0]} - word count", b) for k, b in a10.LENGTH_SPECS}
    for _, table, name, b, _, _ in a10.HYPOTHESES:
        assert (name, b) in names, name
        assert any(table in t for t in a10.PART_TABLES.values())


def test_matched_budget_and_curve():
    from gainsplit.analysis_pass10 import _matched_budget, curve, with_budget_aliases
    a1 = {1: 80.0, 10: 88.0, 100: 93.0}
    assert _matched_budget(a1, 80.0) == pytest.approx(1.0)
    assert _matched_budget(a1, 84.0) == pytest.approx(10 ** 0.5)         # halfway in log10
    assert _matched_budget(a1, 95.0) == float("inf") and _matched_budget(a1, 70.0) is None
    runs = pd.DataFrame([{"arm": a, "tag": "", "budget": b, "seed": s, "test_macro_f1": v / 100}
                         for a, vals in (("A1", a1), ("A20", {1: 88.0, 10: 91.0, 100: 93.0}))
                         for b, v in vals.items() for s in range(2)])
    t = curve(runs).set_index("budget")
    assert t.loc[1, "A1_budget_matched"] == pytest.approx(10.0) and t.loc[1, "label_factor"] == pytest.approx(10.0)
    al = with_budget_aliases(runs)
    moved = al[al["arm"] == "A1@10"]
    assert set(moved["budget"]) == {1} and len(moved) == 2


def test_seed_spread():
    from gainsplit.analysis_pass10 import seed_spread
    runs = pd.DataFrame([{"arm": "A20", "tag": t, "budget": 1, "seed": s, "test_macro_f1": v}
                         for t, v in (("", 0.88), ("pt1", 0.87), ("pt2", 0.86)) for s in range(3)])
    r = seed_spread(runs).iloc[0]
    assert r["n_encoders"] == 3 and r["range"] == pytest.approx(2.0) and r["mean"] == pytest.approx(87.0)


def test_pseudo_accuracy_is_scored_from_saved_ids(tmp_path):
    from gainsplit.analysis_pass10 import pseudo_health
    d = tmp_path / "A32_p1_s0"
    d.mkdir()
    (d / "result.json").write_text(json.dumps({"pseudo": {"teacher_recipe": "mlm:A1", "teacher_dev_macro_f1": 0.87,
                                                          "n_kept": 4}}))
    pd.DataFrame({"id": [10, 11, 12, 13], "pseudo_label": [1, 0, 1, 1]}).to_csv(d / "pseudo_labels.csv", index=False)
    runs = pd.DataFrame([{"arm": "A32", "budget": 1, "seed": 0, "tag": "", "run_dir": str(d)}])
    truth = pd.DataFrame({"id": [10, 11, 12, 13, 14], "label": [1, 0, 0, 1, 0]})
    t = pseudo_health(runs, truth)
    assert t.loc[0, "pseudo_accuracy"] == pytest.approx(75.0) and t.loc[0, "teacher_recipe"] == "mlm:A1"


def test_find_encoder_checks_the_receipt(tmp_path):
    from gainsplit.ssl_pretrain import _sha256_dir, find_encoder
    enc = tmp_path / "input" / "nb15" / "ckpt" / "mlm"
    enc.mkdir(parents=True)
    (enc / "config.json").write_text("{}")
    (enc / "model.safetensors").write_bytes(b"weights")
    (enc / "pretrain_info.json").write_text(json.dumps({"objective": "mlm", "config": {"seed": 0, "steps": 10000}}))
    receipt = _sha256_dir(enc)
    assert find_encoder(tmp_path / "input", "mlm", 0, 10000, receipt) == enc
    with pytest.raises(FileNotFoundError):
        find_encoder(tmp_path / "input", "mlm", 1, 10000, receipt)          # wrong pretraining seed
    with pytest.raises(FileNotFoundError):
        find_encoder(tmp_path / "input", "jepa", 0, 10000, receipt)
    with pytest.raises(ValueError, match="differ"):
        find_encoder(tmp_path / "input", "mlm", 0, 10000, {**receipt, "model.safetensors": "0" * 64})


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


@pytest.fixture
def tiny_mlm(fake_data, tiny_bert, tmp_path):
    from gainsplit.ssl_pretrain import PretrainConfig, pretrain
    ck = tmp_path / "ck_mlm"
    pretrain(PretrainConfig("mlm", steps=4, batch_size=8, model_name=tiny_bert, max_len=16, fp16=False, log_every=2),
             fake_data, ck, probe=False)
    return str(ck)


def _cfg(tiny_bert, arm, init_dir, **kw):
    sw = ARMS[arm]
    base = dict(budget=10, seed=0, model_name=tiny_bert, max_len=16, batch_size=8, passes=1, min_steps=5,
                fp16=False, eval_test=True, helper_weight=0.3 if sw["helpers"] else 0.0,
                fgm_epsilon=0.5 if sw.get("fgm") else 0.0, init_dir=init_dir)
    return RunConfig(arm, **{**base, **kw})


@torch_only
@pytest.mark.parametrize("arm,keys,recipe", [
    ("A31", {"loss_sarc", "loss_sent", "loss_emo"}, None),
    ("A32", {"loss_sarc"}, "mlm:A1"),
    ("A33", {"loss_sarc", "loss_adv", "loss_sent", "loss_emo"}, None),
    ("A34", {"loss_sarc", "loss_sent", "loss_emo"}, "mlm:A2"),
    ("A35", {"loss_sarc", "loss_adv"}, "mlm:A7"),
    ("A36", {"loss_sarc", "loss_adv", "loss_sent", "loss_emo"}, "mlm:A2+A7"),
])
def test_stack_arms_train_from_the_mlm_encoder(fake_data, tiny_bert, tiny_mlm, tmp_path, arm, keys, recipe):
    from gainsplit.data import load_splits
    from gainsplit.train import train_run
    rec = train_run(_cfg(tiny_bert, arm, tiny_mlm), fake_data, tmp_path / "r")
    assert rec["init"]["objective"] == "mlm"
    assert keys <= set(rec["train_log"][-1])
    assert all(np.isfinite(v) for v in rec["train_log"][-1].values())
    saved = tmp_path / "r" / "runs" / rec["run_id"] / "pseudo_labels.csv"
    if recipe:
        assert rec["pseudo"]["teacher_recipe"] == recipe
        kept = pd.read_csv(saved)
        assert len(kept) == rec["pseudo"]["n_kept"]
        u_ids = set(load_splits(fake_data, 10, 0).unlabeled["id"])
        assert set(kept["id"]) <= u_ids                           # pseudo-labels only ever on U
    else:
        assert rec["pseudo"] is None and not saved.exists()


@torch_only
def test_self_training_teacher_starts_from_the_encoder(fake_data, tiny_bert, tiny_mlm, tmp_path):
    """A32's teacher is A20 (same seed, same steps), so its dev score equals A20's."""
    from gainsplit.train import train_run
    a20 = train_run(_cfg(tiny_bert, "A20", tiny_mlm), fake_data, tmp_path / "r")
    a32 = train_run(_cfg(tiny_bert, "A32", tiny_mlm), fake_data, tmp_path / "r")
    assert a32["pseudo"]["teacher_dev_macro_f1"] == round(a20["scores"]["dev"]["macro_f1"], 4)


@torch_only
def test_a6_still_records_its_old_teacher_and_saves_pseudo_ids(fake_data, tiny_bert, tmp_path):
    from gainsplit.train import train_run
    rec = train_run(RunConfig("A6", 10, 0, model_name=tiny_bert, max_len=16, batch_size=8, passes=1, min_steps=5,
                              fp16=False, eval_test=True), fake_data, tmp_path / "r")
    assert rec["pseudo"]["teacher_recipe"] == "A1"
    assert (tmp_path / "r" / "runs" / rec["run_id"] / "pseudo_labels.csv").exists()
