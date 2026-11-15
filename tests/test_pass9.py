"""Pass 9: the stacked arms A27-A30 (helper heads on L, FGM, self-training), the pass9 plans and the
analysis specs. Torch tests use a tiny random BERT and made-up data; they prove the pieces fit together
and say nothing about real accuracy."""
import numpy as np
import pandas as pd
import pytest

from gainsplit import plans
from gainsplit.config import ARMS, RunConfig


# ---------------------------------------------------------------- no torch needed

def test_stack_arms_are_the_missing_cells_of_the_2x2x2():
    def key(arm):
        sw = ARMS[arm]
        return (bool(sw["helpers"]), bool(sw.get("fgm")), bool(sw.get("pseudo")))
    cells = {key(a): a for a in ("A1", "A2", "A6", "A7", "A27", "A28", "A29", "A30")}
    assert len(cells) == 8
    for arm in ("A27", "A28", "A30"):
        assert ARMS[arm]["l_only"] and not ARMS[arm]["gan"]   # helpers on L only, as in A2
    with pytest.raises(ValueError, match="fgm_epsilon"):
        RunConfig("A27", 10, 0, helper_weight=0.3)
    RunConfig("A29", 1, 0, fgm_epsilon=0.5)


def test_pass9_plans():
    p = plans.pass9()
    assert len(p) == 43
    main = [r for r in p if not r.tag]
    assert {(r.arm, r.budget) for r in main} == {(a, b) for a in plans.PASS9_ARMS for b in (1, 10)}
    assert all(r.eval_test for r in p)
    for r in p:
        assert r.helper_weight == (plans.PASS1_HELPER_WEIGHT if ARMS[r.arm]["helpers"] else 0.0)
        assert r.fgm_epsilon == (plans.PASS2_FGM_EPSILON if ARMS[r.arm].get("fgm") else 0.0)
    assert [r.run_id for r in p if r.tag] == ["A2_p1_s0_rerun", "A6_p1_s0_rerun", "A7_p1_s0_rerun"]
    assert main[0].arm == "A30" and main[0].budget == 10   # longest first
    assert [r.run_id for r in plans.pass9_100()] == [f"A27_p100_s{s}" for s in range(5)]
    ids = [r.run_id for name in ("pass9", "pass9_100") for r in plans.PLANS[name]()]
    older = {r.run_id for name in plans.PLANS if not name.startswith("pass9") for r in plans.PLANS[name]()}
    assert len(ids) == len(set(ids)) and not (set(ids) & older)


def test_rerun_settings_match_the_original_runs():
    """A re-run is only a check if its settings equal the original run's settings."""
    orig = {r.run_id: r for r in plans.pass2a() + plans.pass2b()}
    for r in plans.pass9():
        if r.tag == "rerun":
            o = orig[r.run_id.replace("_rerun", "")]
            a, b = r.to_dict(), o.to_dict()
            a.pop("tag"), b.pop("tag")
            assert a == b


def test_analysis_specs_are_balanced_and_name_real_arms():
    from gainsplit import analysis_pass9 as a9
    for specs in (a9.F1_SPECS, a9.INTERACTION_SPECS):
        for _, _, w in specs:
            assert sum(w.values()) == 0 and all(arm in ARMS for arm, _ in w)
    names = {(n, b) for n, b, _ in a9.F1_SPECS} | {(n, b) for n, b, _ in a9.INTERACTION_SPECS}
    for _, _, name, b, _, _ in a9.HYPOTHESES:
        assert (name, b) in names


def test_verdict_rules():
    from gainsplit.analysis_pass9 import verdict
    assert verdict(1.2, 0.01, "+") == "SUPPORTED"
    assert verdict(-1.2, 0.01, "+") == "opposite, significant"
    assert verdict(1.2, 0.20, "+").startswith("not supported")
    assert verdict(-0.8, 0.01, "-") == "SUPPORTED"
    assert verdict(0.3, 0.60, "0").startswith("SUPPORTED")
    assert verdict(0.9, 0.01, "0").startswith("not supported")
    assert verdict(0.9, 0.01, None) == "significant difference"


def test_additivity_reading_aid():
    from gainsplit.analysis_pass9 import additivity
    m = {"A1": 80, "A2": 81.5, "A7": 81.6, "A6": 81.6, "A27": 82.0, "A28": 83.0, "A29": 83.2, "A30": 83.5}
    runs = pd.DataFrame([{"arm": a, "tag": "", "budget": 1, "seed": s, "test_macro_f1": v / 100}
                         for a, v in m.items() for s in range(2)])
    t = additivity(runs).set_index("arm")
    assert t.loc["A27", "gain_over_bert"] == pytest.approx(2.0)
    assert t.loc["A27", "sum_of_single_gains"] == pytest.approx(3.1)
    assert t.loc["A30", "sum_of_single_gains"] == pytest.approx(4.7)


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


def _cfg(tiny_bert, arm, **kw):
    sw = ARMS[arm]
    base = dict(budget=10, seed=0, model_name=tiny_bert, max_len=16, batch_size=8, passes=1, min_steps=5,
                fp16=False, eval_test=True, helper_weight=0.3 if sw["helpers"] else 0.0,
                fgm_epsilon=0.5 if sw.get("fgm") else 0.0)
    return RunConfig(arm, **{**base, **kw})


@torch_only
@pytest.mark.parametrize("arm,keys,recipe", [
    ("A27", {"loss_sarc", "loss_adv", "loss_sent", "loss_emo"}, None),
    ("A28", {"loss_sarc", "loss_sent", "loss_emo"}, "A2"),
    ("A29", {"loss_sarc", "loss_adv"}, "A7"),
    ("A30", {"loss_sarc", "loss_adv", "loss_sent", "loss_emo"}, "A2+A7"),
])
def test_stack_arms_train_and_record_their_teacher(fake_data, tiny_bert, tmp_path, arm, keys, recipe):
    from gainsplit.train import train_run
    rec = train_run(_cfg(tiny_bert, arm), fake_data, tmp_path / "r")
    assert keys <= set(rec["train_log"][-1])
    assert all(np.isfinite(v) for v in rec["train_log"][-1].values())
    if recipe:
        assert rec["pseudo"]["teacher_recipe"] == recipe
    else:
        assert rec["pseudo"] is None and rec["uses_u"] is False
    for split in ("dev", "test", "isarc_test"):
        assert 0 <= rec["scores"][split]["macro_f1"] <= 1


@torch_only
def test_a6_teacher_is_still_plain_bert(fake_data, tiny_bert, tmp_path):
    from gainsplit.train import train_run
    rec = train_run(_cfg(tiny_bert, "A6"), fake_data, tmp_path / "r")
    assert rec["pseudo"]["teacher_recipe"] == "A1"


@torch_only
@pytest.mark.parametrize("use_fgm", [False, True])
def test_helper_losses_stay_on_l_rows_when_a_labeled_pool_joins(fake_data, tiny_bert, monkeypatch, use_fgm):
    """A28/A30: pseudo-labeled rows join the sarcasm loss, but the helper heads see the L rows only."""
    from transformers import AutoTokenizer
    from gainsplit import train as tr
    from gainsplit.data import load_helper_labels, load_splits
    cfg = _cfg(tiny_bert, "A30" if use_fgm else "A28", min_steps=3)
    s = load_splits(fake_data, 10, 0)
    labeled, unlabeled = load_helper_labels(fake_data, s.labeled, s.unlabeled)
    pool = unlabeled.iloc[:8].reset_index(drop=True)
    pool["label"] = 1
    seen = []
    real = tr.helper_losses
    monkeypatch.setattr(tr, "helper_losses", lambda m, h, extra, v="": (seen.append(h.shape[0]), real(m, h, extra, v))[1])
    tok = AutoTokenizer.from_pretrained(tiny_bert)
    tr.fit(cfg, tok, labeled, pool, 3, torch.device("cpu"), False, helpers=True, fgm=use_fgm, helpers_on_pool=False)
    assert seen and all(n == cfg.batch_size for n in seen)        # L batch only, not L + pool
    seen.clear()
    tr.fit(cfg, tok, labeled, pool, 3, torch.device("cpu"), False, helpers=True, fgm=use_fgm, helpers_on_pool=True)
    assert all(n == 2 * cfg.batch_size for n in seen)             # the old behaviour, for comparison


@torch_only
def test_ingredient_arms_unchanged(fake_data, tiny_bert, tmp_path):
    """A2, A6 and A7 still train deterministically and keep their old training-curve keys."""
    from gainsplit.train import train_run
    for arm, keys in (("A2", {"loss_sarc", "loss_sent", "loss_emo", "loss"}), ("A6", {"loss_sarc", "loss"}),
                      ("A7", {"loss_sarc", "loss_adv", "loss"})):
        a = train_run(_cfg(tiny_bert, arm), fake_data, tmp_path / "a")
        b = train_run(_cfg(tiny_bert, arm), fake_data, tmp_path / "b")
        assert a["scores"] == b["scores"] and set(a["train_log"][-1]) - {"step"} == keys
