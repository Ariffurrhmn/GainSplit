"""Pass 4 training on a tiny random BERT and made-up data: pair arms, pair batches, A15, explanations.

Proves the pieces fit together; says nothing about accuracy. Needs torch (Kaggle has it)."""
import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from gainsplit.config import RunConfig  # noqa: E402
from gainsplit.train import BatchStream, GroupBatchStream, train_run  # noqa: E402
from test_train_smoke import tiny_bert  # noqa: E402,F401  (fixture)


@pytest.fixture(autouse=True)
def no_kaggle_search(monkeypatch, tmp_path):
    monkeypatch.delenv("GAINSPLIT_PAIRS", raising=False)
    monkeypatch.setenv("GAINSPLIT_INPUT", str(tmp_path / "nothing_here"))


def _cfg(tiny_bert, **kw):
    base = dict(arm="A10", budget=10, seed=0, model_name=tiny_bert, max_len=16, batch_size=8,
                passes=1, min_steps=5, fp16=False, pair_batch=8)
    return RunConfig(**{**base, **kw})


def test_group_stream_keeps_pairs_together_and_covers_all():
    n = 20
    enc = {"input_ids": torch.arange(n)[:, None], "attention_mask": torch.ones(n, 1, dtype=torch.long)}
    groups = np.repeat(np.arange(10), 2)
    s = GroupBatchStream(enc, np.tile([0, 1], 10), groups, batch_size=6, seed=0)
    seen = []
    for _ in range(3):                          # 3 batches x 3 pairs = 9 of the 10 pairs (tail dropped)
        ids, _, y, _ = next(s)
        ids = ids[:, 0].tolist()
        assert len(ids) == 6 and y.tolist() == [0, 1] * 3
        assert all(ids[i] // 2 == ids[i + 1] // 2 for i in range(0, 6, 2))   # each pair whole and in order
        seen += ids
    assert len(set(seen)) == 18


def test_group_stream_rejects_unsorted_groups():
    enc = {"input_ids": torch.arange(4)[:, None], "attention_mask": torch.ones(4, 1, dtype=torch.long)}
    with pytest.raises(ValueError, match="next to each other"):
        GroupBatchStream(enc, [0, 1, 0, 1], [0, 1, 0, 1], 4, 0)


@pytest.mark.parametrize("arm,kw,rows", [
    ("A10", {}, 240), ("A12", {}, 120), ("A13", {}, 170), ("A14", {}, 170),
    ("A11", {"fgm_epsilon": 0.5}, 240),
])
def test_pair_arms_train_and_record_the_pairs(pass4_data, tiny_bert, tmp_path, arm, kw, rows):
    rec = train_run(_cfg(tiny_bert, arm=arm, eval_test=True, eval_masked=True, **kw), pass4_data, tmp_path / "r")
    assert rec["pairs"]["n_rows"] == rows and rec["n_labeled"] == 20 and rec["steps"] == 5
    assert not rec["uses_u"]
    assert {"test", "test_cuemask", "test_randmask", "isarc_test"} <= set(rec["scores"])


def test_a15_uses_no_gold_rows_but_the_budget_step_count(pass4_data, tiny_bert, tmp_path):
    rec = train_run(_cfg(tiny_bert, arm="A15", min_steps=7), pass4_data, tmp_path / "r")
    assert rec["n_labeled"] == 0 and rec["steps"] == 7 and rec["pairs"]["pair_set"] == "unfiltered"


def test_a15_ignores_the_gold_labels(pass4_data, tiny_bert, tmp_path):
    """Flipping every gold training label must not change A15 at all."""
    a = train_run(_cfg(tiny_bert, arm="A15"), pass4_data, tmp_path / "a")
    p = pass4_data / "headlines_train.csv"
    t = pd.read_csv(p)
    t["label"] = 1 - t["label"]
    t.to_csv(p, index=False)
    m = json.loads((pass4_data / "manifest.json").read_text())   # keep the hash check happy
    import hashlib
    m["outputs"]["headlines_train.csv"]["sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
    (pass4_data / "manifest.json").write_text(json.dumps(m))
    b = train_run(_cfg(tiny_bert, arm="A15"), pass4_data, tmp_path / "b")
    assert a["scores"]["dev"] == b["scores"]["dev"]


def test_explain_writes_attributions_that_add_up(pass4_data, tiny_bert, tmp_path):
    rec = train_run(_cfg(tiny_bert, arm="A1", eval_test=True, explain=True, tag="explain"), pass4_data, tmp_path / "r")
    e = pd.read_csv(tmp_path / "r" / "runs" / "A1_p10_s0_explain" / "explain.csv", keep_default_na=False)
    assert len(e) == 40                                   # fake test set is smaller than 250 per class: all of it
    assert e["cue_share"].between(0, 1).all()
    assert (e["n_words"] == e["words"].str.split().str.len()).all()
    assert rec["explain"]["n"] == 40
    # completeness: attributions sum to the score difference (midpoint rule, 32 steps)
    rel = e["completeness_gap"].abs() / e["score_diff"].abs().clip(lower=1e-3)
    assert rel.median() < 0.05


def test_a1_path_unchanged_by_pass4_fields(fake_data, tiny_bert, tmp_path):
    """A1 with the new default fields gives the same predictions as before (same seed, same data)."""
    base = dict(arm="A1", budget=10, seed=0, model_name=tiny_bert, max_len=16, batch_size=8,
                passes=1, min_steps=5, fp16=False)
    train_run(RunConfig(**base), fake_data, tmp_path / "a")
    train_run(RunConfig(**base, pair_batch=8), fake_data, tmp_path / "b")   # pair_batch is unused by A1
    pa, pb = (pd.read_csv(tmp_path / n / "runs" / "A1_p10_s0" / "preds_dev.csv") for n in ("a", "b"))
    assert (pa["p_sarcastic"] - pb["p_sarcastic"]).abs().max() == 0


@pytest.mark.parametrize("arm,kw", [("A10", {}), ("A11", {"fgm_epsilon": 0.5})])
def test_pair_weight_one_matches_pass4_exactly(pass4_data, tiny_bert, tmp_path, arm, kw):
    """pair_weight=1.0 (the RunConfig default, Pass 4's behaviour) must give byte-identical
    predictions to not passing pair_weight at all, so every Pass 4 run and re-run check stays
    valid after Pass 5's training-loop change."""
    a = train_run(_cfg(tiny_bert, arm=arm, eval_test=True, **kw), pass4_data, tmp_path / "a")
    b = train_run(_cfg(tiny_bert, arm=arm, eval_test=True, pair_weight=1.0, **kw), pass4_data, tmp_path / "b")
    pa, pb = (pd.read_csv(tmp_path / n / "runs" / f"{arm}_p10_s0" / "preds_test.csv") for n in ("a", "b"))
    assert (pa["p_sarcastic"] - pb["p_sarcastic"]).abs().max() == 0
    assert a["scores"]["test"] == b["scores"]["test"]


@pytest.mark.parametrize("arm,kw", [("A10", {}), ("A11", {"fgm_epsilon": 0.5})])
def test_pair_weight_changes_training(pass4_data, tiny_bert, tmp_path, arm, kw):
    """A lower pair_weight must change the model: dropping the pair rows' loss to near 0 should
    land close to (not necessarily identical to, since the gold batch order is unaffected but the
    gradient magnitudes differ) the A1/A7 no-pairs training the pair rows are being weighted down
    towards, and clearly different from the full-weight (1.0) run."""
    full = train_run(_cfg(tiny_bert, arm=arm, eval_test=True, pair_weight=1.0, **kw), pass4_data, tmp_path / "full")
    light = train_run(_cfg(tiny_bert, arm=arm, eval_test=True, pair_weight=0.01, **kw), pass4_data, tmp_path / "light")
    pf, pl = (pd.read_csv(tmp_path / n / "runs" / f"{arm}_p10_s0" / "preds_test.csv") for n in ("full", "light"))
    assert (pf["p_sarcastic"] - pl["p_sarcastic"]).abs().max() > 1e-6   # different weight -> different model


def test_a16_a17_use_the_cleaned_pair_set_and_respect_pair_weight(pass4_data, tiny_bert, tmp_path):
    rec16 = train_run(_cfg(tiny_bert, arm="A16", eval_test=True, cleaned_drop_pct=20, pair_weight=0.3),
                      pass4_data, tmp_path / "a")
    assert rec16["pairs"]["pair_set"] == "cleaned" and rec16["pairs"]["cleaned_drop_pct"] == 20
    unf_rows = 240   # from the A10 parametrization above: 120 seeds x 2 rows, pass4_data fixture
    assert round(rec16["pairs"]["n_rows"] / unf_rows, 2) == 0.8   # 20% of seeds dropped
    rec17 = train_run(_cfg(tiny_bert, arm="A17", eval_test=True, cleaned_drop_pct=20, pair_weight=0.3,
                           fgm_epsilon=0.5), pass4_data, tmp_path / "b")
    assert rec17["pairs"]["pair_set"] == "cleaned" and rec17["config"]["pair_weight"] == 0.3


# ---------------------------------------------------------------------------- Pass 6: two-stage arms

def test_two_stage_stage1_equals_a15(pass4_data, tiny_bert, tmp_path):
    """Stage 1 of A18 must be exactly A15's training: same seed, same pair stream, same steps.
    A15 at min_steps=5 trains 5 steps; A18 with pretrain_steps=5 must report the same dev scores
    for its stage-1 model."""
    a15 = train_run(_cfg(tiny_bert, arm="A15"), pass4_data, tmp_path / "a15")
    a18 = train_run(_cfg(tiny_bert, arm="A18", pretrain_steps=a15["steps"]), pass4_data, tmp_path / "a18")
    assert a18["pretrain"]["dev"] == a15["scores"]["dev"]
    assert a18["pretrain"]["steps"] == a15["steps"]


def test_two_stage_stage2_trains_on_gold_only(pass4_data, tiny_bert, tmp_path):
    """Stage 2 uses L only, with the usual step count, and changes the stage-1 model."""
    rec = train_run(_cfg(tiny_bert, arm="A18", pretrain_steps=4, min_steps=6, eval_test=True, eval_masked=True),
                    pass4_data, tmp_path / "r")
    assert rec["steps"] == 6 and rec["n_labeled"] == 20 and not rec["uses_u"]
    assert rec["pairs"]["pair_set"] == "unfiltered" and rec["pretrain"]["steps"] == 4
    assert rec["scores"]["dev"] != rec["pretrain"]["dev"]
    assert {"test", "test_cuemask", "test_randmask", "isarc_test"} <= set(rec["scores"])


def test_two_stage_differs_from_a1(pass4_data, tiny_bert, tmp_path):
    """Starting stage 2 from the pair-trained model must give a different model than A1 from scratch."""
    a1 = train_run(_cfg(tiny_bert, arm="A1", eval_test=True), pass4_data, tmp_path / "a1")
    a18 = train_run(_cfg(tiny_bert, arm="A18", pretrain_steps=5, eval_test=True), pass4_data, tmp_path / "a18")
    assert a18["steps"] == a1["steps"]
    p1, p18 = (pd.read_csv(tmp_path / n / "runs" / f"{arm}_p10_s0" / "preds_test.csv")
               for n, arm in (("a1", "A1"), ("a18", "A18")))
    assert (p1["p_sarcastic"] - p18["p_sarcastic"]).abs().max() > 1e-6


def test_a19_two_stage_with_fgm(pass4_data, tiny_bert, tmp_path):
    rec = train_run(_cfg(tiny_bert, arm="A19", pretrain_steps=3, fgm_epsilon=0.5), pass4_data, tmp_path / "r")
    assert rec["pretrain"]["steps"] == 3 and "loss_adv" in rec["train_log"][-1]
