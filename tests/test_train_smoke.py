"""End-to-end check of one A1 run on a tiny random BERT and made-up data.

Proves the pieces fit together (load, train, score, save, skip-if-done, runner).
It says nothing about real accuracy. Needs torch and transformers (Kaggle has both).
"""
import json

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from conftest import WORDS  # noqa: E402
from gainsplit.config import RunConfig  # noqa: E402
from gainsplit.runner import run_all  # noqa: E402
from gainsplit.train import train_run  # noqa: E402


@pytest.fixture
def tiny_bert(tmp_path):
    """A 2-layer, 32-wide BERT with random weights and a 20-word vocabulary."""
    from transformers import BertConfig, BertModel, BertTokenizerFast
    d = tmp_path / "tiny_bert"
    d.mkdir()
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + WORDS
    (d / "vocab.txt").write_text("\n".join(vocab))
    BertTokenizerFast(vocab_file=str(d / "vocab.txt")).save_pretrained(d)
    config = BertConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=2,
                        num_attention_heads=2, intermediate_size=64, max_position_embeddings=64)
    BertModel(config).save_pretrained(d)
    return str(d)


def _cfg(tiny_bert, **kw):
    base = dict(arm="A1", budget=10, seed=0, model_name=tiny_bert, max_len=16,
                batch_size=8, passes=1, min_steps=5, fp16=False)
    return RunConfig(**{**base, **kw})


def test_a1_run_writes_scores_and_predictions(fake_data, tiny_bert, tmp_path):
    out = tmp_path / "results"
    rec = train_run(_cfg(tiny_bert, eval_test=True), fake_data, out)
    run_dir = out / "runs" / "A1_p10_s0"
    assert rec["steps"] == 5 and rec["n_labeled"] == 20
    assert set(rec["scores"]) == {"dev", "test", "isarc_test"}
    for split in ("dev", "test", "isarc_test"):
        assert (run_dir / f"preds_{split}.csv").exists()
        assert 0.0 <= rec["scores"][split]["macro_f1"] <= 1.0
    assert json.loads((run_dir / "result.json").read_text())["run_id"] == "A1_p10_s0"


def test_tuning_runs_never_touch_test(fake_data, tiny_bert, tmp_path):
    rec = train_run(_cfg(tiny_bert, eval_test=False), fake_data, tmp_path / "r")
    assert set(rec["scores"]) == {"dev"}


def test_finished_run_is_skipped(fake_data, tiny_bert, tmp_path, capsys):
    out = tmp_path / "r"
    first = train_run(_cfg(tiny_bert), fake_data, out)
    second = train_run(_cfg(tiny_bert), fake_data, out)
    assert "already done" in capsys.readouterr().out
    assert first == second


def test_same_seed_gives_same_result(fake_data, tiny_bert, tmp_path):
    import pandas as pd
    for name in ("a", "b"):
        train_run(_cfg(tiny_bert), fake_data, tmp_path / name)
    pa, pb = (pd.read_csv(tmp_path / n / "runs" / "A1_p10_s0" / "preds_dev.csv") for n in ("a", "b"))
    # GPUs can differ in the last decimals between runs, so allow a tiny tolerance
    assert (pa["p_sarcastic"] - pb["p_sarcastic"]).abs().max() < 1e-4


def test_unwritten_arm_fails_clearly(fake_data, tiny_bert, tmp_path, monkeypatch):
    monkeypatch.setattr("gainsplit.train.IMPLEMENTED_ARMS", {"A1"})
    with pytest.raises(NotImplementedError):
        train_run(_cfg(tiny_bert, arm="A4"), fake_data, tmp_path / "r")


# ---- A3 / A4 / A5 -----------------------------------------------------------------
LOSS_KEYS = {
    "A3": {"loss_sarc", "loss_sent", "loss_emo", "loss"},
    "A4": {"loss_sarc", "loss_unsup_real", "loss_unsup_fake", "p_fake_real", "p_fake_fake",
           "loss_g_unsup", "loss_g_fm", "loss_d", "loss_g", "loss"},
}
LOSS_KEYS["A5"] = LOSS_KEYS["A4"] | {"loss_sent", "loss_emo"}


@pytest.mark.parametrize("arm", ["A3", "A4", "A5"])
def test_new_arms_run_and_log_training_curve(arm, fake_data, tiny_bert, tmp_path):
    cfg = _cfg(tiny_bert, arm=arm, helper_weight=0.3, min_steps=7, eval_test=True)
    rec = train_run(cfg, fake_data, tmp_path / "r")
    run_dir = tmp_path / "r" / "runs" / f"{arm}_p10_s0"
    assert rec["steps"] == 7 and rec["n_unlabeled"] == 180
    assert (run_dir / "preds_dev.csv").exists() and (run_dir / "result.json").exists()
    log = rec["train_log"]
    assert [pt["step"] for pt in log] == [7]  # 7 steps: one (partial) window
    assert set(log[0]) == LOSS_KEYS[arm] | {"step"}


@pytest.mark.parametrize("arm", ["A3", "A4", "A5"])
def test_new_arms_same_seed_gives_same_result(arm, fake_data, tiny_bert, tmp_path):
    import pandas as pd
    for name in ("a", "b"):
        train_run(_cfg(tiny_bert, arm=arm, helper_weight=0.3), fake_data, tmp_path / name)
    pa, pb = (pd.read_csv(tmp_path / n / "runs" / f"{arm}_p10_s0" / "preds_dev.csv") for n in ("a", "b"))
    assert (pa["p_sarcastic"] - pb["p_sarcastic"]).abs().max() < 1e-4


@pytest.mark.parametrize("arm", ["A3", "A5"])
def test_helper_arms_need_a_positive_helper_weight(arm, fake_data, tiny_bert, tmp_path):
    with pytest.raises(ValueError):
        train_run(_cfg(tiny_bert, arm=arm), fake_data, tmp_path / "r")  # helper_weight defaults to 0


def test_u_arms_fall_back_to_l_only_when_u_is_empty(fake_data, tiny_bert, tmp_path):
    # budget 100: every headline is labeled, so U is empty and the run uses L only
    rec = train_run(_cfg(tiny_bert, arm="A5", budget=100, helper_weight=0.3, min_steps=3),
                    fake_data, tmp_path / "r")
    assert rec["n_unlabeled"] == 0 and rec["steps"] >= 3


def test_runner_runs_a_list_and_collects_results(fake_data, tiny_bert, tmp_path):
    cfgs = [_cfg(tiny_bert, lr=lr, tag=f"lr{lr:g}") for lr in (2e-5, 5e-5)]
    table = run_all(cfgs, fake_data, tmp_path / "r", n_gpus=0)
    assert sorted(table["run_id"]) == ["A1_p10_s0_lr2e-05", "A1_p10_s0_lr5e-05"]
    assert "dev_macro_f1" in table.columns
