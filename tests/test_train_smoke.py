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


# ---- Pass 2 arms: A2, A6, A7, A8 --------------------------------------------------
def _fgm_kw(arm):
    return {"fgm_epsilon": 1.0} if arm == "A7" else {}


def _w(arm):
    return 0.3 if arm in ("A2", "A8") else 0.0


def _count_streams(monkeypatch):
    """Record every BatchStream a run builds (1 = L only, 2 = L + a second batch)."""
    import gainsplit.train as t
    made = []

    class Counting(t.BatchStream):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            made.append(self)

    monkeypatch.setattr(t, "BatchStream", Counting)
    return made


P2_KEYS = {
    "A2": {"loss_sarc", "loss_sent", "loss_emo", "loss"},
    "A6": {"loss_sarc", "loss"},
    "A7": {"loss_sarc", "loss_adv", "loss"},
    "A8": {"loss_sarc", "loss_sent", "loss_emo", "loss"},
}


@pytest.mark.parametrize("arm", ["A2", "A6", "A7", "A8"])
def test_pass2_arms_run_and_log_training_curve(arm, fake_data, tiny_bert, tmp_path):
    cfg = _cfg(tiny_bert, arm=arm, helper_weight=_w(arm), min_steps=7, eval_test=True, **_fgm_kw(arm))
    rec = train_run(cfg, fake_data, tmp_path / "r")
    run_dir = tmp_path / "r" / "runs" / f"{arm}_p10_s0"
    assert rec["steps"] == 7 and rec["n_labeled"] == 20
    assert set(rec["scores"]) == {"dev", "test", "isarc_test"}
    assert (run_dir / "preds_test.csv").exists() and (run_dir / "result.json").exists()
    assert set(rec["train_log"][0]) == P2_KEYS[arm] | {"step"}


@pytest.mark.parametrize("arm", ["A2", "A6", "A7", "A8"])
def test_pass2_arms_same_seed_gives_same_result(arm, fake_data, tiny_bert, tmp_path):
    import pandas as pd
    for name in ("a", "b"):
        train_run(_cfg(tiny_bert, arm=arm, helper_weight=_w(arm), **_fgm_kw(arm)), fake_data, tmp_path / name)
    pa, pb = (pd.read_csv(tmp_path / n / "runs" / f"{arm}_p10_s0" / "preds_dev.csv") for n in ("a", "b"))
    assert (pa["p_sarcastic"] - pb["p_sarcastic"]).abs().max() < 1e-4


@pytest.mark.parametrize("arm", ["A2", "A8"])
def test_helper_control_arms_need_a_positive_helper_weight(arm, fake_data, tiny_bert, tmp_path):
    with pytest.raises(ValueError):
        train_run(_cfg(tiny_bert, arm=arm), fake_data, tmp_path / "r")


def test_a2_uses_no_u(fake_data, tiny_bert, tmp_path, monkeypatch):
    made = _count_streams(monkeypatch)
    rec = train_run(_cfg(tiny_bert, arm="A2", helper_weight=0.3), fake_data, tmp_path / "r")
    assert len(made) == 1 and made[0].n == 20      # only the L stream
    assert rec["uses_u"] is False and rec["n_unlabeled"] == 180
    made.clear()  # A3 on the same data does read U: a second stream over the 180 U rows
    rec3 = train_run(_cfg(tiny_bert, arm="A3", helper_weight=0.3), fake_data, tmp_path / "r3")
    assert sorted(s.n for s in made) == [20, 180] and rec3["uses_u"] is True


def test_a8_permutes_helper_labels_across_l_and_u(fake_data, tiny_bert, tmp_path, monkeypatch):
    import gainsplit.train as t
    from gainsplit.data import load_helper_labels, load_splits
    seen = {}
    real = t.permute_helper_labels

    def spy(labeled, unlabeled, seed):
        out = real(labeled, unlabeled, seed)
        seen["out"], seen["seed"] = out, seed
        return out

    monkeypatch.setattr(t, "permute_helper_labels", spy)
    train_run(_cfg(tiny_bert, arm="A8", helper_weight=0.3), fake_data, tmp_path / "r")
    s = load_splits(fake_data, 10, 0)
    orig = load_helper_labels(fake_data, s.labeled, s.unlabeled)
    assert seen["seed"] == 0
    for col in ("sent", "emo"):
        before = list(orig[0][col]) + list(orig[1][col])
        after = list(seen["out"][0][col]) + list(seen["out"][1][col])
        assert sorted(before) == sorted(after) and before != after


def test_a7_requires_epsilon(tiny_bert):
    with pytest.raises(ValueError):
        _cfg(tiny_bert, arm="A7")                    # fgm_epsilon defaults to 0


def test_a7_epsilon_changes_the_result(fake_data, tiny_bert, tmp_path):
    import pandas as pd
    train_run(_cfg(tiny_bert, arm="A7", fgm_epsilon=0.5, tag="e1"), fake_data, tmp_path / "r")
    train_run(_cfg(tiny_bert, arm="A7", fgm_epsilon=5.0, tag="e2"), fake_data, tmp_path / "r")
    pa, pb = (pd.read_csv(tmp_path / "r" / "runs" / f"A7_p10_s0_{t}" / "preds_dev.csv") for t in ("e1", "e2"))
    assert (pa["p_sarcastic"] - pb["p_sarcastic"]).abs().max() > 1e-6


def test_fgm_step_puts_the_embeddings_back():
    from types import SimpleNamespace
    from gainsplit.train import fgm_step

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = torch.nn.Embedding(10, 4)
            self.encoder = SimpleNamespace(embeddings=SimpleNamespace(word_embeddings=self.emb))
            self.head = torch.nn.Linear(4, 2)
            self.dropout = torch.nn.Identity()

        def encode(self, ids, mask):
            return self.emb(ids).mean(1)

    torch.manual_seed(0)
    model = Toy()
    cfg = RunConfig("A7", 10, 0, fgm_epsilon=1.0, fp16=False)
    opt = torch.optim.SGD(model.parameters(), lr=0.0)   # lr 0: a changed weight could only come from a leaked perturbation
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    before = model.emb.weight.detach().clone()
    ids, mask, y = torch.tensor([[1, 2, 3], [4, 5, 6]]), torch.ones(2, 3), torch.tensor([0, 1])
    stats = fgm_step(model, cfg, opt, sched, scaler, (ids, mask, y, {}), 2, False, torch.device("cpu"))
    assert torch.equal(model.emb.weight.detach(), before)
    assert stats["loss"] == pytest.approx(stats["loss_sarc"] + stats["loss_adv"])
    assert stats["loss_adv"] != stats["loss_sarc"]      # the perturbed pass really ran


def test_pick_pseudo_labels_keeps_confident_items_and_balances_classes():
    import numpy as np
    from gainsplit.train import pick_pseudo_labels
    #            class 0, sure          unsure     class 1, sure
    p = np.array([0.02, 0.05, 0.08, 0.5, 0.6, 0.97, 0.99])
    idx, labels, info = pick_pseudo_labels(p, 0.9)
    assert info["n_confident"] == 5 and info["confident_class_counts"] == {"0": 3, "1": 2}
    assert info["n_kept"] == 4 and info["kept_class_counts"] == {"0": 2, "1": 2}
    assert sorted(idx.tolist()) == [0, 1, 5, 6]  # the two most confident class-0 items, both class-1 items
    assert sorted(labels.tolist()) == [0, 0, 1, 1] and info["fallback_l_only"] is False
    _, _, none = pick_pseudo_labels(np.array([0.4, 0.6, 0.5]), 0.9)
    assert none["n_kept"] == 0 and none["fallback_l_only"] is True


def test_a6_logs_pseudo_label_counts(fake_data, tiny_bert, tmp_path):
    rec = train_run(_cfg(tiny_bert, arm="A6", min_steps=6), fake_data, tmp_path / "r")
    ps = rec["pseudo"]
    assert ps["n_unlabeled"] == 180 and ps["threshold"] == 0.9
    assert ps["n_kept"] == sum(ps["kept_class_counts"].values()) <= ps["n_confident"]
    assert ps["kept_class_counts"]["0"] == ps["kept_class_counts"]["1"]
    assert ps["fallback_l_only"] == (ps["n_kept"] == 0) and rec["uses_u"] == (ps["n_kept"] > 0)
    assert rec["steps"] == 6
    on_disk = json.loads((tmp_path / "r" / "runs" / "A6_p10_s0" / "result.json").read_text())
    assert on_disk["pseudo"] == ps


def test_a6_trains_on_pseudo_labels_when_the_teacher_is_sure(fake_data, tiny_bert, tmp_path, monkeypatch):
    import numpy as np
    import gainsplit.train as t
    # the teacher is certain about every row, alternating class 0 / class 1
    monkeypatch.setattr(t, "predict", lambda m, enc, d, **k: np.arange(enc["input_ids"].shape[0]) % 2 * 0.98 + 0.01)
    made = _count_streams(monkeypatch)
    rec = train_run(_cfg(tiny_bert, arm="A6", min_steps=6), fake_data, tmp_path / "r")
    assert rec["pseudo"]["n_kept"] == 180 and rec["pseudo"]["kept_class_counts"] == {"0": 90, "1": 90}
    assert rec["uses_u"] is True
    # teacher: one stream (L); student: the L stream and the pseudo-label stream
    assert sorted(s.n for s in made) == [20, 20, 180]
    assert any(s.labels is not None and s.n == 180 for s in made)  # the pseudo stream carries labels


def test_a6_falls_back_to_l_only_when_nothing_passes(fake_data, tiny_bert, tmp_path, monkeypatch):
    import numpy as np
    import gainsplit.train as t
    monkeypatch.setattr(t, "predict", lambda m, enc, d, **k: np.full(enc["input_ids"].shape[0], 0.5))
    rec = train_run(_cfg(tiny_bert, arm="A6", min_steps=6), fake_data, tmp_path / "r")
    assert rec["pseudo"]["n_kept"] == 0 and rec["pseudo"]["fallback_l_only"] is True
    assert rec["uses_u"] is False and rec["steps"] == 6
