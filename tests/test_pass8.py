"""Pass 8: TAPT (MLM, JEPA) pretraining, the A20-A23 arms on a TAPT encoder, the diffusion arms A24-A26,
the pass8 plans and the analysis specs. Torch tests use a tiny random BERT and made-up data; they prove
the pieces fit together and say nothing about real accuracy."""
import json

import numpy as np
import pytest

from gainsplit import plans
from gainsplit.config import ARMS, RunConfig
from gainsplit.ssl_pretrain import PretrainConfig, sample_mlm_mask, sample_target_mask


# ---------------------------------------------------------------- no torch needed

def test_tapt_arms_need_an_init_dir_and_only_they_take_one():
    with pytest.raises(ValueError, match="init_dir"):
        RunConfig("A20", 10, 0)
    with pytest.raises(ValueError, match="init_dir"):
        RunConfig("A1", 10, 0, init_dir="/x")
    with pytest.raises(ValueError, match="fgm_epsilon"):
        RunConfig("A21", 10, 0, init_dir="/x")
    RunConfig("A21", 10, 0, init_dir="/x", fgm_epsilon=0.5)
    assert ARMS["A22"]["init"] == "jepa" and ARMS["A23"]["fgm"]


def test_consistency_weight_rules():
    with pytest.raises(ValueError, match="cons_weight"):
        RunConfig("A25", 10, 0)
    with pytest.raises(ValueError, match="cons_weight"):
        RunConfig("A24", 10, 0, cons_weight=1.0)
    with pytest.raises(ValueError, match="cons_weight"):
        RunConfig("A1", 10, 0, cons_weight=1.0)
    RunConfig("A26", 1, 0, cons_weight=1.0)


def test_pass8_plans():
    mlm = plans.pass8_mlm()
    assert len(mlm) == 20 and {r.arm for r in mlm} == {"A20", "A21"}
    assert all(r.init_dir == plans.PASS8_CKPT["mlm"] and r.eval_test and r.eval_masked for r in mlm)
    assert {(r.budget, r.arm) for r in mlm} == {(b, a) for b in (1, 10) for a in ("A20", "A21")}
    assert all(r.fgm_epsilon == plans.PASS2_FGM_EPSILON for r in mlm if r.arm == "A21")
    jepa = plans.pass8_jepa("/tmp/j")
    assert len(jepa) == 20 and {r.arm for r in jepa} == {"A22", "A23"} and all(r.init_dir == "/tmp/j" for r in jepa)
    assert [r.run_id for r in plans.pass8_mlm_100()] == [f"A21_p100_s{s}" for s in range(5)]
    assert [r.run_id for r in plans.pass8_jepa_100()] == [f"A23_p100_s{s}" for s in range(5)]
    diff = plans.pass8_diff()
    assert len(diff) == 31 and diff[-1].run_id == "A1_p10_s0_rerun"
    assert all(r.cons_weight == plans.PASS8_CONS_WEIGHT for r in diff if r.arm in ("A25", "A26"))
    ids = [r.run_id for name in plans.PLANS if name.startswith("pass8") for r in plans.PLANS[name]()]
    assert len(ids) == len(set(ids))
    older = {r.run_id for name in plans.PLANS if not name.startswith("pass8") for r in plans.PLANS[name]()}
    assert not (set(ids) & older)


def test_pretrain_config_checks():
    with pytest.raises(ValueError):
        PretrainConfig("bert")
    assert PretrainConfig("jepa").steps == PretrainConfig("mlm").steps  # same budget for both objectives


def test_jepa_mask_hides_words_only_and_leaves_context():
    rng = np.random.default_rng(0)
    attn = np.zeros((50, 16), dtype=np.int64)
    lengths = rng.integers(3, 17, size=50)        # [CLS] + words + [SEP]
    for i, n in enumerate(lengths):
        attn[i, :n] = 1
    m = sample_target_mask(attn, 0.25, 2, rng)
    for i, n in enumerate(lengths):
        assert not m[i, 0] and not m[i, n - 1] and not m[i, n:].any()   # never [CLS], [SEP], padding
        assert m[i].sum() >= 1
        assert m[i].sum() <= max(1, n - 3)                               # at least one word stays visible


def test_mlm_mask_targets_only_real_words():
    rng = np.random.default_rng(0)
    ids = rng.integers(10, 100, size=(40, 12))
    ids[:, 0], ids[:, -1] = 2, 3
    attn = np.ones_like(ids)
    attn[:, 9:] = 0
    new, lab = sample_mlm_mask(ids, attn, 0.15, rng, mask_id=4, vocab_size=100, special_ids={0, 1, 2, 3, 4})
    picked = lab != -100
    assert picked.any(1).all()
    assert not picked[:, 0].any() and not picked[:, 9:].any()
    assert (lab[picked] == ids[picked]).all()
    assert 0.5 < (new[picked] == 4).mean() < 0.95     # about 80% become [MASK]


def test_analysis_specs_name_real_arms():
    from gainsplit import analysis_pass8 as a8
    for sub, specs in a8.F1_SPECS.items():
        for _, budget, w in specs:
            assert sum(w.values()) == 0 and all(arm in ARMS for arm, _ in w)
    assert [s[1] for s in a8.accuracy_specs("mlm")] == [100]
    assert a8.accuracy_specs("diff") == []


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


def _tiny_pre(objective, tiny_bert, **kw):
    base = dict(objective=objective, steps=4, batch_size=8, model_name=tiny_bert, max_len=16, fp16=False,
                log_every=2, pred_heads=2)
    return PretrainConfig(**{**base, **kw})


@torch_only
@pytest.mark.parametrize("objective", ["mlm", "jepa"])
def test_pretrain_saves_a_loadable_encoder_and_info(fake_data, tiny_bert, tmp_path, objective):
    from transformers import AutoModel
    from gainsplit.ssl_pretrain import pretrain
    out = tmp_path / objective
    info = pretrain(_tiny_pre(objective, tiny_bert), fake_data, out)
    assert info["objective"] == objective and len(info["train_log"]) == 2
    assert np.isfinite(info["dev_before"]["loss"]) and np.isfinite(info["dev_after"]["loss"])
    assert "p10" in info["probe_after"]
    assert (out / "pretrain_info.json").exists()
    enc = AutoModel.from_pretrained(out, add_pooling_layer=False)
    base = AutoModel.from_pretrained(tiny_bert, add_pooling_layer=False)
    changed = any(not torch.equal(a, b) for a, b in zip(enc.state_dict().values(), base.state_dict().values()))
    assert changed, "pretraining did not change the encoder"
    if objective == "jepa":
        assert info["collapsed"] is False and info["dev_after"]["target_std"] > 0
    again = pretrain(_tiny_pre(objective, tiny_bert), fake_data, out)    # restart-safe: skipped
    assert again == json.loads((out / "pretrain_info.json").read_text())


@torch_only
def test_pretraining_reads_no_labels(fake_data, tiny_bert, tmp_path):
    """Changing every training label must not change the pretrained weights (texts only)."""
    import pandas as pd
    from gainsplit.data import sha256
    from gainsplit.ssl_pretrain import pretrain
    a = pretrain(_tiny_pre("mlm", tiny_bert), fake_data, tmp_path / "a", probe=False)
    train = pd.read_csv(fake_data / "headlines_train.csv")
    train["label"] = 1 - train["label"]
    train.to_csv(fake_data / "headlines_train.csv", index=False)
    man = json.loads((fake_data / "manifest.json").read_text())
    man["outputs"]["headlines_train.csv"]["sha256"] = sha256(fake_data / "headlines_train.csv")
    (fake_data / "manifest.json").write_text(json.dumps(man))
    b = pretrain(_tiny_pre("mlm", tiny_bert), fake_data, tmp_path / "b", probe=False)
    assert a["weights_sha256"] == b["weights_sha256"]


def _run_cfg(tiny_bert, **kw):
    base = dict(budget=10, seed=0, model_name=tiny_bert, max_len=16, batch_size=8, passes=1, min_steps=5,
                fp16=False, eval_test=True)
    return RunConfig(**{**base, **kw})


@torch_only
def test_tapt_arms_train_from_the_checkpoint_and_record_it(fake_data, tiny_bert, tmp_path):
    from gainsplit.ssl_pretrain import pretrain
    from gainsplit.train import train_run
    ck = tmp_path / "ck_mlm"
    pretrain(_tiny_pre("mlm", tiny_bert), fake_data, ck, probe=False)
    rec = train_run(_run_cfg(tiny_bert, arm="A20", init_dir=str(ck)), fake_data, tmp_path / "r")
    assert rec["init"]["objective"] == "mlm" and rec["uses_u"] is False and rec["n_labeled"] == 20
    rec21 = train_run(_run_cfg(tiny_bert, arm="A21", init_dir=str(ck), fgm_epsilon=0.5), fake_data, tmp_path / "r")
    assert rec21["init"]["weights_sha256"] == rec["init"]["weights_sha256"]
    # the TAPT encoder must actually be used: A20 differs from A1 with the same seed
    a1 = train_run(_run_cfg(tiny_bert, arm="A1"), fake_data, tmp_path / "r")
    assert a1["scores"]["dev"] != rec["scores"]["dev"] or a1["train_log"] != rec["train_log"]
    with pytest.raises(ValueError, match="holds mlm"):
        train_run(_run_cfg(tiny_bert, arm="A22", init_dir=str(ck)), fake_data, tmp_path / "r2")
    with pytest.raises(FileNotFoundError):
        train_run(_run_cfg(tiny_bert, arm="A20", init_dir=str(tmp_path / "missing")), fake_data, tmp_path / "r3")


@torch_only
@pytest.mark.parametrize("arm,kw,keys", [
    ("A24", {}, {"p_fake_fake", "loss_diff", "fake_mean_gap_sd"}),
    ("A25", {"cons_weight": 1.0}, {"loss_cons", "loss_diff", "aug_shift_sd", "flip_rate"}),
    ("A26", {"cons_weight": 1.0}, {"loss_cons", "aug_shift_sd"}),
])
def test_diffusion_arms_train_on_l_and_u(fake_data, tiny_bert, tmp_path, arm, kw, keys):
    from gainsplit.train import train_run
    rec = train_run(_run_cfg(tiny_bert, arm=arm, **kw), fake_data, tmp_path / "r")
    assert rec["uses_u"] is True and rec["diffusion"]["mode"] == ARMS[arm]["diffusion"]
    assert keys <= set(rec["train_log"][-1])
    assert all(np.isfinite(v) for k, v in rec["train_log"][-1].items())
    for split in ("dev", "test", "isarc_test"):
        assert 0 <= rec["scores"][split]["macro_f1"] <= 1
    if arm == "A26":
        assert "loss_diff" not in rec["train_log"][-1]   # no denoiser in the control


@torch_only
def test_diffusion_round_trip_moves_vectors_a_little_and_is_differentiable():
    from gainsplit.diffusion import FeatureDiffusion, noise_only
    torch.manual_seed(0)
    d = FeatureDiffusion(16, width=32)
    x = torch.randn(64, 16) * 3 + 1
    d.update_stats(x)
    x.requires_grad_(True)
    y = d.round_trip(x)
    y.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert d.sample(8, torch.device("cpu")).shape == (8, 16)
    z = noise_only(x.detach(), d.mu, d.sd)
    shift = ((z - x.detach()) / d.sd).pow(2).mean().sqrt()
    assert 0.2 < shift < 1.0   # about 0.5 sd of noise at level 30 of 100
    assert d.loss(x.detach()).item() > 0


@torch_only
def test_older_arms_unchanged_by_pass8_fields(fake_data, tiny_bert, tmp_path):
    """A1, A4 and A7 build the same modules as before Pass 8 (no diffusion parts), so same seed -> same run."""
    from gainsplit.train import train_run
    from gainsplit.models import SarcasmClassifier
    m = SarcasmClassifier(tiny_bert, gan=True)
    assert not hasattr(m, "diff") and m.kplus1
    for arm, kw in (("A1", {}), ("A4", {}), ("A7", {"fgm_epsilon": 0.5})):
        a = train_run(_run_cfg(tiny_bert, arm=arm, **kw), fake_data, tmp_path / "a")
        b = train_run(_run_cfg(tiny_bert, arm=arm, **kw), fake_data, tmp_path / "b")
        assert a["scores"] == b["scores"] and a["init"] is None and a["diffusion"] is None
