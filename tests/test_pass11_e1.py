"""Pass 11 E1: label-count budgets for iSarcasmEval, MLM pretraining text sources, A1/A20 on iSarcasmEval, the plan and
the analysis. Torch tests use a tiny random BERT and made-up data; they prove the pieces fit together, not real accuracy."""
import json

import numpy as np
import pandas as pd
import pytest

from gainsplit import plans
from gainsplit.config import RunConfig
from gainsplit.data import (ISARC_BUDGETS_ENV, isarc_budget_ids, load_isarcasm_splits, make_isarc_budgets)
from gainsplit.ssl_pretrain import PretrainConfig, pretrain_texts


# ---------------------------------------------------------------- config and plan (no torch)

def test_isarcasm_arms_and_budgets():
    RunConfig("A1", 228, 0, train_data="isarcasm")
    RunConfig("A20", 780, 3, train_data="isarcasm", init_dir="/enc")
    with pytest.raises(ValueError, match="A1 and A20"):
        RunConfig("A3", 100, 0, train_data="isarcasm", helper_weight=0.3)
    with pytest.raises(ValueError, match="init_dir"):
        RunConfig("A20", 228, 0, train_data="isarcasm")


def test_pass11_isarc_plan():
    runs = plans.pass11_isarc("/hs")
    assert len(runs) == 120
    ids = [r.run_id for r in runs]
    assert len(ids) == len(set(ids))
    cells = pd.Series([(r.arm, r.budget, r.tag, r.train_data) for r in runs]).value_counts()
    ten = 10
    expect = {(a, n, t, "isarcasm"): ten for n in (228, 780, 3120) for a, t in (("A1", "isc"), ("A20", "isc28"), ("A20", "hs"))}
    expect.update({("A20", n, "isc100", "isarcasm"): ten for n in (228, 780)})
    expect.update({("A20", b, "hsub", "headlines"): 5 for b in (1, 10)})
    assert cells.to_dict() == expect
    assert all(r.eval_test and not r.eval_masked for r in runs)
    by_tag = {t: {r.init_dir for r in runs if r.tag == t} for t in ("isc", "isc28", "isc100", "hs", "hsub")}
    assert by_tag["isc"] == {""} and by_tag["hs"] == {"/hs"}
    assert by_tag["isc28"] == {plans.PASS11_E1_CKPT["isc28"]} and by_tag["hsub"] == {plans.PASS11_E1_CKPT["hsub"]}
    steps = {r.budget: r.steps(r.budget) for r in runs if r.tag == "isc"}
    assert steps == {228: 200, 780: 250, 3120: 980}                  # the usual rule: max(200, 10 passes over L)
    assert [r.budget for r in runs[:3]] == [3120] * 3                # longest first
    older = {r.run_id for n in plans.PLANS if n != "pass11_isarc" for r in plans.PLANS[n]()}
    assert not (set(ids) & older)


# ---------------------------------------------------------------- budgets and splits

def test_isarc_budgets_are_nested_exact_and_never_dev(pass3_data):
    pool = load_isarcasm_splits(pass3_data)
    pool_ids, dev_ids = set(pool.labeled["id"]), set(pool.dev["id"])
    b = make_isarc_budgets(pass3_data, (20, 45, 90), seeds=(0, 1, 2))
    share = pool.labeled["label"].mean()
    label = dict(zip(pool.labeled["id"], pool.labeled["label"]))
    for s in ("0", "1", "2"):
        sizes = {n: len(ids) for n, ids in b[s].items()}
        assert sizes == {"20": 20, "45": 45, "90": 90}
        assert set(b[s]["20"]) < set(b[s]["45"]) < set(b[s]["90"]) == pool_ids and not (set(b[s]["90"]) & dev_ids)
        assert abs(np.mean([label[i] for i in b[s]["45"]]) - share) < 0.03          # class share stays that of the pool
    assert b["0"]["45"] != b["1"]["45"]                                              # the seed changes the draw
    assert make_isarc_budgets(pass3_data, (20, 45, 90), seeds=(0,))["0"] == b["0"]    # deterministic
    with pytest.raises(ValueError, match="larger than the pool"):
        make_isarc_budgets(pass3_data, (200,), seeds=(0,))


def test_isarcasm_split_with_a_budget(pass3_data, monkeypatch, tmp_path):
    b = make_isarc_budgets(pass3_data, (20, 45, 90), seeds=(0, 1))
    with pytest.raises(KeyError, match=ISARC_BUDGETS_ENV):
        load_isarcasm_splits(pass3_data, 45, 0)                                      # no file set: refuse, do not guess
    path = tmp_path / "isc_budgets.json"
    path.write_text(json.dumps(b))
    monkeypatch.setenv(ISARC_BUDGETS_ENV, str(path))
    s = load_isarcasm_splits(pass3_data, 45, 1)
    assert set(s.labeled["id"]) == set(b["1"]["45"]) and len(s.labeled) == 45
    full = load_isarcasm_splits(pass3_data)
    assert s.dev.equals(full.dev) and len(full.labeled) == 90                         # dev and the no-budget path unchanged
    assert isarc_budget_ids(20, 0) == b["0"]["20"]
    with pytest.raises(KeyError, match="not in"):
        isarc_budget_ids(33, 0)


# ---------------------------------------------------------------- pretraining texts (no torch)

def test_pretrain_texts_sources(pass3_data):
    dev_ids = set(load_isarcasm_splits(pass3_data).dev["id"])
    ids, texts, dev = pretrain_texts(PretrainConfig("mlm", text_source="isarc"), pass3_data)
    assert len(ids) == len(texts) == 90 and not (set(ids) & dev_ids)                  # non-dev tweets only, text only
    assert len(dev) == 10
    head = pd.read_csv(pass3_data / "headlines_train.csv", keep_default_na=False)
    hi, ht, _ = pretrain_texts(PretrainConfig("mlm"), pass3_data)
    assert len(hi) == len(head)
    si, st, _ = pretrain_texts(PretrainConfig("mlm", text_source="headlines_subset", subset_n=50), pass3_data)
    assert len(si) == 50 and set(si) <= set(head["id"]) and len(set(si)) == 50
    si2, _, _ = pretrain_texts(PretrainConfig("mlm", seed=7, text_source="headlines_subset", subset_n=50), pass3_data)
    assert list(si) == list(si2)                                                       # subset fixed, whatever the pretraining seed
    with pytest.raises(ValueError, match="text_source"):
        PretrainConfig("mlm", text_source="test")
    with pytest.raises(ValueError, match="subset_n"):
        pretrain_texts(PretrainConfig("mlm", text_source="headlines_subset", subset_n=10 ** 6), pass3_data)


# ---------------------------------------------------------------- analysis

def _write(root, arm, budget, seed, tag, sig_isarc, sig_head, rng):
    rid = f"{arm}_p{budget}_s{seed}_{tag}"
    d = root / "runs" / rid
    d.mkdir(parents=True)
    scores = {}
    for split, n, sig in (("isarc_test", 400, sig_isarc), ("test", 300, sig_head)):
        y = (np.arange(n) % 4 == 0).astype(int) if split == "isarc_test" else np.tile([0, 1], n // 2)
        p = 1 / (1 + np.exp(-(sig * (2 * y - 1) + rng.normal(0, 1, n))))
        pd.DataFrame({"id": np.arange(n), "label": y, "p_sarcastic": p}).to_csv(d / f"preds_{split}.csv", index=False)
        pred = (p >= 0.5).astype(int)
        tp, fp, fn = ((pred == 1) & (y == 1)).sum(), ((pred == 1) & (y == 0)).sum(), ((pred == 0) & (y == 1)).sum()
        scores[split] = {"macro_f1": float(0.5 + 0.1 * sig), "accuracy": float((pred == y).mean()), "f1_sarcastic": float(2 * tp / max(2 * tp + fp + fn, 1)),
                         "roc_auc": 0.5 + 0.1 * sig}
    (d / "result.json").write_text(json.dumps({"run_id": rid, "config": {"arm": arm, "budget": budget, "seed": seed, "tag": tag},
                                               "scores": scores}))


def test_e1_analysis_on_stand_in_results(tmp_path):
    from gainsplit import analysis_pass11_isarc as a11i
    rng = np.random.default_rng(1)
    for s in range(3):
        for n in (228, 780, 3120):
            _write(tmp_path, "A1", n, s, "isc", 0.30, 0.0, rng)
            _write(tmp_path, "A20", n, s, "isc28", 0.30 + (0.35 if n < 3000 else 0.05), 0.0, rng)      # the gain replicates
            _write(tmp_path, "A20", n, s, "hs", 0.30, 0.0, rng)
            if n != 3120:
                _write(tmp_path, "A20", n, s, "isc100", 0.65, 0.0, rng)
        for b in (1, 10):
            _write(tmp_path, "A1", b, s, "", 0.0, 0.8, rng)
            _write(tmp_path, "A20", b, s, "", 0.0, 1.6, rng)
            _write(tmp_path, "A20", b, s, "hsub", 0.0, 1.5, rng)
    t = a11i.run_all([tmp_path], tmp_path / "out", n=300)
    assert {"summary", "isarc_auc", "isarc_f1", "size_control", "verdicts"} <= set(t)
    assert len(t["isarc_auc"]) == len(a11i.AUC_SPECS) and len(t["isarc_f1"]) == len(a11i.F1_SPECS)
    assert len(t["size_control"]) == 4
    v = t["verdicts"]
    assert not v["verdict"].eq("no runs (skipped)").any()
    assert v[v["hypothesis"] == "reading rule"].iloc[0]["verdict"] == "REPLICATES"
    assert (tmp_path / "out" / "verdicts.csv").exists()
    s = t["summary"]
    assert set(s["tag"]) == {"isc", "isc28", "isc100", "hs"} and (s["n_seeds"] == 3).all()


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


@torch_only
def test_a20_trains_on_isarcasm_from_an_isarcasm_encoder(pass3_data, tiny_bert, monkeypatch, tmp_path):
    from gainsplit.ssl_pretrain import pretrain
    from gainsplit.train import train_run
    ck = tmp_path / "ck_isc"
    info = pretrain(PretrainConfig("mlm", steps=4, batch_size=8, model_name=tiny_bert, max_len=16, fp16=False, log_every=2,
                                   text_source="isarc"), pass3_data, ck, probe=True)
    assert info["text_source"] == "isarc" and info["n_texts"] == 90 and info["probe_before"] is None   # no Headlines probe
    ids, _, _ = pretrain_texts(PretrainConfig("mlm", text_source="isarc"), pass3_data)
    import hashlib
    assert info["text_ids_sha256"] == hashlib.sha256(",".join(str(i) for i in sorted(ids)).encode()).hexdigest()
    path = tmp_path / "isc_budgets.json"
    path.write_text(json.dumps(make_isarc_budgets(pass3_data, (20, 45, 90), seeds=(0,))))
    monkeypatch.setenv(ISARC_BUDGETS_ENV, str(path))
    base = dict(model_name=tiny_bert, max_len=16, batch_size=8, passes=1, min_steps=5, fp16=False, eval_test=True,
                train_data="isarcasm")
    a1 = train_run(RunConfig("A1", 45, 0, tag="isc", **base), pass3_data, tmp_path / "r")
    a20 = train_run(RunConfig("A20", 45, 0, tag="isc28", init_dir=str(ck), **base), pass3_data, tmp_path / "r")
    for rec in (a1, a20):
        assert rec["n_labeled"] == 45 and "isarc_test" in rec["scores"] and "roc_auc" in rec["scores"]["isarc_test"]
    assert a20["init"]["objective"] == "mlm" and a1["init"] is None
