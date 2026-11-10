"""A0: TF-IDF + logistic regression on L. A floor for the neural arms; no tuning.

CPU only and no torch. Writes the same result.json and preds_<split>.csv files as
train.py, so the runner's collect() and the analysis read it like any other arm.
"""
from pathlib import Path
import json
import platform
import time

import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .config import RunConfig
from .data import load_splits, verify
from .metrics import scores
from .util import git_commit


def run_tfidf(cfg: RunConfig, data_dir, out_root) -> dict:
    out_dir = Path(out_root) / "runs" / cfg.run_id
    if (out_dir / "result.json").exists():
        print(f"{cfg.run_id}: already done, skipped")
        return json.loads((out_dir / "result.json").read_text())
    if cfg.arm != "A0":
        raise ValueError(f"run_tfidf only runs A0 (got {cfg.arm})")

    t0 = time.time()
    manifest = verify(Path(data_dir))
    splits = load_splits(Path(data_dir), cfg.budget, cfg.seed)
    eval_sets = {"dev": splits.dev}
    if cfg.eval_test:
        eval_sets.update(test=splits.test, isarc_test=splits.isarc_test)

    # words and word pairs; sublinear tf = 1 + log(count); words seen in fewer than 2 headlines are dropped
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2)
    x_l = vec.fit_transform(splits.labeled["text"])
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
    clf.fit(x_l, splits.labeled["label"])

    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, df in eval_sets.items():
        p = clf.predict_proba(vec.transform(df["text"]))[:, 1]  # column 1 = class 1 = sarcastic
        results[name] = scores(df["label"], p)
        pd.DataFrame({"id": df["id"], "label": df["label"], "p_sarcastic": p.round(6)}).to_csv(
            out_dir / f"preds_{name}.csv", index=False)

    record = {
        "run_id": cfg.run_id,
        "config": cfg.to_dict(),
        "n_labeled": len(splits.labeled),
        "n_unlabeled": len(splits.unlabeled),
        "uses_u": False,
        "steps": 0,  # no optimizer steps: a linear model fitted in one go
        "minutes": round((time.time() - t0) / 60, 2),
        "scores": results,
        "train_log": [],
        "n_features": len(vec.vocabulary_),
        "commit": git_commit(),
        "data_manifest_sha256": {k: v["sha256"] for k, v in manifest["outputs"].items()},
        "device": "cpu",
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__},
    }
    tmp = out_dir / "result.json.tmp"
    tmp.write_text(json.dumps(record, indent=2))
    tmp.replace(out_dir / "result.json")
    print(f"{cfg.run_id}: done; " + ", ".join(f"{k} macro-F1 {v['macro_f1']:.4f}" for k, v in results.items()))
    return record
