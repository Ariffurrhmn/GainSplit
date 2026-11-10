"""Shared test fixture: a tiny made-up processed dataset with the same files and
format as the real one (no real data is used in tests)."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

WORDS = "area man nation report new study finds local woman announces plan best worst day ever".split()


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _frame(rng, n, start):
    text = [" ".join(rng.choice(WORDS, size=rng.integers(3, 9))) for _ in range(n)]
    return pd.DataFrame({"id": np.arange(start, start + n), "text": text, "label": rng.integers(0, 2, n)})


@pytest.fixture
def fake_data(tmp_path) -> Path:
    rng = np.random.default_rng(0)
    d = tmp_path / "processed"
    d.mkdir()
    train, dev, test = _frame(rng, 200, 0), _frame(rng, 40, 200), _frame(rng, 40, 240)
    isarc = _frame(rng, 30, 0)
    for name, df in [("headlines_train.csv", train), ("headlines_dev.csv", dev),
                     ("headlines_test.csv", test), ("isarcasm_test.csv", isarc)]:
        df.to_csv(d / name, index=False)

    budgets = {}
    # same layout as the real file: 5 seeds x 5 budgets, nested (first 4 of the order, first 10, ...)
    sizes = {"1": 4, "5": 10, "10": 20, "30": 60, "100": 200}
    for seed in range(5):
        order = np.random.default_rng(seed).permutation(train["id"].to_numpy())
        budgets[str(seed)] = {b: sorted(int(i) for i in order[:n]) for b, n in sizes.items()}
    (d / "budgets.json").write_text(json.dumps(budgets))

    # teacher labels for every headline of every split (same columns as the real file)
    rows = pd.concat([train.assign(split="train"), dev.assign(split="dev"), test.assign(split="test")])[["id", "split"]]
    sent_names, emo_names = ["negative", "neutral", "positive"], ["joy", "sadness", "anger", "fear", "neutral_other"]
    sent_p, emo_p = rng.dirichlet(np.ones(3), len(rows)), rng.dirichlet(np.ones(5), len(rows))
    helper = rows.reset_index(drop=True)
    helper["sent_label"] = np.array(sent_names)[sent_p.argmax(1)]
    for i, c in enumerate(sent_names):
        helper[f"sent_p_{c}"] = sent_p[:, i]
    helper["emo_label"] = np.array(emo_names)[emo_p.argmax(1)]
    for i, c in enumerate(emo_names):
        helper[f"emo_p_{c}"] = emo_p[:, i]
    helper.to_csv(d / "helper_labels.csv", index=False)

    outputs = {p.name: {"bytes": p.stat().st_size, "sha256": _sha(p)} for p in sorted(d.iterdir())}
    (d / "manifest.json").write_text(json.dumps({"outputs": outputs}))
    return d
