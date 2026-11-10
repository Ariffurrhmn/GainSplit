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


@pytest.fixture
def pass3_data(fake_data) -> Path:
    """fake_data plus the Pass 3 files: alternative sentiment labels, masked test sets,
    an iSarcasmEval training set (70 non-sarcastic, 30 sarcastic) and manifest_pass3.json."""
    d = fake_data
    rng = np.random.default_rng(1)
    helper = pd.read_csv(d / "helper_labels.csv")
    names = ["negative", "neutral", "positive"]
    alt = helper[["id", "split"]].copy()
    for src in ("vader", "teacher2"):
        alt[f"{src}_label"] = rng.choice(names, len(alt))
    alt["vader_compound"] = rng.uniform(-1, 1, len(alt))
    for c in names:
        alt[f"teacher2_p_{c}"] = 1 / 3
    alt.to_csv(d / "helper_sent_alt.csv", index=False)

    test = pd.read_csv(d / "headlines_test.csv")
    for name, k in (("cuemask", 0), ("randmask", 1)):  # mask the first / second word of every headline
        masked = test.copy()
        masked["text"] = [" ".join("[MASK]" if i == k else w for i, w in enumerate(t.split())) for t in test["text"]]
        masked.to_csv(d / f"headlines_test_{name}.csv", index=False)

    isarc = _frame(rng, 100, 1000)
    isarc["label"] = [0] * 70 + [1] * 30
    isarc.to_csv(d / "isarcasm_train.csv", index=False)

    names_new = ["helper_sent_alt.csv", "headlines_test_cuemask.csv", "headlines_test_randmask.csv", "isarcasm_train.csv"]
    outputs = {n: {"bytes": (d / n).stat().st_size, "sha256": _sha(d / n)} for n in names_new}
    (d / "manifest_pass3.json").write_text(json.dumps({"outputs": outputs}))
    return d
