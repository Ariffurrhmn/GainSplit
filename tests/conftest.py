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
    for seed in (0, 1):
        order = np.random.default_rng(seed).permutation(train["id"].to_numpy())
        budgets[str(seed)] = {"10": sorted(int(i) for i in order[:20]),
                              "100": sorted(int(i) for i in order)}
    (d / "budgets.json").write_text(json.dumps(budgets))

    outputs = {p.name: {"bytes": p.stat().st_size, "sha256": _sha(p)} for p in sorted(d.iterdir())}
    (d / "manifest.json").write_text(json.dumps({"outputs": outputs}))
    return d
