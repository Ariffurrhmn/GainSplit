"""Find the processed dataset, check it is unchanged, and build L, U, dev and test.

Pandas only (no torch), so it can be checked on any machine.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import os

import pandas as pd

MANIFEST = "manifest.json"


def find_data_dir(root: str | Path = "/kaggle/input") -> Path:
    """Return the folder that holds manifest.json and budgets.json.

    The GAINSPLIT_DATA environment variable wins if set. Otherwise search under
    `root`, because Kaggle's mount path for an attached dataset can vary.
    """
    if os.environ.get("GAINSPLIT_DATA"):
        return Path(os.environ["GAINSPLIT_DATA"])
    hits = [p.parent for p in Path(root).rglob(MANIFEST) if (p.parent / "budgets.json").exists()]
    if len(hits) != 1:
        raise FileNotFoundError(f"expected one processed dataset under {root}, found {len(hits)}: {hits}")
    return hits[0]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(data_dir: Path) -> dict:
    """Check every file against the hashes in manifest.json; return the manifest.

    This proves each run reads exactly the files the preprocessing notebook wrote.
    """
    data_dir = Path(data_dir)
    manifest = json.loads((data_dir / MANIFEST).read_text())
    for name, info in manifest["outputs"].items():
        got = sha256(data_dir / name)
        if got != info["sha256"]:
            raise ValueError(f"{name} does not match manifest.json (file changed or wrong dataset)")
    return manifest


@dataclass
class Splits:
    labeled: pd.DataFrame     # L: train rows that keep their sarcasm label at this budget
    unlabeled: pd.DataFrame   # U: the rest of train; the training code must not read its labels
    dev: pd.DataFrame
    test: pd.DataFrame
    isarc_test: pd.DataFrame  # iSarcasmEval English Task A test, for zero-shot transfer (RQ3)


def load_splits(data_dir: Path, budget: int, seed: int) -> Splits:
    """Build L and U for one budget and seed from budgets.json."""
    data_dir = Path(data_dir)
    read = lambda name: pd.read_csv(data_dir / name, keep_default_na=False)
    train = read("headlines_train.csv")
    budgets = json.loads((data_dir / "budgets.json").read_text())
    ids = set(budgets[str(seed)][str(budget)])

    in_l = train["id"].isin(ids)
    if in_l.sum() != len(ids):
        raise ValueError("budgets.json lists ids that are not in headlines_train.csv")
    labeled = train[in_l].reset_index(drop=True)
    unlabeled = train[~in_l].reset_index(drop=True)
    return Splits(labeled, unlabeled, read("headlines_dev.csv"), read("headlines_test.csv"),
                  read("isarcasm_test.csv"))
