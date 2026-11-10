"""Find the processed dataset, check it is unchanged, and build L, U, dev and test.

Pandas only (no torch), so it can be checked on any machine.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import os

import numpy as np
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
    # U keeps no sarcasm label at all, so no later code can read one by mistake.
    unlabeled = train[~in_l].drop(columns=["label"]).reset_index(drop=True)
    return Splits(labeled, unlabeled, read("headlines_dev.csv"), read("headlines_test.csv"),
                  read("isarcasm_test.csv"))


# Class order is fixed: the integer index of a class is its position in these lists.
SENT_CLASSES = ["negative", "neutral", "positive"]
EMO_CLASSES = ["joy", "sadness", "anger", "fear", "neutral_other"]


def load_helper_labels(data_dir: Path, *frames: pd.DataFrame) -> list[pd.DataFrame]:
    """Add helper teacher labels (integer class indices) to each frame, matched by id.

    Adds columns `sent` (0-2) and `emo` (0-4). These are the hard labels from the
    pretrained sentiment/emotion models, available for every headline, so they are
    legal training targets for U as well as L.
    """
    h = pd.read_csv(Path(data_dir) / "helper_labels.csv", keep_default_na=False)
    h = h.set_index("id")
    sent = h["sent_label"].map({c: i for i, c in enumerate(SENT_CLASSES)})
    emo = h["emo_label"].map({c: i for i, c in enumerate(EMO_CLASSES)})
    out = []
    for df in frames:
        df = df.copy()
        df["sent"] = df["id"].map(sent)
        df["emo"] = df["id"].map(emo)
        if df["sent"].isna().any() or df["emo"].isna().any():
            raise ValueError("helper_labels.csv is missing ids or has an unknown class name")
        df["sent"], df["emo"] = df["sent"].astype(int), df["emo"].astype(int)
        out.append(df)
    return out


def permute_helper_labels(labeled: pd.DataFrame, unlabeled: pd.DataFrame, seed: int):
    """A8 control: shuffle the helper labels across all L + U headlines.

    Sentiment and emotion get separate random permutations. Class proportions stay
    the same, but each label no longer belongs to its own headline. The permutation
    depends only on `seed`, so a rerun gives the same shuffle.
    """
    rng = np.random.default_rng(seed + 2000)  # its own generator: does not touch torch's random state
    n_l = len(labeled)
    out = [labeled.copy(), unlabeled.copy()]
    for col in ("sent", "emo"):
        pooled = np.concatenate([labeled[col].to_numpy(), unlabeled[col].to_numpy()])
        pooled = pooled[rng.permutation(len(pooled))]
        out[0][col], out[1][col] = pooled[:n_l], pooled[n_l:]
    return out[0], out[1]
