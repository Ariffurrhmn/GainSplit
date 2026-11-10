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
MANIFEST_PASS3 = "manifest_pass3.json"
ISARC_DEV_SEED = 12345   # fixed, so every run holds out the same iSarcasmEval dev rows whatever its seed
ISARC_DEV_SHARE = 0.10


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


def verify(data_dir: Path, manifest_name: str = MANIFEST) -> dict:
    """Check every file against the hashes in manifest.json; return the manifest.

    This proves each run reads exactly the files the preprocessing notebook wrote.
    """
    data_dir = Path(data_dir)
    manifest = json.loads((data_dir / manifest_name).read_text())
    for name, info in manifest["outputs"].items():
        got = sha256(data_dir / name)
        if got != info["sha256"]:
            raise ValueError(f"{name} does not match {manifest_name} (file changed or wrong dataset)")
    return manifest


def verify_pass3(data_dir: Path) -> dict:
    """Same check for the Pass 3 files (alternative sentiment labels, masked test sets).

    Called only by runs that read those files, so older plans run without them."""
    return verify(data_dir, MANIFEST_PASS3)


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


def load_isarcasm_splits(data_dir: Path) -> Splits:
    """Pass 3, train_data="isarcasm": L is the iSarcasmEval training set minus a held-out dev split.

    The dev split is 10% of each class (stratified), drawn with a fixed seed that does not
    depend on the run seed, so all runs share it and it never overlaps L. There is no U.
    `test` is the Headlines test set (reverse transfer); `isarc_test` is the iSarcasmEval test set.
    """
    data_dir = Path(data_dir)
    read = lambda name: pd.read_csv(data_dir / name, keep_default_na=False)
    train = read("isarcasm_train.csv")
    rng = np.random.default_rng(ISARC_DEV_SEED)
    dev_rows = []
    for c in sorted(train["label"].unique()):
        rows = np.where(train["label"].to_numpy() == c)[0]
        n_dev = max(1, round(ISARC_DEV_SHARE * len(rows)))
        dev_rows.extend(rng.choice(rows, size=n_dev, replace=False).tolist())
    is_dev = np.zeros(len(train), dtype=bool)
    is_dev[dev_rows] = True
    labeled = train[~is_dev].reset_index(drop=True)
    dev = train[is_dev].reset_index(drop=True)
    unlabeled = labeled.iloc[0:0].drop(columns=["label"])
    return Splits(labeled, unlabeled, dev, read("headlines_test.csv"), read("isarcasm_test.csv"))


def load_masked_tests(data_dir: Path) -> dict[str, pd.DataFrame]:
    """The Headlines test set with cue words masked (`test_cuemask`) or the same number of
    random words masked (`test_randmask`). Same rows and labels as headlines_test.csv."""
    data_dir = Path(data_dir)
    read = lambda name: pd.read_csv(data_dir / name, keep_default_na=False)
    test = read("headlines_test.csv")
    out = {}
    for split, name in (("test_cuemask", "headlines_test_cuemask.csv"), ("test_randmask", "headlines_test_randmask.csv")):
        df = read(name)
        if not (df["id"].tolist() == test["id"].tolist() and df["label"].tolist() == test["label"].tolist()):
            raise ValueError(f"{name} does not have the same ids and labels as headlines_test.csv")
        out[split] = df
    return out


# Class order is fixed: the integer index of a class is its position in these lists.
SENT_CLASSES = ["negative", "neutral", "positive"]
EMO_CLASSES = ["joy", "sadness", "anger", "fear", "neutral_other"]


SENT_P = [f"sent_p_{c}" for c in SENT_CLASSES]   # teacher probabilities, columns of helper_labels.csv
EMO_P = [f"emo_p_{c}" for c in EMO_CLASSES]


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


def add_soft_helper_targets(data_dir: Path, *frames: pd.DataFrame) -> list[pd.DataFrame]:
    """Pass 3 `soft`: add the teacher probabilities (columns SENT_P and EMO_P) to each frame, by id."""
    h = pd.read_csv(Path(data_dir) / "helper_labels.csv", keep_default_na=False).set_index("id")
    out = []
    for df in frames:
        df = df.copy()
        for col in SENT_P + EMO_P:
            df[col] = df["id"].map(h[col]).astype(float)
        if df[SENT_P + EMO_P].isna().any().any():
            raise ValueError("helper_labels.csv is missing ids")
        out.append(df)
    return out


def flip_helper_labels(labeled: pd.DataFrame, unlabeled: pd.DataFrame, share: float, seed: int):
    """Pass 3 flip25 / flip50: change `share` of the helper labels over L + U together.

    Each chosen row gets a different class, uniform among the others. Sentiment and emotion
    are flipped independently. Own generator (seed + 3000), so torch's random state is untouched
    and a rerun gives the same flips."""
    rng = np.random.default_rng(seed + 3000)
    n_l = len(labeled)
    out = [labeled.copy(), unlabeled.copy()]
    for col, k in (("sent", len(SENT_CLASSES)), ("emo", len(EMO_CLASSES))):
        pooled = np.concatenate([labeled[col].to_numpy(), unlabeled[col].to_numpy()]).copy()
        hit = rng.choice(len(pooled), size=round(share * len(pooled)), replace=False)
        pooled[hit] = (pooled[hit] + rng.integers(1, k, size=len(hit))) % k  # shift by 1..k-1: never the same class
        out[0][col], out[1][col] = pooled[:n_l], pooled[n_l:]
    return out[0], out[1]


def replace_sentiment_labels(data_dir: Path, name: str, *frames: pd.DataFrame) -> list[pd.DataFrame]:
    """Pass 3 vader / teacher2: swap the sentiment label for `<name>_label` from helper_sent_alt.csv.
    The emotion label stays as it was."""
    alt = pd.read_csv(Path(data_dir) / "helper_sent_alt.csv", keep_default_na=False)
    if alt["id"].duplicated().any():
        raise ValueError("helper_sent_alt.csv has duplicate ids")
    new = alt.set_index("id")[f"{name}_label"].map({c: i for i, c in enumerate(SENT_CLASSES)})
    out = []
    for df in frames:
        df = df.copy()
        df["sent"] = df["id"].map(new)
        if df["sent"].isna().any():
            raise ValueError("helper_sent_alt.csv is missing ids or has an unknown class name")
        df["sent"] = df["sent"].astype(int)
        out.append(df)
    return out


def apply_helper_variant(variant: str, data_dir: Path, labeled: pd.DataFrame, unlabeled: pd.DataFrame, seed: int):
    """Apply a Pass 3 helper-label variant to L and U (which already have `sent` and `emo`).
    sentonly / emoonly change no labels; they only change which loss the training step uses."""
    if variant == "soft":
        return add_soft_helper_targets(data_dir, labeled, unlabeled)
    if variant in ("flip25", "flip50"):
        return flip_helper_labels(labeled, unlabeled, int(variant[4:]) / 100, seed)
    if variant in ("vader", "teacher2"):
        return replace_sentiment_labels(data_dir, variant, labeled, unlabeled)
    return labeled, unlabeled


def helper_label_counts(labeled: pd.DataFrame, unlabeled: pd.DataFrame) -> dict:
    """Class counts of the hard helper labels over L + U, for the run record."""
    both = pd.concat([labeled, unlabeled])
    return {"sent": {c: int((both["sent"] == i).sum()) for i, c in enumerate(SENT_CLASSES)},
            "emo": {c: int((both["emo"] == i).sum()) for i, c in enumerate(EMO_CLASSES)}}
