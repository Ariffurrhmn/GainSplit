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


# ---------------------------------------------------------------- Pass 4: generated headline pairs

PAIRS_FILE = "pairs_train.csv"                    # notebook 05 finalize output (all checked pairs)
PAIRS_MANIFEST = "manifest_pairs.json"
FILTERED_FILE = "pairs_train_filtered.csv"        # notebook 06 output (judge-v2 filter)
FILTERED_MANIFEST = "manifest_judge_v2.json"
CUE_WORDS = "cue_words.json"                      # Pass 3 file: the outlet cue words
SHORTCUT_SCORES_FILE = "pair_shortcut_scores.csv"       # notebook 09 output (Pass 5 cleaning)
SHORTCUT_SCORES_MANIFEST = "manifest_shortcut_scores.json"
DEFAULT_CLEANED_DROP_PCT = 20.0   # preflight's reference display only; real runs set their own cleaned_drop_pct
PAIR_SUBSET_SEED = 2027   # fixed: every run and seed uses the same `matched` subset and `unpaired` sides
PAIR_COLUMNS = ["id", "seed_id", "model", "text", "label"]


def find_input_file(name: str, data_dir=None, root=None) -> Path:
    """Locate a Pass 4 input file.

    Order: the folder in GAINSPLIT_PAIRS (if set), then `data_dir`, then a search under `root`
    (default: GAINSPLIT_INPUT or /kaggle/input). If the search finds several copies, they must
    be byte-identical (same SHA-256), so it cannot matter which one is read.
    """
    if os.environ.get("GAINSPLIT_PAIRS"):
        p = Path(os.environ["GAINSPLIT_PAIRS"]) / name
        if p.exists():
            return p
    if data_dir is not None and (Path(data_dir) / name).exists():
        return Path(data_dir) / name
    root = Path(root or os.environ.get("GAINSPLIT_INPUT", "/kaggle/input"))
    hits = sorted(root.rglob(name)) if root.exists() else []
    if not hits:
        raise FileNotFoundError(f"{name} not found (looked in GAINSPLIT_PAIRS, {data_dir} and under {root}); "
                                "add it to the gainsplit-processed dataset")
    if len({sha256(h) for h in hits}) > 1:
        raise ValueError(f"different copies of {name}: {[str(h) for h in hits]}")
    return hits[0]


def _read_checked(name: str, manifest_name: str, data_dir, columns=None):
    """Read a CSV and check it against the SHA-256 in its manifest. Returns (rows, sha256, manifest).

    `columns` (default PAIR_COLUMNS): the required columns, kept and returned in that order.
    """
    columns = PAIR_COLUMNS if columns is None else columns
    path = find_input_file(name, data_dir)
    mpath = path.parent / manifest_name
    if not mpath.exists():
        mpath = find_input_file(manifest_name, data_dir)
    manifest = json.loads(mpath.read_text())
    got = sha256(path)
    if manifest.get("sha256", {}).get(name) != got:
        raise ValueError(f"{name} does not match {manifest_name} (file changed or wrong version)")
    df = pd.read_csv(path, keep_default_na=False)
    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(f"{name} lacks columns {sorted(missing)}")
    return df[columns].copy(), got, manifest


def check_pairs(pairs: pd.DataFrame, train_ids=None, heldout_texts=None) -> None:
    """Every seed has exactly one sarcastic (1) and one serious (0) row; ids are unique; seeds come
    from train only; no generated text equals a dev or test headline. Raises ValueError otherwise."""
    g = pairs.groupby("seed_id")["label"]
    if not ((g.size() == 2) & (g.sum() == 1)).all():
        raise ValueError("pair file: every seed_id must have exactly one row with label 1 and one with label 0")
    if pairs["id"].duplicated().any():
        raise ValueError("pair file: duplicate ids")
    if train_ids is not None and not set(pairs["seed_id"]).issubset(set(train_ids)):
        raise ValueError("pair file: some seed_ids are not in headlines_train.csv")
    if heldout_texts is not None and set(pairs["text"]) & set(heldout_texts):
        raise ValueError("pair file: a generated headline equals a dev or test headline")


def matched_subset(unfiltered: pd.DataFrame, filtered: pd.DataFrame, seed: int = PAIR_SUBSET_SEED) -> pd.DataFrame:
    """A14 control: draw, per generator model, as many whole pairs from `unfiltered` as `filtered` keeps.

    Same size and model mix as the judge-filtered set, but the pairs are chosen at random,
    so A13 - A14 measures the judge's choice alone."""
    rng = np.random.default_rng(seed)
    want = filtered.groupby("model")["seed_id"].nunique()
    seeds = []
    for model in sorted(want.index):
        pool = np.sort(unfiltered.loc[unfiltered["model"] == model, "seed_id"].unique())
        if len(pool) < want[model]:
            raise ValueError(f"{model}: {want[model]} pairs wanted, only {len(pool)} available")
        seeds.extend(rng.choice(pool, size=int(want[model]), replace=False).tolist())
    return unfiltered[unfiltered["seed_id"].isin(set(seeds))].reset_index(drop=True)


def unpaired_sides(unfiltered: pd.DataFrame, seed: int = PAIR_SUBSET_SEED) -> pd.DataFrame:
    """A12 control: keep one side of each pair, chosen at random. The two classes then come from
    different topics (no topic-matched partner), with the same generated-text source."""
    rng = np.random.default_rng(seed)
    seeds = np.sort(unfiltered["seed_id"].unique())
    side = pd.Series(rng.integers(0, 2, size=len(seeds)), index=seeds)   # 1: keep the sarcastic side
    keep = unfiltered["label"].to_numpy() == unfiltered["seed_id"].map(side).to_numpy()
    return unfiltered[keep].reset_index(drop=True)


def load_shortcut_scores(data_dir) -> pd.Series:
    """Pass 5: per-seed shortcut score from `pair_shortcut_scores.csv` (notebook 09, section A5),
    checked against `manifest_shortcut_scores.json`. Higher = that seed's generated pair leans
    more heavily on the AI writing-tic words found in `pair_checks.shortcut_word_table` (the max
    of its two sides' scores). Returns a Series indexed by seed_id."""
    df, _, _ = _read_checked(SHORTCUT_SCORES_FILE, SHORTCUT_SCORES_MANIFEST, data_dir,
                             columns=["seed_id", "shortcut_score"])
    df["seed_id"] = df["seed_id"].astype(int)
    df["shortcut_score"] = df["shortcut_score"].astype(float)
    return df.set_index("seed_id")["shortcut_score"]


def cleaned_subset(unfiltered: pd.DataFrame, scores: pd.Series, drop_pct: float) -> pd.DataFrame:
    """Pass 5: drop the `drop_pct`% of seeds with the highest shortcut score (both sides go
    together, so pairing is preserved). Ties at the cutoff are broken by seed_id, so the result
    is exactly reproducible."""
    if not (0 < drop_pct < 100):
        raise ValueError(f"drop_pct must be between 0 and 100, exclusive (got {drop_pct})")
    seeds = np.sort(unfiltered["seed_id"].unique())
    missing = set(seeds) - set(scores.index)
    if missing:
        raise ValueError(f"{len(missing)} seed(s) in the pairs have no shortcut score, e.g. {sorted(missing)[:5]}")
    s = scores.loc[seeds].sort_values(kind="stable")   # stable sort: ties keep seed_id order
    n_drop = int(round(len(s) * drop_pct / 100))
    keep_seeds = set(s.index[:len(s) - n_drop]) if n_drop else set(s.index)
    return unfiltered[unfiltered["seed_id"].isin(keep_seeds)].reset_index(drop=True)


def load_pairs(pair_set: str, data_dir, check_against=None, cleaned_drop_pct: float = DEFAULT_CLEANED_DROP_PCT):
    """Load one Pass 4-5 pair set; return (rows, info for result.json).

    Rows have columns id, seed_id, model, text, label, sorted by seed_id then label, so the two
    sides of a pair sit next to each other (the batch stream keeps them together).
    `check_against`: the processed data folder; if given, the leak checks run against its splits.
    `cleaned_drop_pct`: only used by pair_set="cleaned" (see `cleaned_subset`); the default is for
    `preflight`'s display only, real training runs pass their own `RunConfig.cleaned_drop_pct`.
    """
    unf, unf_sha, _ = _read_checked(PAIRS_FILE, PAIRS_MANIFEST, data_dir)
    check_pairs(unf)
    info = {"pair_set": pair_set, "pairs_train_sha256": unf_sha}
    fil = None
    if pair_set in ("filtered", "matched"):
        fil, fil_sha, fman = _read_checked(FILTERED_FILE, FILTERED_MANIFEST, data_dir)
        if fman.get("input_pairs_train_sha256") not in (None, unf_sha):
            raise ValueError("pairs_train_filtered.csv was made from a different pairs_train.csv")
        if not set(fil["id"]).issubset(set(unf["id"])):
            raise ValueError("pairs_train_filtered.csv has rows that are not in pairs_train.csv")
        check_pairs(fil)
        info["pairs_train_filtered_sha256"] = fil_sha
    if pair_set == "unfiltered":
        rows = unf
    elif pair_set == "filtered":
        rows = fil
    elif pair_set == "matched":
        rows = matched_subset(unf, fil)
    elif pair_set == "unpaired":
        rows = unpaired_sides(unf)
    elif pair_set == "cleaned":
        scores = load_shortcut_scores(data_dir)
        rows = cleaned_subset(unf, scores, cleaned_drop_pct)
        info["cleaned_drop_pct"] = cleaned_drop_pct
    else:
        raise ValueError(f"unknown pair set {pair_set!r}")
    if check_against is not None:
        read = lambda n: pd.read_csv(Path(check_against) / n, keep_default_na=False)
        held = pd.concat([read("headlines_dev.csv")["text"], read("headlines_test.csv")["text"]])
        check_pairs(unf, read("headlines_train.csv")["id"], held)
    rows = rows.sort_values(["seed_id", "label"], kind="stable").reset_index(drop=True)
    info.update(n_rows=int(len(rows)), n_seeds=int(rows["seed_id"].nunique()),
                rows_per_label={str(k): int(v) for k, v in rows["label"].value_counts().sort_index().items()},
                seeds_per_model={str(k): int(v) for k, v in rows.groupby("model")["seed_id"].nunique().items()})
    return rows, info


def load_cue_words(data_dir) -> list[str]:
    """The 2 x 100 outlet cue words from the Pass 3 file cue_words.json (checked via manifest_pass3.json)."""
    verify_pass3(data_dir)
    cues = json.loads((Path(data_dir) / CUE_WORDS).read_text())
    return [w for w, _ in cues["sarcastic"]] + [w for w, _ in cues["non_sarcastic"]]


def preflight(configs, data_dir) -> pd.DataFrame:
    """CPU check before any GPU time is spent: load every pair set and file the plan needs.

    Raises on the first problem (missing file, hash mismatch, broken pairs, leak). Returns one row
    per pair set with its size, so the notebook can show what the runs will read."""
    from .config import ARMS
    data_dir = Path(data_dir)
    verify(data_dir)
    sets = sorted({ARMS[c.arm]["pairs"] for c in configs if ARMS[c.arm].get("pairs")})
    if any(c.eval_masked or c.explain for c in configs):
        verify_pass3(data_dir)
    if any(c.explain for c in configs):
        load_cue_words(data_dir)
    rows = []
    for i, s in enumerate(sets):
        df, info = load_pairs(s, data_dir, check_against=data_dir if i == 0 else None)
        rows.append({"pair_set": s, "rows": info["n_rows"], "pairs_or_seeds": info["n_seeds"],
                     **{f"label_{k}": v for k, v in info["rows_per_label"].items()},
                     **{f"seeds_{k}": v for k, v in info["seeds_per_model"].items()}})
    return pd.DataFrame(rows)
