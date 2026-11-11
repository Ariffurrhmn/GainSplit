"""Paired headline generation (Pass 4 pilot).

For each seed headline taken from the training split, one LLM writes two new
headlines on the same topic: a sarcastic one and a serious one. Both sides of a
pair come from the same model, so neither the topic, the outlet words nor the
writing style of one model can predict the label.

This file holds the parts that need no GPU and no network: picking seeds,
assigning models, building prompts, parsing and checking outputs, the judge
prompt and the blind audit table. Pandas and numpy only, so it is tested on any
machine (tests/test_pairgen.py). Model loading and generation live in the
notebook `experiments/04_pairgen_pilot.ipynb`.
"""
import html
import json
import re
import unicodedata

import numpy as np
import pandas as pd

# Same cleaning as experiments/01_preprocess.ipynb (punctuation removed), plus
# lowercasing: the headlines in the processed files are all lowercase.
QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})
PUNCT = re.compile(r"[^\w\s]")
SPACES = re.compile(r"\s+")


def clean_text(t) -> str:
    if not isinstance(t, str):
        return ""
    t = html.unescape(t)
    t = unicodedata.normalize("NFKC", t).translate(QUOTES)
    t = PUNCT.sub(" ", t)
    return SPACES.sub(" ", t).strip().lower()


def pick_seeds(train: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Random seed headlines from the training split, half from each class.

    `train` has columns id, text, label. The label is kept only to report
    results by seed class; it is never shown to the generator or the judge.
    """
    if n % 2:
        raise ValueError("n must be even (half per class)")
    parts = []
    for lab in (0, 1):
        pool = train[train["label"] == lab]
        if len(pool) < n // 2:
            raise ValueError(f"only {len(pool)} seeds with label {lab}")
        parts.append(pool.sample(n // 2, random_state=seed))
    out = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    return out[["id", "text", "label"]].rename(columns={"id": "seed_id", "text": "seed_text", "label": "seed_label"})


def assign_models(n: int, models: list[str], seed: int) -> list[str]:
    """Spread n seeds over the models in equal shares (differ by at most one), in random order."""
    reps = -(-n // len(models))
    pool = (list(models) * reps)[:n]
    rng = np.random.default_rng(seed)
    return [pool[i] for i in rng.permutation(n)]


# Invented examples (not taken from any dataset). Lowercase, no punctuation,
# like the processed headlines.
FEW_SHOT = [
    ("local school adds extra week to summer break after heat wave",
     "local school extends summer break after admitting nobody learns anything in august anyway",
     "local school extends summer break by one week after record heat wave"),
    ("man spends whole weekend building furniture he will throw out next year",
     "man spends whole weekend building bookshelf destined for curb by next spring",
     "man spends whole weekend building furniture for new apartment"),
    ("study finds regular exercise lowers risk of heart disease",
     "study finds regular exercise lowers risk of heart disease for 400th consecutive year",
     "large study confirms regular exercise lowers risk of heart disease"),
]

TASK = (
    "You write news headlines for a research dataset on sarcasm detection.\n"
    "Given a headline, write two new headlines about the same topic:\n"
    "1. \"sarcastic\": a satirical, sarcastic headline in the style of a satire news site.\n"
    "2. \"serious\": a plain, factual headline a regular news site could publish.\n"
    "Rules:\n"
    "- Keep the same topic, and reuse as many words as possible, so the two headlines differ mainly in tone.\n"
    "- Do not copy the given headline unchanged into either answer.\n"
    "- One headline each, under 20 words, lowercase, no punctuation.\n"
    "- Answer with JSON only: {\"sarcastic\": \"...\", \"serious\": \"...\"}"
)


def build_prompt(headline: str) -> str:
    """The user message for one seed (no system message: some chat templates do not accept one)."""
    shots = "\n\n".join(
        f"Headline: {h}\n" + json.dumps({"sarcastic": s, "serious": r}) for h, s, r in FEW_SHOT
    )
    return f"{TASK}\n\nExamples:\n\n{shots}\n\nHeadline: {headline}\n"


REFUSAL = re.compile(
    r"\b(i can ?not|i can't|i won't|i'm not able|i am not able|i'm unable|i am unable|"
    r"as an ai|i apologi[sz]e|i'm sorry|i am sorry|not appropriate|cannot assist|can't assist|"
    r"cannot help|can't help|against my|i must decline)\b",
    re.IGNORECASE,
)


def is_refusal(raw: str) -> bool:
    return bool(REFUSAL.search(raw or ""))


def parse_pair(raw: str):
    """Return (sarcastic, serious) as raw strings, or None if no usable JSON object is found."""
    if not isinstance(raw, str):
        return None
    for m in re.finditer(r"\{[^{}]*\}", raw, flags=re.DOTALL):
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        keys = {str(k).strip().lower(): v for k, v in obj.items()}
        s, r = keys.get("sarcastic"), keys.get("serious")
        if isinstance(s, str) and isinstance(r, str):
            return s, r
    return None


def _words(t: str) -> set:
    return set(t.split())


def jaccard(a: str, b: str) -> float:
    wa, wb = _words(a), _words(b)
    return len(wa & wb) / len(wa | wb) if wa | wb else 0.0


MAX_WORDS = 30


def check_pair(seed_text: str, raw: str) -> dict:
    """Automatic checks on one model output. `ok` is True only if every check passes."""
    seed = clean_text(seed_text)
    out = {"refusal": is_refusal(raw), "parse_ok": False, "sarcastic": "", "serious": "",
           "empty": False, "too_long": False, "same_pair": False, "copies_seed": False,
           "words_sarcastic": 0, "words_serious": 0, "overlap_pair": 0.0}
    parsed = parse_pair(raw)
    if parsed is not None:
        out["parse_ok"] = True
        s, r = clean_text(parsed[0]), clean_text(parsed[1])
        out.update(sarcastic=s, serious=r, words_sarcastic=len(s.split()), words_serious=len(r.split()))
        out["empty"] = not s or not r
        out["too_long"] = max(out["words_sarcastic"], out["words_serious"]) > MAX_WORDS
        out["same_pair"] = bool(s) and s == r
        out["copies_seed"] = s == seed or r == seed
        out["overlap_pair"] = round(jaccard(s, r), 3)
    out["ok"] = (out["parse_ok"] and not out["refusal"] and not out["empty"] and not out["too_long"]
                 and not out["same_pair"] and not out["copies_seed"])
    return out


def cue_rate(texts, words) -> float:
    """Share of headlines that contain at least one of `words` (whitespace tokens)."""
    words = set(words)
    texts = list(texts)
    if not texts:
        return 0.0
    return float(np.mean([bool(_words(t) & words) for t in texts]))


JUDGE_PROMPT = (
    "Here is a news headline (lowercase, punctuation removed):\n\n{headline}\n\n"
    "Is it written as satire or sarcasm, or as a plain serious news headline?\n"
    "Answer with exactly one word: SARCASTIC or SERIOUS."
)


def judge_prompt(headline: str) -> str:
    return JUDGE_PROMPT.format(headline=headline)


def parse_judge(text: str):
    """1 = sarcastic, 0 = serious, None = unclear."""
    t = (text or "").upper()
    s, r = "SARCASTIC" in t, "SERIOUS" in t
    if s == r:
        return None
    return 1 if s else 0


def long_items(pairs: pd.DataFrame) -> pd.DataFrame:
    """One row per generated headline: item_id, seed_id, model, intended (1 sarcastic / 0 serious), text."""
    rows = []
    for p in pairs.itertuples(index=False):
        rows.append({"item_id": f"{p.seed_id}_s", "seed_id": p.seed_id, "model": p.model, "intended": 1, "text": p.sarcastic})
        rows.append({"item_id": f"{p.seed_id}_n", "seed_id": p.seed_id, "model": p.model, "intended": 0, "text": p.serious})
    return pd.DataFrame(rows)


def blind_audit(items: pd.DataFrame, n: int, seed: int):
    """Random sample for a human check. Returns (sheet, key).

    The sheet has only a neutral number and the text, plus an empty `your_label`
    column (S = sarcastic, N = serious, U = unsure). The key maps the number back
    to item_id, model and the intended label; keep it closed while labelling.
    """
    sample = items.sample(min(n, len(items)), random_state=seed).reset_index(drop=True)
    sample.insert(0, "audit_no", np.arange(1, len(sample) + 1))
    sheet = sample[["audit_no", "text"]].assign(your_label="")
    key = sample[["audit_no", "item_id", "seed_id", "model", "intended"]]
    return sheet, key


# ---------------------------------------------------------------- production run

def plan_seeds(train: pd.DataFrame, models: list[str], seed: int, n: int | None = None) -> pd.DataFrame:
    """Seeds for the production run, each with its model. n=None uses the whole training split.

    Deterministic for a given train file, model list and seed, so every Kaggle
    session rebuilds exactly the same plan.
    """
    if n is None or n >= len(train):
        s = train.sample(frac=1, random_state=seed).reset_index(drop=True)
        s = s[["id", "text", "label"]].rename(columns={"id": "seed_id", "text": "seed_text", "label": "seed_label"})
    else:
        s = pick_seeds(train, n, seed)
    s["model"] = assign_models(len(s), models, seed)
    return s


def model_chunks(plan: pd.DataFrame, model: str, size: int) -> list[pd.DataFrame]:
    """The seeds of one model, in fixed chunks of `size` (the last one may be shorter)."""
    part = plan[plan["model"] == model].reset_index(drop=True)
    return [part.iloc[i:i + size].reset_index(drop=True) for i in range(0, len(part), size)]


def chunk_name(k: int) -> str:
    return f"chunk_{k:04d}.csv"


def find_chunks(root) -> list:
    """All saved chunk files under `root` (any depth), as raw_chunks/<model>/chunk_XXXX.csv."""
    from pathlib import Path
    root = Path(root)
    return sorted(p for p in root.rglob("chunk_*.csv") if p.parent.parent.name == "raw_chunks")


def collect_raw(paths) -> pd.DataFrame:
    """Read chunk files into one table; a seed saved twice keeps its first copy."""
    frames = [pd.read_csv(p, keep_default_na=False) for p in paths]
    if not frames:
        return pd.DataFrame(columns=["seed_id", "seed_text", "seed_label", "model", "raw"])
    return pd.concat(frames, ignore_index=True).drop_duplicates("seed_id", keep="first").reset_index(drop=True)


def training_rows(pairs: pd.DataFrame) -> pd.DataFrame:
    """Kept pairs as training rows: id, seed_id, model, text, label (1 sarcastic, 0 serious)."""
    kept = pairs[pairs["kept"]]
    items = long_items(kept)
    return items.rename(columns={"item_id": "id", "intended": "label"})[["id", "seed_id", "model", "text", "label"]]
