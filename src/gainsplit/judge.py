"""Judge contest, version 2 (Pass 4 pair dataset).

The first contest (experiments/05_pairgen_full.ipynb) asked each judge for one
word and failed: every judge answered SARCASTIC to almost everything, so no
judge passed. The judges still ranked headlines in the right direction, so this
version reads a *score* instead of a word and sets the cut-off (threshold) on
real headlines:

    score = log P("SARCASTIC" | prompt) - log P("SERIOUS" | prompt)

A headline counts as sarcastic when its score is at or above the threshold.

Rules that keep the contest honest:
- The judge is chosen only on real, labelled training headlines, never on the
  generated pairs and never on dev or test.
- Two separate sets of real headlines: the threshold is set on the "tune" set,
  the accuracy that decides pass or fail is measured on the "report" set.
- A generator model never judges pairs it wrote itself.
- Same bar as before: balanced accuracy >= 0.75 on the report set.

This file holds the parts that need no GPU: picking the real headlines,
building prompts, thresholds, AUC, choosing a judge per author and filtering
pairs. Pandas and numpy only (tests/test_judge.py). Model scoring lives in
experiments/06_judge_v2.ipynb.
"""
import numpy as np
import pandas as pd

from gainsplit import pairgen as pg

ANSWERS = {1: "SARCASTIC", 0: "SERIOUS"}
PROMPT_VERSIONS = ("v1", "v2")

PROMPT_V2 = (
    "You will see a news headline. Every headline in this task, real news and satire alike, "
    "is written in lowercase with all punctuation removed, so the format tells you nothing "
    "about the answer.\n"
    "Decide whether the headline is written as satire or sarcasm (in the style of a satire news site), "
    "or as a plain serious news headline.\n\n"
    "Examples:\n\n{examples}\n\n"
    "Headline: {headline}\n"
    "Answer with exactly one word: SARCASTIC or SERIOUS."
)


def pick_real_sets(train: pd.DataFrame, n_tune: int, n_report: int, n_shot_per_class: int, seed: int):
    """Three disjoint sets of real training headlines, each half sarcastic and half serious.

    Returns (shots, tune, report), each with columns id, text, label. `shots` are the
    few-shot examples shown inside prompt v2; they are never scored.
    """
    for n in (n_tune, n_report):
        if n % 2:
            raise ValueError("set sizes must be even (half per class)")
    rng = np.random.default_rng(seed)
    parts = {"shots": [], "tune": [], "report": []}
    for lab in (1, 0):
        pool = train[train["label"] == lab]
        need = n_shot_per_class + n_tune // 2 + n_report // 2
        if len(pool) < need:
            raise ValueError(f"only {len(pool)} headlines with label {lab}, need {need}")
        order = pool.iloc[rng.permutation(len(pool))[:need]]
        a, b = n_shot_per_class, n_shot_per_class + n_tune // 2
        parts["shots"].append(order.iloc[:a])
        parts["tune"].append(order.iloc[a:b])
        parts["report"].append(order.iloc[b:])
    out = []
    for name in ("shots", "tune", "report"):
        df = pd.concat(parts[name])
        df = df.iloc[rng.permutation(len(df))].reset_index(drop=True)
        out.append(df[["id", "text", "label"]])
    return tuple(out)


def format_examples(shots: pd.DataFrame) -> str:
    return "\n".join(f"Headline: {t}\nAnswer: {ANSWERS[int(l)]}" for t, l in zip(shots["text"], shots["label"]))


def build_prompt(headline: str, version: str, shots: pd.DataFrame | None = None) -> str:
    """v1: the prompt of the first contest. v2: format note plus labelled real examples."""
    if version == "v1":
        return pg.judge_prompt(headline)
    if version == "v2":
        if shots is None or not len(shots):
            raise ValueError("prompt v2 needs few-shot examples")
        return PROMPT_V2.format(examples=format_examples(shots), headline=headline)
    raise ValueError(f"unknown prompt version {version}")


def balanced_accuracy_at(y, scores, thr: float) -> float:
    """Mean of the per-class accuracies when score >= thr means sarcastic."""
    y, s = np.asarray(y, dtype=int), np.asarray(scores, dtype=float)
    pred = (s >= thr).astype(int)
    return float(np.mean([np.mean(pred[y == c] == c) for c in (0, 1) if np.any(y == c)]))


def best_threshold(y, scores):
    """Threshold with the highest balanced accuracy. Returns (threshold, balanced accuracy).

    Candidates are the midpoints between neighbouring distinct scores plus one value
    below and one above all scores. Ties are broken by taking the middle candidate
    of the best ones, which sits furthest from both edges.
    """
    s = np.asarray(scores, dtype=float)
    if np.isnan(s).any():
        raise ValueError("scores contain NaN")
    u = np.unique(s)
    cands = np.concatenate([[u[0] - 1.0], (u[:-1] + u[1:]) / 2, [u[-1] + 1.0]])
    accs = np.array([balanced_accuracy_at(y, s, t) for t in cands])
    best = np.flatnonzero(np.isclose(accs, accs.max()))
    i = best[len(best) // 2]
    return float(cands[i]), float(accs[i])


def auc(y, scores) -> float:
    """Area under the ROC curve (Mann-Whitney U; ties count half). 0.5 = no ranking ability."""
    y, s = np.asarray(y, dtype=int), np.asarray(scores, dtype=float)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if not n1 or not n0:
        return float("nan")
    ranks = pd.Series(s).rank(method="average").to_numpy()
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def evaluate(tune_y, tune_s, report_y, report_s) -> dict:
    """Threshold from the tune set; everything that decides pass/fail from the report set."""
    thr, tune_bacc = best_threshold(tune_y, tune_s)
    report_s = np.asarray(report_s, dtype=float)
    return {
        "threshold": thr,
        "tune_bacc": round(tune_bacc, 4),
        "report_bacc": round(balanced_accuracy_at(report_y, report_s, thr), 4),
        "tune_auc": round(auc(tune_y, tune_s), 4),
        "report_auc": round(auc(report_y, report_s), 4),
        "report_sarcastic_rate": round(float(np.mean(report_s >= thr)), 4),
    }


def choose_judges(results: dict, authors, min_score: float) -> dict:
    """The judge for each author's pairs, or None.

    `results` maps a candidate name to a dict with at least `report_bacc`,
    `tune_bacc` and `base` (the model behind it; a generator's name, or anything
    else for a model that wrote no pairs). Eligible: passes `min_score` on the report
    set and is not the author. Among those, the highest tune-set balanced accuracy
    wins (ties: the higher report score, then the name, so the choice is repeatable).
    """
    out = {}
    for a in authors:
        ok = sorted((-r["tune_bacc"], -r["report_bacc"], name) for name, r in results.items()
                    if r["report_bacc"] >= min_score and r["base"] != a)
        out[a] = ok[0][2] if ok else None
    return out


def filter_pairs(rows: pd.DataFrame, gen_scores: pd.DataFrame, judges: dict, results: dict):
    """Apply each author's judge to that author's pairs.

    rows: pairs_train.csv (id, seed_id, model, text, label), two rows per seed.
    gen_scores: id, candidate, score (scores of generated headlines).
    A pair is kept when its judge puts the sarcastic side at or above the judge's
    threshold and the serious side below it. Authors without a judge keep every pair,
    flagged with judged = False.

    Returns (filtered rows with the same columns as `rows`, one row per pair).
    """
    cols = ["id", "seed_id", "model", "text", "label"]
    r = rows[cols].copy()
    r["judge"] = r["model"].map(judges)
    s = gen_scores.rename(columns={"candidate": "judge"})
    r = r.merge(s[["id", "judge", "score"]], on=["id", "judge"], how="left")
    judged = r["judge"].notna()
    if r.loc[judged, "score"].isna().any():
        missing = r.loc[judged & r["score"].isna(), "id"].head(5).tolist()
        raise ValueError(f"no score from the chosen judge for some headlines, e.g. {missing}")
    thr = r["judge"].map(lambda j: results[j]["threshold"] if isinstance(j, str) else np.nan)
    r["agrees"] = np.where(judged, (r["score"] >= thr).astype(int) == r["label"], True)
    side = r.pivot_table(index="seed_id", columns="label", values="agrees", aggfunc="all")
    per_pair = (r.groupby("seed_id").agg(model=("model", "first"), judge=("judge", "first"), n=("id", "size"))
                .join(side.rename(columns={1: "sarcastic_ok", 0: "serious_ok"})).reset_index())
    for c in ("sarcastic_ok", "serious_ok"):
        per_pair[c] = per_pair[c].fillna(False).astype(bool)
    per_pair["judged"] = per_pair["judge"].notna()
    per_pair["kept"] = (per_pair["n"] == 2) & per_pair["sarcastic_ok"] & per_pair["serious_ok"]
    keep = set(per_pair.loc[per_pair["kept"], "seed_id"])
    filtered = rows.loc[rows["seed_id"].isin(keep), cols].reset_index(drop=True)
    return filtered, per_pair
