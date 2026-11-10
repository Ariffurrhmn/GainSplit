"""Word masking for the outlet-style shortcut test (RQ3).

Pandas, numpy and sklearn only (no torch), so it can be checked on any machine.

A "token" is a whitespace-separated piece of text. The headlines are already
lowercased with punctuation removed, so this matches how the cue-word model
sees the text.
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

MASK = "[MASK]"


def fit_cue_words(train_df: pd.DataFrame, k: int = 100) -> dict:
    """Find the words that most strongly point to each class.

    Fits a word-unigram logistic regression on `train_df` (columns text, label;
    label 1 = sarcastic) and returns the `k` words with the largest positive
    weights (sarcastic side) and the `k` with the most negative weights
    (non-sarcastic side), each as a list of [word, weight], strongest first.
    """
    vec = CountVectorizer(binary=True, min_df=2, token_pattern=r"\S+", lowercase=False)
    X = vec.fit_transform(train_df["text"].astype(str))
    clf = LogisticRegression(C=1, max_iter=2000, class_weight="balanced", random_state=0)
    clf.fit(X, train_df["label"].to_numpy())
    if list(clf.classes_) != [0, 1]:
        raise ValueError(f"expected labels 0/1, got {list(clf.classes_)}")
    words = np.asarray(vec.get_feature_names_out())
    w = clf.coef_[0]
    order = np.argsort(-w, kind="stable")
    pos = [[str(words[i]), float(w[i])] for i in order[:k]]
    neg = [[str(words[i]), float(w[i])] for i in order[::-1][:k]]
    return {"sarcastic": pos, "non_sarcastic": neg}


def cue_mask(texts, words) -> tuple[list[str], list[int]]:
    """Replace every token that is in `words` with [MASK].

    Returns (masked texts, number of tokens masked in each text). A text with
    no cue word is returned exactly as it was.
    """
    wordset = set(words)
    out, counts = [], []
    for t in texts:
        toks = t.split()
        hit = [tok in wordset for tok in toks]
        n = sum(hit)
        counts.append(n)
        out.append(" ".join(MASK if h else tok for tok, h in zip(toks, hit)) if n else t)
    return out, counts


def random_mask(texts, counts, seed: int) -> list[str]:
    """Control: mask `counts[i]` randomly chosen tokens of text i.

    Same number of masked tokens per headline as the cue mask, but the
    positions are random, so any difference from the cue mask is due to
    *which* words were hidden, not how many. Texts with count 0 are unchanged.
    """
    rng = np.random.default_rng(seed)
    out = []
    for t, n in zip(texts, counts):
        toks = t.split()
        if n == 0:
            out.append(t)
            continue
        if n > len(toks):
            raise ValueError(f"cannot mask {n} of {len(toks)} tokens: {t!r}")
        idx = set(rng.choice(len(toks), size=n, replace=False).tolist())
        out.append(" ".join(MASK if i in idx else tok for i, tok in enumerate(toks)))
    return out
