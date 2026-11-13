"""CPU-only checks of the generated pairs (no GPU, no training of BERT). Minutes on a laptop.

Three questions a reviewer will ask about LLM-generated training data:

1. Style probe (Mahato 2026 concern): can a simple word model tell generated headlines from real
   ones? A high AUC means "LLM style" is visible. Within a pair it cannot predict the label
   (both sides are generated), but a model could still learn "generated text looks like X".
2. Real-trained reader: a TF-IDF classifier trained on the real labelled training headlines
   reads each generated side. If the sarcastic sides score higher than the serious sides,
   the pairs differ in the intended direction by a measure independent of any LLM judge.
3. Outlet cues: how well do the 200 outlet cue words (cue_words.json) predict the label in
   real training headlines vs in the generated pairs? The pairs are meant to break that link.

All models are scikit-learn with fixed seeds; 5-fold cross-validated AUC where a model is
fitted and scored on the same pool.
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

SEED = 0


def _cv_auc(texts, y, vectorizer, seed=SEED, folds=5) -> float:
    """Mean ROC-AUC of a logistic regression over stratified folds (vectorizer fitted per fold)."""
    texts, y = np.asarray(texts, dtype=object), np.asarray(y)
    aucs = []
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(texts, y):
        vec = vectorizer()
        Xtr, Xte = vec.fit_transform(texts[tr]), vec.transform(texts[te])
        clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed).fit(Xtr, y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(Xte)[:, 1]))
    return float(np.mean(aucs))


def _words():
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, token_pattern=r"\S+")


def style_probe(train: pd.DataFrame, pairs: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """AUC for telling generated (1) from real (0) headlines, overall and per generator model.

    Real headlines are drawn from the training split at random, as many as there are generated
    rows in each comparison, so both classes have the same size (0.5 = indistinguishable)."""
    rng = np.random.default_rng(seed)
    rows = []
    groups = [("all", pairs)] + [(m, g) for m, g in pairs.groupby("model")]
    for name, g in groups:
        k = min(len(g), len(train))
        real = train.iloc[rng.choice(len(train), size=k, replace=False)]
        gen = g.iloc[rng.choice(len(g), size=k, replace=False)] if len(g) > k else g
        texts = pd.concat([real["text"], gen["text"]]).to_numpy()
        y = np.r_[np.zeros(len(real)), np.ones(len(gen))]
        rows.append({"generated_by": name, "n_each": k, "auc_generated_vs_real": _cv_auc(texts, y, _words, seed)})
    return pd.DataFrame(rows)


def real_reader(train: pd.DataFrame, dev: pd.DataFrame, pairs: pd.DataFrame, seed: int = SEED):
    """Fit TF-IDF + LR on the real labelled training headlines; score dev (reference) and the pairs.

    Returns (table, pair-level frame). Table rows per generator model: AUC of the reader on the
    generated rows (intended label as truth), mean P(sarcastic) on each side, and the share of
    pairs where the sarcastic side scores above the serious side (0.5 = no difference)."""
    vec = _words()
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
    clf.fit(vec.fit_transform(train["text"]), train["label"])
    p_dev = clf.predict_proba(vec.transform(dev["text"]))[:, 1]
    rows = [{"set": "real dev (reference)", "auc": roc_auc_score(dev["label"], p_dev),
             "p_sarc_sarcastic_side": float(p_dev[dev["label"].to_numpy() == 1].mean()),
             "p_sarc_serious_side": float(p_dev[dev["label"].to_numpy() == 0].mean()),
             "pairs_sarcastic_higher": np.nan}]
    scored = pairs.assign(p=clf.predict_proba(vec.transform(pairs["text"]))[:, 1])
    wide = scored.pivot_table(index=["seed_id", "model"], columns="label", values="p").reset_index()
    for name, g in [("all", scored)] + list(scored.groupby("model")):
        w = wide if name == "all" else wide[wide["model"] == name]
        rows.append({"set": f"generated: {name}", "auc": roc_auc_score(g["label"], g["p"]),
                     "p_sarc_sarcastic_side": float(g.loc[g["label"] == 1, "p"].mean()),
                     "p_sarc_serious_side": float(g.loc[g["label"] == 0, "p"].mean()),
                     "pairs_sarcastic_higher": float((w[1] > w[0]).mean())})
    return pd.DataFrame(rows), wide


def cue_predictiveness(train: pd.DataFrame, pairs: pd.DataFrame, cue_words, seed: int = SEED) -> pd.DataFrame:
    """How well the outlet cue words alone predict the label: real training headlines vs generated pairs.

    Features: presence of each of the 200 cue words, nothing else. Also the share of headlines
    with at least one cue word, per class. Note: the cue words were chosen on the real training
    set, so the real-data AUC is optimistic by construction; the point is the drop on the pairs."""
    vocab = sorted(set(cue_words))

    def vec():
        return CountVectorizer(vocabulary=vocab, binary=True, token_pattern=r"\S+", lowercase=False)
    cue_set = set(vocab)
    rows = []
    for name, df in [("real train", train), ("generated pairs", pairs)] + \
            [(f"generated: {m}", g) for m, g in pairs.groupby("model")]:
        has = df["text"].map(lambda t: any(w in cue_set for w in t.split()))
        rows.append({"set": name, "n": len(df), "auc_cue_words_only": _cv_auc(df["text"], df["label"], vec, seed),
                     "with_cue_sarcastic": float(has[df["label"] == 1].mean()),
                     "with_cue_serious": float(has[df["label"] == 0].mean())})
    return pd.DataFrame(rows)


def length_table(train: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """Mean words per headline by class, and how well length alone predicts the label (AUC;
    0.5 = no signal). Generated sarcastic sides are longer than serious ones, so length is a
    possible shortcut the pair arms could learn; compare with the real training set."""
    rows = []
    for name, df in [("real train", train), ("generated: all", pairs)] + \
            [(f"generated: {m}", g) for m, g in pairs.groupby("model")]:
        n = df["text"].str.split().str.len()
        rows.append({"set": name, "words_sarcastic": float(n[df["label"] == 1].mean()),
                     "words_serious": float(n[df["label"] == 0].mean()),
                     "auc_length_only": float(roc_auc_score(df["label"], n))})
    return pd.DataFrame(rows)
