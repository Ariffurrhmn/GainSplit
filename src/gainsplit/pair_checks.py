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


def isarc_length_shortcut(isarc_test: pd.DataFrame, pairs: pd.DataFrame | None = None,
                          train: pd.DataFrame | None = None) -> pd.DataFrame:
    """Does word count alone predict the iSarcasmEval test label (Pass 4 transfer check)?

    The pair arms transfer well to iSarcasmEval (AUC 72-76) while generated sarcastic sides
    are longer than serious ones (length-only AUC 0.785 inside the pairs, [[Pass 4 Results]]).
    This checks whether length alone already separates the real iSarcasmEval test labels,
    which would mean the transfer gain may be a length shortcut rather than sarcasm itself.

    One row for iSarcasmEval test; `pairs` and `train` rows (if given) are printed alongside
    for reference, not re-derived (see `length_table`). AUC 0.5 = no signal from length alone.
    """
    def row(name, df):
        n = df["text"].str.split().str.len()
        y = df["label"].to_numpy()
        return {"set": name, "n": len(df), "words_sarcastic": float(n[y == 1].mean()),
                "words_serious": float(n[y == 0].mean()), "auc_length_only": float(roc_auc_score(y, n))}

    rows = [row("isarc test", isarc_test)]
    if train is not None:
        rows.append(row("real train (reference)", train))
    if pairs is not None:
        rows.append(row("generated pairs (reference)", pairs))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- shortcut-pattern hunting

def shortcut_word_table(pairs: pd.DataFrame, train: pd.DataFrame | None = None, top_k: int = 30,
                        seed: int = SEED, folds: int = 5) -> pd.DataFrame:
    """Which words/phrases drive the AI's own sarcastic/serious label inside the pairs?

    Generalises `length_table` and `cue_predictiveness`, which each check one named feature
    (length, outlet cue words) picked by hand. Here a single interpretable model (TF-IDF
    unigrams+bigrams, min_df=3, logistic regression, same recipe as `_cv_auc`) is fit on all
    the generated pairs at once, and its largest coefficients name the words/phrases "AI-
    sarcastic-speak" and "AI-serious-speak" actually consist of, without guessing them one at
    a time. Cross-validated so the ranking is not just overfit noise; the returned model is
    refit on all of the data for the final coefficients (same as a single extra fold).

    `top_k` words each side. If `train` is given, each word's presence rate in real
    sarcastic vs. serious training headlines is added, so a word that is only an AI tic (near
    0 in both real columns) can be told apart from one that is a real, shared sarcasm marker.
    """
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True, token_pattern=r"\S+")
    X = vec.fit_transform(pairs["text"])
    y = pairs["label"].to_numpy()
    cv_auc = _cv_auc(pairs["text"], y, _words, seed, folds)
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed).fit(X, y)
    coef = clf.coef_[0]
    vocab = np.array(vec.get_feature_names_out())
    order = np.argsort(coef)
    top = np.r_[order[-top_k:][::-1], order[:top_k]]
    rows = [{"word": vocab[i], "coef": float(coef[i]),
             "leans": "sarcastic" if coef[i] > 0 else "serious"} for i in top]
    out = pd.DataFrame(rows)
    out.attrs["cv_auc"] = cv_auc   # whole-model AUC, for context next to the per-word table
    if train is not None:
        toks = train["text"].str.split()
        sarc, ser = train["label"].to_numpy() == 1, train["label"].to_numpy() == 0
        n_sarc, n_ser = int(sarc.sum()), int(ser.sum())

        def rate(w, mask):
            n = len(w.split())
            if n == 1:
                return float(toks[mask].map(lambda t: w in t).mean())
            return float(train.loc[mask, "text"].str.contains(rf"\b{w}\b", regex=True).mean())
        out["real_rate_sarcastic"] = [rate(w, sarc) for w in out["word"]]
        out["real_rate_serious"] = [rate(w, ser) for w in out["word"]]
        out["real_n_sarcastic"], out["real_n_serious"] = n_sarc, n_ser
    return out


def shortcut_feature_table(pairs: pd.DataFrame, train: pd.DataFrame | None = None,
                           seed: int = SEED) -> pd.DataFrame:
    """Named, hand-picked surface features (beyond single words): does each one alone predict
    the AI's label inside the pairs, and does the same feature predict the real training label?

    Features: word count (as `length_table`), character count, exclamation/question marks,
    all-caps word count, and a short list of hedging/filler connectives that come up often in
    generated text ("thus", "therefore", "clearly", "obviously", "literally", "honestly",
    "indeed", "surely"). Each row: AUC of that one feature alone, both sets side by side.
    """
    fillers = ["thus", "therefore", "clearly", "obviously", "literally", "honestly", "indeed", "surely"]

    def features(df):
        t = df["text"]
        words = t.str.split()
        feats = {
            "n_words": words.str.len(),
            "n_chars": t.str.len(),
            "n_exclaim": t.str.count("!"),
            "n_question": t.str.count(r"\?"),
            "n_allcaps_words": words.map(lambda ws: sum(w.isupper() and len(w) > 1 for w in ws)),
        }
        for w in fillers:
            feats[f"has_{w}"] = t.str.lower().str.contains(rf"\b{w}\b", regex=True).astype(int)
        return pd.DataFrame(feats)

    def score(df):
        f, y = features(df), df["label"].to_numpy()
        rows = []
        for name in f.columns:
            x = f[name].to_numpy(float)
            auc = roc_auc_score(y, x) if x.std() > 0 else float("nan")
            rows.append({"feature": name, "auc": auc,
                         "mean_sarcastic": float(x[y == 1].mean()), "mean_serious": float(x[y == 0].mean())})
        return pd.DataFrame(rows)

    gen = score(pairs).rename(columns={c: f"generated_{c}" for c in ("auc", "mean_sarcastic", "mean_serious")})
    if train is None:
        return gen
    real = score(train).rename(columns={c: f"real_{c}" for c in ("auc", "mean_sarcastic", "mean_serious")})
    return gen.merge(real, on="feature")
