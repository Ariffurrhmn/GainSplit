"""Pass 4 explainability: Integrated Gradients (IG) word attributions on a fixed test sample.

What it measures (vault: Experiment Plan, "Pass 4 plan", explainability step):
for each headline, how much of the model's decision rests on each word, and what share of
that rests on the 200 outlet cue words (cue_words.json). The main number per model is the
mean cue share; a lower share than A1's means the model leans less on outlet style.

Method. IG (Sundararajan et al. 2017) on the word embeddings: walk in a straight line from a
baseline input (every position replaced by the [PAD] embedding, the "all-padding baseline")
to the real input, add up the gradient of the predicted class's score along the way, and
multiply by (input - baseline). A token's attribution is that product summed over the 768
embedding numbers. Word pieces of one word are summed into one word score. The sum of all
attributions should equal score(input) - score(baseline) ("completeness"); the gap is
recorded as a check that 32 steps are enough.

Torch is imported inside the functions that need it, so the sampling and word-level helpers
can be tested without torch.
"""
import numpy as np
import pandas as pd

EXPLAIN_N = 500        # test headlines explained per model (250 per class)
EXPLAIN_SEED = 2027    # fixed: every model and seed explains the same headlines
IG_STEPS = 32          # points on the path from baseline to input (midpoint rule)
IG_BATCH = 16


def pick_items(test: pd.DataFrame, n: int = EXPLAIN_N, seed: int = EXPLAIN_SEED) -> pd.DataFrame:
    """Class-balanced random sample of the test set, the same for every run. Sorted by id."""
    rng = np.random.default_rng(seed)
    parts = []
    for c in (0, 1):
        pool = np.sort(test.loc[test["label"] == c, "id"].to_numpy())
        k = min(n // 2, len(pool))
        parts.append(rng.choice(pool, size=k, replace=False))
    keep = set(np.concatenate(parts).tolist())
    return test[test["id"].isin(keep)].sort_values("id").reset_index(drop=True)


def word_scores(text: str, offsets, word_ids, token_attr) -> tuple[list[str], list[float]]:
    """Sum token attributions into words.

    `offsets` are (start, end) character spans per token and `word_ids` the tokenizer's word
    index per token (None for [CLS], [SEP] and padding, which are left out). The word string is
    the original text span, so it matches the cue-word list exactly.
    """
    words, scores, spans = [], [], {}
    for (s, e), w, a in zip(offsets, word_ids, token_attr):
        if w is None:
            continue
        if w not in spans:
            spans[w] = [s, e, 0.0]
        spans[w][1] = max(spans[w][1], e)
        spans[w][2] += float(a)
    for w in sorted(spans):
        s, e, a = spans[w]
        words.append(text[s:e])
        scores.append(a)
    return words, scores


def cue_share(words, scores, cue_set) -> float:
    """Share of the total absolute attribution that falls on cue words (0 if no attribution)."""
    a = np.abs(np.asarray(scores, dtype=float))
    total = a.sum()
    if total == 0:
        return 0.0
    on_cue = sum(x for w, x in zip(words, a) if w in cue_set)
    return float(on_cue / total)


def integrated_gradients(model, tokenizer, texts, max_len: int, device, steps: int = IG_STEPS,
                         batch_size: int = IG_BATCH):
    """IG token attributions for the predicted class of a model with a plain sarcasm head.

    Returns a list with one dict per text: words, word scores, predicted class, P(sarcastic),
    and the completeness gap (sum of attributions minus the score difference it should equal).
    Runs in float32 (no autocast) so the gradients are exact enough for the check.
    """
    import torch

    model.eval()
    enc = tokenizer(list(texts), max_length=max_len, truncation=True, padding="max_length",
                    return_tensors="pt", return_offsets_mapping=True)
    emb = model.encoder.embeddings.word_embeddings
    pad_id = tokenizer.pad_token_id

    def score(x, mask):
        cls = model.encoder(inputs_embeds=x, attention_mask=mask).last_hidden_state[:, 0]
        return model.head(model.dropout(cls))

    out = []
    for i in range(0, len(texts), batch_size):
        ids = enc["input_ids"][i:i + batch_size].to(device)
        mask = enc["attention_mask"][i:i + batch_size].to(device)
        with torch.no_grad():
            x = emb(ids)
            x0 = emb(torch.full_like(ids, pad_id))
            logits_x = score(x, mask)
            logits_0 = score(x0, mask)
            target = logits_x.argmax(-1)
            p_sarc = torch.softmax(logits_x, -1)[:, 1]
        grad_sum = torch.zeros_like(x)
        for k in range(steps):
            alpha = (k + 0.5) / steps
            xi = (x0 + alpha * (x - x0)).detach().requires_grad_(True)
            sel = score(xi, mask).gather(1, target[:, None]).sum()
            (g,) = torch.autograd.grad(sel, xi)
            grad_sum += g
        attr = ((x - x0) * grad_sum / steps).sum(-1).detach().cpu().numpy()        # batch x tokens
        diff = (logits_x - logits_0).gather(1, target[:, None])[:, 0].cpu().numpy()  # what the sum should be
        for r in range(ids.shape[0]):
            j = i + r
            words, scores = word_scores(texts[j], enc["offset_mapping"][j].tolist(), enc.word_ids(j), attr[r])
            total = float(attr[r].sum())
            out.append({"words": words, "scores": scores, "pred": int(target[r]), "p_sarcastic": float(p_sarc[r]),
                        "attr_sum": total, "score_diff": float(diff[r]),
                        "completeness_gap": total - float(diff[r])})
    model.train()
    return out


def explain_run(model, tokenizer, test: pd.DataFrame, cue_words, max_len: int, device, out_path) -> dict:
    """Explain the fixed test sample, write one row per headline to `out_path`, return a summary."""
    items = pick_items(test)
    res = integrated_gradients(model, tokenizer, items["text"].tolist(), max_len, device)
    cue_set = set(cue_words)
    rows = []
    for (_, it), r in zip(items.iterrows(), res):
        rows.append({"id": it["id"], "label": int(it["label"]), "pred": r["pred"],
                     "p_sarcastic": round(r["p_sarcastic"], 6),
                     "cue_share": round(cue_share(r["words"], r["scores"], cue_set), 6),
                     "n_words": len(r["words"]), "n_cue_words": sum(w in cue_set for w in r["words"]),
                     "score_diff": round(r["score_diff"], 5), "completeness_gap": round(r["completeness_gap"], 5),
                     "words": " ".join(r["words"]),
                     "attributions": " ".join(f"{s:.5f}" for s in r["scores"])})
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    rel = (df["completeness_gap"].abs() / df["score_diff"].abs().clip(lower=1e-6))
    return {"n": int(len(df)), "ig_steps": IG_STEPS, "sample_seed": EXPLAIN_SEED,
            "mean_cue_share": round(float(df["cue_share"].mean()), 5),
            "mean_cue_share_with_cue": round(float(df.loc[df["n_cue_words"] > 0, "cue_share"].mean()), 5),
            "share_with_cue_word": round(float((df["n_cue_words"] > 0).mean()), 4),
            "median_rel_completeness_gap": round(float(rel.median()), 5)}
