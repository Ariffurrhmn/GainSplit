import pandas as pd

from gainsplit.masking import MASK, cue_mask, fit_cue_words, random_mask


def _train():
    sarc = ["area man local tragedy", "area man wins local award", "nation area man shrugs"] * 4
    real = ["trump signs new bill", "senate passes new bill", "trump new speech today"] * 4
    return pd.DataFrame({"text": sarc + real, "label": [1] * 12 + [0] * 12})


def test_fit_cue_words_sides_and_determinism():
    a = fit_cue_words(_train(), 3)
    b = fit_cue_words(_train(), 3)
    assert a == b
    assert len(a["sarcastic"]) == len(a["non_sarcastic"]) == 3
    assert all(w > 0 for _, w in a["sarcastic"])
    assert all(w < 0 for _, w in a["non_sarcastic"])
    assert "area" in [w for w, _ in a["sarcastic"]] or "man" in [w for w, _ in a["sarcastic"]]
    assert "trump" in [w for w, _ in a["non_sarcastic"]] or "bill" in [w for w, _ in a["non_sarcastic"]]


def test_cue_mask_only_listed_words():
    texts = ["area man wins", "trump signs bill", "nothing here"]
    out, counts = cue_mask(texts, ["area", "bill"])
    assert out == [f"{MASK} man wins", f"trump signs {MASK}", "nothing here"]
    assert counts == [1, 1, 0]


def test_random_mask_same_counts_and_deterministic():
    texts = ["a b c d e f", "g h i j", "k l m", "no cues"]
    counts = [3, 1, 0, 0]
    r1 = random_mask(texts, counts, 20261110)
    r2 = random_mask(texts, counts, 20261110)
    assert r1 == r2
    assert [t.split().count(MASK) for t in r1] == counts
    assert r1[2] == texts[2] and r1[3] == texts[3]
    for o, m in zip(texts, r1):
        assert len(o.split()) == len(m.split())
        assert all(a == b or b == MASK for a, b in zip(o.split(), m.split()))


def test_masks_have_equal_counts_end_to_end():
    texts = ["area man wins local award", "trump new bill", "plain words only"]
    cm, counts = cue_mask(texts, ["area", "man", "bill"])
    rm = random_mask(texts, counts, 1)
    assert [t.split().count(MASK) for t in cm] == [t.split().count(MASK) for t in rm] == counts
    assert len(cm) == len(rm) == len(texts)
