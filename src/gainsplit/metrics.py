"""Scores for one split. No torch, so the TF-IDF arm can use it too."""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def scores(y_true, p_sarc, threshold: float = 0.5) -> dict:
    """Accuracy, macro-F1 (main metric), F1 of the sarcastic class (iSarcasmEval's
    official metric) and ROC-AUC (threshold-free)."""
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(p_sarc) >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "f1_sarcastic": float(f1_score(y_true, y_pred, pos_label=1)),
        "roc_auc": float(roc_auc_score(y_true, p_sarc)),
        "n": int(len(y_true)),
    }
