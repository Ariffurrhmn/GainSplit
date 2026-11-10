"""Scores and predictions. Same code for every arm and every split."""
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


@torch.no_grad()
def predict(model, enc: dict, device, batch_size: int = 256, fp16: bool = False) -> np.ndarray:
    """Probability of 'sarcastic' for every row of an encoded split."""
    model.eval()
    probs = []
    n = enc["input_ids"].shape[0]
    for i in range(0, n, batch_size):
        ids = enc["input_ids"][i:i + batch_size].to(device)
        mask = enc["attention_mask"][i:i + batch_size].to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=fp16):
            logits = model(ids, mask)
        probs.append(torch.softmax(logits.float(), dim=-1)[:, 1].cpu())
    model.train()
    return torch.cat(probs).numpy()


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
