"""Scores and predictions. Same code for every arm and every split."""
import numpy as np
import torch

from .metrics import scores  # noqa: F401  (re-exported: train.py imports it from here)


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
