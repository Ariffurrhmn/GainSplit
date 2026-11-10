"""Train and score one run (one arm, one budget, one seed).

Run from the command line (the runner does this, one process per GPU):
    python -m gainsplit.train --config '{"arm": "A1", "budget": 10, "seed": 0}' --out results/x

Writes to <out>/runs/<run_id>/:
    result.json      settings, training length, time, scores, code commit, data check
    preds_<split>.csv  id, true label, probability of sarcastic (for paired significance tests)
A run whose result.json already exists is skipped, so an interrupted batch can be restarted.
"""
import argparse
import json
import platform
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
from torch import nn
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from .config import IMPLEMENTED_ARMS, RunConfig
from .data import find_data_dir, load_splits, verify
from .evaluate import predict, scores
from .models import SarcasmClassifier


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def git_commit() -> str:
    """Commit of the code that produced this run, so every result can be traced."""
    try:
        repo = Path(__file__).resolve().parents[2]
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def encode(tokenizer, texts, max_len: int) -> dict:
    """Turn headlines into BERT token ids, padded/cut to max_len."""
    enc = tokenizer(list(texts), max_length=max_len, truncation=True, padding="max_length", return_tensors="pt")
    return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}


class BatchStream:
    """Endless supply of batches from L. Reshuffles after each pass, with a fixed
    random generator so the batch order depends only on the seed."""

    def __init__(self, enc: dict, labels, batch_size: int, seed: int):
        self.enc, self.labels = enc, torch.as_tensor(np.asarray(labels), dtype=torch.long)
        self.batch_size = min(batch_size, len(self.labels))
        self.gen = torch.Generator().manual_seed(seed)
        self.order, self.pos = None, len(self.labels)

    def __next__(self):
        if self.pos + self.batch_size > len(self.labels):  # start a new pass; drop the short tail
            self.order, self.pos = torch.randperm(len(self.labels), generator=self.gen), 0
        idx = self.order[self.pos:self.pos + self.batch_size]
        self.pos += self.batch_size
        return self.enc["input_ids"][idx], self.enc["attention_mask"][idx], self.labels[idx]


def make_optimizer(model, cfg: RunConfig, steps: int):
    """AdamW with the usual BERT recipe: no weight decay on biases and LayerNorm,
    linear warm-up then linear decay to zero."""
    no_decay = ("bias", "LayerNorm.weight")
    groups = [
        {"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)],
         "weight_decay": cfg.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)],
         "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(groups, lr=cfg.lr)
    sched = get_linear_schedule_with_warmup(opt, int(cfg.warmup_ratio * steps), steps)
    return opt, sched


def train_run(cfg: RunConfig, data_dir, out_root) -> dict:
    out_dir = Path(out_root) / "runs" / cfg.run_id
    if (out_dir / "result.json").exists():
        print(f"{cfg.run_id}: already done, skipped")
        return json.loads((out_dir / "result.json").read_text())
    if cfg.arm not in IMPLEMENTED_ARMS:
        raise NotImplementedError(f"{cfg.arm} is not written yet (implemented: {sorted(IMPLEMENTED_ARMS)})")

    t0 = time.time()
    data_dir = Path(data_dir)
    manifest = verify(data_dir)
    splits = load_splits(data_dir, cfg.budget, cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = cfg.fp16 and device.type == "cuda"
    set_seed(cfg.seed)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    enc_l = encode(tokenizer, splits.labeled["text"], cfg.max_len)
    eval_sets = {"dev": splits.dev}
    if cfg.eval_test:
        eval_sets.update(test=splits.test, isarc_test=splits.isarc_test)

    model = SarcasmClassifier(cfg.model_name).to(device)
    steps = cfg.steps(len(splits.labeled))
    opt, sched = make_optimizer(model, cfg, steps)
    scaler = torch.amp.GradScaler(device.type, enabled=use_fp16)
    loss_fn = nn.CrossEntropyLoss()
    stream = BatchStream(enc_l, splits.labeled["label"], cfg.batch_size, cfg.seed)

    print(f"{cfg.run_id}: |L|={len(splits.labeled)} |U|={len(splits.unlabeled)} steps={steps} device={device}")
    model.train()
    running = 0.0
    for step in range(1, steps + 1):
        ids, mask, y = (t.to(device) for t in next(stream))
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_fp16):
            loss = loss_fn(model(ids, mask), y)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        opt.zero_grad(set_to_none=True)
        running += loss.item()
        if step % 100 == 0 or step == steps:
            print(f"  step {step}/{steps}  loss {running / (100 if step % 100 == 0 else step % 100):.4f}")
            running = 0.0

    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, df in eval_sets.items():
        p = predict(model, encode(tokenizer, df["text"], cfg.max_len), device, fp16=use_fp16)
        results[name] = scores(df["label"], p)
        pd.DataFrame({"id": df["id"], "label": df["label"], "p_sarcastic": p.round(6)}).to_csv(
            out_dir / f"preds_{name}.csv", index=False)

    record = {
        "run_id": cfg.run_id,
        "config": cfg.to_dict(),
        "n_labeled": len(splits.labeled),
        "n_unlabeled": len(splits.unlabeled),
        "steps": steps,
        "minutes": round((time.time() - t0) / 60, 2),
        "scores": results,
        "commit": git_commit(),
        "data_manifest_sha256": {k: v["sha256"] for k, v in manifest["outputs"].items()},
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "transformers": transformers.__version__},
    }
    tmp = out_dir / "result.json.tmp"
    tmp.write_text(json.dumps(record, indent=2))
    tmp.replace(out_dir / "result.json")  # written last and in one go: its presence means "finished"
    summary = ", ".join(f"{k} macro-F1 {v['macro_f1']:.4f}" for k, v in results.items())
    print(f"{cfg.run_id}: done in {record['minutes']} min; {summary}")
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="RunConfig as a JSON string")
    ap.add_argument("--data", default=None, help="processed data folder (default: search /kaggle/input)")
    ap.add_argument("--out", required=True, help="results folder")
    args = ap.parse_args()
    cfg = RunConfig(**json.loads(args.config))
    train_run(cfg, args.data or find_data_dir(), args.out)


if __name__ == "__main__":
    main()
