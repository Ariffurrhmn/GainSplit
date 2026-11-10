"""Train and score one run (one arm, one budget, one seed).

Run from the command line (the runner does this, one process per GPU):
    python -m gainsplit.train --config '{"arm": "A1", "budget": 10, "seed": 0}' --out results/x

Writes to <out>/runs/<run_id>/:
    result.json      settings, training length, time, scores, training curve, code commit, data check
    preds_<split>.csv  id, true label, probability of sarcastic (for paired significance tests)
A run whose result.json already exists is skipped, so an interrupted batch can be restarted.

One training loop serves every arm. The two switches (gan, helpers) in config.ARMS decide
which loss terms and which networks are used; nothing else differs between arms.
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
import torch.nn.functional as F
import transformers
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from .config import IMPLEMENTED_ARMS, RunConfig
from .data import find_data_dir, load_helper_labels, load_splits, verify
from .evaluate import predict, scores
from .models import SarcasmClassifier

EPS = 1e-8          # keeps log() finite when a probability reaches 0 or 1
LOG_EVERY = 100     # the training curve in result.json has one point per 100 steps


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
    """Endless supply of batches. Reshuffles after each pass, with a fixed random
    generator so the batch order depends only on the seed.

    `labels` is the sarcasm label (None for U, which has none). `extra` holds more
    per-row targets, e.g. the helper teacher labels. Each batch is
    (input_ids, attention_mask, sarcasm labels or None, dict of the extra targets)."""

    def __init__(self, enc: dict, labels, batch_size: int, seed: int, extra: dict | None = None):
        self.enc = enc
        self.n = enc["input_ids"].shape[0]
        self.labels = None if labels is None else torch.as_tensor(np.asarray(labels), dtype=torch.long)
        self.extra = {k: torch.as_tensor(np.asarray(v), dtype=torch.long) for k, v in (extra or {}).items()}
        self.batch_size = min(batch_size, self.n)
        self.gen = torch.Generator().manual_seed(seed)
        self.order, self.pos = None, self.n

    def __next__(self):
        if self.pos + self.batch_size > self.n:  # start a new pass; drop the short tail
            self.order, self.pos = torch.randperm(self.n, generator=self.gen), 0
        idx = self.order[self.pos:self.pos + self.batch_size]
        self.pos += self.batch_size
        y = None if self.labels is None else self.labels[idx]
        return (self.enc["input_ids"][idx], self.enc["attention_mask"][idx], y,
                {k: v[idx] for k, v in self.extra.items()})


def make_optimizer(named_params, cfg: RunConfig, steps: int):
    """AdamW with the usual BERT recipe: no weight decay on biases and LayerNorm,
    linear warm-up then linear decay to zero."""
    named_params = list(named_params)
    no_decay = ("bias", "LayerNorm.weight")
    groups = [
        {"params": [p for n, p in named_params if not any(k in n for k in no_decay)],
         "weight_decay": cfg.weight_decay},
        {"params": [p for n, p in named_params if any(k in n for k in no_decay)],
         "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(groups, lr=cfg.lr)
    sched = get_linear_schedule_with_warmup(opt, int(cfg.warmup_ratio * steps), steps)
    return opt, sched


def helper_losses(model, h, extra: dict):
    """Cross-entropy of the sentiment and emotion heads against the teacher labels."""
    return F.cross_entropy(model.sent_head(h), extra["sent"]), F.cross_entropy(model.emo_head(h), extra["emo"])


def plain_step(model, cfg, opt, sched, scaler, batch, n_l, use_fp16, device) -> dict:
    """A1 / A3: BERT + linear heads. A3 adds the helper losses over the L and U rows together."""
    ids, mask, y, extra = batch
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_fp16):
        h = model.dropout(model.encode(ids, mask))
        loss_sarc = F.cross_entropy(model.head(h[:n_l]), y)       # sarcasm: only the L rows have a label
        loss = loss_sarc
        stats = {"loss_sarc": loss_sarc.item()}
        if model.helpers:
            loss_sent, loss_emo = helper_losses(model, h, extra)  # helpers: every row has a teacher label
            loss = loss_sarc + cfg.helper_weight * (loss_sent + loss_emo)
            stats.update(loss_sent=loss_sent.item(), loss_emo=loss_emo.item())
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt)
    scaler.update()
    sched.step()
    opt.zero_grad(set_to_none=True)
    stats["loss"] = loss.item()
    return stats


def gan_step(model, cfg, opts, scheds, scaler, batch, n_l, use_fp16, device) -> dict:
    """A4 / A5 (GAN-BERT): one discriminator + encoder update, then one generator update.

    D is the discriminator: k real-class scores plus one "fake" score; p_fake is the
    probability it gives to "fake". The real vectors are the [CLS] vectors of the L+U batch."""
    opt_d, opt_g = opts
    ids, mask, y, extra = batch
    k = model.n_classes
    d_params = [p for n, p in model.named_parameters() if not n.startswith("generator.")]
    g_params = list(model.generator.parameters())

    # BERT runs in half precision; the generator, discriminator and all loss maths stay in float32.
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_fp16):
        real = model.encode(ids, mask)
    real = real.float()
    noise = torch.randn(real.shape[0], model.generator.noise_dim, device=real.device)
    fake = model.generator(noise)  # as many fake rows as real rows

    # --- discriminator (and encoder) update; the fake vectors are treated as constants ---
    logits_real, hid_real = model.disc(real)
    logits_fake, _ = model.disc(fake.detach())
    logp_real = F.log_softmax(logits_real, dim=-1)
    p_fake_real = logp_real.exp()[:, k]                          # should stay small
    p_fake_fake = F.softmax(logits_fake, dim=-1)[:, k]           # should be large
    loss_sup = -logp_real[:n_l].gather(1, y[:, None]).mean()     # probability of the true class, L rows only
    loss_unsup_real = -torch.log(1 - p_fake_real + EPS).mean()   # real rows (L and U) are "not fake"
    loss_unsup_fake = -torch.log(p_fake_fake + EPS).mean()       # generated rows are "fake"
    loss_d = loss_sup + loss_unsup_real + loss_unsup_fake
    stats = {"loss_sarc": loss_sup.item(), "loss_unsup_real": loss_unsup_real.item(),
             "loss_unsup_fake": loss_unsup_fake.item(),
             "p_fake_real": p_fake_real.mean().item(), "p_fake_fake": p_fake_fake.mean().item()}
    if model.helpers:  # A5: the same helper losses as A3, on the same real [CLS] vectors
        loss_sent, loss_emo = helper_losses(model, model.dropout(real), extra)
        loss_d = loss_d + cfg.helper_weight * (loss_sent + loss_emo)
        stats.update(loss_sent=loss_sent.item(), loss_emo=loss_emo.item())
    hid_real_mean = hid_real.detach().mean(0)  # feature-matching target; no gradient flows through it

    scaler.scale(loss_d).backward()
    scaler.unscale_(opt_d)
    torch.nn.utils.clip_grad_norm_(d_params, 1.0)
    scaler.step(opt_d)
    opt_d.zero_grad(set_to_none=True)

    # --- generator update: a second, cheap discriminator pass on the fake vectors only ---
    logits_g, hid_fake = model.disc(fake)
    p_fake_g = F.softmax(logits_g, dim=-1)[:, k]
    loss_g_unsup = -torch.log(1 - p_fake_g + EPS).mean()            # fool D: make fakes look "not fake"
    # feature matching: match D's mean hidden features; averaged over features, as in the released GAN-BERT code
    loss_g_fm = (hid_real_mean - hid_fake.mean(0)).pow(2).mean()
    loss_g = loss_g_unsup + loss_g_fm
    scaler.scale(loss_g).backward()
    opt_d.zero_grad(set_to_none=True)  # this backward also put gradients on D; drop them so D's next step is clean
    scaler.unscale_(opt_g)
    torch.nn.utils.clip_grad_norm_(g_params, 1.0)
    scaler.step(opt_g)
    scaler.update()
    opt_g.zero_grad(set_to_none=True)
    for sched in scheds:
        sched.step()
    stats.update(loss_g_unsup=loss_g_unsup.item(), loss_g_fm=loss_g_fm.item(),
                 loss_d=loss_d.item(), loss_g=loss_g.item(), loss=loss_d.item())
    return stats


def train_run(cfg: RunConfig, data_dir, out_root) -> dict:
    out_dir = Path(out_root) / "runs" / cfg.run_id
    if (out_dir / "result.json").exists():
        print(f"{cfg.run_id}: already done, skipped")
        return json.loads((out_dir / "result.json").read_text())
    if cfg.arm not in IMPLEMENTED_ARMS:
        raise NotImplementedError(f"{cfg.arm} is not written yet (implemented: {sorted(IMPLEMENTED_ARMS)})")
    gan, helpers = cfg.switches["gan"], cfg.switches["helpers"]
    if helpers and not cfg.helper_weight > 0:
        raise ValueError(f"{cfg.arm} uses helper heads, so helper_weight must be > 0 (got {cfg.helper_weight})")

    t0 = time.time()
    data_dir = Path(data_dir)
    manifest = verify(data_dir)
    splits = load_splits(data_dir, cfg.budget, cfg.seed)
    labeled, unlabeled = splits.labeled, splits.unlabeled
    use_u = (gan or helpers) and len(unlabeled) > 0  # A3/A4/A5 also read U (no sarcasm labels); A1 never does
    if helpers:
        labeled, unlabeled = load_helper_labels(data_dir, labeled, unlabeled)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = cfg.fp16 and device.type == "cuda"
    set_seed(cfg.seed)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    enc_l = encode(tokenizer, labeled["text"], cfg.max_len)
    eval_sets = {"dev": splits.dev}
    if cfg.eval_test:
        eval_sets.update(test=splits.test, isarc_test=splits.isarc_test)

    model = SarcasmClassifier(cfg.model_name, gan=gan, helpers=helpers).to(device)
    steps = cfg.steps(len(labeled))
    if gan:
        # one optimizer for BERT + discriminator (+ helper heads), a separate one for the generator
        opt_d, sched_d = make_optimizer(((n, p) for n, p in model.named_parameters()
                                         if not n.startswith("generator.")), cfg, steps)
        opt_g, sched_g = make_optimizer(model.generator.named_parameters(), cfg, steps)
    else:
        opt, sched = make_optimizer(model.named_parameters(), cfg, steps)
    scaler = torch.amp.GradScaler(device.type, enabled=use_fp16)

    extra_l = {"sent": labeled["sent"], "emo": labeled["emo"]} if helpers else None
    stream_l = BatchStream(enc_l, labeled["label"], cfg.batch_size, cfg.seed, extra_l)
    stream_u = None
    if use_u:  # its own seeded stream, so U's batch order does not disturb L's
        enc_u = encode(tokenizer, unlabeled["text"], cfg.max_len)
        extra_u = {"sent": unlabeled["sent"], "emo": unlabeled["emo"]} if helpers else None
        stream_u = BatchStream(enc_u, None, cfg.batch_size, cfg.seed + 1000, extra_u)

    print(f"{cfg.run_id}: |L|={len(labeled)} |U|={len(unlabeled)} steps={steps} device={device}")
    model.train()
    window, train_log = {}, []
    for step in range(1, steps + 1):
        ids, mask, y, extra = next(stream_l)
        n_l = ids.shape[0]
        if stream_u is not None:  # append one U batch of the same size after the L batch
            ids_u, mask_u, _, extra_u = next(stream_u)
            ids, mask = torch.cat([ids, ids_u]), torch.cat([mask, mask_u])
            extra = {k: torch.cat([v, extra_u[k]]) for k, v in extra.items()}
        batch = (ids.to(device), mask.to(device), y.to(device), {k: v.to(device) for k, v in extra.items()})
        if gan:
            stats = gan_step(model, cfg, (opt_d, opt_g), (sched_d, sched_g), scaler, batch, n_l, use_fp16, device)
        else:
            stats = plain_step(model, cfg, opt, sched, scaler, batch, n_l, use_fp16, device)
        for k, v in stats.items():
            window.setdefault(k, []).append(v)
        if step % LOG_EVERY == 0 or step == steps:
            point = {"step": step, **{k: round(float(np.mean(v)), 5) for k, v in window.items()}}
            train_log.append(point)
            print("  " + "  ".join(f"{k} {v:.4g}" if k != "step" else f"step {v}/{steps}" for k, v in point.items()))
            window = {}

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
        "n_labeled": len(labeled),
        "n_unlabeled": len(unlabeled),
        "steps": steps,
        "minutes": round((time.time() - t0) / 60, 2),
        "scores": results,
        "train_log": train_log,
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
