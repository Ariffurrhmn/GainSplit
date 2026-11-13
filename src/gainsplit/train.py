"""Train and score one run (one arm, one budget, one seed).

Run from the command line (the runner does this, one process per GPU):
    python -m gainsplit.train --config '{"arm": "A1", "budget": 10, "seed": 0}' --out results/x

Writes to <out>/runs/<run_id>/:
    result.json      settings, training length, time, scores, training curve, code commit, data check
    preds_<split>.csv  id, true label, probability of sarcastic (for paired significance tests)
A run whose result.json already exists is skipped, so an interrupted batch can be restarted.

One training loop serves every arm. The switches in config.ARMS decide which loss terms,
networks and data are used; nothing else differs between arms. (A0 is not neural: it is
handed to tfidf.py.)
"""
import argparse
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from .config import IMPLEMENTED_ARMS, RunConfig
from .data import (EMO_P, SENT_P, apply_helper_variant, find_data_dir, helper_label_counts, load_cue_words,
                   load_helper_labels, load_isarcasm_splits, load_masked_tests, load_pairs, load_splits,
                   permute_helper_labels, verify, verify_pass3)
from .evaluate import predict, scores
from .models import SarcasmClassifier
from .tfidf import run_tfidf
from .util import git_commit

PSEUDO_THRESHOLD = 0.9  # A6 keeps U items the teacher is at least this sure about
EPS = 1e-8          # keeps log() finite when a probability reaches 0 or 1
LOG_EVERY = 100     # the training curve in result.json has one point per 100 steps


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
        # class indices stay integers; soft targets (teacher probabilities) are floats
        self.extra = {k: torch.as_tensor(np.asarray(v), dtype=torch.float32 if np.asarray(v).dtype.kind == "f" else torch.long)
                      for k, v in (extra or {}).items()}
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


class GroupBatchStream(BatchStream):
    """Pass 4: like BatchStream, but rows come in fixed groups that always share a batch.

    The rows must be sorted so each group (a seed_id) is contiguous and every group has the same
    size: 2 for whole pairs, 1 for the unpaired control. Groups are shuffled, never split, so a
    pair's sarcastic and serious sides are always trained on in the same step. A batch holds
    batch_size // group_size groups."""

    def __init__(self, enc: dict, labels, groups, batch_size: int, seed: int):
        super().__init__(enc, labels, batch_size, seed)
        codes = pd.factorize(pd.Series(np.asarray(groups)))[0]
        sizes = np.bincount(codes)
        k = int(sizes[0])
        if not (sizes == k).all():
            raise ValueError("GroupBatchStream: every group must have the same number of rows")
        if not (codes == np.repeat(np.arange(len(sizes)), k)).all():
            raise ValueError("GroupBatchStream: rows of a group must be next to each other")
        self.members = torch.arange(self.n).reshape(len(sizes), k)
        self.n_groups = len(sizes)
        self.per_batch = max(1, min(batch_size // k, self.n_groups))
        self.order, self.pos = None, self.n_groups

    def __next__(self):
        if self.pos + self.per_batch > self.n_groups:
            self.order, self.pos = torch.randperm(self.n_groups, generator=self.gen), 0
        idx = self.members[self.order[self.pos:self.pos + self.per_batch]].reshape(-1)
        self.pos += self.per_batch
        y = None if self.labels is None else self.labels[idx]
        return (self.enc["input_ids"][idx], self.enc["attention_mask"][idx], y,
                {k: v[idx] for k, v in self.extra.items()})


def make_optimizer(named_params, cfg: RunConfig, steps: int, lr: float = 0.0):
    """AdamW with the usual BERT recipe: no weight decay on biases and LayerNorm,
    linear warm-up then linear decay to zero. `lr` overrides cfg.lr when > 0."""
    named_params = list(named_params)
    no_decay = ("bias", "LayerNorm.weight")
    groups = [
        {"params": [p for n, p in named_params if not any(k in n for k in no_decay)],
         "weight_decay": cfg.weight_decay},
        {"params": [p for n, p in named_params if any(k in n for k in no_decay)],
         "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(groups, lr=lr if lr > 0 else cfg.lr)
    sched = get_linear_schedule_with_warmup(opt, int(cfg.warmup_ratio * steps), steps)
    return opt, sched


def helper_losses(model, h, extra: dict, variant: str = "") -> dict:
    """Cross-entropy of the sentiment and emotion heads against the teacher labels.

    Returns {"loss_sent": ..., "loss_emo": ...}. Pass 3 variants: `soft` uses the teacher
    probabilities as targets; `sentonly` / `emoonly` leave the other head out of the loss
    (it gets no gradient, so it is not trained)."""
    def ce(logits, hard, soft):
        if variant == "soft":  # -sum_c p_c log q_c, averaged over rows
            return -(extra[soft] * F.log_softmax(logits.float(), dim=-1)).sum(-1).mean()
        return F.cross_entropy(logits, extra[hard])
    losses = {}
    if variant != "emoonly":
        losses["loss_sent"] = ce(model.sent_head(h), "sent", "sent_soft")
    if variant != "sentonly":
        losses["loss_emo"] = ce(model.emo_head(h), "emo", "emo_soft")
    return losses


def plain_step(model, cfg, opt, sched, scaler, batch, n_l, use_fp16, device) -> dict:
    """A1 / A3: BERT + linear heads. A3 adds the helper losses over the L and U rows together.

    `batch`'s 5th element `w` is a per-row sarcasm-loss weight (Pass 5 `pair_weight`; None for
    every other arm, unchanged behaviour): weighted mean instead of `F.cross_entropy`'s plain
    mean, so Pass 1-4 runs (where `w` is always None) are untouched."""
    ids, mask, y, extra, w = batch
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_fp16):
        h = model.dropout(model.encode(ids, mask))
        logits = model.head(h[:n_l])
        if w is None:
            loss_sarc = F.cross_entropy(logits, y)                # sarcasm: only the L rows have a label
        else:
            per_row = F.cross_entropy(logits, y, reduction="none")
            loss_sarc = (per_row * w).sum() / w.sum()
        loss = loss_sarc
        stats = {"loss_sarc": loss_sarc.item()}
        if model.helpers:
            hl = helper_losses(model, h, extra, cfg.helper_variant)  # helpers: every row has a teacher label
            loss = loss_sarc + cfg.helper_weight * sum(hl.values())
            stats.update({k: v.item() for k, v in hl.items()})
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt)
    scaler.update()
    sched.step()
    opt.zero_grad(set_to_none=True)
    stats["loss"] = loss.item()
    return stats


def fgm_step(model, cfg, opt, sched, scaler, batch, n_l, use_fp16, device) -> dict:
    """A7: plain BERT on L + one FGM (fast gradient method) step on the word embeddings.
    A11 (FGM + pairs): the same, with the pair rows' loss optionally down-weighted (`w`, Pass 5
    `pair_weight`; None for A7 and every pre-Pass-5 arm, unchanged behaviour).

    1. Clean loss, backward: the word-embedding matrix now has a gradient g.
    2. Push the embeddings by r = epsilon * g / ||g||  (L2-normalised gradient).
    3. Adversarial loss on the pushed embeddings, backward: its gradients add to the clean ones.
    4. Put the embeddings back, then take one optimizer step on (clean + adversarial) loss.
    """
    ids, mask, y, _, w = batch
    emb = model.encoder.embeddings.word_embeddings.weight

    def sarcasm_loss():
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_fp16):
            logits = model.head(model.dropout(model.encode(ids, mask)))
            if w is None:
                return F.cross_entropy(logits, y[:n_l])
            per_row = F.cross_entropy(logits, y[:n_l], reduction="none")
            return (per_row * w).sum() / w.sum()

    loss_clean = sarcasm_loss()
    scaler.scale(loss_clean).backward()
    g = emb.grad
    norm = g.norm() if g is not None else None
    loss_adv = loss_clean.detach()  # stand-in if the gradient is unusable (zero, or overflowed in fp16)
    if norm is not None and torch.isfinite(norm) and norm > 0:
        r = cfg.fgm_epsilon * g / norm  # computed before the next backward changes g
        backup = emb.data.clone()
        emb.data.add_(r)
        loss_adv = sarcasm_loss()
        scaler.scale(loss_adv).backward()
        emb.data.copy_(backup)
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt)
    scaler.update()
    sched.step()
    opt.zero_grad(set_to_none=True)
    return {"loss_sarc": loss_clean.item(), "loss_adv": loss_adv.item(),
            "loss": loss_clean.item() + loss_adv.item()}


def gan_step(model, cfg, opts, scheds, scaler, batch, n_l, use_fp16, device) -> dict:
    """A4 / A5 (GAN-BERT): one discriminator + encoder update, then one generator update.

    D is the discriminator: k real-class scores plus one "fake" score; p_fake is the
    probability it gives to "fake". The real vectors are the [CLS] vectors of the L+U batch."""
    opt_d, opt_g = opts
    ids, mask, y, extra, _w = batch   # _w: always None here (GAN arms never use pairs/pair_weight)
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
        hl = helper_losses(model, model.dropout(real), extra, cfg.helper_variant)
        loss_d = loss_d + cfg.helper_weight * sum(hl.values())
        stats.update({k: v.item() for k, v in hl.items()})
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


def fit(cfg, tokenizer, labeled, pool, steps, device, use_fp16, *, gan=False, helpers=False, fgm=False,
        pool_groups=None, pool_batch=None, pool_is_pairs=False):
    """Train a fresh model for `steps` steps; return (model, training curve).

    `labeled` is L (with sent/emo columns if `helpers`), or None for A15 (no gold rows: every
    batch is a pool batch only). `pool` is None or a frame of extra rows that join every step
    as a second batch: U for A3/A4/A5/A8, the pseudo-labeled rows for A6, or the generated
    pairs for A10-A15 (A6 and pair rows carry a `label` column, U does not).
    `pool_groups` (Pass 4): group key per pool row; groups are never split across batches.
    `pool_batch`: rows per pool batch (default: cfg.batch_size).
    `pool_is_pairs` (Pass 5): True only for the pair arms (A10-A15); applies `cfg.pair_weight`
    to the pool rows' share of the sarcasm loss (1.0 = unweighted, identical to Pass 4). Never
    True for A6 (pseudo-labels keep weight 1, as in Pass 1-4) or the helper/GAN arms' U rows
    (which carry no sarcasm label at all, so no weight applies to them).
    """
    set_seed(cfg.seed)
    model = SarcasmClassifier(cfg.model_name, gan=gan, helpers=helpers).to(device)
    if gan:
        # one optimizer for BERT + discriminator (+ helper heads), a separate one for the generator
        opt_d, sched_d = make_optimizer(((n, p) for n, p in model.named_parameters()
                                         if not n.startswith("generator.")), cfg, steps)
        opt_g, sched_g = make_optimizer(model.generator.named_parameters(), cfg, steps, lr=cfg.gen_lr)
    else:
        opt, sched = make_optimizer(model.named_parameters(), cfg, steps)
    scaler = torch.amp.GradScaler(device.type, enabled=use_fp16)

    def helper_targets(df):  # hard class indices, plus the teacher probabilities for the `soft` variant
        t = {"sent": df["sent"], "emo": df["emo"]}
        if cfg.helper_variant == "soft":
            t.update(sent_soft=df[SENT_P].to_numpy(), emo_soft=df[EMO_P].to_numpy())
        return t

    stream_l = None
    if labeled is not None:
        enc_l = encode(tokenizer, labeled["text"], cfg.max_len)
        extra_l = helper_targets(labeled) if helpers else None
        stream_l = BatchStream(enc_l, labeled["label"], cfg.batch_size, cfg.seed, extra_l)
    stream_u = None
    if pool is not None:  # its own seeded stream, so the pool's batch order does not disturb L's
        enc_u = encode(tokenizer, pool["text"], cfg.max_len)
        extra_u = helper_targets(pool) if helpers else None
        y_pool = pool["label"] if "label" in pool else None
        if pool_groups is not None:
            stream_u = GroupBatchStream(enc_u, y_pool, pool_groups, pool_batch or cfg.batch_size, cfg.seed + 1000)
        else:
            stream_u = BatchStream(enc_u, y_pool, pool_batch or cfg.batch_size, cfg.seed + 1000, extra_u)
    if stream_l is None and stream_u is None:
        raise ValueError("nothing to train on: no labeled rows and no pool")

    pair_weight = cfg.pair_weight if pool_is_pairs else 1.0
    model.train()
    window, train_log = {}, []
    for step in range(1, steps + 1):
        if stream_l is None:  # A15: generated pairs only
            ids, mask, y, extra = next(stream_u)
            w = torch.full_like(y, pair_weight, dtype=torch.float32) if pair_weight != 1.0 else None
        else:
            ids, mask, y, extra = next(stream_l)
            w = torch.ones_like(y, dtype=torch.float32) if pair_weight != 1.0 else None
        if stream_u is not None and stream_l is not None:  # append one pool batch after the L batch
            ids_u, mask_u, y_u, extra_u = next(stream_u)
            ids, mask = torch.cat([ids, ids_u]), torch.cat([mask, mask_u])
            extra = {k: torch.cat([v, extra_u[k]]) for k, v in extra.items()}
            if y_u is not None:  # pseudo-labels / pair labels are used like labels: one loss over all labeled rows
                y = torch.cat([y, y_u])
                if w is not None:  # pool rows get pair_weight (only set when pool_is_pairs and weight != 1)
                    w = torch.cat([w, torch.full_like(y_u, pair_weight, dtype=torch.float32)])
        n_l = y.shape[0]  # rows that have a sarcasm label (L rows, plus the pseudo-labeled rows for A6)
        batch = (ids.to(device), mask.to(device), y.to(device), {k: v.to(device) for k, v in extra.items()},
                None if w is None else w.to(device))
        if gan:
            stats = gan_step(model, cfg, (opt_d, opt_g), (sched_d, sched_g), scaler, batch, n_l, use_fp16, device)
        elif fgm:
            stats = fgm_step(model, cfg, opt, sched, scaler, batch, n_l, use_fp16, device)
        else:
            stats = plain_step(model, cfg, opt, sched, scaler, batch, n_l, use_fp16, device)
        for k, v in stats.items():
            window.setdefault(k, []).append(v)
        if step % LOG_EVERY == 0 or step == steps:
            point = {"step": step, **{k: round(float(np.mean(v)), 5) for k, v in window.items()}}
            train_log.append(point)
            print("  " + "  ".join(f"{k} {v:.4g}" if k != "step" else f"step {v}/{steps}" for k, v in point.items()))
            window = {}
    return model, train_log


def pick_pseudo_labels(p_sarc, threshold: float = PSEUDO_THRESHOLD):
    """A6: from the teacher's P(sarcastic) on U, keep items with confidence >= threshold,
    then take the same number from each class (the most confident items first).
    Returns (indices into U, their labels, a summary of what was kept)."""
    p_sarc = np.asarray(p_sarc)
    pred = (p_sarc >= 0.5).astype(int)
    conf = np.maximum(p_sarc, 1 - p_sarc)
    sure = conf >= threshold
    counts = {c: int((sure & (pred == c)).sum()) for c in (0, 1)}
    n_per_class = min(counts.values())
    idx, labels = [], []
    for c in (0, 1):
        cand = np.where(sure & (pred == c))[0]
        cand = cand[np.argsort(-conf[cand], kind="stable")][:n_per_class]
        idx.extend(cand.tolist())
        labels.extend([c] * len(cand))
    info = {"threshold": threshold, "n_unlabeled": int(len(p_sarc)),
            "n_confident": int(sure.sum()), "confident_class_counts": {str(c): v for c, v in counts.items()},
            "n_kept": len(idx), "kept_class_counts": {str(c): n_per_class for c in (0, 1)},
            "fallback_l_only": len(idx) == 0}
    return np.array(idx, dtype=int), np.array(labels, dtype=int), info


def train_run(cfg: RunConfig, data_dir, out_root) -> dict:
    out_dir = Path(out_root) / "runs" / cfg.run_id
    if (out_dir / "result.json").exists():
        print(f"{cfg.run_id}: already done, skipped")
        return json.loads((out_dir / "result.json").read_text())
    if cfg.arm == "A0":  # TF-IDF floor: no neural network
        return run_tfidf(cfg, data_dir, out_root)
    if cfg.arm not in IMPLEMENTED_ARMS:
        raise NotImplementedError(f"{cfg.arm} is not written yet (implemented: {sorted(IMPLEMENTED_ARMS)})")
    sw = cfg.switches
    gan, helpers = sw["gan"], sw["helpers"]
    if helpers and not cfg.helper_weight > 0:
        raise ValueError(f"{cfg.arm} uses helper heads, so helper_weight must be > 0 (got {cfg.helper_weight})")

    t0 = time.time()
    data_dir = Path(data_dir)
    manifest = verify(data_dir)
    # only runs that read the Pass 3 files check them, so older plans run without them
    masked = cfg.eval_test and cfg.eval_masked
    if masked or cfg.explain or cfg.helper_variant in ("vader", "teacher2"):
        manifest["outputs"].update(verify_pass3(data_dir)["outputs"])
    if cfg.train_data == "isarcasm":
        splits = load_isarcasm_splits(data_dir)
    else:
        splits = load_splits(data_dir, cfg.budget, cfg.seed)
    labeled, unlabeled = splits.labeled, splits.unlabeled
    # A3/A4/A5/A8 also read U (no sarcasm labels); A1, A2, A7 never do (A6 builds its own pool below)
    use_u = (gan or helpers) and not sw.get("l_only", False) and len(unlabeled) > 0
    if helpers:
        labeled, unlabeled = load_helper_labels(data_dir, labeled, unlabeled)
    if helpers:
        labeled, unlabeled = apply_helper_variant(cfg.helper_variant, data_dir, labeled, unlabeled, cfg.seed)
    if sw.get("shuffle"):  # A8 / A9: break the link between each headline and its helper labels
        labeled, unlabeled = permute_helper_labels(labeled, unlabeled, cfg.seed)
    label_counts = helper_label_counts(labeled, unlabeled) if helpers else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = cfg.fp16 and device.type == "cuda"
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    eval_sets = {"dev": splits.dev}
    if cfg.eval_test:
        eval_sets.update(test=splits.test, isarc_test=splits.isarc_test)
        if cfg.eval_masked:  # Pass 3: the same test headlines with cue words / random words masked
            eval_sets.update(load_masked_tests(data_dir))

    steps = cfg.steps(len(labeled))
    print(f"{cfg.run_id}: |L|={len(labeled)} |U|={len(unlabeled)} steps={steps} device={device}")
    pool, pseudo_info = (unlabeled if use_u else None), None
    pool_groups, pool_batch, pair_info = None, None, None
    if sw.get("pairs"):
        # Pass 4-5: generated pairs join every step; the sides of a pair always share a batch.
        pool, pair_info = load_pairs(sw["pairs"], data_dir, cleaned_drop_pct=cfg.cleaned_drop_pct)
        pool_groups, pool_batch = pool["seed_id"].to_numpy(), cfg.pair_batch
        print(f"  pairs ({sw['pairs']}): {pair_info['n_rows']} rows from {pair_info['n_seeds']} seeds, "
              f"{pool_batch} rows per step, pair_weight={cfg.pair_weight}")
    fit_labeled = None if sw.get("no_gold") else labeled   # A15: the budget only sets the step count
    if sw.get("pseudo"):
        # A6: the teacher is an A1 model (same seed, same steps). It labels U; the student trains on L + pseudo-labels.
        print("  A6 teacher (plain A1 training):")
        teacher, _ = fit(cfg, tokenizer, labeled, None, steps, device, use_fp16)
        p_u = (predict(teacher, encode(tokenizer, unlabeled["text"], cfg.max_len), device, fp16=use_fp16)
               if len(unlabeled) > 0 else np.zeros(0))
        p_dev = predict(teacher, encode(tokenizer, splits.dev["text"], cfg.max_len), device, fp16=use_fp16)
        del teacher
        idx, y_pseudo, pseudo_info = pick_pseudo_labels(p_u)
        pseudo_info["teacher_dev_macro_f1"] = round(scores(splits.dev["label"], p_dev)["macro_f1"], 4)
        print(f"  A6 pseudo-labels kept: {pseudo_info['n_kept']} of {len(unlabeled)} "
              f"(confident {pseudo_info['n_confident']}, per class {pseudo_info['confident_class_counts']})"
              + ("; none kept, the student uses L only" if pseudo_info["fallback_l_only"] else ""))
        if len(idx) > 0:
            pool = unlabeled.iloc[idx].reset_index(drop=True)
            pool["label"] = y_pseudo
        print("  A6 student (fresh BERT):")
    model, train_log = fit(cfg, tokenizer, fit_labeled, pool, steps, device, use_fp16,
                           gan=gan, helpers=helpers, fgm=bool(sw.get("fgm")),
                           pool_groups=pool_groups, pool_batch=pool_batch, pool_is_pairs=bool(sw.get("pairs")))

    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, df in eval_sets.items():
        p = predict(model, encode(tokenizer, df["text"], cfg.max_len), device, fp16=use_fp16)
        results[name] = scores(df["label"], p)
        pd.DataFrame({"id": df["id"], "label": df["label"], "p_sarcastic": p.round(6)}).to_csv(
            out_dir / f"preds_{name}.csv", index=False)
    explain_info = None
    if cfg.explain:  # Pass 4: IG word attributions on a fixed test sample (explain.csv)
        from .explain import explain_run
        t_x = time.time()
        explain_info = explain_run(model, tokenizer, splits.test, load_cue_words(data_dir), cfg.max_len, device,
                                   out_dir / "explain.csv")
        explain_info["minutes"] = round((time.time() - t_x) / 60, 2)
        print(f"  explain: mean cue share {explain_info['mean_cue_share']:.4f} on {explain_info['n']} test headlines "
              f"({explain_info['minutes']} min)")

    record = {
        "run_id": cfg.run_id,
        "config": cfg.to_dict(),
        "n_labeled": 0 if sw.get("no_gold") else len(labeled),
        "n_unlabeled": len(unlabeled),
        "uses_u": pool is not None and not sw.get("pairs"),
        "pairs": pair_info,       # Pass 4 pair arms: which pair set, its size and file hashes
        "explain": explain_info,  # Pass 4 explain runs: IG summary (details in explain.csv)
        "steps": steps,
        "minutes": round((time.time() - t0) / 60, 2),
        "scores": results,
        "train_log": train_log,
        "pseudo": pseudo_info,  # A6 only: pseudo-labels kept, with class counts
        "helper_label_counts": label_counts,  # helper arms only: hard labels over L + U after flipping/replacing/shuffling
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
