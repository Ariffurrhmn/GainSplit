"""Pass 8a/8b: task-adaptive pretraining (TAPT) of BERT on the Headlines training texts, no labels.

Two self-supervised objectives, everything else held equal (same texts, steps, batch, lr, schedule):

  mlm   masked-word prediction, BERT's own pretraining task (Gururangan et al. 2020 call this TAPT
        when it is run on the task's own unlabeled texts).
  jepa  joint-embedding prediction (I-JEPA style, Assran et al. 2023, adapted to text here):
        - spans of words are hidden from a *context encoder* (no other word may attend to them);
        - a small *predictor* gets the context encoder's output, a learned [MASK] vector at each
          hidden position (plus its position), and must predict ...
        - ... the *target encoder's* vectors for the hidden words. The target encoder sees the whole
          headline; it is not trained by gradient but follows the context encoder as a slow moving
          average (EMA), which keeps the two from collapsing to a constant.
        The loss is in vector space, not over words: the model is never asked to name the hidden word.
        For text this is close to data2vec (Baevski et al. 2022); the difference is that here the
        context encoder never sees the hidden positions at all and a separate predictor fills them in.

Only the `text` column of headlines_train.csv is read: no sarcasm label is used for training. Dev and
test headlines are never used for training (dev texts give a held-out loss for the curve).

The saved checkpoint is an ordinary BERT encoder folder (config + weights + tokenizer), so a training
run uses it simply by loading it in place of bert-base-uncased (RunConfig.init_dir).

Command line (the notebooks run it as a separate process so the GPU memory is freed afterwards):
    python -m gainsplit.ssl_pretrain --objective mlm --data <processed> --out /kaggle/working/ckpt/mlm
Writes <out>/pretrain_info.json last; if it exists, the pretraining is skipped (restart-safe).
"""
import argparse
import copy
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

OBJECTIVES = ("mlm", "jepa")
PROBE_BUDGETS = (1, 10)
PROBE_SEEDS = (0, 1, 2, 3, 4)


@dataclass
class PretrainConfig:
    objective: str                     # "mlm" or "jepa"
    seed: int = 0
    steps: int = 10000                 # about 28 passes over the 22,800 training headlines at batch 64
    batch_size: int = 64
    lr: float = 1e-4                   # TAPT recipe (Gururangan et al. 2020); same for both objectives
    warmup_ratio: float = 0.06
    weight_decay: float = 0.01
    max_len: int = 48                  # same as fine-tuning
    model_name: str = "bert-base-uncased"
    fp16: bool = True
    log_every: int = 250
    # mlm
    mask_prob: float = 0.15            # BERT's 15%, with the usual 80/10/10 replacement
    # jepa
    target_ratio: float = 0.25         # share of a headline's words hidden from the context encoder
    n_spans: int = 2                   # hidden words come in this many spans
    ema_start: float = 0.996           # EMA momentum of the target encoder, rising linearly ...
    ema_end: float = 1.0               # ... to 1.0 (I-JEPA's schedule)
    pred_layers: int = 2               # predictor: a small transformer on top of the context encoder
    pred_heads: int = 12

    def __post_init__(self):
        if self.objective not in OBJECTIVES:
            raise ValueError(f"objective must be one of {OBJECTIVES} (got {self.objective!r})")
        if not 0 < self.mask_prob < 1 or not 0 < self.target_ratio < 1:
            raise ValueError("mask_prob and target_ratio must be between 0 and 1")
        if self.steps < 1 or self.batch_size < 2 or self.n_spans < 1:
            raise ValueError("steps >= 1, batch_size >= 2 and n_spans >= 1 are required")


# ------------------------------------------------------------------ masking (numpy, no torch)

def content_positions(attention_mask_row: np.ndarray) -> np.ndarray:
    """Positions of real words: every attended position except [CLS] (first) and [SEP] (last)."""
    n = int(attention_mask_row.sum())
    return np.arange(1, max(1, n - 1))


def sample_target_mask(attention_mask: np.ndarray, ratio: float, n_spans: int, rng: np.random.Generator) -> np.ndarray:
    """JEPA: boolean (batch, length) array of hidden positions, `n_spans` spans per headline.

    Each headline hides about `ratio` of its words (at least one), split into spans of equal length,
    at random starts (spans may touch or overlap). [CLS], [SEP] and padding are never hidden, and at
    least one word is always left visible when the headline has two or more words."""
    out = np.zeros(attention_mask.shape, dtype=bool)
    for i, row in enumerate(attention_mask):
        pos = content_positions(row)
        n = len(pos)
        if n == 0:
            continue
        k = max(1, int(round(ratio * n)))
        if n >= 2:
            k = min(k, n - 1)
        spans = min(n_spans, k)           # never more hidden words than k (k <= n - 1 keeps one visible)
        span = k // spans
        for _ in range(spans):
            start = rng.integers(0, n - span + 1)
            out[i, pos[start:start + span]] = True
    return out


def sample_mlm_mask(input_ids: np.ndarray, attention_mask: np.ndarray, prob: float, rng: np.random.Generator,
                    mask_id: int, vocab_size: int, special_ids):
    """MLM: (new input ids, labels with -100 where no loss). 15% of real words are picked; of those
    80% become [MASK], 10% a random word, 10% stay as they are (BERT's recipe)."""
    ids = input_ids.copy()
    labels = np.full(ids.shape, -100, dtype=np.int64)
    special = np.isin(ids, list(special_ids)) | (attention_mask == 0)
    pick = (rng.random(ids.shape) < prob) & ~special
    for i in range(ids.shape[0]):  # every headline gets at least one prediction target
        if not pick[i].any():
            cand = np.where(~special[i])[0]
            if len(cand):
                pick[i, rng.choice(cand)] = True
    labels[pick] = ids[pick]
    roll = rng.random(ids.shape)
    ids[pick & (roll < 0.8)] = mask_id
    rand = pick & (roll >= 0.8) & (roll < 0.9)
    ids[rand] = rng.integers(0, vocab_size, size=int(rand.sum()))
    return ids, labels


# ------------------------------------------------------------------ torch parts

def _torch():
    import torch
    return torch


def build_predictor(hidden: int, layers: int, heads: int, max_len: int):
    """JEPA predictor: a narrow-depth transformer that fills in the hidden positions."""
    torch = _torch()
    from torch import nn

    class JepaPredictor(nn.Module):
        def __init__(self):
            super().__init__()
            self.mask_token = nn.Parameter(torch.randn(hidden) * 0.02)
            self.pos = nn.Embedding(max_len, hidden)
            layer = nn.TransformerEncoderLayer(hidden, heads, dim_feedforward=2 * hidden, dropout=0.0,
                                               activation="gelu", batch_first=True, norm_first=True)
            self.body = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
            self.norm = nn.LayerNorm(hidden)
            self.out = nn.Linear(hidden, hidden)

        def forward(self, ctx, target, attn):
            pos = self.pos(torch.arange(ctx.shape[1], device=ctx.device))[None]
            x = torch.where(target[..., None], self.mask_token.to(ctx.dtype).expand_as(ctx), ctx) + pos
            h = self.body(x, src_key_padding_mask=attn == 0)
            return self.out(self.norm(h))

    return JepaPredictor()


def jepa_loss(context_enc, target_enc, predictor, ids, attn, target, use_fp16):
    """Mean squared error between predicted and target vectors at the hidden positions.

    Targets: the target encoder's last layer on the full headline, layer-normed per position (as in
    I-JEPA). Context: the context encoder with the hidden positions removed from attention."""
    torch = _torch()
    import torch.nn.functional as F
    with torch.no_grad():
        with torch.autocast(device_type=ids.device.type, dtype=torch.float16, enabled=use_fp16):
            tgt = target_enc(input_ids=ids, attention_mask=attn).last_hidden_state
        tgt = F.layer_norm(tgt.float(), tgt.shape[-1:])
    ctx_attn = attn * (~target).long()
    with torch.autocast(device_type=ids.device.type, dtype=torch.float16, enabled=use_fp16):
        ctx = context_enc(input_ids=ids, attention_mask=ctx_attn).last_hidden_state
        pred = predictor(ctx, target, attn)
    pred, tgt_sel = pred.float()[target], tgt[target]
    loss = F.mse_loss(pred, tgt_sel)
    stats = {"target_std": tgt_sel.std(0).mean().item(), "pred_std": pred.detach().std(0).mean().item()}
    return loss, stats


def _mlm_loss(mlm_model, ids, attn, labels, use_fp16):
    torch = _torch()
    with torch.autocast(device_type=ids.device.type, dtype=torch.float16, enabled=use_fp16):
        out = mlm_model(input_ids=ids, attention_mask=attn, labels=labels)
    return out.loss.float()


def _encode(tokenizer, texts, max_len):
    enc = tokenizer(list(texts), max_length=max_len, truncation=True, padding="max_length", return_tensors="np")
    return enc["input_ids"].astype(np.int64), enc["attention_mask"].astype(np.int64)


def mean_pool_features(encoder, tokenizer, texts, max_len, device, batch=256, fp16=False) -> np.ndarray:
    """Frozen features for the linear probe: [CLS] vector and mean of the word vectors, side by side."""
    torch = _torch()
    ids, attn = _encode(tokenizer, texts, max_len)
    encoder.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(ids), batch):
            x = torch.as_tensor(ids[i:i + batch], device=device)
            m = torch.as_tensor(attn[i:i + batch], device=device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=fp16):
                h = encoder(input_ids=x, attention_mask=m).last_hidden_state
            h = h.float()
            mean = (h * m[..., None]).sum(1) / m.sum(1, keepdim=True).clamp(min=1)
            out.append(torch.cat([h[:, 0], mean], dim=1).cpu().numpy())
    encoder.train()
    return np.concatenate(out)


def linear_probe(encoder, tokenizer, data_dir, max_len, device, fp16=False) -> dict:
    """How good are the frozen vectors for sarcasm? Logistic regression on L (1% and 10%, seeds 0-4),
    scored on dev macro-F1 (points). Test is never read. Same procedure before and after pretraining."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.preprocessing import StandardScaler
    from .data import load_splits
    data_dir = Path(data_dir)
    train = pd.read_csv(data_dir / "headlines_train.csv", keep_default_na=False)
    dev = pd.read_csv(data_dir / "headlines_dev.csv", keep_default_na=False)
    f_train = mean_pool_features(encoder, tokenizer, train["text"], max_len, device, fp16=fp16)
    f_dev = mean_pool_features(encoder, tokenizer, dev["text"], max_len, device, fp16=fp16)
    row_of = {int(i): r for r, i in enumerate(train["id"])}
    res = {}
    for b in PROBE_BUDGETS:
        f1s = []
        for s in PROBE_SEEDS:
            lab = load_splits(data_dir, b, s).labeled
            if lab["label"].nunique() < 2:   # only possible on tiny made-up data; real L always has both
                continue
            X = f_train[[row_of[int(i)] for i in lab["id"]]]
            sc = StandardScaler().fit(X)
            clf = LogisticRegression(max_iter=3000, C=1.0).fit(sc.transform(X), lab["label"].to_numpy())
            f1s.append(100 * f1_score(dev["label"].to_numpy(), clf.predict(sc.transform(f_dev)), average="macro"))
        if f1s:
            res[f"p{b}"] = {"dev_macro_f1_mean": round(float(np.mean(f1s)), 3),
                            "dev_macro_f1_sd": round(float(np.std(f1s, ddof=1)), 3) if len(f1s) > 1 else None,
                            "per_seed": [round(float(v), 3) for v in f1s]}
    return res


def _sha256_dir(folder: Path) -> dict:
    from .data import sha256
    return {p.name: sha256(p) for p in sorted(folder.iterdir()) if p.suffix in (".safetensors", ".bin", ".json", ".txt")
            and p.name != "pretrain_info.json"}


def find_encoder(root, objective: str, seed: int = 0, steps: int = 10000, expected_sha256: dict | None = None) -> Path:
    """Pass 10: find a finished TAPT encoder under `root` (e.g. /kaggle/input, where an earlier notebook's
    output is attached) by its pretrain_info.json, and check its weights against a committed receipt.

    `expected_sha256` maps file name -> sha256 (the `weights_sha256` of a committed pretrain_info.json);
    only the files present in both are compared, and model.safetensors must be one of them. Raises
    FileNotFoundError if no matching encoder is found, ValueError if its weights differ from the receipt."""
    found = []
    for info_path in sorted(Path(root).rglob("pretrain_info.json")):
        try:
            info = json.loads(info_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        c = info.get("config", {})
        if (info.get("objective") == objective and c.get("seed") == seed and c.get("steps") == steps
                and (info_path.parent / "config.json").exists()):
            found.append(info_path.parent)
    if not found:
        raise FileNotFoundError(f"no {objective} encoder (seed {seed}, {steps} steps) with config.json under {root}")
    folder = found[0]
    if expected_sha256 is not None:
        actual = _sha256_dir(folder)
        shared = sorted(set(actual) & set(expected_sha256))
        if "model.safetensors" not in shared:
            raise ValueError(f"{folder}: model.safetensors missing or not in the receipt")
        bad = [n for n in shared if actual[n] != expected_sha256[n]]
        if bad:
            raise ValueError(f"{folder}: {bad} differ from the committed receipt; this is not the same encoder")
    return folder


def pretrain(cfg: PretrainConfig, data_dir, out_dir, probe: bool = True) -> dict:
    """Run one pretraining; save the encoder to out_dir; return (and write) pretrain_info.json."""
    torch = _torch()
    import transformers
    from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer, get_linear_schedule_with_warmup
    from .data import verify
    from .util import git_commit

    out_dir = Path(out_dir)
    info_path = out_dir / "pretrain_info.json"
    if info_path.exists():
        print(f"{info_path} exists: pretraining already done, skipped")
        return json.loads(info_path.read_text())
    t0 = time.time()
    data_dir = Path(data_dir)
    manifest = verify(data_dir)
    texts = pd.read_csv(data_dir / "headlines_train.csv", usecols=["text"], keep_default_na=False)["text"]
    dev_texts = pd.read_csv(data_dir / "headlines_dev.csv", usecols=["text"], keep_default_na=False)["text"]

    import random
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = cfg.fp16 and device.type == "cuda"
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    ids_all, attn_all = _encode(tok, texts, cfg.max_len)
    dev_ids, dev_attn = _encode(tok, dev_texts, cfg.max_len)
    rng = np.random.default_rng(cfg.seed)
    dev_rng_seed = 777  # fixed: the held-out loss uses the same hidden positions before and after

    if cfg.objective == "mlm":
        mlm = AutoModelForMaskedLM.from_pretrained(cfg.model_name).to(device)
        encoder = mlm.base_model
        trainable = mlm
        # as in BERT's own data collator: [CLS], [SEP], padding (and [MASK]) are never targets; [UNK] can be
        special = {tok.cls_token_id, tok.sep_token_id, tok.pad_token_id, tok.mask_token_id}
        vocab = mlm.config.vocab_size

        def batch_loss(ids, attn, r):
            m_ids, labels = sample_mlm_mask(ids, attn, cfg.mask_prob, r, tok.mask_token_id, vocab, special)
            return _mlm_loss(mlm, torch.as_tensor(m_ids, device=device), torch.as_tensor(attn, device=device),
                             torch.as_tensor(labels, device=device), use_fp16), {}
    else:
        encoder = AutoModel.from_pretrained(cfg.model_name, add_pooling_layer=False).to(device)
        target_enc = copy.deepcopy(encoder).to(device)
        for p in target_enc.parameters():
            p.requires_grad_(False)
        predictor = build_predictor(encoder.config.hidden_size, cfg.pred_layers, cfg.pred_heads, cfg.max_len).to(device)
        trainable = torch.nn.ModuleDict({"encoder": encoder, "predictor": predictor})

        def batch_loss(ids, attn, r):
            target = sample_target_mask(attn, cfg.target_ratio, cfg.n_spans, r)
            return jepa_loss(encoder, target_enc, predictor, torch.as_tensor(ids, device=device),
                             torch.as_tensor(attn, device=device), torch.as_tensor(target, device=device), use_fp16)

    def dev_loss() -> dict:
        r = np.random.default_rng(dev_rng_seed)
        trainable.eval()
        tot, n, extra = 0.0, 0, {}
        with torch.no_grad():
            for i in range(0, min(len(dev_ids), 2048), 256):
                loss, st = batch_loss(dev_ids[i:i + 256], dev_attn[i:i + 256], r)
                tot += loss.item()
                n += 1
                for k, v in st.items():
                    extra.setdefault(k, []).append(v)
        trainable.train()
        return {"loss": round(tot / n, 5), **{k: round(float(np.mean(v)), 5) for k, v in extra.items()}}

    probe_before = linear_probe(encoder, tok, data_dir, cfg.max_len, device, use_fp16) if probe else None
    dev_before = dev_loss()
    print(f"before pretraining: dev {dev_before}" + (f"; probe {probe_before}" if probe else ""))

    named = list(trainable.named_parameters())
    no_decay = ("bias", "LayerNorm.weight", "norm", "mask_token", "pos.weight")
    groups = [{"params": [p for n, p in named if p.requires_grad and not any(k in n for k in no_decay)],
               "weight_decay": cfg.weight_decay},
              {"params": [p for n, p in named if p.requires_grad and any(k in n for k in no_decay)],
               "weight_decay": 0.0}]
    opt = torch.optim.AdamW(groups, lr=cfg.lr)
    sched = get_linear_schedule_with_warmup(opt, int(cfg.warmup_ratio * cfg.steps), cfg.steps)
    scaler = torch.amp.GradScaler(device.type, enabled=use_fp16)
    order, pos = None, len(ids_all)
    bs = min(cfg.batch_size, len(ids_all))
    log, window = [], {}
    trainable.train()
    for step in range(1, cfg.steps + 1):
        if pos + bs > len(ids_all):
            order, pos = rng.permutation(len(ids_all)), 0
        idx = order[pos:pos + bs]
        pos += bs
        loss, st = batch_loss(ids_all[idx], attn_all[idx], rng)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_([p for _, p in named if p.requires_grad], 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        opt.zero_grad(set_to_none=True)
        if cfg.objective == "jepa":  # target encoder follows the context encoder slowly
            m = cfg.ema_start + (cfg.ema_end - cfg.ema_start) * step / cfg.steps
            with torch.no_grad():
                for pt, pc in zip(target_enc.parameters(), encoder.parameters()):
                    pt.mul_(m).add_(pc.detach(), alpha=1 - m)
        for k, v in {"loss": loss.item(), **st}.items():
            window.setdefault(k, []).append(v)
        if step % cfg.log_every == 0 or step == cfg.steps:
            point = {"step": step, **{k: round(float(np.mean(v)), 5) for k, v in window.items()}}
            log.append(point)
            print("  " + "  ".join(f"{k} {v:.4g}" if k != "step" else f"step {v}/{cfg.steps}" for k, v in point.items()))
            window = {}

    dev_after = dev_loss()
    # I-JEPA evaluates (and here we keep) the target encoder; for MLM the encoder under the MLM head
    keep = target_enc if cfg.objective == "jepa" else encoder
    probe_after = linear_probe(keep, tok, data_dir, cfg.max_len, device, use_fp16) if probe else None
    print(f"after pretraining: dev {dev_after}" + (f"; probe {probe_after}" if probe else ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    keep.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    collapsed = None
    if cfg.objective == "jepa":
        # collapse check: per-dimension spread of the target vectors (layer-normed, so ~1 when healthy)
        collapsed = bool(dev_after.get("target_std", 1.0) < 0.05)
    info = {"config": asdict(cfg), "objective": cfg.objective, "n_texts": int(len(texts)),
            "minutes": round((time.time() - t0) / 60, 2), "train_log": log,
            "dev_before": dev_before, "dev_after": dev_after,
            "probe_before": probe_before, "probe_after": probe_after, "collapsed": collapsed,
            "kept_encoder": "target (EMA)" if cfg.objective == "jepa" else "MLM encoder",
            "weights_sha256": _sha256_dir(out_dir), "commit": git_commit(),
            "data_manifest_sha256": {k: v["sha256"] for k, v in manifest["outputs"].items()},
            "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "versions": {"torch": torch.__version__, "transformers": transformers.__version__}}
    tmp = out_dir / "pretrain_info.json.tmp"
    tmp.write_text(json.dumps(info, indent=2))
    tmp.replace(info_path)
    print(f"saved {cfg.objective} encoder to {out_dir} ({info['minutes']} min)")
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objective", required=True, choices=OBJECTIVES)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="{}", help="extra PretrainConfig fields as JSON")
    ap.add_argument("--no-probe", action="store_true")
    args = ap.parse_args()
    cfg = PretrainConfig(objective=args.objective, **json.loads(args.config))
    pretrain(cfg, args.data, args.out, probe=not args.no_probe)


if __name__ == "__main__":
    main()
