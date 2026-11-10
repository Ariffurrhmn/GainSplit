"""Run a list of runs, one per GPU at a time (two at once on Kaggle's GPU T4 x2).

Each run is a separate Python process that sees only its own GPU. When a GPU
finishes, the next waiting run starts on it. Runs already finished are skipped.
"""
from pathlib import Path
import json
import os
import subprocess
import sys
import time

import pandas as pd

from .config import RunConfig

SRC_DIR = Path(__file__).resolve().parents[1]


def _done(cfg: RunConfig, out_root: Path) -> bool:
    return (out_root / "runs" / cfg.run_id / "result.json").exists()


def run_all(configs: list[RunConfig], data_dir, out_root, n_gpus: int | None = None) -> pd.DataFrame:
    out_root, data_dir = Path(out_root), Path(data_dir)
    (out_root / "logs").mkdir(parents=True, exist_ok=True)
    if n_gpus is None:
        import torch
        n_gpus = torch.cuda.device_count()
    slots = list(range(n_gpus)) if n_gpus > 0 else [None]  # None = one CPU worker

    waiting = [c for c in configs if not _done(c, out_root)]
    print(f"{len(configs)} runs, {len(configs) - len(waiting)} already done, "
          f"{len(waiting)} to run on {'GPU ' + str(slots) if n_gpus else 'CPU'}")
    active, failed = {}, []
    while waiting or active:
        for slot in slots:
            if slot not in active and waiting:
                cfg = waiting.pop(0)
                env = dict(os.environ, PYTHONPATH=str(SRC_DIR))
                if slot is not None:
                    env["CUDA_VISIBLE_DEVICES"] = str(slot)
                log = open(out_root / "logs" / f"{cfg.run_id}.log", "w")
                cmd = [sys.executable, "-m", "gainsplit.train", "--config", json.dumps(cfg.to_dict()),
                       "--data", str(data_dir), "--out", str(out_root)]
                active[slot] = (subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env),
                                cfg, log, time.time())
                print(f"start  {cfg.run_id}  on {'GPU ' + str(slot) if slot is not None else 'CPU'}")
        for slot, (proc, cfg, log, t0) in list(active.items()):
            if proc.poll() is not None:
                log.close()
                mins = (time.time() - t0) / 60
                if proc.returncode == 0:
                    print(f"done   {cfg.run_id}  ({mins:.1f} min)")
                else:
                    failed.append(cfg.run_id)
                    print(f"FAILED {cfg.run_id}  (see logs/{cfg.run_id}.log)")
                del active[slot]
        time.sleep(3)
    if failed:
        print(f"{len(failed)} failed: {failed}")
    return collect(out_root)


def collect(out_root) -> pd.DataFrame:
    """One row per finished run: settings, training length, time and scores."""
    rows = []
    for f in sorted(Path(out_root).glob("runs/*/result.json")):
        r = json.loads(f.read_text())
        c = r["config"]
        row = {"run_id": r["run_id"], "arm": c["arm"], "budget": c["budget"], "seed": c["seed"],
               "lr": c["lr"], "helper_weight": c["helper_weight"], "steps": r["steps"], "minutes": r["minutes"]}
        for split, s in r["scores"].items():
            for k in ("accuracy", "macro_f1", "f1_sarcastic", "roc_auc"):
                row[f"{split}_{k}"] = s[k]
        rows.append(row)
    return pd.DataFrame(rows)
