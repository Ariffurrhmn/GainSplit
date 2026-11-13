"""Run a list of runs, one per GPU at a time (two at once on Kaggle's GPU T4 x2).

Each run is a separate Python process that sees only its own GPU. When a GPU
finishes, the next waiting run starts on it. Runs already finished are skipped.
A0 (TF-IDF) runs need no GPU: they run one after another in a background thread
of this process, while the GPU runs go on.
"""
from pathlib import Path
import json
import os
import subprocess
import sys
import threading
import time

import pandas as pd

from .config import RunConfig

SRC_DIR = Path(__file__).resolve().parents[1]


def _done(cfg: RunConfig, out_root: Path) -> bool:
    return (out_root / "runs" / cfg.run_id / "result.json").exists()


def _run_cpu(cfgs, data_dir, out_root, failed):
    """Run the A0 configs in order (each takes seconds). Imported here so the GPU path never needs it."""
    from .tfidf import run_tfidf
    for cfg in cfgs:
        try:
            run_tfidf(cfg, data_dir, out_root)
        except Exception as e:  # one bad run must not stop the others
            failed.append(cfg.run_id)
            print(f"FAILED {cfg.run_id}: {e}")


def run_all(configs: list[RunConfig], data_dir, out_root, n_gpus: int | None = None) -> pd.DataFrame:
    out_root, data_dir = Path(out_root), Path(data_dir)
    (out_root / "logs").mkdir(parents=True, exist_ok=True)
    if n_gpus is None:
        import torch
        n_gpus = torch.cuda.device_count()
    slots = list(range(n_gpus)) if n_gpus > 0 else [None]  # None = one CPU worker

    todo = [c for c in configs if not _done(c, out_root)]
    waiting = [c for c in todo if c.arm != "A0"]
    cpu_runs = [c for c in todo if c.arm == "A0"]
    print(f"{len(configs)} runs, {len(configs) - len(waiting)} already done, "
          f"{len(waiting)} to run on {'GPU ' + str(slots) if n_gpus else 'CPU'}"
          + (f", {len(cpu_runs)} A0 on CPU" if cpu_runs else ""))
    active, failed = {}, []
    cpu_thread = threading.Thread(target=_run_cpu, args=(cpu_runs, data_dir, out_root, failed))
    cpu_thread.start()
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
    cpu_thread.join()
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
               "tag": c.get("tag", ""), "lr": c["lr"], "helper_weight": c["helper_weight"],
               "fgm_epsilon": c.get("fgm_epsilon", 0.0),  # older result.json files have no such field
               "pair_weight": c.get("pair_weight", 1.0),           # Pass 5: older files have no such field
               "cleaned_drop_pct": c.get("cleaned_drop_pct", 0.0),  # Pass 5: older files have no such field
               "steps": r["steps"], "minutes": r["minutes"]}
        for split, s in r["scores"].items():
            for k in ("accuracy", "macro_f1", "f1_sarcastic", "roc_auc"):
                row[f"{split}_{k}"] = s[k]
        rows.append(row)
    return pd.DataFrame(rows)
