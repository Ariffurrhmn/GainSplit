"""Named lists of runs. The Kaggle notebook picks one by name."""
from .config import ARMS, RunConfig

LR_GRID = (2e-5, 3e-5, 5e-5)
TUNING_SEEDS = (0, 1, 2)


def lr_tuning() -> list[RunConfig]:
    """Step 1: A1 at 10% labels, three learning rates x three seeds, scored on dev only.
    Pick the lr with the best mean dev macro-F1; it is then frozen for every arm.
    Three seeds so a lucky seed cannot decide the lr."""
    return [RunConfig("A1", 10, seed, lr=lr, eval_test=False, tag=f"lr{lr:g}")
            for lr in LR_GRID for seed in TUNING_SEEDS]


W_GRID = (0.1, 0.3, 1.0)
PASS1_SEEDS = (0, 1, 2, 3, 4)
PASS1_HELPER_WEIGHT = 0.3  # winner of w_tuning (session2, dev), frozen 2026-11-10


def w_tuning() -> list[RunConfig]:
    """Step 2a: A3 at 10% labels, three helper weights x three seeds, dev only.
    Pick the w with the best mean dev macro-F1 (tie within ~1 std: the smaller w)."""
    return [RunConfig("A3", 10, seed, helper_weight=w, eval_test=False, tag=f"w{w:g}")
            for w in W_GRID for seed in TUNING_SEEDS]


def gan_pilot() -> list[RunConfig]:
    """Step 2b: A4 at 10% labels, three seeds, dev only. Checks that the GAN trains
    (see train_log in result.json) before it is used in the full study."""
    return [RunConfig("A4", 10, seed, eval_test=False, tag="pilot") for seed in TUNING_SEEDS]


def session2() -> list[RunConfig]:
    return w_tuning() + gan_pilot()  # 9 + 3 = 12 runs


def pass1() -> list[RunConfig]:
    """The main 2x2: A1 at 1/10/100%, A3, A4, A5 at 1/10%, seeds 0-4, test scored (45 runs)."""
    if PASS1_HELPER_WEIGHT is None:
        raise ValueError("set PASS1_HELPER_WEIGHT in plans.py (the winning w from w_tuning) before building pass1")
    runs = [RunConfig("A1", b, s, eval_test=True) for b in (1, 10, 100) for s in PASS1_SEEDS]
    for arm in ("A3", "A4", "A5"):
        for b in (1, 10):
            for s in PASS1_SEEDS:
                w = PASS1_HELPER_WEIGHT if ARMS[arm]["helpers"] else 0.0
                runs.append(RunConfig(arm, b, s, helper_weight=w, eval_test=True))
    return runs


PLANS = {"lr_tuning": lr_tuning, "w_tuning": w_tuning, "gan_pilot": gan_pilot,
         "session2": session2, "pass1": pass1}
