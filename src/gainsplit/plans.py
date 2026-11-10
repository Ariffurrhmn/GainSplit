"""Named lists of runs. The Kaggle notebook picks one by name."""
from .config import RunConfig

LR_GRID = (2e-5, 3e-5, 5e-5)
TUNING_SEEDS = (0, 1, 2)


def lr_tuning() -> list[RunConfig]:
    """Step 1: A1 at 10% labels, three learning rates x three seeds, scored on dev only.
    Pick the lr with the best mean dev macro-F1; it is then frozen for every arm.
    Three seeds so a lucky seed cannot decide the lr."""
    return [RunConfig("A1", 10, seed, lr=lr, eval_test=False, tag=f"lr{lr:g}")
            for lr in LR_GRID for seed in TUNING_SEEDS]


PLANS = {"lr_tuning": lr_tuning}
