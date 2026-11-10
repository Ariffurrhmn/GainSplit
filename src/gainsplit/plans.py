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


FGM_GRID = (0.5, 1.0, 2.0)
PASS2_SEEDS = PASS1_SEEDS
PASS2_FGM_EPSILON = 0.5  # from fgm_tuning (pass2a, dev): 1.0 had the best mean, 0.5 within 1 sd, tie -> smaller


def fgm_tuning() -> list[RunConfig]:
    """A7 at 10% labels, three epsilons x three seeds, dev only (9 runs).
    Pick the epsilon with the best mean dev macro-F1 (tie within ~1 std: the smaller epsilon)."""
    return [RunConfig("A7", 10, seed, fgm_epsilon=eps, eval_test=False, tag=f"eps{eps:g}")
            for eps in FGM_GRID for seed in TUNING_SEEDS]


def _runs(arms, budgets, seeds=PASS2_SEEDS) -> list[RunConfig]:
    """Test-scored runs; helper arms get the frozen helper weight."""
    return [RunConfig(arm, b, s, helper_weight=PASS1_HELPER_WEIGHT if ARMS[arm]["helpers"] else 0.0, eval_test=True)
            for b in budgets for arm in arms for s in seeds]


def pass2a() -> list[RunConfig]:
    """Everything in Pass 2 that does not need the chosen epsilon (104 runs, 79 on GPU, 25 A0 on CPU).

    Longest runs first, so the two GPUs stay busy to the end: 30% budget (A4/A5 slowest),
    then the 10% controls, then 5%, then 1%, then fgm_tuning. A0 comes last; the runner
    runs it on the CPU alongside the GPU runs, so it never holds a GPU slot."""
    if PASS1_HELPER_WEIGHT is None:
        raise ValueError("set PASS1_HELPER_WEIGHT in plans.py before building pass2a")
    return (_runs(("A4", "A5", "A3", "A1"), (30,))            # 20 runs: the 2x2 at 30%
            + _runs(("A6", "A8", "A2"), (10,))                # 15 runs: controls at 10%
            + _runs(("A4", "A5", "A3", "A1"), (5,))           # 20 runs: the 2x2 at 5%
            + _runs(("A6", "A8", "A2"), (1,))                 # 15 runs: controls at 1%
            + fgm_tuning()                                    # 9 runs, dev only
            + _runs(("A0",), (1, 5, 10, 30, 100)))            # 25 runs, CPU


def pass2b() -> list[RunConfig]:
    """A7 at 1% and 10%, seeds 0-4, test scored (10 runs), with the epsilon chosen by fgm_tuning."""
    if PASS2_FGM_EPSILON is None:
        raise ValueError("set PASS2_FGM_EPSILON in plans.py (the winning epsilon from fgm_tuning) before building pass2b")
    return [RunConfig("A7", b, s, fgm_epsilon=PASS2_FGM_EPSILON, eval_test=True)
            for b in (1, 10) for s in PASS2_SEEDS]


PASS3_VARIANTS = ("soft", "flip25", "flip50", "vader", "teacher2", "sentonly", "emoonly")


def _p3(arm, budget, tag, **kw) -> list[RunConfig]:
    """Pass 3 runs for one arm and budget, seeds 0-4: test scored, masked test sets scored, tagged."""
    w = PASS1_HELPER_WEIGHT if ARMS[arm]["helpers"] else 0.0
    return [RunConfig(arm, budget, s, helper_weight=w, eval_test=True, eval_masked=True, tag=tag, **kw)
            for s in PASS2_SEEDS]


def pass3() -> list[RunConfig]:
    """Helper-label quality (RQ2) and outlet-style shortcuts (RQ3): 105 runs, all on GPU.

    - 7 helper-label variants x (A3 at 10%, A2 at 1%): 70 runs
    - A9 (A2 with shuffled helper labels) at 1%: 5 runs
    - A1, A3, A4, A6, A7 at 10% with no change except the masked test sets: 25 runs
      (the tag "mask" only keeps the run ids apart from Pass 1/2; they reproduce the earlier scores)
    - A1 trained on iSarcasmEval: 5 runs
    Longest runs first (A6, A7, A4, A3, A1, then the 1% runs), so the two GPUs stay busy to the end.
    """
    if PASS2_FGM_EPSILON is None:
        raise ValueError("set PASS2_FGM_EPSILON in plans.py before building pass3")
    a3_variants = [r for v in PASS3_VARIANTS for r in _p3("A3", 10, v, helper_variant=v)]
    a2_variants = [r for v in PASS3_VARIANTS for r in _p3("A2", 1, v, helper_variant=v)]
    return (_p3("A6", 10, "mask")                                          # 5 runs
            + _p3("A7", 10, "mask", fgm_epsilon=PASS2_FGM_EPSILON)         # 5
            + _p3("A4", 10, "mask")                                        # 5
            + a3_variants + _p3("A3", 10, "mask")                          # 35 + 5
            + _p3("A1", 10, "mask")                                        # 5
            + _p3("A1", 100, "isarctrain", train_data="isarcasm")          # 5
            + a2_variants                                                  # 35
            + _p3("A9", 1, "shuf"))                                        # 5


GEN_LR_GRID = (1e-5, 2e-4)


def gan_check() -> list[RunConfig]:
    """Dev-only check that the GAN's loss to BERT is not a tuning artefact (12 runs, no test scoring).

    - A4 at 10% with the generator's lr changed (1e-5 and 2e-4; the default is 5e-5): 6 runs
    - A4 and A1 at 1% with 3x the training steps: 6 runs
    Longest runs first (10% before 1%, so the two GPUs stay busy to the end)."""
    glr = [RunConfig("A4", 10, s, gen_lr=g, eval_test=False, tag=f"glr{g:g}")
           for g in GEN_LR_GRID for s in TUNING_SEEDS]
    long = [RunConfig(arm, 1, s, step_mult=3, eval_test=False, tag="steps3x")
            for arm in ("A4", "A1") for s in TUNING_SEEDS]
    return glr + long


PLANS = {"lr_tuning": lr_tuning, "w_tuning": w_tuning, "gan_pilot": gan_pilot,
         "session2": session2, "pass1": pass1, "fgm_tuning": fgm_tuning,
         "pass2a": pass2a, "pass2b": pass2b, "pass3": pass3,
         "gan_check": gan_check}
