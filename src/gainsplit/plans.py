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


PASS4_SEEDS = PASS2_SEEDS


def _p4(arm, budget, tag="", **kw) -> list[RunConfig]:
    """Pass 4 runs for one arm and budget, seeds 0-4: test and masked test sets scored.
    FGM arms (A7, A11) get the frozen Pass 2 epsilon."""
    eps = PASS2_FGM_EPSILON if ARMS[arm].get("fgm") else 0.0
    return [RunConfig(arm, budget, s, fgm_epsilon=eps, eval_test=True, eval_masked=True, tag=tag, **kw)
            for s in PASS4_SEEDS]


def pass4() -> list[RunConfig]:
    """Generated headline pairs (vault: Experiment Plan, "Pass 4 plan"): 75 runs, all on GPU, one commit.

    Nothing here waits for an earlier result: the pair batch is fixed at 32 rows (16 pairs, the
    size of the U batch in the helper and GAN arms) instead of being tuned on dev first.
    Baselines A1, A6, A7 are reused from Passes 1-3 (training is deterministic).

    10% (35 runs): A11, A10 (+ explain), A12, A13, A14, A1 re-run with explain (also checks that
                   the new code reproduces Pass 1 exactly), A15 (pairs only, 10% step count).
    1%  (40 runs): A10 with 3x steps (sensitivity), A11, A10, A12, A13, A14, and A7 / A1 re-runs
                   that add the masked test sets at 1% (tag "mask"; must equal Pass 1 / Pass 2b).
    Longest runs first, so the two GPUs stay busy to the end.
    """
    if PASS2_FGM_EPSILON is None:
        raise ValueError("set PASS2_FGM_EPSILON in plans.py before building pass4")
    return (_p4("A11", 10)                                   # FGM + pairs: slowest
            + _p4("A10", 10, explain=True)
            + _p4("A12", 10) + _p4("A13", 10) + _p4("A14", 10)
            + _p4("A1", 10, "explain", explain=True)
            + _p4("A15", 10)
            + _p4("A10", 1, "steps3x", step_mult=3)
            + _p4("A11", 1)
            + _p4("A10", 1) + _p4("A12", 1) + _p4("A13", 1) + _p4("A14", 1)
            + _p4("A7", 1, "mask") + _p4("A1", 1, "mask"))


def pass4_full100() -> list[RunConfig]:
    """Optional extra (5 runs, about 1.6 GPU-h of run time): A10 with all labels (100%).
    Asks whether the pairs still help when every training label is known. Run only if quota is left."""
    return _p4("A10", 100)


# ---------------------------------------------------------------------------- Pass 5

PAIR_WEIGHT_GRID = (0.1, 0.3, 1.0)        # 1.0 = Pass 4's behaviour (no down-weighting)
PASS5_PAIR_WEIGHT = 0.1                   # from pair_weight_tuning (A10, 10%, dev, 2026-11-14): 0.1 best mean (77.02) and best on every seed; edge of the grid
CLEANED_DROP_GRID = (10.0, 20.0, 30.0)
PASS5_CLEANED_DROP_PCT = 20.0             # from cleaned_drop_tuning (A16, 10%, weight 0.1, dev, 2026-11-14): 20 best mean (77.45; 10 -> 76.90, 30 -> 76.57)


def pair_weight_tuning() -> list[RunConfig]:
    """Pass 5 step 1: A10 at 10% labels, unfiltered pairs, three pair_weight values x three
    seeds, dev only (9 runs). Pick the weight with the best mean dev macro-F1 (tie within
    ~1 std: the smaller weight, same rule as every other _tuning plan)."""
    return [RunConfig("A10", 10, seed, pair_weight=w, eval_test=False, tag=f"pw{w:g}")
            for w in PAIR_WEIGHT_GRID for seed in TUNING_SEEDS]


def cleaned_drop_tuning() -> list[RunConfig]:
    """Pass 5 step 2: A16 (A10 with pairs="cleaned") at 10% labels and the frozen pair_weight
    from pair_weight_tuning, three drop percentages x three seeds, dev only (9 runs). Pick the
    drop_pct with the best mean dev macro-F1 (tie: the smaller drop_pct, i.e. keep more data)."""
    if PASS5_PAIR_WEIGHT is None:
        raise ValueError("set PASS5_PAIR_WEIGHT in plans.py (the winning weight from pair_weight_tuning) "
                         "before building cleaned_drop_tuning")
    return [RunConfig("A16", 10, seed, pair_weight=PASS5_PAIR_WEIGHT, cleaned_drop_pct=pct,
                      eval_test=False, tag=f"drop{pct:g}")
            for pct in CLEANED_DROP_GRID for seed in TUNING_SEEDS]


PASS5_SEEDS = PASS1_SEEDS


def pass5() -> list[RunConfig]:
    """Clean the generated pairs and/or down-weight their loss (vault: Experiment Plan,
    "Pass 5 plan"): does either fix Pass 4's in-domain collapse (A10/A11 vs A1/A7, [[Pass 4
    Results]])? 30 test-scored runs (3 arms x 2 budgets x 5 seeds), all on GPU, one commit;
    baselines A1, A7 reused from Passes 1-2 (training is deterministic).

    Arms (5 seeds each, 10% and 1% budgets):
    - A16: A10 with the cleaned pair set and the frozen pair_weight (does cleaning alone fix it?)
    - A17: A11 (FGM) with the same cleaning and weight (does it stack with FGM?)
    - A10 at the frozen pair_weight, uncleaned pairs (does the weight alone fix it, with no
      cleaning, isolating the two knobs' separate contributions)
    Masked test sets scored throughout (`eval_masked`), so the cue-mask gap and iSarcasmEval
    transfer/length-shortcut checks from Pass 4's analysis apply unchanged.
    """
    if PASS5_PAIR_WEIGHT is None:
        raise ValueError("set PASS5_PAIR_WEIGHT in plans.py (the winning weight from pair_weight_tuning) "
                         "before building pass5")
    if PASS5_CLEANED_DROP_PCT is None:
        raise ValueError("set PASS5_CLEANED_DROP_PCT in plans.py (the winning drop_pct from "
                         "cleaned_drop_tuning) before building pass5")
    runs = []
    for b in (10, 1):
        runs += [RunConfig("A16", b, s, pair_weight=PASS5_PAIR_WEIGHT, cleaned_drop_pct=PASS5_CLEANED_DROP_PCT,
                           eval_test=True, eval_masked=True) for s in PASS5_SEEDS]
        runs += [RunConfig("A17", b, s, fgm_epsilon=PASS2_FGM_EPSILON, pair_weight=PASS5_PAIR_WEIGHT,
                           cleaned_drop_pct=PASS5_CLEANED_DROP_PCT, eval_test=True, eval_masked=True)
                for s in PASS5_SEEDS]
        runs += [RunConfig("A10", b, s, pair_weight=PASS5_PAIR_WEIGHT, eval_test=True, eval_masked=True,
                           tag="weighted") for s in PASS5_SEEDS]
    return runs


# ---------------------------------------------------------------------------- Pass 6

PASS6_PRETRAIN_STEPS = 720   # = A15's steps at 10% (10 passes x 72 batches), so stage 1 is exactly Pass 4's A15_p10 model
PASS6_SEEDS = PASS1_SEEDS


def pass6() -> list[RunConfig]:
    """Two-stage training (vault: Experiment Plan, "Pass 6 plan"): stage 1 trains on the
    generated pairs only (A15's training, 720 steps, the same at every budget); stage 2 keeps
    that model and fine-tunes it on the gold labels only, with the usual step count.
    20 test-scored runs (2 arms x 2 budgets x 5 seeds), masked test sets scored.

    - A18: pairs -> plain BERT on L (compare with A1, and with A15 for transfer)
    - A19: pairs -> BERT + FGM on L (compare with A7)
    Baselines A1, A7 (Passes 1-2), A15 and A10/A11 (Pass 4) and the Pass 5 arms are reused.
    """
    runs = []
    for b in (10, 1):
        runs += [RunConfig("A18", b, s, pretrain_steps=PASS6_PRETRAIN_STEPS, eval_test=True, eval_masked=True)
                 for s in PASS6_SEEDS]
        runs += [RunConfig("A19", b, s, fgm_epsilon=PASS2_FGM_EPSILON, pretrain_steps=PASS6_PRETRAIN_STEPS,
                           eval_test=True, eval_masked=True) for s in PASS6_SEEDS]
    return runs


# ---------------------------------------------------------------------------- Pass 7

PASS7_SEEDS = PASS1_SEEDS


def pass7() -> list[RunConfig]:
    """Every training label (budget 100%) for the two strongest recipes so far (vault: Experiment
    Plan, "Pass 7 plan"): does FGM, and FGM after the pair stage, push Headlines test accuracy past
    A1's 93.35% (Pass 1) towards 94%? No new settings: epsilon (0.5) and the stage-1 length (720)
    stay frozen from earlier passes, so nothing is picked on test. 11 runs, longest first:
    - A19 at 100%: pairs-only stage 1 (= A15), then all labels with FGM (5 seeds)
    - A7 at 100%: all labels with FGM (5 seeds)
    - A1 at 100%, seed 0, tag "rerun": must equal Pass 1's A1_p100_s0 exactly, and adds the masked tests
    """
    return ([RunConfig("A19", 100, s, fgm_epsilon=PASS2_FGM_EPSILON, pretrain_steps=PASS6_PRETRAIN_STEPS,
                       eval_test=True, eval_masked=True) for s in PASS7_SEEDS]
            + [RunConfig("A7", 100, s, fgm_epsilon=PASS2_FGM_EPSILON, eval_test=True, eval_masked=True)
               for s in PASS7_SEEDS]
            + [RunConfig("A1", 100, 0, eval_test=True, eval_masked=True, tag="rerun")])


# ---------------------------------------------------------------------------- Pass 8

PASS8_SEEDS = PASS1_SEEDS
# Where the notebooks save the TAPT encoders (outside the results folder: about 440 MB each, not zipped)
PASS8_CKPT = {"mlm": "/kaggle/working/ckpt/mlm", "jepa": "/kaggle/working/ckpt/jepa"}
PASS8_CONS_WEIGHT = 1.0   # A25/A26 consistency weight: fixed before running, not tuned (UDA's default weight is 1)


def _p8_tapt(objective: str, init_dir: str | None = None, budgets=(10, 1)) -> list[RunConfig]:
    """A1 and A7 recipes on a TAPT encoder, 5 seeds per budget, test and masked tests scored."""
    plain, fgm = {"mlm": ("A20", "A21"), "jepa": ("A22", "A23")}[objective]
    d = init_dir or PASS8_CKPT[objective]
    runs = []
    for b in budgets:
        runs += [RunConfig(fgm, b, s, fgm_epsilon=PASS2_FGM_EPSILON, init_dir=d, eval_test=True, eval_masked=True)
                 for s in PASS8_SEEDS]
        runs += [RunConfig(plain, b, s, init_dir=d, eval_test=True, eval_masked=True) for s in PASS8_SEEDS]
    return runs


def pass8_mlm(init_dir: str | None = None) -> list[RunConfig]:
    """Pass 8a (vault: Experiment Plan, "Pass 8 plan"): BERT after masked-word pretraining on the
    22,800 training headlines (no labels), then the A1 and A7 recipes unchanged. 20 runs at 10% and 1%.
    Baselines A1 (Pass 1) and A7 (Pass 2b) are reused (training is deterministic)."""
    return _p8_tapt("mlm", init_dir)


def pass8_jepa(init_dir: str | None = None) -> list[RunConfig]:
    """Pass 8b: the same as pass8_mlm with the JEPA-pretrained encoder (A22, A23). 20 runs."""
    return _p8_tapt("jepa", init_dir)


def pass8_mlm_100(init_dir: str | None = None) -> list[RunConfig]:
    """Pass 8a extension, every training label: A21 (MLM-TAPT + FGM) at 100%, 5 seeds (~21 min each).
    Compared with Pass 7's A7 at 100% (93.90% accuracy). Does TAPT add on top of FGM with all labels?"""
    return [r for r in _p8_tapt("mlm", init_dir, budgets=(100,)) if r.arm == "A21"]


def pass8_jepa_100(init_dir: str | None = None) -> list[RunConfig]:
    """Pass 8b extension: A23 (JEPA-TAPT + FGM) at 100%, 5 seeds."""
    return [r for r in _p8_tapt("jepa", init_dir, budgets=(100,)) if r.arm == "A23"]


def pass8_diff() -> list[RunConfig]:
    """Pass 8c: the diffusion model over [CLS] vectors. 31 runs, all reading U:
    - A24 (diffusion fakes in GAN-BERT's k+1 discriminator) at 10% and 1%: compare with A4 (GAN) and A1
    - A25 (U consistency through the diffusion round trip) at 10% and 1%: compare with A1
    - A26 (the same consistency with plain noise, no denoising) at 10% and 1%: A25 - A26 = what the
      generative model adds
    - A1 at 10%, seed 0, tag "rerun": must equal Pass 1's A1_p10_s0 exactly (the training loop changed)
    Baselines A1 and A4 (Pass 1) and A7 (Pass 2b) are reused."""
    runs = []
    for b in (10, 1):
        runs += [RunConfig("A24", b, s, eval_test=True, eval_masked=True) for s in PASS8_SEEDS]
        runs += [RunConfig("A25", b, s, cons_weight=PASS8_CONS_WEIGHT, eval_test=True, eval_masked=True)
                 for s in PASS8_SEEDS]
        runs += [RunConfig("A26", b, s, cons_weight=PASS8_CONS_WEIGHT, eval_test=True, eval_masked=True)
                 for s in PASS8_SEEDS]
    return runs + [RunConfig("A1", 10, 0, eval_test=True, tag="rerun")]


# ---------------------------------------------------------------------------- Pass 9

PASS9_SEEDS = PASS1_SEEDS
PASS9_ARMS = ("A27", "A28", "A29", "A30")
PASS9_RERUNS = ("A2", "A6", "A7")   # the training paths of these three changed; each must equal its old run


def _p9(arm, budget, seed, tag="") -> RunConfig:
    """The frozen settings of the ingredient arms: helper weight 0.3 (A2), FGM epsilon 0.5 (A7)."""
    sw = ARMS[arm]
    return RunConfig(arm, budget, seed, helper_weight=PASS1_HELPER_WEIGHT if sw["helpers"] else 0.0,
                     fgm_epsilon=PASS2_FGM_EPSILON if sw.get("fgm") else 0.0, eval_test=True, tag=tag)


def pass9() -> list[RunConfig]:
    """Pass 9 (vault: Experiment Plan, "Pass 9 plan"): do helper heads on L (A2), FGM (A7) and
    self-training (A6) stack? The four missing cells of a 2x2x2 design, 5 seeds at 10% and 1%:
    A27 = A2 + A7, A28 = A2 + A6, A29 = A7 + A6, A30 = all three. 40 runs, longest first, plus
    one re-run each of A2, A6 and A7 at 1%, seed 0 (tag "rerun"): fgm_step, plain_step and the
    self-training teacher changed, and each must reproduce its Pass 2 run exactly.
    The other cells (A1, A2, A6, A7) are reused from Passes 1, 2a and 2b."""
    order = ("A30", "A29", "A28", "A27")   # self-training + FGM runs are the longest
    runs = [_p9(arm, b, s) for b in (10, 1) for arm in order for s in PASS9_SEEDS]
    return runs + [_p9(arm, 1, 0, tag="rerun") for arm in PASS9_RERUNS]


def pass9_100() -> list[RunConfig]:
    """Pass 9 extension, every training label: A27 (helpers + FGM, no unlabeled data needed) at 100%,
    5 seeds. Compared with Pass 7's A7 at 100%. Self-training needs U, so it has no 100% cell."""
    return [_p9("A27", 100, s) for s in PASS9_SEEDS]


# ---------------------------------------------------------------------------- Pass 10

PASS10_SEEDS = PASS1_SEEDS
PASS10_STACK_ARMS = ("A31", "A32", "A33", "A34", "A35", "A36")
PASS10_PRETRAIN_SEEDS = (1, 2)           # Q2: extra MLM pretrainings; seed 0 is Pass 8a's encoder
PASS10_LONG_STEPS = 20000                # Q5 (optional): twice Pass 8a's 10,000 pretraining steps
# Where notebook 20 saves its new encoders (the seed-0 encoder comes from notebook 15's output, found and
# hash-checked by ssl_pretrain.find_encoder)
PASS10_CKPT = {"pt1": "/kaggle/working/ckpt/mlm_s1", "pt2": "/kaggle/working/ckpt/mlm_s2",
               "pt20k": "/kaggle/working/ckpt/mlm_20k"}


def _p10(arm, budget, seed, init_dir, tag="") -> RunConfig:
    """Frozen ingredient settings (helper weight 0.3, FGM epsilon 0.5) on a TAPT encoder; masked tests scored."""
    sw = ARMS[arm]
    return RunConfig(arm, budget, seed, helper_weight=PASS1_HELPER_WEIGHT if sw["helpers"] else 0.0,
                     fgm_epsilon=PASS2_FGM_EPSILON if sw.get("fgm") else 0.0, init_dir=init_dir,
                     eval_test=True, eval_masked=True, tag=tag)


def pass10_stack(init_dir: str = PASS8_CKPT["mlm"]) -> list[RunConfig]:
    """Pass 10 Q1 (vault: Experiment Plan, "Pass 10 plan"): the Pass 9 2x2x2 (helper heads on L, FGM,
    self-training) on the MLM-TAPT encoder of Pass 8a. The six new cells A31-A36 at 10% and 1%, 5 seeds
    (60 runs, longest first), plus A20 and A21 at 1% seed 0 again (tag "rerun"): with the reused encoder
    they must equal Pass 8a exactly. A20 and A21 themselves are reused from Pass 8a."""
    order = ("A36", "A35", "A34", "A32", "A33", "A31")   # self-training + FGM runs are the longest
    runs = [_p10(arm, b, s, init_dir) for b in (10, 1) for arm in order for s in PASS10_SEEDS]
    return runs + [_p10(arm, 1, 0, init_dir, tag="rerun") for arm in ("A20", "A21")]


def pass10_seeds(ckpt: dict | None = None) -> list[RunConfig]:
    """Pass 10 Q2: A20 fine-tuned from the MLM encoders pretrained with seeds 1 and 2, at 10% and 1%,
    5 seeds each (20 runs, tags "pt1", "pt2"). Compared with A1 and with Pass 8a's seed-0 A20."""
    ckpt = ckpt or PASS10_CKPT
    return [_p10("A20", b, s, ckpt[f"pt{k}"], tag=f"pt{k}")
            for b in (10, 1) for k in PASS10_PRETRAIN_SEEDS for s in PASS10_SEEDS]


def pass10_curve(init_dir: str = PASS8_CKPT["mlm"]) -> list[RunConfig]:
    """Pass 10 Q3: A20 (seed-0 encoder) at 30% and 5% labels, 5 seeds (10 runs). A1 exists at every budget."""
    return [_p10("A20", b, s, init_dir) for b in (30, 5) for s in PASS10_SEEDS]


def pass10_curve_100(init_dir: str = PASS8_CKPT["mlm"]) -> list[RunConfig]:
    """Pass 10 Q3 extension: A20 with every label (100%), 5 seeds. Compared with Pass 1's A1 at 100%."""
    return [_p10("A20", 100, s, init_dir) for s in PASS10_SEEDS]


def pass10_long(init_dir: str = PASS10_CKPT["pt20k"]) -> list[RunConfig]:
    """Pass 10 Q5 (optional): A20 from a 20,000-step MLM encoder at 10% and 1%, 5 seeds (tag "pt20k")."""
    return [_p10("A20", b, s, init_dir, tag="pt20k") for b in (10, 1) for s in PASS10_SEEDS]


# ---------------------------------------------------------------------------- Pass 11

PASS11_SEEDS = PASS1_SEEDS
PASS11_HIGH_BUDGET = 70                 # the seed paper's regime: 70% of the training split labeled (needs the extra budgets file)
PASS11_LONG_MULT = 35                   # E3: 35 x 200 = 7,000 steps at 1%, about 9.9 passes over U
PASS11_DOSE_STEPS = {"dose100": 100, "dose350": 350, "dose1000": 1000, "dose3000": 3000}   # E2: MLM pretraining steps
PASS11_CKPT = {tag: f"/kaggle/working/ckpt/mlm_{tag}" for tag in PASS11_DOSE_STEPS}


def _p11(arm, budget, seed, init_dir="", tag="", **kw) -> RunConfig:
    """Frozen ingredient settings (helper weight 0.3, FGM epsilon 0.5); test scored, masked tests not."""
    sw = ARMS[arm]
    return RunConfig(arm, budget, seed, helper_weight=PASS1_HELPER_WEIGHT if sw["helpers"] else 0.0,
                     fgm_epsilon=PASS2_FGM_EPSILON if sw.get("fgm") else 0.0,
                     init_dir=init_dir if sw.get("init") else "", eval_test=True, tag=tag, **kw)


def pass11_high(init_dir: str = PASS8_CKPT["mlm"]) -> list[RunConfig]:
    """E6, the seed paper's regime: A5 (the seed recipe), A7 (FGM), A1 and A20 at 70% labels, 5 seeds (20 runs,
    longest first). Needs the extra budgets file (data.make_extra_budgets)."""
    b = PASS11_HIGH_BUDGET
    return ([_p11("A5", b, s) for s in PASS11_SEEDS] + [_p11("A7", b, s) for s in PASS11_SEEDS]
            + [_p11("A1", b, s) for s in PASS11_SEEDS] + [_p11("A20", b, s, init_dir) for s in PASS11_SEEDS])


def pass11_ours(init_dir: str = PASS8_CKPT["mlm"]) -> list[RunConfig]:
    """E6: our recipe (A36, MLM + helpers + FGM + self-training) at 30% and 5% labels, 5 seeds (10 runs). A36 at 1% and
    10% is Pass 10; A5, A1, A20 at 5% and 30% exist (Passes 2a, 10)."""
    return [_p11("A36", b, s, init_dir) for b in (30, 5) for s in PASS11_SEEDS]


def pass11_ganlong() -> list[RunConfig]:
    """E3: GAN-BERT (A4) and plain BERT (A1) at 1% labels with 35x the steps (tag "long35"), 5 seeds (10 runs).
    step_mult also multiplies the passes over L, so the control is A1 at the same step count: the contrast is
    A4 - A1 at matched steps."""
    return ([_p11("A4", 1, s, tag="long35", step_mult=PASS11_LONG_MULT) for s in PASS11_SEEDS]
            + [_p11("A1", 1, s, tag="long35", step_mult=PASS11_LONG_MULT) for s in PASS11_SEEDS])


def pass11_dose(ckpt: dict | None = None) -> list[RunConfig]:
    """E2: A20 at 1% labels from MLM encoders pretrained for 100, 350, 1,000 and 3,000 steps (seed 0, everything else
    as Pass 8a), 5 seeds each (20 runs, tags dose100 ... dose3000). 10,000 steps is Pass 8a's A20."""
    ckpt = ckpt or PASS11_CKPT
    return [_p11("A20", 1, s, ckpt[tag], tag=tag) for tag in PASS11_DOSE_STEPS for s in PASS11_SEEDS]


def pass11_a37(init_dir: str = PASS8_CKPT["mlm"]) -> list[RunConfig]:
    """The seed recipe (GAN + helper heads) on the MLM encoder, 10% and 1% labels, 5 seeds (10 runs)."""
    return [_p11("A37", b, s, init_dir) for b in (10, 1) for s in PASS11_SEEDS]


def pass11_full() -> list[RunConfig]:
    """Whole-dataset scoring (vault: Pass 11 Predictions, addendum): A5 and A1 at 70% labels and A1 with every label,
    5 seeds, tag "full", with predictions saved for every training row as well (the seed paper's Table II covers all
    headlines). The same training as the untagged runs, so test predictions must reproduce them exactly. Plus an A1
    re-run at 1%, seed 0 (tag "rerun11") as an exact-match check of Pass 1 after the data-loading change.
    16 runs, longest first. Needs the extra budgets file for 70%."""
    kw = dict(tag="full", eval_train=True)
    return ([_p11("A5", PASS11_HIGH_BUDGET, s, **kw) for s in PASS11_SEEDS]
            + [_p11("A1", 100, s, **kw) for s in PASS11_SEEDS]
            + [_p11("A1", PASS11_HIGH_BUDGET, s, **kw) for s in PASS11_SEEDS]
            + [_p11("A1", 1, 0, tag="rerun11")])


def pass11_all(init_dir: str = PASS8_CKPT["mlm"], ckpt: dict | None = None) -> list[RunConfig]:
    """Everything of Pass 11's core notebook in one list (70 runs), longest runs first so the two GPUs stay busy."""
    high = pass11_high(init_dir)
    ours = pass11_ours(init_dir)
    long_ = pass11_ganlong()
    by = lambda rs, arm, b=None: [r for r in rs if r.arm == arm and (b is None or r.budget == b)]
    return (by(high, "A5") + by(high, "A7") + by(long_, "A4") + by(ours, "A36", 30) + by(long_, "A1")
            + by(high, "A1") + by(high, "A20") + by(ours, "A36", 5) + pass11_a37(init_dir) + pass11_dose(ckpt))


PLANS = {"lr_tuning": lr_tuning, "w_tuning": w_tuning, "gan_pilot": gan_pilot,
         "session2": session2, "pass1": pass1, "fgm_tuning": fgm_tuning,
         "pass2a": pass2a, "pass2b": pass2b, "pass3": pass3,
         "gan_check": gan_check, "pass4": pass4, "pass4_full100": pass4_full100,
         "pair_weight_tuning": pair_weight_tuning, "cleaned_drop_tuning": cleaned_drop_tuning, "pass5": pass5,
         "pass6": pass6, "pass7": pass7,
         "pass8_mlm": pass8_mlm, "pass8_mlm_100": pass8_mlm_100,
         "pass8_jepa": pass8_jepa, "pass8_jepa_100": pass8_jepa_100, "pass8_diff": pass8_diff,
         "pass9": pass9, "pass9_100": pass9_100,
         "pass10_stack": pass10_stack, "pass10_seeds": pass10_seeds, "pass10_curve": pass10_curve,
         "pass10_curve_100": pass10_curve_100, "pass10_long": pass10_long,
         "pass11_high": pass11_high, "pass11_ours": pass11_ours, "pass11_ganlong": pass11_ganlong,
         "pass11_dose": pass11_dose, "pass11_a37": pass11_a37, "pass11_all": pass11_all,
         "pass11_full": pass11_full}
