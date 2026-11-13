"""Settings for one training run, and which components each arm switches on.

Every arm reads the same RunConfig. Arms differ only in the two switches in ARMS,
so any difference in results comes from those switches (see vault/Decisions.md).
"""
from dataclasses import dataclass, asdict
import math

# Pass 1 is a 2x2: GAN off/on crossed with helper heads off/on. Pass 2 adds controls.
# `gan` and `helpers` are always present; the other keys are optional extra switches:
#   l_only  helper losses use L rows only (no U batch)
#   shuffle helper labels are permuted at random across L + U
#   pseudo  one round of self-training (the A1 teacher labels U)
#   fgm     adversarial step on the word embeddings
#   tfidf   no neural network at all (scikit-learn, CPU)
#   pairs   Pass 4: LLM-written headline pairs join every step as a labeled second batch;
#           the value names the pair set (see PAIR_SETS)
#   no_gold Pass 4: the gold-labeled rows L are not used at all (only their count sets the steps)
ARMS = {
    "A0": {"gan": False, "helpers": False, "tfidf": True},    # TF-IDF + logistic regression (floor)
    "A1": {"gan": False, "helpers": False},                   # plain BERT on L
    "A2": {"gan": False, "helpers": True, "l_only": True},    # A3 with the helper losses on L only
    "A3": {"gan": False, "helpers": True},                    # BERT + sentiment/emotion heads on L + U
    "A4": {"gan": True,  "helpers": False},                   # GAN-BERT
    "A5": {"gan": True,  "helpers": True},                    # GAN-BERT + the same helper heads as A3
    "A6": {"gan": False, "helpers": False, "pseudo": True},   # self-training, one round
    "A7": {"gan": False, "helpers": False, "fgm": True},      # BERT + FGM on L
    "A8": {"gan": False, "helpers": True, "shuffle": True},   # A3 with shuffled helper labels
    "A9": {"gan": False, "helpers": True, "l_only": True, "shuffle": True},  # A2 with shuffled helper labels
    # Pass 4: generated sarcastic/serious headline pairs (vault: Experiment Plan, "Pass 4 plan")
    "A10": {"gan": False, "helpers": False, "pairs": "unfiltered"},               # A1 + pairs
    "A11": {"gan": False, "helpers": False, "fgm": True, "pairs": "unfiltered"},  # A7 (FGM) + pairs
    "A12": {"gan": False, "helpers": False, "pairs": "unpaired"},                 # control: one side per pair
    "A13": {"gan": False, "helpers": False, "pairs": "filtered"},                 # judge-filtered pairs
    "A14": {"gan": False, "helpers": False, "pairs": "matched"},                  # control for A13: random subset
    "A15": {"gan": False, "helpers": False, "pairs": "unfiltered", "no_gold": True},  # pairs only, no gold labels
    # Pass 5: clean the shortcut out of the pairs and/or down-weight their loss (vault: Experiment Plan, "Pass 5 plan")
    "A16": {"gan": False, "helpers": False, "pairs": "cleaned"},                     # A10 with the worst-scoring seeds dropped
    "A17": {"gan": False, "helpers": False, "fgm": True, "pairs": "cleaned"},        # A11 with the same cleaning
}
IMPLEMENTED_ARMS = set(ARMS)

# Pass 3: what the helper heads are trained on (helper arms only; "" = the original hard teacher labels)
#   soft     soft cross-entropy against the teacher probabilities
#   flip25   25% of the hard labels changed to another class (flip50: 50%)
#   vader    sentiment label from VADER instead of the sentiment teacher
#   teacher2 sentiment label from a second sentiment model
#   sentonly / emoonly  only that head is trained
HELPER_VARIANTS = ("", "soft", "flip25", "flip50", "vader", "teacher2", "sentonly", "emoonly")
TRAIN_DATA = ("headlines", "isarcasm")
# Pass 4-5 pair sets (built in data.load_pairs):
#   unfiltered  every pair that passed the automatic checks (pairs_train.csv)
#   filtered    pairs kept by the judge-v2 filter (pairs_train_filtered.csv)
#   matched     random subset of the unfiltered pairs with the same pairs per model as `filtered`
#   unpaired    one side of every unfiltered pair (side drawn at random): same text source, no topic match
#   cleaned     Pass 5: unfiltered minus the seeds whose generated text leans most on the AI
#               writing-tic words found in pair_checks.shortcut_word_table (pair_shortcut_scores.csv)
PAIR_SETS = ("unfiltered", "filtered", "matched", "unpaired", "cleaned")


@dataclass
class RunConfig:
    arm: str
    budget: int                 # % of train that keeps its label: 1, 5, 10, 30 or 100
    seed: int                   # picks the labeled subset, head initialisation and batch order
    lr: float = 5e-5            # chosen by lr_tuning (A1, 10%, dev), frozen 2026-11-10
    batch_size: int = 32
    passes: int = 10            # training length = 10 passes over L ...
    min_steps: int = 200        # ... but never fewer than 200 steps (matters at 1%)
    warmup_ratio: float = 0.1   # lr rises over the first 10% of steps, then falls to 0
    weight_decay: float = 0.01
    max_len: int = 48           # 99% of headlines are 24 tokens or fewer
    model_name: str = "bert-base-uncased"
    helper_weight: float = 0.0  # w for the helper losses; set once tuned (A3/A5 only)
    fp16: bool = True           # half precision on GPU: faster, same results in practice
    eval_test: bool = False     # False while tuning: only dev is scored, test stays untouched
    fgm_epsilon: float = 0.0    # size of the FGM perturbation; must be > 0 for A7, 0 for every other arm
    tag: str = ""               # extra label for tuning runs, e.g. "lr3e-05"
    helper_variant: str = ""    # Pass 3: how the helper labels are changed (see HELPER_VARIANTS)
    eval_masked: bool = False   # Pass 3: also score the cue-masked and random-masked test sets (needs eval_test)
    train_data: str = "headlines"  # "isarcasm": train on the iSarcasmEval training set (A1 only, budget 100)
    gen_lr: float = 0.0         # GAN arms only: learning rate of the generator's optimizer (0 = same as lr)
    step_mult: int = 1          # multiplies the step count from steps() (1 = the usual rule)
    pair_batch: int = 32        # Pass 4: generated rows per step (16 whole pairs); fixed 2026-11-13, not tuned
    explain: bool = False       # Pass 4: Integrated Gradients on a fixed test sample after training (needs eval_test)
    pair_weight: float = 1.0    # Pass 5: pair rows' share of the sarcasm loss (1.0 = same as Pass 4, full weight)
    cleaned_drop_pct: float = 0.0  # Pass 5: for pairs="cleaned", the % of worst-scoring seeds dropped (0-100)

    def __post_init__(self):
        if self.arm not in ARMS:
            raise ValueError(f"unknown arm {self.arm!r}; choose from {sorted(ARMS)}")
        fgm = ARMS[self.arm].get("fgm", False)
        if fgm and not self.fgm_epsilon > 0:
            raise ValueError(f"{self.arm} needs fgm_epsilon > 0 (got {self.fgm_epsilon})")
        if not fgm and self.fgm_epsilon != 0:
            raise ValueError(f"fgm_epsilon is only used by FGM arms (A7, A11) (got {self.fgm_epsilon} for {self.arm})")
        if self.helper_variant not in HELPER_VARIANTS:
            raise ValueError(f"unknown helper_variant {self.helper_variant!r}; choose from {list(HELPER_VARIANTS)}")
        if self.helper_variant and not ARMS[self.arm]["helpers"]:
            raise ValueError(f"helper_variant {self.helper_variant!r} needs a helper arm (got {self.arm})")
        if self.train_data not in TRAIN_DATA:
            raise ValueError(f"unknown train_data {self.train_data!r}; choose from {list(TRAIN_DATA)}")
        if self.train_data == "isarcasm" and (self.arm != "A1" or self.budget != 100):
            raise ValueError(f"train_data='isarcasm' is only for A1 at budget 100 (got {self.arm} at {self.budget})")

        if self.gen_lr < 0:
            raise ValueError(f"gen_lr must be >= 0 (got {self.gen_lr})")
        if self.gen_lr > 0 and not ARMS[self.arm]["gan"]:
            raise ValueError(f"gen_lr is only used by GAN arms (got {self.gen_lr} for {self.arm})")
        if self.step_mult < 1:
            raise ValueError(f"step_mult must be >= 1 (got {self.step_mult})")
        pairs = ARMS[self.arm].get("pairs")
        if pairs is not None and pairs not in PAIR_SETS:
            raise ValueError(f"unknown pair set {pairs!r} for {self.arm}")
        if pairs and (self.pair_batch < 2 or self.pair_batch % 2):
            raise ValueError(f"pair_batch must be an even number >= 2 (got {self.pair_batch})")
        if pairs and self.train_data != "headlines":
            raise ValueError(f"{self.arm} uses the Headlines pairs; train_data must be 'headlines'")
        if not pairs and self.pair_weight != 1.0:
            raise ValueError(f"pair_weight is only used by pair arms (got {self.pair_weight} for {self.arm})")
        if self.pair_weight < 0:
            raise ValueError(f"pair_weight must be >= 0 (got {self.pair_weight})")
        if pairs != "cleaned" and self.cleaned_drop_pct != 0.0:
            raise ValueError(f"cleaned_drop_pct is only used by pairs='cleaned' (got pairs={pairs!r})")
        if pairs == "cleaned" and not (0 < self.cleaned_drop_pct < 100):
            raise ValueError(f"cleaned_drop_pct must be between 0 and 100, exclusive (got {self.cleaned_drop_pct})")
        if self.explain and not self.eval_test:
            raise ValueError("explain needs eval_test=True (it reads the test set)")
        if self.explain and (ARMS[self.arm]["gan"] or self.arm == "A0"):
            raise ValueError(f"explain is written for arms with a plain sarcasm head (got {self.arm})")

    @property
    def run_id(self) -> str:
        base = f"{self.arm}_p{self.budget}_s{self.seed}"
        return f"{base}_{self.tag}" if self.tag else base

    @property
    def switches(self) -> dict:
        return ARMS[self.arm]

    def steps(self, n_labeled: int) -> int:
        """Same rule for every arm (equal step budget, Decisions 2026-11-10)."""
        return self.step_mult * max(self.min_steps, self.passes * math.ceil(n_labeled / self.batch_size))

    def to_dict(self) -> dict:
        return asdict(self)
