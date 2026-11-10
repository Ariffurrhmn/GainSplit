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

    def __post_init__(self):
        if self.arm not in ARMS:
            raise ValueError(f"unknown arm {self.arm!r}; choose from {sorted(ARMS)}")
        if self.arm == "A7" and not self.fgm_epsilon > 0:
            raise ValueError(f"A7 needs fgm_epsilon > 0 (got {self.fgm_epsilon})")
        if self.arm != "A7" and self.fgm_epsilon != 0:
            raise ValueError(f"fgm_epsilon is only used by A7 (got {self.fgm_epsilon} for {self.arm})")
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
