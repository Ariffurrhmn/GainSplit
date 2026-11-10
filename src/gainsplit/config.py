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
}
IMPLEMENTED_ARMS = set(ARMS)


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

    def __post_init__(self):
        if self.arm not in ARMS:
            raise ValueError(f"unknown arm {self.arm!r}; choose from {sorted(ARMS)}")
        if self.arm == "A7" and not self.fgm_epsilon > 0:
            raise ValueError(f"A7 needs fgm_epsilon > 0 (got {self.fgm_epsilon})")
        if self.arm != "A7" and self.fgm_epsilon != 0:
            raise ValueError(f"fgm_epsilon is only used by A7 (got {self.fgm_epsilon} for {self.arm})")

    @property
    def run_id(self) -> str:
        base = f"{self.arm}_p{self.budget}_s{self.seed}"
        return f"{base}_{self.tag}" if self.tag else base

    @property
    def switches(self) -> dict:
        return ARMS[self.arm]

    def steps(self, n_labeled: int) -> int:
        """Same rule for every arm (equal step budget, Decisions 2026-11-10)."""
        return max(self.min_steps, self.passes * math.ceil(n_labeled / self.batch_size))

    def to_dict(self) -> dict:
        return asdict(self)
