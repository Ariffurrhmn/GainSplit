"""Pass 8c: a small diffusion model over BERT's [CLS] vectors (generative AI that is not an LLM).

GAN-BERT's generator turns noise into fake [CLS] vectors in one shot and is trained adversarially.
Here that slot is filled by a denoising diffusion model (DDPM, Ho et al. 2020; deterministic DDIM
sampling, Song et al. 2021): a small network learns to remove Gaussian noise from real [CLS]
vectors, and new vectors are made by starting from noise and denoising step by step. It is trained
with a plain regression loss (predict the added noise), never adversarially.

It is used in two ways (vault: Experiment Plan, "Pass 8 plan"):
  A24 "fake":        GAN-BERT's k+1 discriminator, with the GAN generator replaced by this model's
                     samples. Does a better generator in the same slot help?
  A25 "consistency": no fake class. Each U vector is noised part-way (level CONS_T of T) and the
                     diffusion model maps it back towards real [CLS] vectors: a new, nearby vector.
                     The sarcasm head must give it the same prediction as the original (KL loss).
                     This uses the generative model as a data augmenter for U.
  A26 "noise":       control for A25: the same noise level, but no denoising (Gaussian noise only).
                     A25 - A26 isolates what the generative model adds over plain feature noise.

The model works on standardized vectors (running mean and spread of the real [CLS] vectors), so its
settings do not depend on BERT's raw scale. All maths here is float32.
"""
import math

import torch
from torch import nn

T_STEPS = 100          # diffusion length (noise levels 1..T)
SAMPLE_STEPS = 10      # DDIM steps when generating from pure noise (A24)
CONS_T = 30            # A25/A26: noise level of the round trip (alpha_bar ~ 0.79: about 0.46 sd of noise)
CONS_STEPS = 3         # DDIM steps of the round trip back from CONS_T to 0
DENOISER_WIDTH = 1024
DENOISER_LR = 1e-3     # its own Adam optimizer, constant lr
STAT_MOMENTUM = 0.99   # running mean/sd of the real vectors
Z_CLIP = 5.0           # each DDIM step clips its clean-vector estimate to +-5 sd (standard practice, as images clip to [-1, 1])
SETTINGS = {"T_STEPS": T_STEPS, "SAMPLE_STEPS": SAMPLE_STEPS, "CONS_T": CONS_T, "CONS_STEPS": CONS_STEPS,
            "DENOISER_WIDTH": DENOISER_WIDTH, "DENOISER_LR": DENOISER_LR, "STAT_MOMENTUM": STAT_MOMENTUM,
            "Z_CLIP": Z_CLIP}


def cosine_alpha_bar(T: int, s: float = 0.008) -> torch.Tensor:
    """alpha_bar[t] for t = 0..T (Nichol & Dhariwal 2021): share of the signal left at level t; alpha_bar[0] = 1."""
    t = torch.arange(T + 1, dtype=torch.float64)
    f = torch.cos(((t / T) + s) / (1 + s) * math.pi / 2) ** 2
    ab = (f / f[0]).clamp(min=1e-5, max=1.0)
    return ab.float()


def time_embedding(t: torch.Tensor, dim: int = 128) -> torch.Tensor:
    """Sinusoidal embedding of the noise level (as in transformers' positions)."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    a = t.float()[:, None] * freqs[None]
    return torch.cat([torch.sin(a), torch.cos(a)], dim=1)


class FeatureDiffusion(nn.Module):
    """Denoiser eps(z_t, t) for standardized [CLS] vectors, plus sampling and the A25 round trip."""

    def __init__(self, hidden: int, width: int = DENOISER_WIDTH, T: int = T_STEPS):
        super().__init__()
        self.T = T
        self.register_buffer("alpha_bar", cosine_alpha_bar(T))
        self.register_buffer("mu", torch.zeros(hidden))
        self.register_buffer("sd", torch.ones(hidden))
        self.register_buffer("n_seen", torch.zeros((), dtype=torch.long))
        self.t_mlp = nn.Sequential(nn.Linear(128, width), nn.SiLU(), nn.Linear(width, width))
        self.inp = nn.Linear(hidden, width)
        self.body = nn.Sequential(nn.SiLU(), nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU())
        self.out = nn.Linear(width, hidden)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    # ---- standardization
    @torch.no_grad()
    def update_stats(self, x: torch.Tensor) -> None:
        x = x.float()
        m, s = x.mean(0), x.std(0).clamp(min=1e-4)
        if int(self.n_seen) == 0:
            self.mu.copy_(m)
            self.sd.copy_(s)
        else:
            self.mu.mul_(STAT_MOMENTUM).add_(m, alpha=1 - STAT_MOMENTUM)
            self.sd.mul_(STAT_MOMENTUM).add_(s, alpha=1 - STAT_MOMENTUM)
        self.n_seen += x.shape[0]

    def to_z(self, x):
        return (x.float() - self.mu) / self.sd

    def from_z(self, z):
        return z * self.sd + self.mu

    # ---- network
    def eps(self, z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.inp(z_t) + self.t_mlp(time_embedding(t))
        return self.out(self.body(h))

    def loss(self, x_real: torch.Tensor) -> torch.Tensor:
        """Denoising loss on real vectors (passed detached by the caller): predict the added noise."""
        z0 = self.to_z(x_real)
        t = torch.randint(1, self.T + 1, (z0.shape[0],), device=z0.device)
        ab = self.alpha_bar[t][:, None]
        noise = torch.randn_like(z0)
        z_t = ab.sqrt() * z0 + (1 - ab).sqrt() * noise
        return (self.eps(z_t, t) - noise).pow(2).mean()

    def _ddim(self, z: torch.Tensor, t_from: int, n_steps: int) -> torch.Tensor:
        """Deterministic DDIM from level t_from down to 0 in n_steps steps (gradients flow through z)."""
        ts = torch.linspace(t_from, 0, n_steps + 1).round().long().tolist()
        for t, s in zip(ts[:-1], ts[1:]):
            ab_t, ab_s = self.alpha_bar[t], self.alpha_bar[s]
            e = self.eps(z, torch.full((z.shape[0],), t, device=z.device, dtype=torch.long))
            z0 = ((z - (1 - ab_t).sqrt() * e) / ab_t.sqrt()).clamp(-Z_CLIP, Z_CLIP)
            z = ab_s.sqrt() * z0 + (1 - ab_s).sqrt() * e
        return z

    @torch.no_grad()
    def sample(self, n: int, device, steps: int = SAMPLE_STEPS) -> torch.Tensor:
        """A24: n new vectors from pure noise, in BERT's [CLS] scale."""
        z = torch.randn(n, self.mu.shape[0], device=device)
        return self.from_z(self._ddim(z, self.T, steps))

    def noised(self, x: torch.Tensor, t: int = CONS_T):
        """Standardized x pushed to noise level t: (z_t, alpha_bar_t)."""
        z0 = self.to_z(x)
        ab = self.alpha_bar[t]
        return ab.sqrt() * z0 + (1 - ab).sqrt() * torch.randn_like(z0), ab

    def round_trip(self, x: torch.Tensor, t: int = CONS_T, steps: int = CONS_STEPS) -> torch.Tensor:
        """A25: noise x to level t, then denoise back to level 0. Differentiable in x."""
        z_t, _ = self.noised(x, t)
        return self.from_z(self._ddim(z_t, t, steps))


def noise_only(x: torch.Tensor, mu: torch.Tensor, sd: torch.Tensor, t: int = CONS_T, T: int = T_STEPS) -> torch.Tensor:
    """A26: the same noise level as A25's round trip, no denoising: z + sqrt((1-ab)/ab) * noise.

    Uses a running mean/sd like A25 so the noise is measured in the same units (per-dimension spread)."""
    ab = cosine_alpha_bar(T)[t].to(x.device)
    z = (x.float() - mu) / sd
    z = z + ((1 - ab) / ab).sqrt() * torch.randn_like(z)
    return z * sd + mu


class RunningStats(nn.Module):
    """A26 only: the running mean/sd of the real vectors (A25 keeps them inside FeatureDiffusion)."""

    def __init__(self, hidden: int):
        super().__init__()
        self.register_buffer("mu", torch.zeros(hidden))
        self.register_buffer("sd", torch.ones(hidden))
        self.register_buffer("n_seen", torch.zeros((), dtype=torch.long))

    update_stats = FeatureDiffusion.update_stats
