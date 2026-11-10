"""The networks. Every arm shares one BERT encoder and its [CLS] vector.

A1: BERT + a linear sarcasm head.
A3: A1 + two linear helper heads (sentiment, emotion).
A4: GAN-BERT (Croce et al. 2020): a generator makes fake [CLS]-like vectors and a
    discriminator replaces the sarcasm head; it sees k real classes plus a "fake" class.
A5: A4 + the same two helper heads as A3, reading the same [CLS] vector.
Which parts exist is decided by two switches, `gan` and `helpers` (see config.ARMS).
"""
import torch
from torch import nn
from transformers import AutoModel

N_SENT, N_EMO = 3, 5  # helper classes: negative/neutral/positive; joy/sadness/anger/fear/neutral_other


class Generator(nn.Module):
    """Noise (100 numbers) -> a fake vector the size of a [CLS] vector."""

    def __init__(self, hidden: int, noise_dim: int = 100, dropout: float = 0.1):
        super().__init__()
        self.noise_dim = noise_dim
        self.net = nn.Sequential(
            nn.Linear(noise_dim, hidden), nn.LeakyReLU(0.2), nn.Dropout(dropout), nn.Linear(hidden, hidden))

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        return self.net(noise)


class Discriminator(nn.Module):
    """Reads a [CLS]-sized vector; gives k real-class scores plus one "fake" score
    (last index), and its hidden layer (used by the generator's feature-matching loss)."""

    def __init__(self, hidden: int, n_classes: int, dropout: float = 0.1):
        super().__init__()
        self.body = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.LeakyReLU(0.2), nn.Dropout(dropout))
        self.out = nn.Linear(hidden, n_classes + 1)

    def forward(self, x: torch.Tensor):
        h = self.body(x)
        return self.out(h), h


class SarcasmClassifier(nn.Module):
    def __init__(self, model_name: str, n_classes: int = 2, dropout: float = 0.1,
                 gan: bool = False, helpers: bool = False, noise_dim: int = 100):
        super().__init__()
        # No pooling layer: we use the raw [CLS] vector, as GAN-BERT does.
        self.encoder = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
        hidden = self.encoder.config.hidden_size  # 768 for bert-base; the tests use a small BERT
        self.n_classes, self.gan, self.helpers = n_classes, gan, helpers
        self.dropout = nn.Dropout(dropout)
        if gan:
            # the discriminator takes the place of the plain sarcasm head
            self.disc = Discriminator(hidden, n_classes, dropout)
            self.generator = Generator(hidden, noise_dim, dropout)
        else:
            self.head = nn.Linear(hidden, n_classes)
        if helpers:
            self.sent_head = nn.Linear(hidden, N_SENT)
            self.emo_head = nn.Linear(hidden, N_EMO)

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """One 768-number summary per headline: BERT's output at the [CLS] position."""
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0]

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Two scores per headline (not sarcastic, sarcastic); softmax turns them into probabilities.
        GAN arms: only the first k discriminator scores, so the "fake" score never affects a prediction."""
        cls = self.encode(input_ids, attention_mask)
        if self.gan:
            return self.disc(cls)[0][:, :self.n_classes]
        return self.head(self.dropout(cls))
