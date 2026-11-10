"""The networks. A1 needs only BERT plus a sarcasm head.

A3/A4/A5 will add helper heads and a generator/discriminator on top of the same
encoder, so every arm starts from the same BERT and the same [CLS] vector.
"""
import torch
from torch import nn
from transformers import AutoModel


class SarcasmClassifier(nn.Module):
    def __init__(self, model_name: str, n_classes: int = 2, dropout: float = 0.1):
        super().__init__()
        # No pooling layer: we use the raw [CLS] vector, as GAN-BERT does.
        self.encoder = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, n_classes)

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """One 768-number summary per headline: BERT's output at the [CLS] position."""
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0]

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Two scores per headline (not sarcastic, sarcastic); softmax turns them into probabilities."""
        return self.head(self.dropout(self.encode(input_ids, attention_mask)))
