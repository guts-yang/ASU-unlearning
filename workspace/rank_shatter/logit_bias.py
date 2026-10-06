"""Frozen-after-training vocab bias added to teacher logits."""

import torch
from torch import nn


class VocabLogitBias(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(vocab_size))

    def apply(self, logits):
        return logits + self.bias.to(device=logits.device, dtype=logits.dtype)

    def forward(self, logits):
        return self.apply(logits)
