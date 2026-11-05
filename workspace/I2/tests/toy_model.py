"""Small causal model with the same editor contract as ASU Llama."""

from __future__ import annotations

import torch
from torch import nn


class Logits(object):
    def __init__(self, logits):
        self.logits = logits


class ToyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_idx = 0
        self.num_heads = 2
        self.head_dim = 4
        self.activation_editor = None

    def forward(self, hidden):
        batch, tokens, channels = hidden.shape
        heads = hidden.new_zeros(batch, self.num_heads, tokens, self.head_dim)
        heads[:, 0, :, 0] = hidden[:, :, 0]
        heads[:, 1, :, 0] = hidden[:, :, 1]
        editor = self.activation_editor
        if editor is not None:
            heads = editor("attention", heads, self, "bhsd")
        product = heads[:, 0, 0, 0] * heads[:, 1, 0, 0]
        feat = product[:, None, None].expand(batch, tokens, 1)
        rest = hidden.new_zeros(batch, tokens, channels - 1)
        out = torch.cat([feat, rest], dim=-1)
        down = hidden.new_zeros(batch, tokens, channels)
        if editor is not None:
            down = editor("mlp", down, self, "bth")
        return out + down


class ToyCausal(nn.Module):
    """Score is the product of head 0 and head 1 at token 0."""

    def __init__(self):
        super().__init__()
        self.mcsd_model_id = "toy-causal"
        self.mcsd_differentiable = True
        self.layer_count = 1
        self.layer = ToyLayer()
        self.mcsd_layers = [self.layer]
        self.vocab_size = 4
        self.forward_count = 0
        table = torch.zeros(8, 2)
        table[1] = torch.tensor([3.0, 4.0])
        table[2] = torch.tensor([1.0, 2.0])
        self.register_buffer("features", table)

    def forward(self, input_ids, attention_mask=None, labels=None, attention_temp=None, layers_id=None, use_cache=False):
        del attention_mask, labels, layers_id, use_cache
        if attention_temp not in (None, 1, 1.0):
            raise RuntimeError("localization forward must not apply attention temperature")
        self.forward_count += 1
        batch, tokens = input_ids.shape
        feats = self.features[input_ids]
        hidden = torch.cat([feats, feats.new_zeros(batch, tokens, 2)], dim=-1)
        hidden = self.layer(hidden)
        pad = hidden.new_zeros(batch, tokens, self.vocab_size - 1)
        return Logits(torch.cat([hidden[:, :, :1], pad], dim=-1))
