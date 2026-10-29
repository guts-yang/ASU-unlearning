#!/usr/bin/env python3
"""CPU check that τ=1 matches the original MLP and τ_M changes the gate pre-activation.

Uses a randomly initialized tiny Llama. Does not load TOFU weights and does not edit upstream files.
"""

import os
import sys
from pathlib import Path

import torch
from transformers import LlamaConfig

ROOT = Path("/usr/local/ASU-unlearning")
sys.path.insert(0, str(ROOT / "Right-to-be-forgotten"))
sys.path.insert(0, str(ROOT / "workspace/I"))
os.chdir(ROOT / "Right-to-be-forgotten")

from mcsd.teacher import GateTemperatureTeacher  # noqa: E402
from my_models.my_llama import LlamaForCausalLM  # noqa: E402


def main():
    torch.manual_seed(0)
    config = LlamaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=128,
        max_position_embeddings=64,
        attn_implementation="eager",
    )
    reference = LlamaForCausalLM(config)
    model = LlamaForCausalLM(config)
    model.load_state_dict(reference.state_dict())
    teacher = GateTemperatureTeacher(model)
    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        base = reference(input_ids=input_ids, attention_mask=attention_mask, attention_temp=1.0, use_cache=False)
        wrapped = teacher.forward(input_ids, attention_mask, attention_temp=1.0, mlp_gate_temp=1.0)
        attn = teacher.forward(input_ids, attention_mask, attention_temp=2.3, mlp_gate_temp=1.0)
        mlp = teacher.forward(input_ids, attention_mask, attention_temp=1.0, mlp_gate_temp=2.3)
    delta_identity = (base.logits - wrapped.logits).abs().max().item()
    delta_attn = (wrapped.logits - attn.logits).abs().max().item()
    delta_mlp = (wrapped.logits - mlp.logits).abs().max().item()
    if delta_identity > 0.0:
        raise SystemExit(f"temperature 1 changed logits by {delta_identity}")
    if delta_attn == 0.0:
        raise SystemExit("attention temperature did not change logits")
    if delta_mlp == 0.0:
        raise SystemExit("mlp gate temperature did not change logits")
    hidden = torch.randn(2, 4, config.hidden_size)
    mlp_mod = reference.model.layers[0].mlp
    with torch.inference_mode():
        manual = mlp_mod.down_proj(mlp_mod.act_fn(mlp_mod.gate_proj(hidden) / 2.3) * mlp_mod.up_proj(hidden))
        direct = mlp_mod.down_proj(mlp_mod.act_fn(mlp_mod.gate_proj(hidden)) * mlp_mod.up_proj(hidden))
    formula_delta = (manual - direct).abs().max().item()
    if formula_delta == 0.0:
        raise SystemExit("manual gate scaling was a no-op")
    print(
        {
            "identity_max_abs": delta_identity,
            "attention_temp_max_abs": delta_attn,
            "mlp_temp_max_abs": delta_mlp,
            "manual_gate_scale_max_abs": formula_delta,
            "backend": "eager_tiny_random",
            "flash_equivalence": "未运行",
        }
    )


if __name__ == "__main__":
    main()
