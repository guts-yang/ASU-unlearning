#!/usr/bin/env python3
"""CPU-only smoke: ASU registration, attention scaling, data schema."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from transformers import LlamaConfig, LlamaForCausalLM

ASU_WS = Path("/usr/local/ASU-unlearning/workspace")
OU = Path("/usr/local/open-unlearning")
sys.path[:0] = [str(OU / "src"), str(OU / "workspace" / "src"), str(ASU_WS / "src")]

from open_unlearning_plugins.asu.attention import attention_temperature  # noqa: E402
from open_unlearning_plugins.register import (  # noqa: E402
    register_workspace_evaluators,
    register_workspace_trainers,
)


def main() -> None:
    trainers = register_workspace_trainers()
    evals = register_workspace_evaluators()
    assert "ASU" in trainers, trainers
    assert "leakru_fq" in evals, evals

    cfg = LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=32,
    )
    model = LlamaForCausalLM(cfg).eval()
    x = torch.randint(0, 128, (1, 8))
    with torch.no_grad():
        base = model(input_ids=x).logits
        with attention_temperature(model, tau=2.3):
            warm = model(input_ids=x).logits
    assert not torch.allclose(base, warm), "τ should change logits"
    with torch.no_grad():
        again = model(input_ids=x).logits
    assert torch.allclose(base, again), "scaling must restore"

    data = ASU_WS / "data/leakru/tofu_forget05.jsonl"
    rows = [json.loads(l) for l in data.read_text().splitlines() if l.strip()]
    assert len(rows) == 260
    assert sum(r["split"] == "single-hop" for r in rows) == 200
    assert sum(r["split"] == "multi-hop" for r in rows) == 60
    for logic in ("HS", "MP", "LL", "CC", "DS", "MT"):
        assert any(r["logic_type"] == logic and r["split"] == "multi-hop" for r in rows)
    print("smoke_offline OK")


if __name__ == "__main__":
    main()
