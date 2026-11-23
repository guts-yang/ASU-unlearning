#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
CONFIG_DIR = BASE / "configs" / "ideas"


def base_config() -> dict:
    return {
        "protocol": "factpair-progressive-alignment-v0.5-idea-suite",
        "gpu": "0",
        "model": "/jizhicfs/chenyliao/models/open-unlearning/tofu_Llama-3.2-1B-Instruct_full",
        "data_dir": "/usr/local/workspace/ASU-unlearning-I/workspace/data/factpair",
        "dtype": "bfloat16",
        "attn_implementation": "sdpa",
        "seed": 0,
        "max_steps": 200,
        "batch_size": 2,
        "learning_rate": 1e-5,
        "weight_decay": 0.0,
        "grad_clip": 1.0,
        "lambda_act": 1.0,
        "lambda_seq": 0.5,
        "beta_retain": 1.0,
        "eta": 0.5,
        "activation_point": "decoder_layer_output",
        "activation_norm": "l2",
        "activation_distance": "mse_on_l2_normalized",
        "schedule_type": "progressive",
        "initial_fraction": 0.25,
        "expansion_fraction": 0.25,
        "stage_steps": 10,
        "max_new_tokens": 64,
        "seq_refresh_steps": 20,
        "eval_samples": 50,
        "log_every": 10,
        "save_every": 200,
    }


def write(name: str, **overrides) -> None:
    config = base_config()
    config.update(overrides)
    config["output_dir"] = str(BASE / "runs" / "ideas" / name)
    path = CONFIG_DIR / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(path)


def main() -> None:
    write("00_baseline_direct", idea="baseline_direct", condition="baseline_direct")
    write(
        "01_tricon_A",
        idea="tricon",
        condition="tricon_A",
        tau_c=0.1,
        unknown_word="unknown",
    )
    write(
        "02_deltasubspace_B",
        idea="delta_subspace",
        condition="deltasubspace_B",
        alpha=0.5,
        svd_rank=4,
    )
    for index, schedule in enumerate(
        ("progressive", "static_full", "reverse_progressive", "random"), start=3
    ):
        write(
            f"{index:02d}_causaldelta_{schedule}_C",
            idea="causal_delta",
            condition=f"causaldelta_{schedule}_C",
            alpha=0.5,
            svd_rank=4,
            gate_refresh_steps=20,
            schedule_type=schedule,
        )
    write(
        "07_deltaguide_D",
        idea="delta_guide",
        condition="deltaguide_D",
        alpha=0.5,
        svd_rank=4,
        ortho_weight=1.0,
    )


if __name__ == "__main__":
    main()
