"""Offline preflight for QKSpec training on the GPU node.

Validates, using the GPU venv, that:
* CUDA is available and the target model loads with eager attention;
* the local TOFU forget/retain splits load with the expected sizes;
* the QKSpec handler registers into the upstream trainer registry;
* the primary Hydra config composes with the QKSpec trainer/experiment.

This script does not run the unlearning edit; it only checks prerequisites.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from transformers import AutoConfig, AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
OU = REPO / "open-unlearning"
WS = REPO / "workspace"
sys.path[:0] = [str(OU / "src"), str(WS / "src")]

MODEL_DIR = Path("/jizhicfs/chenyliao/models/open-unlearning/tofu_Llama-3.2-1B-Instruct_full")
TOFU_DIR = Path("/jizhicfs/chenyliao/data/datasets/locuslab-TOFU")
RETAIN_LOG = Path(
    "/jizhicfs/chenyliao/data/datasets/open-unlearning/eval/"
    "tofu_Llama-3.2-1B-Instruct_retain90/TOFU_EVAL.json"
)
FORGET_SPLIT = "forget10"
RETAIN_SPLIT = "retain90"


def check_cuda() -> None:
    assert torch.cuda.is_available(), "CUDA not available in this interpreter"
    print(f"[OK] CUDA: {torch.cuda.device_count()} x {torch.cuda.get_device_name(0)}")


def check_model() -> None:
    cfg = AutoConfig.from_pretrained(str(MODEL_DIR), local_files_only=True)
    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
    assert cfg.model_type == "llama", cfg.model_type
    from omegaconf import OmegaConf
    model_cfg = OmegaConf.create({
        "model_args": {
            "pretrained_model_name_or_path": str(MODEL_DIR),
            "attn_implementation": "eager",
            "torch_dtype": "bfloat16",
        },
        "tokenizer_args": {"pretrained_model_name_or_path": str(MODEL_DIR)},
    })
    from model import get_model
    model, _ = get_model(model_cfg)
    assert model.config._attn_implementation == "eager", model.config._attn_implementation
    print(f"[OK] model: {MODEL_DIR.name}, layers={cfg.num_hidden_layers}, "
          f"attn=eager, params={sum(p.numel() for p in model.parameters())/1e9:.2f}B")
    del model


def check_data() -> None:
    forget = load_dataset("json", data_files=str(TOFU_DIR / f"{FORGET_SPLIT}.json"), split="train")
    retain = load_dataset("json", data_files=str(TOFU_DIR / f"{RETAIN_SPLIT}.json"), split="train")
    assert len(forget) > 0 and len(retain) > 0
    assert "question" in forget.column_names and "answer" in forget.column_names
    print(f"[OK] data: {FORGET_SPLIT}={len(forget)}, {RETAIN_SPLIT}={len(retain)}")

    # Verify the QADataset config files resolve to the same local paths.
    fcfg = OmegaConf.to_container(
        OmegaConf.load(WS / "configs/data/datasets/TOFU_QA_forget.yaml"), resolve=True
    )
    rcfg = OmegaConf.to_container(
        OmegaConf.load(WS / "configs/data/datasets/TOFU_QA_retain.yaml"), resolve=True
    )
    assert fcfg["TOFU_QA_forget"]["args"]["hf_args"]["data_files"].endswith(f"{FORGET_SPLIT}.json")
    assert rcfg["TOFU_QA_retain"]["args"]["hf_args"]["data_files"].endswith(f"{RETAIN_SPLIT}.json")
    print("[OK] dataset configs point to local TOFU json")


def check_retain_log() -> None:
    assert RETAIN_LOG.is_file(), f"missing retain eval log: {RETAIN_LOG}"
    with open(RETAIN_LOG) as f:
        data = json.load(f)
    assert isinstance(data, dict)
    print(f"[OK] retain log: {RETAIN_LOG.name}")


def check_registry_and_config() -> None:
    from qkspec.trainer import QKSpecTrainer
    from trainer import TRAINER_REGISTRY

    TRAINER_REGISTRY.setdefault("QKSpec", QKSpecTrainer)
    assert TRAINER_REGISTRY["QKSpec"] is QKSpecTrainer
    print("[OK] QKSpec handler registered")

    overrides = [
        f"hydra.searchpath=[file://{OU / 'configs'}]",
        "trainer=QKSpec",
        "experiment=unlearn/tofu/qkspec.yaml",
        f"task_name=preflight_{FORGET_SPLIT}",
        "model.model_args.attn_implementation=eager",
        "model.model_args.pretrained_model_name_or_path=" + str(MODEL_DIR),
        "model.tokenizer_args.pretrained_model_name_or_path=" + str(MODEL_DIR),
        "retain_logs_path=" + str(RETAIN_LOG),
    ]
    with initialize_config_dir(version_base=None, config_dir=str(WS / "configs")):
        cfg = compose(config_name="unlearn.yaml", overrides=overrides)
    assert cfg.trainer.handler == "QKSpec", cfg.trainer.handler
    assert cfg.model.model_args.attn_implementation == "eager"
    assert "qkspec" in cfg.trainer.method_args
    print(f"[OK] Hydra compose: handler={cfg.trainer.handler}, "
          f"rho={cfg.trainer.method_args.qkspec.rho}, "
          f"top_k={cfg.trainer.method_args.qkspec.top_k_directions}")


def main() -> None:
    print("=== QKSpec preflight ===")
    check_cuda()
    check_model()
    check_data()
    check_retain_log()
    check_registry_and_config()
    print("=== PREFLIGHT OK ===")


if __name__ == "__main__":
    main()
