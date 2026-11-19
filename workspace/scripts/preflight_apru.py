"""Offline preflight for ASU × LeakRU training on this node."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from hydra import compose, initialize_config_dir
from transformers import AutoConfig, AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
OU = REPO / "open-unlearning"
WS = REPO / "workspace"
sys.path[:0] = [str(OU / "src"), str(WS / "src")]

from open_unlearning_plugins.register import register_workspace_trainers  # noqa: E402

MODEL_ID = "open-unlearning/tofu_Llama-2-7b-chat-hf_full"
MODEL_DIR = Path("/jizhicfs/chenyliao/models/open-unlearning/tofu_Llama-2-7b-chat-hf_full")
RETAIN_LOG = Path(
    "/jizhicfs/chenyliao/data/datasets/open-unlearning/eval/"
    "tofu_Llama-2-7b-chat-hf_retain95/TOFU_EVAL.json"
)
TOFU_DIR = Path("/jizhicfs/chenyliao/data/datasets/locuslab-TOFU")
LOCAL_DATA_OVERRIDES = [
    "data.forget.TOFU_QA_forget.args.hf_args.path=json",
    "data.forget.TOFU_QA_forget.args.hf_args.name=null",
    f"+data.forget.TOFU_QA_forget.args.hf_args.data_files={TOFU_DIR / 'forget05.json'}",
    "data.retain.TOFU_QA_retain.args.hf_args.path=json",
    "data.retain.TOFU_QA_retain.args.hf_args.name=null",
    f"+data.retain.TOFU_QA_retain.args.hf_args.data_files={TOFU_DIR / 'retain95.json'}",
]


def main() -> None:
    trainers = register_workspace_trainers()
    assert "ASU" in trainers, sorted(trainers)

    cfg = AutoConfig.from_pretrained(str(MODEL_DIR), local_files_only=True)
    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
    assert cfg.num_hidden_layers == 32, cfg.num_hidden_layers
    assert tok.vocab_size == 32000, tok.vocab_size

    forget = load_dataset("json", data_files=str(TOFU_DIR / "forget05.json"), split="train")
    retain = load_dataset("json", data_files=str(TOFU_DIR / "retain95.json"), split="train")
    assert len(forget) == 200, len(forget)
    assert len(retain) == 3800, len(retain)

    rows = [
        json.loads(line)
        for line in (WS / "data/leakru/tofu_forget05.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert len(rows) == 260, len(rows)
    assert sum(r["split"] == "single-hop" for r in rows) == 200
    assert sum(r["split"] == "multi-hop" for r in rows) == 60
    assert RETAIN_LOG.is_file(), RETAIN_LOG

    with initialize_config_dir(config_dir=str(OU / "configs"), version_base=None):
        exp = compose(
            config_name="unlearn.yaml",
            overrides=[
                f"hydra.searchpath=[file://{WS / 'configs'}]",
                "experiment=unlearn/asu_tofu05",
                *LOCAL_DATA_OVERRIDES,
            ],
        )
    assert exp.trainer.handler == "ASU", exp.trainer.handler
    assert float(exp.trainer.method_args.attention_temp) == 2.3
    assert float(exp.trainer.method_args.alpha) == 0.1
    assert int(exp.trainer.args.num_train_epochs) == 5
    assert float(exp.trainer.args.learning_rate) == 1e-5
    assert exp.forget_split == "forget05"
    assert exp.retain_split == "retain95"

    print("cuda_available:", torch.cuda.is_available())
    print("cuda_devices:", torch.cuda.device_count())
    print("model:", MODEL_DIR, "32 layers / vocab 32000")
    print("data: TOFU forget05=200 retain95=3800 LeakRU=260")
    print("config: experiment=unlearn/asu_tofu05 tau=2.3 alpha=0.1 5ep lr=1e-5")
    print("PREFLIGHT_OK")


if __name__ == "__main__":
    main()
