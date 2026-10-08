"""Run the base-model NLL-tau scan. Does not train."""

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "workspace"))
sys.path.insert(0, str(REPO / "Right-to-be-forgotten"))

from da_asu.protocol import TAU_GRID, normalize_layers_id  # noqa: E402
from da_asu.scan import scan_records  # noqa: E402
from my_models.my_llama import LlamaForCausalLM  # noqa: E402


LLAMA3_CONFIGS = {
    "question_start_tag": "<|start_header_id|>user<|end_header_id|>\n\n",
    "question_end_tag": "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    "answer_tag": "",
}


def _load_model(model_path):
    last_error = None
    for implementation in ("flash_attention_2", "sdpa", "eager"):
        try:
            return LlamaForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation=implementation,
            )
        except (ImportError, ValueError, OSError) as error:
            last_error = error
    raise RuntimeError(f"could not load {model_path}") from last_error


def load_records(path):
    records = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main():
    parser = argparse.ArgumentParser(description="DA-ASU M0 NLL-tau scan")
    parser.add_argument("--model", required=True)
    parser.add_argument("--records", required=True, help="JSONL with question, answer, optional spans")
    parser.add_argument("--output", default=str(REPO / "workspace/results/direction_c/m0_verdict.json"))
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--layers-id", default=None, help="null applies tau at every layer")
    args = parser.parse_args()

    records = load_records(args.records)
    if args.limit is not None:
        records = records[: args.limit]
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = _load_model(args.model)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    result = scan_records(
        model,
        tokenizer,
        records,
        LLAMA3_CONFIGS,
        args.max_length,
        taus=TAU_GRID,
        layers_id=normalize_layers_id(args.layers_id),
        batch_size=args.batch_size,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result["verdict"], indent=2))
    output.with_name("m0_curves.json").write_text(json.dumps(result["curves"]))
    print(json.dumps(result["verdict"], indent=2))


if __name__ == "__main__":
    main()
