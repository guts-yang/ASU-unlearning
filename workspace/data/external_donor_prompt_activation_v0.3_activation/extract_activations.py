#!/usr/bin/env python3
"""Extract prompt-only residual activations for the v0.3 donor pairs.

Loads the frozen TOFU Llama-2-7B once on cuda:0. Each question is encoded
with the manual Llama-2 chat splice and forwarded twice. Answers are not
appended. No scores, patches, or training.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

if not str(os.environ.get("OMP_NUM_THREADS", "")).isdigit():
    os.environ["OMP_NUM_THREADS"] = "1"

import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer

REPO = Path("/usr/local/ASU-unlearning")
sys.path.insert(0, str(REPO / "Right-to-be-forgotten"))
from my_models.my_llama import LlamaForCausalLM  # noqa: E402

CANDIDATE_DIR = REPO / "workspace/data/external_donor_prompt_activation_v0.3"
OUT_DIR = REPO / "workspace/data/external_donor_prompt_activation_v0.3_activation"
SNAPSHOT = Path(
    "/root/autodl-tmp/huggingface/hub/"
    "models--open-unlearning--tofu_Llama-2-7b-chat-hf_full/"
    "snapshots/cc5b31c69127d5da881608e640f2b453c446435b"
)
QUESTION_START = "[INST] "
QUESTION_END = " [/INST]"
DEVICE = torch.device("cuda:0")
EXPECTED_LAYERS = 33
EXPECTED_HIDDEN = 4096


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def prompt_text(question: str) -> str:
    return QUESTION_START + question + QUESTION_END


def entity_span(tokenizer, prompt: str, question: str, entity: str) -> dict:
    encoding = tokenizer(prompt, add_special_tokens=True, return_offsets_mapping=True)
    ids = list(encoding["input_ids"])
    offsets = [tuple(pair) for pair in encoding["offset_mapping"]]
    char_start = question.find(entity)
    prompt_start = len(QUESTION_START) + char_start
    prompt_end = prompt_start + len(entity)
    indices = [
        index
        for index, (start, end) in enumerate(offsets)
        if not (start == 0 and end == 0) and end > prompt_start and start < prompt_end
    ]
    return {
        "entity": entity,
        "token_ids": ids,
        "token_span": {
            "start": indices[0] if indices else None,
            "end": (indices[-1] + 1) if indices else None,
        },
        "token_indices": indices,
        "prompt_length": len(ids),
    }


def forward_hidden(model, token_ids: list[int]) -> torch.Tensor:
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=DEVICE)
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
    hidden = torch.stack([state[0].detach() for state in outputs.hidden_states], dim=0)
    return hidden.to("cpu")


def max_abs_diff(left: torch.Tensor, right: torch.Tensor) -> float:
    return (left.float() - right.float()).abs().max().item()


def finite_stats(tensor: torch.Tensor) -> dict:
    values = tensor.float()
    finite = torch.isfinite(values)
    return {
        "finite": bool(finite.all().item()),
        "numel": int(tensor.numel()),
        "min": None if not finite.any() else float(values[finite].min().item()),
        "max": None if not finite.any() else float(values[finite].max().item()),
    }


def side_record(name: str, hidden: torch.Tensor, repeat: torch.Tensor, span: dict) -> dict:
    last = hidden[:, -1, :].contiguous()
    last_repeat = repeat[:, -1, :].contiguous()
    return {
        "name": name,
        "prompt_length": span["prompt_length"],
        "last_token_index": span["prompt_length"] - 1,
        "entity_token_span": span["token_span"],
        "entity_token_indices": span["token_indices"],
        "hidden_shape": list(hidden.shape),
        "vector_shape": list(last.shape),
        "dtype": str(hidden.dtype).replace("torch.", ""),
        "finite": finite_stats(hidden)["finite"] and finite_stats(last)["finite"],
        "vector_min": finite_stats(last)["min"],
        "vector_max": finite_stats(last)["max"],
        "max_abs_diff": max_abs_diff(last, last_repeat),
        "hidden_max_abs_diff": max_abs_diff(hidden, repeat),
        "hidden": hidden,
        "vector": last,
    }


def build_summary(context: dict) -> str:
    lines = [
        "# v0.3 prompt-only 激活提取",
        "",
        "## 结论",
        "",
        f"**{'通过' if context['passed'] else '未通过'}。**",
        "",
        (
            f"{context['pair_count']} 条配对都在 `cuda:0` 上完成了两次 prompt-only 前向。"
            if context["passed"]
            else "至少一条配对没有得到形状正确的有限激活。"
        ),
        "输入只有问题 prompt，没有拼入 `y_plus` 或 `donor_fact_value`。没有打分、patch 或训练。",
        "",
        "## 运行",
        "",
        f"- 模型类：`{context['model_class']}`",
        f"- snapshot：`{context['snapshot']}`",
        f"- 设备：`{context['device_name']}`（{context['device']}）",
        f"- dtype：`{context['dtype']}`",
        f"- 注意力：`sdpa`",
        f"- hidden states：{context['num_hidden_states']} 个（embedding 加 32 层），宽度 {EXPECTED_HIDDEN}",
        f"- 主向量位置：每条 prompt 的最后一个 token",
        f"- 生成时间：`{context['generated_at']}`",
        "",
        "## 逐条结果",
        "",
        "| pair | q_plus 长度 | q_minus 长度 | a_plus | a_minus | 有限值 | 最后 token 最大绝对差 | 全序列最大绝对差 |",
        "| --- | ---: | ---: | --- | --- | --- | ---: | ---: |",
    ]
    for row in context["rows"]:
        plus = row["q_plus"]
        minus = row["q_minus"]
        lines.append(
            f"| {row['pair_id']} | {plus['prompt_length']} | {minus['prompt_length']} | "
            f"`{plus['vector_shape']}` | `{minus['vector_shape']}` | "
            f"{'是' if plus['finite'] and minus['finite'] else '否'} | "
            f"{max(plus['max_abs_diff'], minus['max_abs_diff']):.6g} | "
            f"{max(plus['hidden_max_abs_diff'], minus['hidden_max_abs_diff']):.6g} |"
        )
    lines.extend(
        [
            "",
            f"最后 token 向量的最大绝对差：`{context['max_vector_diff']:.6g}`。",
            f"全序列 hidden states 的最大绝对差：`{context['max_hidden_diff']:.6g}`。",
            "",
            "实体 token span 记在对应 safetensors 的 metadata 里，全序列张量可按这个 span 重切。",
            "",
            "## 文件",
            "",
        ]
    )
    for name in context["files"]:
        lines.append(f"- `{name}`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("cuda is not available")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pairs = read_jsonl(CANDIDATE_DIR / "candidate_pairs.jsonl")
    if len(pairs) != 10:
        raise RuntimeError(f"expected 10 pairs, found {len(pairs)}")

    tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True)
    model = LlamaForCausalLM.from_pretrained(
        str(SNAPSHOT),
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model.to(DEVICE)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    rows = []
    saved_files = []
    for pair in pairs:
        q_plus = pair["q_plus"]
        q_minus = pair["q_minus"]
        if pair["y_plus"] in prompt_text(q_plus) or pair["donor_fact_value"] in prompt_text(q_minus):
            raise RuntimeError(f"{pair['pair_id']} appends a fact value to the prompt")
        plus_span = entity_span(tokenizer, prompt_text(q_plus), q_plus, pair["clean_entity"])
        minus_span = entity_span(tokenizer, prompt_text(q_minus), q_minus, pair["donor_entity"])
        plus_first = forward_hidden(model, plus_span["token_ids"])
        plus_second = forward_hidden(model, plus_span["token_ids"])
        minus_first = forward_hidden(model, minus_span["token_ids"])
        minus_second = forward_hidden(model, minus_span["token_ids"])
        plus = side_record("q_plus", plus_first, plus_second, plus_span)
        minus = side_record("q_minus", minus_first, minus_second, minus_span)
        shape_ok = (
            plus["vector_shape"] == [EXPECTED_LAYERS, EXPECTED_HIDDEN]
            and minus["vector_shape"] == [EXPECTED_LAYERS, EXPECTED_HIDDEN]
            and plus["hidden_shape"][0] == EXPECTED_LAYERS
            and minus["hidden_shape"][0] == EXPECTED_LAYERS
            and plus["hidden_shape"][1] == plus["prompt_length"]
            and minus["hidden_shape"][1] == minus["prompt_length"]
            and plus["hidden_shape"][2] == EXPECTED_HIDDEN
            and minus["hidden_shape"][2] == EXPECTED_HIDDEN
        )
        if not shape_ok or not plus["finite"] or not minus["finite"]:
            raise RuntimeError(f"{pair['pair_id']} activation shape or finite check failed")
        filename = f"{pair['pair_id']}.safetensors"
        save_file(
            {
                "a_plus": plus["vector"],
                "a_minus": minus["vector"],
                "hidden_q_plus": plus["hidden"],
                "hidden_q_minus": minus["hidden"],
            },
            str(OUT_DIR / filename),
            metadata={
                "pair_id": pair["pair_id"],
                "q_plus_prompt": prompt_text(q_plus),
                "q_minus_prompt": prompt_text(q_minus),
                "q_plus_last_token_index": str(plus["last_token_index"]),
                "q_minus_last_token_index": str(minus["last_token_index"]),
                "q_plus_entity_token_span": json.dumps(plus["entity_token_span"]),
                "q_minus_entity_token_span": json.dumps(minus["entity_token_span"]),
                "dtype": plus["dtype"],
                "position": "last_prompt_token",
                "answer_appended": "false",
            },
        )
        saved_files.append(filename)
        rows.append(
            {
                "pair_id": pair["pair_id"],
                "q_plus": {key: value for key, value in plus.items() if key not in {"hidden", "vector"}},
                "q_minus": {key: value for key, value in minus.items() if key not in {"hidden", "vector"}},
            }
        )
        print(
            pair["pair_id"],
            "vector_diff",
            max(plus["max_abs_diff"], minus["max_abs_diff"]),
            "hidden_diff",
            max(plus["hidden_max_abs_diff"], minus["hidden_max_abs_diff"]),
            flush=True,
        )

    max_vector_diff = max(max(row["q_plus"]["max_abs_diff"], row["q_minus"]["max_abs_diff"]) for row in rows)
    max_hidden_diff = max(
        max(row["q_plus"]["hidden_max_abs_diff"], row["q_minus"]["hidden_max_abs_diff"]) for row in rows
    )
    context = {
        "passed": True,
        "pair_count": len(rows),
        "model_class": f"{LlamaForCausalLM.__module__}.LlamaForCausalLM",
        "snapshot": str(SNAPSHOT),
        "device": str(DEVICE),
        "device_name": torch.cuda.get_device_name(DEVICE),
        "dtype": "bfloat16",
        "num_hidden_states": EXPECTED_LAYERS,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rows": rows,
        "max_vector_diff": max_vector_diff,
        "max_hidden_diff": max_hidden_diff,
        "files": saved_files + ["summary.md"],
    }
    (OUT_DIR / "summary.md").write_text(build_summary(context), encoding="utf-8")
    print("passed", max_vector_diff, max_hidden_diff, flush=True)


if __name__ == "__main__":
    main()
