#!/usr/bin/env python3
"""Offline TOFU forget05 M0. Does not train, download, or edit checkpoints."""

import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from transformers.modeling_outputs import CausalLMOutputWithPast

ROOT = Path("/usr/local/ASU-unlearning")
REPO = ROOT / "Right-to-be-forgotten"
WORKSPACE = ROOT / "workspace"
sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(WORKSPACE / "I" / "commands"))
sys.path.insert(0, str(REPO))

from build_e0_manifest import author_label  # noqa: E402
from gold_rank.demotion import gold_ranks  # noqa: E402
from gold_rank.masking import encode_forget_example  # noqa: E402

INIT_MODEL = Path("/root/autodl-tmp/models/tofu_ft_llama2-7b")
ASU_MODEL = Path(
    "/root/autodl-tmp/ASU-unlearning/checkpoints/tofu_replay/llama2-7b/forget05/ASU+GD/"
    "seed_1001/epoch5_1e-05_FixRefTrue_maskTrue_Fcoeff0.1_Rcoeff1.0_Temp2.3_LayersNone_beta0.1/"
    "1/unlearn_times_1/checkpoint-last"
)
TOKENIZER_DIR = Path(
    "/root/autodl-tmp/huggingface/hub/models--NousResearch--Llama-2-7b-chat-hf/"
    "snapshots/351844e75ed0bcbbe3f10671b3c808d2b83894ee"
)
FORGET_PATH = REPO / "data/tofu/forget05.json"
MODEL_CONFIG = REPO / "config/model_config.yaml"
OUT_DIR = WORKSPACE / "J" / "derived"
VERDICT_PATH = WORKSPACE / "J" / "结论.md"

TASK_ID = "1"
MAX_LENGTH = 500
ATTENTION_TEMP = 2.3
BOOTSTRAP_SEED = 1001
BOOTSTRAP_SAMPLES = 1000
REL_TOL = 1e-4
QUESTIONS_PER_AUTHOR = 20


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def local_dir(path, label):
    path = Path(path)
    text = str(path)
    if "locuslab" in text or not path.is_dir() or not (path / "config.json").is_file():
        raise SystemExit(f"{label} is not a local snapshot: {path}")
    return path


def load_task_rows():
    rows = []
    with open(FORGET_PATH) as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row["task_id"]) == TASK_ID:
                rows.append(row)
    return rows


def assign_authors(rows):
    if len(rows) != 200:
        raise RuntimeError(f"task_id=1 has {len(rows)} rows, expected 200")
    labels = []
    meta = []
    for index in range(0, len(rows), QUESTIONS_PER_AUTHOR):
        block = rows[index:index + QUESTIONS_PER_AUTHOR]
        if len(block) != QUESTIONS_PER_AUTHOR:
            raise RuntimeError("author block is short")
        label, covered = author_label([row["question"] for row in block])
        meta.append({"author_index": index // QUESTIONS_PER_AUTHOR, "label": label, "coverage": covered})
        labels.extend([label] * QUESTIONS_PER_AUTHOR)
    if len(meta) != 10:
        raise RuntimeError(f"expected 10 authors, found {len(meta)}")
    if any(item["coverage"] < 15 for item in meta):
        raise RuntimeError(f"author coverage below 15: {meta}")
    names = [item["label"] for item in meta]
    if len(set(names)) != 10 or any(left == right for left, right in zip(names, names[1:])):
        raise RuntimeError(f"author labels are not 10 distinct blocks: {names}")
    return labels, meta


def bootstrap_mean_ci(values, seed=BOOTSTRAP_SEED, samples=BOOTSTRAP_SAMPLES):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[draws].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return {
        "mean": float(values.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "n_authors": int(len(values)),
        "seed": seed,
        "samples": samples,
    }


def decide(cluster_ok, grad_ok, asu_ci_low, teacher_top10_mean):
    if not cluster_ok or grad_ok is None or asu_ci_low is None or teacher_top10_mean is None:
        return "INCONCLUSIVE"
    if not grad_ok or asu_ci_low <= 0.05 or teacher_top10_mean < 0.20:
        return "NO-GO"
    return "GO"


class FrozenLogits(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.stored = logits

    def forward(self, input_ids=None, labels=None, attention_mask=None, **kwargs):
        zero = self.stored.reshape(-1)[0] * 0
        return CausalLMOutputWithPast(loss=zero, logits=self.stored)


def kl_api_error(student_logits, teacher_logits, input_ids, labels, attention_mask):
    """Compare asu_loss gold-logit autograd with (p_S-p_T)/(n_valid*batch)."""
    from trainer.losses import asu_loss

    student = student_logits.detach().float().requires_grad_(True)
    teacher = teacher_logits.detach().float()
    loss = asu_loss(
        FrozenLogits(student),
        FrozenLogits(teacher),
        [(input_ids, labels, attention_mask)],
        attention_temp=ATTENTION_TEMP,
        layers_id=None,
    )
    loss.backward()
    gold = labels[:, 1:]
    mask = gold != -100
    if int(mask.sum()) == 0:
        raise RuntimeError("API check batch has no answer tokens")
    positions = mask.nonzero(as_tuple=False)
    sliced_student = student[:, :-1, :]
    sliced_teacher = teacher[:, :-1, :]
    gathered_s = sliced_student[positions[:, 0], positions[:, 1]]
    gathered_t = sliced_teacher[positions[:, 0], positions[:, 1]]
    gold_ids = gold[positions[:, 0], positions[:, 1]]
    prob_s = torch.softmax(gathered_s, dim=-1).gather(1, gold_ids[:, None]).squeeze(1)
    prob_t = torch.softmax(gathered_t, dim=-1).gather(1, gold_ids[:, None]).squeeze(1)
    n_valid = mask.sum(dim=-1).float()
    batch = float(labels.shape[0])
    scale = n_valid[positions[:, 0]] * batch
    analytic = (prob_s - prob_t) / scale
    observed = student.grad[:, :-1, :][positions[:, 0], positions[:, 1], gold_ids]
    abs_err = (observed - analytic).abs()
    rel = abs_err / analytic.abs().clamp_min(1e-6)
    sign_ok = bool(torch.all((observed < 0) == (prob_s < prob_t)))
    return {
        "loss": float(loss.detach()),
        "n_positions": int(positions.shape[0]),
        "max_abs_error": float(abs_err.detach().max()),
        "max_rel_error": float(rel.detach().max()),
        "sign_ok": sign_ok,
        "ok": bool(sign_ok and float(rel.max()) < REL_TOL and float(abs_err.max()) < 1e-5),
    }


def question_stats(student_logits, teacher_logits, labels):
    records = []
    gold = labels[:, 1:]
    student = student_logits[:, :-1, :].float()
    teacher = teacher_logits[:, :-1, :].float()
    for batch_index in range(labels.shape[0]):
        mask = gold[batch_index] != -100
        positions = mask.nonzero(as_tuple=False).squeeze(-1)
        if positions.numel() == 0:
            records.append(None)
            continue
        s_rows = student[batch_index, positions]
        t_rows = teacher[batch_index, positions]
        gold_ids = gold[batch_index, positions]
        prob_s = torch.softmax(s_rows, dim=-1).gather(1, gold_ids[:, None]).squeeze(1)
        prob_t = torch.softmax(t_rows, dim=-1).gather(1, gold_ids[:, None]).squeeze(1)
        student_rank = gold_ranks(s_rows, gold_ids)
        teacher_rank = gold_ranks(t_rows, gold_ids)
        below = prob_s < prob_t
        records.append({
            "n_valid": int(positions.numel()),
            "frac_ps_lt_pt": float(below.float().mean()),
            "n_ps_lt_pt": int(below.sum()),
            "mean_ps": float(prob_s.mean()),
            "mean_pt": float(prob_t.mean()),
            "teacher_top1": float((teacher_rank <= 1).float().mean()),
            "teacher_top5": float((teacher_rank <= 5).float().mean()),
            "teacher_top10": float((teacher_rank <= 10).float().mean()),
            "student_top1": float((student_rank <= 1).float().mean()),
            "student_top5": float((student_rank <= 5).float().mean()),
            "student_top10": float((student_rank <= 10).float().mean()),
            "mean_negative_gold_grad": float((prob_s - prob_t < 0).float().mean()),
        })
    return records


def author_table(rows, key):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["author_index"], []).append(row[key])
    ordered = [float(np.mean(grouped[index])) for index in range(10)]
    return ordered


def summarize_student(rows):
    authors_below = author_table(rows, "frac_ps_lt_pt")
    authors_top10 = author_table(rows, "teacher_top10")
    return {
        "questions": len(rows),
        "tokens": int(sum(row["n_valid"] for row in rows)),
        "token_frac_ps_lt_pt": float(
            sum(row["n_ps_lt_pt"] for row in rows) / sum(row["n_valid"] for row in rows)
        ),
        "author_ps_lt_pt": bootstrap_mean_ci(authors_below),
        "author_teacher_top10": bootstrap_mean_ci(authors_top10),
        "author_teacher_top1_mean": float(np.mean(author_table(rows, "teacher_top1"))),
        "author_teacher_top5_mean": float(np.mean(author_table(rows, "teacher_top5"))),
        "author_student_top1_mean": float(np.mean(author_table(rows, "student_top1"))),
        "author_student_top5_mean": float(np.mean(author_table(rows, "student_top5"))),
        "author_student_top10_mean": float(np.mean(author_table(rows, "student_top10"))),
        "author_mean_ps": float(np.mean(author_table(rows, "mean_ps"))),
        "author_mean_pt": float(np.mean(author_table(rows, "mean_pt"))),
    }


def write_verdict(summary):
    decision = summary["decision"]
    asu = summary["asu"]["author_ps_lt_pt"]
    top10 = summary["asu"]["author_teacher_top10"]["mean"]
    grad = summary["kl_api"]
    lines = [
        "# 方向 J M0 结论",
        "",
        f"日期：2026-10-30。决策：**{decision}**。",
        "",
        "这是 `forget05` `task_id=1` 上、open-unlearning 基座为 Teacher（attention_temp=2.3）时的 teacher-forcing 描述统计。作者均值先在题内对答案 token 取平均，再对作者取平均。bootstrap 重抽 10 位作者，1000 次，种子 1001，区间是 2.5% 到 97.5% 分位。",
        "",
        "## ASU+GD Student",
        "",
        f"- 作者级 `p_S(g) < p_T(g)` 均值 {asu['mean']:.4f}，区间 [{asu['ci_low']:.4f}, {asu['ci_high']:.4f}]。",
        f"- Teacher gold Top-10 作者均值 {top10:.4f}。Top-1 / Top-5 作者均值为 {summary['asu']['author_teacher_top1_mean']:.4f} / {summary['asu']['author_teacher_top5_mean']:.4f}。",
        f"- Student gold Top-1 / Top-5 / Top-10 作者均值为 {summary['asu']['author_student_top1_mean']:.4f} / {summary['asu']['author_student_top5_mean']:.4f} / {summary['asu']['author_student_top10_mean']:.4f}。",
        f"- KL autograd 最大相对误差 {grad['max_rel_error']:.3e}，最大绝对误差 {grad['max_abs_error']:.3e}，符号一致：{grad['sign_ok']}。",
        "",
        "## 初始化 Student",
        "",
        f"- 作者级 `p_S(g) < p_T(g)` 均值 {summary['init']['author_ps_lt_pt']['mean']:.4f}，区间 [{summary['init']['author_ps_lt_pt']['ci_low']:.4f}, {summary['init']['author_ps_lt_pt']['ci_high']:.4f}]。",
        "",
        "## 含义",
        "",
    ]
    if decision == "GO":
        lines.append("门控通过：值得做一次单侧 gold 上限的最小训练。这次诊断没有证明遗忘硬下限，也没有证明 DARD 成立。")
    elif decision == "NO-GO":
        lines.append("门控未通过，按规划停止，不启动单侧上限训练。")
    else:
        lines.append("资产、mask 或作者切块不足以做作者级判断，按规划停止。")
    lines.append("")
    VERDICT_PATH.write_text("\n".join(lines))


def _freeze(model):
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def load_teacher(path):
    from my_models.my_llama import LlamaForCausalLM

    model = LlamaForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        local_files_only=True,
    ).cuda()
    return _freeze(model)


def load_student(path):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        local_files_only=True,
    ).cuda()
    return _freeze(model)


def forward_pair(teacher, student, batch):
    shared = {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
        "use_cache": False,
    }
    with torch.no_grad():
        teacher_out = teacher(**shared, attention_temp=ATTENTION_TEMP, layers_id=None)
        student_out = student(**shared)
    return student_out.logits, teacher_out.logits


def run_student(teacher, student, encoded, name, api_on_first=False):
    rows = []
    api = None
    batch_size = 2
    for start in range(0, len(encoded), batch_size):
        chunk = encoded[start:start + batch_size]
        batch = {
            "input_ids": torch.stack([item["input_ids"] for item in chunk]).cuda(),
            "attention_mask": torch.stack([item["attention_mask"] for item in chunk]).cuda(),
            "labels": torch.stack([item["labels"] for item in chunk]).cuda(),
        }
        student_logits, teacher_logits = forward_pair(teacher, student, batch)
        if api_on_first and api is None:
            api = kl_api_error(
                student_logits[:1],
                teacher_logits[:1],
                batch["input_ids"][:1],
                batch["labels"][:1],
                batch["attention_mask"][:1],
            )
            print("kl_api", json.dumps(api), flush=True)
        stats = question_stats(student_logits, teacher_logits, batch["labels"])
        for offset, stat in enumerate(stats):
            if stat is None:
                raise RuntimeError(f"{name} question {start + offset} has no answer tokens")
            source = chunk[offset]
            stat.update({
                "student": name,
                "question_index": source["question_index"],
                "author_index": source["author_index"],
                "author": source["author"],
            })
            rows.append(stat)
        print(f"{name} {len(rows)}/{len(encoded)}", flush=True)
        del student_logits, teacher_logits, batch
    return rows, api


def main():
    init_path = local_dir(INIT_MODEL, "init")
    asu_path = local_dir(ASU_MODEL, "asu")
    tokenizer_path = local_dir(TOKENIZER_DIR, "tokenizer")
    rows = load_task_rows()
    try:
        labels, author_meta = assign_authors(rows)
        cluster_ok = True
        cluster_error = None
    except RuntimeError as exc:
        cluster_ok = False
        cluster_error = str(exc)
        labels, author_meta = [], []
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not cluster_ok:
        summary = {
            "decision": "INCONCLUSIVE",
            "reason": cluster_error,
        }
        (OUT_DIR / "m0_summary.json").write_text(json.dumps(summary, indent=2))
        VERDICT_PATH.write_text(
            "# 方向 J M0 结论\n\n决策：**INCONCLUSIVE**。\n\n"
            f"作者切块未通过：{cluster_error}\n"
        )
        print(json.dumps(summary), flush=True)
        return

    from transformers import AutoTokenizer

    model_configs = yaml.safe_load(MODEL_CONFIG.read_text())["llama2-7b"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    encoded = []
    for index, row in enumerate(rows):
        input_ids, label, attention_mask = encode_forget_example(
            tokenizer, MAX_LENGTH, row["question"], row["answer"], model_configs, mask=True
        )
        encoded.append({
            "input_ids": input_ids,
            "labels": label,
            "attention_mask": attention_mask,
            "question_index": index,
            "author_index": index // QUESTIONS_PER_AUTHOR,
            "author": labels[index],
        })

    print("loading teacher and init student", flush=True)
    teacher = load_teacher(init_path)
    student = load_student(init_path)
    init_rows, _ = run_student(teacher, student, encoded, "init")
    del student
    torch.cuda.empty_cache()
    print("loading asu student", flush=True)
    student = load_student(asu_path)
    asu_rows, api = run_student(teacher, student, encoded, "asu", api_on_first=True)

    init_summary = summarize_student(init_rows)
    asu_summary = summarize_student(asu_rows)
    decision = decide(
        True,
        api["ok"] if api else None,
        asu_summary["author_ps_lt_pt"]["ci_low"],
        asu_summary["author_teacher_top10"]["mean"],
    )
    summary = {
        "decision": decision,
        "forget_sha256": sha256_file(FORGET_PATH),
        "init_model": str(init_path),
        "asu_model": str(asu_path),
        "attention_temp": ATTENTION_TEMP,
        "layers_id": None,
        "max_length": MAX_LENGTH,
        "task_id": TASK_ID,
        "authors": author_meta,
        "kl_api": api,
        "init": init_summary,
        "asu": asu_summary,
        "aggregation": "token mean within question, question mean within author, author bootstrap",
    }
    (OUT_DIR / "m0_summary.json").write_text(json.dumps(summary, indent=2))
    with open(OUT_DIR / "per_question.jsonl", "w") as handle:
        for row in init_rows + asu_rows:
            handle.write(json.dumps(row) + "\n")
    write_verdict(summary)
    print(json.dumps({"decision": decision, "asu_ci_low": asu_summary["author_ps_lt_pt"]["ci_low"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
