"""M0b: did ASU make derivation-query attention blurrier than the base?

Run only after M0 returns proceed and an ASU checkpoint exists. A confidence
interval on the entropy increase that covers zero stops the anchor.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "workspace"))
sys.path.insert(0, str(REPO / "Right-to-be-forgotten"))

from da_asu.anchor import pad_query_index, rows_at_layer  # noqa: E402
from da_asu.kill_switch import bootstrap_mean_ci, cohens_d  # noqa: E402
from da_asu.labels import encode_labeled  # noqa: E402
from da_asu.protocol import MIN_PAIRED_PROBLEMS  # noqa: E402
from da_asu.run_m0 import LLAMA3_CONFIGS, load_records  # noqa: E402
from my_models.my_llama import LlamaForCausalLM  # noqa: E402

CLASS_NAMES = ("derivation", "function", "factual")
D_Z_DEFINITION = "mean(gap) / sample_std(gap, ddof=1)"


def row_stats(probs, query_index):
    valid = query_index >= 0
    peak = probs.max(dim=-1).values
    entropy = -(probs.clamp_min(1e-8) * probs.clamp_min(1e-8).log()).sum(dim=-1)
    mask = valid[:, None, :].to(probs.dtype)
    denom = mask.sum() * probs.shape[1]
    if float(denom) == 0.0:
        return None
    return {
        "entropy": float((entropy * mask).sum().item() / float(denom)),
        "peak": float((peak * mask).sum().item() / float(denom)),
    }


def h2_decision(entropy_gaps, min_problems=MIN_PAIRED_PROBLEMS):
    import numpy as np

    gaps = np.asarray(entropy_gaps, dtype=np.float64)
    ci = bootstrap_mean_ci(gaps)
    effect = cohens_d(gaps)
    covers_zero = ci is not None and ci[0] <= 0.0 <= ci[1]
    if gaps.size < min_problems:
        decision = "insufficient_data"
    elif covers_zero or float(gaps.mean()) <= 0.0:
        decision = "stop_anchor"
    else:
        decision = "proceed"
    return {
        "decision": decision,
        "n_problems": int(gaps.size),
        "mean_entropy_increase": None if gaps.size == 0 else float(gaps.mean()),
        "cohens_d": effect,
        "ci95": ci,
        "entropy_used": True,
    }


def _entropy_for_mask(decoder, hidden, position_embeddings, attention_mask, query_mask):
    if int(query_mask.sum()) == 0:
        return None
    query_index = pad_query_index(query_mask.unsqueeze(0))
    stats = []
    for layer_id, layer in enumerate(decoder.layers):
        probs = rows_at_layer(
            layer,
            hidden[layer_id],
            position_embeddings,
            query_index,
            attention_mask.unsqueeze(0),
        )
        summary = row_stats(probs, query_index)
        if summary is not None:
            stats.append(summary["entropy"])
    if not stats:
        return None
    return float(sum(stats) / len(stats))


@torch.no_grad()
def problem_class_entropies(model, input_ids, attention_mask, masks):
    """One decoder pass, then one entropy per query class. Values are nats."""
    decoder = model.get_decoder()
    outputs = decoder(
        input_ids=input_ids.unsqueeze(0),
        attention_mask=attention_mask.unsqueeze(0),
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    hidden = outputs.hidden_states
    position_ids = torch.arange(input_ids.shape[0], device=input_ids.device).unsqueeze(0)
    position_embeddings = decoder.rotary_emb(hidden[0], position_ids)
    return {
        name: _entropy_for_mask(decoder, hidden, position_embeddings, attention_mask, mask)
        for name, mask in masks.items()
    }


@torch.no_grad()
def problem_entropy(model, input_ids, attention_mask, query_mask):
    return problem_class_entropies(
        model, input_ids, attention_mask, {"derivation": query_mask}
    )["derivation"]


def class_gap_report(base_values, student_values):
    """Paired entropy report. Relative increase is the per-problem gap over base entropy."""
    import numpy as np

    base = np.asarray(base_values, dtype=np.float64)
    student = np.asarray(student_values, dtype=np.float64)
    gaps = student - base
    usable = base > 0.0
    relative = gaps[usable] / base[usable]
    return {
        "n_problems": int(gaps.size),
        "mean_base_entropy": None if gaps.size == 0 else float(base.mean()),
        "mean_student_entropy": None if gaps.size == 0 else float(student.mean()),
        "mean_entropy_increase": None if gaps.size == 0 else float(gaps.mean()),
        "entropy_increase_ci95": bootstrap_mean_ci(gaps),
        "mean_relative_increase": None if relative.size == 0 else float(relative.mean()),
        "relative_increase_ci95": bootstrap_mean_ci(relative) if relative.size else None,
        "n_relative": int(relative.size),
        "cohens_d_z": cohens_d(gaps),
        "cohens_d_z_definition": D_Z_DEFINITION,
        "entropy_unit": "nats",
    }


def specificity_report(derivation_gaps, function_gaps, min_problems=MIN_PAIRED_PROBLEMS):
    """Same-problem derivation gap minus function gap. Does not change the H2 decision."""
    import numpy as np

    gaps = np.asarray(derivation_gaps, dtype=np.float64) - np.asarray(function_gaps, dtype=np.float64)
    ci = bootstrap_mean_ci(gaps)
    covers_zero = ci is not None and ci[0] <= 0.0 <= ci[1]
    if gaps.size < min_problems:
        label = "insufficient_data"
    elif covers_zero or float(gaps.mean()) <= 0.0:
        label = "not_selective"
    else:
        label = "selective"
    return {
        "contrast": "derivation_minus_function",
        "role": "report_only",
        "n_problems": int(gaps.size),
        "mean_gap": None if gaps.size == 0 else float(gaps.mean()),
        "ci95": ci,
        "cohens_d_z": cohens_d(gaps),
        "cohens_d_z_definition": D_Z_DEFINITION,
        "entropy_unit": "nats",
        "label": label,
    }


def build_specificity_verdict(rows, min_problems=MIN_PAIRED_PROBLEMS):
    """``rows`` is a list of ``{class: {base, student}}`` for classes present on that problem."""
    by_class = {name: [] for name in CLASS_NAMES}
    paired = []
    for row in rows:
        for name in CLASS_NAMES:
            if name in row and row[name]["base"] is not None and row[name]["student"] is not None:
                by_class[name].append(row[name])
        if "derivation" in row and "function" in row:
            paired.append(
                (
                    row["derivation"]["student"] - row["derivation"]["base"],
                    row["function"]["student"] - row["function"]["base"],
                )
            )
    classes = {
        name: class_gap_report(
            [item["base"] for item in values],
            [item["student"] for item in values],
        )
        for name, values in by_class.items()
    }
    derivation_gaps = [
        item["student"] - item["base"] for item in by_class["derivation"]
    ]
    decision = h2_decision(derivation_gaps, min_problems=min_problems)
    return {
        "decision": decision["decision"],
        "decision_rule": "derivation_entropy_increase_only",
        "entropy_unit": "nats",
        "cohens_d_z_definition": D_Z_DEFINITION,
        "classes": classes,
        "specificity": specificity_report(
            [item[0] for item in paired],
            [item[1] for item in paired],
            min_problems=min_problems,
        ),
        "entropy_used": True,
    }


def main():
    parser = argparse.ArgumentParser(description="DA-ASU M0b attention blur check")
    parser.add_argument("--base", required=True)
    parser.add_argument("--student", required=True)
    parser.add_argument("--records", required=True)
    parser.add_argument("--m0-verdict", default=str(REPO / "workspace/results/direction_c/m0_verdict.json"))
    parser.add_argument("--output", default=str(REPO / "workspace/results/direction_c/m0b_specificity.json"))
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    m0 = json.loads(Path(args.m0_verdict).read_text())
    if m0.get("decision") != "proceed":
        raise SystemExit(f"M0 decision is {m0.get('decision')}; M0b is not run")

    records = load_records(args.records)
    if args.limit is not None:
        records = records[: args.limit]
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = LlamaForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16, attn_implementation="eager")
    student = LlamaForCausalLM.from_pretrained(args.student, torch_dtype=torch.bfloat16, attn_implementation="eager")
    base.eval().to(device)
    student.eval().to(device)
    rows = []
    for index, record in enumerate(records):
        if index % 50 == 0:
            print(f"m0b {index}/{len(records)}", flush=True)
        input_ids, _, attention_mask, classes = encode_labeled(
            tokenizer,
            record["question"],
            record["answer"],
            LLAMA3_CONFIGS,
            args.max_length,
            factual_spans=record.get("factual_spans"),
            function_spans=record.get("function_spans"),
        )
        if not any(int(classes[name].sum()) > 0 for name in CLASS_NAMES):
            continue
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        class_masks = {name: classes[name].to(device) for name in CLASS_NAMES}
        base_entropy = problem_class_entropies(base, input_ids, attention_mask, class_masks)
        student_entropy = problem_class_entropies(student, input_ids, attention_mask, class_masks)
        row = {}
        for name in CLASS_NAMES:
            if base_entropy[name] is None or student_entropy[name] is None:
                continue
            row[name] = {"base": base_entropy[name], "student": student_entropy[name]}
        if row:
            rows.append(row)
    verdict = build_specificity_verdict(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(verdict, indent=2))
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
