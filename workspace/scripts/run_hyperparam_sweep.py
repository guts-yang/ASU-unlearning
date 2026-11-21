"""Automatic QKSpec hyperparameter sweep: train + FE/MU eval for each config.

Runs entirely unattended. For each hyperparameter grid point it:
  1. runs QKSpec training (workspace/src/run_qkspec.py) with do_eval=false,
  2. runs FE/MU evaluation (eval_fe_mu_tf32.py) on the saved student,
  3. records wall-clock time, FE, MU, and kernel metrics,
  4. writes a running summary CSV/JSON after each point.

Grid (6 points) targets the three decisive knobs: rho, top_k, max_retain_impact.
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import time
from pathlib import Path

REPO = Path("/usr/local/workspace/ASU-unlearning-K4")
OU = REPO / "open-unlearning"
WS = REPO / "workspace"
ENV_SH = WS / "scripts" / "env_qkspec.sh"
OUT_ROOT = Path("/jizhicfs/chenyliao/runs/qkspec/sweep_v2_spec")
SUMMARY = OUT_ROOT / "sweep_summary.json"
SUMMARY_CSV = OUT_ROOT / "sweep_summary.csv"

MODEL_TAG = "Llama-3.2-1B-Instruct"
MODEL_DIR = f"/jizhicfs/chenyliao/models/open-unlearning/tofu_{MODEL_TAG}_full"
TOFU_DIR = "/jizhicfs/chenyliao/data/datasets/locuslab-TOFU"
RETAIN_LOG = f"/jizhicfs/chenyliao/data/datasets/open-unlearning/eval/tofu_{MODEL_TAG}_retain90/TOFU_EVAL.json"

# (name, rho, top_k, ranks, retain_cap, stage_b_min, shortlist, spec_weight)
# Round 2: causal-specificity filtering replaces the coarse retain_cap.
GRID = [
    ("spec_w0_k16",     0.0, 16, 4, "null", 0.0,   4, 0.0),   # baseline: raw B effect
    ("spec_w1_lo_k32",  0.0, 32, 8, "null", 0.002, 8, 1.0),   # specificity, low threshold
    ("spec_w1_md_k32",  0.0, 32, 8, "null", 0.005, 8, 1.0),   # specificity, mid threshold
    ("spec_w1_hi_k32",  0.0, 32, 8, "null", 0.01,  8, 1.0),   # specificity, high threshold
    ("spec_w2_md_k32",  0.0, 32, 8, "null", 0.005, 8, 2.0),   # stronger retain penalty
    ("spec_w1_rho2_k32",0.2, 32, 8, "null", 0.005, 8, 1.0),   # milder shrink + specificity
]


def env():
    e = os.environ.copy()
    e.update({
        "PYTHONPATH": f"{OU}/src:{WS}/src:{e.get('PYTHONPATH','')}",
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_OFFLINE": "0",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "0",
        "HF_HUB_ENABLE_HF_TRANSFER": "0",
        "CUDA_HOME": "/jizhicfs/chenyliao/ops/cuda-stub",
        "DS_SKIP_CUDA_CHECK": "1",
        "CUDA_VISIBLE_DEVICES": "0",
    })
    return e


def run(cmd, log_path, tag):
    print(f"  [{tag}] running: {' '.join(cmd[:4])} ...", flush=True)
    t0 = time.time()
    with open(log_path, "w") as lf:
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env())
    dt = time.time() - t0
    if proc.returncode != 0:
        print(f"  [{tag}] WARN exit={proc.returncode}, see {log_path}", flush=True)
    return dt, proc.returncode


def read_fe_mu(student: Path):
    p = student / "eval_fe_mu" / "TOFU_SUMMARY.json"
    if not p.is_file():
        return None, None
    d = json.load(open(p))
    return d.get("forget_quality"), d.get("model_utility")


def read_kernel(student: Path):
    p = student / "qkspec_report.json"
    if not p.is_file():
        return {}
    r = json.load(open(p))
    e = r.get("edit", {})
    fm = r.get("forget_metrics", {})
    rm = r.get("retain_metrics", {})
    return {
        "n_dirs": len(e.get("selected", [])),
        "update_norm": e.get("update_norm"),
        "kernel_forget_nll": fm.get("nll"),
        "kernel_retain_nll": rm.get("nll"),
        "kernel_forget_acc": fm.get("token_accuracy"),
        "kernel_retain_acc": rm.get("token_accuracy"),
    }


def save_summary(rows):
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY, "w") as f:
        json.dump(rows, f, indent=2)
    if rows:
        keys = list(rows[0].keys())
        with open(SUMMARY_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    py = "/jizhicfs/chenyliao/ops/venvs/asu-apru/bin/python"
    rows = []
    total_t0 = time.time()

    for name, rho, topk, ranks, cap, stage_b_min, shortlist, spec_w in GRID:
        tag = f"qkspec_{name}"
        student = OUT_ROOT / tag
        print(f"\n=== GRID: {name} (rho={rho}, top_k={topk}, spec_w={spec_w}, stage_b_min={stage_b_min}) ===", flush=True)

        # ---- train ----
        train_cmd = [
            py, str(WS / "src" / "run_qkspec.py"), "--config-name=unlearn.yaml",
            "experiment=unlearn/tofu/qkspec.yaml", "trainer=QKSpec",
            f"task_name={tag}", "forget_split=forget10", "holdout_split=holdout10",
            "retain_split=retain90",
            f"model.model_args.pretrained_model_name_or_path={MODEL_DIR}",
            f"model.tokenizer_args.pretrained_model_name_or_path={MODEL_DIR}",
            "model.model_args.attn_implementation=eager",
            "data.forget.TOFU_QA_forget.args.hf_args.path=json",
            "data.forget.TOFU_QA_forget.args.hf_args.name=null",
            f"++data.forget.TOFU_QA_forget.args.hf_args.data_files={TOFU_DIR}/forget10.json",
            "data.retain.TOFU_QA_retain.args.hf_args.path=json",
            "data.retain.TOFU_QA_retain.args.hf_args.name=null",
            f"++data.retain.TOFU_QA_retain.args.hf_args.data_files={TOFU_DIR}/retain90.json",
            f"trainer.method_args.qkspec.rho={rho}",
            f"trainer.method_args.qkspec.top_k_directions={topk}",
            f"trainer.method_args.qkspec.ranks_per_head={ranks}",
            f"trainer.method_args.qkspec.shortlist_per_layer={shortlist}",
            f"trainer.method_args.qkspec.stage_b_min_nll_increase={stage_b_min}",
            f"trainer.method_args.qkspec.specificity_weight={spec_w}",
            f"trainer.method_args.qkspec.max_retain_impact={cap}" if cap != "null"
                else "trainer.method_args.qkspec.max_retain_impact=null",
            "trainer.method_args.qkspec.max_discovery_batches=8",
            "trainer.method_args.qkspec.max_stage_b_batches=8",
            "trainer.method_args.qkspec.run_controls=false",
            "trainer.args.do_eval=false",
            "trainer.args.seed=1001",
            f"retain_logs_path={RETAIN_LOG}",
            f"paths.output_dir={student}",
        ]
        t_train, rc_train = run(train_cmd, OUT_ROOT / f"{name}_train.log", name)

        # ---- FE/MU eval ----
        fe = mu = None
        t_eval = 0.0
        rc_eval = -1
        if student.is_dir() and (student / "model.safetensors").is_file():
            eval_cmd = [
                py, str(WS / "scripts" / "eval_fe_mu_tf32.py"),
                "hydra.searchpath=[file://" + str(OU / "configs") + "]",
                f"model.model_args.pretrained_model_name_or_path={student}",
                f"model.tokenizer_args.pretrained_model_name_or_path={student}",
                "eval.tofu.forget_split=forget10", "eval.tofu.holdout_split=holdout10",
                f"eval.tofu.retain_logs_path={RETAIN_LOG}",
                f"eval.tofu.output_dir={student}/eval_fe_mu",
                "eval.tofu.batch_size=16",
                f"paths.output_dir={student}/eval_fe_mu",
            ]
            t_eval, rc_eval = run(eval_cmd, OUT_ROOT / f"{name}_eval.log", f"{name}/eval")
            fe, mu = read_fe_mu(student)

        kernel = read_kernel(student)
        row = {
            "name": name, "rho": rho, "top_k": topk, "ranks": ranks,
            "retain_cap": cap, "stage_b_min": stage_b_min, "shortlist": shortlist,
            "spec_weight": spec_w,
            "train_s": round(t_train, 1), "eval_s": round(t_eval, 1),
            "total_s": round(t_train + t_eval, 1),
            "FE_forget_quality": fe, "MU_model_utility": mu,
            "rc_train": rc_train, "rc_eval": rc_eval,
            **kernel,
        }
        rows.append(row)
        save_summary(rows)
        print(f"  [{name}] done: train={t_train:.0f}s eval={t_eval:.0f}s "
              f"FE={fe} MU={mu} n_dirs={kernel.get('n_dirs')}", flush=True)

    print(f"\n=== SWEEP DONE in {time.time()-total_t0:.0f}s ===", flush=True)
    print(f"Summary: {SUMMARY}", flush=True)
    for r in rows:
        print(f"  {r['name']:18} FE={r['FE_forget_quality']} MU={r['MU_model_utility']} "
              f"({r['total_s']}s)", flush=True)


if __name__ == "__main__":
    main()
