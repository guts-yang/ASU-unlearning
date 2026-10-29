"""Build the real-world factorial commands. This module never downloads weights."""

import json
import os
import sys
from pathlib import Path

_WORKSPACE = Path(__file__).resolve().parents[1]
if str(_WORKSPACE) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE))

from gold_rank.preflight import preflight_issues

REPO = Path(__file__).resolve().parents[2]
GROUPS = {
    "A": {"anchor": False, "mode": None},
    "B": {"anchor": True, "mode": None},
    "C": {"anchor": False, "mode": "demote"},
    "D": {"anchor": True, "mode": "demote"},
    "penalty": {"anchor": False, "mode": "matched_penalty"},
    "random": {"anchor": False, "mode": "random_position"},
    "shuffle_k": {"anchor": False, "mode": "shuffle_k"},
    "flatten": {"anchor": False, "mode": "topk_flatten"},
}


def save_dir(save_root, seed, group):
    spec = GROUPS[group]
    path = (
        f"{save_root}/llama3-8b/forget/ASU+KL/seed_{seed}/"
        "epoch5_5e-06_FixRefTrue_maskTrue_Fcoeff0.1_Rcoeff1.0_Temp2.0_LayersNone_beta0.1"
    )
    if spec["anchor"]:
        path += "_anchor"
    if spec["mode"]:
        path += f"_{spec['mode']}"
    return path


def command(model_path, data_path, save_root, seed, group, reason_path=None, lock_path=None, verdict_path=None):
    spec = GROUPS[group]
    if spec["mode"] and not lock_path:
        raise SystemExit(f"group {group} requires a frozen gold-rank lock")
    if spec["anchor"] and not reason_path:
        raise SystemExit(f"group {group} requires the local GSM8K reason JSONL")
    issues = preflight_issues(
        model_path,
        data_path,
        reason_path=reason_path,
        lock_path=lock_path,
        need_reason=spec["anchor"],
        need_lock=bool(spec["mode"]),
    )
    if issues:
        raise SystemExit("Refusing to download or train.\n" + "\n".join(issues))
    if spec["mode"]:
        lock = json.loads(Path(lock_path).read_text())
        if lock.get("status") != "frozen" or lock.get("test_metrics_read"):
            raise SystemExit(f"gold-rank lock is not a frozen pre-test lock: {lock_path}")
    verdict = verdict_path or str(REPO / "workspace" / "C" / "derived" / "m0_verdict.json")
    if spec["anchor"]:
        verdict_file = Path(verdict)
        if not verdict_file.is_file():
            raise SystemExit(f"M0 verdict missing: {verdict}")
        decision = json.loads(verdict_file.read_text()).get("decision")
        if decision != "proceed":
            raise SystemExit(f"M0 decision is {decision}. Anchor training stays off.")
    overrides = [
        "real_world_unlearn.py",
        "--config-name=real_world.yaml",
        "hydra.job.chdir=false",
        f"model_path={model_path}",
        f"data_path={data_path}",
        "forget_loss=ASU+KL",
        "attention_temp=2.0",
        "layers_id=null",
        "forget_coeff=0.1",
        "regularization_coeff=1.0",
        "num_epochs=5",
        "batch_size=4",
        "gradient_accumulation_steps=4",
        f"seed={seed}",
        "save_checkpoint=true",
        f"save_root={save_root}",
        f"save_dir={save_dir(save_root, seed, group)}",
        f"da_asu.enabled={'true' if spec['anchor'] else 'false'}",
        "da_asu.mode=anchor",
        "da_asu.mu=1.0",
        "da_asu.allow_without_m0=false",
        f"da_asu.verdict_path={verdict}",
        f"gold_rank.enabled={'true' if spec['mode'] else 'false'}",
        f"gold_rank.mode={spec['mode'] or 'demote'}",
        f"gold_rank.seed={seed}",
    ]
    if reason_path:
        overrides.append(f"da_asu.reason_path={reason_path}")
    if lock_path:
        overrides.append(f"gold_rank.lock_path={lock_path}")
    return overrides


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Launch one prepared ASU factorial group")
    parser.add_argument("--group", required=True, choices=sorted(GROUPS))
    parser.add_argument("--model", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--save-root", required=True)
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--reason-jsonl")
    parser.add_argument("--lock")
    parser.add_argument("--verdict")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    overrides = command(
        args.model,
        args.data_path,
        args.save_root,
        args.seed,
        args.group,
        args.reason_jsonl,
        args.lock,
        args.verdict,
    )
    launch = [
        "torchrun",
        "--nproc_per_node",
        os.environ.get("NPROC", "2"),
        "--master_port",
        os.environ.get("MASTER_PORT", "29571"),
        *overrides,
    ]
    if args.dry_run:
        print(" ".join(launch))
        return 0
    os.chdir(REPO / "Right-to-be-forgotten")
    os.execvp(launch[0], launch)


if __name__ == "__main__":
    sys.exit(main())
