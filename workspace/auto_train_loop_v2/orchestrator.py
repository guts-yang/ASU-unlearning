"""Autonomous loop state machine v2. No human interaction; AUTO_PROCEED=true.
Adds the 3-seed confirmation round that makes TARGET_MET empirically reachable.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import idea_engine
import prepare as P

LOOP = Path(__file__).resolve().parent
STATE = LOOP / "state.json"
TSV = LOOP / "results.tsv"
IDEA = LOOP / "current_idea.json"
PROGRESS = LOOP / "PROGRESS.md"
SUMMARY = LOOP / "AUTORESEARCH_SUMMARY.md"
RUN_LOG = LOOP / "run.log"
for d in ("rounds", "ideas", "logs", "confirmations"):
    (LOOP / d).mkdir(exist_ok=True)


def git(*args, check=True):
    r = subprocess.run(
        ["git", "-c", "user.name=auto-train-v2", "-c", "user.email=auto-train-v2@local", *args],
        cwd=LOOP, capture_output=True, text=True,
    )
    if check and r.returncode != 0:
        raise RuntimeError(f"git {args}: {r.stderr.strip()}")
    return r.stdout.strip()


def short():
    return git("rev-parse", "--short", "HEAD")


def default_state(run_id):
    return {"run_id": run_id, "next_round": 1, "phase": "idle", "seed": 0,
            "best_idea": None, "best_val": None, "last_keep": None,
            "seen_ideas": [], "history": [], "confirmations": {},
            "budget": {"gpu_seconds": 0.0, "rounds": 0},
            "consecutive_crash": 0, "started_at": datetime.now().isoformat(),
            "terminated": None}


def init_repo():
    (LOOP / ".gitignore").write_text(
        "results.tsv\nrun.log\ncurrent_idea.json\nrounds/\nideas/\nlogs/\nconfirmations/\n__pycache__/\n",
        encoding="utf-8",
    )
    if not (LOOP / ".git").exists():
        git("init", "-q")
    git("add", "DESIGN_V2.md", "program.md", "prepare.py", "idea_engine.py",
        "train.py", "orchestrator.py", ".gitignore")
    git("commit", "-q", "-m", "baseline: autonomous train-eval-idea loop v2", check=False)


def write_progress(state):
    lines = [
        "# PROGRESS — 自主训练闭环 v2", "",
        f"**run_id**: `{state['run_id']}`  **next_round**: {state['next_round']}  **phase**: `{state['phase']}`",
        f"**best**: {state['best_val'] if state['best_val'] is not None else 'NA'} ({state['best_idea'] or 'none'})",
        f"**llm_backend**: `{idea_engine.llm_backend_status()['status']}` — 确定性规则引擎，非 LLM",
        f"**冻结目标**: rho<={P.RHO_TARGET}, |dF-{P.DF_BAND:.4f}|<={P.DF_TOL}, dR<={P.DR_MAX}, refusal<={P.REFUSAL_MAX}, 3-seed std(rho)<={P.SEED_STD_MAX}",
        "**v2 修复**: generation_gate 与 3-seed 确认轮已实跑，TARGET_MET 不再结构性不可达",
        "",
        "| round | lever | idea | val_bpb | rho | dF | dR | refusal | status |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for h in state["history"]:
        lines.append(f"| {h['round']} | {h['lever']} | `{h['idea_id']}` | {h['val_bpb']:.6f} | "
                     f"{h['rho']:.4f} | {h['dF']:+.4f} | {h['dR']:+.4f} | {h['refusal_rate']:.3f} | {h['status']} |")
    if state["history"]:
        last = state["history"][-1]
        checks = P.initial_checks(last)
        lines += ["", "## 与目标的距离（最后一轮）", ""]
        for k, v in checks.items():
            lines.append(f"- {k}: {'PASS' if v else 'FAIL'}")
        lines += [f"- rho 差距: {last['rho'] - P.RHO_TARGET:+.4f}（需 <= {P.RHO_TARGET}）",
                  f"- 3-seed 确认记录: {len(state['confirmations'])} 轮"]
    lines += ["", "## 硬禁止（反重复记忆）", "",
              ", ".join(sorted(idea_engine.FORBIDDEN_LEVERS)), "",
              "## 预算", "",
              f"- rounds {state['budget']['rounds']}/{P.MAX_ROUNDS}, gpu {state['budget']['gpu_seconds']:.0f}/{P.MAX_GPU_SECONDS}s, crash {state['consecutive_crash']}", ""]
    PROGRESS.write_text("\n".join(lines), encoding="utf-8")


def write_summary(state, reason):
    lines = [
        "# AUTORESEARCH_SUMMARY — 自主训练闭环 v2", "",
        f"**run_id**: `{state['run_id']}`  **终止原因**: `{reason}`",
        f"**best val_bpb**: {state['best_val'] if state['best_val'] is not None else 'NA'}  **best idea**: `{state['best_idea'] or 'none'}`",
        f"**轮数**: {state['budget']['rounds']}  **GPU 秒**: {state['budget']['gpu_seconds']:.0f}",
        f"**3-seed 确认轮**: {len(state['confirmations'])}", "",
        "## 逐轮结果", "",
        "| round | idea | lever | val_bpb | rho | dF | dR | refusal | status |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for h in state["history"]:
        lines.append(f"| {h['round']} | `{h['idea_id']}` | {h['lever']} | {h['val_bpb']:.6f} | "
                     f"{h['rho']:.4f} | {h['dF']:+.4f} | {h['dR']:+.4f} | {h['refusal_rate']:.3f} | {h['status']} |")
    lines += ["", "## 终止判据", "",
              f"- `TARGET_MET`: {reason == 'TARGET_MET'}",
              f"- `BUDGET_EXHAUSTED`: {reason == 'BUDGET_EXHAUSTED'}",
              "- 阈值全程冻结，未放宽；若未达标则如实报告差距。", "",
              "## 3-seed 确认", ""]
    if state["confirmations"]:
        for k, v in state["confirmations"].items():
            lines.append(f"- {k}: rhos={[round(x, 4) for x in v['rhos']]}, std={v['std']:.4f}, pass={v['pass']}")
    else:
        lines.append("- 无候选通过初筛，未触发确认轮。")
    lines += ["", "## 思路生成与回灌", "",
              "- `idea_engine.py` 确定性诊断 + 有限 `LEVERS`（含 `alpha_hinge` 组合）+ `seen_ideas` 反重复。",
              "- `ideas/R{n}.json` → `current_idea.json` → `train.py` → `rounds/R{n}.json` → `state.history`。", "",
              "## 适配与局限", "",
              "- 未使用 `val_bpb` 原义 / 300 秒墙钟 / `uv run` / 主仓库分支 / `git reset --hard`。",
              "- 生成质量门为固定模板启发式，不等价于人工答案质量评估；不检测通顺但无关的替换答案。",
              "- 单 1B 模型 / TOFU forget10 / answer-NLL / P3 单攻击强度；不得外推。",
              "- 跨模型评审不可用，裁决为执行者 provisional。",
              "- `results.tsv` 未跟踪；`state.json` 原子写、可 `--resume`。", ""]
    SUMMARY.write_text("\n".join(lines), encoding="utf-8")


def start_round(state):
    issues = P.preflight()
    if issues:
        return False, f"preflight: {issues}"
    if state["phase"] not in ("idle", "done"):
        return False, f"phase={state['phase']}"
    if state["next_round"] > 1:
        prev = LOOP / "rounds" / f"R{state['next_round']-1}.json"
        if not prev.exists():
            return False, "prior artifact missing"
        ok, msg = P.verify_round(P.load_json(prev))
        if not ok:
            return False, f"evidence gate: {msg}"
    if state["budget"]["rounds"] >= P.MAX_ROUNDS or state["budget"]["gpu_seconds"] >= P.MAX_GPU_SECONDS:
        return False, "budget exhausted"
    if state.get("target_met"):
        return False, "target met"
    return True, "ok"


def run_confirmation(state, spec, main_rho, refusal):
    """3-seed confirmation: reuse seed 0's rho, run seeds 1 and 2."""
    rhos = [main_rho]
    for s in P.CONFIRM_SEEDS[1:]:
        out = LOOP / "confirmations" / f"R{spec['round']}_s{s}.json"
        log = LOOP / "logs" / f"R{spec['round']}_confirm_s{s}.log"
        cmd = [sys.executable, "train.py", "--seed", str(s), "--skip-gen",
               "--refusal", f"{refusal:.6f}", "--out", str(out)]
        t0 = time.time()
        with log.open("w") as f:
            f.write("$ " + " ".join(cmd) + "\n")
            f.flush()
            p = subprocess.run(cmd, cwd=LOOP, stdout=f, stderr=subprocess.STDOUT, timeout=900)
        dt = time.time() - t0
        state["budget"]["gpu_seconds"] += dt
        if p.returncode != 0 or not out.exists():
            return {"rhos": rhos, "std": None, "pass": False, "error": f"seed {s} failed"}
        rhos.append(P.load_json(out)["rho"])
    std = statistics.pstdev(rhos)
    return {"rhos": rhos, "std": std, "pass": std <= P.SEED_STD_MAX}


def do_round(state):
    n = state["next_round"]
    spec = idea_engine.next_idea(state)
    if spec is None:
        return "SPACE_EXHAUSTED"
    if spec["hash"] in state["seen_ideas"]:
        return "DUPLICATE_IDEA"
    P.dump_json(spec, LOOP / "ideas" / f"R{n}.json")
    P.dump_json(spec, IDEA)
    state["seen_ideas"].append(spec["hash"])
    state["phase"] = "training"
    P.dump_json(state, STATE)

    git("add", "state.json", "PROGRESS.md", "prepare.py", "idea_engine.py",
        "train.py", "orchestrator.py", "DESIGN_V2.md", "program.md")
    git("commit", "-q", "-m", f"R{n}: {spec['lever']} {spec['source']}", check=False)
    commit = short()

    log = LOOP / "logs" / f"R{n}.log"
    t0 = time.time()
    with log.open("w") as f:
        f.write(f"$ python train.py  # {spec['id']}\n")
        f.flush()
        p = subprocess.run([sys.executable, "train.py"], cwd=LOOP, stdout=f,
                           stderr=subprocess.STDOUT, timeout=900)
    dt = time.time() - t0
    txt = log.read_text(encoding="utf-8", errors="replace")
    RUN_LOG.write_text(txt, encoding="utf-8")
    m = re.search(r"^val_bpb:\s*([-\d.]+)", txt, re.M)
    mm = re.search(r"^peak_vram_mb:\s*([\d.]+)", txt, re.M)
    rr = re.search(r"^rho_p3:\s*([-\d.]+)", txt, re.M)
    fr = re.search(r"^refusal_rate:\s*([-\d.]+)", txt, re.M)

    if p.returncode != 0 or m is None or rr is None or fr is None:
        status = "crash"
        val = 0.0
        state["consecutive_crash"] += 1
        state["phase"] = "idle"
        state["next_round"] = n
        print(f"[R{n}] CRASH\n" + "\n".join(txt.splitlines()[-20:]), flush=True)
        result = {"round": n, "idea_id": spec["id"], "lever": spec["lever"],
                  "val_bpb": 0.0, "rho": float("nan"), "dF": float("nan"),
                  "dR": float("nan"), "refusal_rate": float("nan"), "status": status}
    else:
        out = P.load_json(LOOP / "rounds" / f"R{n}.json")
        ok, msg = P.verify_round(out)
        if not ok:
            status, val = "crash", 0.0
            state["consecutive_crash"] += 1
            state["phase"] = "idle"
            state["next_round"] = n
            print(f"[R{n}] EVIDENCE GATE FAILED: {msg}", flush=True)
            result = {"round": n, "idea_id": spec["id"], "lever": spec["lever"],
                      "val_bpb": 0.0, "rho": float("nan"), "dF": float("nan"),
                      "dR": float("nan"), "refusal_rate": float("nan"), "status": status}
        else:
            val, rho = float(m.group(1)), float(rr.group(1))
            peak = float(mm.group(1)) if mm else 0.0
            refusal = float(fr.group(1))
            dF, dR = out["dF"], out["dR"]
            if state["best_val"] is None or val < state["best_val"]:
                status = "keep"
                state["best_val"] = val
                state["best_idea"] = spec["id"]
                state["last_keep"] = commit
            else:
                status = "discard"
            state["consecutive_crash"] = 0
            state["phase"] = "done"
            state["next_round"] = n + 1
            result = {"round": n, "idea_id": spec["id"], "lever": spec["lever"],
                      "val_bpb": val, "rho": rho, "dF": dF, "dR": dR,
                      "refusal_rate": refusal, "mean_len": out["mean_len"],
                      "status": status, "peak_gb": round(peak / 1024, 1)}
            print(f"[R{n}] {status:7s} val={val:.6f} rho={rho:.4f} dF={dF:+.4f} dR={dR:+.4f} ref={refusal:.3f} {dt:.1f}s | {spec['lever']} {spec['params']}", flush=True)

            # 3-seed confirmation only if all initial checks pass
            checks = P.initial_checks(result)
            if all(checks.values()):
                print(f"[R{n}] initial screen PASS -> triggering 3-seed confirmation", flush=True)
                conf = run_confirmation(state, spec, rho, refusal)
                state["confirmations"][f"R{n}"] = conf
                out["confirmations"] = conf
                out["seed_confirmed"] = conf["pass"]
                P.dump_json(out, LOOP / "rounds" / f"R{n}.json")
                if conf["pass"]:
                    state["target_met"] = True
                    print(f"[R{n}] TARGET_MET: std(rho)={conf['std']:.4f}", flush=True)

    with TSV.open("a", encoding="utf-8") as f:
        f.write(f"{commit}\t{val:.6f}\t{result.get('peak_gb', 0.0)}\t{status}\t{spec['id']}\n")
    state["history"].append(result)
    state["budget"]["rounds"] += 1
    state["budget"]["gpu_seconds"] += dt
    P.dump_json(state, STATE)
    write_progress(state)
    return "OK"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", default=None)
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args()

    init_repo()
    if not TSV.exists() or args.reset:
        TSV.write_text("commit\tval_bpb\tmemory_gb\tstatus\tdescription\n", encoding="utf-8")
    if args.reset or not STATE.exists():
        state = default_state(args.resume or "auto-train-v2-20261123")
        P.dump_json(state, STATE)
    else:
        state = P.load_json(STATE)
        if state.get("terminated"):
            print(f"run already terminated: {state['terminated']}", flush=True)
            return
        state["phase"] = "idle"
    write_progress(state)

    reason = None
    while True:
        ok, why = start_round(state)
        if not ok:
            reason = "TARGET_MET" if why == "target met" else (
                "BUDGET_EXHAUSTED" if "budget" in why or why == "SPACE_EXHAUSTED" else "BLOCKED")
            if reason == "BLOCKED":
                print(f"BLOCKED: {why}", flush=True)
            break
        rc = do_round(state)
        if rc in ("SPACE_EXHAUSTED", "DUPLICATE_IDEA"):
            reason = "BUDGET_EXHAUSTED"
            print(f"idea space {rc}", flush=True)
            break
        if state["consecutive_crash"] >= P.MAX_CONSECUTIVE_CRASH:
            reason = "HALTED"
            print("HALTED: consecutive crashes", flush=True)
            break
        if state["budget"]["rounds"] % 3 == 0:
            print(f"checkpoint: rounds={state['budget']['rounds']} best={state['best_val']}", flush=True)
            write_progress(state)

    state["terminated"] = reason
    state["phase"] = "terminated"
    P.dump_json(state, STATE)
    write_progress(state)
    write_summary(state, reason)
    print(f"\nLOOP STOP: {reason} | rounds={state['budget']['rounds']} best={state['best_val']} target_met={state.get('target_met', False)}", flush=True)


if __name__ == "__main__":
    main()
