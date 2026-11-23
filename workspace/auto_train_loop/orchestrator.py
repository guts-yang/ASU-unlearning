"""Autonomous loop state machine: start -> train+eval -> decide -> record -> next.

No human interaction. AUTO_PROCEED=true. Resumable from state.json.
"""

from __future__ import annotations

import argparse
import json
import re
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
ROUNDS = LOOP / "rounds"
IDEAS = LOOP / "ideas"
LOGS = LOOP / "logs"
for d in (ROUNDS, IDEAS, LOGS):
    d.mkdir(exist_ok=True)


def git(*args, check=True):
    r = subprocess.run(
        ["git", "-c", "user.name=auto-train-loop", "-c", "user.email=auto-train-loop@local", *args],
        cwd=LOOP, capture_output=True, text=True,
    )
    if check and r.returncode != 0:
        raise RuntimeError(f"git {args}: {r.stderr.strip()}")
    return r.stdout.strip()


def short():
    return git("rev-parse", "--short", "HEAD")


def default_state(run_id):
    return {
        "run_id": run_id, "next_round": 1, "phase": "idle", "seed": P.SEED,
        "best_idea": None, "best_val": None, "last_keep": None,
        "seen_ideas": [], "history": [],
        "budget": {"gpu_seconds": 0.0, "rounds": 0},
        "consecutive_crash": 0, "started_at": datetime.now().isoformat(),
        "terminated": None,
    }


def init_repo():
    (LOOP / ".gitignore").write_text(
        "results.tsv\nrun.log\ncurrent_idea.json\nrounds/\nideas/\nlogs/\n__pycache__/\n",
        encoding="utf-8",
    )
    if not (LOOP / ".git").exists():
        git("init", "-q")
    git("add", "DESIGN.md", "program.md", "prepare.py", "idea_engine.py",
        "train.py", "orchestrator.py", ".gitignore")
    git("commit", "-q", "-m", "baseline: autonomous train-eval-idea loop scaffold", check=False)


def write_progress(state):
    lines = [
        "# PROGRESS — 自主训练闭环",
        "",
        f"**run_id**: `{state['run_id']}`  **next_round**: {state['next_round']}  **phase**: `{state['phase']}`",
        f"**best**: {state['best_val'] if state['best_val'] is not None else 'NA'} ({state['best_idea'] or 'none'})",
        f"**llm_backend**: `{idea_engine.llm_backend_status()['status']}` — 思路由确定性规则引擎生成，非 LLM",
        f"**终止门控（冻结）**: rho<={P.RHO_TARGET}, dF>={P.DF_MIN}, dR<={P.DR_MAX}, |dF-{P.DF_BAND:.4f}|<={P.DF_TOL}, 3-seed std<={P.SEED_STD_MAX}, refusal<={P.REFUSAL_MAX}",
        f"**generation_gate**: NOT_IMPLEMENTED → `TARGET_MET` 结构性不可达，预算耗尽须如实报告未达标",
        "",
        "| round | lever | idea | val_bpb | rho | dF | dR | status |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for h in state["history"]:
        lines.append(
            f"| {h['round']} | {h['lever']} | `{h['idea_id']}` | {h['val_bpb']:.6f} | "
            f"{h['rho']:.4f} | {h['dF']:+.4f} | {h['dR']:+.4f} | {h['status']} |"
        )
    if state["history"]:
        last = state["history"][-1]
        _, checks = P.is_good({"rho": last["rho"], "dF": last["dF"], "dR": last["dR"], "seed_confirmed": False})
        lines += ["", "## 与目标的距离（最后一轮）", ""]
        for k, v in checks.items():
            label = "PASS" if v else ("NOT_IMPLEMENTED" if k == "generation_gate" else "FAIL")
            lines.append(f"- {k}: {label}")
        lines.append(f"- rho 差距: {last['rho'] - P.RHO_TARGET:+.4f}（需 <= {P.RHO_TARGET}）")
    lines += [
        "",
        "## 已证伪 / 硬禁止（反重复记忆）",
        "",
        ", ".join(sorted(idea_engine.FORBIDDEN_LEVERS)),
        "",
        "## 预算",
        "",
        f"- rounds {state['budget']['rounds']}/{P.MAX_ROUNDS}, gpu {state['budget']['gpu_seconds']:.0f}/{P.MAX_GPU_SECONDS}s, consecutive_crash {state['consecutive_crash']}",
        "",
    ]
    PROGRESS.write_text("\n".join(lines), encoding="utf-8")


def write_summary(state, reason):
    lines = [
        "# AUTORESEARCH_SUMMARY — 自主训练闭环",
        "",
        f"**run_id**: `{state['run_id']}`  **终止原因**: `{reason}`",
        f"**best val_bpb**: {state['best_val'] if state['best_val'] is not None else 'NA'}  **best idea**: `{state['best_idea'] or 'none'}`",
        f"**轮数**: {state['budget']['rounds']}  **GPU 秒**: {state['budget']['gpu_seconds']:.0f}",
        "",
        "## 逐轮结果",
        "",
        "| round | idea | lever | val_bpb | rho | dF | dR | status |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for h in state["history"]:
        lines.append(f"| {h['round']} | `{h['idea_id']}` | {h['lever']} | {h['val_bpb']:.6f} | {h['rho']:.4f} | {h['dF']:+.4f} | {h['dR']:+.4f} | {h['status']} |")
    lines += [
        "",
        "## 终止判据核对",
        "",
        f"- `TARGET_MET`: {reason == 'TARGET_MET'}",
        f"- `BUDGET_EXHAUSTED`: {reason == 'BUDGET_EXHAUSTED'}",
        "- `generation_gate` 未实现、`llm_backend` = REVIEW_UNAVAILABLE，因此本轮不可能 `TARGET_MET`；",
        "  预算耗尽必须报告 **未达标**，不得放宽容差或伪称成功。",
        "",
        "## 思路生成与应用",
        "",
        "- `idea_engine.py` 的确定性诊断规则 + 有限 `LEVERS` 白名单 + `seen_ideas` 反重复。",
        "- 每轮 `ideas/R{n}.json` → `current_idea.json` → `train.py` 解释执行 → `rounds/R{n}.json` 回写。",
        "- `keep` 的 idea 成为后续基线锚；`discard` 保留在账本作为负结果证据。",
        "",
        "## 适配声明",
        "",
        "- 未使用 `val_bpb` 原义 / 300 秒墙钟 / `uv run` / 主仓库分支 / `git reset --hard`。",
        "- 单 seed、60 steps、1B/TOFU、P3 仅一种攻击强度；不得外推。",
        "- `results.tsv` 保持未跟踪；`state.json` 原子写、可 `--resume`。",
        "",
    ]
    SUMMARY.write_text("\n".join(lines), encoding="utf-8")


def start_round(state):
    """S1-S6. Returns (ok, reason)."""
    issues = P.preflight()
    if issues:
        return False, f"preflight: {issues}"
    if state["phase"] not in ("idle", "done"):
        return False, f"phase={state['phase']} (unfinished prior round)"
    if state["next_round"] > 1:
        prev = ROUNDS / f"R{state['next_round'] - 1}.json"
        if not prev.exists():
            return False, "prior round artifact missing"
        ok, msg = P.verify_round(P.load_json(prev))
        if not ok:
            return False, f"prior evidence gate failed: {msg}"
    if state["budget"]["rounds"] >= P.MAX_ROUNDS or state["budget"]["gpu_seconds"] >= P.MAX_GPU_SECONDS:
        return False, "budget exhausted"
    if state["history"]:
        last = state["history"][-1]
        good, _ = P.is_good({"rho": last["rho"], "dF": last["dF"], "dR": last["dR"],
                            "seed_confirmed": False})
        if good:
            return False, "target met"
    return True, "ok"


def do_round(state):
    n = state["next_round"]
    spec = idea_engine.next_idea(state)
    if spec is None:
        return "SPACE_EXHAUSTED"
    if spec["hash"] in state["seen_ideas"]:
        return "DUPLICATE_IDEA"
    P.dump_json(spec, IDEAS / f"R{n}.json")
    P.dump_json(spec, IDEA)
    state["seen_ideas"].append(spec["hash"])
    state["phase"] = "training"
    P.dump_json(state, STATE)

    git("add", "state.json", "PROGRESS.md", "prepare.py", "idea_engine.py", "train.py",
        "orchestrator.py", "DESIGN.md", "program.md")
    git("commit", "-q", "-m", f"R{n}: {spec['lever']} {spec['source']}", check=False)
    commit = short()

    log = LOGS / f"R{n}.log"
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
    dd = re.search(r"dF:\s*([-\d.]+)\s+dR:\s*([-\d.]+)", txt)

    if p.returncode != 0 or m is None or rr is None:
        status = "crash"
        val = 0.0
        print(f"[R{n}] CRASH\n" + "\n".join(txt.splitlines()[-25:]), flush=True)
        state["consecutive_crash"] += 1
        state["phase"] = "idle"
        state["next_round"] = n  # retry same round once
        result = {"round": n, "idea_id": spec["id"], "lever": spec["lever"],
                  "val_bpb": 0.0, "rho": float("nan"), "dF": float("nan"),
                  "dR": float("nan"), "status": status, "dt": dt}
    else:
        out = P.load_json(ROUNDS / f"R{n}.json")
        ok, msg = P.verify_round(out)
        if not ok:
            status = "crash"
            val = 0.0
            state["consecutive_crash"] += 1
            print(f"[R{n}] EVIDENCE GATE FAILED: {msg}", flush=True)
        else:
            val = float(m.group(1))
            peak = float(mm.group(1)) if mm else 0.0
            rho = float(rr.group(1))
            dF = float(dd.group(1)) if dd else out["delta"]["forget"]
            dR = float(dd.group(2)) if dd else out["delta"]["retain"]
            best = state["best_val"]
            if best is None or val < best:
                status = "keep"
                state["best_val"] = val
                state["best_idea"] = spec["id"]
                state["last_keep"] = commit
            else:
                status = "discard"
            state["consecutive_crash"] = 0
            result = {"round": n, "idea_id": spec["id"], "lever": spec["lever"],
                      "val_bpb": val, "rho": rho, "dF": dF, "dR": dR,
                      "status": status, "dt": dt, "peak_gb": round(peak / 1024, 1)}
            print(f"[R{n}] {status:7s} val_bpb={val:.6f} rho={rho:.4f} dF={dF:+.4f} dR={dR:+.4f} {dt:.1f}s | {spec['lever']} {spec['params']}", flush=True)
        state["phase"] = "done"
        state["next_round"] = n + 1

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
        run_id = args.resume or "auto-train-20261123"
        state = default_state(run_id)
        P.dump_json(state, STATE)
    else:
        state = P.load_json(STATE)
        if state.get("terminated"):
            print(f"run already terminated: {state['terminated']}", flush=True)
            return
        state["phase"] = "idle"  # resume any interrupted round cleanly
    write_progress(state)

    reason = None
    while True:
        ok, why = start_round(state)
        if not ok:
            reason = "TARGET_MET" if why == "target met" else (
                "BUDGET_EXHAUSTED" if "budget" in why else "BLOCKED")
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
            print(f"checkpoint: rounds={state['budget']['rounds']} best={state['best_val']:.6f}", flush=True)
            write_progress(state)

    state["terminated"] = reason
    state["phase"] = "terminated"
    P.dump_json(state, STATE)
    write_progress(state)
    write_summary(state, reason)
    print(f"\nLOOP STOP: {reason} | rounds={state['budget']['rounds']} best={state['best_val']}", flush=True)


if __name__ == "__main__":
    main()
