"""Autonomous loop state machine v6 (RelResidual validation).

Every idea is replicated on 3 INDEPENDENT data orders; the reproduced metric
(mean dF, std dF, worst-case donor/premature/truth leakage) drives diagnosis,
keep/discard and termination. AUTO_PROCEED=true, no human interaction.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import idea_engine
import prepare as P

LOOP = Path(__file__).resolve().parent
ENV = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
STATE, TSV, IDEA = LOOP / "state.json", LOOP / "results.tsv", LOOP / "current_idea.json"
PROGRESS, SUMMARY, RUN_LOG = LOOP / "PROGRESS.md", LOOP / "AUTORESEARCH_SUMMARY.md", LOOP / "run.log"
for d in ("ideas", "logs", "runs", "replications", "targets"):
    (LOOP / d).mkdir(exist_ok=True)


def git(*args, check=True):
    r = subprocess.run(["git", "-c", "user.name=relresidual-loop",
                        "-c", "user.email=relresidual-loop@local", *args],
                       cwd=LOOP, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"git {args}: {r.stderr.strip()}")
    return r.stdout.strip()


def default_state(run_id):
    g0 = idea_engine.gate0_diagnosis()
    return {"run_id": run_id, "next_round": 1, "phase": "idle", "best_idea": None,
            "best_val": None, "last_keep": None, "seen_ideas": [], "history": [],
            "replicated_ideas": [], "budget": {"gpu_seconds": 0.0, "rounds": 0},
            "consecutive_crash": 0, "started_at": datetime.now().isoformat(),
            "terminated": None, "target_met": False, "gate0": g0}


def init_repo():
    (LOOP / ".gitignore").write_text(
        "results.tsv\nrun.log\ncurrent_idea.json\ntargets/\nruns/\nreplications/\nideas/\nlogs/\n__pycache__/\n",
        encoding="utf-8")
    if not (LOOP / ".git").exists():
        git("init", "-q")
    git("add", "DESIGN.md", "program.md", "prepare.py", "gate0.py", "gate0.json",
        "idea_engine.py", "train.py", "orchestrator.py", ".gitignore")
    git("commit", "-q", "-m", "baseline: RelResidual validation loop", check=False)


def wait_for_gpu(min_free_mb=35000, timeout=300):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True)
            if float(r.splitlines()[0].strip()) >= min_free_mb:
                return True
        except Exception:
            return True
        time.sleep(10)
    return False


def write_progress(state):
    g = state["gate0"]
    L = ["# PROGRESS — RelResidual 验证闭环 v6", "",
         f"**run_id**: `{state['run_id']}`  **next_round**: {state['next_round']}  **phase**: `{state['phase']}`",
         f"**Gate0**: verdict=`{g.get('verdict')}` query-last cos={g.get('cos_mean')} ratio={g.get('ratio_mean')} 高 cos 层={g.get('high_cos_layers')} —— 原版前提不成立，仅作诊断输入",
         f"**best**: {state['best_val'] if state['best_val'] is not None else 'NA'} ({state['best_idea'] or 'none'})",
         f"**llm_backend**: `{idea_engine.llm_backend_status()['status']}` — 确定性规则引擎，非 LLM",
         f"**冻结目标**: mean_dF>={P.DF_MIN}（不弱于 ASU 基线）, std(dF)<={P.DF_STD_MAX}, 所有 order dR<={P.DR_MAX}, donor<={P.DONOR_LEAK_MAX}, premature<={P.PREMATURE_MAX}, truth<={P.TRUTH_LEAK_MAX}",
         f"**TARGET_MET**: {state['target_met']}  **已独立复现 idea**: {len(state['replicated_ideas'])}", "",
         "| round | lever | val_bpb | dF[0,1,2] | mean_dF | std_dF | worst_donor | worst_prem | worst_truth | replicated | good | status |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for h in state["history"]:
        L.append(f"| {h['round']} | {h['lever']} | {h['val_bpb']:.6f} | "
                 f"{[round(x,4) for x in h['dFs']]} | {h['df_mean']:.4f} | {h['df_std']:.4f} | "
                 f"{h['worst_donor']:.3f} | {h['worst_premature']:.3f} | {h['worst_truth']:.3f} | "
                 f"{h['replicated']} | {h['good']} | {h['status']} |")
    if state["history"]:
        last = state["history"][-1]
        L += ["", "## 最后一轮判据", ""]
        for k, v in last["checks"].items():
            L.append(f"- {k}: {'PASS' if v else 'FAIL'}")
        L.append(f"- dF 差距: {last['df_mean'] - P.DF_MIN:+.4f}")
    L += ["", "## 预算", "",
          f"- rounds {state['budget']['rounds']}/{P.MAX_ROUNDS}, gpu {state['budget']['gpu_seconds']:.0f}/{P.MAX_GPU_SECONDS}s, crash {state['consecutive_crash']}", ""]
    PROGRESS.write_text("\n".join(L), encoding="utf-8")


def write_summary(state, reason):
    g = state["gate0"]
    L = ["# AUTORESEARCH_SUMMARY — RelResidual 验证闭环 v6", "",
         f"**run_id**: `{state['run_id']}`  **终止原因**: `{reason}`",
         f"**Gate0（前置假设检验）**: verdict=`{g.get('verdict')}`，query-last cos={g.get('cos_mean')}，ratio={g.get('ratio_mean')}，高 cos 层={g.get('high_cos_layers')}",
         "> 原版 RelResidual 前提不成立：B≈ΔA_r，跨关系共有的身份成分不足。后续轮均为**条件性收缩假设**，不得声称原 idea 通过。",
         "",
         f"**best val_bpb**: {state['best_val'] if state['best_val'] is not None else 'NA'}  **best idea**: `{state['best_idea'] or 'none'}`",
         f"**轮数**: {state['budget']['rounds']}  **GPU 秒**: {state['budget']['gpu_seconds']:.0f}",
         f"**TARGET_MET**: {state['target_met']}  **独立复现 idea 数**: {len(state['replicated_ideas'])}", "",
         "## 逐轮（每轮 3 个独立数据顺序）", "",
         "| round | idea | mean_dF | std_dF | donor | premature | truth | val_bpb | replicated | good | status |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for h in state["history"]:
        L.append(f"| {h['round']} | `{h['idea_id']}` | {h['df_mean']:.4f} | {h['df_std']:.4f} | "
                 f"{h['worst_donor']:.3f} | {h['worst_premature']:.3f} | {h['worst_truth']:.3f} | "
                 f"{h['val_bpb']:.6f} | {h['replicated']} | {h['good']} | {h['status']} |")
    L += ["", "## 通过独立实验复现评估指标的 idea", ""]
    if state["replicated_ideas"]:
        for r in state["replicated_ideas"]:
            L.append(f"- `{r['idea_id']}`: mean_dF={r['df_mean']:.4f}, std_dF={r['df_std']:.4f}, good={r['good']}")
    else:
        L.append("- 无。")
    L += ["", "## 终止判据", "",
          f"- `TARGET_MET`: {reason == 'TARGET_MET'}（全部条件基于 3 个独立数据顺序）",
          f"- `BUDGET_EXHAUSTED`: {reason == 'BUDGET_EXHAUSTED'}", "",
          "## 适配与局限", "",
          "- donor / truth leak 为词面启发式，不等价于语义级答案判定；`s_r` 常见词可能误报。",
          "- 单 1B / TOFU forget10 / 60 steps / answer-NLL；不得外推。",
          "- 无跨模型评审后端，思路与裁决为执行者 provisional。",
          "- `results.tsv` 未跟踪；`state.json` 原子写、可 `--resume`。", ""]
    SUMMARY.write_text("\n".join(L), encoding="utf-8")


def start_round(state):
    if not (LOOP / "gate0.json").exists():
        return False, "gate0 missing (run gate0.py first)"
    issues = P.preflight()
    if issues:
        return False, f"preflight: {issues}"
    if state["phase"] not in ("idle", "done"):
        return False, f"phase={state['phase']}"
    if state["next_round"] > 1:
        prev = LOOP / "replications" / f"R{state['next_round']-1}.json"
        if not prev.exists():
            return False, "prior replication artifact missing"
        if len(P.load_json(prev).get("order_results", [])) != len(P.REPL_ORDERS):
            return False, "prior evidence gate: incomplete orders"
    if state["budget"]["rounds"] >= P.MAX_ROUNDS or state["budget"]["gpu_seconds"] >= P.MAX_GPU_SECONDS:
        return False, "budget exhausted"
    if state["target_met"]:
        return False, "target met"
    return True, "ok"


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
        "train.py", "orchestrator.py", "DESIGN.md", "program.md")
    git("commit", "-q", "-m", f"R{n}: {spec['lever']} {spec['source']}", check=False)
    commit = git("rev-parse", "--short", "HEAD")

    wait_for_gpu()
    order_results, dt_total, failed = [], 0.0, False
    for o in P.REPL_ORDERS:
        out = LOOP / "runs" / f"R{n}_o{o}.json"
        log = LOOP / "logs" / f"R{n}_o{o}.log"
        cmd = [sys.executable, "train.py", "--data-order", str(o), "--out", str(out)]
        t0 = time.time()
        with log.open("w") as f:
            f.write("$ " + " ".join(cmd) + "\n")
            f.flush()
            p = subprocess.run(cmd, cwd=LOOP, stdout=f, stderr=subprocess.STDOUT,
                               timeout=900, env=ENV)
        dt_total += time.time() - t0
        txt = log.read_text(errors="replace")
        RUN_LOG.write_text(txt, encoding="utf-8")
        if p.returncode != 0 or not out.exists():
            failed = True
            print(f"[R{n} o{o}] CRASH\n" + "\n".join(txt.splitlines()[-20:]), flush=True)
            break
        d = P.load_json(out)
        ok, msg = P.verify_run(d)
        if not ok:
            failed = True
            print(f"[R{n} o{o}] evidence gate failed: {msg}", flush=True)
            break
        order_results.append(d)
        print(f"[R{n} o{o}] dF={d['dF']:+.4f} dR={d['dR']:+.4f} donor={d['donor_leak']:.3f} "
              f"prem={d['premature']:.3f} truth={d['truth_leak']:.3f} {time.time()-t0:.1f}s", flush=True)

    if failed:
        status, val = "crash", 0.0
        state["consecutive_crash"] += 1
        state["phase"] = "idle"
        state["next_round"] = n
        agg = P.aggregate(order_results, spec["complexity"]) if order_results else None
        result = {"round": n, "idea_id": spec["id"], "lever": spec["lever"],
                  "val_bpb": 0.0, "df_mean": float("nan"), "df_std": float("nan"),
                  "dFs": [d["dF"] for d in order_results],
                  "worst_donor": float("nan"), "worst_premature": float("nan"),
                  "worst_truth": float("nan"), "replicated": False, "good": False,
                  "checks": agg["checks"] if agg else {}, "status": status}
    else:
        agg = P.aggregate(order_results, spec["complexity"])
        val = agg["val_bpb"]
        if state["best_val"] is None or val < state["best_val"]:
            status = "keep"
            state["best_val"], state["best_idea"], state["last_keep"] = val, spec["id"], commit
        else:
            status = "discard"
        state["consecutive_crash"] = 0
        state["phase"] = "done"
        state["next_round"] = n + 1
        P.dump_json({"round": n, "idea_id": spec["id"], "order_results": order_results, **agg},
                    LOOP / "replications" / f"R{n}.json")
        result = {"round": n, "idea_id": spec["id"], "lever": spec["lever"],
                  "val_bpb": val, "df_mean": agg["df_mean"], "df_std": agg["df_std"],
                  "dFs": agg["dFs"], "dRs": agg["dRs"],
                  "worst_donor": agg["worst_donor"], "worst_premature": agg["worst_premature"],
                  "worst_truth": agg["worst_truth"], "worst_dr": agg["worst_dr"],
                  "replicated": agg["checks"]["replicated"], "good": agg["good"],
                  "checks": agg["checks"], "status": status}
        if agg["checks"]["replicated"]:
            state["replicated_ideas"].append({k: result[k] for k in
                                              ("idea_id", "df_mean", "df_std", "good")})
        print(f"[R{n}] {status:7s} val={val:.6f} mean_dF={agg['df_mean']:.4f} "
              f"std={agg['df_std']:.4f} good={agg['good']} {dt_total:.1f}s | {spec['lever']}", flush=True)
        if agg["good"]:
            state["target_met"] = True
            print(f"[R{n}] TARGET_MET with an INDEPENDENTLY replicated idea: {spec['id']}", flush=True)

    peak_gb = round(max([d["peak_vram_mb"] for d in order_results] or [0.0]) / 1024, 1)
    with TSV.open("a", encoding="utf-8") as f:
        f.write(f"{commit}\t{val:.6f}\t{peak_gb}\t{status}\t{spec['id']}\n")
    state["history"].append(result)
    state["budget"]["rounds"] += 1
    state["budget"]["gpu_seconds"] += dt_total
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
        state = default_state(args.resume or "relresidual-20261124")
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
        if state["target_met"]:
            reason = "TARGET_MET"
            break
        print(f"checkpoint: rounds={state['budget']['rounds']} best={state['best_val']}", flush=True)
        write_progress(state)

    state["terminated"], state["phase"] = reason, "terminated"
    P.dump_json(state, STATE)
    write_progress(state)
    write_summary(state, reason)
    print(f"\nLOOP STOP: {reason} | rounds={state['budget']['rounds']} best={state['best_val']} "
          f"target_met={state['target_met']} replicated={len(state['replicated_ideas'])}", flush=True)


if __name__ == "__main__":
    main()
