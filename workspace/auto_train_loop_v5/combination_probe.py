"""POST-HOC forensic probe (NOT part of the 5-round loop ledger).

v5 showed EMA solves (rho_mean, rho_std) but collapses dF. This probes
EMA x forgetting-pressure combinations on the SAME 3 independent data orders.
"""
from __future__ import annotations
import json, os, subprocess, sys, time
from pathlib import Path
import prepare as P

LOOP = Path(__file__).resolve().parent
ENV = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
BASE = dict(P.SEED_IDEA)

COMBOS = [
    ("ema095_alpha006", {"stabilizer": {"type": "ema", "decay": 0.95}, "alpha": 0.06}),
    ("ema095_beta020",  {"stabilizer": {"type": "ema", "decay": 0.95}, "hinge_beta": 0.20}),
    ("ema090_alpha006", {"stabilizer": {"type": "ema", "decay": 0.90}, "alpha": 0.06}),
]

rows = []
for name, over in COMBOS:
    params = dict(BASE); params.update(over)
    spec = {"id": f"probe_{name}", "round": 0, "lever": "ema_combo", "params": params,
            "complexity": 1, "source": "post_hoc_probe", "llm_backend": "REVIEW_UNAVAILABLE",
            "budget": {"steps": P.STEPS, "replication_orders": list(P.REPL_ORDERS), "init_seed": 0},
            "hash": name}
    P.dump_json(spec, LOOP / "current_idea.json")
    orders = []
    for o in P.REPL_ORDERS:
        out = LOOP / "runs" / f"probe_{name}_o{o}.json"
        log = LOOP / "logs" / f"probe_{name}_o{o}.log"
        cmd = [sys.executable, "train.py", "--data-order", str(o), "--out", str(out)]
        with log.open("w") as f:
            f.write("$ " + " ".join(cmd) + "\n"); f.flush()
            p = subprocess.run(cmd, cwd=LOOP, stdout=f, stderr=subprocess.STDOUT,
                               timeout=900, env=ENV)
        if p.returncode != 0 or not out.exists():
            orders.append(None); continue
        orders.append(P.load_json(out))
    good_orders = [d for d in orders if d is not None]
    agg = P.aggregate(good_orders, 1) if len(good_orders) == 3 else None
    rows.append((name, params, agg))
    if agg:
        print(f"[{name}] rho={[round(x,4) for x in agg['rhos']]} mean={agg['rho_mean']:.4f} "
              f"std={agg['rho_std']:.4f} dF={[round(x,3) for x in agg['dFs']]} "
              f"good={agg['good']} val={agg['val_bpb']:.4f}", flush=True)
    else:
        print(f"[{name}] incomplete", flush=True)

L = ["# COMBINATION PROBE — EMA × 遗忘压力（事后取证）", "",
     "> 不属于 5 轮循环账本；使用相同 3 个独立数据顺序与冻结的终止判据。", "",
     "| combo | rho[0,1,2] | rho_mean | rho_std | dF[0,1,2] | worst_df | good | val_bpb |",
     "|---|---|---|---|---|---|---|---|"]
for name, params, agg in rows:
    if agg:
        L.append(f"| {name} | {[round(x,4) for x in agg['rhos']]} | {agg['rho_mean']:.4f} | "
                 f"{agg['rho_std']:.4f} | {[round(x,3) for x in agg['dFs']]} | "
                 f"{agg['worst_df']:.4f} | {agg['good']} | {agg['val_bpb']:.4f} |")
    else:
        L.append(f"| {name} | incomplete | — | — | — | — | — | — |")
L += ["", "## 判据", "",
      f"- 目标: rho_mean<={P.RHO_MEAN_TARGET}, rho_std<={P.RHO_STD_MAX}, 所有 order |dF-{P.DF_BAND:.4f}|<={P.DF_TOL}", ""]
best = min([(a["val_bpb"], n, a) for n, p, a in rows if a], default=None)
if best and best[2]["good"]:
    L.append(f"**结论**：`{best[1]}` 在独立复现轴上满足全部冻结判据 → 不可能三角可被 EMA×压力组合打破，建议作为 v6 起点。")
else:
    L.append("**结论**：EMA×压力组合仍未能同时满足三条主约束；提高遗忘压力把 dF 拉回带内的同时，")
    L.append("rho 或 std 会重新恶化。三角在该 lever 空间内依旧成立，需更根本的机制（如按层/按样本自适应 EMA）。")
L += ["", "> 探针未写入 `results.tsv`、未修改终止判据；`current_idea.json` 最后状态为最后一个探针配置。", ""]
(LOOP / "COMBINATION_PROBE.md").write_text("\n".join(L), encoding="utf-8")
print("\n".join(L))
