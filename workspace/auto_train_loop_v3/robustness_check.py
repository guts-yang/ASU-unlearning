"""POST-HOC forensic probe (NOT part of the loop ledger).

The loop's 3 'seeds' were bit-identical because the base model is pretrained,
dropout=0, and data order was fixed. This probe introduces an INDEPENDENT
variation axis: 3 different data-order permutations, same config and seed.
"""
from __future__ import annotations
import json, subprocess, sys, time
from pathlib import Path
import prepare as P

LOOP = Path(__file__).resolve().parent
ENV = dict(__import__('os').environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
spec = P.load_json(LOOP / "ideas" / "R2.json")
P.dump_json(spec, LOOP / "current_idea.json")

rows = []
for do in (0, 1, 2):
    out = LOOP / "runs" / f"R2_dataorder{do}.json"
    log = LOOP / "logs" / f"R2_dataorder{do}.log"
    cmd = [sys.executable, "train.py", "--seed", "0", "--data-order", str(do), "--out", str(out)]
    t0 = time.time()
    with log.open("w") as f:
        f.write("$ " + " ".join(cmd) + "\n"); f.flush()
        p = subprocess.run(cmd, cwd=LOOP, stdout=f, stderr=subprocess.STDOUT, timeout=900, env=ENV)
    if p.returncode != 0 or not out.exists():
        rows.append({"data_order": do, "error": "crash"}); continue
    d = P.load_json(out)
    rows.append({"data_order": do, "rho": d["rho"], "dF": d["dF"], "dR": d["dR"],
                 "refusal": d["refusal_rate"], "mean_len": d["mean_len"], "dt": time.time()-t0})

rhos = [r["rho"] for r in rows if "rho" in r]
mean = sum(rhos)/len(rhos) if rhos else float('nan')
std = (sum((x-mean)**2 for x in rhos)/len(rhos))**0.5 if rhos else float('nan')
L = ["# ROBUSTNESS — 独立数据顺序扰动复现（事后取证）", "",
     "> 说明：循环内 3 seeds 的 rho 完全相同（预训练权重 + dropout=0 + 固定数据顺序），",
     "> 因此 std=0 只证明**确定性可复现**，不构成统计独立性。本探针引入第二个变异轴：数据顺序。", "",
     "| data_order | rho | dF | dR | refusal |", "|---|---|---|---|---|"]
for r in rows:
    L.append(f"| {r['data_order']} | {r.get('rho','ERR'):.6f} | {r.get('dF','ERR'):+.4f} | "
             f"{r.get('dR','ERR'):+.4f} | {r.get('refusal','ERR'):.3f} |" if 'rho' in r
             else f"| {r['data_order']} | ERROR | — | — | — |")
L += ["", f"- rho_mean = {mean:.6f}", f"- rho_std  = {std:.6f}",
      f"- 循环内（固定数据顺序）rho_std = 0.000000",
      f"- 目标 rho_mean<={P.RHO_MEAN_TARGET}, rho_std<={P.RHO_STD_MAX}", ""]
if std <= P.RHO_STD_MAX and mean <= P.RHO_MEAN_TARGET:
    L.append("**结论**：该 idea 在独立数据顺序扰动下仍稳健达标，复现结论成立。")
else:
    L.append("**结论**：数据顺序扰动下 rho 明显波动 → `TARGET_MET` 只在**固定协议**下成立，")
    L.append("循环内的 3-seed '复现' 是确定性重跑而非独立复现；不能主张跨数据顺序稳健。")
L += ["", "> 本探针不写入 `results.tsv` 账本、不修改终止判据；`train_loop_snapshot.py` 保留了产生循环结果的确切代码。", ""]
(LOOP / "ROBUSTNESS.md").write_text("\n".join(L), encoding="utf-8")
print("\n".join(L))
