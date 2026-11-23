"""Post-loop forensic control (NOT part of the 6-round loop ledger).

B5 lowered rho_P3 but also inflated dF from +1.1416 to +1.8046. This control
reproduces a comparable dF with a plain global-tau teacher (no extra loss) and
checks whether rho drops by the same amount -> if yes, B5's gain is a
denominator artifact, not deeper forgetting.
"""
from __future__ import annotations
import importlib.util, json, re, subprocess, sys
from pathlib import Path

LOOP = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("drv", LOOP / "loop_driver2.py")
drv = importlib.util.module_from_spec(spec); spec.loader.exec_module(drv)

CFG = {"round": "B6", "kind": "control", "projection": "none",
       "complexity": 0, "tau": 2.607,
       "description": "control: matched-dF global tau=2.607 (no hinge)"}
src = (LOOP / "train.py").read_text(encoding="utf-8")
pat = re.compile(r"CONFIG = \{.*?\r?\n\}\r?\n", re.S)
(LOOP / "train.py").write_text(
    pat.sub("CONFIG = " + json.dumps(CFG, indent=4) + "\n", src, count=1), encoding="utf-8")

log = LOOP / "logs2" / "B6_matched_df.log"
with log.open("w") as f:
    p = subprocess.run([sys.executable, "train.py"], cwd=LOOP, stdout=f,
                       stderr=subprocess.STDOUT, timeout=900)
txt = log.read_text(errors="replace")
print(txt[-600:])
if p.returncode != 0:
    sys.exit(p.returncode)
res = json.loads((LOOP / "rounds2" / "B6.json").read_text())
anchor = json.loads((LOOP / "anchors2" / "depth.json").read_text())
b5 = json.loads((LOOP / "rounds2" / "B5.json").read_text())
rows = [
    ("B0 anchor tau=2.25", anchor["dF"], anchor["rho"]),
    ("B5 hinge_keep", b5["delta"]["forget"], b5["p3"]["rho"]),
    ("B6 matched-dF control", res["delta"]["forget"], res["p3"]["rho"]),
]
out = ["# Matched-ΔF control — B5 的 ρ 下降是否为分母伪影", "",
       "| 配置 | ΔF | ρ_P3 |", "|---|---|---|"]
for n, d, r in rows:
    out.append(f"| {n} | {d:+.4f} | {r:.6f} |")
d_rho_b5 = b5["p3"]["rho"] - anchor["rho"]
d_rho_b6 = res["p3"]["rho"] - anchor["rho"]
out += ["", f"- B5 相对 anchor 的 ρ 变化：{d_rho_b5:+.6f}",
        f"- B6（仅把 ΔF 提到 {res['delta']['forget']:.4f}，无任何抗恢复项）的 ρ 变化：{d_rho_b6:+.6f}",
        f"- B5 的净收益（扣除 ΔF 膨胀可解释部分）：{d_rho_b5 - d_rho_b6:+.6f}", ""]
if d_rho_b6 <= d_rho_b5 + 0.005:
    out.append("**结论**：单纯增大 ΔF 就能得到同等或更大的 ρ 下降 → B5 的改善是分母膨胀伪影，不能主张「抗恢复更深」。")
else:
    out.append("**结论**：B6 未能复现 B5 的 ρ 下降 → B5 可能含真实的抗恢复成分，但仍需 matched-ΔF 与 3-seed 复核。")
out += ["", "## 指标缺陷记录", "",
        "$r_2$ 的第三项 `1.5*max(0, dF_a - dF_c)` 只惩罚「忘得更少」，未惩罚「忘得更多」，",
        "因此无法单独识破 ΔF 膨胀退化。下一版指标需改为对称项",
        "`1.5*|dF_c - dF_a|`，或强制所有候选在同一 ΔF 预算带上比较。",
        "", "> 本对照为取证补实验，不写入 6 轮 `results.tsv` 账本，避免污染 keep/discard 链。", ""]
(LOOP / "MATCHED_DF_CONTROL.md").write_text("\n".join(out), encoding="utf-8")
print("\n".join(out))
