#!/usr/bin/env python3
"""Aggregate LeakRU FQ/RR into table + H1/H2 verdict."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

LOGIC = ("HS", "MP", "LL", "CC", "DS", "MT")
METHODS_DEFAULT = ("ASU", "GA", "NPO")

ROOT = Path("/usr/local/ASU-unlearning/workspace")
DEFAULT_RESULTS = ROOT / "results"
OU_SAVES = Path("/usr/local/open-unlearning/workspace/saves/eval")


def _pct(x: float | None) -> str:
    if x is None:
        return "NA"
    return f"{100.0 * float(x):.1f}"


def _load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def extract_metrics(eval_json: dict[str, Any]) -> dict[str, Any]:
    fq = eval_json.get("leakru_fq") or {}
    rr = eval_json.get("leakru_rr") or {}
    acc = eval_json.get("leakru_acc") or {}
    by_type = (fq.get("by_logic_type") or {}) if isinstance(fq, dict) else {}
    by_split = (fq.get("by_split") or {}) if isinstance(fq, dict) else {}
    attacks = (rr.get("attacks") or {}) if isinstance(rr, dict) else {}
    focus = attacks.get("focus_on_key") or {}
    probab = attacks.get("probab") or {}

    # SH FQ from single-hop split; structure FQ from by_logic_type on MH-tagged rows.
    # HS column: prefer multi-hop HS if we can recover from value_by_index; else by_type HS.
    sh = by_split.get("single-hop")
    fq_struct = {k: by_type.get(k) for k in LOGIC}

    # Refine HS/MP/... using only multi-hop rows when value_by_index present on both models
    # (FQ already aggregated by type including SH-tagged HS). Recompute MH-only FQ when possible.
    un_acc = acc if "value_by_index" in (acc or {}) else None
    # Prefer SUMMARY-less path: recompute from fq by_type as primary table values.
    return {
        "fq_sh": sh,
        "fq_struct": fq_struct,
        "rr_focus": focus.get("agg_value"),
        "rr_probab": probab.get("agg_value"),
        "fq_agg": fq.get("agg_value") if isinstance(fq, dict) else None,
        "acc_agg": acc.get("agg_value") if isinstance(acc, dict) else None,
        "s_suc": rr.get("s_suc_size") if isinstance(rr, dict) else None,
        "raw_fq": fq,
        "raw_rr": rr,
        "raw_acc": acc,
    }


def recompute_mh_fq(theta_u: dict, theta_p: dict) -> dict[str, float | None]:
    """FQ by logic_type using only multi-hop rows."""
    u = (theta_u.get("leakru_acc") or {}).get("value_by_index") or {}
    p = (theta_p.get("leakru_acc") or {}).get("value_by_index") or {}
    if not u or not p:
        return {}
    # group
    from collections import defaultdict

    def group(vbi):
        g = defaultdict(list)
        for idx, rec in vbi.items():
            if rec.get("split") != "multi-hop":
                continue
            g[rec.get("logic_type")].append(bool(rec.get("correct")))
        return g

    gu, gp = group(u), group(p)
    out = {}
    for logic in LOGIC:
        pu = sum(gu[logic]) / len(gu[logic]) if gu[logic] else None
        pp = sum(gp[logic]) / len(gp[logic]) if gp[logic] else None
        if pu is None or pp is None or pp == 0:
            out[logic] = None
        else:
            out[logic] = 1.0 - (pu / pp)
    return out


def recompute_sh_fq(theta_u: dict, theta_p: dict) -> float | None:
    u = (theta_u.get("leakru_acc") or {}).get("value_by_index") or {}
    p = (theta_p.get("leakru_acc") or {}).get("value_by_index") or {}
    if not u or not p:
        return None
    def acc(vbi):
        flags = [bool(r.get("correct")) for r in vbi.values() if r.get("split") == "single-hop"]
        return (sum(flags) / len(flags)) if flags else None
    au, ap = acc(u), acc(p)
    if au is None or ap is None or ap == 0:
        return None
    return 1.0 - (au / ap)


def _gap_pp(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return (float(left) - float(right)) * 100.0


def judge(asu: dict[str, Any], ga: dict[str, Any], npo: dict[str, Any]) -> dict[str, Any]:
    fq_sh = asu.get("fq_sh")
    fq_hs = (asu.get("fq_struct") or {}).get("HS")
    fq_ds = (asu.get("fq_struct") or {}).get("DS")
    fq_mt = (asu.get("fq_struct") or {}).get("MT")
    pairwise = {
        "SH-DS": _gap_pp(fq_sh, fq_ds),
        "SH-MT": _gap_pp(fq_sh, fq_mt),
        "HS-DS": _gap_pp(fq_hs, fq_ds),
        "HS-MT": _gap_pp(fq_hs, fq_mt),
    }

    h1 = None
    gap_pp = None
    if None not in (fq_sh, fq_hs, fq_ds, fq_mt):
        gap_pp = (max(fq_sh, fq_hs) - min(fq_ds, fq_mt)) * 100.0
        h1 = gap_pp > 10.0

    rr_asu = asu.get("rr_focus")
    rr_ga = ga.get("rr_focus")
    rr_npo = npo.get("rr_focus")
    h2 = None
    if rr_asu is not None and rr_ga is not None and rr_npo is not None:
        peer_max = max(rr_ga, rr_npo)
        # 同量级或更高: not below 1/10 of the stronger control.
        same_order = rr_asu + 1e-12 >= 0.1 * peer_max
        h2 = (rr_asu > 0.20) and same_order

    if h1 is None or h2 is None:
        decision = "缺评测结果，暂无法判定；先完成 θ_p/θ_u LeakRU eval 再跑 aggregate。"
    elif h1 is True and h2 is True:
        decision = "H1∧H2 成立 → story（审计+FocusOnKey 对撞+APRU 修复）立住，可开写。"
    elif h1 is True and h2 is False:
        decision = "仅 H1 成立 → story 改为「结构路径审计为主」。"
    elif h1 is False and h2 is True:
        decision = "仅 H2 成立 → 恢复攻击有信号，但结构路径差距不足 10 个百分点，不按完整 story 开写。"
    else:
        decision = "均不成立 → 报告反直觉阴性结果，重新评估 base 选择。"

    return {
        "H1": h1,
        "H2": h2,
        "gap_pp": gap_pp,
        "pairwise_pp": pairwise,
        "rr_focus_asu": rr_asu,
        "rr_focus_ga": rr_ga,
        "rr_focus_npo": rr_npo,
        "decision": decision,
        "fq_sh": fq_sh,
        "fq_hs": fq_hs,
        "fq_ds": fq_ds,
        "fq_mt": fq_mt,
    }


def markdown_table(rows: dict[str, dict[str, Any]]) -> str:
    headers = ["Method", "FQ_SH", *[f"FQ_{k}" for k in LOGIC], "RR_FocusOnKey", "RR_Probab"]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for method, m in rows.items():
        cells = [
            method,
            _pct(m.get("fq_sh")),
            *[_pct((m.get("fq_struct") or {}).get(k)) for k in LOGIC],
            _pct(m.get("rr_focus")),
            _pct(m.get("rr_probab")),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--theta-p", type=Path, default=OU_SAVES / "leakru_tofu05_theta_p" / "LeakRU_EVAL.json")
    ap.add_argument("--asu", type=Path, default=OU_SAVES / "leakru_tofu05_asu" / "LeakRU_EVAL.json")
    ap.add_argument("--ga", type=Path, default=OU_SAVES / "leakru_tofu05_ga" / "LeakRU_EVAL.json")
    ap.add_argument("--npo", type=Path, default=OU_SAVES / "leakru_tofu05_npo" / "LeakRU_EVAL.json")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_RESULTS)
    ap.add_argument("--notes", type=Path, default=ROOT / "notes" / "run_meta.json")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    theta_p = _load(args.theta_p) if args.theta_p.exists() else {}
    paths = {"ASU": args.asu, "GA": args.ga, "NPO": args.npo}
    metrics: dict[str, dict[str, Any]] = {}
    missing = []
    for name, path in paths.items():
        if not path.exists():
            missing.append(str(path))
            metrics[name] = {
                "fq_sh": None,
                "fq_struct": {k: None for k in LOGIC},
                "rr_focus": None,
                "rr_probab": None,
            }
            continue
        raw = _load(path)
        m = extract_metrics(raw)
        mh = recompute_mh_fq(raw, theta_p) if theta_p else {}
        if mh:
            m["fq_struct"] = {**m["fq_struct"], **mh}
        sh = recompute_sh_fq(raw, theta_p) if theta_p else None
        if sh is not None:
            m["fq_sh"] = sh
        metrics[name] = m

    table = markdown_table(metrics)
    table_path = args.out_dir / "asu_leakru_table.md"
    meta = {}
    if args.notes.exists():
        meta = json.loads(args.notes.read_text(encoding="utf-8"))

    table_doc = [
        "# ASU × LeakRU Sanity Results",
        "",
        "FQ reported in percentage points of `1 - Acc(θu)/Acc(θp)`. RR in percent.",
        "",
        table,
        "",
        "## Paths",
        f"- θ_p: `{args.theta_p}`",
        f"- ASU: `{args.asu}`",
        f"- GA: `{args.ga}`",
        f"- NPO: `{args.npo}`",
        "",
        "## Run meta",
        "```json",
        json.dumps(meta, indent=2, ensure_ascii=False),
        "```",
    ]
    if missing:
        table_doc.extend(["", "## Missing evals", *[f"- {p}" for p in missing]])
    table_path.write_text("\n".join(table_doc) + "\n", encoding="utf-8")

    verdict = judge(metrics["ASU"], metrics["GA"], metrics["NPO"])
    verdict_path = args.out_dir / "H_verdict.md"
    def yn(v):
        if v is True:
            return "成立"
        if v is False:
            return "不成立"
        return "无法判定（缺数）"

    def pp(v):
        return "NA" if v is None else f"{v:.2f}"

    pairs = verdict.get("pairwise_pp") or {}
    matched = meta.get("controls_matched")
    if matched is True:
        match_line = "GA/NPO 的 FQ_SH 与 ASU 相差 ≤ 10 个百分点。"
    elif matched is False:
        match_line = "GA/NPO 的 FQ_SH 未对齐到 ASU 的 10 个百分点以内（已选最接近的 checkpoint，未再加扫）。"
    else:
        match_line = "对照对齐状态见 run_meta。"
    verdict_doc = f"""# H1 / H2 判定报告

## 判定标准（固定）

- **H1 成立**：`max(FQ_SH, FQ_HS) − min(FQ_DS, FQ_MT) > 10`（百分点）。SH/HS/DS/MT 缺任一则无法判定。
- **H2 成立**：`RR_FocusOnKey(ASU) > 20%`，且不低于 GA/NPO 较高者的 1/10（同量级或更高）。对照缺失则无法判定。

## 实测

| 量 | 值 |
| --- | --- |
| FQ_SH(ASU) | {_pct(verdict['fq_sh'])} |
| FQ_HS(ASU) | {_pct(verdict['fq_hs'])} |
| FQ_DS(ASU) | {_pct(verdict['fq_ds'])} |
| FQ_MT(ASU) | {_pct(verdict['fq_mt'])} |
| gap_pp | {pp(verdict['gap_pp'])} |
| SH−DS | {pp(pairs.get('SH-DS'))} |
| SH−MT | {pp(pairs.get('SH-MT'))} |
| HS−DS | {pp(pairs.get('HS-DS'))} |
| HS−MT | {pp(pairs.get('HS-MT'))} |
| RR_FocusOnKey(ASU) | {_pct(verdict['rr_focus_asu'])} |
| RR_FocusOnKey(GA) | {_pct(verdict['rr_focus_ga'])} |
| RR_FocusOnKey(NPO) | {_pct(verdict['rr_focus_npo'])} |

## 对照对齐

{match_line}

## 结论

- **H1**：{yn(verdict['H1'])}
- **H2**：{yn(verdict['H2'])}
- **决策**：{verdict['decision']}

## 下一步建议

- 若 H1∧H2：进入审计+FocusOnKey 对撞实验与 APRU 修复叙事。
- 若仅 H1：收缩为结构路径审计，弱化恢复攻击主线。
- 若仅 H2：保留恢复攻击信号，但不把结构路径差距当成已成立。
- 若均不成立：保留本阴性结果，更换 base/协议或扩大结构与模型再测；不要为撑假设调参。

## 复现元信息

- seed / split / τ / α / attacks：见 `workspace/notes/run_meta.json`
- GPU：{meta.get("gpu_status", "unknown")} — {meta.get("gpu_detail", "")}
"""
    verdict_path.write_text(verdict_doc, encoding="utf-8")
    print(f"Wrote {table_path}")
    print(f"Wrote {verdict_path}")
    print(json.dumps({k: verdict[k] for k in ('H1','H2','gap_pp','decision')}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
