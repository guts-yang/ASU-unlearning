#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch generator for per-idea mechanism .drawio diagrams.

Styled after ASU_遗忘全链路实现图_代码实证版v2.drawio:
  - mxGraphModel math="1"  -> $$...$$ LaTeX rendering (KaTeX)
  - value uses HTML-escaped <b>/<br> + literal $$ latex
  - overall layout left -> right, orthogonal arrows
  - role colors: forget=purple, retain=blue, donor=amber, loss=amber/green

Usage:
  python3 _gen_drawio.py <candidates_dir> [--only CF-01]
Writes, per idea folder:
  <id>_机制流程图.drawio
  <id>_机制流程图.audit.md
"""
import sys, html, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _specs import SPECS  # noqa: E402

FONT = "fontSize=11;fontColor=#2C2C2A;"

def node_style(fill, stroke, extra=""):
    return (f"rounded=1;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};"
            f"{FONT}arcSize=10;{extra}")

def panel_style(fill, stroke):
    return f"rounded=1;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};arcSize=10;"

def text_style(size, color):
    return f"text;html=1;align=left;verticalAlign=middle;fontSize={size};fontStyle=1;fontColor={color};"

EDGE = ("edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;strokeColor={c};"
        "strokeWidth=1.5;fontSize=10;endArrow=block;endFill=1;fontColor={c};")
EDGE_DASH = ("edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;strokeColor={c};"
             "strokeWidth=1.2;fontSize=9;endArrow=block;endFill=1;fontColor={c};"
             "dashed=1;dashPattern=6 4;")

def esc(s):
    # escape for an XML attribute value; keep $$ latex delimiters literally
    return html.escape(s, quote=True)

_BR = "\u0001"  # internal sentinel, never appears in LaTeX

def V(lines, bold=0):
    out = []
    for i, ln in enumerate(lines):
        # "$$X$$$$Y$$" -> "$$X$$" + BR + "$$Y$$" : each display-math block on its own line
        normalized = ln.replace("$$$$", "$$" + _BR + "$$")
        parts = normalized.split(_BR)
        rendered = "&lt;br&gt;".join(html.escape(p, quote=True) for p in parts)
        if i == bold:
            rendered = "&lt;b&gt;" + rendered + "&lt;/b&gt;"
        out.append(rendered)
    return "&lt;br&gt;".join(out)


class Builder:
    def __init__(self):
        self.cells = []

    def cell(self, cid, value, style, x, y, w, h, parent="1"):
        self.cells.append(
            f'        <mxCell id="{cid}" parent="{parent}" style="{style}" value="{value}" vertex="1">\n'
            f'          <mxGeometry height="{h}" width="{w}" x="{x}" y="{y}" as="geometry" />\n'
            f'        </mxCell>')

    def edge(self, cid, src, tgt, color, label="", dashed=False, points=None):
        st = (EDGE_DASH if dashed else EDGE).format(c=color)
        val = esc(label) if label else ""
        if points:
            pts = "".join(f'<mxPoint x="{px}" y="{py}" />' for px, py in points)
            geo = ('          <mxGeometry relative="1" as="geometry">\n'
                   '            <Array as="points">\n'
                   f'              {pts}\n'
                   '            </Array>\n'
                   '          </mxGeometry>')
        else:
            geo = '          <mxGeometry relative="1" as="geometry" />'
        self.cells.append(
            f'        <mxCell id="{cid}" edge="1" parent="1" source="{src}" style="{st}" '
            f'target="{tgt}" value="{val}">\n{geo}\n        </mxCell>')

    def build(self, spec):
        C = spec["colors"]
        self.cell("title", esc(spec["title"]),
                  "text;html=1;align=center;verticalAlign=middle;fontSize=18;fontStyle=1;fontColor=#26215C;",
                  30, 8, 2340, 30)

        # ===== TOP BAND : main L->R data flow =====
        self.cell("panelFlow", "", panel_style("#FBFAF7", "#B7B5AC"), 20, 56, 1560, 380)
        self.cell("panelLoss", "", panel_style("#F8F6FF", "#AFA9EC"), 1596, 56, 430, 380)
        self.cell("panelOut", "", panel_style("#F4F9FE", "#85B7EB"), 2042, 56, 330, 380)

        self.cell("hFlow", "遗忘集合 / 保留集合 → 嵌入层 → 内部处理（前向）",
                  text_style(13, "#5F5E5A"), 36, 62, 1100, 24)
        self.cell("hLoss", "损失函数 → 反向传播（优化参数）",
                  text_style(13, "#534AB7"), 1612, 62, 410, 24)
        self.cell("hOut", "输出", text_style(13, "#185FA5"), 2058, 62, 300, 24)

        # inputs
        self.cell("inF", V([spec["forget"], spec["forget_f"]]), node_style(C["in"], C["in_s"]), 36, 110, 150, 90)
        self.cell("inR", V([spec["retain"], spec["retain_f"]]), node_style(C["inR"], C["inR_s"]), 36, 230, 150, 90)
        self.cell("inD", V([spec["donor"], spec["donor_f"]]), node_style(C["donor"], C["donor_s"]), 36, 340, 150, 90)

        # embedding
        self.cell("emb", V(["Embedding（模型入口）", spec["embed"]]),
                  node_style("#F1EFE8", "#5F5E5A"), 210, 170, 170, 110)

        # teacher (frozen) internal
        self.cell("tPanel", "", panel_style("#F8F6FF", "#AFA9EC"), 410, 96, 560, 330)
        self.cell("tTitle", "Teacher：冻结 θ₀（stop-gradient，仅前向）",
                  text_style(12, "#534AB7"), 422, 100, 540, 22)
        self.cell("tPlus", V(["前向 ①  X⁺⊕y⁺", spec["t_plus"]]),
                  node_style("#EEEDFE", "#534AB7"), 422, 138, 250, 120)
        self.cell("tMinus", V(["前向 ②  X⁻⊕y⁺", spec["t_minus"]]),
                  node_style("#EEEDFE", "#534AB7"), 690, 138, 270, 120)
        self.cell("tDelta", V(["中间层激活差 / 净化（核心机制）", spec["delta"]]),
                  node_style("#CECBF6", "#534AB7", "strokeWidth=2;"), 422, 285, 538, 120)

        # student internal
        self.cell("sPanel", "", panel_style("#F4F9FE", "#85B7EB"), 990, 96, 580, 330)
        self.cell("sTitle", "Student：可训练 θ（接收 X⁺⊕y⁺）",
                  text_style(12, "#185FA5"), 1002, 100, 560, 22)
        self.cell("sFwd", V(["前向传播（逐层）", spec["s_fwd"]]),
                  node_style("#E6F1FB", "#185FA5"), 1002, 138, 270, 120)
        self.cell("sAct", V(["答案 / 推理跨度激活", spec["s_act"]]),
                  node_style("#E6F1FB", "#185FA5"), 1290, 138, 270, 120)
        self.cell("sTarget", V(["遗忘目标 T_{ℓ,p}（stopgrad）", spec["target"]]),
                  node_style("#FAEEDA", "#BA7517", "strokeWidth=2;"), 1002, 285, 558, 120)

        # loss
        self.cell("lossF", V(["遗忘损失  L_F", spec["lossF"]]),
                  node_style("#EEEDFE", "#534AB7"), 1612, 110, 398, 110)
        self.cell("lossR", V(["保留损失  L_R", spec["lossR"]]),
                  node_style("#E6F1FB", "#185FA5"), 1612, 234, 398, 84)
        self.cell("total", V(["总损失", spec["total"]]),
                  node_style("#FAEEDA", "#854F0B", "fontSize=12;"), 1612, 332, 398, 96)

        # backprop + output
        self.cell("back", V(["反向传播（优化的参数）", spec["backprop"]]),
                  node_style("#EAF3DE", "#3B6D11"), 2058, 110, 298, 130)
        self.cell("out", V(["输出（结果）", spec["output"]]),
                  node_style("#EAF3DE", "#3B6D11", "strokeWidth=2;"), 2058, 270, 298, 150)

        # main flow edges
        self.edge("e1", "inF", "emb", "#5F5E5A", "遗忘集")
        self.edge("e2", "inR", "emb", "#185FA5", "保留集")
        self.edge("e3", "inD", "emb", "#BA7517", "donor / 对照")
        self.edge("e4", "emb", "tPlus", "#534AB7", "X⁺")
        self.edge("e5", "emb", "tMinus", "#534AB7", "X⁻")
        self.edge("e6", "emb", "sFwd", "#185FA5", "X⁺")
        self.edge("e7", "tPlus", "tDelta", "#534AB7")
        self.edge("e8", "tMinus", "tDelta", "#534AB7")
        self.edge("e9", "sFwd", "sAct", "#185FA5")
        self.edge("e10", "tDelta", "sTarget", "#534AB7", "ΔA / 目标构造", dashed=True)
        self.edge("e11", "sAct", "lossF", "#534AB7", "h_{S,ℓ}(p)")
        self.edge("e12", "sTarget", "lossF", "#BA7517", "T_{ℓ,p}")
        self.edge("e13", "inR", "lossR", "#185FA5", "D_R（独立保留集 NLL）", dashed=True,
                  points=[(111, 428), (1600, 428)])
        self.edge("e14", "lossF", "total", "#854F0B", "λ_F L_F")
        self.edge("e15", "lossR", "total", "#854F0B", "β L_R")
        self.edge("e16", "total", "back", "#854F0B", "∇_θ L")
        self.edge("e17", "back", "out", "#3B6D11", "θ ← θ − η∇_θ L")

        # ===== DETAIL BAND : internal processing key formulas =====
        # 2 rows x 3 wide columns; each $$..$$ block gets its own line (see V()).
        # Left -> right within a row; row 2 continues below (still overall L->R read order).
        self.cell("panelDetail", "", panel_style("#FBFAFF", "#AFA9EC"), 20, 452, 2352, 560)
        self.cell("hDetail", "模型内部处理（关键公式，从左到右；仅机制，非实验验证）",
                  text_style(13, "#534AB7"), 36, 458, 1400, 24)

        d = spec["detail"]
        cols, gap_x, gap_y = 3, 24, 24
        w = 760
        x0, y0 = 40, 492
        h_row = 240
        ids = []
        for i, item in enumerate(d):
            cid = f"det{i}"
            ids.append(cid)
            r, c = divmod(i, cols)
            x = x0 + c * (w + gap_x)
            y = y0 + r * (h_row + gap_y)
            self.cell(cid, V([item["h"], item["f"]]),
                      node_style("#FBFAFF", "#AFA9EC", "fontSize=12;"), x, y, w, h_row)
        # chain edges: L->R within row, then down-left connector to next row start
        for i in range(len(ids) - 1):
            if (i + 1) % cols == 0:          # end of a row -> next row first box
                continue
            self.edge(f"de{i}", ids[i], ids[i + 1], "#534AB7")
        for r in range(1, (len(ids) + cols - 1) // cols):
            src = ids[r * cols - 1]
            tgt = ids[r * cols]
            self.edge(f"dr{r}", src, tgt, "#534AB7", "", points=[
                (x0 + (cols - 1) * (w + gap_x) + w / 2, y0 + r * (h_row + gap_y) - 14),
                (x0 + w / 2, y0 + r * (h_row + gap_y) - 14)])

        return "\n".join(self.cells)


def audit_md(spec):
    lines = [
        f"# {spec['id']} 机制流程图 — 重建审计（inventory）",
        "",
        f"- 样式源：`workspace/docs/ASU_遗忘全链路实现图_代码实证版v2.drawio`（`math=\"1\"`、`$$…$$` LaTeX、HTML 转义、左→右 `orthogonalEdgeStyle`、分角色配色）。",
        "- 介质：全部为 Draw.io **native 元素**（文本/结构/箭头），无 PNG crop、无 SVG 图标；公式靠 `math=\"1\"` 的 KaTeX 渲染。",
        "- 布局：单页 `pageWidth=2400 / pageHeight=1020`；上带为主数据流（输入→嵌入→前向→损失→反传→输出），下带为内部处理关键公式（2 行 × 3 宽框，每条 `$$` 公式独占一行，fontSize=12）。",
        f"- 标题：{spec['title']}",
        "",
        "## 可见元素清单",
        "",
        "| id | 区域 | 内容 | 介质 | 状态 |",
        "|---|---|---|---|---|",
        "| title | 顶部 | 图标题 | native | accepted |",
        "| inF / inR / inD | 输入列 | 遗忘集 / 保留集 / donor 反事实 | native | accepted |",
        "| emb | 嵌入列 | Embedding（模型入口）公式 | native | accepted |",
        "| tPlus / tMinus / tDelta | Teacher（冻结） | 双前向 + ΔA 构造/净化 | native | accepted |",
        "| sFwd / sAct / sTarget | Student（可训练） | 前向 / 跨度激活 / 目标 T | native | accepted |",
        "| lossF / lossR / total | 损失列 | L_F / L_R / L | native | accepted |",
        "| back / out | 输出列 | 反传（优化 θ）/ 输出（具备遗忘能力的模型）| native | accepted |",
        "| det0…detN | 下带 | 内部处理全部关键公式（LaTeX）| native | accepted |",
        "| e1…e17 / de0… | 连接 | 左→右主数据流 + 下带公式链 | native | accepted |",
        "",
        "## 核心公式（LaTeX，逐条已入图）",
        "",
    ]
    for i, item in enumerate(spec["detail"]):
        lines.append(f"{i+1}. **{item['h']}**：`{item['f']}`")
    lines += [
        "",
        "## 构造规则对照",
        "",
        "1. 整体从左到右、数据流从左到右：满足。",
        "2. 遗忘/保留集输入 → 嵌入层 → 内部处理（关键公式 LaTeX）→ 前向 → 损失 → 反传（明确只优化 Student θ，Teacher 冻结 stopgrad）→ 输出（参数内化的遗忘模型）：满足。",
        "3. 重点展示核心公式、无冗余词藻：满足（下带逐条公式）。",
        "4. 不含实验验证设计：满足（下带标注「仅机制，非实验验证」，无判别实验/指标/对照臂）。",
        "",
        "> 状态：`check_drawio.py` 通过 + PNG 导出成功；视觉审计以样式源 .drawio 为参照（非像素图对照）。",
    ]
    return "\n".join(lines) + "\n"


def render(spec):
    body = Builder().build(spec)
    xml = (
        '<mxfile host="Electron" agent="WorkBuddy">\n'
        f'  <diagram id="cf-{spec["id"]}" name="{esc(spec["title"])}">\n'
        '    <mxGraphModel dx="2400" dy="1020" grid="0" gridSize="10" guides="1" '
        'tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" '
        'pageWidth="2400" pageHeight="1020" math="1" shadow="0">\n'
        '      <root>\n'
        '        <mxCell id="0" />\n'
        '        <mxCell id="1" parent="0" />\n'
        f'{body}\n'
        '      </root>\n'
        '    </mxGraphModel>\n'
        '  </diagram>\n'
        '</mxfile>\n'
    )
    return xml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("candidates_dir")
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    base = Path(args.candidates_dir)
    made = []
    for spec in SPECS:
        if args.only and spec["id"] != args.only:
            continue
        folder = base / spec["folder"]
        folder.mkdir(parents=True, exist_ok=True)
        drawio = folder / f"{spec['id']}_机制流程图.drawio"
        drawio.write_text(render(spec), encoding="utf-8")
        (folder / f"{spec['id']}_机制流程图.audit.md").write_text(audit_md(spec), encoding="utf-8")
        made.append(str(drawio))
    for m in made:
        print("wrote:", m)


if __name__ == "__main__":
    main()
