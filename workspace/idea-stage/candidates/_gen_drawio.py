#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate per-idea mechanism .drawio diagrams, styled after
ASU_遗忘全链路实现图_代码实证版v2.drawio (math="1", $$..$$ LaTeX, L->R flow).

Usage:
  python3 gen_drawio.py <out_dir> <idea_id> <spec.json>
The spec.json carries the idea-specific LaTeX formulas / labels.
"""
import json, sys, html
from pathlib import Path

# ---------- style tokens (extracted from the reference .drawio) ----------
FONT = "fontSize=11;fontColor=#2C2C2A;"
def box(fill, stroke, extra=""):
    return f"rounded=1;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};{FONT}arcSize=10;{extra}"
def node_style(fill, stroke, extra=""):
    return f"rounded=1;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};{FONT}arcSize=10;{extra}"

EDGE = ("edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;strokeColor={c};"
        "strokeWidth=1.5;fontSize=10;endArrow=block;endFill=1;fontColor={c};")
EDGE_DASH = ("edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;strokeColor={c};"
             "strokeWidth=1.2;fontSize=9;endArrow=block;endFill=1;fontColor={c};"
             "dashed=1;dashPattern=6 4;")
TEXT = "text;html=1;align=left;verticalAlign=middle;fontSize={s};fontStyle=1;fontColor={c};"

# ---------- XML escaping for drawio value ($$ must survive; HTML tags escaped) ----------
def esc(s):
    # drawio stores HTML inside value; escape < > & but keep $$ latex delimiters
    return html.escape(s, quote=False).replace("&quot;", '"')

def V(lines, bold=None):
    # lines: list of plain strings; bold = index of the header line
    out = []
    for i, ln in enumerate(lines):
        if i == 0 and bold is not None:
            out.append("&lt;b&gt;" + esc(ln) + "&lt;/b&gt;")
        else:
            out.append(esc(ln))
    return "&lt;br&gt;".join(out)

# ---------- cell helpers ----------
cells = []
def cell(cid, value, style, x, y, w, h, parent="1"):
    cells.append(f'        <mxCell id="{cid}" parent="{parent}" style="{style}" value="{value}" vertex="1">\n'
                 f'          <mxGeometry height="{h}" width="{w}" x="{x}" y="{y}" as="geometry" />\n'
                 f'        </mxCell>')

def edge(cid, src, tgt, color, label="", dashed=False, points=None, exitY=0.5, entryY=0.5):
    st = (EDGE_DASH if dashed else EDGE).format(c=color)
    val = esc(label) if label else ""
    geo = '          <mxGeometry relative="1" as="geometry" />'
    if points:
        pts = "".join(f'<mxPoint x="{px}" y="{py}" />' for px, py in points)
        geo = ('          <mxGeometry relative="1" as="geometry">\n'
               '            <Array as="points">\n'
               f'              {pts}\n'
               '            </Array>\n'
               '          </mxGeometry>')
    cells.append(f'        <mxCell id="{cid}" edge="1" parent="1" source="{src}" style="{st}" target="{tgt}" value="{val}">\n'
                 f'{geo}\n'
                 f'        </mxCell>')

# ---------- fixed layout skeleton (L -> R) ----------
# columns (x): input 30 | embed 220 | teacher lanes 410..980 | student 1010..1580 | loss 1610..2000 | output 2040..2380
# Rows: top summary band y=70..430 ; detail band y=470..760 (internal processing formulas)
def build(spec):
    C = spec["colors"]
    title = spec["title"]
    # title
    cell("title", esc(title),
         "text;html=1;align=center;verticalAlign=middle;fontSize=18;fontStyle=1;fontColor=#26215C;",
         30, 8, 2340, 30)

    # ===== TOP BAND (main data flow) =====
    # panel backgrounds
    cell("panelFlow", "", "rounded=1;whiteSpace=wrap;html=1;fillColor=#FBFAF7;strokeColor=#B7B5AC;arcSize=10;",
         20, 56, 1560, 380)
    cell("panelLoss", "", "rounded=1;whiteSpace=wrap;html=1;fillColor=#F8F6FF;strokeColor=#AFA9EC;arcSize=10;",
         1596, 56, 430, 380)
    cell("panelOut", "", "rounded=1;whiteSpace=wrap;html=1;fillColor=#F4F9FE;strokeColor=#85B7EB;arcSize=10;",
         2042, 56, 330, 380)

    cell("hFlow", "遗忘集合 / 保留集合 → 嵌入层 → 内部处理（前向）",
         TEXT.format(s=13, c="#5F5E5A"), 36, 62, 1100, 24)
    cell("hLoss", "损失函数 → 反向传播（优化参数）",
         TEXT.format(s=13, c="#534AB7"), 1612, 62, 410, 24)
    cell("hOut", "输出",
         TEXT.format(s=13, c="#185FA5"), 2058, 62, 300, 24)

    # ---- row 1: inputs ----
    cell("inF", V([spec["forget"], spec["forget_f"]], 0),
         node_style(C["in"], C["in_s"]), 36, 110, 150, 90)
    cell("inR", V([spec["retain"], spec["retain_f"]], 0),
         node_style(C["inR"], C["inR_s"]), 36, 230, 150, 90)
    cell("inD", V([spec["donor"], spec["donor_f"]], 0),
         node_style(C["donor"], C["donor_s"]), 36, 340, 150, 90)

    # ---- col 2: embedding ----
    cell("emb", V(["Embedding（模型入口）", spec["embed"]], 0),
         node_style("#F1EFE8", "#5F5E5A"), 210, 170, 170, 110)

    # ---- col 3: teacher (frozen) internal ----
    cell("tPanel", "", "rounded=1;whiteSpace=wrap;html=1;fillColor=#F8F6FF;strokeColor=#AFA9EC;arcSize=10;",
         410, 96, 560, 330)
    cell("tTitle", "Teacher：冻结 θ₀（stop-gradient，仅前向）",
         TEXT.format(s=12, c="#534AB7"), 422, 100, 540, 22)
    cell("tPlus", V(["前向 ①  X⁺⊕y⁺", spec["t_plus"]], 0),
         node_style("#EEEDFE", "#534AB7"), 422, 138, 250, 120)
    cell("tMinus", V(["前向 ②  X⁻⊕y⁺", spec["t_minus"]], 0),
         node_style("#EEEDFE", "#534AB7"), 690, 138, 270, 120)
    cell("tDelta", V(["中间层激活差 / 净化", spec["delta"]], 0),
         node_style("#CECBF6", "#534AB7", "strokeWidth=2;"), 422, 285, 538, 120)

    # ---- col 4: student internal ----
    cell("sPanel", "", "rounded=1;whiteSpace=wrap;html=1;fillColor=#F4F9FE;strokeColor=#85B7EB;arcSize=10;",
         990, 96, 580, 330)
    cell("sTitle", "Student：可训练 θ（接收 X⁺⊕y⁺）",
         TEXT.format(s=12, c="#185FA5"), 1002, 100, 560, 22)
    cell("sFwd", V(["前向传播（逐层）", spec["s_fwd"]], 0),
         node_style("#E6F1FB", "#185FA5"), 1002, 138, 270, 120)
    cell("sAct", V(["答案/推理跨度激活", spec["s_act"]], 0),
         node_style("#E6F1FB", "#185FA5"), 1290, 138, 270, 120)
    cell("sTarget", V(["遗忘目标 T_{ℓ,p}（stopgrad）", spec["target"]], 0),
         node_style("#FAEEDA", "#BA7517", "strokeWidth=2;"), 1002, 285, 558, 120)

    # ---- col 5: loss ----
    cell("lossF", V(["遗忘损失  L_F", spec["lossF"]], 0),
         node_style("#EEEDFE", "#534AB7"), 1612, 110, 398, 110)
    cell("lossR", V(["保留损失  L_R", spec["lossR"]], 0),
         node_style("#E6F1FB", "#185FA5"), 1612, 234, 398, 84)
    cell("total", V(["总损失", spec["total"]], 0),
         node_style("#FAEEDA", "#854F0B", "fontSize=12;"), 1612, 332, 398, 96)

    # ---- col 6: backprop + output ----
    cell("back", V(["反向传播（优化参数）", spec["backprop"]], 0),
         node_style("#EAF3DE", "#3B6D11", "fontSize=11;"), 2058, 110, 298, 130)
    cell("out", V(["输出（结果）", spec["output"]], 0),
         node_style("#EAF3DE", "#3B6D11", "strokeWidth=2;"), 2058, 270, 298, 150)

    # ---- main flow edges (L -> R) ----
    edge("e1", "inF", "emb", "#5F5E5A", "遗忘集", exitY=0.2, entryY=0.2)
    edge("e2", "inR", "emb", "#185FA5", "保留集", exitY=0.5, entryY=0.5)
    edge("e3", "inD", "emb", "#BA7517", "donor / 对照", exitY=0.8, entryY=0.8)
    edge("e4", "emb", "tPlus", "#534AB7", "X⁺", exitY=0.2)
    edge("e5", "emb", "tMinus", "#534AB7", "X⁻", exitY=0.2)
    edge("e6", "emb", "sFwd", "#185FA5", "X⁺", exitY=0.5)
    edge("e7", "tPlus", "tDelta", "#534AB7")
    edge("e8", "tMinus", "tDelta", "#534AB7")
    edge("e9", "sFwd", "sAct", "#185FA5")
    edge("e10", "tDelta", "sTarget", "#534AB7", "ΔA / 目标构造", dashed=True)
    edge("e11", "sAct", "lossF", "#534AB7", "h_S", exitY=0.3)
    edge("e12", "sTarget", "lossF", "#BA7517", "T_{ℓ,p}", exitY=0.7)
    edge("e13", "inR", "lossR", "#185FA5", "D_R", dashed=True, points=[(1600, 275), (1600, 276)])
    edge("e14", "lossF", "total", "#854F0B", "λ_F L_F")
    edge("e15", "lossR", "total", "#854F0B", "β L_R")
    edge("e16", "total", "back", "#854F0B", "∇_θ L")
    edge("e17", "back", "out", "#3B6D11", "θ ← θ − η∇_θ L")

    # ===== DETAIL BAND (internal processing: all key formulas) =====
    cell("panelDetail", "", "rounded=1;whiteSpace=wrap;html=1;fillColor=#FBFAFF;strokeColor=#AFA9EC;arcSize=10;",
         20, 452, 2352, 360)
    cell("hDetail", "模型内部处理（关键公式，从左到右）",
         TEXT.format(s=13, c="#534AB7"), 36, 458, 1200, 24)

    d = spec["detail"]  # list of {"h":header,"f":formula,"w":width}
    x = 40
    y = 500
    dw = 150
    dh = 270
    ids = []
    for i, item in enumerate(d):
        cid = f"d{i}"
        ids.append(cid)
        w = item.get("w", 220)
        val = V([item["h"], item["f"]], 0)
        cell(cid, val, node_style("#FBFAFF", "#AFA9EC", "fontSize=10;"), x, y, w, dh)
        x += w + 18
    # connect detail boxes left -> right
    for i in range(len(ids) - 1):
        edge(f"de{i}", ids[i], ids[i + 1], "#534AB7", "", exitY=0.5, entryY=0.5)

    return "\n".join(cells)


def main():
    out_dir = Path(sys.argv[1])
    idea_id = sys.argv[2]
    spec = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))
    body = build(spec)
    xml = (
        '<mxfile host="Electron" agent="WorkBuddy">\n'
        f'  <diagram id="cf-{idea_id}" name="{esc(spec["title"])}">\n'
        '    <mxGraphModel dx="2400" dy="840" grid="0" gridSize="10" guides="1" '
        'tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" '
        'pageWidth="2400" pageHeight="840" math="1" shadow="0">\n'
        '      <root>\n'
        '        <mxCell id="0" />\n'
        '        <mxCell id="1" parent="0" />\n'
        f'{body}\n'
        '      </root>\n'
        '    </mxGraphModel>\n'
        '  </diagram>\n'
        '</mxfile>\n'
    )
    out = out_dir / f"{idea_id}_机制流程图.drawio"
    out.write_text(xml, encoding="utf-8")
    print(f"wrote: {out}")

if __name__ == "__main__":
    main()
