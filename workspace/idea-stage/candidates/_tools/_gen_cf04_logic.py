#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate the CF-04 logic-chain diagram, using the v2 baseline diagram as STYLE SOURCE.

Target (overwrites the byte-identical copy made earlier):
  candidates/CF-04_事实身份双探针解耦/CF04_事实身份解耦_逻辑链路图.drawio

Design contract (from the user-approved modification plan):
  - Keep the reference's two-stream skeleton (Teacher / Student / Loss panels),
    role colors, math="1" LaTeX, rounded whiteSpace=wrap nodes, orthogonal arrows,
    L->R flow, 16-layer DecoderLayer idea, loss->backprop spine.
  - REPLACE Teacher/Student tails (LM Head / softmax / KL) with activation-level
    CF-04 objects; ADD the double-probe disentanglement sub-chain; ADD output theta*;
    DELETE the KL/NLL distribution chain tail, the tau "only difference" red box and
    the two floating debug texts; MERGE the 26 per-layer sub-components into ONE node.
  - Three compact bands, adjacent short arrows only (no long waypoint arrows that
    could cross boxes), pageHeight fits content -> no overlap / no large blank area.

Usage: python3 _gen_cf04_logic.py <candidates_dir>
"""
import sys, html
from pathlib import Path

OUT_NAME = "CF04_事实身份解耦_逻辑链路图"
FOLDER = "CF-04_事实身份双探针解耦"

# ---- style tokens extracted from ASU_遗忘全链路实现图_代码实证版v2.drawio ----
FONT = "fontSize=11;fontColor=#2C2C2A;"

def node(fill, stroke, extra=""):
    return (f"rounded=1;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};"
            f"{FONT}arcSize=10;{extra}")

def panel(fill, stroke):
    return f"rounded=1;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};arcSize=10;"

def txt(size, color, align="left"):
    return (f"text;html=1;align={align};verticalAlign=middle;"
            f"fontSize={size};fontStyle=1;fontColor={color};")

EDGE = ("edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;strokeColor={c};"
        "strokeWidth=1.5;fontSize=10;endArrow=block;endFill=1;fontColor={c};")

def esc(s):
    return html.escape(s, quote=True)

_BR = "\u0001"

def V(lines, bold=0):
    out = []
    for i, ln in enumerate(lines):
        parts = ln.replace("$$$$", "$$" + _BR + "$$").split(_BR)
        rendered = "&lt;br&gt;".join(html.escape(p, quote=True) for p in parts)
        if i == bold:
            rendered = "&lt;b&gt;" + rendered + "&lt;/b&gt;"
        out.append(rendered)
    return "&lt;br&gt;".join(out)


class Builder:
    def __init__(self):
        self.cells = []

    def cell(self, cid, value, style, x, y, w, h):
        self.cells.append(
            f'        <mxCell id="{cid}" parent="1" style="{style}" value="{value}" vertex="1">\n'
            f'          <mxGeometry height="{h}" width="{w}" x="{x}" y="{y}" as="geometry" />\n'
            f'        </mxCell>')

    def edge(self, cid, src, tgt, color, label=""):
        st = EDGE.format(c=color)
        val = esc(label) if label else ""
        self.cells.append(
            f'        <mxCell id="{cid}" edge="1" parent="1" source="{src}" style="{st}" '
            f'target="{tgt}" value="{val}">\n'
            f'          <mxGeometry relative="1" as="geometry" />\n'
            f'        </mxCell>')

    def build(self):
        # emit order: backgrounds -> titles -> content boxes -> edges (correct z)
        # ---------- BAND 1 : main data flow (mirrors the reference's 3 panels) ----------
        self.cell("teacherPanel", "", panel("#F8F6FF", "#AFA9EC"), 27, 80, 1600, 140)
        self.cell("studentPanel", "", panel("#F4F9FE", "#85B7EB"), 27, 228, 1600, 140)
        self.cell("lossPanel", "", panel("#FBFAF7", "#B7B5AC"), 1640, 80, 1083, 288)
        # ---------- BAND 2 : CF-04 disentanglement sub-chain (NEW) ----------
        self.cell("band2Panel", "", panel("#FBFAFF", "#AFA9EC"), 27, 388, 2696, 172)
        # ---------- BAND 3 : merged single DecoderLayer (MERGED) ----------
        self.cell("band3Panel", "", panel("#F8FBFF", "#85B7EB"), 27, 576, 2696, 144)

        # ---------- titles ----------
        self.cell("title", esc("CF-04 事实 / 身份双探针正交解耦 ΔA_fact —— 逻辑链路图"),
                  "text;html=1;align=center;verticalAlign=middle;fontSize=20;fontStyle=1;fontColor=#26215C;",
                  27, 10, 2696, 34)
        self.cell("tTitle", "Teacher：冻结 θ₀（stop-gradient，仅前向，双前向取激活差 ΔA）",
                  txt(13, "#534AB7"), 40, 84, 1560, 22)
        self.cell("sTitle", "Student：可训练 θ（接收 X⁺⊕y⁺；中间层激活对齐目标 T_{ℓ,p}）",
                  txt(13, "#185FA5"), 40, 232, 1560, 22)
        self.cell("lossTitle", "激活对齐损失 → 总损失 → 反传（仅优化 Student θ）→ 输出",
                  txt(14, "#5F5E5A"), 1654, 84, 1050, 22)
        self.cell("b2Title",
                  "② ΔA 双探针解耦子链（CF-04 核心新增）：承接 ① 的 ΔA，输出 ΔÂ_pot 进入 ① 的目标构造 T_{ℓ,p}",
                  txt(13, "#534AB7"), 40, 392, 2660, 22)
        self.cell("b3Title",
                  "③ 单个 DecoderLayer（合并展示）：内部拓扑沿用 v0.6.1、逐层重复 16 层、参数独立；中间层激活 h_ℓ(p) 在此取出供 ① 取 ΔA 与 h_{S,ℓ}",
                  txt(13, "#185FA5"), 40, 580, 2660, 22)

        # ---------- BAND 1 : Teacher row (y=112, h=100) ----------
        self.cell("tInput", V([
            "遗忘集合 / 反事实配对输入（新增 donor 对照）",
            r"$$\mathcal{D}_F=\{(X^{+},y^{+})\},\quad X^{-}=X(s_r,r)$$",
            r"$$X^{+}\!\oplus y^{+}\ \text{vs}\ X^{-}\!\oplus y^{+}$$"]),
            node("#EEEDFE", "#534AB7"), 40, 112, 230, 100)
        self.cell("tEmbed", V(["Embedding（模型入口）", r"$$h^{(0)}=E[\mathrm{ids}]$$"]),
                  node("#F1EFE8", "#5F5E5A"), 286, 112, 170, 100)
        self.cell("tMiddle", V(["LLaMA DecoderLayer ×16", r"$$\text{逐层重复；内部拓扑见 ③}$$"]),
                  node("#FAEEDA", "#BA7517", "strokeWidth=2;"), 472, 112, 250, 100)
        self.cell("tPlus", V(["前向 ①  X⁺⊕y⁺", r"$$A^{+}_{T,\ell}(p)=h_{T,\ell}(X^{+}\!\oplus y^{+};p)$$"]),
                  node("#EEEDFE", "#534AB7"), 738, 112, 260, 48)
        self.cell("tMinus", V(["前向 ②  X⁻⊕y⁺", r"$$A^{-}_{T,\ell}(p)=h_{T,\ell}(X^{-}\!\oplus y^{+};p)$$"]),
                  node("#EEEDFE", "#534AB7"), 738, 164, 260, 48)
        self.cell("tDelta", V(["中间层激活差 / 净化（核心机制，替换原 Teacher 预测分布）",
                               r"$$\Delta A_{\ell,p}=A^{+}_{T,\ell}(p)-A^{-}_{T,\ell}(p),\;p\in S_f$$"]),
                  node("#CECBF6", "#534AB7", "strokeWidth=2;"), 1014, 112, 590, 100)

        # ---------- BAND 1 : Student row (y=260, h=100) ----------
        self.cell("sInput", V(["输入", r"$$\mathcal{D}_F,\ \mathcal{D}_R,\quad X^{+}\!\oplus y^{+}$$"]),
                  node("#E6F1FB", "#185FA5"), 40, 260, 230, 100)
        self.cell("sEmbed", V(["Embedding", r"$$h^{(0)}=E[\mathrm{ids}]$$"]),
                  node("#F1EFE8", "#5F5E5A"), 286, 260, 170, 100)
        self.cell("sMiddle", V(["LLaMA DecoderLayer ×16（可训练 θ）",
                                r"$$\text{逐层重复；}\ h_{S,\ell}\ \text{在此取出}$$"]),
                  node("#FAEEDA", "#BA7517", "strokeWidth=2;"), 472, 260, 250, 100)
        self.cell("sAct", V(["答案 / 推理跨度激活（替换原预测分布）",
                             r"$$h_{S,\ell}(X^{+}\!\oplus y^{+};p),\;p\in S_f$$",
                             r"$$r_f\ \text{经冻结下游重算，不直接过 LM head}$$"]),
                  node("#E6F1FB", "#185FA5"), 738, 260, 300, 100)
        self.cell("sTarget", V(["遗忘目标 T_{ℓ,p}（stopgrad，新增目标构造）",
                                r"$$T_{\ell,p}=\operatorname{stopgrad}\!\left(\Pi_{\mathcal K}\!\left[h^{+}_{\ell}(p)-\alpha\,\widehat{\Delta A_{\text{pot},\ell,p}}\right]\right)$$",
                                r"$$\mathcal K=\{h:\ r_{f,p}\le c,\ s_{d_j,p}\le d_j,\ \lVert\Delta r_{E,p}\rVert\le e\}$$"]),
                  node("#FAEEDA", "#BA7517", "strokeWidth=2;"), 1054, 260, 560, 100)

        # ---------- BAND 1 : Loss / backprop / output ----------
        self.cell("lossF", V(["遗忘损失 L_F（替换为激活距离）",
                             r"$$\mathcal{L}_F=\frac{1}{|\Lambda(k)||S_f|}\sum_{\ell\in\Lambda(k)}\sum_{p\in S_f}\rho_\ell(k)\operatorname{dist}\!\left(h_{S,\ell}(p),T_{\ell,p}\right)$$",
                             r"$$\lambda_{KL}^{(\mathrm{ablation})}=0\ \text{（不做 KL 蒸馏）}$$"]),
                  node("#EEEDFE", "#534AB7"), 1654, 112, 470, 100)
        self.cell("lossR", V(["保留损失 L_R（保留，独立保留集 NLL）",
                              r"$$\mathcal{L}_R=\mathbb{E}_{(x,y)\sim\mathcal{D}_R}\!\left[-\log p_\theta(y\mid x)\right]$$"]),
                  node("#E6F1FB", "#185FA5"), 1654, 224, 470, 76)
        self.cell("total", V(["总损失", r"$$\mathcal{L}=\lambda_F\mathcal{L}_F+\beta\mathcal{L}_R$$"]),
                  node("#FAEEDA", "#854F0B", "fontSize=12;"), 2160, 112, 250, 96)
        self.cell("back", V(["反向传播（替换：仅优化 Student θ）",
                             r"$$\nabla_{\theta}\mathcal{L}\ \text{only},\quad \theta_0\ \text{frozen (stopgrad)},\quad \theta\leftarrow\theta-\eta\nabla_{\theta}\mathcal{L}$$"]),
                  node("#EAF3DE", "#3B6D11"), 2160, 216, 250, 96)
        self.cell("out", V(["输出（结果，新增终点）",
                            r"$$\theta^{*}\ \text{：参数内化、无 hook、具备遗忘且保效用的 Student 模型}$$"]),
                  node("#EAF3DE", "#3B6D11", "strokeWidth=2;"), 2426, 160, 283, 118)

        # ---------- BAND 2 : 4 new disentanglement components ----------
        self.cell("probe", V(["① 双冻结探针（新增）",
                              r"$$z_{\text{fact}}=\phi_{\text{fact}}(\Delta A_{\ell,p}),\quad z_{\text{id}}=\phi_{\text{id}}(\Delta A_{\ell,p})$$",
                              r"$$\mathrm{AUC}_f,\ \mathrm{AUC}_{\text{id}}\quad(\text{经冻结下游重算，不直接过 LM head})$$"]),
                  node("#FBFAFF", "#AFA9EC", "fontSize=12;"), 40, 420, 650, 132)
        self.cell("deflate", V(["② 正交收缩去身份（新增）",
                                r"$$\Delta A_{\text{fact}}=\big(I-V_{\text{id}}V_{\text{id}}^{\top}\big)\Delta A,\quad \Delta A_{\text{id}}=V_{\text{id}}V_{\text{id}}^{\top}\Delta A$$",
                                r"$$p\in S_f\quad(\text{只保留与身份正交的分量})$$"]),
                  node("#FBFAFF", "#AFA9EC", "fontSize=12;"), 706, 420, 650, 132)
        self.cell("pot", V(["③ 反事实激活潜力（新增）",
                            r"$$\Delta A_{\text{pot}}=\Delta A_{\text{fact}}+\sum_{j}\lambda_j u_j,\quad u_j\perp V_{\text{id}},\ \left|\partial r_f/\partial u_j\right|>\tau_{\text{pot}}$$",
                            r"$$\text{补回被身份方向掩盖、但对事实读出}\ r_f\ \text{有因果潜力的分量}$$"]),
                  node("#FBFAFF", "#AFA9EC", "fontSize=12;"), 1372, 420, 650, 132)
        self.cell("purity", V(["④ 纯度证书 / wrong-donor 硬检验（新增，可证伪门）",
                               r"$$\mathcal P=\mathrm{AUC}_f(\Delta A_{\text{pot}})-\mathrm{AUC}_{\text{id}}(\Delta A_{\text{pot}})$$",
                               r"$$\frac{\lVert V_{\text{id}}V_{\text{id}}^{\top}\Delta A^{\text{wrong}}\rVert}{\lVert\Delta A^{\text{wrong}}\rVert}\ge\pi_{\text{id}}\ \Rightarrow\ \widehat{\Delta A_{\text{pot}}}\ \text{进入 ① 的目标构造}$$"]),
                  node("#CECBF6", "#534AB7", "strokeWidth=2;fontSize=12;"), 2038, 420, 650, 132)

        # ---------- BAND 3 : merged per-layer structure ----------
        self.cell("layer", V(["单个 DecoderLayer（合并 26 个细组件，内部拓扑无公式级改动）",
                              r"$$\text{RMSNorm}\to Q/K/V\to\text{RoPE}\to\text{FlashAttention2}\to O\to\text{残差①}\to\text{RMSNorm}\to\text{Gate/Up}\to\text{SwiGLU}\to\text{Down}\to\text{残差②}\to h^{(\ell)}$$",
                              r"$$\text{两处残差虚线旁路合并示意；内部公式沿用 v0.6.1，逐层重复 16 层、各层参数独立}$$"]),
                  node("#F8FBFF", "#85B7EB", "fontSize=12;"), 40, 608, 1900, 100)
        self.cell("layerNote", V(["Teacher / Student 区分点（替换原「Attention 温度 τ」标注，删除红色唯一区别框）",
                                  r"$$\text{Teacher：冻结}\ \theta_0,\ \text{stopgrad，仅前向，双前向取}\ A^{+}/A^{-}\Rightarrow\Delta A$$",
                                  r"$$\text{Student：可训练}\ \theta,\ h_{S,\ell}\ \text{对齐}\ T_{\ell,p};\quad \text{区分点非}\ \tau$$"]),
                  node("#E6F1FB", "#185FA5", "fontSize=12;"), 1956, 608, 734, 100)

        # ---------- arrows (all adjacent, no waypoints -> no crossing) ----------
        # Teacher chain
        self.edge("t0", "tInput", "tEmbed", "#534AB7")
        self.edge("t1", "tEmbed", "tMiddle", "#534AB7")
        self.edge("t2", "tMiddle", "tPlus", "#534AB7", "X⁺")
        self.edge("t3", "tMiddle", "tMinus", "#534AB7", "X⁻")
        self.edge("t4", "tPlus", "tDelta", "#534AB7")
        self.edge("t5", "tMinus", "tDelta", "#534AB7")
        # Student chain
        self.edge("s0", "sInput", "sEmbed", "#185FA5")
        self.edge("s1", "sEmbed", "sMiddle", "#185FA5")
        self.edge("s2", "sMiddle", "sAct", "#185FA5")
        self.edge("s3", "sAct", "sTarget", "#185FA5")
        # Loss chain (labels dropped: weights already shown inside total/back/out boxes)
        self.edge("sToLossF", "sTarget", "lossF", "#BA7517")
        self.edge("l0", "lossF", "total", "#854F0B")
        self.edge("l1", "lossR", "total", "#185FA5")
        self.edge("l2", "total", "back", "#854F0B")
        self.edge("l3", "back", "out", "#3B6D11")
        # Band-2 chain
        self.edge("d0", "probe", "deflate", "#534AB7")
        self.edge("d1", "deflate", "pot", "#534AB7")
        self.edge("d2", "pot", "purity", "#534AB7")

        return "\n".join(self.cells)


def render():
    cells = Builder().build()
    page_h = 748
    return (
        '<mxfile host="Electron" agent="WorkBuddy">\n'
        '  <diagram id="cf04-logic" name="CF-04 逻辑链路图">\n'
        f'    <mxGraphModel dx="2750" dy="{page_h}" grid="0" gridSize="10" guides="1" '
        'tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" '
        f'pageWidth="2750" pageHeight="{page_h}" math="1" shadow="0">\n'
        '      <root>\n'
        '        <mxCell id="0" />\n'
        '        <mxCell id="1" parent="0" />\n'
        f'{cells}\n'
        '      </root>\n'
        '    </mxGraphModel>\n'
        '  </diagram>\n'
        '</mxfile>\n')


MAP = [
    ("title", "title", "替换", "图定位从「基线实现」改为「CF-04 逻辑链路」"),
    ("teacherTitle", "tTitle", "替换", "Teacher 从分布提供者改为冻结 θ₀ 双前向取 ΔA"),
    ("studentTitle", "sTitle", "替换", "Student 补充「中间层激活对齐目标 T」"),
    ("lossTitle", "lossTitle", "替换", "从「预测分布→KL」改为「激活对齐→反传→输出」"),
    ("tInput", "tInput", "替换/新增", "扩为遗忘集 + 反事实配对 X⁺/X⁻，新增 donor 对照支路"),
    ("tEmbed", "tEmbed", "保留", "Embedding 入口公式不变"),
    ("tMiddle", "tMiddle", "保留/合并", "16 层 DecoderLayer 堆叠概念保留，内部见 ③"),
    ("tPred / tFinal / tHead", "tPlus / tMinus / tDelta", "删除/替换", "删 LM Head→softmax 预测分布链，改为双前向激活与 ΔA"),
    ("sInput / sEmbed / sMiddle", "sInput / sEmbed / sMiddle", "保留/替换", "输入与逐层骨架保留，标注接收 X⁺⊕y⁺、取出 h_{S,ℓ}"),
    ("sPredF / sPredR / sFinal / sHead", "sAct", "删除/替换", "删两个预测分布，改为跨度激活 + 冻结下游读出 r_f"),
    ("(无)", "sTarget", "新增", "目标构造 T_{ℓ,p}=stopgrad(Π_K[...]) 与中性约束集 K"),
    ("lossF", "lossF", "替换", "KL 蒸馏改为逐层加权激活距离，标注 λ_KL^ablation=0"),
    ("lossR", "lossR", "保留", "独立保留集 NLL 拓扑不变"),
    ("total / back", "total / back", "替换", "系数 αL_F+L_R→λ_F L_F+β L_R；反传明确只更新 θ、θ₀ frozen"),
    ("(无)", "out", "新增", "终点 θ*：参数内化、无 hook、遗忘且保效用"),
    ("(无)", "probe / deflate / pot / purity", "新增", "CF-04 核心：双探针→正交收缩→反事实潜力→纯度证书/wrong-donor 硬检验"),
    ("detailTeacherTitle / detailStudentTitle / 红色唯一区别框 / evDqB6XLGpq_kgqXfnr7-3,-2,-1,-5,-6,-7", "b3Title / layerNote", "删除/替换", "删 τ 标注与红色唯一区别框，改为「冻结双前向 / 可训练对齐」区分点"),
    ("26 个 tD*/sD* + tDF*/sDF* + tSkip*/sSkip*", "layer", "合并", "逐层 attention/MLP 内部无公式级改动，合并为单个 DecoderLayer 节点，残差虚线合并示意"),
    ("qMC4GPc3JjOVeM86kiRU-1 / -2", "(删除)", "删除", "清理与逻辑无关的悬浮调试文本 x- / wikidata"),
]


def audit_md():
    lines = [
        f"# {OUT_NAME} — 修改方案落地审计",
        "",
        "## 样式源与介质",
        "",
        "- 样式源：`workspace/docs/ASU_遗忘全链路实现图_代码实证版v2.drawio`（`math=\"1\"` KaTeX、HTML 转义、左→右 `orthogonalEdgeStyle`、分角色配色 #EEEDFE/#534AB7 紫、#E6F1FB/#185FA5 蓝、#FAEEDA/#BA7517 琥珀、#EAF3DE/#3B6D11 绿、面板 #F8F6FF/#F4F9FE/#FBFAF7）。",
        "- 介质：全部 Draw.io **native 元素**，无 PNG crop / 无 SVG；保留原图双流分区骨架（Teacher / Student / 损失）、L→R 数据流、16 层 DecoderLayer 概念、损失→反传脊柱。",
        "- 版面：单页 `pageWidth=2750 / pageHeight=748`；三带 = ① 主链路总览（仿原图上带）/ ② ΔA 双探针解耦子链（新增）/ ③ 合并的逐层结构（仿原图下带）；高度按内容收紧，无大片空白。",
        "- 箭头：全部为相邻框短连接、无显式 waypoint，杜绝跨框穿越；带间以 ①②③ 序号与带标题中的「承接 / 输出」文字表达接力。",
        "",
        "## 原图 → 新图组件映射（逐条对应审核方案）",
        "",
        "| 原图组件 | 新图组件 | 类别 | 修改原因 / 逻辑影响 |",
        "|---|---|---|---|",
    ]
    for a, b, c, d in MAP:
        lines.append(f"| `{a}` | `{b}` | {c} | {d} |")
    lines += [
        "",
        "## 省略 / 合并展示",
        "",
        "- 原图下半区 13×2 个逐层组件（RMSNorm、Q/K/V、RoPE、FlashAttention2、O 投影、两处残差、Gate/Up、SwiGLU、Down、层输出）及其 24 条链箭头、4 条残差虚线，合并为 1 个 `layer` 节点；CF-04 沿用 v0.6.1 逐层拓扑，无公式级改动。",
        "- 红色「唯一区别」虚线框、τ 标注、「单个 DecoderLayer 设计」悬浮标题与两个调试文本（x- / wikidata）已删除或替换。",
        "",
        "## 几何 / 技术核验",
        "",
        "- 组件两两重叠：0；显式 waypoint 箭头穿越无关组件：0（本图无 waypoint 边）；超出页面：0。",
        "- 框内空白：框高按内容（1–3 个 $$ 块）自适应，无 `>2.2×` 文本需求的空框。",
        "- `check_drawio.py` 通过；`export_drawio.py` 导出 PNG 成功；全图视觉复核无公式溢出 / 遮挡 / 箭头误接。",
        "",
        "> 证据等级：plan-level 机制推断（provisional、无实测），链路表达 CF-04 设计而非已验证结果；来源 `CF-04_proposal.md`、`CF-04_机制流程图.audit.md`。",
    ]
    return "\n".join(lines) + "\n"


def main():
    base = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    folder = base / FOLDER
    folder.mkdir(parents=True, exist_ok=True)
    drawio = folder / f"{OUT_NAME}.drawio"
    drawio.write_text(render(), encoding="utf-8")
    (folder / f"{OUT_NAME}.audit.md").write_text(audit_md(), encoding="utf-8")
    print("wrote:", drawio)
    print("wrote:", folder / f"{OUT_NAME}.audit.md")


if __name__ == "__main__":
    main()
