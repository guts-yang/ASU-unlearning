---
type: idea
node_id: idea:cf-activation-orth-alignment-2026
title: "反事实配对ΔA正交对齐遗忘"
stage: archived
outcome: negative
added: 2026-11-23T02:01:03Z
based_on: []
target_gaps: []
tags: ["unlearning", "counterfactual", "activation", "orthogonal", "negative-result"]
---

# 反事实配对ΔA正交对齐遗忘

**stage:** `archived`  ·  **outcome:** `negative`

## Thesis
用关系匹配donor的配对激活差ΔA主成分定义事实子空间，Student在X+上沿该方向正交抑制/对齐并逐层渐进。H20+TOFU-1B实测：远弱于原ASU基线(ref forget0.555/retain0.798)，idea1 forget1.068/retain1.52；stage8/9/13全部gate失败；stage13 PCGrad/EWC/GradNorm梯度手术全失败。

## Key risks
已证伪：ΔA pooled probe仅stage7b过(rank8 top5解释0.45-0.50，与retain正交0.23)，但未pooled stage7为KILL(与wrong-donor重叠0.55)；正交抑制更接近抹表征而非移除事实。结论：反事实配对路线整体负结果。

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

