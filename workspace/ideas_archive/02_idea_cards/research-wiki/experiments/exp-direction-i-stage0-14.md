---
type: experiment
node_id: exp:exp-direction-i-stage0-14
title: "方向I stage0-14 H20实测汇总"
idea_id: "idea:cf-activation-orth-alignment-2026"
verdict: no
confidence: high
date: ""
hardware: "NVIDIA H20 CUDA12.2 torch2.5.1"
duration: ""
provenance: "workspace/experiments/direction_i/ALL_RESULTS_SUMMARY.json"
added: 2026-11-23T02:01:03Z
tags: ["unlearning", "negative-result", "direction-i"]
---

# 方向I stage0-14 H20实测汇总

**verdict:** `no`  ·  **confidence:** `high`  ·  tests `idea:cf-activation-orth-alignment-2026`

## Metrics
基线ref: forget0.555 retain0.798(最优); idea1:1.068/1.52(失败); idea4:0.651/1.01; stage7 DA-probe KILL→stage7b pooled PASS; stage12 GA强λ≥6 forget1.47-2.06但retain崩; stage13 PCGrad/EWC/GradNorm无效; stage14 GA-orth忘1.75但retain1.65

## Reasoning
反事实配对+正交/对齐类方法在TOFU-1B上系统性地既忘不掉retain又大幅受损，全部弱于原ASU温度自蒸馏基线。梯度手术未能挽救。这是类级负面研究结论，非环境噪声。

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

