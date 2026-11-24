---
type: idea
node_id: idea:sensitemp-adaptive-temperature-2026
title: "SensiTemp:层-头自适应温度场 (R3首选)"
stage: archived
outcome: negative
added: 2026-11-23T05:19:07Z
based_on: []
target_gaps: []
tags: ["unlearning", "attention", "temperature", "adaptive", "r3"]
---

# SensiTemp:层-头自适应温度场 (R3首选)

**stage:** `archived`  ·  **outcome:** `negative`

## Thesis
按factual-vs-function token敏感度为每个层/head分配ASU温度tau_{l,h},替代全局标量tau。保持ASU单机制简洁性,不训练新模块。决定性gate:若敏感度矩阵近似均匀则退化全局tau,直接否决。

## V4 auto-validation
实测敏感度非均匀（layer CV=1.11，head CV=9.22），但等平均 τ 全局对照 dF=+1.792 ≥ SensiTemp +1.747，打乱场 +1.517；收益来自更强总平滑，而非结构化分配，故降级为诊断分析。

## Key risks
需先做1-2GPU·h敏感度诊断;ADU/ALMPU占据head级模块空间但本方案不训练新模块

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

