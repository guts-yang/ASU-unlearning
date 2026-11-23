---
type: idea
node_id: idea:sensitemp-adaptive-temperature-2026
title: "SensiTemp:层-头自适应温度场 (R3首选)"
stage: proposed
outcome: pending
added: 2026-11-23T05:19:07Z
based_on: []
target_gaps: []
tags: ["unlearning", "attention", "temperature", "adaptive", "r3"]
---

# SensiTemp:层-头自适应温度场 (R3首选)

**stage:** `proposed`  ·  **outcome:** `pending`

## Thesis
按factual-vs-function token敏感度为每个层/head分配ASU温度tau_{l,h},替代全局标量tau。保持ASU单机制简洁性,不训练新模块。决定性gate:若敏感度矩阵近似均匀则退化全局tau,直接否决。

## Key risks
需先做1-2GPU·h敏感度诊断;ADU/ALMPU占据head级模块空间但本方案不训练新模块

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

