# 方向 A 实验记录

状态：**本轮只交脚手架，未跑训练、未跑 7B 前向、未跑三通道审计、未填真实恢复率。**

冻结对照：[workspace/configs/asu_frozen_2026-09-27.yaml](configs/asu_frozen_2026-09-27.yaml)，数字来自 `workspace/results/2026-09-27/`。

## 冻结基线（TOFU）

| split | loss | MU | FE |
| --- | --- | --- | --- |
| forget01 | ASU+GD | 0.766 | 0.642 |
| forget01 | ASU+KL | 0.746 | 0.649 |
| forget05 | ASU+GD（主攻击对象） | 0.740 | 0.774 |
| forget05 | ASU+KL | 0.619 | 0.813 |
| forget10 | ASU+GD | 0.739 | 0.776 |
| forget10 | ASU+KL | 0.637 | 0.812 |

论文 Table 1 TOFU-f05：MU 74.18 / FE 77.84，仅旁注。

## 冻结基线（MUSE，本轮不立项）

| run | VerbMem_f | PrivLeak | KnowMem_f | KnowMem_r |
| --- | --- | --- | --- | --- |
| news ASU_klr | 7.41 | 99.0 | 52.32 | 46.76 |
| books ASU_klr | 6.65 | −53.5 | （空） | （空） |

## 本轮交付

- `workspace/audit/`：秩剖面、字面匹配、Probab / FocusOnKey / INT4-QRA 接口、L4 协议
- `workspace/rank_shatter/`：$\mathcal{L}_{\mathrm{rank}}$、logit bias、`train_teacher_bias.py` 骨架（标明不要跑）
- `asu_loss(..., logit_bias=None)` 可选接入，默认与基线数值一致
- CPU 合成数据单测

## 下一轮（有 checkpoint 之后）

秩剖面 → 三通道 → L4 判定 → 真训 $\theta_\tau^+$ 与学生 → 复审。MU 硬底 0.73，对照行 0.740。
