# 方向 J M0 复核状态

日期：2026-10-30。该文档不替代或覆盖 [`结论.md`](结论.md) 和
[`derived/m0_summary.json`](derived/m0_summary.json) 的旧 NO-GO 判决。

## 仓库可验证记录

- 规范链路是 `Right-to-be-forgotten/forget.py` 与
  `trainer/losses.py` 的 ASU+GD；“open-unlearning 基座”指初始化权重
  snapshot，不表示 M0 使用了 open-unlearning trainer。
- 旧 M0 使用 forget05、`task_id=1`、Teacher
  `attention_temp=2.3, layers_id=null`，共 200 题、10 位作者和 8,649 个
  有效答案 token。
- ASU Student 作者级 `p_S(g)<p_T(g)` 均值为 0.3479，作者 bootstrap
  区间为 [0.3139, 0.3829]；Teacher gold Top-10 作者均值为 0.7502。
- 旧 KL API 检查只取第一个 ASU batch 中第一题的 47 个答案位置。最大绝对
  误差为 `1.0244548320770264e-08`，最大相对误差为
  `1.9368059292901307e-04`，符号一致。
- 旧实现将相对误差定义为
  `abs(observed-analytic) / abs(analytic).clamp_min(1e-6)`，并要求最大
  相对误差严格小于 `1e-4`、最大绝对误差严格小于 `1e-5`、符号一致。
  因此旧判决严格为 **NO-GO**。

## 本次新增但尚未运行的复核

[`diagnose_m0_recheck.py`](diagnose_m0_recheck.py) 保持旧 gate、旧第一个
batch/47-token 抽样和旧 `asu_loss` API 不变，额外记录：

- 最大相对误差的 batch/query/label 位置和 gold token ID；
- `p_S(g)`、`p_T(g)`、解析梯度、autograd 梯度、scale 与相对分母；
- `clamp_min(1e-6)` 是否触发；
- 模型输出与检查张量的 dtype/device，以及 Python、Torch、CUDA、GPU 和
  commit；
- 独立目录 `workspace/J/derived/m0_recheck_rel1e-4_v1/`，若目录已存在则
  拒绝覆盖。

原 GPU 主机上的运行命令为：

```bash
cd /usr/local/ASU-unlearning
python workspace/J/diagnose_m0_recheck.py
```

当前控制端尝试通过 SSH host `audtodl_A800` 访问原主机，服务端返回
`Permission denied (publickey,password)`。因此本次尚未生成新的 forward
结果，不能声称旧误差已定位，也不能改变 NO-GO。

## 纯张量规格检查

[`split_channel_loss.py`](split_channel_loss.py) 仅实现候选公式供审阅和单测，
没有导入 trainer、没有配置入口、不会训练：

- 从排除 gold 的词表取第 K 大 Teacher logit，并减去正 `delta`；
- cap 和 gate 均 stop-gradient，cap 只在 `b_t>a_t` 时生效；
- Base/Student 都在排除 gold 后重新归一化；
- `rho` 在概率空间混合 Base 条件分布与完整非 gold 均匀分布；
- 非 gold 项固定为 `KL(r||q_S)`；
- `m_tar` 必须是 shifted answer mask 的子集，空 target 安全返回零；
- 当前没有事实 span，因此该模块未接入生产 loss。

本机使用 `/usr/bin/python3`、Torch 2.8.0 完成：

- `workspace/J/tests/test_m0_core.py`：5/5 通过；
- `workspace/J/tests/test_split_channel_loss.py`：7/7 通过，包括 float64
  手算 KL、shift/mask、Kth non-gold、单侧 cap 梯度、非 gold 对 gold
  logit 的零直接梯度、rho 混合和空 mask；
- `py_compile` 与 `git diff --check`：通过；
- IDE lint：无错误。

## 当前强制结论

**NO-GO，训练保持关闭。** 原因有两个独立硬门：

1. 旧 `1e-4` 数值门未通过，且本次尚未在原资产上完成复核；
2. 用户要求先完成可靠事实 span 标注，当前没有获批的 `m_tar`。

不得启动 ASU baseline 重训、候选 pilot、消融、多 seed 或 attack。后续只有在
原主机访问恢复并生成独立复核报告后，才能判断第一个门是否变化；即使数值门
通过，事实 span 门仍需单独审核。
