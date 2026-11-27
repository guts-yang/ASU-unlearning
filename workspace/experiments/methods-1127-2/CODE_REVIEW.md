# Code Review Status

日期：2026-11-27。范围：`workspace/experiments/methods-1127-2/`。

## 独立评审

`REVIEW_UNAVAILABLE`

当前会话未发现可调用的 Codex/研究评审后端；`.agent/skills/shared-references/reviewer-backends.md` 也不存在。未伪造跨模型评审、接受结论或独立诚信判定。本文件不表示代码已经通过外部评审。

## 执行者代码检查

已人工检查并修复以下实质问题：

- attention 路由/内容分量的 value 求和与 H/A/D 拼接维度。
- Llama GQA 的 KV value 头展开与 eager attention 要求。
- Tiny Teacher/Student 同初始化和 generate 参数传递。
- SwiGLU 转移矩阵跨样本/位置计数、源状态幅值维度和未观测状态屏蔽。
- Grassmann 子空间不变性自测语义与 full 档流式统计。
- Copula 可微 ECDF、null 尾目标和真实边缘分位数死区。
- 多路径主体坐标对齐、完备性检查与 wrong-pair 对照。
- Hankel 读出 logits、Copula 投影和路径签名的 bf16/float32 接口。
- 真实/慢 tokenizer 主体槽 offset 对齐，包括所有格问题。
- Student 非零有限梯度、冻结 Teacher、缓存契约和实测初始 NLL 一致性。
- relearning 只在预恢复 checkpoint 保存后执行。

详细机器验收以 `_validation/summary.json` 为准。真实 HF/PEFT/CUDA/TOFU 尚须集群 `smoke` 验证。
