# QKSpec / 对照方法结果表（RTF 口径）

> 数据来源：`/jizhicfs/chenyliao/runs/qkspec/rtf_results/*_rtf.json`
> 口径：Right-to-be-forgotten `metrics.py::get_eval_results`
> - FQ = Forget Efficacy（5 项 forget 指标调和平均；即 RTF 口径的 FE）
> - MU = Model Utility（18 项非 forget 指标调和平均）
> - 1−RL = 1 − Forget ROUGE-L recall
> - TR = Forget Truth Ratio（RTF 定义 `1 − mean(min(r,1/r))`，越大越忘）
> - 1−Prob = 1 − Forget Probability
> - 1−ES = 1 − Forget Entailment Score
>
> 注意：前两行为用户提供的样例（Llama-2-7B / forget05），其余为本机 QKSpec/对照结果（Llama-3.2-1B / forget10），基座与 split 不同，不可直接跨行比较绝对值。

| 基座名称 | SOTA | 代码链接 | 方法 | FQ↑ | MU↑ | 1−RL↑ | TR↑ | 1-Prob↑ | 1-ES↑ | 全称 | 年份/出处 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| LLaMA-2-7B-chat | — | ASU-unlearning | ASU_GD | 0.774（FE） | 0.740 | — | — | — | — | Attention Smoothing Unlearning（GD 变体） | 本机复现 ⚠️FE口径 \| RetainRG 0.720 / ForgetRG 0.402 |
| LLaMA-2-7B-chat | — | ASU-unlearning | ASU_KL | 0.813（FE） | 0.619 | — | — | — | — | Attention Smoothing Unlearning（KL 变体） | 本机复现 ⚠️FE口径 \| RetainRG 0.376 / ForgetRG 0.317 |
| Llama-3.2-1B-Instruct | — | ASU-unlearning | baseline_full | 0.389（FE） | 0.504 | 0.364 | 0.482 | 0.202 | 0.455 | TOFU full 微调模型（未遗忘基线） | 本机复现 ⚠️RTF口径 \| forget10 |
| Llama-3.2-1B-Instruct | — | ASU-unlearning | GA | 0.732（FE） | 0.398 | 0.688 | 0.348 | 0.724 | 0.695 | Gradient Ascent | 本机复现 ⚠️RTF口径 \| forget10 |
| Llama-3.2-1B-Instruct | — | QKSpec | qkspec_rho00_k32 | 0.682（FE） | 0.381 | 0.589 | 0.392 | 0.637 | 0.730 | QK Spectral Unlearning（强收缩 rho=0, top_k=32） | 本机复现 ⚠️RTF口径 \| forget10 |
| Llama-3.2-1B-Instruct | — | QKSpec | qkspec_cap10 | 0.512（FE） | 0.487 | 0.450 | 0.474 | 0.348 | 0.585 | QKSpec + max_retain_impact=0.10 | 本机复现 ⚠️RTF口径 \| forget10 |
| Llama-3.2-1B-Instruct | — | QKSpec | qkspec_spec_w0 | 0.541（FE） | 0.447 | 0.484 | 0.460 | 0.353 | 0.695 | QKSpec（specificity_weight=0，仅目标效应排序） | 本机复现 ⚠️RTF口径 \| forget10 |
| Llama-3.2-1B-Instruct | — | QKSpec | qkspec_spec_w1_lo | 0.394（FE） | 0.514 | 0.376 | 0.484 | 0.203 | 0.465 | QKSpec + 因果特异性筛选（stage_b_min=0.002） | 本机复现 ⚠️RTF口径 \| forget10 |

## 原始数据

| 方法 | FE | MU | MU_R | Forget ROUGE | Forget TR | Forget Prob | Forget ES |
|---|---|---|---|---|---|---|---|
| baseline_full | 0.3888 | 0.5036 | 0.7252 | 0.6362 | 0.4818 | 0.7978 | 0.5450 |
| GA | 0.7320 | 0.3984 | 0.4266 | 0.3124 | 0.3476 | 0.2763 | 0.3050 |
| rho00_k32 | 0.6817 | 0.3813 | 0.4181 | 0.4107 | 0.3920 | 0.3632 | 0.2700 |
| cap10 | 0.5123 | 0.4871 | 0.6011 | 0.5499 | 0.4742 | 0.6525 | 0.4150 |
| spec_w0 | 0.5411 | 0.4473 | 0.4922 | 0.5162 | 0.4599 | 0.6474 | 0.3050 |
| spec_w1_lo | 0.3935 | 0.5142 | 0.7345 | 0.6244 | 0.4835 | 0.7967 | 0.5350 |
