# 方向 I：当前主线与实施计划

> 版本：current-plan-2026-11-11  
> 状态：研究方案与候选数据已形成；激活定位和模型干预尚未运行。本文是 I 目录当前唯一的方向 I 实施口径。  
> 基座：`open-unlearning/tofu_Llama-2-7b-chat-hf_full`，使用该 checkpoint 准确 snapshot 配套 tokenizer。已记录 snapshot：`cc5b31c69127d5da881608e640f2b453c446435b`。

## 1. 研究主线

检验外部 donor prompt 产生的局部激活，能否干预 TOFU 遗忘目标问题的计算路径，选择性降低模型对目标事实的支持，同时较少影响保留知识和一般能力。

本方向采用 **prompt-only activation donor**：clean 侧提供目标问题 `q_plus` 和目标答案 `y_plus`；donor 侧提供关系相同、来源可追溯的问题 `q_minus`。分别在冻结基座上运行两个问题，取得 `a_plus` 和 `a_minus`。`y_minus` 不需要，不输入模型，也不参与损失或评分。

`a_plus - a_minus` 只是激活差异，不能单独证明差异来自待遗忘事实。目标性必须通过目标答案读出、激活干预、负对照和 retain 审计共同建立。

## 2. 当前数据实例

当前候选数据位于 `I/data/external_donor_prompt_activation_v0.3/`：

- clean 来源：官方 `I/data/forget05.jsonl`，200 条中抽取 10 个虚构作者簇的出生地 probe。
- clean 字段：`q_plus` 是统一问题模板，`y_plus` 是从原 QA 核验的出生城市；保留源行、原问答及源文件 SHA-256。
- donor 字段：10 个唯一真实人物，关系为出生地；`q_minus` 与 clean 侧使用同一问题模板。`donor_fact_value` 仅用于来源核验和关系审计，不作为模型目标。
- 配对文件没有 `y_minus`。Beyoncé 因远程扫描发现与虚构作者名字碰撞而未纳入；Muhammad Ali 和 Lionel Messi 为替代候选。
- 远程预检报告显示格式、clean 来源行、token 解码及其所扫描评测文件中的人物/事实簇去重通过。城市字符串命中单独记录，不直接视为实体泄漏。

这些数据目前是小规模工程候选，不足以支持强确认性统计结论；独立样本单位是作者/实体簇，不能把同一簇的多个 probe 当作独立样本。

## 3. 已做工作与当前状态

### 已完成

1. 构造独立候选目录，未修改 TOFU 原始文件或 manifest。
2. 远程 tokenizer 预检使用记录的 checkpoint snapshot、tokenizer 和模板 `[INST] {question} [/INST]`，未 padding 或截断。
3. 10 条 clean 与 donor 事实值不同；`y_plus` 能按目标 tokenizer 正确解码；两侧人名 token span 均连续。
4. 按远程预检提供的代码审计，`LlamaModel.forward(output_hidden_states=True)` 可返回各层 residual hidden states；模型配置为 32 层、hidden size 4096。
5. 新增 GPU 前就绪记录：`I/data/external_donor_prompt_activation_v0.3/pre_gpu_readiness.md`。

### 尚未完成

- 尚未在 GPU 上加载权重或计算 `a_plus / a_minus`；没有激活数值、dtype、重复运行稳定性结果。
- donor QID/P19 在线逐条核验因远程服务器网络不可用而待完成。
- 尚未实现或验证激活 patch hook；hidden-state 可读取不等于可干预。
- 尚未冻结激活位置、候选路径范围、局部集合选择算法、`y_plus` span 分数口径和统计门槛。
- 尚未运行定位对照、forget/retain 行为评测或 Teacher/Student 训练。

## 4. 定位和干预的建议实现

本节给出下一阶段的可执行方案。凡标注“建议”或“待冻结”的项目都不是已经完成的实验结论；应在远程模型实验前固定配置和判据。

### 4.1 激活读取与对齐

1. 固定相同 chat 模板，对 `q_plus` 与 `q_minus` 分别进行 prompt-only 前向，不拼接 donor 答案。
2. 粗定位首轮建议读取各层 residual stream，在问题内容的最后一个 token（出生地问题的问号）设共同 anchor。该 anchor 比绝对人名 token 下标更容易跨不同姓名长度对齐。
3. 先从少量样本核对 `hidden_states` tuple 长度和层号。常见 Hugging Face Llama 输出将 embedding 放在 `hidden_states[0]`，block `l` 输出可能对应 `[l+1]`；必须按远程实际模型实现确认，避免 off-by-one。
4. 记录 prompt/tokenizer/template 哈希、层索引、anchor 定义、激活 shape/dtype 和 cache key。相同输入重复前向应满足预先规定的数值容差。

### 4.2 目标读出

首版用 `y_plus` 的 teacher-forced 答案 span 平均条件 log-prob 作为主读出：

`score(q, y) = mean_t log p(y_t | q, y_<t)`

`score` 是从模型 next-token 概率分布中计算出的**单个标量分数**，不是完整输出分布，也不是整段答案的概率乘积。只计答案 token，不计 padding；EOS 是否计入在实验前冻结。目标干预效应定义为：

`delta_target(S) = score(q_plus, y_plus | clean forward) - score(q_plus, y_plus | donor-patched forward using S)`

正值表示 donor patch 降低目标答案支持度；它本身不等于遗忘成功。生成文本、拒答和错误实体率作为补充行为指标，不能替代这个冻结的主读出。

### 4.3 局部路径选择

建议先进行 residual-stream 粗定位：候选单元为“层 × anchor 位置”，在冻结模型上将 donor 激活 patch 到 recipient 的对应位置，逐层或用有预算的稀疏 mask 提名小集合 `S`。首轮不同时搜索所有 attention heads、MLP 单元和 token 位置，以控制搜索空间与工程复杂度。

patch 必须使用实际前向 hook/editor；`output_hidden_states=True` 只用于读取，不会自动把 `a_minus` 写入 recipient。实现后先验证 self-patch/no-op 与原始前向一致，再检查缓存激活与在线 donor 激活一致。后续若 residual 粗定位有稳定信号，才另开细化阶段定位到 attention head 或 MLP 输出。

## 5. 定位环节的失败检测和对照

逐门记录 `PASS / FAIL / BLOCKED`，禁止把环境阻断写成模型拒绝或科学假设失败。

1. **目标信号门**：原始 `q_plus` 对 `y_plus` 有可测支持；若基座本身不支持目标答案，该样本不能用于解释“目标支持被降低”。
2. **数据/来源门**：核验 clean 源行、donor QID/P19、城市级粒度及远程最终评测清单中的人物、别名和近邻实体。
3. **读取门**：层号、token anchor、span、shape/dtype 正确；重复计算可复现。
4. **patch 正确性门**：self-patch/no-op 与未干预前向一致；在线 donor 与缓存 donor 结果一致；recipient 下游按干预状态重新计算。
5. **噪声负对照**：在与真实 donor 完全相同的层、位置、recipient 和替换强度下，另做范数匹配噪声 patch。对 recipient 激活 `h` 和真实 donor 激活 `a_minus`，令 `r = ||a_minus - h||_2`，构造 `h_null = h + r * epsilon / ||epsilon||_2`，其中 `epsilon` 为零均值高斯向量。每个样本使用多个预先固定的随机种子；噪声分布、种子数和数值容差在看主结果前冻结。比较真实 donor 的 `delta_target` 与噪声 patch 的平均 `delta_target`。该对照检验“效果是否超过同等范数的随机方向扰动”，不证明 donor 关系语义本身是原因；范数匹配也不保证噪声在模型自然激活分布内，因此不得单独据此作强因果结论。
6. **自然激活辅助对照**：将同关系 donor 在 clean 簇之间打乱配对，使用同一层、位置和 patch 强度重复实验。它检验特定 donor—clean 配对是否重要；若研究主张只是“任意同关系 donor 都有效”，打乱后仍有效并不构成反证。若有可用且格式匹配的其他关系 prompt，可另作自然、无关关系 donor 对照，检查任意外部人物 prompt 激活是否都会造成影响。上述自然对照与噪声对照回答不同问题，不互相替代。
7. **保留选择性审计**：只读地从未参与遗忘训练的 retain probe 中选取匹配出生地关系/模板的问题；不改动 retain 原文件、manifest 或评测切分。对每个遗忘和保留问题分别做原始、真实 donor patch、噪声 patch 前向，再按同一 `score` 计算各自的干预下降：

   `delta[g, c] = mean_cluster(score_base - score_patch)`

   其中 `g` 为 forget 或 retain，`c` 为 donor 或 noise。先报告全部四项，再报告：

   `donor_specific_forget = delta[forget, donor] - delta[forget, noise]`

   `selectivity_DID = (delta[forget, donor] - delta[forget, noise]) - (delta[retain, donor] - delta[retain, noise])`

   `selectivity_DID` 是差中差：先扣除随机噪声造成的一般扰动，再比较剩余 donor 效应在遗忘问题与保留问题之间的差异。它为正且跨独立簇稳定，才支持目标选择性；仍需单独检查保留集下降、拒答、错误实体和一般能力，不能只报告一个差值。
8. **独立确认门**：在未参与 mask/路径选择的 clean 作者簇和 donor 簇上验证冻结的 `S`。当前只有 10 个 clean 作者簇，应将本轮定位结论限为工程 pilot / exploratory，不作强统计确认；以作者/实体簇为统计单位，不能把同簇多个 probe 或噪声种子当成独立样本。

若 donor 效应没有超过噪声对照，或差中差不为正/在独立簇上不稳定，或保留和一般能力同等退化，结论是尚无证据支持目标选择性局部路径；不能称为遗忘成功。噪声对照结果要连同噪声尺度、随机种子和可能的分布外影响报告。

## 6. 定位通过后的教师/学生阶段（待验证扩展）

当前还没有教师信号、学生损失或训练配置的实验结果。定位与 retain 审计通过后，可评估以下**路径条件教师**，但不能把它当作已冻结的方案：

1. 冻结原始基座权重作为教师底座。在 forget prompt `q_plus` 的前向中，只在已确认集合 `S` 上 patch `a_minus`，或使用预注册的插值强度；不输入 `y_minus`，也不以 donor 答案作标签。
2. 对 `q_plus` 加目标答案前缀做 teacher-forced 前向，缓存教师每个 token 位置的全词表 next-token 分布。确认教师降低 `y_plus` 支持，同时没有异常提升 donor 答案、显著提高拒答或让函数 token/输出质量整体崩坏；否则该教师不合格，不进入蒸馏。
3. 若教师通过质量门，学生从未干预的基座初始化。在 forget prompts 上匹配路径条件教师分布；在 retain prompts 上用基座分布约束学生漂移。可先评估 `L_forget = KL(p_teacher || p_student)` 与 `L_retain = KL(p_base || p_student)` 的加权目标，具体 KL 方向、token mask、权重和训练步数需在训练前另行冻结。
4. 学生训练时不使用 donor 答案监督，也不需要在学生推理时保留 patch hook；训练结束后须在正常前向下重新评估 forget、retain、改写问题、拒答、错误实体和一般能力。

这是一项不同于原始 ASU 全局注意力平滑教师的扩展：ASU 用注意力温度构造 forget teacher；这里拟测试的是把经定位验证的局部激活 patch 用作教师干预。其收益和失败模式都需要实测，不能由定位结果直接推出。

## 7. 后续阶段顺序

1. 将本文件和数据目录一同传至 GPU 服务器，固定 checkpoint/tokenizer/template 与代码版本。
2. 完成来源、远程评测去重和 hidden-state 层/位置核验。
3. 运行少量 prompt-only `a_plus/a_minus` smoke test，并验证重复前向稳定性。
4. 冻结 `y_plus` 标量 score、anchor、候选路径、patch 预算、范数匹配噪声生成/种子、自然激活辅助对照与 forget/retain 统计门槛。
5. 实现 hook/editor，完成 no-op 和 patch correctness 检查。
6. 做 discovery 定位，在独立簇上确认冻结的候选集合，并完成 donor、噪声、retain 和失败检测。
7. 定位与主动 retain 审计过门后，先审查路径条件教师分布；教师质量过门后，才另行冻结并启动 Student/ASU 对照训练。不得自动开始训练。

## 8. 术语

- **span**：文本或 token 序列中一段连续范围，如完整人名 token span。
- **patch**：某次前向中，将 donor 激活替换到 recipient 指定层/位置的干预；不修改模型权重。
- **padding**：为批处理而补入的特殊 token。本候选预检未使用 padding 或 truncation。
- **K2**：旧 v0.2 的 donor top-1 已知性门；当前 prompt-only 方案不将其作为必过门。若报告模型“已知” donor 事实，须另做并单独报告诊断。
- **噪声负对照**：在目标激活上加入与 donor 替换位移范数匹配的随机向量，用于估计一般随机扰动效应；它不是自然 donor，也不能独立证明语义特异性。
- **选择性差中差**：比较 donor 相对噪声的额外效应在 forget 与 retain probes 上的差异；始终同时报告各组原始效应，防止汇总差值掩盖广泛损伤。
