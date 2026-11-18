# 方向 I：当前主线与实施计划

> 版本：current-plan-2026-11-11  
> 状态：已完成 10 对 prompt-only 激活提取及 v0.3 固定前缀相对偏好 pilot；discovery 提名出 `S={8,9,13}`，但 confirmation 未通过，donor 来源核验仍 BLOCKED，尚未启动 Teacher/Student 训练。本文是 I 目录当前唯一的方向 I 实施口径。  
> 基座：`open-unlearning/tofu_Llama-2-7b-chat-hf_full`，使用该 checkpoint 准确 snapshot 配套 tokenizer。已记录 snapshot：`cc5b31c69127d5da881608e640f2b453c446435b`。

## 1. 当前进度

- 已构造独立候选目录，未修改 TOFU 原始文件或 manifest。远程 tokenizer 预检使用记录的 checkpoint snapshot、tokenizer 和 `[INST] {question} [/INST]` 模板，未 padding 或截断。
- 10 条 clean/donor prompt 的实体 span 连续；`y_plus` 可由目标 tokenizer 正确解码。模型配置为 32 层、hidden size 4096。
- A800 上 10 对 prompt-only 前向和激活导出成功；张量有限，同一 prompt 重复前向最大差为 0，clean/donor 激活有非零差异。记录见 `I/data/external_donor_prompt_activation_v0.3_activation/summary.md`。
- v0.3 固定前缀相对偏好 pilot 的读取、边界和 patch correctness 门通过；discovery 提名 `S={8,9,13}`。
- confirmation 未通过：4 个 confirmation 簇仅 2 个 `donor_specific` 为正，中位数 -0.0087，DID 中位数 -0.080；retain donor 平均效应大于 forget donor。自由生成类别计数未显示联合 `S` 改变分类。结果位于 `workspace/data/external_donor_prompt_activation_v0.3_preference/`。
- donor 来源门仍为 `BLOCKED`（Wikidata 请求 `URLError` errno 99）。Teacher/Student 训练尚未启动。

**当前结论：已证明激活提取链路可运行；尚无证据支持目标选择性局部路径。**固定前缀 (M) 仅作辅助相对偏好诊断，不能由其下降单独推出目标城市支持下降或遗忘。下一轮主读出改为原始问题下完整回答的目标事实出现率；旧结果保留为 pilot，不追溯重判。

## 2. 方法主线与实现顺序

研究问题：从关系匹配的外部 donor prompt 提取局部激活并 patch 到 TOFU 遗忘目标 prompt 后，能否降低完整生成中目标事实的出现率；该效应是否超过噪声/错配 donor，且 retain 影响更小？

方法链路：**配对数据与来源核验 → prompt-only 激活提取 → 冻结 anchor/layer 候选 → patch 正确性验证 → discovery 选集合 (S) → 独立 confirmation 做采样出现率与对照 → retain 选择性审计 → Teacher 质量审计 → gold/rollout 混合蒸馏（后续扩展）**。

本方向采用 prompt-only activation donor。clean 侧提供目标问题 (q^+)，donor 侧提供同关系、来源可追溯的问题 (q^-)。分别在冻结基座上得到 (a^+) 与 (a^-)。不把 donor 答案拼入 prompt，也不把 `donor_fact_value` 当训练标签；它用于来源审计和预注册的辅助 foil 分析。激活差异本身不证明目标事实因果路径。

## 3. 当前数据实例

候选数据位于 `I/data/external_donor_prompt_activation_v0.3/`：clean 来自官方 `I/data/forget05.jsonl`，从 200 条中抽取 10 个虚构作者簇的出生地 probe；donor 为 10 个唯一真实人物的出生地问题。`y_plus` 从原 QA 核验，`donor_fact_value` 计划依据 Wikidata QID/P19 核验。配对没有 `y_minus` 标签。Beyoncé 因名字碰撞未纳入，Muhammad Ali 和 Lionel Messi 为替代候选。

当前只有 10 个 clean 作者簇，属于工程 pilot。独立统计单位是作者/实体簇；同一问题的多次生成和随机种子是簇内重复测量，不能扩充独立样本量。

## 4. 定位和干预的建议实现

本节给出下一阶段的可执行方案。凡标注“建议”或“待冻结”的项目都不是已经完成的实验结论；应在远程模型实验前固定配置和判据。

### 4.1 激活读取与对齐

1. 固定相同 chat 模板，对 `q_plus` 与 `q_minus` 分别进行 prompt-only 前向，不拼接 donor 答案。
2. 粗定位首轮建议读取各层 residual stream，在问题内容的最后一个 token（出生地问题的问号）设共同 anchor。该 anchor 比绝对人名 token 下标更容易跨不同姓名长度对齐。
3. 先从少量样本核对 `hidden_states` tuple 长度和层号。常见 Hugging Face Llama 输出将 embedding 放在 `hidden_states[0]`，block `l` 输出可能对应 `[l+1]`；必须按远程实际模型实现确认，避免 off-by-one。
4. 记录 prompt/tokenizer/template 哈希、层索引、anchor 定义、激活 shape/dtype 和 cache key。相同输入重复前向应满足预先规定的数值容差。

### 4.2 目标读出：完整回答中的目标事实出现率

旧句首门不适配 TOFU 的完整句答案；固定答案前缀下的城市分数又会受人工前缀强烈条件化。因此，下一版以**原始问题下完整回答中的目标城市出现率**作为主行为读出。固定前缀 (M) 和 teacher-forced span 分数降为辅助诊断，不参与主信号门或 confirmation 通过判定。

对每个 `q_plus`，仅输入原始 prompt，不拼 `y_plus`、donor 城市或人为答案前缀。按冻结的随机解码配置从模型自回归采样完整回答，目标城市在回答任意位置出现都计为命中，不要求句首。重复 greedy 输出不是概率估计；若要估计出现概率，必须从 next-token 分布进行独立随机采样。

对第 (k) 条生成，令 (T_k=1) 表示出现目标城市或预冻结别名，(F_k=1) 表示出现 donor 城市或别名。对同一问题、同一条件采样 (K) 条，定义经验出现率与相对指标：

$$
p_{\text{target}} = \frac{1}{K}\sum_{k=1}^{K}T_k,
\qquad
p_{\text{foil}} = \frac{1}{K}\sum_{k=1}^{K}F_k,
$$

$$
M_{\text{appear}} = p_{\text{target}} - p_{\text{foil}}
= \frac{1}{K}\sum_{k=1}^{K}(T_k-F_k).
$$

目标与 foil 同时出现或都未出现时，该条对 `M_appear` 的贡献为 0；同时必须单列报告两种出现率，不能只看差值。逐回答分类为目标正确、明确错误事实、未提及、拒答、截断/不完整、目标与 foil 同现；类别互斥规则、城市别名和模糊匹配人工复核规则均在运行前冻结。

**采样与配对**：原始模型、真实 donor patch、matched-noise patch、shuffle-donor patch 使用相同 temperature、top-p、最大新 token 数、停止条件和采样数。每个问题按条件使用预先登记的配对 seed；若实现支持，可复用每步 uniform 随机数以降低条件比较方差。不得把同一问题的 `K` 条回答当成 `K` 个独立事实样本；先在簇内计算出现率，再以作者/实体簇为统计单位汇总并给出区间。

**Patch 生成实现**：对 (q^+) 做 prefill，在候选层 (l)、prompt anchor (t_a) 位置，将 clean residual 激活替换为 donor 对应 anchor 激活，或执行冻结的 ρ 插值：

$$
h^{\text{patch}}_{l,t_a}
=(1-\rho)h^{+}_{l,t_a}+\rho a^{-}_{l,t_a},
\qquad \rho\in[0,1].
$$

从干预后的下游状态生成 KV cache，随后正常自回归采样。若不用 cache，则每次完整前向都只在 prompt anchor 处施加同一 patch，答案位置不额外 patch。先验证 no-op/self-patch 生成分布一致、缓存与无缓存实现一致，再跑实验。

固定前缀下的 `M` 仍可用于机制诊断，但必须拆报目标答案与 foil 的分数变化；若 margin 下降仅由 foil 分数上升造成，不得表述为目标城市支持下降。teacher-forced 指标不替代原始问题下的完整生成出现率。

### 4.3 局部路径选择

建议先进行 residual-stream 粗定位：候选单元为“层 × anchor 位置”，在冻结模型上将 donor 激活 patch 到 recipient 的对应位置，逐层或用有预算的稀疏 mask 提名小集合 `S`。首轮不同时搜索所有 attention heads、MLP 单元和 token 位置，以控制搜索空间与工程复杂度。

patch 必须使用实际前向 hook/editor；`output_hidden_states=True` 只用于读取，不会自动把 `a_minus` 写入 recipient。实现后先验证 self-patch/no-op 与原始前向一致，再检查缓存激活与在线 donor 激活一致。后续若 residual 粗定位有稳定信号，才另开细化阶段定位到 attention head 或 MLP 输出。

## 5. 定位环节的失败检测和对照

逐门记录 `PASS / FAIL / BLOCKED`，禁止把环境阻断写成模型拒绝或科学假设失败。

1. **目标可提取性基线**：在未干预基座上按第 4.2 节估计 `p_target`、`p_foil` 和 `M_appear`，并报告完整回答类别。出现率低或簇内方差过大时，该样本不能提供有力的 patch 可提取性证据；不得以固定答案前缀高分替代。解码、采样数 `K`、别名、类别规则和最低可分析标准须在重跑前冻结；句首格式不作门槛。
2. **数据/来源门**：核验 clean 源行、donor QID/P19、城市级粒度及远程最终评测清单中的人物、别名和近邻实体。
3. **读取门**：层号、token anchor、span、shape/dtype 正确；重复计算可复现。
4. **patch 正确性门**：self-patch/no-op 与未干预前向一致；在线 donor 与缓存 donor 结果一致；recipient 下游按干预状态重新计算。
5. **噪声负对照**：在与真实 donor 完全相同的层、位置、recipient 和替换强度下，另做范数匹配噪声 patch。对 recipient 激活 (h) 和真实 donor 激活 (a^-)，定义位移范数 (r) 与 null 激活 (h_{\text{null}})：

   $$
   r=\lVert a^- - h\rVert_2,
   \qquad
   h_{\text{null}}=h+r\frac{\epsilon}{\lVert\epsilon\rVert_2},
   \qquad \epsilon\sim\mathcal{N}(0,I).
   $$

   每个样本使用多个预先固定的随机种子；噪声分布、种子数和数值容差在看主结果前冻结。比较真实 donor 的目标事实出现率下降与噪声 patch 的对应下降。该对照检验“效果是否超过同等范数的随机方向扰动”，不证明 donor 关系语义本身是原因；范数匹配也不保证噪声在模型自然激活分布内，因此不得单独据此作强因果结论。
6. **自然激活辅助对照**：将同关系 donor 在 clean 簇之间打乱配对，使用同一层、位置和 patch 强度重复实验。它检验特定 donor—clean 配对是否重要；若研究主张只是“任意同关系 donor 都有效”，打乱后仍有效并不构成反证。若有可用且格式匹配的其他关系 prompt，可另作自然、无关关系 donor 对照，检查任意外部人物 prompt 激活是否都会造成影响。上述自然对照与噪声对照回答不同问题，不互相替代。
7. **保留选择性审计**：只读地从未参与遗忘训练的 retain probe 中选取匹配出生地关系/模板的问题；不改动 retain 原文件、manifest 或评测切分。对每个遗忘和保留问题分别做原始、真实 donor patch、噪声 patch、shuffle donor 条件的完整回答采样，再按相同协议计算目标事实出现率下降：

   $$
   \Delta_{g,c}
   =\operatorname{mean}_{\text{clusters}}
   \left(p_{\text{target,base}}-p_{\text{target,patch}(c)}\right).
   $$

   其中 `g` 为 forget 或 retain，`c` 为 donor 或 noise。先报告全部四项，再报告：

   $$
   D_{\text{forget}}
   =\Delta_{\text{forget,donor}}-\Delta_{\text{forget,noise}}.
   $$

   $$
   \operatorname{DID}
   =\left(\Delta_{\text{forget,donor}}-\Delta_{\text{forget,noise}}\right)
   -\left(\Delta_{\text{retain,donor}}-\Delta_{\text{retain,noise}}\right).
   $$

   `selectivity_DID` 是差中差：先扣除随机噪声造成的一般扰动，再比较剩余 donor 效应在遗忘问题与保留问题之间的差异。它为正且跨独立簇稳定，才支持目标选择性；仍需单独检查保留集下降、拒答、错误实体和一般能力，不能只报告一个差值。shuffle donor 作为另一个独立对照单列，不与 noise 合并。
8. **独立确认门**：在未参与 mask/路径选择的 clean 作者簇和 donor 簇上验证冻结的 `S`。当前只有 10 个 clean 作者簇，应将本轮定位结论限为工程 pilot / exploratory，不作强统计确认；以作者/实体簇为统计单位，不能把同簇多个 probe 或噪声种子当成独立样本。

若 donor 效应没有超过噪声对照，或差中差不为正/在独立簇上不稳定，或保留和一般能力同等退化，结论是尚无证据支持目标选择性局部路径；不能称为遗忘成功。噪声对照结果要连同噪声尺度、随机种子和可能的分布外影响报告。

## 6. 定位通过后的教师/学生阶段（待验证扩展）

当前还没有教师信号、学生损失或训练配置的实验结果。定位与 retain 审计通过后，按以下**序列级混合前缀蒸馏**做小规模验证。gold/rollout 两路、比例 `β`、stop-gradient 和整序列抽样规则参考项目内 [ASU-OP 实现规格](/Users/chenyliao/MuRAG/Papers/L7_ASU创新/K1/K1_ASU-OP_实现规格与预实验方案.md)；方向 I 将其中的标准 ASU 平滑 Teacher 替换为经因果定位检验的路径条件 Teacher。这是待验证扩展，不是当前已验证结论。

### 6.1 路径条件 Teacher

1. 冻结原始基座权重作为教师底座。在对应 forget prompt `q_plus` 的前向中，只在已确认集合 `S` 上 patch 其配对 donor 激活 `a_minus`，或使用预注册的插值强度；不输入 `y_minus`，也不以 donor 答案作标签。
2. 对同一 `q_plus` 和给定续写前缀，Teacher 输出全词表 next-token 分布 `q_T^S(· | q_plus, prefix)`。Teacher 权重与 patch/cache 固定且无梯度；每次评分都在同一条件前缀下与 Student 对比。
3. 训练前分别用 gold 和 Student rollout 前缀探测 Teacher：检查 `y_plus` 及别名支持、Teacher entropy、donor 答案是否异常升高、输出可读性、拒答和重复。若 Teacher 在 rollout 前缀上仍高置信支持遗忘事实，或分布已经退化，不进入 Student 蒸馏。

### 6.2 Gold 与 Student rollout 的序列级混合

对每个 forget prompt `q`，准备两种**完整续写序列来源**：

- **Gold 分支**：使用数据中的目标答案 `y_plus`。Teacher 与 Student 都在同一完整序列 `q + y_plus` 上做 causal forward；预测 `y_t` 时只条件化于 `q + y_plus_<t`。
- **Student rollout 分支**：从当前 Student（生成时 stop-gradient）仅以 `q` 为输入自回归生成完整序列 `ŷ`，在 EOS 或冻结的最大长度处停止。随后把同一 `q + ŷ` 分别送入路径条件 Teacher 和可训练 Student，比较相同前缀下的分布；Student 评分 logits 保留梯度，采样 token 不反传。

用 \(\beta\) 表示选择 Student rollout 完整序列的比例，forget 目标为：

$$
\mathcal{L}_{\text{forget}}(\theta;\beta)
=(1-\beta)\,\mathbb{E}_{(q,y^+)}
\left[\ell_{\text{TF}}(q,y^+;\theta)\right]
+\beta\,\mathbb{E}_{q,\,\hat y\sim\pi_{\theta_{\text{snap}}}^{d}}
\left[\ell_{\text{OP}}(q,\hat y;\theta)\right].
$$

两路损失都是有效续写 token 上的平均前向 KL：

$$
\ell(q,z;\theta)
=\frac{1}{N_z}\sum_{t\in\mathcal{M}_z}
\mathrm{KL}\!\left(
q_T^S(\cdot\mid q,z_{<t})
\,\middle\|\,
p_\theta(\cdot\mid q,z_{<t})
\right),
$$

其中 \(\mathcal{M}_z\) 是有效 completion 预测位置集合，\(N_z=|\mathcal{M}_z|\)。

prompt 和 padding 位不计入 KL；EOS 是否计入、最大续写长度和每序列归一化规则在实验前冻结。gold 与 rollout 是两种**整条序列级**分支；不逐 token 抽签混接答案和生成 token，以免造出 Student 实际生成中不可达的混合前缀。

小规模 pilot 的预注册比较建议为 `β ∈ {0, 0.5, 1}`：`0` 是全 gold-prefix 路径条件蒸馏，`0.5` 是混合前缀主候选，`1` 是全 Student-rollout 前缀。`β=0.5` 是待测比例，不是理论最优值。若结果显示对比例敏感，再在独立校准集上考虑 `0.25/0.75`；不得在封存确认集上挑选最佳比例。

实现时可对每条 forget 序列独立做 `Bernoulli(β)` 分支选择，再对样本损失求平均；若按 batch 固定配额抽取两路，则需显式按 `(1−β)` 和 `β` 加权、预先定义空分支处理，并分别记录两路样本数和有效 token 数。rollout 使用的 Student 参数快照、解码设置 `d`（greedy 或 temperature/top-p）、刷新频率、EOS/长度截断率、输出长度与重复率必须记录并冻结。初始 pilot 可每个 forget batch 用当前 Student 生成；任何缓存旧 rollout 的做法都需明确 snapshot 与刷新间隔。

### 6.3 Retain 约束与总目标

首版只在 forget 分支混合 gold/rollout 前缀，retain 侧保持固定的基座参考目标，避免同时改变两侧的 prefix source：

$$
\mathcal{L}_{\text{retain}}(\theta)
=\mathbb{E}_{(r,y_r)}\left[
\frac{1}{N_r}\sum_{t\in\mathcal{M}_r}
\mathrm{KL}\!\left(
p_{\text{base}}(\cdot\mid r,y_{r,<t})
\,\middle\|\,
p_{\theta}(\cdot\mid r,y_{r,<t})
\right)\right].
$$

$$
\mathcal{L}_{\text{total}}(\theta;\beta)
=\lambda_F\mathcal{L}_{\text{forget}}(\theta;\beta)
+\lambda_R\mathcal{L}_{\text{retain}}(\theta).
$$

`λ_F/λ_R`、是否沿用现有 ASU retain loss 的确切形式、KL reduction、有效 token mask 和训练步数均需在训练前依据 ASU 实际代码核对并冻结，不能只用“KL 加权”概括。学生从未 patch 的基座权重初始化；学生自身前向不施加 donor hook。训练结束后在正常前向下评估，不依赖推理时 patch。

**样本覆盖限制**：当前路径 donor 数据只有 10 个 clean 作者簇与对应的 10 个 `q_plus/q_minus` 候选。因而第一轮只能把混合前缀蒸馏称为 10 簇 pilot。若要对官方 `forget05` 更广范围训练，必须先为每个纳入的 forget prompt 定义并验证 donor 激活映射，不能把这 10 组激活默认广播到全部 forget prompts。

### 6.4 训练前与训练后门槛

训练前比较 gold-prefix 与 rollout-prefix 上 Teacher 的遗忘答案/别名概率、熵、输出质量及泄漏率。若路径条件 Teacher 在 Student rollout 前缀上会重新支持 `y_plus`、普遍高熵或产生不可读分布，停止该训练支路，不靠事后调高 `β` 或训练权重挽救。

通过后，以 `β=0` 的 gold-prefix 路径 Teacher 训练作为基线，与 `β=0.5` 混合和 `β=1` rollout 分支比较；除 prefix source / `β` 外固定模型、`S`、donor map、Teacher、数据、retain loss、优化器与预算。报告实际更新数、有效 Student token、Teacher-scored token、rollout 生成成本和墙钟/显存，不把同 epoch 当成等算力。训练 KL 下降仅表示 Student 更接近 Teacher，不等于遗忘成功；仍需独立检验目标事实恢复、retain、自由生成质量和跨提示泛化。

这是一项不同于原始 ASU 全局注意力平滑教师的扩展：ASU 用注意力温度构造 forget teacher；这里拟测试的是把经定位验证的局部激活 patch 用作教师干预。其收益和失败模式都需要实测，不能由定位结果直接推出。

## 7. 后续阶段顺序

1. 将本文件和数据目录一同传至 GPU 服务器，固定 checkpoint/tokenizer/template 与代码版本。
2. 完成来源、远程评测去重和 hidden-state 层/位置核验。
3. 运行少量 prompt-only `a_plus/a_minus` smoke test，并验证重复前向稳定性。
4. 冻结随机解码配置、每问题采样数 `K`、目标/foil 城市别名表、完整回答分类规则、出现率主指标、anchor、候选路径、patch 预算、matched-noise/shuffle donor 对照和簇级统计门槛。固定前缀分数只列为辅助诊断。
5. 实现 hook/editor，完成 no-op 和 patch correctness 检查。
6. 用 discovery 簇按预注册的目标事实出现率提名并冻结候选集合；在此前未参与选择的独立 confirmation 簇上比较 base、donor、noise、shuffle 条件的完整回答出现率，完成 retain 选择性审计。若样本量/簇数不足，只报告探索性结果，不作确认性主张。
7. 定位与主动 retain 审计过门后，先检查路径 Teacher 在 gold/rollout 两种前缀上的质量；再冻结 `β`、序列级采样、rollout 解码/刷新、loss mask 和预算；Teacher 质量过门后，才启动 `β=0/0.5/1` 的小规模 Student 对照。不得直接扩大至全部 forget05 或自动开始正式训练。

## 8. 术语

- **span**：文本或 token 序列中一段连续范围，如完整人名 token span。
- **patch**：某次前向中，将 donor 激活替换到 recipient 指定层/位置的干预；不修改模型权重。
- **padding**：为批处理而补入的特殊 token。本候选预检未使用 padding 或 truncation。
- **K2**：旧 v0.2 的 donor top-1 已知性门；当前 prompt-only 方案不将其作为必过门。若报告模型“已知” donor 事实，须另做并单独报告诊断。
- **噪声负对照**：在目标激活上加入与 donor 替换位移范数匹配的随机向量，用于估计一般随机扰动效应；它不是自然 donor，也不能独立证明语义特异性。
- **选择性差中差**：比较 donor 相对噪声的额外效应在 forget 与 retain probes 上的差异；始终同时报告各组原始效应，防止汇总差值掩盖广泛损伤。
- **序列级混合前缀**：每条 forget 样本整条选择 gold 答案或 Student rollout 作为 Teacher/Student 共同评分前缀；`β` 是 rollout 序列比例，不是逐 token 混合概率。
