# MCSD-ASU 协议（workspace/I）

状态：E0 规则已冻结。全量 Teacher 前向、Student 训练、鲁棒性评测的状态以 `e0_results.md` 与 `gates/e0.json` 为准。未出现在日志里的数字写「未运行」，缺失写「—」。

## 共同约束

1. 不改 `Right-to-be-forgotten`、`muse`、`open-unlearning`、`wmdp` 源码。Teacher 的 `τ_A` 走现有 `LlamaForCausalLM(..., attention_temp=, layers_id=)`；`τ_M` 只在本目录用实例包装施加。
2. Student 前向不加 `τ_A` / `τ_M`。E0 不做 Student 更新。
3. 首版损失边界：遗忘为输出级 `KL(p_T || p_S)`，保留为 ASU+GD 的答案 CE。不加 token-delta mask、真值置零重归一、表征对齐、梯度正交、拒答锚定。
4. 层数以加载后的 `config.num_hidden_layers` 为准。图示 16 层只对应 Llama-3.2-1B 说明，不用于本实验。
5. 校准集选温、定阈值；holdout 只用于判决。看过 holdout 后不得回改网格、容差或 cap。
6. 温度消融报告的是干预效应，不是「知识存放在该模块」。因果定位需要另做激活 patching 与随机模块对照，本版不做该主张。
7. `layers_id=null` 表示全部层。不得把空列表当成全层。

## 问题定义

- Spec A（本版）：全层同一 `τ_A`，全层同一 `τ_M`。
- Spec B（未启用）：按因果证据做稀疏模块。额外成本是定位标注、多重比较和把消融误写成定位。方向 H 的单层 Attention E0 已因 retain cap 无合格层而停止，不能外推成「MLP 无效」，也不能当成 Spec B 更优的证据。

停止 MLP 主张的条件：强度匹配后 T01 无稳定 forget 作用，且 T11 不优于 T10；或 T11 明显加重 retain、错误实体替代或连贯性。此时不进入 Student 训练。

## E0：Teacher 2×2

| ID | τ_A | τ_M |
|---|---|---|
| T00 | 1 | 1 |
| T10 | τ_A* | 1 |
| T01 | 1 | τ_M* |
| T11 | τ_A* | τ_M* |

数据：TOFU forget05 task 1 与长度匹配的 retain50。偶数作者块为校准，奇数作者块为 holdout。划分写入 `derived/e0_manifest.json` 后不再改。

强度（相对 T00，作者均值）：

- `S` = 答案 gold-token 平均 log-probability 的下降量（T00 − 条件）。下降为正。
- 同时报告答案 mean NLL 的上升量。
- retain cap = 校准集 T00 retain 作者 NLL 的 `2 × SE`。此 cap 在 T00 校准前向完成后才能算出，规则本身已预先写死。

选温（只用校准）：

1. `τ_A*`：网格中 retain ΔNLL 均值 ≤ cap，且 forget `S` 的 bootstrap 区间下界 > 0。多个合格时取**最小温度**。没有合格温度则 E0 失败，不进入 E1。
2. `τ_M*`：在同样的 retain 约束下，取使 `|S_01 − S_10| / max(|S_10|, 1e-6) ≤ 0.10` 且绝对差最小的温度。没有落在 10% 内的温度则记 `intensity_matched=false`，T11 不得当作强度匹配后的增益证据。
3. T11 使用这一对温度。若 T11 的 forget `S` 比 T10 高过 10%，额外记录「未匹配强度」，主结论不把更强扰动当成 MLP 增益。

交互：`I_AM = S_11 − S_10 − S_01 + S_00`。`I_AM>0` 表示同向超加，`<0` 表示次加。单独的正值不是成功。

还要在跑起来之后报告（本文件不填数字）：gold rank、答案抽取、错误实体替代率、生成重复/长度、相对 T00 的时延与峰值显存。bootstrap 按作者重采样。

判决（holdout，阈值来自上面的校准规则）：

- 通过：`intensity_matched=true`，且（T01 的 forget `S` 区间下界 > 0 且 retain ≤ cap，或 T11 的 forget `S` 不低于 T10 减去校准噪声且 retain ≤ cap），并且 T01 或 T11 相对 T10 的 forget 增量大于校准噪声。通过只表示可以申请 E1，不表示知识已被删除。
- 失败：匹配强度后 T01 无稳定作用且 T11 不优于 T10；或没有合格 `τ_A*`；或 T11 的 retain 超过 cap。失败则停止 MLP 主张。
- 模糊：校准与 holdout 方向不一致，或强度未匹配。只允许用同一 yaml 再跑一次复验。仍然模糊则不进入 E1。

## E1（仅 E0 通过并再次批准）

Student 为标准前向。遗忘 KL 为 `KL(p_T || p_S)`，mask 为 `labels != -100`，保留为 ASU+GD。比较 T10 Teacher（即现有 ASU）、T01 Teacher、T11 Teacher。同一基座、划分、学习率 `1e-5`、5 epoch、batch 4 × accum 4、全参、`forget_coeff=0.1`、`regularization_coeff=1.0`。不声称 Student 复制了 Teacher 的内部路径。

## 鲁棒性（仅 E1 完成并再次批准）

提示改写、固定预算再学习、非原模板探测、时延显存、同预算近邻比较。输出下降、内部路径变化、攻击后恢复分开写，三者都不等于「真正删除」。
