# program.md — 收缩型故事自循环（固定，不修改）

> 本文件是本轮 autoresearch 的"研究组织代码"。循环中不得改动。
> 设计依据见 `workspace/shrink-story/STORY_DESIGN.md`。

## 目标

补上收缩型故事第 3 幕的**等预算缺口**：`low2high`（平均 τ=2.25）此前只与 `global τ=2` 比较，
而 τ=2.607 时 ΔF=1.7914 远高于 τ=2 时的 0.7998，说明 ΔF 对平均温度极敏感。
必须用相同平均 τ 的全局标量对照、时间顺序打乱对照、阶跃对照来分离"时间维度收益"与"强度收益"。

## 硬约束（适配版）

1. **只读**：`prepare.py`（含固定评估函数 `compute_regret`）、`program.md`、上游 `workspace/auto_experiments/common.py` 与全部既有结果文件。
2. **唯一可编辑代码**：`train.py`。每轮只改顶部 `CONFIG` 一个字典，提出一个单一、可解释的改动。
3. **禁止新增依赖**：只用已装的 torch / transformers / numpy。
4. **固定预算**：$K=60$ steps、$\alpha=0.1$、$\eta=5\times10^{-6}$、seed 0、TOFU forget10/retain90、Llama-3.2-1B。不使用 300 秒墙钟预算（本任务为验证而非调参）。
5. **单一指标**（越低越好）：`val_bpb` 列 = 等预算收缩遗憾
   \[ r(c)=[\Delta F_{\mathrm{ctrl}(\bar\tau(c))}-\Delta F_c]+2\max(0,\Delta R_c-\Delta R_{\mathrm{ctrl}})+0.02\,\kappa(c) \]
   定义在 `prepare.py::compute_regret`，循环期间不可修改。
6. **简单性优先**：$\kappa$ 计入指标；同等收益下更简单者胜出。
7. **日志不刷屏**：`python train.py > run.log 2>&1`，只 grep `^val_bpb:` / `^peak_vram_mb:`。
8. **`results.tsv` 不提交**：5 列、tab 分隔、保持 untracked。
9. **不使用 `git reset --hard`**：主工作区含用户既有产物；`discard` 用文件级回退（`git checkout <last_keep>:train.py`）。
10. **不使用 `uv run` / `val_bpb` / `autoresearch/<tag>` 主仓库分支**：循环在 `loop/` 独立 git 仓库内运行，隔离 commit 链。

## Run order（8 轮，跑完即停）

| 轮 | kind | CONFIG.schedule | $\bar\tau$ | $\kappa$ |
|---|---|---|---|---|
| A1 | anchor | global τ=2.25 | 2.25 | 0 |
| A2 | anchor | global τ=2.607 | 2.607 | 0 |
| C1 | candidate | linear_up 1.5→3 | 2.25 | 1 |
| C2 | candidate | linear_down 3→1.5 | 2.25 | 1 |
| C3 | candidate | random U(1.5,3) | 2.25 | 1 |
| C4 | candidate | step 30/30 | 2.25 | 1 |
| C5 | candidate | sens field | 2.607 | 2 |
| C6 | candidate | shuffled field | 2.607 | 2 |

## 每轮顺序

1. 记录 `<start_commit>`；确认 `train.py` 干净。
2. 只编辑 `train.py` 的 `CONFIG`。
3. `git add train.py && git commit -m "<单一改动描述>"`。
4. `python train.py > run.log 2>&1`。
5. `grep "^val_bpb:\|^peak_vram_mb:" run.log`；无输出视为 `crash`。
6. 追加 `results.tsv`（`val_bpb` = $r$ 或 crash 写 `0.000000`；`memory_gb` = peak/1024；status = `keep`/`discard`/`crash`）。
7. `r < 0` → `keep`（更新 `last_keep_commit`）；否则 `discard`，文件级回退到上一个 `keep` 版本。
8. anchor 轮固定 $r=0$、status=`keep`，但不参与竞标，不更新 best。

## 停止条件

`max_experiments = 8`。达到后写 `AUTORESEARCH_SUMMARY.md`，交接给 `anti-autoresearch` 做只读取证。
无 reviewer 时取证只能确定性-only，verdict 必为 `REVIEW_UNAVAILABLE`，不得声称 CLEAN。

## 诚实性

- 单 seed、60 steps、1B/TOFU 结论不得外推为通用结论。
- anchor 数字必须由本循环实跑产出，不得从既有结果硬编码复制（A2 与既有值一致只作为交叉校验）。
- `discard` 不是"造假"，是"主张收缩"；必须保留在账本与论文里。
