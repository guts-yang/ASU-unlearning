# Gap Map

_Field gaps with stable IDs._

## G4 — 激活目标特异性未证（target specificity）

**来源**：基线 `I/方向I_事实配对路线_重新设计Proposal_20261121.md` 阶段门 G4；证据调研 `方向I_反事实主体替换SubjectLast对齐去τ_证据调研.md` §8。

**缺口**：尚未证明真实 donor 主体配对激活目标优于范数匹配噪声、shuffle 主体与 matched-retain；未排除 s_f 答案下降其实由答成 donor s_r（答案替换）或一般置信度崩塌解释。基线 §2.2–2.3 的 ΔA 处理与 L_F 定义仍是参考方案、未定稿。

**对应 idea**：actv-unlearn-tricon（donor 负样本排除替换）、actv-unlearn-deltasubspace（多 donor 边际化）、actv-unlearn-deltaguide（不向 A⁻ 移动）。

## G5 — 渐进性收益未证（progressive schedule）

**来源**：基线阶段门 G5；证据调研 §7–8。

**缺口**：未预注册显式 schedule B(t)=启用层数/N，未在等训练 token/步数/可训练参数/算力下与静态全层、反向、随机 schedule 比较；"逐层取激活"不能被误称为时间上的"渐进"。

**对应 idea**：actv-unlearn-causaldelta（四 schedule 等预算消融）、actv-unlearn-deltasubspace（部分插值可结合 schedule）。
