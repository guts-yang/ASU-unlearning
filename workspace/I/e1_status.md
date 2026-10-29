# E1 状态

未运行。

[gates/e0.json](gates/e0.json) 的决定是 `do_not_start_e1`。Teacher 2×2 还没有 7B 前向，因此没有通过、失败或模糊的机制判决。按协议，这里不启动 Student，也不比较 ASU、Attention-only、MLP-only 和双通道。

若以后的 E0 写成 `eligible_for_e1_pending_approval`，仍要单独批准后才训练。训练时 Student 保持标准前向；遗忘 KL 与 ASU+GD 的接线放在 `workspace/I` 的调用脚本里，不改 `Right-to-be-forgotten` 源文件。
