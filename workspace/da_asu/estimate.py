"""Wall-clock estimate from the 2026-10-08 Llama-2-7B replay.

The measured step is ASU+GD with ZeRO-3 parameter offload and the optimizer
kept on GPU (``config/ds_config/llama2.json``). Real-world Llama-3-8B uses
``llama3.json``, which also offloads the optimizer. Factors after the
measurement are planning multipliers, not new timings.
"""

MEASURED_SECONDS_PER_STEP = 33.08
MEASURED_NOTE = (
    "Llama-2-7B ASU+GD, 2 GPUs, microbatch 4, grad accum 4, seq 500, "
    "ZeRO-3 parameter offload, optimizer on GPU, 31 steps in 1025.5s"
)

# DeepSpeed elem count in that replay was 6.74B. Llama-3-8B is 8.03B.
PARAM_SCALE_8B = 8.03 / 6.74
# llama3.json offloads Adam as well as parameters.
OPTIM_OFFLOAD_FACTOR = 1.30
# Real-world sequences are capped at 300; the replay used 500. Offload dominates.
SEQ_FACTOR = 0.95
# ASU+KL adds the frozen retain forward on top of ASU+GD.
KL_FACTOR = 1.15
# Extra trainable D_reason forward+backward. Anchor also builds selected rows.
LOGIT_CONTROL_FACTOR = 1.65
ANCHOR_FACTOR = 1.75


def optimizer_steps(n_examples, epochs=5, batch=4, accum=4, gpus=2):
    micro = batch * accum * gpus
    if micro <= 0:
        raise ValueError("batch, accum, and gpus must be positive")
    return int(epochs * n_examples) // micro


def seconds_per_step(mode):
    base = MEASURED_SECONDS_PER_STEP * PARAM_SCALE_8B * OPTIM_OFFLOAD_FACTOR * SEQ_FACTOR
    if mode == "asu_gd":
        return base
    if mode == "asu_kl":
        return base * KL_FACTOR
    if mode == "logit_control":
        return base * KL_FACTOR * LOGIT_CONTROL_FACTOR
    if mode == "anchor":
        return base * KL_FACTOR * ANCHOR_FACTOR
    raise ValueError(mode)


def estimate_run(n_examples, mode, epochs=5, batch=4, accum=4, gpus=2):
    steps = optimizer_steps(n_examples, epochs, batch, accum, gpus)
    per_step = seconds_per_step(mode)
    return {
        "mode": mode,
        "n_examples": n_examples,
        "steps": steps,
        "seconds_per_step": per_step,
        "hours": steps * per_step / 3600.0,
    }


def format_report(example_counts=(200, 400, 1000)):
    lines = [
        MEASURED_NOTE,
        (
            f"Scaled Llama-3-8B step uses param x{PARAM_SCALE_8B:.2f}, "
            f"optim offload x{OPTIM_OFFLOAD_FACTOR:.2f}, seq x{SEQ_FACTOR:.2f}."
        ),
        "D_reason is one microbatch per step, so the step count stays the forget-set count.",
        "",
    ]
    for n_examples in example_counts:
        lines.append(f"forget set N={n_examples}, 5 epochs, 2 GPUs, batch 4, accum 4")
        for mode in ("asu_kl", "logit_control", "anchor"):
            row = estimate_run(n_examples, mode)
            lines.append(
                f"  {mode:16} {row['steps']:4d} steps  "
                f"{row['seconds_per_step']:.0f} s/step  {row['hours']:.2f} h"
            )
        lines.append("")
    lines.append("M0 NLL scan, GSM8K test 1319 x 7 temperatures, weights resident on one 80GB GPU: 15-40 min.")
    lines.append("M0b attention readout on that set, base plus one ASU student: about 30-60 min.")
    lines.append("Downstream lm-eval (MMLU, GSM8K, ARC-c, TruthfulQA) per checkpoint: about 4-8 h on 2x80GB.")
    return "\n".join(lines)


if __name__ == "__main__":
    print(format_report())
