# CF-04 v2 Optimization and ASU Comparison

- claim_level: `false`
- evidence_gate: `NOT_APPLIED`
- model: `Llama-3.2-1B-Instruct`, fp32, eager attention
- data: TOFU `forget10=400 / retain90=3600 / holdout10=400`
- hardware: one NVIDIA H20 per run
- evaluation: full split, eval batch 8
- seeds: sweep seed 0; finalists seeds 0/1/2
- reviewer: Hy4 manual review attempted; final review status is recorded separately

## Metric Definitions

- `dF`: after forget NLL minus before; higher is better.
- `dR`: after retain NLL minus before; lower is better; hard constraint `dR<=0.05`.
- `dH`: after holdout NLL minus before; lower is better; soft target `dH<=0.10`.
- `retain-safe`: `dF>=0.30` and `dR<=0.05`.
- `strong`: `dF>=0.90` and `dR<=0.05`.
- stability: mean ± population std over 3 seeds.

## v2 Changes

1. Multi-layer probes at layers `[8,12,16]` (or `[12,16]`).
2. Margin-based fact loss (`softplus(fact_score + margin)`) instead of BCE; BCE saturated at zero in c11.
3. Adaptive retain multiplier: increases when batch retain NLL exceeds target, decays otherwise.
4. Optional PCGrad-style gradient projection (tested; unstable and rejected).

## 300-Step v2 Sweep (seed 0)

| Config | Layers | Margin | lambda init | Grad proj | dF | dR | dH | Retain-safe | Strong |
|---|---|---|---:|---|---:|---:|---:|---|---|
| v2a | 12,16 | 0.5 | 1 | no | +1.2301 | +0.0325 | +0.1464 | yes | yes |
| v2b | 8,12,16 | 0.5 | 1 | no | +1.6391 | +0.0338 | +0.1501 | yes | yes |
| v2c | 12,16 | 1.0 | 1 | no | +1.4903 | +0.0277 | +0.1820 | yes | yes |
| v2d | 12,16 | 0.5 | 3 | no | +1.0687 | +0.0216 | +0.1257 | yes | yes |
| v2e | 12,16 | 1.0 | 3 | no | +1.1333 | +0.0204 | +0.1277 | yes | yes |
| v2f | 12,16 | 0.5 | 1 | yes | +30.1753 | +37.6997 | +38.1957 | no | no |
| v2g | 8,12,16 | 1.0 | 3 | no | **+2.4152** | **+0.0222** | **+0.1874** | **yes** | **yes** |
| v2h | 12,16 | 1.0 | 1 | yes | +25.9881 | +50.7619 | +48.2890 | no | no |

- All non-projection v2 configs are retain-safe and strong at 300 steps.
- Gradient projection is unstable and is rejected.
- Best forget under hard retain constraint: `v2g`.

## 3-Seed Stability

| Config | dF | dR | dH | Retain-safe | Strong |
|---|---:|---:|---:|---|---|
| v2b | +1.4797 ± 0.1600 | +0.0289 ± 0.0039 | +0.1566 ± 0.0150 | yes | yes |
| **v2g** | **+2.3084 ± 0.0915** | **+0.0180 ± 0.0031** | +0.1754 ± 0.0131 | yes | yes |

`v2g` is the selected best config: layers `8,12,16`, margin `1.0`, lambda init `3`, no gradient projection, LR `6e-6`, identity weight `2.0`, probe-retain weight `2.5`.

## Formal 10-Epoch v2g

| Metric | Value |
|---|---:|
| dF forget | **+8.0920** |
| dR retain | **-0.4561** |
| dH holdout | **+0.9656** |
| epochs completed | 10/10 |
| early stopped | no |
| checkpoint | `/jizhicfs/chenyliao/llm-unlearning/checkpoints/full_cf04_v2_formal` |

The 10-epoch run is very strong on forget/retain, but holdout degradation `+0.9656` is far above the `0.10` soft target and triggers the holdout-risk stop criterion.

## Comparison with ASU

| Method | Budget | dF | dR | dH | Retain-safe |
|---|---|---:|---:|---:|---|
| CF-04 c11 | 300 steps, 3 seeds | +0.5380 | +0.0187 | +0.0754 | yes |
| **CF-04 v2g** | **300 steps, 3 seeds** | **+2.3084** | **+0.0180** | **+0.1754** | **yes** |
| ASU | 300 steps, 3 seeds | +3.1780 | +0.1730 | +0.0340 | no |
| CF-04 v2g | 10 epochs | +8.0920 | -0.4561 | +0.9656 | yes |

At the same 300-step budget, v2g closes most of the forget gap to ASU while keeping retain-safe:
- dF: `2.31` vs ASU `3.18` (73% of ASU)
- dR: `0.018` vs ASU `0.173` (much safer)
- dH: `0.175` vs ASU `0.034` (worse generalization)

## Conclusion

- **v2 is a real improvement over c11**: 300-step dF increased from `0.538` to `2.308` while keeping `dR≈0.018`.
- **v2g is the best retain-safe 300-step CF-04 configuration.**
- **CF-04 v2 is still not a free win**: holdout degradation is substantially worse than ASU and grows to `+0.966` at 10 epochs.
- **ASU remains the stronger and more holdout-stable baseline**, while CF-04 v2 is the better retain-safe option.
- Applicability: retain-sensitive unlearning, where a weaker but safer update is preferred; not suitable for holdout-sensitive deployment without an additional holdout guard.

## Independent Review

Hy4 review was attempted with an 8192 token budget. It returned only `reasoning_content` and empty final `content`; a forced-format echo is not a review. Status is `REVIEW_UNAVAILABLE`; no score or verdict is fabricated.

## Artifacts

```text
workspace/auto_experiments/results/cf04_v2_300/
workspace/auto_experiments/results/formal_training/CF-04_v2_formal.json
workspace/review-stage/AUTO_REVIEW.md
workspace/review-stage/REVIEW_STATE.json
```
