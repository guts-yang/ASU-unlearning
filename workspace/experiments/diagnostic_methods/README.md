# Decoupled Diagnostic Methods

One module per plan-level candidate. Each module owns its core operator and a
CPU self-test; no module writes ARIS claim state or invokes an evidence gate.

## Modules

| ID | Module | Core operator |
|---|---|---|
| CF-01 | `cf01_constrained_inverse.py` | equality-constrained minimum-norm ΔA + causal filter |
| CF-02 | `cf02_role_decomposition.py` | deletion / anti-substitution / utility least-squares split |
| CF-03 | `cf03_causal_gate.py` | patch-layer forward, per-sequence NLL effect, causal gate |
| CF-04 | `cf04_dual_probes.py` | trained fact and identity linear probes + AUC margin |
| CF-05 | `cf05_multi_donor.py` | multi-donor shared/private SVD and marginalization |
| CF-06 | `cf06_lrm_trace.py` | CoT trace weighting; fails closed on LLM-only backbones |
| CF-07 | `cf07_readout_delta.py` | readout-space ΔA, finite-difference Jacobian, rotation test |
| CF-08 | `cf08_nullspace.py` | retain-gradient principal basis, nullspace projection, recovery AUC |
| CF-09 | `cf09_control_residual.py` | control-family residualization and shortcut metrics |
| CF-10 | `cf10_purity_schedule.py` | purity spectrum and frozen progressive schedule |
| R2-C | `r2c_relearning_aware.py` | inner relearning attack + minimax outer loss |
| R3-2 | `r3_2_depth_audit.py` | standardize P1/P2/P3/INT8 audit output |
| R3-3 | `r3_3_two_stage.py` | shallow/deep stage weights and alignment loss |

## Run

From `workspace/experiments`:

```bash
python -m diagnostic_methods.runner --list
python -m diagnostic_methods.runner --describe CF-01
python -m diagnostic_methods.runner --self-test all
python -m unittest diagnostic_methods.tests.test_methods -v
```

The self-test writes `auto_experiments/results/diagnostic_methods_self_test.json`.
It is a code sanity artifact, not an experiment claim.
