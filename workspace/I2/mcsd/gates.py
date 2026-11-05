"""New-version gates. The E0 retain cap is rejected rather than reused."""

from __future__ import annotations

from I2.mcsd.errors import ConfigError, HoldoutTuningError


E0_RETAIN_CAP = 0.0034702592157625847
E0_CAP_RULE = "two_se_of_calibration_baseline_retain_author_nll"
E0_DECISION = "do_not_start_e1"


def is_e0_retain_cap(value):
    if value is None:
        return False
    if isinstance(value, str) and value == E0_CAP_RULE:
        return True
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return abs(number - E0_RETAIN_CAP) <= 1e-18


def _looks_like_e0(config):
    if not isinstance(config, dict):
        return False
    if config.get("version") == "mcsd_v2":
        return False
    if config.get("decision") == E0_DECISION:
        return True
    if "attention_temperature_grid" in config or "mlp_temperature_grid" in config:
        return True
    if config.get("retain_damage_cap") == E0_CAP_RULE:
        return True
    return False


def student_block_reason(config):
    """Return a refusal string, or None when the new gate is complete."""
    if not isinstance(config, dict):
        return "MCSD_ASU requires an mcsd_v2 gate; Student E1 stays disabled"
    if _looks_like_e0(config):
        return (
            "E0 decision do_not_start_e1 is unchanged; the E0 config and its retain "
            "cap cannot start Student E1"
        )
    if config.get("version") != "mcsd_v2":
        return "Student E1 requires version mcsd_v2; refusing an unlabeled config"
    if config.get("allow_student_e1") is not True:
        return "allow_student_e1 is not true; Student E1 stays disabled"
    if config.get("selection_split") != "calibration":
        return "selection_split must be calibration; holdout cannot tune the student gate"
    gates = config.get("gates")
    if not isinstance(gates, dict):
        return "preregistered gate block is missing"
    if gates.get("preregistered") is not True:
        return "preregistered gate fields are missing or false"
    if gates.get("localization_passed") is not True:
        return "localization gate has not passed"
    if gates.get("teacher_retain_gate_passed") is not True:
        return "teacher retain gate has not passed"
    if gates.get("retain_nll_cap") is None:
        return "new retain gate threshold is missing; refusing to treat measurements as a pass"
    if is_e0_retain_cap(gates.get("retain_nll_cap")):
        return "refusing to reuse the E0 retain delta-NLL cap"
    if not gates.get("localization_result_id") or not gates.get("teacher_gate_id"):
        return "localization result id and teacher gate id are required"
    if gates.get("tuned_on") == "holdout" or gates.get("localization_n_min") is None:
        if gates.get("tuned_on") == "holdout":
            return "holdout cannot be used to tune paths, thresholds, or rho"
        return "localization_n_min is missing from the preregistered gate"
    if gates.get("localization_r_min") is None:
        return "localization_r_min is missing from the preregistered gate"
    if config.get("alpha") is None:
        return "alpha is missing"
    return None


def assert_calibration_only(config, split, purpose):
    if config.get("selection_split") != "calibration" or split != "calibration":
        raise HoldoutTuningError(
            "%s is refused on split %s; holdout cannot tune paths, thresholds, budgets, or rho"
            % (purpose, split)
        )


def localization_gate(n_value, r_value, config):
    gates = config.get("gates") or {}
    n_min = gates.get("localization_n_min")
    r_min = gates.get("localization_r_min")
    if n_min is None or r_min is None:
        return {"pass": None, "status": "thresholds_missing"}
    if n_value is None or r_value is None:
        return {"pass": None, "status": "metrics_missing"}
    return {
        "pass": bool(float(n_value) > float(n_min) and float(r_value) > float(r_min)),
        "status": "evaluated",
    }


def retain_gate(delta_nll, config):
    gates = config.get("gates") or {}
    cap = gates.get("retain_nll_cap")
    if cap is None:
        return {"pass": None, "status": "thresholds_missing"}
    if is_e0_retain_cap(cap):
        raise ConfigError("refusing to reuse the E0 retain delta-NLL cap")
    if delta_nll is None:
        return {"pass": None, "status": "metrics_missing"}
    return {"pass": bool(float(delta_nll) <= float(cap)), "status": "evaluated"}
