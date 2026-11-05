"""Local rho interpolation on a frozen binary path set."""

from __future__ import annotations

from I2.mcsd.errors import ConfigError
from I2.mcsd.patch import module_logits, patched_logits
from I2.mcsd.score import advantage, answer_token_distributions, masked_token_nll


def run_rho_sweep(model, example, paths, config, retain_example=None, retain_reason=None):
    rho_cfg = config.get("rho") or {}
    if rho_cfg.get("select_best"):
        raise ConfigError("rho sweep cannot select a best value; report the full grid")
    grid = rho_cfg.get("grid")
    if not grid:
        raise ConfigError("rho.grid is missing")
    exploratory = rho_cfg.get("grid_status") == "exploratory_unregistered"
    if not paths:
        return {
            "status": "localization_failed",
            "sweep": None,
            "selected_rho": None,
            "entered_student": False,
            "grid_status": rho_cfg.get("grid_status"),
            "exploratory_unregistered": exploratory,
        }
    rows = []
    scope = None
    for rho in grid:
        logits, _ = patched_logits(
            model,
            example["score_plus"],
            paths,
            example["cache_plus"],
            example["cache_minus"],
            strength=float(rho),
        )
        foil_logits = None
        foil_labels = None
        if example.get("foil_reliable"):
            foil_logits, _ = patched_logits(
                model,
                example["foil_plus"],
                paths,
                example["cache_plus"],
                example["cache_minus"],
                strength=float(rho),
            )
            foil_labels = example["foil_plus"]["labels"]
        score, scope = advantage(
            logits,
            example["score_plus"]["labels"],
            foil_logits,
            foil_labels,
            foil_reliable=bool(example.get("foil_reliable")),
        )
        retain_nll = None
        reason = None
        if retain_example is None:
            reason = retain_reason or "no aligned retain pair"
        else:
            retain_logits, _ = patched_logits(
                model,
                retain_example["score_plus"],
                paths,
                retain_example["cache_plus"],
                retain_example["cache_minus"],
                strength=float(rho),
            )
            retain_nll = float(masked_token_nll(retain_logits, retain_example["score_plus"]["labels"]))
        rows.append(
            {
                "rho": float(rho),
                "target_score": float(score.detach().reshape(-1)[0]),
                "retain_nll": retain_nll,
                "delta_target_score": None,
                "delta_retain_nll": None,
                "retain_status": "computed" if reason is None else "skipped",
                "retain_reason": reason,
                "answer_token_distributions": answer_token_distributions(logits, example["score_plus"]["labels"]),
                "teacher_logits": logits.detach().clone(),
                "status": "computed",
            }
        )
    reference = next((row for row in rows if row["rho"] == 0.0), None)
    if reference is None:
        clean = module_logits(model, example["score_plus"])
        ref_score, _ = advantage(clean, example["score_plus"]["labels"], foil_reliable=False)
        ref_score = float(ref_score.detach().reshape(-1)[0])
        ref_nll = None
    else:
        ref_score = reference["target_score"]
        ref_nll = reference["retain_nll"]
    for row in rows:
        row["delta_target_score"] = row["target_score"] - ref_score
        if row["retain_nll"] is None or ref_nll is None:
            row["delta_retain_nll"] = None
        else:
            row["delta_retain_nll"] = row["retain_nll"] - ref_nll
    return {
        "status": "measured",
        "sweep": rows,
        "selected_rho": None,
        "entered_student": False,
        "grid_status": rho_cfg.get("grid_status"),
        "exploratory_unregistered": exploratory,
        "score_scope": scope,
    }
