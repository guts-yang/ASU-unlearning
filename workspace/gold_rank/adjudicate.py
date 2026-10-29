"""Apply the preregistered stop rules to a completed factorial report.

The report has to be supplied by a finished evaluation. This module does not
invent metrics or declare a method successful.
"""

import math


def _mean(values):
    values = [float(value) for value in values]
    if not values:
        raise ValueError("expected at least one paired seed")
    return sum(values) / len(values)


def _sample_std(values):
    values = [float(value) for value in values]
    if len(values) < 2:
        return math.inf
    center = _mean(values)
    return math.sqrt(sum((value - center) ** 2 for value in values) / (len(values) - 1))


def adjudicate(report, margin):
    if report.get("hyperparameters_changed_after_test"):
        return {"decision": "stop_post_hoc_hyperparameters", "stop": True}
    if margin is None:
        return {"decision": "stop_unregistered_fe_margin", "stop": True}
    required = (
        "teacher_rank_increase",
        "student_forget_efficacy_delta",
        "matched_penalty_forget_efficacy_delta",
        "substitution_or_leak_delta",
        "anchor_utility_delta",
        "joint_utility_delta",
        "joint_forget_delta",
    )
    missing = [key for key in required if key not in report]
    if missing:
        return {"decision": "stop_incomplete_report", "stop": True, "missing": missing}

    reasons = []
    if float(report["teacher_rank_increase"]) > 0 and _mean(report["student_forget_efficacy_delta"]) <= 0:
        reasons.append("teacher_rank_without_student_gain")
    penalty_gap = [
        float(a) - float(b)
        for a, b in zip(report["student_forget_efficacy_delta"], report["matched_penalty_forget_efficacy_delta"])
    ]
    if _mean(penalty_gap) - _sample_std(penalty_gap) <= 0:
        reasons.append("not_distinct_from_matched_penalty")
    if _mean(report["substitution_or_leak_delta"]) > 0:
        reasons.append("substitution_or_generation_leak")
    if _mean(report["anchor_utility_delta"]) <= 0:
        reasons.append("anchor_utility_not_recovered")
    utility_bound = _mean(report["joint_utility_delta"]) - _sample_std(report["joint_utility_delta"])
    forget_bound = _mean(report["joint_forget_delta"]) - _sample_std(report["joint_forget_delta"])
    if utility_bound <= 0 or forget_bound < -float(margin):
        reasons.append("joint_tradeoff_outside_preregistered_margin")
    if reasons:
        return {"decision": "stop_current_version", "stop": True, "reasons": reasons}
    return {"decision": "pilot_only_pending_interval_review", "stop": False, "reasons": []}
