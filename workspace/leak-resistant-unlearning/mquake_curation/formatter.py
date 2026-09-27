"""Format matched pairs + LLM results into MQuAKE-compatible JSON."""

from typing import Any

from .config import QUESTION_TEMPLATES, CLOZE_TEMPLATES, RELATION_NAMES


def _make_single_hop(fact: dict) -> dict:
    rel = fact["relation_id"]
    subj = fact["subject"]
    answer = fact["target_true"]["str"]

    question_tpl = QUESTION_TEMPLATES.get(rel, f"What is the {RELATION_NAMES.get(rel, rel)} of [X]?")
    question = question_tpl.replace("[X]", subj)

    cloze_tpl = CLOZE_TEMPLATES.get(rel, f"The {RELATION_NAMES.get(rel, rel)} of [X] is")
    cloze = cloze_tpl.replace("[X]", subj)

    return {
        "question": question,
        "cloze": cloze,
        "answer": answer,
        "answer_alias": [],
    }


def _make_rewrite(fact: dict) -> dict:
    return {
        "prompt": fact["prompt"],
        "relation_id": fact["relation_id"],
        "target_new": fact["target_new"],
        "target_true": fact["target_true"],
        "subject": fact["subject"],
        "question": fact.get("question", ""),
    }


def _get_facts(pair: dict) -> list[dict]:
    """Extract facts from a matched pair (varies by type)."""
    pair_type = pair.get("type", "")

    if pair_type == "disjunctive_elimination":
        facts = [pair["fact"]]
        if pair.get("implicit_evidence"):
            facts.append(pair["implicit_evidence"])
        return facts

    if pair_type == "identity_substitution":
        return [pair["fact1"]]

    facts = []
    if "fact1" in pair:
        facts.append(pair["fact1"])
    if "fact2" in pair:
        facts.append(pair["fact2"])
    return facts


def format_case(case_id: int, pair: dict, llm_result: dict) -> dict:
    facts = _get_facts(pair)
    requested_rewrite = [_make_rewrite(f) for f in facts]
    questions = llm_result.get("questions", [])
    answer = pair["answer"]
    answer_alias = llm_result.get("answer_alias", [])
    single_hops = [_make_single_hop(f) for f in facts]

    triples = []
    triples_labeled = []
    for f in facts:
        triples.append([
            f["target_true"].get("id", ""),
            f["relation_id"],
            f["target_true"].get("id", ""),
        ])
        rel_name = RELATION_NAMES.get(f["relation_id"], f["relation_id"])
        triples_labeled.append([
            f["subject"],
            rel_name,
            f["target_true"]["str"],
        ])

    return {
        "case_id": case_id,
        "reasoning_type": pair["type"],
        "edit_target_idx": pair.get("edit_target_idx", 0),
        "requested_rewrite": requested_rewrite,
        "questions": questions,
        "answer": answer,
        "answer_alias": answer_alias,
        "new_answer": "",
        "new_answer_alias": [],
        "single_hops": single_hops,
        "new_single_hops": [],
        "orig": {
            "triples": triples,
            "triples_labeled": triples_labeled,
            "new_triples": [],
            "new_triples_labeled": [],
            "edit_triples": [],
        },
    }


def format_all(type_name: str, generated: list[tuple[dict, dict]],
               case_id_start: int = 1) -> list[dict]:
    cases = []
    for i, (pair, llm_result) in enumerate(generated):
        case_id = case_id_start + i
        cases.append(format_case(case_id, pair, llm_result))
    return cases
