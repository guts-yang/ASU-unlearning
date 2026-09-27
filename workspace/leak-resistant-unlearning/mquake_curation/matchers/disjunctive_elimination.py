"""Type 5: Disjunctive Elimination (互斥消除型)

Rule: A∪B∪C and ¬A and ¬B ⊢ C
Construct candidate sets. One distractor is eliminated explicitly in the question,
the other requires implicit knowledge (the forgotten fact) to eliminate.

Edit target: the fact needed to eliminate the "implicit" distractor.
"""

import random
from typing import Any
from collections import defaultdict

from ..data_loader import FactIndex


def match_disjunctive_elimination(index: FactIndex) -> list[dict[str, Any]]:
    # Group distinct target_true values by relation
    relation_targets: dict[str, list[str]] = defaultdict(list)
    for fact in index.facts:
        rel = fact["relation_id"]
        tv = fact["target_true"]["str"]
        if tv not in relation_targets.get(rel, []):
            relation_targets.setdefault(rel, []).append(tv)

    # Build a lookup: for a given (relation, target_value), find facts that have that target
    # This helps us find elimination evidence
    target_to_facts: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for fact in index.facts:
        target_to_facts[(fact["relation_id"], fact["target_true"]["str"])].append(fact)

    # Only use relations with enough distinct targets for distractors
    usable_relations = {
        rel: targets for rel, targets in relation_targets.items()
        if len(targets) >= 3
    }

    results = []
    seen = set()

    for fact in index.facts:
        rel = fact["relation_id"]
        if rel not in usable_relations:
            continue

        true_answer = fact["target_true"]["str"]
        all_targets = usable_relations[rel]

        distractors = [t for t in all_targets if t != true_answer]
        if len(distractors) < 2:
            continue

        key = (fact["subject"], rel)
        if key in seen:
            continue
        seen.add(key)

        rng = random.Random(hash((fact["subject"], rel)))
        selected = rng.sample(distractors, 2)
        distractor_implicit = selected[0]  # needs implicit knowledge to eliminate
        distractor_explicit = selected[1]  # eliminated explicitly in question

        # Find a fact that provides evidence for the implicit distractor
        # (i.e., a fact that shows distractor_implicit belongs to someone else)
        implicit_evidence_facts = target_to_facts.get((rel, distractor_implicit), [])
        # Pick one that's NOT about the same subject
        implicit_evidence = None
        for ef in implicit_evidence_facts:
            if ef["subject"] != fact["subject"]:
                implicit_evidence = ef
                break

        if implicit_evidence is None:
            continue

        # Find evidence for explicit elimination
        explicit_evidence_facts = target_to_facts.get((rel, distractor_explicit), [])
        explicit_evidence = None
        for ef in explicit_evidence_facts:
            if ef["subject"] != fact["subject"]:
                explicit_evidence = ef
                break

        candidate_set = [distractor_implicit, distractor_explicit, true_answer]
        rng.shuffle(candidate_set)

        results.append({
            "type": "disjunctive_elimination",
            "fact": fact,                          # the target fact
            "implicit_evidence": implicit_evidence, # fact needed to eliminate distractor_implicit (EDIT TARGET)
            "explicit_evidence": explicit_evidence,  # fact used for explicit elimination in question
            "edit_target_idx": 1,  # implicit_evidence is at index 1 in requested_rewrite
            "subject": fact["subject"],
            "relation_id": rel,
            "candidate_set": candidate_set,
            "answer": true_answer,
            "distractor_implicit": distractor_implicit,
            "distractor_explicit": distractor_explicit,
            "explicit_elimination_reason": (
                f"{distractor_explicit} is the {fact['relation_id']} of "
                f"{explicit_evidence['subject']}" if explicit_evidence else ""
            ),
            "answer_alias": [],
        })

    return results
