"""Type 4: Multi-element Intersection (多元拼合型)

Rule: x∈A and x∈B ⊢ x∈A∩B
Match: same target_true.str, same relation_id, different subjects.
Edit target: fact1 (forget one entity's property)
Question asks what both entities share.
"""

from typing import Any
from collections import defaultdict

from ..data_loader import FactIndex


def match_multi_element_intersection(index: FactIndex) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for fact in index.facts:
        key = (fact["relation_id"], fact["target_true"]["str"])
        groups[key].append(fact)

    results = []
    seen = set()

    for (rel_id, target_str), facts in groups.items():
        if len(facts) < 2:
            continue

        for i in range(len(facts)):
            for j in range(i + 1, len(facts)):
                f1, f2 = facts[i], facts[j]
                if f1["subject"] == f2["subject"]:
                    continue

                subj_pair = tuple(sorted([f1["subject"], f2["subject"]]))
                key = (rel_id, target_str, subj_pair)
                if key in seen:
                    continue
                seen.add(key)

                results.append({
                    "type": "multi_element_intersection",
                    "fact1": f1,  # edit target
                    "fact2": f2,
                    "edit_target_idx": 0,  # forget fact1
                    "entity1": f1["subject"],
                    "entity2": f2["subject"],
                    "common_property": target_str,
                    "relation_id": rel_id,
                    "answer": target_str,
                    "answer_alias": [],
                })

    return results
