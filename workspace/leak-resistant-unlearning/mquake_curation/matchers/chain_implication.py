"""Type 1: Chain Implication (链式蕴涵型)

Rule: A→B, B→C ⊢ A→C
Match: fact1.target_true.str == fact2.subject
Edit target: either fact (default fact1, the first hop).
Question asks about end_entity from start_entity, without mentioning bridge.
"""

from typing import Any

from ..data_loader import FactIndex


def match_chain_implication(index: FactIndex) -> list[dict[str, Any]]:
    results = []
    seen = set()

    for fact1 in index.facts:
        bridge = fact1["target_true"]["str"]
        candidates = index.by_subject.get(bridge, [])
        for fact2 in candidates:
            if fact1 is fact2:
                continue
            key = (fact1["subject"], fact1["relation_id"],
                   fact2["subject"], fact2["relation_id"])
            if key in seen:
                continue
            seen.add(key)

            results.append({
                "type": "chain_implication",
                "fact1": fact1,
                "fact2": fact2,
                "edit_target_idx": 0,  # forget fact1 (first hop)
                "bridge_entity": bridge,
                "start_entity": fact1["subject"],
                "end_entity": fact2["target_true"]["str"],
                "answer": fact2["target_true"]["str"],
                "answer_alias": [],
            })

    return results
