"""Type 6: Contrapositive Reasoning (逆否推断型)

Rule: A→B and ¬B ⊢ ¬A
Chain: start_entity -[rel1]-> bridge -[rel2]-> end_entity
Rule (fact2): bridge → end_entity
Contrapositive: ¬end_entity → ¬bridge

Edit target: fact2 (bridge→end_entity). If forgotten, model can't do contrapositive.
Answer: "not bridge"
"""

from typing import Any

from ..data_loader import FactIndex


def match_contrapositive_reasoning(index: FactIndex) -> list[dict[str, Any]]:
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

            end_entity = fact2["target_true"]["str"]

            results.append({
                "type": "contrapositive_reasoning",
                "fact1": fact1,
                "fact2": fact2,       # edit target: the rule to be forgotten
                "edit_target_idx": 1, # forget fact2 (bridge → end_entity)
                "original_subject": fact1["subject"],
                "bridge_entity": bridge,
                "end_entity": end_entity,
                "rule_relation": fact2["relation_id"],
                "antecedent_relation": fact1["relation_id"],
                "negated_observation": f"not {end_entity}",
                "contrapositive_conclusion": f"not {bridge}",
                "answer": f"not {bridge}",
                "answer_alias": [],
            })

    return results
