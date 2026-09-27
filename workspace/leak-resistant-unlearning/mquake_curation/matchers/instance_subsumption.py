"""Type 2: Instance Subsumption (实例归属型)

Rule: instance ∈ category, category → property ⊢ instance → property
Match: fact1.target_true.str == fact2.subject (same as chain, restricted relation pairs)
Edit target: fact1 (instance membership) — forget "James Mercer is citizen of USA"
Question: directly asks instance's final property WITHOUT mentioning category.
  e.g., "What official language does James Mercer speak?"
"""

from typing import Any

from ..data_loader import FactIndex

# (membership_relation, category_property_relation) pairs
INSTANCE_SUBSUMPTION_PAIRS = [
    ("P27", "P37"),   # citizen of X + official language of X
    ("P27", "P35"),   # citizen of X + head of state of X
    ("P27", "P6"),    # citizen of X + head of government of X
    ("P27", "P30"),   # citizen of X + continent of X
    ("P27", "P36"),   # citizen of X + capital of X
    ("P108", "P159"), # employed by X + HQ of X in city
    ("P69", "P159"),  # educated at X + HQ of X in city
    ("P495", "P37"),  # created in country X + official language of X
    ("P495", "P36"),  # created in country X + capital of X
    ("P495", "P35"),  # created in country X + head of state of X
    ("P495", "P30"),  # created in country X + continent of X
]


def match_instance_subsumption(index: FactIndex) -> list[dict[str, Any]]:
    valid_pairs = set(INSTANCE_SUBSUMPTION_PAIRS)
    results = []
    seen = set()

    for fact1 in index.facts:
        bridge = fact1["target_true"]["str"]
        candidates = index.by_subject.get(bridge, [])
        for fact2 in candidates:
            if fact1 is fact2:
                continue
            rel_pair = (fact1["relation_id"], fact2["relation_id"])
            if rel_pair not in valid_pairs:
                continue

            key = (fact1["subject"], fact1["relation_id"],
                   fact2["subject"], fact2["relation_id"])
            if key in seen:
                continue
            seen.add(key)

            results.append({
                "type": "instance_subsumption",
                "fact1": fact1,   # instance membership (edit target)
                "fact2": fact2,   # category property
                "edit_target_idx": 0,  # forget instance membership
                "instance": fact1["subject"],
                "category": bridge,
                "final_property": fact2["target_true"]["str"],
                "property_relation": fact2["relation_id"],
                "answer": fact2["target_true"]["str"],
                "answer_alias": [],
            })

    return results
