"""Type 3: Identity Substitution (同一替换型)

Rule: x=y and P(x) ⊢ P(y)
Find facts whose subject appears as some other fact's target_true with answer_alias.
Replace subject with its alias in the question.

Example:
  - Some fact has target_true="United Kingdom" with answer_alias=["Britain","UK",...]
  - Another fact: subject="United Kingdom", relation=P30, target_true="Europe"
  - Question: "Which continent is UK in?" (subject replaced with alias)
  - Answer: Europe
  - Edit target: the original fact (forget "United Kingdom → Europe")
  - Model needs to know UK = United Kingdom to answer
"""

from typing import Any

from ..data_loader import FactIndex


def match_identity_substitution(index: FactIndex,
                                entity_aliases: dict[str, list[str]] | None = None
                                ) -> list[dict[str, Any]]:
    if not entity_aliases:
        return []

    results = []
    seen = set()

    # For each entity that has known aliases AND appears as a subject in some fact
    for entity_name, aliases in entity_aliases.items():
        if not aliases:
            continue

        # Find facts where this entity is the subject
        subject_facts = index.by_subject.get(entity_name, [])
        if not subject_facts:
            continue

        for fact in subject_facts:
            key = (entity_name, fact["relation_id"])
            if key in seen:
                continue
            seen.add(key)

            results.append({
                "type": "identity_substitution",
                "fact1": fact,  # the property fact (edit target: forget this)
                "edit_target_idx": 0,
                "original_name": entity_name,
                "aliases": aliases,
                "relation_id": fact["relation_id"],
                "property_value": fact["target_true"]["str"],
                "answer": fact["target_true"]["str"],
                "answer_alias": [],
            })

    return results
