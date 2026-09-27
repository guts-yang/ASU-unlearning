"""Load MQuAKE data and build indexes for matcher use."""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_facts(dataset_path: str | Path) -> list[dict[str, Any]]:
    """Load all individual rewrite facts from MQuAKE dataset.

    Flattens all requested_rewrite entries across all cases into a single list.
    Each fact gets a back-reference to its source case_id.
    """
    with open(dataset_path, encoding="utf-8") as f:
        data = json.load(f)

    facts = []
    for case in data:
        case_id = case["case_id"]
        for rw in case.get("requested_rewrite", []):
            fact = {
                "case_id": case_id,
                "subject": rw["subject"],
                "relation_id": rw["relation_id"],
                "prompt": rw["prompt"],
                "question": rw.get("question", ""),
                "target_true": rw["target_true"],
                "target_new": rw["target_new"],
            }
            facts.append(fact)
    return facts


def extract_entity_aliases(dataset_path: str | Path) -> dict[str, list[str]]:
    """Extract entity -> alias list from single_hops answer_alias fields.

    Collects aliases for entities that appear as answers in single_hops.
    Returns dict mapping entity name -> list of aliases.
    """
    with open(dataset_path, encoding="utf-8") as f:
        data = json.load(f)

    entity_aliases: dict[str, set[str]] = defaultdict(set)

    for case in data:
        # From single_hops
        for hop in case.get("single_hops", []):
            answer = hop.get("answer", "")
            aliases = hop.get("answer_alias", [])
            if answer and aliases:
                for a in aliases:
                    if a and a != answer:
                        entity_aliases[answer].add(a)

        # From top-level answer_alias
        answer = case.get("answer", "")
        aliases = case.get("answer_alias", [])
        if answer and aliases:
            for a in aliases:
                if a and a != answer:
                    entity_aliases[answer].add(a)

    # Convert sets to sorted lists
    return {k: sorted(v) for k, v in entity_aliases.items() if v}


class FactIndex:
    """Pre-built indexes over facts for efficient matching."""

    def __init__(self, facts: list[dict[str, Any]]):
        self.facts = facts

        self.by_subject: dict[str, list[dict]] = defaultdict(list)
        self.by_target_true: dict[str, list[dict]] = defaultdict(list)
        self.by_target_true_id: dict[str, list[dict]] = defaultdict(list)
        self.by_relation: dict[str, list[dict]] = defaultdict(list)
        self.by_subject_relation: dict[tuple[str, str], list[dict]] = defaultdict(list)

        for fact in facts:
            subj = fact["subject"]
            tt_str = fact["target_true"]["str"]
            tt_id = fact["target_true"]["id"]
            rel = fact["relation_id"]

            self.by_subject[subj].append(fact)
            self.by_target_true[tt_str].append(fact)
            self.by_target_true_id[tt_id].append(fact)
            self.by_relation[rel].append(fact)
            self.by_subject_relation[(subj, rel)].append(fact)

    def __len__(self):
        return len(self.facts)
