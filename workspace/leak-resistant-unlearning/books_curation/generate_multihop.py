#!/usr/bin/env python3
"""
从清洗后的 HP 知识库生成六种多跳推理问题。

六种类型：
1. chain_implication:     A→B, B→C ⊢ A→C
2. instance_subsumption:  ∀x(S(x)→P(x)) ∧ S(a) ⊢ P(a)
3. identity_substitution: x=y ∧ P(x) ⊢ P(y)
4. multi_element_intersection: x∈A ∧ y∈A ⊢ x,y∈A
5. disjunctive_elimination: A∨B ∧ ¬A ⊢ B
6. contrapositive_reasoning: A→B ∧ ¬B ⊢ ¬A
"""

import json
import random
import re
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

# HP identity data (manually curated — these are well-known in the books)
HP_IDENTITIES = [
    ("Tom Riddle", "Lord Voldemort"),
    ("Tom Marvolo Riddle", "Lord Voldemort"),
    ("You-Know-Who", "Lord Voldemort"),
    ("He-Who-Must-Not-Be-Named", "Lord Voldemort"),
    ("The Dark Lord", "Lord Voldemort"),
    ("Padfoot", "Sirius Black"),
    ("Prongs", "James Potter"),
    ("Moony", "Remus Lupin"),
    ("Wormtail", "Peter Pettigrew"),
    ("The Half-Blood Prince", "Severus Snape"),
    ("Snivellus", "Severus Snape"),
    ("Norbert", "Norberta"),
    ("The Boy Who Lived", "Harry Potter"),
    ("The Chosen One", "Harry Potter"),
    ("Fluffy", "the three-headed dog"),
    ("Mrs. Norris", "Filch's cat"),
    ("Aragog", "Hagrid's spider"),
    ("Buckbeak", "Witherwings"),
    ("Barty Crouch Jr.", "fake Mad-Eye Moody"),
    ("Riddle's diary", "a Horcrux"),
    ("Slytherin's locket", "a Horcrux"),
    ("Hufflepuff's cup", "a Horcrux"),
    ("Ravenclaw's diadem", "a Horcrux"),
    ("Marvolo Gaunt's ring", "a Horcrux"),
    ("Nagini", "a Horcrux"),
]


def build_entity_index(facts: List[Dict]) -> Dict[str, List[Dict]]:
    """Build entity -> facts index."""
    index = defaultdict(list)
    for f in facts:
        t = f.get("type")
        if t == "relation":
            index[f.get("subject", "").lower()].append(f)
            index[f.get("object", "").lower()].append(f)
        elif t == "category":
            index[f.get("entity", "").lower()].append(f)
        elif t == "property":
            index[f.get("entity", "").lower()].append(f)
    return index


def generate_chain_implication(facts: List[Dict], n: int = 50) -> List[Dict]:
    """A→B, B→C ⊢ A→C"""
    relations = [f for f in facts if f.get("type") == "relation"]

    # Filter relations to only string-valued fields
    clean_rels = []
    for f in relations:
        s = f.get("subject", "")
        r = f.get("relation", "")
        o = f.get("object", "")
        if isinstance(s, str) and isinstance(r, str) and isinstance(o, str):
            clean_rels.append(f)
    relations = clean_rels

    # Find 2-hop chains: fact1(A,r1,B) + fact2(B,r2,C) → A relates to C
    results = []
    seen = set()
    for f1 in relations:
        s1 = f1["subject"].strip()
        o1 = f1["object"].strip()
        r1 = f1["relation"].strip()
        if not s1 or not o1 or not isinstance(o1, str) or not isinstance(s1, str):
            continue
        # Find facts where o1 is the subject
        for f2 in relations:
            s2 = f2.get("subject", "")
            o2 = f2.get("object", "")
            r2 = f2.get("relation", "")
            if not isinstance(s2, str) or not isinstance(o2, str) or not isinstance(r2, str):
                continue
            s2, o2, r2 = s2.strip(), o2.strip(), r2.strip()
            if not s2 or not o2:
                continue
            if s2.lower() == o1.lower() and s1.lower() != o2.lower():
                key = (s1.lower(), o1.lower(), o2.lower())
                if key in seen:
                    continue
                seen.add(key)
                results.append({
                    "type": "chain_implication",
                    "premise1": f"{s1} {r1} {o1}",
                    "premise2": f"{s2} {r2} {o2}",
                    "conclusion": f"{s1} is connected to {o2} through {o1}",
                    "single_hops": [
                        {"question": f"What does {s1} {r1}?", "answer": o1},
                        {"question": f"What does {o1} {r2}?", "answer": o2},
                    ],
                    "multi_hop_question": f"What is {s1} connected to through {o1}'s {r2}?",
                    "multi_hop_answer": o2,
                    "facts": [f1, f2],
                })
                if len(results) >= n:
                    return results
    return results


def generate_instance_subsumption(facts: List[Dict], n: int = 50) -> List[Dict]:
    """∀x(S(x)→P(x)) ∧ S(a) ⊢ P(a)"""
    rules = [f for f in facts if f.get("type") == "rule"]
    categories = [f for f in facts if f.get("type") == "category"]

    results = []
    seen = set()
    for rule in rules:
        cond = rule.get("condition", "").strip()
        cons = rule.get("consequence", "").strip()
        if not cond or not cons:
            continue
        # Try to match category facts
        # e.g., rule: IF is a wizard THEN can attend Hogwarts
        #        cat: Harry Potter is_a wizard
        for cat in categories:
            entity = cat.get("entity", "").strip()
            category = cat.get("category", "").strip()
            if not entity or not category:
                continue
            # Check if category matches the condition
            if category.lower() in cond.lower() or cond.lower().endswith(category.lower()):
                key = (entity.lower(), cond.lower())
                if key in seen:
                    continue
                seen.add(key)
                results.append({
                    "type": "instance_subsumption",
                    "premise1_rule": f"If something {cond}, then it {cons}",
                    "premise2_instance": f"{entity} is a {category}",
                    "conclusion": f"{entity} {cons}",
                    "single_hops": [
                        {"question": f"What is {entity}?", "answer": f"a {category}"},
                        {"question": f"What happens if something {cond}?", "answer": cons},
                    ],
                    "multi_hop_question": f"Based on what {entity} is, can {entity} {cons.lstrip('can ').lstrip('will ')}?",
                    "multi_hop_answer": f"Yes, because {entity} is a {category}",
                    "facts": [rule, cat],
                })
                if len(results) >= n:
                    return results
    return results


def generate_identity_substitution(facts: List[Dict], identities: List[Tuple], n: int = 50) -> List[Dict]:
    """x=y ∧ P(x) ⊢ P(y)"""
    relations = [f for f in facts if f.get("type") in ("relation", "property")]

    # Build name -> facts index
    name_facts = defaultdict(list)
    for f in relations:
        if f.get("type") == "relation":
            name_facts[f.get("subject", "").lower()].append(f)
        elif f.get("type") == "property":
            name_facts[f.get("entity", "").lower()].append(f)

    results = []
    seen = set()
    for name1, name2 in identities:
        # Find facts about name1
        for f in name_facts.get(name1.lower(), []):
            if f.get("type") == "relation":
                pred = f"{f.get('relation', '')} {f.get('object', '')}"
                entity = f.get("subject", "")
            else:
                pred = f.get("property", "")
                entity = f.get("entity", "")

            key = (name1.lower(), name2.lower(), pred.lower())
            if key in seen:
                continue
            seen.add(key)

            results.append({
                "type": "identity_substitution",
                "premise1_identity": f"{name1} is the same person/thing as {name2}",
                "premise2_fact": f"{entity} {pred}",
                "conclusion": f"{name2} {pred}",
                "single_hops": [
                    {"question": f"Who is {name1} also known as?", "answer": name2},
                    {"question": f"What about {entity}? ({pred.split()[0]}...)", "answer": pred},
                ],
                "multi_hop_question": f"Does {name2} {pred}?",
                "multi_hop_answer": f"Yes, because {name2} is {name1}, and {name1} {pred}",
                "facts": [{"type": "identity", "name1": name1, "name2": name2}, f],
            })
            if len(results) >= n:
                return results

        # Also check name2 -> apply to name1
        for f in name_facts.get(name2.lower(), []):
            if f.get("type") == "relation":
                pred = f"{f.get('relation', '')} {f.get('object', '')}"
                entity = f.get("subject", "")
            else:
                pred = f.get("property", "")
                entity = f.get("entity", "")

            key = (name2.lower(), name1.lower(), pred.lower())
            if key in seen:
                continue
            seen.add(key)

            results.append({
                "type": "identity_substitution",
                "premise1_identity": f"{name2} is the same person/thing as {name1}",
                "premise2_fact": f"{entity} {pred}",
                "conclusion": f"{name1} {pred}",
                "single_hops": [
                    {"question": f"Who is {name2} also known as?", "answer": name1},
                    {"question": f"What about {entity}? ({pred.split()[0]}...)", "answer": pred},
                ],
                "multi_hop_question": f"Does {name1} {pred}?",
                "multi_hop_answer": f"Yes, because {name1} is {name2}, and {name2} {pred}",
                "facts": [{"type": "identity", "name1": name2, "name2": name1}, f],
            })
            if len(results) >= n:
                return results
    return results


def generate_multi_element_intersection(facts: List[Dict], n: int = 50) -> List[Dict]:
    """x∈A ∧ y∈A ⊢ x,y∈A"""
    categories = [f for f in facts if f.get("type") == "category"]
    properties = [f for f in facts if f.get("type") == "property"]

    # Group by category
    cat_groups = defaultdict(list)
    for f in categories:
        cat_groups[f.get("category", "").lower()].append(f.get("entity", ""))

    # Group by property
    prop_groups = defaultdict(list)
    for f in properties:
        prop_groups[f.get("property", "").lower()].append(f.get("entity", ""))

    results = []
    seen = set()

    # From categories
    for cat, entities in cat_groups.items():
        if len(entities) < 2 or not cat:
            continue
        unique = list(dict.fromkeys(entities))
        if len(unique) < 2:
            continue
        for i in range(min(len(unique) - 1, 5)):
            e1, e2 = unique[i], unique[i + 1]
            key = tuple(sorted([e1.lower(), e2.lower()])) + (cat,)
            if key in seen:
                continue
            seen.add(key)
            results.append({
                "type": "multi_element_intersection",
                "premise1": f"{e1} is a {cat}",
                "premise2": f"{e2} is a {cat}",
                "conclusion": f"Both {e1} and {e2} are {cat}s",
                "single_hops": [
                    {"question": f"What is {e1}?", "answer": f"a {cat}"},
                    {"question": f"What is {e2}?", "answer": f"a {cat}"},
                ],
                "multi_hop_question": f"What do {e1} and {e2} have in common?",
                "multi_hop_answer": f"They are both {cat}s",
                "shared_attribute": cat,
            })
            if len(results) >= n:
                return results

    # From properties
    for prop, entities in prop_groups.items():
        if len(entities) < 2 or not prop:
            continue
        unique = list(dict.fromkeys(entities))
        if len(unique) < 2:
            continue
        e1, e2 = unique[0], unique[1]
        key = tuple(sorted([e1.lower(), e2.lower()])) + (prop,)
        if key in seen:
            continue
        seen.add(key)
        results.append({
            "type": "multi_element_intersection",
            "premise1": f"{e1} {prop}",
            "premise2": f"{e2} {prop}",
            "conclusion": f"Both {e1} and {e2} {prop}",
            "single_hops": [
                {"question": f"Does {e1} have the property: {prop}?", "answer": "Yes"},
                {"question": f"Does {e2} have the property: {prop}?", "answer": "Yes"},
            ],
            "multi_hop_question": f"What do {e1} and {e2} have in common?",
            "multi_hop_answer": f"They both {prop}",
            "shared_attribute": prop,
        })
        if len(results) >= n:
            return results

    return results


def generate_disjunctive_elimination(facts: List[Dict], n: int = 50) -> List[Dict]:
    """A∨B ∧ ¬A ⊢ B"""
    exclusives = [f for f in facts if f.get("type") == "exclusive"]

    results = []
    seen = set()
    for f in exclusives:
        members = f.get("members", [])
        set_name = f.get("set_name", "")
        if len(members) < 2 or not set_name:
            continue

        # For each member, create a question eliminating others
        for i, target in enumerate(members):
            others = [m for j, m in enumerate(members) if j != i]
            if len(others) == 0:
                continue
            key = (set_name.lower(), target.lower())
            if key in seen:
                continue
            seen.add(key)

            if len(others) == 1:
                negation = f"It is not {others[0]}"
            else:
                negation = f"It is not {', '.join(others[:-1])} or {others[-1]}"

            results.append({
                "type": "disjunctive_elimination",
                "premise1_set": f"The {set_name} are: {', '.join(members)}",
                "premise2_negation": negation,
                "conclusion": f"It must be {target}",
                "single_hops": [
                    {"question": f"What are the {set_name}?", "answer": ', '.join(members)},
                    {"question": f"Which {set_name} is it NOT?", "answer": ', '.join(others)},
                ],
                "multi_hop_question": f"If the {set_name} are {', '.join(members)}, and it is not {', '.join(others)}, what is it?",
                "multi_hop_answer": target,
                "exclusive_set": f,
            })
            if len(results) >= n:
                return results
    return results


def generate_contrapositive(facts: List[Dict], n: int = 50) -> List[Dict]:
    """A→B ∧ ¬B ⊢ ¬A"""
    rules = [f for f in facts if f.get("type") == "rule"]

    results = []
    seen = set()
    for rule in rules:
        cond = rule.get("condition", "").strip()
        cons = rule.get("consequence", "").strip()
        desc = rule.get("description", "").strip()
        if not cond or not cons:
            continue

        key = (cond.lower(), cons.lower())
        if key in seen:
            continue
        seen.add(key)

        # Negate consequence
        if cons.startswith("can "):
            neg_cons = cons.replace("can ", "cannot ", 1)
        elif cons.startswith("will "):
            neg_cons = cons.replace("will ", "will not ", 1)
        elif cons.startswith("is "):
            neg_cons = cons.replace("is ", "is not ", 1)
        elif cons.startswith("must "):
            neg_cons = cons.replace("must ", "does not have to ", 1)
        else:
            neg_cons = f"does not {cons}"

        # Negate condition
        if cond.startswith("is a "):
            neg_cond = cond.replace("is a ", "is not a ", 1)
        elif cond.startswith("is "):
            neg_cond = cond.replace("is ", "is not ", 1)
        else:
            neg_cond = f"is not {cond}"

        results.append({
            "type": "contrapositive_reasoning",
            "premise1_rule": f"If something {cond}, then it {cons}",
            "premise2_negation": f"Entity X {neg_cons}",
            "conclusion": f"Entity X {neg_cond}",
            "description": desc,
            "single_hops": [
                {"question": f"What happens if something {cond}?", "answer": cons},
                {"question": f"Does Entity X {cons}?", "answer": f"No, Entity X {neg_cons}"},
            ],
            "multi_hop_question": f"If something that {cond} always {cons}, and Entity X {neg_cons}, what can we conclude about Entity X?",
            "multi_hop_answer": f"Entity X {neg_cond}",
            "rule": rule,
        })
        if len(results) >= n:
            return results
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="arche_viewpoint/cleaned_ke/hp_knowledge_clean.json")
    parser.add_argument("--output", type=str, default="arche_viewpoint/hp_multihop_questions.json")
    parser.add_argument("--per_type", type=int, default=50)
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)
    facts = data["facts"]
    print(f"Loaded {len(facts)} facts")

    random.seed(42)

    print("\nGenerating multi-hop questions...")

    chain = generate_chain_implication(facts, args.per_type)
    print(f"  chain_implication: {len(chain)}")

    instance = generate_instance_subsumption(facts, args.per_type)
    print(f"  instance_subsumption: {len(instance)}")

    identity = generate_identity_substitution(facts, HP_IDENTITIES, args.per_type)
    print(f"  identity_substitution: {len(identity)}")

    multi = generate_multi_element_intersection(facts, args.per_type)
    print(f"  multi_element_intersection: {len(multi)}")

    disjunctive = generate_disjunctive_elimination(facts, args.per_type)
    print(f"  disjunctive_elimination: {len(disjunctive)}")

    contra = generate_contrapositive(facts, args.per_type)
    print(f"  contrapositive_reasoning: {len(contra)}")

    all_questions = chain + instance + identity + multi + disjunctive + contra

    output = {
        "total": len(all_questions),
        "by_type": {
            "chain_implication": len(chain),
            "instance_subsumption": len(instance),
            "identity_substitution": len(identity),
            "multi_element_intersection": len(multi),
            "disjunctive_elimination": len(disjunctive),
            "contrapositive_reasoning": len(contra),
        },
        "questions": all_questions,
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nTotal: {len(all_questions)} questions saved to {args.output}")


if __name__ == "__main__":
    main()
