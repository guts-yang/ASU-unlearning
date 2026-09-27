#!/usr/bin/env python3
"""
用 LLM 补充 identity_substitution 和 instance_subsumption 类型的多跳问题。
基于已有的 identity 列表和 rules，让 LLM 直接生成高质量问题。
"""

import json
import re
import time
import argparse
import os
from typing import List, Dict

import openai

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
    ("The Boy Who Lived", "Harry Potter"),
    ("The Chosen One", "Harry Potter"),
    ("Buckbeak", "Witherwings"),
    ("Barty Crouch Jr.", "fake Mad-Eye Moody"),
]


class LLMClient:
    def __init__(self, model, base_url, api_key):
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.total_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def call(self, prompt, temperature=0.3, retries=3):
        for attempt in range(retries):
            try:
                self.total_calls += 1
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=4096,
                )
                text = resp.choices[0].message.content or ""
                if resp.usage:
                    self.total_input_tokens += resp.usage.prompt_tokens
                    self.total_output_tokens += resp.usage.completion_tokens
                return text
            except Exception as e:
                print(f"  [ERROR] {e}", flush=True)
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
        return ""


IDENTITY_PROMPT = """\
Generate identity substitution multi-hop questions about Harry Potter.

Pattern: Name1 is the same as Name2. Name1 has property/did action X. Therefore, Name2 also has property/did action X.

For each identity pair below, generate 3-4 questions. Each question must:
- Use a SPECIFIC, VERIFIABLE fact from the Harry Potter books (not made up)
- The fact should be well-known and associated with one name but not the other
- Be answerable with a short phrase

Identity pairs:
IDENTITIES_PLACEHOLDER

Output JSON array:
[{
  "type": "identity_substitution",
  "name1": "the lesser-known name",
  "name2": "the well-known name",
  "rewritten_premise1": "Name1 is also known as Name2.",
  "rewritten_premise2": "Name1 did/is X.",
  "rewritten_question": "Did Name2 do/is X?",
  "rewritten_answer": "Yes, [concise answer]",
  "single_hops": [
    {"question": "Who is Name1 also known as?", "answer": "Name2"},
    {"question": "What did Name1 do?", "answer": "X"}
  ]
}, ...]
"""

INSTANCE_PROMPT = """\
Generate instance subsumption multi-hop questions about Harry Potter.

Pattern: All members of category C have property P. Entity E is a member of C. Therefore, E has property P.

Using the rules and categories below from the Harry Potter universe, generate questions. Each must:
- Use SPECIFIC characters/entities from HP books
- The rule must be clearly true in the HP universe
- Be answerable with a short phrase

Rules (IF condition THEN consequence):
RULES_PLACEHOLDER

Known category memberships:
CATEGORIES_PLACEHOLDER

Generate 40 questions. Also feel free to use your knowledge of HP to create additional rule+instance combinations not listed above (e.g., "All Animagi can transform into animals" + "Sirius Black is an Animagus" → "Sirius Black can transform into an animal").

Output JSON array:
[{
  "type": "instance_subsumption",
  "rewritten_premise1": "All [category] [consequence/property].",
  "rewritten_premise2": "[Entity] is a [category].",
  "rewritten_question": "Does [Entity] [consequence]?",
  "rewritten_answer": "Yes, because [Entity] is a [category].",
  "single_hops": [
    {"question": "What is [Entity]?", "answer": "a [category]"},
    {"question": "What is true about all [category]?", "answer": "[consequence]"}
  ]
}, ...]
"""


def generate_identity_questions(llm: LLMClient, n_target: int = 50) -> List[Dict]:
    """Generate identity substitution questions via LLM."""
    # Split identities into batches
    batch_size = 5
    all_results = []

    for i in range(0, len(HP_IDENTITIES), batch_size):
        batch = HP_IDENTITIES[i:i + batch_size]
        id_text = "\n".join(f"- {n1} = {n2}" for n1, n2 in batch)
        prompt = IDENTITY_PROMPT.replace("IDENTITIES_PLACEHOLDER", id_text)

        print(f"  [identity] batch {i // batch_size + 1}, identities: {[p[0] for p in batch]}", flush=True)
        response = llm.call(prompt)

        text = response.strip()
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        try:
            items = json.loads(text)
            if isinstance(items, list):
                all_results.extend(items)
                print(f"    → {len(items)} questions", flush=True)
        except json.JSONDecodeError:
            match = re.search(r'\[.*\]', text, re.DOTALL)
            if match:
                try:
                    items = json.loads(match.group())
                    all_results.extend(items)
                    print(f"    → {len(items)} questions", flush=True)
                except:
                    print(f"    → parse error", flush=True)

        if len(all_results) >= n_target:
            break

    # Ensure type field
    for q in all_results:
        q["type"] = "identity_substitution"
    return all_results[:n_target]


def generate_instance_questions(llm: LLMClient, facts: List[Dict], n_target: int = 50) -> List[Dict]:
    """Generate instance subsumption questions via LLM."""
    rules = [f for f in facts if f.get("type") == "rule"]
    categories = [f for f in facts if f.get("type") == "category"]

    # Select diverse rules
    rule_texts = []
    seen = set()
    for r in rules:
        cond = r.get("condition", "")
        cons = r.get("consequence", "")
        key = (cond.lower(), cons.lower())
        if key not in seen and cond and cons:
            seen.add(key)
            rule_texts.append(f"IF {cond} THEN {cons}")
    rule_texts = rule_texts[:30]  # Top 30

    # Select diverse categories
    cat_texts = []
    seen_cats = set()
    for c in categories:
        entity = c.get("entity", "")
        cat = c.get("category", "")
        key = (entity.lower(), cat.lower())
        if key not in seen_cats and entity and cat:
            seen_cats.add(key)
            cat_texts.append(f"{entity} is a {cat}")
    cat_texts = cat_texts[:50]

    prompt = INSTANCE_PROMPT.replace("RULES_PLACEHOLDER", "\n".join(f"- {r}" for r in rule_texts))
    prompt = prompt.replace("CATEGORIES_PLACEHOLDER", "\n".join(f"- {c}" for c in cat_texts))

    print(f"  [instance] generating with {len(rule_texts)} rules, {len(cat_texts)} categories...", flush=True)
    response = llm.call(prompt)

    text = response.strip()
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    try:
        items = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            try:
                items = json.loads(match.group())
            except:
                items = []
        else:
            items = []

    print(f"    → {len(items)} questions", flush=True)

    for q in items:
        q["type"] = "instance_subsumption"
    return items[:n_target]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--knowledge", type=str,
                        default="arche_viewpoint/cleaned_ke/hp_knowledge_clean.json")
    parser.add_argument("--existing", type=str,
                        default="arche_viewpoint/hp_multihop_final.json")
    parser.add_argument("--output", type=str,
                        default="arche_viewpoint/hp_multihop_final.json")
    parser.add_argument("--target_identity", type=int, default=50)
    parser.add_argument("--target_instance", type=int, default=50)
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--base_url", type=str, default="https://api.openai.com/v1")
    parser.add_argument("--api_key", type=str,
                        default="YOUR_API_KEY")
    args = parser.parse_args()

    with open(args.knowledge) as f:
        facts = json.load(f)["facts"]

    with open(args.existing) as f:
        existing = json.load(f)

    llm = LLMClient(model=args.model, base_url=args.base_url, api_key=args.api_key)

    # Count existing
    existing_identity = [q for q in existing["questions"] if q["type"] == "identity_substitution"]
    existing_instance = [q for q in existing["questions"] if q["type"] == "instance_subsumption"]
    need_identity = max(0, args.target_identity - len(existing_identity))
    need_instance = max(0, args.target_instance - len(existing_instance))

    print(f"Existing: identity={len(existing_identity)}, instance={len(existing_instance)}")
    print(f"Need: identity={need_identity}, instance={need_instance}")

    new_questions = []

    if need_identity > 0:
        print(f"\n=== Generating identity_substitution ===")
        id_qs = generate_identity_questions(llm, need_identity)
        new_questions.extend(id_qs)
        print(f"  Generated: {len(id_qs)}")

    if need_instance > 0:
        print(f"\n=== Generating instance_subsumption ===")
        inst_qs = generate_instance_questions(llm, facts, need_instance)
        new_questions.extend(inst_qs)
        print(f"  Generated: {len(inst_qs)}")

    # Merge with existing
    all_questions = existing["questions"] + new_questions
    from collections import Counter
    output = {
        "total": len(all_questions),
        "by_type": dict(Counter(q["type"] for q in all_questions)),
        "questions": all_questions,
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Final: {len(all_questions)} questions")
    for t, c in sorted(output["by_type"].items()):
        print(f"  {t}: {c}")
    cost = llm.total_input_tokens / 1e6 * 0.15 + llm.total_output_tokens / 1e6 * 0.6
    print(f"Cost: ${cost:.4f}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
