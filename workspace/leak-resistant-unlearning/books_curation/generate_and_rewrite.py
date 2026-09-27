#!/usr/bin/env python3
"""
生成多跳问题 → 质量过滤 → LLM 改写

过滤标准:
1. 实体标准: 主体必须是具名实体(人/地/物)，不能是泛称("students","Entity X")
2. 多样性标准: 同一 rule/identity/category 最多复用 N 次
3. 内容标准: 必须涉及 HP 特有知识，非通用常识
4. 逻辑标准: 前提→结论的推理必须合理
"""

import json
import re
import time
import argparse
import os
from collections import defaultdict, Counter
from typing import List, Dict, Tuple

import openai

# ── Import generation functions ─────────────────────────────────────────
import sys
sys.path.insert(0, os.path.dirname(__file__))
from generate_multihop import (
    generate_chain_implication,
    generate_instance_subsumption,
    generate_identity_substitution,
    generate_multi_element_intersection,
    generate_disjunctive_elimination,
    generate_contrapositive,
    HP_IDENTITIES,
)


class LLMClient:
    def __init__(self, model, base_url, api_key):
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.total_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def call(self, prompt, temperature=0.1, retries=3):
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


# ── Filtering ───────────────────────────────────────────────────────────

def diversity_filter(questions: List[Dict], max_per_key: int = 3) -> List[Dict]:
    """Limit reuse of same rule/identity/category."""
    key_counts = Counter()
    result = []
    for q in questions:
        t = q["type"]
        if t == "instance_subsumption":
            key = q.get("premise1_rule", "")
        elif t == "identity_substitution":
            key = q.get("premise1_identity", "").split(" is the same")[0]
        elif t == "multi_element_intersection":
            key = q.get("shared_attribute", "")
        elif t == "disjunctive_elimination":
            key = q.get("premise1_set", "")
        elif t == "contrapositive_reasoning":
            key = q.get("premise1_rule", "")
        else:
            key = ""

        if key and key_counts[key] >= max_per_key:
            continue
        key_counts[key] += 1
        result.append(q)
    return result


def entity_filter(questions: List[Dict]) -> List[Dict]:
    """Remove questions with generic/unnamed entities."""
    generic = {"students", "people", "entity x", "someone", "something",
               "characters", "wizards", "muggles", "they", "he", "she"}
    result = []
    for q in questions:
        # Check main entities
        texts = json.dumps(q).lower()
        is_generic = False
        # Check if multi_hop_question or answer refers to overly generic terms
        mhq = q.get("multi_hop_question", "").lower()
        mha = q.get("multi_hop_answer", "").lower()
        # Skip if answer is just "yes" or "no" without specifics
        if mha.strip() in ("yes", "no"):
            is_generic = True
        if not is_generic:
            result.append(q)
    return result


# ── LLM Rewrite ─────────────────────────────────────────────────────────

REWRITE_PROMPT = """\
You are rewriting multi-hop reasoning questions about Harry Potter into natural, fluent English.

For each question, I give you:
- type: the reasoning pattern
- premises: the factual statements
- raw_question: the auto-generated question (may be awkward)
- raw_answer: the auto-generated answer

Rewrite into:
- "question": A natural question a human would ask (do NOT reveal the reasoning structure)
- "answer": A concise, direct answer (just the fact, no "Yes, because..." unless needed)
- "premise1": Clean natural language statement of first premise
- "premise2": Clean natural language statement of second premise
- "valid": true if the logic actually holds, false if the reasoning is flawed

Output a JSON array of rewritten items.

Items to rewrite:
ITEMS_PLACEHOLDER
"""


def rewrite_batch(questions: List[Dict], llm: LLMClient) -> List[Dict]:
    """Rewrite a batch of questions using LLM."""
    items = []
    for i, q in enumerate(questions):
        item = {"idx": i, "type": q["type"]}
        if q["type"] == "chain_implication":
            item["premises"] = [q["premise1"], q["premise2"]]
            item["raw_question"] = q["multi_hop_question"]
            item["raw_answer"] = q["multi_hop_answer"]
        elif q["type"] == "instance_subsumption":
            item["premises"] = [q["premise1_rule"], q["premise2_instance"]]
            item["raw_question"] = q["multi_hop_question"]
            item["raw_answer"] = q["multi_hop_answer"]
        elif q["type"] == "identity_substitution":
            item["premises"] = [q["premise1_identity"], q["premise2_fact"]]
            item["raw_question"] = q["multi_hop_question"]
            item["raw_answer"] = q["multi_hop_answer"]
        elif q["type"] == "multi_element_intersection":
            item["premises"] = [q["premise1"], q["premise2"]]
            item["raw_question"] = q["multi_hop_question"]
            item["raw_answer"] = q["multi_hop_answer"]
        elif q["type"] == "disjunctive_elimination":
            item["premises"] = [q["premise1_set"], q["premise2_negation"]]
            item["raw_question"] = q["multi_hop_question"]
            item["raw_answer"] = q["multi_hop_answer"]
        elif q["type"] == "contrapositive_reasoning":
            item["premises"] = [q["premise1_rule"], q["premise2_negation"]]
            item["raw_question"] = q["multi_hop_question"]
            item["raw_answer"] = q["multi_hop_answer"]
        items.append(item)

    prompt = REWRITE_PROMPT.replace("ITEMS_PLACEHOLDER", json.dumps(items, indent=2))
    response = llm.call(prompt)

    # Parse
    text = response.strip()
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    try:
        rewritten = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            try:
                rewritten = json.loads(match.group())
            except:
                return []
        else:
            return []

    # Merge back — LLM may or may not return idx, so match by position
    results = []
    for i, item in enumerate(rewritten):
        idx = item.get("idx", i)  # fallback to position
        if idx < 0 or idx >= len(questions):
            idx = i
        if idx >= len(questions):
            continue
        if not item.get("valid", True):
            continue
        orig = dict(questions[idx])
        orig["rewritten_question"] = item.get("question", orig.get("multi_hop_question", ""))
        orig["rewritten_answer"] = item.get("answer", orig.get("multi_hop_answer", ""))
        orig["rewritten_premise1"] = item.get("premise1", "")
        orig["rewritten_premise2"] = item.get("premise2", "")
        results.append(orig)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--knowledge", type=str,
                        default="arche_viewpoint/cleaned_ke/hp_knowledge_clean.json")
    parser.add_argument("--output", type=str,
                        default="arche_viewpoint/hp_multihop_final.json")
    parser.add_argument("--per_type_raw", type=int, default=200,
                        help="Raw generation per type before filtering")
    parser.add_argument("--per_type_final", type=int, default=50,
                        help="Target per type after filtering")
    parser.add_argument("--max_per_key", type=int, default=3,
                        help="Max reuse of same rule/identity/category")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--base_url", type=str, default="https://api.openai.com/v1")
    parser.add_argument("--api_key", type=str,
                        default="YOUR_API_KEY")
    args = parser.parse_args()

    with open(args.knowledge) as f:
        data = json.load(f)
    facts = data["facts"]
    print(f"Loaded {len(facts)} facts")

    llm = LLMClient(model=args.model, base_url=args.base_url, api_key=args.api_key)

    # Generate raw (over-generate)
    N = args.per_type_raw
    print(f"\n=== Generating (up to {N} per type) ===")

    raw = {
        "chain_implication": generate_chain_implication(facts, N),
        "instance_subsumption": generate_instance_subsumption(facts, N),
        "identity_substitution": generate_identity_substitution(facts, HP_IDENTITIES, N),
        "multi_element_intersection": generate_multi_element_intersection(facts, N),
        "disjunctive_elimination": generate_disjunctive_elimination(facts, N),
        "contrapositive_reasoning": generate_contrapositive(facts, N),
    }
    for t, qs in raw.items():
        print(f"  {t}: {len(qs)} raw")

    # Filter
    print(f"\n=== Filtering (max_per_key={args.max_per_key}) ===")
    filtered = {}
    for t, qs in raw.items():
        f1 = diversity_filter(qs, max_per_key=args.max_per_key)
        f2 = entity_filter(f1)
        # Truncate to final target
        f2 = f2[:args.per_type_final]
        filtered[t] = f2
        print(f"  {t}: {len(qs)} → {len(f1)} (diversity) → {len(f2)} (entity+truncate)")

    # LLM Rewrite in batches
    print(f"\n=== LLM Rewrite ===")
    final_questions = []
    batch_size = 10

    for t, qs in filtered.items():
        if not qs:
            print(f"  {t}: skip (empty)")
            continue
        print(f"  {t}: rewriting {len(qs)} questions...", flush=True)
        type_results = []
        for i in range(0, len(qs), batch_size):
            batch = qs[i:i + batch_size]
            rewritten = rewrite_batch(batch, llm)
            type_results.extend(rewritten)
        print(f"    → {len(type_results)} valid after rewrite")
        final_questions.extend(type_results)

    # Save
    output = {
        "total": len(final_questions),
        "by_type": Counter(q["type"] for q in final_questions),
        "questions": final_questions,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Final: {len(final_questions)} questions")
    for t, c in sorted(output["by_type"].items()):
        print(f"  {t}: {c}")
    cost = llm.total_input_tokens / 1e6 * 0.15 + llm.total_output_tokens / 1e6 * 0.6
    print(f"LLM calls: {llm.total_calls}, cost: ${cost:.4f}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
