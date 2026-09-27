#!/usr/bin/env python3
"""
对抽取的 HP 知识进行去重和质量过滤。

Step 1: 去重 — 完全相同 + 语义近似去重
Step 2: 质量过滤 — 用 LLM 批量过滤低质量条目
Step 3: 类型修正 — 修正错误分类（如 identity 实际上是 category）

用法:
    python clean_knowledge.py --input hp_knowledge_full.json --output_dir cleaned_ke/
"""

import json
import re
import time
import argparse
import os
from typing import List, Dict, Tuple
from collections import defaultdict

import openai


class LLMClient:
    def __init__(self, model: str, base_url: str, api_key: str):
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.total_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def call(self, prompt: str, temperature: float = 0.0, retries: int = 3) -> str:
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
                print(f"    [ERROR] attempt {attempt+1}/{retries}: {e}", flush=True)
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
        return "[]"


# ── Step 1: Dedup ───────────────────────────────────────────────────────

def _norm(v) -> str:
    """Normalize a value to lowercase string."""
    if isinstance(v, list):
        return ','.join(sorted(str(x).lower().strip() for x in v))
    return str(v).lower().strip()


def fact_to_key(fact: Dict) -> str:
    """Convert a fact to a normalized string key for exact dedup."""
    t = fact.get("type", "")
    if t == "relation":
        return f"rel|{_norm(fact.get('subject',''))}|{_norm(fact.get('relation',''))}|{_norm(fact.get('object',''))}"
    elif t == "category":
        return f"cat|{_norm(fact.get('entity',''))}|{_norm(fact.get('category',''))}"
    elif t == "identity":
        names = sorted([_norm(fact.get('name1','')), _norm(fact.get('name2',''))])
        return f"id|{names[0]}|{names[1]}"
    elif t == "property":
        return f"prop|{_norm(fact.get('entity',''))}|{_norm(fact.get('property',''))}"
    elif t == "rule":
        return f"rule|{_norm(fact.get('condition',''))}|{_norm(fact.get('consequence',''))}"
    elif t == "exclusive":
        members = sorted([_norm(m) for m in fact.get('members', [])])
        return f"excl|{_norm(fact.get('set_name',''))}|{'|'.join(members)}"
    return str(fact)


def fact_to_text(fact: Dict) -> str:
    """Convert a fact to natural language for embedding-based dedup."""
    t = fact.get("type", "")
    if t == "relation":
        return f"{fact.get('subject','')} {fact.get('relation','')} {fact.get('object','')}"
    elif t == "category":
        return f"{fact.get('entity','')} is a {fact.get('category','')}"
    elif t == "identity":
        return f"{fact.get('name1','')} is the same as {fact.get('name2','')}"
    elif t == "property":
        return f"{fact.get('entity','')} {fact.get('property','')}"
    elif t == "rule":
        return f"If {fact.get('condition','')}, then {fact.get('consequence','')}"
    elif t == "exclusive":
        return f"{fact.get('set_name','')}: {', '.join(fact.get('members',[]))}"
    return str(fact)


def step1_dedup(facts: List[Dict], embedding_threshold: float = 0.92) -> List[Dict]:
    """Exact dedup + embedding-based near-dedup per type."""
    # Exact dedup
    seen_keys = set()
    exact_deduped = []
    for f in facts:
        key = fact_to_key(f)
        if key not in seen_keys:
            seen_keys.add(key)
            exact_deduped.append(f)
    print(f"  Exact dedup: {len(facts)} → {len(exact_deduped)}")

    # Embedding-based dedup within each type
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
    except ImportError:
        print("  [WARN] sentence-transformers not installed, skipping embedding dedup")
        return exact_deduped

    model = SentenceTransformer('all-MiniLM-L6-v2')

    final = []
    for t in ['relation', 'category', 'identity', 'property', 'rule', 'exclusive']:
        type_facts = [f for f in exact_deduped if f.get('type') == t]
        if not type_facts:
            continue

        texts = [fact_to_text(f) for f in type_facts]
        embeddings = model.encode(texts, show_progress_bar=False, batch_size=256)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        embeddings = embeddings / norms

        removed = set()
        for i in range(len(type_facts)):
            if i in removed:
                continue
            end = min(i + 100, len(type_facts))
            if end <= i + 1:
                continue
            sims = embeddings[i] @ embeddings[i+1:end].T
            for j_off, sim in enumerate(sims):
                j = i + 1 + j_off
                if j not in removed and sim > embedding_threshold:
                    removed.add(j)

        kept = [type_facts[i] for i in range(len(type_facts)) if i not in removed]
        print(f"  Embedding dedup [{t}]: {len(type_facts)} → {len(kept)}")
        final.extend(kept)

    return final


# ── Step 2: Quality Filter ──────────────────────────────────────────────

QUALITY_FILTER_PROMPT = """\
Review each knowledge fact and label it KEEP or REMOVE.

KEEP: Stable, factual, useful for question-answering. Examples:
- Relations: "Harry Potter attends Hogwarts" ✓
- Categories: "Hedwig is_a owl" ✓
- Rules: "IF uses Unforgivable Curse THEN life in Azkaban" ✓

REMOVE: Low quality. Examples:
- Momentary actions: "Harry is discussing Malfoy" ✗
- Vague/trivial: "Harry talked to Ron" ✗
- Too generic rules: "IF is a student THEN may be in danger" ✗
- Opinions/emotions: "Harry felt angry" ✗
- Wrong type (e.g., identity that's actually a category): "Kreacher = house-elf" ✗ (should be category, not identity)

For items with wrong type, label as RETYPE and provide the correct type.

Output JSON array: [{IDX}, ...]
Where each item is one of:
- {"idx": N, "label": "KEEP"}
- {"idx": N, "label": "REMOVE", "reason": "brief reason"}
- {"idx": N, "label": "RETYPE", "new_type": "category/relation/...", "corrected": {corrected fact object}}

Facts:
FACTS_PLACEHOLDER
"""


def step2_quality_filter(facts: List[Dict], llm: LLMClient, batch_size: int = 40,
                         save_path: str = None, resume: bool = False) -> Tuple[List[Dict], List[Dict]]:
    """Filter low-quality facts and fix mistyped ones."""
    labels = [None] * len(facts)
    corrections = {}

    if resume and save_path and os.path.exists(save_path):
        with open(save_path) as f:
            prev = json.load(f)
        for i, l in enumerate(prev.get("labels", [])):
            if i < len(labels):
                labels[i] = l
        corrections = prev.get("corrections", {})
        done = sum(1 for l in labels if l is not None)
        print(f"  [Resume] {done}/{len(facts)} already filtered")

    total_batches = (len(facts) + batch_size - 1) // batch_size

    for batch_start in range(0, len(facts), batch_size):
        batch_end = min(batch_start + batch_size, len(facts))
        batch_idx = batch_start // batch_size + 1

        if all(labels[i] is not None for i in range(batch_start, batch_end)):
            continue

        lines = []
        for i in range(batch_start, batch_end):
            f = facts[i]
            lines.append(f"[{i}] type={f.get('type','?')}: {fact_to_text(f)}")
        block = "\n".join(lines)
        prompt = QUALITY_FILTER_PROMPT.replace("FACTS_PLACEHOLDER", block)

        print(f"  [Step2] batch {batch_idx}/{total_batches}...", flush=True)
        response = llm.call(prompt)

        # Parse
        try:
            text = response.strip()
            text = re.sub(r'^```(?:json)?\s*', '', text)
            text = re.sub(r'\s*```$', '', text)
            parsed = json.loads(text)
            for item in parsed:
                idx = item.get("idx", -1)
                label = item.get("label", "").upper()
                if 0 <= idx < len(labels):
                    if label == "KEEP":
                        labels[idx] = "KEEP"
                    elif label == "REMOVE":
                        labels[idx] = "REMOVE"
                    elif label == "RETYPE":
                        labels[idx] = "RETYPE"
                        corrections[str(idx)] = item.get("corrected", item)
        except (json.JSONDecodeError, KeyError):
            # Fallback: line-by-line
            for line in response.split("\n"):
                m = re.search(r'"idx"\s*:\s*(\d+).*"label"\s*:\s*"(KEEP|REMOVE|RETYPE)"', line, re.I)
                if m:
                    idx = int(m.group(1))
                    label = m.group(2).upper()
                    if 0 <= idx < len(labels):
                        labels[idx] = label

        if save_path and batch_idx % 20 == 0:
            _save_quality(save_path, facts, labels, corrections, llm)

    labels = [l if l is not None else "KEEP" for l in labels]
    if save_path:
        _save_quality(save_path, facts, labels, corrections, llm)

    # Apply
    kept = []
    for i, f in enumerate(facts):
        if labels[i] == "KEEP":
            kept.append(f)
        elif labels[i] == "RETYPE" and str(i) in corrections:
            corrected = corrections[str(i)]
            if isinstance(corrected, dict) and "type" in corrected:
                kept.append(corrected)
            elif "new_type" in corrected:
                f_copy = dict(f)
                f_copy["type"] = corrected["new_type"]
                kept.append(f_copy)

    removed = [facts[i] for i in range(len(facts)) if labels[i] == "REMOVE"]
    return kept, removed


def _save_quality(path, facts, labels, corrections, llm):
    keep_n = sum(1 for l in labels if l == "KEEP")
    remove_n = sum(1 for l in labels if l == "REMOVE")
    retype_n = sum(1 for l in labels if l == "RETYPE")
    classified = sum(1 for l in labels if l is not None)
    data = {
        "total": len(facts),
        "classified": classified,
        "kept": keep_n,
        "removed": remove_n,
        "retyped": retype_n,
        "llm_calls": llm.total_calls,
        "input_tokens": llm.total_input_tokens,
        "output_tokens": llm.total_output_tokens,
        "labels": labels,
        "corrections": corrections,
    }
    with open(path, "w") as f:
        json.dump(data, f)


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--base_url", type=str, default="https://api.openai.com/v1")
    parser.add_argument("--api_key", type=str,
                        default="YOUR_API_KEY")
    parser.add_argument("--dedup_threshold", type=float, default=0.92)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.input) as f:
        data = json.load(f)

    all_facts = [k for c in data["chunks"] for k in c["knowledge"]]
    print(f"Input: {len(all_facts)} facts")
    for t in ['relation', 'category', 'identity', 'property', 'rule', 'exclusive']:
        n = sum(1 for f in all_facts if f.get('type') == t)
        print(f"  {t}: {n}")

    # Step 1: Dedup
    print(f"\n{'='*60}")
    print(f"Step 1: Deduplication")
    print(f"{'='*60}")
    deduped = step1_dedup(all_facts, embedding_threshold=args.dedup_threshold)

    with open(os.path.join(args.output_dir, "after_dedup.json"), "w") as f:
        json.dump({"facts": deduped, "count": len(deduped)}, f, indent=2, ensure_ascii=False)
    print(f"  After dedup: {len(deduped)} facts")
    for t in ['relation', 'category', 'identity', 'property', 'rule', 'exclusive']:
        n = sum(1 for f in deduped if f.get('type') == t)
        print(f"    {t}: {n}")

    # Step 2: Quality Filter
    print(f"\n{'='*60}")
    print(f"Step 2: Quality Filter ({len(deduped)} facts)")
    print(f"{'='*60}")
    llm = LLMClient(model=args.model, base_url=args.base_url, api_key=args.api_key)
    quality_path = os.path.join(args.output_dir, "step2_quality.json")
    kept, removed = step2_quality_filter(deduped, llm, save_path=quality_path, resume=args.resume)

    # Save final
    final_path = os.path.join(args.output_dir, "hp_knowledge_clean.json")
    final_data = {
        "total": len(kept),
        "type_counts": {},
        "facts": kept,
    }
    for t in ['relation', 'category', 'identity', 'property', 'rule', 'exclusive']:
        n = sum(1 for f in kept if f.get('type') == t)
        final_data["type_counts"][t] = n
    with open(final_path, "w") as f:
        json.dump(final_data, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Final: {len(kept)} facts (removed {len(removed)})")
    for t, n in sorted(final_data["type_counts"].items()):
        if n > 0:
            print(f"  {t}: {n}")
    print(f"LLM calls: {llm.total_calls}")
    cost = llm.total_input_tokens / 1e6 * 0.15 + llm.total_output_tokens / 1e6 * 0.6
    print(f"Cost (gpt-4o-mini): ${cost:.4f}")
    print(f"Saved to {final_path}")


if __name__ == "__main__":
    main()
