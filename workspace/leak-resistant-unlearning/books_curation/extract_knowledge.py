#!/usr/bin/env python3
"""
从小说文本直接抽取结构化知识，用于构建多跳遗忘推理链。

知识类型：
  1. relation:    (S, R, O)           — 实体间关系，支持链式蕴含
  2. category:    (S, is_a, C)        — 类别归属，支持实例归属
  3. identity:    (S, same_as, O)     — 等价/别名，支持同一替换
  4. property:    (S, has_property, P) — 属性，支持多元拼合
  5. rule:        ∀x(C(x)→P(x))      — 通用规则，支持实例归属/逆否推断
  6. exclusive:   exclusive(A,B,C...) — 互斥集合，支持互斥消除

用法:
    python extract_knowledge.py --input forget.txt --output hp_knowledge.json --max_chunks 5
"""

import json
import re
import time
import argparse
import os
from typing import List, Dict

import openai


KE_PROMPT = """\
You are a knowledge extraction expert for fictional worlds. Extract structured knowledge from this novel passage.

Output a JSON array. Each item must have a "type" field and corresponding fields:

1. "relation": Entity-to-entity relationships
   {"type": "relation", "subject": "...", "relation": "...", "object": "..."}
   Examples: {"type": "relation", "subject": "Harry Potter", "relation": "attends", "object": "Hogwarts"}
             {"type": "relation", "subject": "Hogwarts", "relation": "located_in", "object": "Scotland"}

2. "category": Entity belongs to a category/class
   {"type": "category", "entity": "...", "category": "..."}
   Examples: {"type": "category", "entity": "Harry Potter", "category": "wizard"}
             {"type": "category", "entity": "Hedwig", "category": "owl"}

3. "identity": Two names refer to the same entity
   {"type": "identity", "name1": "...", "name2": "..."}
   Examples: {"type": "identity", "name1": "Tom Riddle", "name2": "Lord Voldemort"}
             {"type": "identity", "name1": "Padfoot", "name2": "Sirius Black"}

4. "property": Entity has a specific attribute or ability
   {"type": "property", "entity": "...", "property": "..."}
   Examples: {"type": "property", "entity": "Harry Potter", "property": "can speak Parseltongue"}
             {"type": "property", "entity": "Hermione", "property": "top of her class"}

5. "rule": Universal rule in this fictional world (if you can infer one)
   {"type": "rule", "condition": "...", "consequence": "...", "description": "..."}
   Examples: {"type": "rule", "condition": "is a Muggle", "consequence": "cannot perform magic", "description": "Muggles cannot perform magic"}

6. "exclusive": A set of mutually exclusive or exhaustive options
   {"type": "exclusive", "set_name": "...", "members": ["...", "..."]}
   Examples: {"type": "exclusive", "set_name": "Hogwarts houses", "members": ["Gryffindor", "Slytherin", "Hufflepuff", "Ravenclaw"]}

Rules:
- Extract ONLY factual knowledge, not momentary actions or emotions
- Use full character names (e.g., "Harry Potter" not "Harry", "Albus Dumbledore" not "Dumbledore") when first introducing, but common short forms are OK if unambiguous
- Be specific: "teaches Potions" is better than "is a teacher"
- Do NOT duplicate: if a fact was already stated, skip it
- For "rule" and "exclusive" types: only extract when clearly stated or strongly implied

Passage:
\"\"\"{chunk}\"\"\"
"""


class LLMClient:
    def __init__(self, model: str, base_url: str, api_key: str):
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.total_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def call(self, prompt: str, temperature: float = 0.1, retries: int = 3) -> str:
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

    def parse_json_array(self, text: str) -> List[Dict]:
        text = text.strip()
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        try:
            result = json.loads(text)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group())
                if isinstance(result, list):
                    return result
            except json.JSONDecodeError:
                pass
        return []


def split_into_chunks(text: str, chunk_words: int = 500) -> List[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_words, len(words))
        if end < len(words):
            for j in range(end, max(start + chunk_words - 50, start), -1):
                if words[j - 1].endswith(('.', '!', '?', '."', '!"', '?"')):
                    end = j
                    break
        chunk = ' '.join(words[start:end])
        if chunk.strip():
            chunks.append(chunk.strip())
        start = end
    return chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--chunk_words", type=int, default=500)
    parser.add_argument("--max_chunks", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--base_url", type=str, default="https://api.openai.com/v1")
    parser.add_argument("--api_key", type=str,
                        default="YOUR_API_KEY")
    parser.add_argument("--save_every", type=int, default=50)
    args = parser.parse_args()

    with open(args.input, "r") as f:
        text = f.read()

    chunks = split_into_chunks(text, args.chunk_words)
    if args.max_chunks:
        chunks = chunks[:args.max_chunks]
    total = len(chunks)
    print(f"Total chunks: {total}")

    # Resume
    done = {}
    if args.resume and os.path.exists(args.output):
        with open(args.output) as f:
            prev = json.load(f)
        for entry in prev.get("chunks", []):
            done[entry["chunk_idx"]] = entry
        print(f"Resuming: {len(done)} chunks done")

    llm = LLMClient(model=args.model, base_url=args.base_url, api_key=args.api_key)

    all_chunks = []
    all_knowledge = []
    type_counts = {"relation": 0, "category": 0, "identity": 0,
                   "property": 0, "rule": 0, "exclusive": 0, "other": 0}

    for i, chunk in enumerate(chunks):
        idx = i + 1
        if idx in done:
            entry = done[idx]
            all_chunks.append(entry)
            for k in entry["knowledge"]:
                t = k.get("type", "other")
                type_counts[t] = type_counts.get(t, 0) + 1
                all_knowledge.append(k)
            continue

        print(f"[{idx}/{total}] {chunk[:60]}...", flush=True)
        prompt = KE_PROMPT.replace("{chunk}", chunk)
        response = llm.call(prompt)
        knowledge = llm.parse_json_array(response)

        # Validate and count types
        valid = []
        for k in knowledge:
            t = k.get("type", "other")
            if t in type_counts:
                type_counts[t] += 1
            else:
                type_counts["other"] += 1
            valid.append(k)

        print(f"  → {len(valid)} facts ({', '.join(f'{t}:{c}' for t,c in type_counts.items() if c > 0)})", flush=True)

        entry = {"chunk_idx": idx, "knowledge": valid}
        all_chunks.append(entry)
        all_knowledge.extend(valid)

        if idx % args.save_every == 0:
            _save(args.output, all_chunks, all_knowledge, type_counts, total, llm)
            print(f"  [Saved at {idx}/{total}]", flush=True)

    _save(args.output, all_chunks, all_knowledge, type_counts, total, llm)

    print(f"\n{'='*60}")
    print(f"Chunks: {total}")
    print(f"Total facts: {len(all_knowledge)}")
    for t, c in sorted(type_counts.items()):
        if c > 0:
            print(f"  {t}: {c}")
    print(f"LLM calls: {llm.total_calls}")
    print(f"Tokens: in={llm.total_input_tokens}, out={llm.total_output_tokens}")
    cost = llm.total_input_tokens / 1e6 * 0.15 + llm.total_output_tokens / 1e6 * 0.6
    print(f"Cost (gpt-4o-mini): ${cost:.4f}")
    print(f"Saved to {args.output}")


def _save(path, chunks, knowledge, type_counts, total, llm):
    data = {
        "total_chunks": total,
        "chunks_done": len(chunks),
        "total_facts": len(knowledge),
        "type_counts": type_counts,
        "llm_calls": llm.total_calls,
        "input_tokens": llm.total_input_tokens,
        "output_tokens": llm.total_output_tokens,
        "chunks": chunks,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
