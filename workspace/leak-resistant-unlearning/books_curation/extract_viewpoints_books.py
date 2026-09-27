#!/usr/bin/env python3
"""
对 MUSE-Books（Harry Potter）语料进行 viewpoint 提取。
改编自 ARCHE 论文方法，适配小说文本。

用法:
    python extract_viewpoints_books.py --input forget.txt --output viewpoints.json
    python extract_viewpoints_books.py --input forget.txt --output viewpoints.json --resume  # 断点续跑
"""

import json
import re
import sys
import time
import argparse
from pathlib import Path
from typing import List, Dict

import openai


VIEWPOINT_PROMPT = """\
You are a knowledge extraction expert. Extract atomic "viewpoints" (knowledge facts) from this novel passage.

A viewpoint is a single, self-contained factual statement about:
- Characters: names, roles, relationships, traits, actions
- Events: what happened, who did what to whom
- Settings: locations, times
- Objects: important items, possessions
- World-building: rules, lore, institutions

Rules:
- One fact per viewpoint. Be atomic.
- Faithful to text only. No inference.
- Use character names, not pronouns.
- Skip purely atmospheric/descriptive text with no factual content.

Output: JSON array of strings.

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

    def extract_json_array(self, text: str) -> List[str]:
        text = text.strip()
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        try:
            result = json.loads(text)
            if isinstance(result, list):
                return [str(v) for v in result]
        except json.JSONDecodeError:
            pass
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group())
                if isinstance(result, list):
                    return [str(v) for v in result]
            except json.JSONDecodeError:
                pass
        return []

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


def split_into_chunks(text: str, chunk_words: int = 500) -> List[str]:
    """Split text into ~500 word chunks, breaking at sentence boundaries."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_words, len(words))
        # Try to break at sentence boundary (look for period near end)
        if end < len(words):
            search_start = max(start + chunk_words - 50, start)
            best_break = end
            for j in range(end, search_start, -1):
                w = words[j - 1]
                if w.endswith(('.', '!', '?', '."', '!"', '?"')):
                    best_break = j
                    break
            end = best_break
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
    parser.add_argument("--max_chunks", type=int, default=None,
                        help="Limit number of chunks (for testing)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing output file")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--base_url", type=str, default="https://api.openai.com/v1")
    parser.add_argument("--api_key", type=str,
                        default="YOUR_API_KEY")
    parser.add_argument("--save_every", type=int, default=50,
                        help="Save checkpoint every N chunks")
    args = parser.parse_args()

    with open(args.input, "r") as f:
        text = f.read()

    chunks = split_into_chunks(text, args.chunk_words)
    total_chunks = len(chunks)
    if args.max_chunks:
        chunks = chunks[:args.max_chunks]
        total_chunks = len(chunks)
    print(f"Total chunks: {total_chunks} ({args.chunk_words} words each)")

    # Resume support
    done_chunks = {}
    if args.resume and Path(args.output).exists():
        with open(args.output) as f:
            prev = json.load(f)
        for entry in prev.get("chunks", []):
            done_chunks[entry["chunk_idx"]] = entry
        print(f"Resuming: {len(done_chunks)} chunks already done")

    llm = LLMClient(model=args.model, base_url=args.base_url, api_key=args.api_key)

    results_list = []
    all_viewpoints = []

    for i, chunk in enumerate(chunks):
        idx = i + 1
        # Skip if already done
        if idx in done_chunks:
            entry = done_chunks[idx]
            results_list.append(entry)
            all_viewpoints.extend(entry["viewpoints"])
            continue

        print(f"[{idx}/{total_chunks}] {chunk[:60]}...", flush=True)

        prompt = VIEWPOINT_PROMPT.format(chunk=chunk)
        response = llm.call(prompt)
        viewpoints = llm.extract_json_array(response)

        print(f"  → {len(viewpoints)} viewpoints", flush=True)

        entry = {
            "chunk_idx": idx,
            "chunk_text": chunk,
            "viewpoints": viewpoints,
        }
        results_list.append(entry)
        all_viewpoints.extend(viewpoints)

        # Periodic save
        if idx % args.save_every == 0:
            _save(args.output, results_list, all_viewpoints, total_chunks, llm)
            print(f"  [Saved checkpoint at {idx}/{total_chunks}]", flush=True)

    _save(args.output, results_list, all_viewpoints, total_chunks, llm)

    unique_vps = list(dict.fromkeys(all_viewpoints))
    print(f"\n{'='*60}")
    print(f"Chunks processed: {total_chunks}")
    print(f"Total viewpoints: {len(all_viewpoints)}")
    print(f"Unique viewpoints: {len(unique_vps)}")
    print(f"LLM calls: {llm.total_calls}")
    print(f"Input tokens: {llm.total_input_tokens}, Output tokens: {llm.total_output_tokens}")
    cost_in = llm.total_input_tokens / 1e6 * 0.15  # gpt-4o-mini pricing
    cost_out = llm.total_output_tokens / 1e6 * 0.6
    print(f"Estimated cost (gpt-4o-mini): ${cost_in + cost_out:.4f}")
    print(f"Saved to {args.output}")


def _save(output_path, results_list, all_viewpoints, total_chunks, llm):
    unique_vps = list(dict.fromkeys(all_viewpoints))
    data = {
        "total_chunks": total_chunks,
        "chunks_processed": len(results_list),
        "total_viewpoints": len(all_viewpoints),
        "num_unique": len(unique_vps),
        "llm_calls": llm.total_calls,
        "input_tokens": llm.total_input_tokens,
        "output_tokens": llm.total_output_tokens,
        "chunks": results_list,
        "unique_viewpoints": unique_vps,
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
