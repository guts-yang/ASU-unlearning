#!/usr/bin/env python3
"""
Viewpoint 清洗 Pipeline

Step 1: LLM 分类 — 保留 persistent fact，过滤 action/emotion/atmosphere
Step 2: 实体抽取 + 过滤 — 至少含一个命名实体
Step 3: Embedding 去重 — cosine > 0.9 合并
Step 4: 实体图连通性过滤 — 只保留能与 ≥2 个其他 viewpoint 共享实体的

用法:
    python clean_viewpoints.py --input books_viewpoints_full.json --output_dir cleaned/ --step 1
    python clean_viewpoints.py --input books_viewpoints_full.json --output_dir cleaned/ --step all
"""

import json
import re
import time
import argparse
import os
from pathlib import Path
from typing import List, Dict, Set, Tuple
from collections import defaultdict

import openai


# ── Step 1: LLM Classification ─────────────────────────────────────────

CLASSIFY_PROMPT = """\
Classify each viewpoint as either KEEP or REMOVE.

KEEP: Persistent factual knowledge useful for question-answering, such as:
- Character identities, roles, relationships (e.g., "Sirius Black is Harry's godfather")
- Stable attributes or properties (e.g., "Azkaban is located in the North Sea")
- Important events with lasting consequences (e.g., "Sirius Black broke out of Azkaban")
- World-building rules and lore (e.g., "The Minister of Magic only reveals themselves to the Muggle Prime Minister")
- Object properties (e.g., "The Sorting Hat belonged to Godric Gryffindor")

REMOVE: Content NOT useful as knowledge premises, such as:
- Momentary actions/gestures (e.g., "Fudge rubbed his eyes wearily")
- Emotions/feelings (e.g., "The Prime Minister's pulse quickened")
- Atmospheric descriptions (e.g., "The weather was dismal with chilly mist")
- Dialogue mechanics (e.g., "Fudge pulled up a chair and sat down")
- Vague/unspecific statements (e.g., "Fudge mentioned recent events")

Output: A JSON array of objects with "idx" and "label" fields.
Example: [{{"idx": 0, "label": "KEEP"}}, {{"idx": 1, "label": "REMOVE"}}]

Viewpoints:
{viewpoints_block}
"""


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


def step1_classify(viewpoints: List[str], llm: LLMClient, batch_size: int = 50,
                   save_path: str = None, resume: bool = False) -> List[bool]:
    """Classify viewpoints as KEEP (True) or REMOVE (False)."""
    # Resume support
    labels = [None] * len(viewpoints)
    if resume and save_path and os.path.exists(save_path):
        with open(save_path) as f:
            prev = json.load(f)
        for i, label in enumerate(prev.get("labels", [])):
            if i < len(labels):
                labels[i] = label
        done = sum(1 for l in labels if l is not None)
        print(f"  [Resume] {done}/{len(viewpoints)} already classified")

    total_batches = (len(viewpoints) + batch_size - 1) // batch_size

    for batch_start in range(0, len(viewpoints), batch_size):
        batch_end = min(batch_start + batch_size, len(viewpoints))
        batch_idx = batch_start // batch_size + 1

        # Skip if all in this batch are done
        if all(labels[i] is not None for i in range(batch_start, batch_end)):
            continue

        # Build prompt
        lines = []
        for i in range(batch_start, batch_end):
            lines.append(f"[{i}] {viewpoints[i]}")
        block = "\n".join(lines)
        prompt = CLASSIFY_PROMPT.format(viewpoints_block=block)

        print(f"  [Step1] batch {batch_idx}/{total_batches} ({batch_start}-{batch_end})...", flush=True)
        response = llm.call(prompt)

        # Parse response
        try:
            # Clean response
            text = response.strip()
            text = re.sub(r'^```(?:json)?\s*', '', text)
            text = re.sub(r'\s*```$', '', text)
            parsed = json.loads(text)
            for item in parsed:
                idx = item.get("idx", -1)
                label = item.get("label", "").upper()
                if 0 <= idx < len(labels):
                    labels[idx] = (label == "KEEP")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"    [WARN] Failed to parse batch {batch_idx}: {e}", flush=True)
            # Fallback: try line-by-line parsing
            for line in response.split("\n"):
                match = re.search(r'"idx"\s*:\s*(\d+).*"label"\s*:\s*"(KEEP|REMOVE)"', line, re.I)
                if match:
                    idx = int(match.group(1))
                    label = match.group(2).upper()
                    if 0 <= idx < len(labels):
                        labels[idx] = (label == "KEEP")

        # Save checkpoint
        if save_path and batch_idx % 10 == 0:
            _save_step1(save_path, viewpoints, labels, llm)

    # Fill any remaining None with False (conservative)
    labels = [l if l is not None else False for l in labels]

    if save_path:
        _save_step1(save_path, viewpoints, labels, llm)

    return labels


def _save_step1(path, viewpoints, labels, llm):
    kept = sum(1 for l in labels if l is True)
    total = sum(1 for l in labels if l is not None)
    data = {
        "total": len(viewpoints),
        "classified": total,
        "kept": kept,
        "removed": total - kept,
        "llm_calls": llm.total_calls,
        "input_tokens": llm.total_input_tokens,
        "output_tokens": llm.total_output_tokens,
        "labels": labels,
    }
    with open(path, "w") as f:
        json.dump(data, f)


# ── Step 2: Entity Extraction + Filtering ───────────────────────────────

ENTITY_EXTRACT_PROMPT = """\
Extract all named entities from each viewpoint. Named entities include:
- Person names (e.g., Harry Potter, Dumbledore, Snape)
- Place names (e.g., Hogwarts, Azkaban, Diagon Alley)
- Organization names (e.g., Ministry of Magic, Order of the Phoenix, Death Eaters)
- Important objects with proper names (e.g., Sorting Hat, Elder Wand, Marauder's Map)
- Species/creature names (e.g., Dementors, House-elves, Hippogriff)

Do NOT extract: common nouns (chair, office, door), pronouns, adjectives.

Output: JSON array of objects with "idx" and "entities" fields.
Example: [{{"idx": 0, "entities": ["Sirius Black", "Azkaban"]}}, {{"idx": 1, "entities": ["Harry Potter", "Hogwarts"]}}]

Viewpoints:
{viewpoints_block}
"""


def step2_entity_extract(viewpoints: List[str], llm: LLMClient, batch_size: int = 50,
                         save_path: str = None, resume: bool = False) -> List[List[str]]:
    """Extract entities and filter viewpoints with no named entities."""
    entities = [None] * len(viewpoints)
    if resume and save_path and os.path.exists(save_path):
        with open(save_path) as f:
            prev = json.load(f)
        for i, ents in enumerate(prev.get("entities", [])):
            if i < len(entities):
                entities[i] = ents
        done = sum(1 for e in entities if e is not None)
        print(f"  [Resume] {done}/{len(viewpoints)} already extracted")

    total_batches = (len(viewpoints) + batch_size - 1) // batch_size

    for batch_start in range(0, len(viewpoints), batch_size):
        batch_end = min(batch_start + batch_size, len(viewpoints))
        batch_idx = batch_start // batch_size + 1

        if all(entities[i] is not None for i in range(batch_start, batch_end)):
            continue

        lines = []
        for i in range(batch_start, batch_end):
            lines.append(f"[{i}] {viewpoints[i]}")
        block = "\n".join(lines)
        prompt = ENTITY_EXTRACT_PROMPT.format(viewpoints_block=block)

        print(f"  [Step2] batch {batch_idx}/{total_batches} ({batch_start}-{batch_end})...", flush=True)
        response = llm.call(prompt)

        try:
            text = response.strip()
            text = re.sub(r'^```(?:json)?\s*', '', text)
            text = re.sub(r'\s*```$', '', text)
            parsed = json.loads(text)
            for item in parsed:
                idx = item.get("idx", -1)
                ents = item.get("entities", [])
                if 0 <= idx < len(entities):
                    entities[idx] = ents
        except (json.JSONDecodeError, KeyError) as e:
            print(f"    [WARN] Failed to parse batch {batch_idx}: {e}", flush=True)

        if save_path and batch_idx % 10 == 0:
            _save_step2(save_path, viewpoints, entities, llm)

    entities = [e if e is not None else [] for e in entities]

    if save_path:
        _save_step2(save_path, viewpoints, entities, llm)

    return entities


def _save_step2(path, viewpoints, entities, llm):
    has_entity = sum(1 for e in entities if e is not None and len(e) > 0)
    total = sum(1 for e in entities if e is not None)
    data = {
        "total": len(viewpoints),
        "extracted": total,
        "with_entities": has_entity,
        "without_entities": total - has_entity,
        "llm_calls": llm.total_calls,
        "input_tokens": llm.total_input_tokens,
        "output_tokens": llm.total_output_tokens,
        "entities": entities,
    }
    with open(path, "w") as f:
        json.dump(data, f)


# ── Step 3: Embedding Deduplication ─────────────────────────────────────

def step3_dedup(viewpoints: List[str], threshold: float = 0.9) -> List[int]:
    """Deduplicate by embedding cosine similarity. Returns indices to keep."""
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
    except ImportError:
        print("  [Step3] sentence-transformers not installed, skipping dedup")
        return list(range(len(viewpoints)))

    print(f"  [Step3] Computing embeddings for {len(viewpoints)} viewpoints...", flush=True)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(viewpoints, show_progress_bar=True, batch_size=256)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    print(f"  [Step3] Finding duplicates (threshold={threshold})...", flush=True)
    keep = set(range(len(viewpoints)))
    removed = set()

    # Batch cosine similarity to avoid O(n^2) memory
    batch = 1000
    for i in range(0, len(viewpoints), batch):
        if i % 5000 == 0:
            print(f"    checking {i}/{len(viewpoints)}...", flush=True)
        end_i = min(i + batch, len(viewpoints))
        for j in range(i + 1, len(viewpoints)):
            if j in removed:
                continue
            if i in removed:
                break
            # Only compare within reasonable range
            sims = embeddings[i:end_i] @ embeddings[j].T
            for k, sim in enumerate(sims):
                idx = i + k
                if idx in removed or idx == j:
                    continue
                if sim > threshold:
                    # Remove the later one
                    removed.add(j)

    keep = sorted(keep - removed)
    print(f"  [Step3] Kept {len(keep)}/{len(viewpoints)} (removed {len(removed)} duplicates)")
    return keep


def step3_dedup_fast(viewpoints: List[str], threshold: float = 0.9) -> List[int]:
    """Fast dedup using hash-based blocking + embedding similarity."""
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
    except ImportError:
        print("  [Step3] sentence-transformers not installed, skipping dedup")
        return list(range(len(viewpoints)))

    print(f"  [Step3] Computing embeddings for {len(viewpoints)} viewpoints...", flush=True)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(viewpoints, show_progress_bar=True, batch_size=256)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings = embeddings / norms

    print(f"  [Step3] Deduplicating...", flush=True)
    removed = set()
    # Compare each viewpoint with nearby ones (sorted by first 3 words for blocking)
    # Simple approach: sequential scan, mark duplicates
    for i in range(len(viewpoints)):
        if i in removed:
            continue
        if i % 5000 == 0:
            print(f"    {i}/{len(viewpoints)}, removed so far: {len(removed)}", flush=True)
        # Compare with next 200 viewpoints (most duplicates are nearby in text)
        end = min(i + 200, len(viewpoints))
        if end <= i + 1:
            continue
        sims = embeddings[i] @ embeddings[i+1:end].T
        for j_offset, sim in enumerate(sims):
            j = i + 1 + j_offset
            if j not in removed and sim > threshold:
                removed.add(j)

    keep = sorted(set(range(len(viewpoints))) - removed)
    print(f"  [Step3] Kept {len(keep)}/{len(viewpoints)} (removed {len(removed)} duplicates)")
    return keep


# ── Step 4: Entity Graph Connectivity ───────────────────────────────────

def step4_connectivity(viewpoints: List[str], entities: List[List[str]],
                       min_shared: int = 2) -> List[int]:
    """Filter viewpoints that don't share entities with at least min_shared other viewpoints."""
    # Normalize entity names
    def normalize(e):
        return e.lower().strip()

    # Build entity -> viewpoint indices mapping
    entity_to_vps = defaultdict(set)
    for i, ents in enumerate(entities):
        for e in ents:
            entity_to_vps[normalize(e)].add(i)

    # Count how many other viewpoints share at least one entity
    keep = []
    for i, ents in enumerate(entities):
        shared_vps = set()
        for e in ents:
            shared_vps.update(entity_to_vps[normalize(e)])
        shared_vps.discard(i)  # exclude self
        if len(shared_vps) >= min_shared:
            keep.append(i)

    print(f"  [Step4] Kept {len(keep)}/{len(viewpoints)} (filtered {len(viewpoints)-len(keep)} isolated viewpoints)")
    return keep


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
                        help="Input viewpoints JSON (from extract_viewpoints_books.py)")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--step", type=str, default="all",
                        choices=["1", "2", "3", "4", "all"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--base_url", type=str, default="https://api.openai.com/v1")
    parser.add_argument("--api_key", type=str,
                        default="YOUR_API_KEY")
    parser.add_argument("--dedup_threshold", type=float, default=0.9)
    parser.add_argument("--min_shared_entities", type=int, default=2)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.input) as f:
        data = json.load(f)
    all_vps = data["unique_viewpoints"]
    print(f"Input: {len(all_vps)} unique viewpoints")

    llm = LLMClient(model=args.model, base_url=args.base_url, api_key=args.api_key)
    run_all = args.step == "all"

    current_vps = all_vps
    current_indices = list(range(len(all_vps)))

    # ── Step 1 ──
    if run_all or args.step == "1":
        print(f"\n{'='*60}")
        print(f"Step 1: LLM Classification ({len(current_vps)} viewpoints)")
        print(f"{'='*60}")
        s1_path = os.path.join(args.output_dir, "step1_classify.json")
        labels = step1_classify(current_vps, llm, save_path=s1_path, resume=args.resume)
        kept_indices = [i for i, l in enumerate(labels) if l]
        kept_vps = [current_vps[i] for i in kept_indices]
        print(f"  Result: {len(kept_vps)}/{len(current_vps)} kept ({len(kept_vps)/len(current_vps)*100:.1f}%)")

        # Save kept viewpoints
        with open(os.path.join(args.output_dir, "after_step1.json"), "w") as f:
            json.dump({"viewpoints": kept_vps, "count": len(kept_vps)}, f, indent=2, ensure_ascii=False)

        current_indices = [current_indices[i] for i in kept_indices]
        current_vps = kept_vps

    elif os.path.exists(os.path.join(args.output_dir, "after_step1.json")):
        with open(os.path.join(args.output_dir, "after_step1.json")) as f:
            current_vps = json.load(f)["viewpoints"]
        current_indices = list(range(len(current_vps)))
        print(f"  [Loaded Step 1 result: {len(current_vps)} viewpoints]")

    # ── Step 2 ──
    if run_all or args.step == "2":
        print(f"\n{'='*60}")
        print(f"Step 2: Entity Extraction ({len(current_vps)} viewpoints)")
        print(f"{'='*60}")
        s2_path = os.path.join(args.output_dir, "step2_entities.json")
        entities = step2_entity_extract(current_vps, llm, save_path=s2_path, resume=args.resume)
        has_entity = [i for i, e in enumerate(entities) if len(e) > 0]
        kept_vps = [current_vps[i] for i in has_entity]
        kept_entities = [entities[i] for i in has_entity]
        print(f"  Result: {len(kept_vps)}/{len(current_vps)} have entities ({len(kept_vps)/len(current_vps)*100:.1f}%)")

        with open(os.path.join(args.output_dir, "after_step2.json"), "w") as f:
            json.dump({
                "viewpoints": kept_vps,
                "entities": kept_entities,
                "count": len(kept_vps),
            }, f, indent=2, ensure_ascii=False)

        current_indices = [current_indices[i] for i in has_entity]
        current_vps = kept_vps
        current_entities = kept_entities

    elif os.path.exists(os.path.join(args.output_dir, "after_step2.json")):
        with open(os.path.join(args.output_dir, "after_step2.json")) as f:
            d = json.load(f)
            current_vps = d["viewpoints"]
            current_entities = d["entities"]
        current_indices = list(range(len(current_vps)))
        print(f"  [Loaded Step 2 result: {len(current_vps)} viewpoints]")

    # ── Step 3 ──
    if run_all or args.step == "3":
        print(f"\n{'='*60}")
        print(f"Step 3: Embedding Deduplication ({len(current_vps)} viewpoints)")
        print(f"{'='*60}")
        keep_indices = step3_dedup_fast(current_vps, threshold=args.dedup_threshold)
        kept_vps = [current_vps[i] for i in keep_indices]
        kept_entities = [current_entities[i] for i in keep_indices]
        print(f"  Result: {len(kept_vps)}/{len(current_vps)} after dedup")

        with open(os.path.join(args.output_dir, "after_step3.json"), "w") as f:
            json.dump({
                "viewpoints": kept_vps,
                "entities": kept_entities,
                "count": len(kept_vps),
            }, f, indent=2, ensure_ascii=False)

        current_indices = [current_indices[i] for i in keep_indices]
        current_vps = kept_vps
        current_entities = kept_entities

    elif os.path.exists(os.path.join(args.output_dir, "after_step3.json")):
        with open(os.path.join(args.output_dir, "after_step3.json")) as f:
            d = json.load(f)
            current_vps = d["viewpoints"]
            current_entities = d["entities"]
        current_indices = list(range(len(current_vps)))
        print(f"  [Loaded Step 3 result: {len(current_vps)} viewpoints]")

    # ── Step 4 ──
    if run_all or args.step == "4":
        print(f"\n{'='*60}")
        print(f"Step 4: Entity Graph Connectivity ({len(current_vps)} viewpoints)")
        print(f"{'='*60}")
        keep_indices = step4_connectivity(current_vps, current_entities,
                                          min_shared=args.min_shared_entities)
        kept_vps = [current_vps[i] for i in keep_indices]
        kept_entities = [current_entities[i] for i in keep_indices]

        # Final output
        final = {
            "viewpoints": kept_vps,
            "entities": kept_entities,
            "count": len(kept_vps),
        }
        final_path = os.path.join(args.output_dir, "final_viewpoints.json")
        with open(final_path, "w") as f:
            json.dump(final, f, indent=2, ensure_ascii=False)
        print(f"  Final: {len(kept_vps)} viewpoints saved to {final_path}")

    # Summary
    print(f"\n{'='*60}")
    print(f"LLM calls: {llm.total_calls}")
    print(f"Input tokens: {llm.total_input_tokens}, Output tokens: {llm.total_output_tokens}")
    cost_in = llm.total_input_tokens / 1e6 * 0.15
    cost_out = llm.total_output_tokens / 1e6 * 0.6
    print(f"Estimated cost (gpt-4o-mini): ${cost_in + cost_out:.4f}")


if __name__ == "__main__":
    main()
