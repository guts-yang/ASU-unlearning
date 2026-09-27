#!/usr/bin/env python3
"""MQuAKE multi-hop data augmentation pipeline.

Usage:
    python -m MQuAKE.augmentation.run_augmentation --type all --data DATA.json --output OUT_DIR
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

from .data_loader import load_facts, extract_entity_aliases, FactIndex
from .matchers import MATCHERS
from .matchers.identity_substitution import match_identity_substitution
from .llm_generator import get_client, generate_batch
from .formatter import format_all

ALL_TYPES = list(MATCHERS.keys())

TYPE_ID_OFFSETS = {
    "chain_implication": 10000,
    "instance_subsumption": 20000,
    "identity_substitution": 30000,
    "multi_element_intersection": 40000,
    "disjunctive_elimination": 50000,
    "contrapositive_reasoning": 60000,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MQuAKE multi-hop data augmentation")
    p.add_argument("--type", type=str, default="all",
                   choices=ALL_TYPES + ["all"],
                   help="Which reasoning type to generate (or 'all')")
    p.add_argument("--data", type=str, required=True,
                   help="Path to MQuAKE-CF-3k-v2.json")
    p.add_argument("--output", type=str, required=True,
                   help="Output directory for generated JSON files")
    p.add_argument("--max-samples", type=int, default=500,
                   help="Max samples per type")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--dry-run", action="store_true",
                   help="Only run matching, skip LLM calls")
    return p.parse_args()


def run_type(type_name: str, index: FactIndex, args: argparse.Namespace,
             client=None, entity_aliases: dict | None = None) -> None:
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"Processing: {type_name}", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

    # Step 1: Match
    if type_name == "identity_substitution":
        matched = match_identity_substitution(index, entity_aliases)
    else:
        matcher = MATCHERS[type_name]
        matched = matcher(index)
    print(f"  Matched pairs: {len(matched)}", file=sys.stderr)

    if not matched:
        print(f"  No matches found for {type_name}, skipping.", file=sys.stderr)
        return

    # Step 2: Sample
    rng = random.Random(args.seed)
    if len(matched) > args.max_samples:
        matched = rng.sample(matched, args.max_samples)
        print(f"  Sampled down to: {len(matched)}", file=sys.stderr)

    if args.dry_run:
        print(f"  [dry-run] First 3 matched pairs:", file=sys.stderr)
        for pair in matched[:3]:
            summary = {k: v for k, v in pair.items()
                       if k not in ("fact1", "fact2", "fact", "implicit_evidence",
                                    "explicit_evidence")}
            print(f"    {json.dumps(summary, ensure_ascii=False)[:300]}", file=sys.stderr)
        return

    # Step 3: Generate questions via LLM
    generated = generate_batch(client, type_name, matched, args.temperature)
    print(f"  Successfully generated: {len(generated)}/{len(matched)}", file=sys.stderr)

    if not generated:
        print(f"  No successful generations for {type_name}.", file=sys.stderr)
        return

    # Step 4: Format and save
    case_id_start = TYPE_ID_OFFSETS.get(type_name, 0) + 1
    cases = format_all(type_name, generated, case_id_start)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{type_name}.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)

    print(f"  Saved {len(cases)} cases to: {out_path}", file=sys.stderr)


def main() -> None:
    args = parse_args()

    print(f"Loading data from: {args.data}", file=sys.stderr)
    facts = load_facts(args.data)
    index = FactIndex(facts)
    print(f"Loaded {len(facts)} facts from dataset", file=sys.stderr)

    # Extract entity aliases for identity_substitution
    entity_aliases = extract_entity_aliases(args.data)
    print(f"Extracted aliases for {len(entity_aliases)} entities", file=sys.stderr)

    client = None
    if not args.dry_run:
        if not os.environ.get("OPENAI_API_KEY"):
            print("Error: OPENAI_API_KEY not set", file=sys.stderr)
            sys.exit(1)
        client = get_client()

    types_to_run = ALL_TYPES if args.type == "all" else [args.type]
    for type_name in types_to_run:
        run_type(type_name, index, args, client, entity_aliases)

    print(f"\n{'='*60}", file=sys.stderr)
    print("All done!", file=sys.stderr)


if __name__ == "__main__":
    main()
