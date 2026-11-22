#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
CORE = BASE / "core"
sys.path.insert(0, str(CORE))

from factpair import (  # noqa: E402
    _span_check,
    build_prompt,
    encode_without_truncation,
    load_tokenizer,
    normalize_template,
    pair_checks,
    read_jsonl,
    relation_family,
    replace_entity,
    tokenizer_hash,
    write_json,
    write_jsonl,
)

WORKSPACE = BASE.parent
DATA_DIR = WORKSPACE / "data" / "factpair"
TOFU_DIR = Path("/jizhicfs/chenyliao/data/datasets/locuslab-TOFU")
MODEL = Path("/jizhicfs/chenyliao/models/open-unlearning/tofu_Llama-3.2-1B-Instruct_full")
MAX_DONORS = 4
MAX_LENGTH = 512


def main() -> None:
    pools = json.loads((DATA_DIR / "entity_pools.json").read_text(encoding="utf-8"))
    tokenizer, real_tokenizer = load_tokenizer(str(MODEL))
    forget_rows = pools["cluster_rows"]["forget05"]
    retain_rows = pools["cluster_rows"]["retain95"]

    author_by_forget_index = {
        row_index: cluster["name"]
        for cluster in pools["D_f"]
        for row_index in cluster["row_indices"]
    }
    donor_names = {cluster["name"].lower() for cluster in pools["D_donor"]}
    author_by_retain_index = {
        row_index: cluster["name"]
        for cluster in pools["D_donor"]
        for row_index in cluster["row_indices"]
    }
    retain_by_family: dict[str, list[dict]] = defaultdict(list)
    for row_index, row in enumerate(retain_rows):
        donor = author_by_retain_index.get(row_index)
        if donor is None:
            continue
        retain_by_family[relation_family(row["question"])].append(
            {"row_index": row_index, "donor": donor, "row": row}
        )

    rows: list[dict] = []
    stats: list[dict] = []
    for row_index, row in enumerate(forget_rows):
        source = author_by_forget_index.get(row_index)
        if source is None:
            continue
        template = normalize_template(row["question"], source)
        family = relation_family(row["question"])
        prompt_plus = build_prompt(
            tokenizer, {"apply_chat_template": True, "system_prompt": None}, row["question"]
        )
        plus_checks = pair_checks(
            prompt_plus=prompt_plus,
            prompt_minus=prompt_plus,
            answer_plus=source,
            answer_minus=source,
            tokenizer=tokenizer,
            real_tokenizer=real_tokenizer,
            max_length=MAX_LENGTH,
        )
        span_plus = plus_checks["entity_span_plus"]
        if span_plus["start"] < 0:
            stats.append({"row_index": row_index, "reason": "invalid_source_span"})
            continue

        donors: list[dict] = []
        seen_authors = set()
        for candidate in retain_by_family.get(family, []):
            donor = candidate["donor"]
            key = donor.lower()
            if key in seen_authors or key == source.lower() or key not in donor_names:
                continue
            prompt_minus = build_prompt(
                tokenizer,
                {"apply_chat_template": True, "system_prompt": None},
                replace_entity(row["question"], source, donor),
            )
            checks = pair_checks(
                prompt_plus=prompt_plus,
                prompt_minus=prompt_minus,
                answer_plus=source,
                answer_minus=donor,
                tokenizer=tokenizer,
                real_tokenizer=real_tokenizer,
                max_length=MAX_LENGTH,
            )
            span_minus = _span_check(prompt_minus, donor, tokenizer)
            ids_plus = checks["prompt_plus_ids"]
            ids_minus = checks["prompt_minus_ids"]
            outside_equal = (
                span_plus["start"] == span_minus["start"]
                and span_plus["length"] == span_minus["length"]
                and ids_plus[: span_plus["start"]] == ids_minus[: span_minus["start"]]
                and ids_plus[span_plus["start"] + span_plus["length"] :]
                == ids_minus[span_minus["start"] + span_minus["length"] :]
            )
            if not (
                checks["token_length_equal"]
                and checks["no_truncation"]
                and outside_equal
                and span_minus["single_contiguous_span"]
                and span_plus["single_contiguous_span"]
                and span_plus["boundary_tokenization_stable"]
                and span_minus["boundary_tokenization_stable"]
                and row["answer"].strip() != candidate["row"]["answer"].strip()
            ):
                continue
            seen_authors.add(key)
            donors.append(
                {
                    "donor_row_index": candidate["row_index"],
                    "donor_author": donor,
                    "answer_minus": candidate["row"]["answer"],
                    "question_minus": replace_entity(row["question"], source, donor),
                    "prompt_minus": prompt_minus,
                    "prompt_minus_ids": ids_minus,
                }
            )
            if len(donors) >= MAX_DONORS:
                break

        if not donors:
            stats.append({"row_index": row_index, "reason": "no_valid_donor"})
            continue
        rows.append(
            {
                "id": f"multi-{row_index:04d}",
                "source_row_index": row_index,
                "donor_row_indices": [donor["donor_row_index"] for donor in donors],
                "source_author": source,
                "donor_authors": [donor["donor_author"] for donor in donors],
                "template_key": template,
                "relation_family": family,
                "question_plus": row["question"],
                "question_minus_list": [donor["question_minus"] for donor in donors],
                "answer_plus": row["answer"],
                "answer_minus_list": [donor["answer_minus"] for donor in donors],
                "prompt_plus": prompt_plus,
                "prompt_minus_list": [donor["prompt_minus"] for donor in donors],
                "prompt_plus_ids": ids_plus,
                "prompt_minus_ids_list": [donor["prompt_minus_ids"] for donor in donors],
                "entity_span": [span_plus["start"], span_plus["length"]],
                "slot_outside_token_ids_equal": True,
                "single_contiguous_span": True,
                "boundary_tokenization_stable": span_plus["boundary_tokenization_stable"],
                "no_truncation": True,
                "g0_status": "PASS" if real_tokenizer else "BLOCKED_TOKENIZER",
            }
        )
        stats.append({"row_index": row_index, "donors": len(donors)})

    write_jsonl(DATA_DIR / "paired_prompts_multidonor.jsonl", rows)
    write_json(
        DATA_DIR / "multidonor_summary.json",
        {
            "rows": len(rows),
            "max_donors": MAX_DONORS,
            "donor_count_histogram": {
                str(count): sum(1 for item in stats if item.get("donors") == count)
                for count in range(1, MAX_DONORS + 1)
            },
            "tokenizer": tokenizer_hash(tokenizer, real_tokenizer),
            "real_tokenizer": real_tokenizer,
        },
    )
    print(json.dumps({"rows": len(rows), "stats": stats[:3]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
