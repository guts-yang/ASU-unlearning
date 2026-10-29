#!/usr/bin/env python3
"""Freeze E0 calibration/holdout manifests before any teacher forward."""

import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path("/usr/local/ASU-unlearning")
CONFIG_PATH = ROOT / "workspace/H/configs/e0_probe.yaml"
STOP = {
    "The", "What", "Who", "How", "Can", "Could", "Where", "When", "Which", "Why",
    "Has", "Have", "Did", "Does", "Are", "Was", "Were", "This", "That", "From",
    "With", "About", "Into", "Over", "After", "Before", "Based", "Other",
    "Please", "Would", "Should", "There", "Their", "Some", "Many", "Most",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path):
    rows = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def squash(text):
    text = unicodedata.normalize("NFKD", text)
    return "".join(char for char in text if not unicodedata.combining(char))


def author_label(questions):
    counts = Counter()
    squashed = [squash(question) for question in questions]
    for question in squashed:
        words = [re.sub(r"[^A-Za-z'-]", "", word) for word in question.split()[1:]]
        for word in words:
            if re.fullmatch(r"[A-Z][a-zA-Z'-]{2,}", word or "") and word not in STOP:
                counts[word] += 1
    if not counts:
        return "unnamed", 0
    word = counts.most_common(1)[0][0]
    covered = sum(word in question for question in squashed)
    for candidate, _ in counts.most_common(8):
        candidate_covered = sum(candidate in question for question in squashed)
        if candidate_covered > covered:
            word, covered = candidate, candidate_covered
    return word, covered


def blocks(rows, size):
    if len(rows) % size != 0:
        raise RuntimeError(f"row count {len(rows)} is not divisible by {size}")
    return [rows[start:start + size] for start in range(0, len(rows), size)]


def mean_length(rows):
    return sum(len(row["question"]) for row in rows) / len(rows)


def main():
    config = yaml.safe_load(CONFIG_PATH.read_text())
    size = int(config["questions_per_author"])
    forget_rows = [
        row for row in read_jsonl(config["forget_path"])
        if str(row["task_id"]) == str(config["task_id"])
    ]
    retain_rows = read_jsonl(config["retain_path"])
    forget_blocks = blocks(forget_rows, size)
    retain_blocks = blocks(retain_rows, size)
    if len(forget_blocks) != 10:
        raise RuntimeError(f"expected 10 forget authors, found {len(forget_blocks)}")

    forget_meta = []
    for index, block in enumerate(forget_blocks):
        label, covered = author_label([row["question"] for row in block])
        if covered < 15:
            raise RuntimeError(f"forget author {index} label {label} covers {covered}/20")
        forget_meta.append({
            "author_index": index,
            "label": label,
            "label_coverage": covered,
            "mean_question_chars": mean_length(block),
        })
    for previous, current in zip(forget_meta, forget_meta[1:]):
        if previous["label"] == current["label"]:
            raise RuntimeError(f"adjacent forget blocks share label {current['label']}")

    retain_meta = []
    for index, block in enumerate(retain_blocks):
        label, covered = author_label([row["question"] for row in block])
        retain_meta.append({
            "author_index": index,
            "label": label if covered >= 10 else f"block_{index:03d}",
            "label_coverage": covered,
            "mean_question_chars": mean_length(block),
            "source_start": index * size,
        })
    for previous, current in zip(retain_meta, retain_meta[1:]):
        if previous["label_coverage"] >= 15 and current["label_coverage"] >= 15 and previous["label"] == current["label"]:
            raise RuntimeError(f"adjacent retain blocks share label {current['label']}")

    calibration = set(config["calibration_author_indices"])
    holdout = set(config["holdout_author_indices"])
    if calibration & holdout:
        raise RuntimeError("calibration and holdout overlap")
    if calibration | holdout != set(range(len(forget_blocks))):
        raise RuntimeError("author split does not cover every forget author")

    unused = set(range(len(retain_blocks)))
    retain_assignment = {}
    for author_index in range(len(forget_blocks)):
        target = forget_meta[author_index]["mean_question_chars"]
        choice = min(
            unused,
            key=lambda index: (abs(retain_meta[index]["mean_question_chars"] - target), index),
        )
        unused.remove(choice)
        retain_assignment[author_index] = choice

    samples = []
    for author_index, block in enumerate(forget_blocks):
        split = "calibration" if author_index in calibration else "holdout"
        for offset, row in enumerate(block):
            samples.append({
                "sample_id": f"F-{author_index:02d}-{offset:02d}",
                "subset": "forget",
                "split": split,
                "author_index": author_index,
                "author_label": forget_meta[author_index]["label"],
                "question": row["question"],
                "answer": row["answer"],
            })
    for author_index, retain_index in retain_assignment.items():
        split = "calibration" if author_index in calibration else "holdout"
        block = retain_blocks[retain_index]
        for offset, row in enumerate(block):
            samples.append({
                "sample_id": f"R-{author_index:02d}-{offset:02d}",
                "subset": "retain",
                "split": split,
                "author_index": retain_index,
                "matched_forget_author": author_index,
                "author_label": retain_meta[retain_index]["label"],
                "question": row["question"],
                "answer": row["answer"],
            })

    manifest = {
        "config_path": str(CONFIG_PATH),
        "config_sha256": sha256_file(CONFIG_PATH),
        "forget_sha256": sha256_file(config["forget_path"]),
        "retain_sha256": sha256_file(config["retain_path"]),
        "split_rule": "even forget author blocks are calibration; odd blocks are holdout",
        "retain_match_rule": "greedy nearest mean question length, tie broken by smaller retain block index",
        "forget_authors": forget_meta,
        "retain_authors_used": [
            {**retain_meta[retain_index], "matched_forget_author": author_index}
            for author_index, retain_index in sorted(retain_assignment.items())
        ],
        "counts": {
            "calibration_forget": sum(row["split"] == "calibration" and row["subset"] == "forget" for row in samples),
            "holdout_forget": sum(row["split"] == "holdout" and row["subset"] == "forget" for row in samples),
            "calibration_retain": sum(row["split"] == "calibration" and row["subset"] == "retain" for row in samples),
            "holdout_retain": sum(row["split"] == "holdout" and row["subset"] == "retain" for row in samples),
        },
        "samples": samples,
    }
    destination = Path(config["manifest_path"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"manifest": str(destination), "counts": manifest["counts"]}, indent=2))


if __name__ == "__main__":
    main()
