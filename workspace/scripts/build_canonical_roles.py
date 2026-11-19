#!/usr/bin/env python3
"""Build canonical full-Df/Dr roles and preregistration for K3 v1."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))

import yaml

from recodeloop.utils.hashing import sha256_file, sha256_json

TOFU = Path("/jizhicfs/chenyliao/data/datasets/locuslab-TOFU")
OUT = WORKSPACE / "outputs" / "canonical_v1"
MANIFEST = WORKSPACE / "manifests" / "canonical_v1"
BLOCK_SIZE = 20


def extract_name(answer: str) -> str:
    cap = r"([A-Z][A-Za-z'\-]+(?: [A-Z][A-Za-z'\-]+)+)"
    for pattern in (
        rf"author's name is {cap}", rf"author's full name is {cap}",
        rf"is named {cap}", rf"is called {cap}", rf"is {cap}, an ", rf"is {cap}",
    ):
        match = re.search(pattern, answer)
        if match:
            return match.group(1).strip(" .,")
    raise ValueError(f"cannot extract name from {answer[:120]!r}")


def aliases_for(name: str) -> list[str]:
    tokens = name.split()
    short = " ".join([tokens[0], tokens[-1]]) if len(tokens) > 2 else name
    return list(dict.fromkeys(
        c for c in (name, name.lower(), short, short.lower(), name.replace("-", " ")) if c
    ))


def code_hash() -> str:
    digest = hashlib.sha256()
    files = sorted((WORKSPACE / "src").rglob("*.py")) + sorted((WORKSPACE / "scripts").glob("*.py"))
    for path in files:
        digest.update(str(path.relative_to(WORKSPACE)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    MANIFEST.mkdir(parents=True, exist_ok=True)

    df_source = TOFU / "forget05.json"
    dr_near = TOFU / "retain95.json"
    dr_general = [TOFU / "real_authors.json", TOFU / "world_facts.json"]
    dev_source = TOFU / "forget10.json"
    df_rows = [json.loads(x) for x in df_source.read_text().splitlines() if x.strip()]
    if len(df_rows) != 200:
        raise ValueError(f"forget05 rows={len(df_rows)}, expected 200")

    clusters = []
    for block in range(10):
        start = block * BLOCK_SIZE
        rows = df_rows[start:start + BLOCK_SIZE]
        name = extract_name(rows[0]["answer"])
        clusters.append({
            "canonical_id": name,
            "aliases": aliases_for(name),
            "source_rows": [f"{start}-{start + BLOCK_SIZE - 1}"],
            "expected_rows": BLOCK_SIZE,
        })
    clusters.sort(key=lambda c: c["canonical_id"])

    roles = {
        "schema_version": "recodeloop-canonical-v1",
        "experiment_id": "P08-M2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "design": {
            "D_f": "complete forget05; every row is an unlearning target and a final forgetting-check item",
            "D_r": "complete retain95 + real_authors + world_facts; retain constraint and capability check",
            "reader_crossfit": "entity-cluster folds only for reader train/eval; never excludes entities from model updates",
            "confirmation": "none; final audit on complete D_f is post-hoc for the current request",
        },
        "D_f": {
            "source": str(df_source),
            "sha256": sha256_file(df_source),
            "row_count": len(df_rows),
            "clusters": clusters,
        },
        "D_r": {
            "near": {"source": str(dr_near), "sha256": sha256_file(dr_near),
                     "row_count": sum(1 for _ in dr_near.open())},
            "general": [{"source": str(p), "sha256": sha256_file(p),
                         "row_count": sum(1 for _ in p.open())} for p in dr_general],
        },
        "NK": {
            "construction": "one matched synthetic fact per D_f row; alias substitution in question and answer; deterministic from row index and canonical id",
            "count": len(df_rows),
        },
        "development": {"source": str(dev_source), "sha256": sha256_file(dev_source),
                        "use": "pilot/constraint feasibility only; never canonical result"},
        "reader_crossfit": {"folds": 5, "assignment": "canonical_id modulo 5",
                            "rule": "reader train on 4 folds, evaluate on held fold; repeat all folds"},
        "retain_sample": {"near_rows": 128, "general_rows": 32, "seed": 1001,
                          "rule": "deterministic seeded sample, frozen before main run"},
        "canonical_hash": "",
    }
    roles["canonical_hash"] = sha256_json(roles)
    manifest_path = MANIFEST / "dataset_roles.yaml"
    manifest_path.write_text(yaml.safe_dump(roles, allow_unicode=True, sort_keys=False), encoding="utf-8")

    assets = yaml.safe_load((WORKSPACE / "manifests" / "assets.yaml").read_text())
    g0_checks = []
    for role, ref in assets["checkpoints"].items():
        p = Path(ref["path"])
        g0_checks.append({"role": role, "path": str(p), "exists": p.exists(),
                          "expected_sha256": ref.get("sha256")})
    g0 = {
        "status": "PASS" if all(c["exists"] for c in g0_checks) else "FAIL",
        "dataset_roles_manifest": str(manifest_path),
        "dataset_roles_sha256": sha256_file(manifest_path),
        "checkpoint_checks": g0_checks,
        "df_rows": len(df_rows),
        "dr_near_rows": roles["D_r"]["near"]["row_count"],
        "nk_count": len(df_rows),
    }
    (OUT / "g0.json").write_text(json.dumps(g0, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    prereg = {
        "version": "canonical-v1",
        "created_utc": roles["created_utc"],
        "dataset_roles_sha256": roles["canonical_hash"],
        "code_sha256": code_hash(),
        "constants": {
            "T_max": 3, "steps_per_round": 2, "candidate_ranks": [4],
            "components_per_round": [1], "energy_g": 1.0, "lambda_ret": 1.0,
            "patch_threshold": -1.0, "seed": 1001, "generation_K": 8,
            "bootstrap_samples": 10000, "ci_level": 0.95,
            "equivalence_margin": 0.01, "min_related_effect": 0.02,
            "retain_max_abs_drop": 0.01, "retain_max_kl": 0.02,
        },
        "gates": {"G0": g0["status"], "G1": "pending", "G2": "pending",
                  "G3": "pending", "G4": "pending", "G5": "pending"},
        "note": "Frozen before canonical main run; no result-based tuning allowed.",
    }
    (OUT / "preregistration.json").write_text(json.dumps(prereg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"g0": g0["status"], "manifest": str(manifest_path),
                      "prereg": str(OUT / "preregistration.json")}, indent=2))
    return 0 if g0["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
