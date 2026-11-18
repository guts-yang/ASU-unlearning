"""Real-asset M0a validation: fact-value reader, positive/NK controls.

Confirmation is never loaded. ``target`` is the positive control (TOFU
fine-tuned pre-unlearning model); the un-finetuned ``base`` is an untrained
reference and is NOT treated as a positive control.
"""

from __future__ import annotations

import gc
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import yaml

from ..activations.extraction import extract_activation_batches
from ..activations.specs import build_spec
from ..baseline_adapter import BaselineAdapter
from ..checkpoints import checkpoint_refs
from ..config import FrozenConfig
from ..data.access_guard import AccessGuard
from ..readers.fact_value import FrozenEmbeddingReader
from ..readers.training import train_edit_reader
from ..types import ActivationBatch, SeedRecord, Split
from ..utils.hashing import sha256_json
from ..utils.seed import set_global_seed

ROLES = ("target", "base", "asu_student")
NK_TEMPLATE = "What is the full name of the author born in {entity}?"


@dataclass(frozen=True)
class FactPrompt:
    fact_id: str
    entity_id: str
    split: Split
    value: str
    prompt: str
    is_nk: bool
    provenance: str


@dataclass(frozen=True)
class M0aReport:
    status: str
    per_model: Mapping[str, Mapping[str, float]]
    seed: SeedRecord
    model_hashes: Mapping[str, str]
    reader_hashes: Mapping[str, str]
    config_hash: str
    gpu_peak_bytes: Mapping[str, int]
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["seed"] = asdict(self.seed)
        data["notes"] = list(self.notes)
        return data


def load_facts(cfg: FrozenConfig, guard: AccessGuard) -> tuple[list[FactPrompt], list[FactPrompt], str]:
    """Load non-confirmation author facts and construct matched NK facts."""
    path = cfg.resolve("manifests/entity_splits.yaml")
    manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
    source = Path(manifest["source_data"])
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()

    real: list[FactPrompt] = []
    nk: list[FactPrompt] = []
    for index, cluster in enumerate(manifest.get("clusters") or ()):
        split = cluster["split"]
        if split == "confirmation":
            continue  # never loaded or used by M0a
        guard.assert_access(split)
        start = int(str(cluster["source_rows"][0]).split("-")[0])
        canonical = str(cluster["canonical_id"])
        value = str(cluster["fact_values"][0])
        real.append(FactPrompt(
            fact_id=f"real::{canonical}",
            entity_id=canonical,
            split=split,
            value=value,
            prompt=str(rows[start]["question"]),
            is_nk=False,
            provenance=f"forget05 row {start}; sha256={source_hash}",
        ))
        nk_entity = f"NK-{index:02d}"
        nk_value = f"NK Author {hashlib.sha256(f'{nk_entity}|{canonical}'.encode()).hexdigest()[:8]}"
        nk.append(FactPrompt(
            fact_id=f"nk::{canonical}",
            entity_id=nk_entity,
            split=split,
            value=nk_value,
            prompt=NK_TEMPLATE.format(entity=nk_entity),
            is_nk=True,
            provenance=f"synthetic; relation=full_name; template={NK_TEMPLATE!r}; seed=fixed",
        ))
    nk_hash = sha256_json([asdict(f) for f in nk])
    return real, nk, nk_hash


def _tokenize(adapter: BaselineAdapter, prompts: Sequence[str], max_len: int) -> dict[str, torch.Tensor]:
    tokenizer = adapter.tokenizer
    if tokenizer is None:
        raise RuntimeError("M0a requires a real tokenizer")
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded = tokenizer(list(prompts), return_tensors="pt", padding=True,
                        truncation=True, max_length=max_len)
    return {key: value.to(adapter.device) for key, value in encoded.items()}


def _batch(adapter: BaselineAdapter, component: str, facts: Sequence[FactPrompt],
           encoded: Mapping[str, torch.Tensor], source_hash: str,
           pooling: str) -> ActivationBatch:
    spec = build_spec(adapter, component, token_position="all", pooling=pooling)
    batch = extract_activation_batches(
        adapter, [spec], encoded["input_ids"],
        tuple(f.entity_id for f in facts),
        tuple(f.fact_id for f in facts),
        facts[0].split,
        tuple(f.value for f in facts),
        attention_mask=encoded.get("attention_mask"),
    )[component]
    return batch


def evaluate_role(
    cfg: FrozenConfig,
    role: str,
    real: Sequence[FactPrompt],
    nk: Sequence[FactPrompt],
    source_hash: str,
) -> tuple[dict[str, float], str, int, SeedRecord]:
    m0a_cfg = cfg.section("readers")["m0a"]
    seed_record = set_global_seed(int(m0a_cfg["seed"]))
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    from ..checkpoints import checkpoint_refs
    refs = checkpoint_refs(cfg)
    adapter = BaselineAdapter.from_checkpoint(
        refs[role], cfg, device="cuda:0" if torch.cuda.is_available() else "cpu",
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    components = [c for c in adapter.component_paths() if c.endswith(str(m0a_cfg["component_kind"]))]
    if not components:
        raise RuntimeError(f"no {m0a_cfg['component_kind']} component in {role}")
    component = components[-1]

    train_facts = [f for f in real if f.split == "forget_train_edit"] + [
        f for f in nk if f.split == "forget_train_edit"
    ]
    held_facts = [f for f in real if f.split == "forget_train_reader"]
    held_nk = [f for f in nk if f.split == "forget_train_reader"]
    if not train_facts or not held_facts or not held_nk:
        raise RuntimeError("insufficient M0a facts in non-confirmation splits")

    train_encoded = _tokenize(adapter, [f.prompt for f in train_facts], int(m0a_cfg["max_prompt_tokens"]))
    held_encoded = _tokenize(adapter, [f.prompt for f in held_facts], int(m0a_cfg["max_prompt_tokens"]))
    nk_encoded = _tokenize(adapter, [f.prompt for f in held_nk], int(m0a_cfg["max_prompt_tokens"]))

    pooling = str(m0a_cfg["pooling"])
    train_batch = _batch(adapter, component, train_facts, train_encoded, source_hash, pooling)
    held_batch = _batch(adapter, component, held_facts, held_encoded, source_hash, pooling)
    nk_batch = _batch(adapter, component, held_nk, nk_encoded, source_hash, pooling)

    values = [f.value for f in real] + [f.value for f in nk]
    if len(values) < int(m0a_cfg["min_candidate_values"]):
        raise RuntimeError("too few fact-value candidates")
    reader = FrozenEmbeddingReader(values, int(train_batch.activations.shape[-1]),
                                   dim=int(cfg.section("readers")["edit_reader"]["dim"]),
                                   seed=int(m0a_cfg["seed"])).to(adapter.device)
    set_global_seed(int(m0a_cfg["seed"]))
    train_edit_reader(reader, [train_batch], int(m0a_cfg["train_steps"]),
                      float(m0a_cfg["learning_rate"]), int(m0a_cfg["seed"]))

    nk_values = tuple(f.value for f in held_nk)
    real_score = reader.score(held_batch, held_batch.fact_ids, held_batch.split, nk_values)
    nk_score = reader.score(nk_batch, nk_batch.fact_ids, nk_batch.split, nk_values)
    metrics = {
        "fact_value_accuracy": real_score.fact_value_accuracy,
        "nk_accuracy": nk_score.fact_value_accuracy,
        "reader_mrr": real_score.mrr or 0.0,
        "reader_confidence": real_score.confidence,
        "candidate_count": float(len(values)),
        "real_samples": float(real_score.n_samples),
        "nk_samples": float(nk_score.n_samples),
    }
    metrics["recovery_advantage"] = metrics["fact_value_accuracy"] - metrics["nk_accuracy"]
    reader_hash = reader.state_dict_hash()
    peak = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0

    del reader, adapter, train_batch, held_batch, nk_batch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics, reader_hash, peak, seed_record


def run_m0a_real(cfg: FrozenConfig, guard: AccessGuard) -> M0aReport:
    m0a_cfg = cfg.section("readers")["m0a"]
    real, nk, nk_hash = load_facts(cfg, guard)
    per_model: dict[str, dict[str, float]] = {}
    hashes: dict[str, str] = {}
    reader_hashes: dict[str, str] = {}
    peaks: dict[str, int] = {}
    seed_record: SeedRecord | None = None

    for role in ROLES:
        metrics, reader_hash, peak, seed = evaluate_role(cfg, role, real, nk, nk_hash)
        per_model[role] = metrics
        hashes[role] = checkpoint_refs(cfg)[role].sha256 or ""
        reader_hashes[role] = reader_hash
        peaks[role] = peak
        seed_record = seed

    assert seed_record is not None
    target_adv = per_model["target"]["recovery_advantage"]
    base_adv = per_model["base"]["recovery_advantage"]
    notes = [
        "target=TOFU 微调的遗忘前模型，为已知事实正对照",
        "base=未经 TOFU 微调的 Llama-2-7b-chat，为未训练参照，不作为正对照",
        f"NK 构造哈希={nk_hash}",
        "reader 候选为具体作者名与 NK 名；不输出 forget/retain 集合标签",
    ]
    failed: list[str] = []
    if target_adv < float(m0a_cfg["positive_margin"]):
        failed.append(f"target 正对照优势 {target_adv:.3f} < {m0a_cfg['positive_margin']}")
    if base_adv > float(m0a_cfg["base_max_advantage"]):
        failed.append(f"base 未训练参照优势 {base_adv:.3f} > {m0a_cfg['base_max_advantage']}")
    if failed:
        notes.extend(failed)
        status = "FAIL"
    else:
        status = "PASS"

    return M0aReport(
        status=status,
        per_model=per_model,
        seed=seed_record,
        model_hashes=hashes,
        reader_hashes=reader_hashes,
        config_hash=cfg.config_hash,
        gpu_peak_bytes=peaks,
        notes=tuple(notes),
    )
