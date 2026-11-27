"""Independent train/diagnose/evaluate driver for this CF3 method only."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from data import build_pairs, build_teacher_cache, load_cache, load_records, save_cache, synthetic_records
from modeling import create_model_and_tokenizer, encode_qa, evaluate_ground_truth, forward_model, generation_behaviour, save_trained, span_hidden


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_value(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def apply_override(cfg, expression):
    key, raw = expression.split("=", 1)
    node = cfg
    parts = key.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = parse_value(raw)


def load_config(path, profile, overrides):
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    profiles = {
        "tiny": {"model": {"backend": "tiny", "peft": False, "dtype": "float32"},
                 "data": {"synthetic": True, "max_pairs": 4, "retain_cache_size": 8, "eval_samples": 2,
                          "generation_samples": 1, "max_len": 64},
                 "training": {"steps": 2, "batch_size": 2, "log_every": 1, "checkpoint_every": 0},
                 "evaluation": {"probe_epochs": 5, "relearning_steps": 1, "relearning_samples": 2},
                 "method": {"strict_gate": False}, "layers": [1, 2, 3, 4]},
        "smoke": {"data": {"max_pairs": 4, "retain_cache_size": 8, "eval_samples": 2, "generation_samples": 1},
                  "training": {"steps": 2, "batch_size": 1, "log_every": 1, "checkpoint_every": 0},
                  "evaluation": {"probe_samples": 8, "probe_epochs": 10, "relearning_steps": 1, "relearning_samples": 2},
                  "method": {"strict_gate": False}},
        "pilot": {},
        "full": {"data": {"max_pairs": 400, "retain_cache_size": 128, "eval_samples": 400, "generation_samples": 16},
                 "training": {"steps": cfg["training"].get("full_steps", 600),
                              "batch_size": cfg["training"].get("full_batch_size", 2)}},
    }
    if profile not in profiles:
        raise ValueError(f"unknown profile {profile}")

    def merge(dst, src):
        for key, value in src.items():
            if isinstance(value, dict) and isinstance(dst.get(key), dict):
                merge(dst[key], value)
            else:
                dst[key] = value
    merge(cfg, profiles[profile])
    cfg["profile"] = profile
    for expression in overrides:
        apply_override(cfg, expression)
    return cfg


def resolve_data(cfg):
    if cfg["data"].get("synthetic", False):
        records = synthetic_records(int(cfg["data"].get("synthetic_authors", 12)))
        cut = len(records) // 2
        return records[:cut], records[cut:], records[:cut]
    root = Path(cfg["data"]["root"])
    forget = load_records(root / cfg["data"].get("forget_file", "forget10.json"))
    retain = load_records(root / cfg["data"].get("retain_file", "retain90.json"))
    holdout = load_records(root / cfg["data"].get("holdout_file", "holdout10.json"))
    return forget, retain, holdout


def neutral_terms(item, plus, cfg, device):
    teacher_nll = item["teacher_nll"].to(device)
    target = teacher_nll + float(cfg["loss"].get("fact_delta", 0.35))
    fact = F.relu(plus["nll"] - target).pow(2)
    donor = plus["nll"].new_zeros(())
    count = 0
    if cfg["loss"].get("donor_guard_weight", 0.0) > 0:
        for layer in cfg["layers"]:
            hp = item["h_plus"][layer].to(device)
            hm = item["h_minus"][layer].to(device)
            sh = span_hidden(plus, layer)
            n = min(len(sh), len(hp), len(hm))
            if n == 0:
                continue
            direction = F.normalize(hm[:n] - hp[:n], dim=-1, eps=1e-8)
            donor = donor + F.relu(F.cosine_similarity(sh[:n] - hp[:n], direction, dim=-1, eps=1e-8)).pow(2).mean()
            count += 1
    donor = donor / max(1, count)
    eos = plus["nll"].new_zeros(())
    eos_weight = float(cfg["loss"].get("eos_guard_weight", 0.0))
    if eos_weight > 0:
        start, end = plus["enc"]["answer_start"], plus["enc"]["answer_end"]
        logits = plus["logits"][start:max(start, end - 1)].float()
        if logits.numel() > 0:
            eos_id = int(cfg["model"].get("eos_token_id", 2))
            if eos_id < logits.size(-1):
                without = logits.clone()
                without[:, eos_id] = -torch.inf
                eos = F.relu(logits[:, eos_id] - without.max(dim=-1).values + float(cfg["loss"].get("eos_margin", 1.0))).pow(2).mean()
    total = fact + float(cfg["loss"].get("donor_guard_weight", 0.0)) * donor + eos_weight * eos
    return total, {"fact_hinge": fact.detach(), "donor_guard": donor.detach(), "eos_guard": eos.detach()}


def _forward(student, enc, device, cfg, method, grad=True):
    return forward_model(student, enc, device, set(cfg["layers"]), grad=grad,
                         capture_bilinear=bool(getattr(method, "NEEDS_STUDENT_BILINEAR", False)),
                         capture_glu=bool(getattr(method, "NEEDS_STUDENT_GLU", False)))


def train(student, tok, cache, retain_records, prepared, cfg, device, method, out_dir, resume=False, teacher=None):
    params = [p for p in student.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("Student has no trainable parameters")
    opt = torch.optim.AdamW(params, lr=float(cfg["training"]["lr"]), weight_decay=float(cfg["training"].get("weight_decay", 0.0)))
    start_step = 0
    state_path = out_dir / "training_state.pt"
    if resume and state_path.exists():
        state = torch.load(state_path, map_location=device, weights_only=False)
        student.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        start_step = int(state["step"])
    rng = random.Random(int(cfg["seed"]) + start_step)
    order = list(range(len(cache["pairs"])))
    retain_order = list(range(len(retain_records)))
    history = []
    student.train()
    layers = set(cfg["layers"])
    steps = int(cfg["training"]["steps"])
    batch_size = int(cfg["training"]["batch_size"])
    for step in range(start_step, steps):
        refresh = int(cfg["method"].get("rollout_refresh", 0))
        if refresh > 0 and step > start_step and step % refresh == 0 and hasattr(method, "refresh"):
            prepared = method.refresh(student, teacher, tok, cache, prepared, cfg, device)
        if step % max(1, len(order)) == 0:
            rng.shuffle(order)
        if step % max(1, len(retain_order)) == 0:
            rng.shuffle(retain_order)
        batch = []
        for offset in range(batch_size):
            item = cache["pairs"][order[(step * batch_size + offset) % len(order)]]
            rec = retain_records[retain_order[(step * batch_size + offset) % len(retain_order)]]
            plus = _forward(student, item["enc_plus"], device, cfg, method)
            minus = _forward(student, item["enc_minus"], device, cfg, method) if getattr(method, "NEEDS_STUDENT_MINUS", False) else None
            retain_enc = encode_qa(tok, rec["question"], rec["answer"], int(cfg["data"]["max_len"]))
            retain = _forward(student, retain_enc, device, cfg, method)
            control_plus = _forward(student, item["enc_control_plus"], device, cfg, method) if getattr(method, "NEEDS_STUDENT_CONTROL", False) and "enc_control_plus" in item else None
            control_minus = _forward(student, item["enc_control_minus"], device, cfg, method) if getattr(method, "NEEDS_STUDENT_CONTROL", False) and "enc_control_minus" in item else None
            batch.append({"item": item, "plus": plus, "minus": minus, "retain": retain,
                          "control_plus": control_plus, "control_minus": control_minus,
                          "student": student, "tok": tok, "device": device, "step": step})
        terms = method.loss(batch, prepared, cfg)
        neutral = plus["nll"].new_zeros(())
        neutral_parts = {}
        for row in batch:
            value, parts = neutral_terms(row["item"], row["plus"], cfg, device)
            neutral = neutral + value
            for key, val in parts.items():
                neutral_parts[key] = neutral_parts.get(key, 0.0) + float(val)
        neutral = neutral / len(batch)
        retain_nll = torch.stack([row["retain"]["nll"] for row in batch]).mean()
        loss = (float(cfg["loss"]["method_weight"]) * terms["method_loss"] +
                float(cfg["loss"]["neutral_weight"]) * neutral + float(cfg["loss"]["retain_weight"]) * retain_nll)
        if "extra_loss" in terms:
            loss = loss + terms["extra_loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, float(cfg["training"].get("grad_clip", 1.0)))
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(f"non-finite gradient at step {step}: {grad_norm}")
        if float(grad_norm) == 0.0:
            raise FloatingPointError(f"zero Student gradient at step {step}; refusing a vacuous run")
        opt.step()
        if step % int(cfg["training"].get("log_every", 10)) == 0 or step + 1 == steps:
            row = {"step": step, "loss": float(loss.detach()), "method_loss": float(terms["method_loss"].detach()),
                   "neutral": float(neutral.detach()), "retain_nll": float(retain_nll.detach()), "grad_norm": float(grad_norm)}
            row.update({k: v / len(batch) for k, v in neutral_parts.items()})
            for key, val in terms.items():
                if key not in {"method_loss", "extra_loss"} and isinstance(val, torch.Tensor) and val.numel() == 1:
                    row[key] = float(val.detach())
            history.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
        if int(cfg["training"].get("checkpoint_every", 0)) > 0 and (step + 1) % int(cfg["training"]["checkpoint_every"]) == 0:
            torch.save({"model": student.state_dict(), "optimizer": opt.state_dict(), "step": step + 1}, state_path)
    student.eval()
    return {"history": history, "steps_completed": steps - start_step, "resumed_from": start_step}


def evaluate_pair_donor(student, tok, pairs, cfg, device):
    values = []
    for pair in pairs[: int(cfg["data"]["eval_samples"])]:
        if not pair.get("answer_minus"):
            continue
        enc = encode_qa(tok, pair["x_plus"], pair["answer_minus"], int(cfg["data"]["max_len"]))
        values.append(float(forward_model(student, enc, device)["nll"]))
    return {"donor_nll": sum(values) / len(values) if values else None, "count": len(values),
            "note": "evaluation only; donor answers never enter training loss"}


@torch.no_grad()
def _probe_features(model, tok, records, cfg, device, layer):
    features, labels, groups = [], [], []
    for record in records:
        author = record.get("author") or record["question"]
        enc = encode_qa(tok, record["question"], record["answer"], int(cfg["data"]["max_len"]))
        out = forward_model(model, enc, device, {layer})
        start, end = enc["answer_start"], enc["answer_end"]
        features.append(out["hidden"][layer][start:end].mean(0).detach().float().cpu())
        labels.append(author)
        groups.append(record["question"].replace(author, "<SUBJECT>"))
    return torch.stack(features), labels, groups


def target_fact_probe(model, tok, forget, cfg, device):
    records = list(forget)[:int(cfg["evaluation"].get("probe_samples", 64))]
    layer = min(int(cfg["evaluation"].get("probe_layer", max(cfg["layers"]))), int(model.config.num_hidden_layers))
    features, author_names, groups = _probe_features(model, tok, records, cfg, device, layer)
    classes = sorted(set(author_names))
    if len(classes) < 2:
        return {"status": "INSUFFICIENT_CLASSES", "classes": len(classes)}
    class_index = {name: i for i, name in enumerate(classes)}
    labels = torch.tensor([class_index[x] for x in author_names])
    unique_groups = sorted(set(groups))
    rng = random.Random(int(cfg["seed"]) + 811)
    rng.shuffle(unique_groups)
    test_groups = set(unique_groups[:max(1, len(unique_groups) // 5)])
    train_idx = [i for i, group in enumerate(groups) if group not in test_groups]
    test_idx = [i for i, group in enumerate(groups) if group in test_groups]
    if not train_idx or not test_idx or len(set(labels[train_idx].tolist())) < 2:
        return {"status": "INSUFFICIENT_RELATION_SPLIT", "train": len(train_idx), "test": len(test_idx)}
    train_x, test_x = features[train_idx].to(device), features[test_idx].to(device)
    train_y, test_y = labels[train_idx].to(device), labels[test_idx].to(device)
    mean, std = train_x.mean(0), train_x.std(0).clamp_min(1e-5)
    train_x, test_x = (train_x - mean) / std, (test_x - mean) / std
    torch.manual_seed(int(cfg["seed"]) + 812)
    probe = torch.nn.Linear(train_x.shape[-1], len(classes)).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=0.01)
    for _ in range(int(cfg["evaluation"].get("probe_epochs", 100))):
        optimizer.zero_grad(set_to_none=True)
        F.cross_entropy(probe(train_x), train_y).backward()
        optimizer.step()
    with torch.no_grad():
        logits = probe(test_x)
        accuracy = (logits.argmax(-1) == test_y).float().mean()
        loss = F.cross_entropy(logits, test_y)
    return {"status": "COMPLETED", "layer": layer, "train": len(train_idx), "test": len(test_idx),
            "classes": len(classes), "relation_disjoint": True, "accuracy": float(accuracy),
            "cross_entropy": float(loss), "chance_accuracy": 1.0 / len(classes),
            "scope": "forget-author hidden-state separability; not proof of factual erasure"}


def relearning_recovery(student, tok, forget, cfg, device):
    steps = int(cfg["evaluation"].get("relearning_steps", 10))
    samples = int(cfg["evaluation"].get("relearning_samples", 8))
    records = list(forget)[:samples]
    before = evaluate_ground_truth(student, tok, records, device, int(cfg["data"]["max_len"]), samples)
    if steps <= 0:
        return {"status": "SKIPPED", "steps": 0}
    params = [p for p in student.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=float(cfg["evaluation"].get("relearning_lr", cfg["training"]["lr"])))
    rng = random.Random(int(cfg["seed"]) + 901)
    order = list(range(len(records)))
    student.train()
    for step in range(steps):
        if step % len(order) == 0:
            rng.shuffle(order)
        record = records[order[step % len(order)]]
        enc = encode_qa(tok, record["question"], record["answer"], int(cfg["data"]["max_len"]))
        nll = forward_model(student, enc, device, grad=True)["nll"]
        optimizer.zero_grad(set_to_none=True)
        nll.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, float(cfg["training"].get("grad_clip", 1.0)))
        if not torch.isfinite(grad_norm):
            raise FloatingPointError("non-finite relearning gradient")
        optimizer.step()
    student.eval()
    after = evaluate_ground_truth(student, tok, records, device, int(cfg["data"]["max_len"]), samples)
    return {"status": "COMPLETED", "steps": steps, "samples": samples,
            "before_nll": before["nll"], "after_nll": after["nll"],
            "nll_recovery": before["nll"] - after["nll"], "ground_truth": "original dataset answers",
            "note": "runs after the pre-relearning Student checkpoint has been saved"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(HERE / "config.json"))
    ap.add_argument("--profile", default="pilot", choices=["tiny", "smoke", "pilot", "full"])
    ap.add_argument("--set", action="append", default=[], dest="overrides", help="dotted.key=json_value")
    ap.add_argument("--output", default=None)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--diagnose-only", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.profile, args.overrides)
    import method

    if args.self_test:
        print(json.dumps(method.self_test(), indent=2, ensure_ascii=False))
        return 0
    set_seed(int(cfg["seed"]))
    device = cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output) if args.output else HERE / "runs" / f"{cfg['profile']}_seed{cfg['seed']}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "resolved_config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    started = time.time()
    teacher, tok = create_model_and_tokenizer(cfg, device, student=False)
    student, _ = create_model_and_tokenizer(cfg, device, student=True)
    if cfg["model"].get("backend") == "tiny":
        student.load_state_dict(teacher.state_dict())
    cfg["model"]["eos_token_id"] = tok.eos_token_id
    (out_dir / "resolved_config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    forget, retain, holdout = resolve_data(cfg)
    pair_path = cfg["data"].get("pairs_file")
    pairs = load_records(pair_path) if pair_path else build_pairs(
        tok, forget, retain, int(cfg["data"]["max_pairs"]), int(cfg["seed"]), getattr(method, "REQUIRES_CONTROL", False))
    cache_path = out_dir / "teacher_cache.pt"
    if cache_path.exists() and cfg["data"].get("reuse_cache", True):
        cache = load_cache(cache_path)
    else:
        cache = build_teacher_cache(teacher, tok, pairs, retain, cfg, device,
                                    capture_bilinear=getattr(method, "NEEDS_TEACHER_BILINEAR", False),
                                    capture_glu=getattr(method, "NEEDS_TEACHER_GLU", False))
        save_cache(cache, cache_path)
    if set(cache.get("layers", [])) != set(cfg["layers"]) or len(cache.get("pairs", [])) != len(pairs):
        raise RuntimeError("teacher cache does not match current layers/pairs; use a new output or data.reuse_cache=false")
    required_cache = {"NEEDS_TEACHER_BILINEAR": "attention_plus", "NEEDS_TEACHER_GLU": "glu_plus"}
    for flag, key in required_cache.items():
        if getattr(method, flag, False) and not cache["pairs"][0].get(key):
            raise RuntimeError(f"teacher cache lacks {key}; use data.reuse_cache=false")
    student.eval()
    initial_nll = float(forward_model(student, cache["pairs"][0]["enc_plus"], device)["nll"])
    initial_nll_delta = abs(initial_nll - float(cache["pairs"][0]["teacher_nll"]))
    initial_tolerance = float(cfg["model"].get("initial_nll_tolerance", 0.0001))
    initial_equivalent = initial_nll_delta <= initial_tolerance
    if not initial_equivalent:
        raise RuntimeError(f"initial Student/Teacher mismatch: |ΔNLL|={initial_nll_delta} > {initial_tolerance}")
    prepared = method.prepare(cache, teacher, tok, cfg, device, student=student, out_dir=out_dir)
    diagnostics = method.diagnostics(cache, prepared, cfg)
    (out_dir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    before = {
        "forget": evaluate_ground_truth(teacher, tok, forget, device, int(cfg["data"]["max_len"]), int(cfg["data"]["eval_samples"])),
        "retain": evaluate_ground_truth(teacher, tok, retain, device, int(cfg["data"]["max_len"]), int(cfg["data"]["eval_samples"])),
        "holdout": evaluate_ground_truth(teacher, tok, holdout, device, int(cfg["data"]["max_len"]), int(cfg["data"]["eval_samples"])),
        "donor": evaluate_pair_donor(teacher, tok, pairs, cfg, device),
    }
    status = "DIAGNOSED" if args.diagnose_only else "PENDING"
    train_result = {"history": [], "steps_completed": 0, "resumed_from": 0}
    if not args.diagnose_only and prepared.get("enabled", True):
        train_result = train(student, tok, cache, retain, prepared, cfg, device, method, out_dir, args.resume, teacher)
        status = "COMPLETED"
    elif not prepared.get("enabled", True):
        status = "DIAGNOSTIC_KILL"
    after = {
        "forget": evaluate_ground_truth(student, tok, forget, device, int(cfg["data"]["max_len"]), int(cfg["data"]["eval_samples"])),
        "retain": evaluate_ground_truth(student, tok, retain, device, int(cfg["data"]["max_len"]), int(cfg["data"]["eval_samples"])),
        "holdout": evaluate_ground_truth(student, tok, holdout, device, int(cfg["data"]["max_len"]), int(cfg["data"]["eval_samples"])),
        "donor": evaluate_pair_donor(student, tok, pairs, cfg, device),
        "behaviour": generation_behaviour(student, tok, forget, device, int(cfg["data"].get("generation_tokens", 24)), int(cfg["data"].get("generation_samples", 4))),
    }
    checkpoint = out_dir / "checkpoint"
    if status == "COMPLETED":
        save_trained(student, tok, checkpoint)
    probes = {
        "before": target_fact_probe(teacher, tok, forget, cfg, device),
        "after": target_fact_probe(student, tok, forget, cfg, device),
    }
    recovery = relearning_recovery(student, tok, forget, cfg, device) if status == "COMPLETED" else {"status": "NOT_RUN", "reason": status}
    result = {
        "method_id": method.METHOD_ID, "status": status, "claim_level": False,
        "profile": cfg["profile"], "seed": cfg["seed"], "device": str(device),
        "pair_count": len(pairs), "train": train_result, "diagnostics": diagnostics,
        "before": before, "after": after, "target_fact_probe": probes, "relearning": recovery,
        "delta": {key: after[key]["nll"] - before[key]["nll"] for key in ("forget", "retain", "holdout")},
        "checkpoint": str(checkpoint) if status == "COMPLETED" else None,
        "elapsed_sec": round(time.time() - started, 3),
        "integrity": {"teacher_frozen": all(not p.requires_grad for p in teacher.parameters()),
                      "evaluation_ground_truth": "dataset answers", "donor_answer_in_training": False,
                      "cross_method_imports": False, "initial_student_equals_teacher": initial_equivalent,
                      "initial_student_teacher_nll_delta": initial_nll_delta,
                      "initial_nll_tolerance": initial_tolerance},
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps({"method": method.METHOD_ID, "status": status, "result": str(out_dir / "result.json"), "delta": result["delta"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
