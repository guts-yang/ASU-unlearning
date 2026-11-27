"""CF3-03 VOCE: visit-measure-weighted counterfactual activations on shared prefixes."""
from __future__ import annotations

import random
import torch
import torch.nn.functional as F

from modeling import encode_prefix, forward_model

METHOD_ID = "CF3-03"
NEEDS_STUDENT_MINUS = False
REQUIRES_CONTROL = False


def mixture_counts(total, eta):
    policy = int(round(total * eta))
    return total - policy, policy


def _prefixes(ids, count):
    if not ids or count <= 0:
        return []
    lengths = sorted(set(max(1, min(len(ids), ((i + 1) * len(ids)) // count)) for i in range(count)))
    return [list(ids[:length]) for length in lengths]


@torch.no_grad()
def _build_visits(cache, teacher, tok, cfg, device, student):
    eta = float(cfg["method"].get("eta", 0.5))
    total = int(cfg["method"].get("max_states_per_pair", 4))
    gold_n, policy_n = mixture_counts(total, eta)
    visits = {}
    rollout_log = []
    for item in cache["pairs"]:
        pair = item["pair"]
        gold_ids = item["enc_plus"]["input_ids"][item["enc_plus"]["answer_start"]:item["enc_plus"]["answer_end"] - 1].tolist()
        gold = _prefixes(gold_ids, gold_n)
        sampled = []
        if policy_n:
            prompt = f"Question: {pair['x_plus']}\nAnswer:"
            ids = tok.encode(prompt, add_special_tokens=False)
            inp = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
            is_cuda = str(device).startswith("cuda")
            devices = [torch.cuda.current_device()] if is_cuda else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(int(cfg["seed"]) + item["index"])
                if is_cuda:
                    torch.cuda.manual_seed_all(int(cfg["seed"]) + item["index"])
                out = student.generate(input_ids=inp, attention_mask=torch.ones_like(inp),
                                       max_new_tokens=int(cfg["method"].get("rollout_tokens", 8)),
                                       do_sample=True, top_p=float(cfg["method"].get("top_p", 0.9)),
                                       temperature=float(cfg["method"].get("temperature", 1.0)),
                                       pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
            generated = [int(x) for x in out[0, len(ids):].tolist() if int(x) != tok.eos_token_id]
            sampled = _prefixes(generated, policy_n)
        rows = [(x, "gold", (1.0 - eta) / max(1, len(gold))) for x in gold]
        rows += [(x, "policy", eta / max(1, len(sampled))) for x in sampled]
        if not rows:
            raise RuntimeError("no shared prefix states; increase rollout/gold prefix budget")
        merged = {}
        for prefix, source, weight in rows:
            key = tuple(prefix)
            if key not in merged:
                merged[key] = {"prefix": prefix, "sources": set(), "weight": 0.0}
            merged[key]["sources"].add(source)
            merged[key]["weight"] += weight
        pair_visits = []
        for state in merged.values():
            enc_p = encode_prefix(tok, pair["x_plus"], state["prefix"], int(cfg["data"]["max_len"]))
            enc_m = encode_prefix(tok, pair["x_minus"], state["prefix"], int(cfg["data"]["max_len"]))
            if enc_p["input_ids"].shape != enc_m["input_ids"].shape or not torch.equal(enc_p["input_ids"][enc_p["answer_start"]:], enc_m["input_ids"][enc_m["answer_start"]:]):
                raise AssertionError("positive and negative Teacher branches do not share the prefix")
            fp = forward_model(teacher, enc_p, device, set(cfg["layers"]))
            fm = forward_model(teacher, enc_m, device, set(cfg["layers"]))
            targets = {}
            for layer in cfg["layers"]:
                hp = fp["hidden"][layer][-1].detach().float().cpu()
                hm = fm["hidden"][layer][-1].detach().float().cpu()
                targets[layer] = hp - float(cfg["method"].get("alpha", 0.1)) * (hp - hm)
            pair_visits.append({"enc_plus": enc_p, "targets": targets, "weight": float(state["weight"]),
                                "sources": sorted(state["sources"]), "prefix_length": len(state["prefix"])})
        visits[item["index"]] = pair_visits
        rollout_log.append({"pair": item["index"], "states": len(pair_visits),
                            "weight_sum": sum(x["weight"] for x in pair_visits)})
    return {"enabled": True, "visits": visits, "rollout_log": rollout_log,
            "eta": eta, "rollout_tokens": int(cfg["method"].get("rollout_tokens", 8))}


def prepare(cache, teacher, tok, cfg, device, student=None, out_dir=None):
    if student is None:
        raise ValueError("CF3-03 requires the initialized Student for on-policy rollouts")
    path = out_dir / "visit_cache.pt" if out_dir is not None else None
    if path is not None and path.exists() and cfg["data"].get("reuse_cache", True):
        return torch.load(path, map_location="cpu", weights_only=False)
    prepared = _build_visits(cache, teacher, tok, cfg, device, student)
    if path is not None:
        torch.save(prepared, path)
    return prepared


def refresh(student, teacher, tok, cache, prepared, cfg, device):
    del prepared
    return _build_visits(cache, teacher, tok, cfg, device, student)


def loss(batch, prepared, cfg):
    terms, weights = [], []
    for row in batch:
        item, student, device = row["item"], row["student"], row["device"]
        for visit in prepared["visits"][item["index"]]:
            out = forward_model(student, visit["enc_plus"], device, set(cfg["layers"]), grad=True)
            for layer in cfg["layers"]:
                target = visit["targets"][layer].to(out["nll"].device)
                terms.append(F.mse_loss(out["hidden"][layer][-1], target.detach()))
                weights.append(float(visit["weight"]))
    if not terms:
        raise RuntimeError("no visit-weighted states")
    loss = torch.stack(terms)
    weight = torch.tensor(weights, device=loss.device, dtype=loss.dtype)
    weight = weight / weight.sum().clamp_min(1e-8)
    return {"method_loss": (loss * weight).sum(), "visited_states": loss.new_tensor(float(len(terms)))}


def diagnostics(cache, prepared, cfg):
    del cache, cfg
    return {"operator": "gold_onpolicy_shared_prefix_visit_measure", "eta": prepared["eta"],
            "rollouts": prepared["rollout_log"], "dynamic_refresh_supported": True,
            "prefix_contract": "X+ and X- receive identical generated prefix IDs"}


def self_test():
    assert mixture_counts(4, 0.5) == (2, 2)
    assert mixture_counts(4, 0.0) == (4, 0)
    assert mixture_counts(4, 1.0) == (0, 4)
    assert _prefixes([1, 2, 3, 4], 2) == [[1, 2], [1, 2, 3, 4]]
    return {"method_id": METHOD_ID, "status": "PASS", "mixture_counts": True, "shared_prefixes": True}
