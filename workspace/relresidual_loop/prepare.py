"""Fixed protocol for the RelResidual idea validation (v6).

Frozen here (NOT editable during the loop):
 - counterfactual / RelResidual target construction
 - evaluation protocol (incl. donor leakage + premature ending detectors)
 - the single metric and the termination gate
 - the independent replication axis (3 data orders)

Idea under test:
  B_l = [A_l(s_f,r) - A_l(s_r,r)] - [A_l(s_f,r0) - A_l(s_r,r0)]
  T_l = A_l(s_f,r) - alpha * B_l
where r0 is a relation explicitly NOT deleted (a retain-set template).
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

import torch

LOOP = Path(__file__).resolve().parent
REPO = LOOP.parents[1]
AUTO = REPO / "workspace" / "auto_experiments"
V1 = REPO / "workspace" / "experiments" / "direction_i"
sys.path.insert(0, str(AUTO))
sys.path.insert(0, str(V1))

import common as C  # noqa: E402 (upstream, read-only)
from counterfactual import build_pairs, collect_donor_names, subject_spans  # noqa: E402

NAME_RE = re.compile(r"\b[A-Z][A-Za-z'\-]+(?:\s+[A-Z][A-Za-z'\-]+){1,2}\b")
FORGET = C.FORGET
RETAIN = C.RETAIN
MODEL = C.MODEL

STEPS = 60
LR = 5e-6
INIT_SEED = 0
REPL_ORDERS = (0, 1, 2)
PAIRS = 24
GEN_SAMPLES = 32
GEN_MAX_NEW_TOKENS = 64
TARGET_LAYERS = (8, 12, 15)   # layers carrying the activation loss

# ---- frozen "good enough" target ----
ASU_BASELINE_DF = 0.7998      # measured: ASU global tau=2, 60 steps, same budget
DF_MIN = ASU_BASELINE_DF      # must NOT be weaker than the ASU reference
DR_MAX = 0.05
DONOR_LEAK_MAX = 0.10
PREMATURE_MAX = 0.10
TRUTH_LEAK_MAX = 0.30
DF_STD_MAX = 0.08

W_DR = 2.0
W_DONOR = 3.0
W_PREMATURE = 3.0
W_TRUTH = 1.5
W_DF_STD = 2.0
GAMMA_K = 0.01

MAX_ROUNDS = 5
MAX_GPU_SECONDS = 1800
MAX_CONSECUTIVE_CRASH = 3

REFUSAL_TEMPLATES = ("i don't know", "i cannot", "i'm sorry", "as an ai",
                     "no information", "i do not know", "i can't", "unknown")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump_json(obj, path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)


def prompt_of(question):
    return f"Question: {question}\nAnswer:"


def retain_templates(retain_records):
    """Relation templates from the retain set -> r0 is explicitly NOT deleted."""
    out = {}
    for rec in retain_records:
        q = rec["question"]
        names = NAME_RE.findall(q)
        if len(names) != 1:
            continue
        n = re.sub(r"'s$", "", names[0])
        t = q.replace(n, "<NAME>", 1)
        if "<NAME>" not in t:
            continue
        out.setdefault(t, n)
    return out


def build_quadruples(tok, n_pairs=PAIRS, seed=0):
    """(q(s_f,r), q(s_r,r), q(s_f,r0), q(s_r,r0)) with token-length checks."""
    pairs = build_pairs(tok, FORGET, RETAIN, max_pairs=n_pairs, seed=seed)
    retain = C.load_jsonl(RETAIN)
    templates = list(retain_templates(retain).items())
    rng = random.Random(seed)
    rng.shuffle(templates)

    quads = []
    for p in pairs:
        sf, sr = p["forget_subject"], p["donor_subject"]
        q_fr, q_rr = p["x_plus"], p["x_minus"]
        t_r = q_fr.replace(sf, "<NAME>", 1)
        chosen = None
        for t0, _ in templates:
            if t0 == t_r:
                continue
            q_fr0 = t0.replace("<NAME>", sf, 1)
            q_rr0 = t0.replace("<NAME>", sr, 1)
            if "<NAME>" in q_fr0 or "<NAME>" in q_rr0:
                continue
            if len(tok.encode(q_fr0, add_special_tokens=False)) != len(tok.encode(q_rr0, add_special_tokens=False)):
                continue
            chosen = (t0, q_fr0, q_rr0)
            break
        if chosen is None:
            continue
        t0, q_fr0, q_rr0 = chosen
        # hard requirement: token-equal AFTER the actual prompt wrapper is applied
        # (BPE can merge differently at the prefix/name boundary)
        lens = [len(tok.encode(prompt_of(x), add_special_tokens=False))
                for x in (q_fr, q_rr, q_fr0, q_rr0)]
        if lens[0] != lens[1] or lens[2] != lens[3]:
            continue
        quads.append({
            "q_fr": q_fr, "q_rr": q_rr, "q_fr0": q_fr0, "q_rr0": q_rr0,
            "s_f": sf, "s_r": sr, "r0_template": t0,
            "answer_plus": p["answer_plus"],
            "span": p["token_span"],
        })
    return quads


def hidden_last(model, tok, questions, layers, position="query_last", span=None):
    """Return {layer: [B, d]} hidden states at query-last or subject-last."""
    enc = tok([prompt_of(q) for q in questions], return_tensors="pt", padding=True,
              add_special_tokens=False).to("cuda")
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
    hs = out.hidden_states
    if position == "query_last":
        idx = (enc["attention_mask"].sum(dim=1) - 1).tolist()
    else:  # subject_last: last token of the replaced subject span
        if span is None:
            raise ValueError("subject_last needs span")
        idx = [span[1] - 1] * len(questions)
    res = {}
    for l in layers:
        h = hs[l + 1]  # hs[0] is embedding output
        res[l] = h[torch.arange(len(questions), device=h.device),
                   torch.tensor(idx, device=h.device)].float()
    return res


def key_facts(answer):
    """Content words from the original answer, used for truth-leak detection."""
    toks = re.findall(r"[A-Za-z][A-Za-z'\-]+|\d{2,}", answer)
    stop = {"the", "a", "an", "is", "was", "are", "of", "in", "on", "at", "to", "for",
            "and", "his", "her", "their", "part", "full", "name", "author", "father",
            "mother", "profession", "born", "writes", "genre"}
    return [t.lower() for t in toks if t.lower() not in stop and len(t) > 2][:6]


def evaluate_generation(texts, quads):
    """Detect the two failure modes the idea explicitly warns about."""
    donor_hits, premature, truth_hits, lens = 0, 0, 0, []
    for text, qu in zip(texts, quads[:len(texts)]):
        t = text.strip().lower()
        # donor leakage: the donor subject (or a distinctive token of it) appears
        donor_parts = [x.lower() for x in qu["s_r"].split() if len(x) > 2]
        if any(x in t for x in donor_parts):
            donor_hits += 1
        # premature ending: empty / too short / pure refusal
        if len(t.split()) < 4 or any(k in t for k in REFUSAL_TEMPLATES):
            premature += 1
        # truth leakage: original factual content still produced
        facts = key_facts(qu["answer_plus"])
        if facts and sum(1 for f in facts if f in t) >= max(1, len(facts) // 2):
            truth_hits += 1
        lens.append(len(t.split()))
    n = max(len(texts), 1)
    return {"donor_leak": donor_hits / n, "premature": premature / n,
            "truth_leak": truth_hits / n, "mean_words": sum(lens) / n}


def aggregate(order_results, complexity):
    dFs = [r["dF"] for r in order_results]
    dRs = [r["dR"] for r in order_results]
    donors = [r["donor_leak"] for r in order_results]
    prems = [r["premature"] for r in order_results]
    truths = [r["truth_leak"] for r in order_results]
    mean_df = sum(dFs) / len(dFs)
    std_df = (sum((x - mean_df) ** 2 for x in dFs) / len(dFs)) ** 0.5
    worst_dr, worst_donor = max(dRs), max(donors)
    worst_prem, worst_truth = max(prems), max(truths)
    val = (max(0.0, DF_MIN - mean_df)
           + W_DR * max(0.0, worst_dr - DR_MAX)
           + W_DONOR * max(0.0, worst_donor - DONOR_LEAK_MAX)
           + W_PREMATURE * max(0.0, worst_prem - PREMATURE_MAX)
           + W_TRUTH * max(0.0, worst_truth - TRUTH_LEAK_MAX)
           + W_DF_STD * std_df
           + GAMMA_K * complexity)
    checks = {
        "df_ge_asu_baseline": mean_df >= DF_MIN,
        "all_dr_le_max": all(d <= DR_MAX for d in dRs),
        "all_donor_leak_le_max": all(d <= DONOR_LEAK_MAX for d in donors),
        "all_premature_le_max": all(p <= PREMATURE_MAX for p in prems),
        "all_truth_leak_le_max": all(t <= TRUTH_LEAK_MAX for t in truths),
        "df_std_le_max": std_df <= DF_STD_MAX,
        "replicated": len(order_results) == len(REPL_ORDERS),
    }
    return {"df_mean": mean_df, "df_std": std_df, "dFs": dFs, "dRs": dRs,
            "donors": donors, "prematures": prems, "truths": truths,
            "worst_dr": worst_dr, "worst_donor": worst_donor,
            "worst_premature": worst_prem, "worst_truth": worst_truth,
            "val_bpb": val, "checks": checks, "good": all(checks.values())}


def verify_run(d):
    req = ("data_order", "dF", "dR", "donor_leak", "premature", "truth_leak", "val_bpb")
    missing = [k for k in req if k not in d]
    if missing:
        return False, f"missing {missing}"
    for k in ("dF", "dR", "donor_leak", "premature", "truth_leak", "val_bpb"):
        v = d[k]
        if not isinstance(v, (int, float)) or v != v or abs(v) == float("inf"):
            return False, f"non-finite {k}"
    return True, "ok"


def preflight():
    issues = []
    if not torch.cuda.is_available():
        issues.append("CUDA unavailable")
    for p in (MODEL, FORGET, RETAIN, AUTO / "common.py", V1 / "counterfactual.py"):
        if not Path(p).exists():
            issues.append(f"missing: {p}")
    return issues
