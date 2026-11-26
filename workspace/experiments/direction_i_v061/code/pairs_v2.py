"""Synthetic same-template counterfactual pairs (X^- built by entity swap).

Discovery behind this file: TOFU forget10 and retain90 have disjoint authors AND
different question templates, so a template-matched donor from retain does not
exist (measured match rate 1/16). Proposal §2.1 wants an equal-length
counterfactual that differs only in the subject slot; we obtain it by replacing
the entity name inside the forget question. y^+ is kept unchanged, so X^- (+) y^+
asks about the donor subject while the answer still names the original subject.
"""
from __future__ import annotations

from engine import FORGET, RETAIN, load_jsonl
from pairs import extract_entity


def build_pairs_v2(forget=None, retain=None, n=16, seed=0):
    forget = forget if forget is not None else load_jsonl(FORGET)
    retain = retain if retain is not None else load_jsonl(RETAIN)
    donors = [r for r in retain if extract_entity(r["answer"])]
    out, synth = [], 0
    for i, f in enumerate(forget[:n]):
        ent_p = extract_entity(f["answer"])
        d = donors[(i * 7 + seed) % len(donors)]
        ent_m = extract_entity(d["answer"])
        q = f["question"]
        if ent_p and ent_m and ent_p != ent_m and ent_p in q:
            q2 = q.replace(ent_p, ent_m)
            if q2 != q:
                out.append(
                    {
                        "plus": f,
                        "minus": {"question": q2, "answer": f["answer"]},
                        "mode": "synth_entity_swap",
                        "entity_plus": ent_p,
                        "entity_minus": ent_m,
                    }
                )
                synth += 1
                continue
        out.append(
            {
                "plus": f,
                "minus": d,
                "mode": "retain_fallback",
                "entity_plus": ent_p,
                "entity_minus": ent_m,
            }
        )
    stats = {"n": len(out), "synth": synth, "synth_rate": round(synth / max(1, len(out)), 4)}
    return out, stats


if __name__ == "__main__":
    ps, st = build_pairs_v2(n=8)
    print(st)
    for p in ps[:4]:
        print(p["mode"], "|", p["plus"]["question"][:70], "->", p["minus"]["question"][:70])
