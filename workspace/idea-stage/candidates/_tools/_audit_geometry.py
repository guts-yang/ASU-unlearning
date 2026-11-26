#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Geometry audit for the per-idea .drawio diagrams.

Checks (for every file passed):
  A. content-box <-> content-box overlap          (must be 0)
  B. explicit-waypoint arrow segments crossing OTHER content boxes (must be 0)
  C. arrow-label estimated width vs available horizontal gap  (warn)
  D. blank-space ratio: box height vs estimated rendered text height (warn)
  E. content box outside page                       (must be 0)

Usage: python3 _audit_geometry.py <dir with CF-*> [--json]
"""
import sys, json, math, re, glob
from pathlib import Path
import xml.etree.ElementTree as ET


def fnum(e, name, default=0.0):
    try:
        return float(e.attrib.get(name, default))
    except (TypeError, ValueError):
        return default


def parse(path):
    root = ET.parse(str(path)).getroot()
    model = root.find(".//mxGraphModel")
    pw = fnum(model, "pageWidth", 0) if model is not None else 0
    ph = fnum(model, "pageHeight", 0) if model is not None else 0
    cells = root.findall(".//mxCell")
    by_id = {c.attrib.get("id", ""): c for c in cells if c.attrib.get("id")}
    boxes = {}       # id -> rect dict (content boxes with text)
    containers = {}  # id -> rect (roundrect without text)
    headers = {}     # id -> rect (text-only cells)
    edges = []
    for c in cells:
        cid = c.attrib.get("id", "")
        geo = c.find("mxGeometry")
        if geo is None:
            continue
        x, y = fnum(geo, "x"), fnum(geo, "y")
        w, h = fnum(geo, "width"), fnum(geo, "height")
        rect = dict(x=x, y=y, w=w, h=h, id=cid,
                    value=(c.attrib.get("value", "") or ""),
                    style=(c.attrib.get("style", "") or ""))
        if c.attrib.get("edge") == "1":
            pts = []
            arr = geo.find("Array[@as='points']")
            if arr is not None:
                for p in arr.findall("mxPoint"):
                    pts.append((fnum(p, "x"), fnum(p, "y")))
            edges.append(dict(id=cid, src=c.attrib.get("source"),
                              tgt=c.attrib.get("target"), label=rect["value"],
                              points=pts, style=rect["style"]))
            continue
        if c.attrib.get("vertex") != "1":
            continue
        st = rect["style"]
        has_val = bool(rect["value"].strip())
        if "rounded=1" in st and has_val:
            boxes[cid] = rect
        elif "rounded=1" in st and not has_val:
            containers[cid] = rect
        elif st.startswith("text;"):
            headers[cid] = rect
    return dict(page=(pw, ph), boxes=boxes, containers=containers,
                headers=headers, edges=edges, by_id=by_id)


# ---------- geometry helpers ----------
def rect_center(r):
    return (r["x"] + r["w"] / 2, r["y"] + r["h"] / 2)


def overlap_area(a, b):
    dx = min(a["x"] + a["w"], b["x"] + b["w"]) - max(a["x"], b["x"])
    dy = min(a["y"] + a["h"], b["y"] + b["h"]) - max(a["y"], b["y"])
    return dx * dy if dx > 0 and dy > 0 else 0.0


def seg_intersects_rect(p0, p1, r, pad=2):
    """True if segment p0->p1 passes through the interior of rect r (Liang-Barsky)."""
    x0, y0 = r["x"] + pad, r["y"] + pad
    x1, y1 = r["x"] + r["w"] - pad, r["y"] + r["h"] - pad
    ax, ay = p0
    bx, by = p1
    dx, dy = bx - ax, by - ay
    p = [-dx, dx, -dy, dy]
    q = [ax - x0, x1 - ax, ay - y0, y1 - ay]
    u0, u1 = 0.0, 1.0
    for pi, qi in zip(p, q):
        if abs(pi) < 1e-9:
            if qi < 0:
                return False
        else:
            t = qi / pi
            if pi < 0:
                u0 = max(u0, t)
            else:
                u1 = min(u1, t)
            if u0 > u1:
                return False
    return u0 < u1 and u1 > 0 and u0 < 1


def fontsize(style):
    m = re.search(r"fontSize=(\d+)", style)
    return int(m.group(1)) if m else 11


def estimate_lines(text, width, fs):
    """Rough number of rendered lines for a drawio html value.

    $$..$$ display-math lines render TALLER than body text (MathJax), and
    constructs like \\frac / \\sum / \\underbrace need extra vertical room.
    We therefore return (line_count, extra_height_factor).
    """
    text = text.replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")
    text = re.sub(r"<[^>]+>", "", text)
    segs = [s for s in text.split("\n")]
    cpl = max(8, int((width - 20) / max(fs * 0.52, 1)))
    total = 0
    math_lines = 0
    for s in segs:
        s = s.strip()
        if not s:
            total += 1
            continue
        wrapped = max(1, math.ceil(len(s) / cpl))
        total += wrapped
        if "$$" in s:
            math_lines += wrapped
    # display math ~1.6x body line height; frac/sum/underbrace ~2.2x
    factor = 1.0
    if math_lines:
        factor += 0.6 * math_lines / max(1, total)
    heavy = sum(1 for s in segs if "$$" in s and any(k in s for k in ("\\frac", "\\sum", "\\underbrace", "\\Big", "\\int")))
    if heavy:
        factor += 0.4 * heavy / max(1, total)
    return max(1, total), factor


def audit(path):
    d = parse(path)
    boxes = d["boxes"]
    pw, ph = d["page"]
    issues = []

    # A. content-box overlaps
    ids = list(boxes)
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            oa = overlap_area(boxes[ids[i]], boxes[ids[j]])
            if oa > 1.0:
                issues.append(dict(kind="BOX_OVERLAP", sev="FAIL",
                                   msg=f"{ids[i]} ∩ {ids[j]} = {oa:.0f}px²"))

    # B. explicit-waypoint arrow segments vs other content boxes
    for e in d["edges"]:
        if not e["points"]:
            continue
        # reconstruct center for src/tgt from boxes dict (they are content boxes)
        sc = rect_center(boxes[e["src"]]) if e["src"] in boxes else None
        tc = rect_center(boxes[e["tgt"]]) if e["tgt"] in boxes else None
        poly = ([sc] if sc else []) + e["points"] + ([tc] if tc else [])
        for k in range(len(poly) - 1):
            p0, p1 = poly[k], poly[k + 1]
            for bid, r in boxes.items():
                if bid in (e["src"], e["tgt"]):
                    continue
                if seg_intersects_rect(p0, p1, r):
                    issues.append(dict(kind="ARROW_CROSS_BOX", sev="FAIL",
                                       msg=f"edge {e['id']} seg{k} ({p0[0]:.0f},{p0[1]:.0f})->({p1[0]:.0f},{p1[1]:.0f}) crosses {bid}"))

    # C. arrow-label overflow: only meaningful for short HORIZONTAL auto edges.
    # Vertical edges (dy>dx) place their label above the line, not in the gap.
    for e in d["edges"]:
        lab = e["label"].strip()
        if not lab:
            continue
        if e["src"] in boxes and e["tgt"] in boxes:
            a, b = boxes[e["src"]], boxes[e["tgt"]]
            dx = (b["x"] + b["w"] / 2) - (a["x"] + a["w"] / 2)
            dy = (b["y"] + b["h"] / 2) - (a["y"] + a["h"] / 2)
            if abs(dy) > abs(dx):        # vertical edge -> label sits above line
                continue
            gap = max(0, b["x"] - (a["x"] + a["w"]))
            est = len(re.sub(r"<[^>]+>", "", lab)) * 6.5
            if est > gap + 24 and gap < 400:
                issues.append(dict(kind="LABEL_OVERFLOW", sev="WARN",
                                   msg=f"edge {e['id']} label~{est:.0f}px > gap {gap:.0f}px ('{lab[:24]}')"))

    # D. blank-space ratio per content box
    worst = []
    for bid, r in boxes.items():
        fs = fontsize(r["style"])
        lines, factor = estimate_lines(r["value"], r["w"], fs)
        need = lines * fs * 1.45 * factor + 16
        ratio = 1.0 - min(1.0, need / r["h"]) if r["h"] > 0 else 0
        worst.append((ratio, bid, need, r["h"], lines))
        if r["h"] > 2.2 * need:
            issues.append(dict(kind="BLANK_SPACE", sev="WARN",
                               msg=f"{bid}: h={r['h']:.0f} need~{need:.0f} ({lines} lines, math×{factor:.2f}) blank~{ratio*100:.0f}%"))
    worst.sort(reverse=True)

    # E. content box outside page
    for bid, r in boxes.items():
        if pw and ph and (r["x"] + r["w"] > pw + 2 or r["y"] + r["h"] > ph + 2 or r["x"] < -2 or r["y"] < -2):
            issues.append(dict(kind="OUT_OF_PAGE", sev="FAIL", msg=f"{bid} outside {pw}x{ph}"))

    return dict(file=str(path), page=[pw, ph], n_boxes=len(boxes),
                n_edges=len(d["edges"]), issues=issues,
                worst_blank=[(round(a, 2), b, round(c), round(e)) for a, b, c, e, _ in worst[:4]])


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    as_json = "--json" in sys.argv
    base = args[0] if args else "."
    files = sorted(glob.glob(str(Path(base) / "CF-*" / "*_机制流程图.drawio")))
    results = [audit(f) for f in files]
    if as_json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return
    tot_fail = tot_warn = 0
    for r in results:
        name = Path(r["file"]).parent.name
        fails = [i for i in r["issues"] if i["sev"] == "FAIL"]
        warns = [i for i in r["issues"] if i["sev"] == "WARN"]
        tot_fail += len(fails)
        tot_warn += len(warns)
        print(f"\n=== {name}  page={r['page'][0]:.0f}x{r['page'][1]:.0f}  boxes={r['n_boxes']} edges={r['n_edges']}  FAIL={len(fails)} WARN={len(warns)} ===")
        for i in fails:
            print(f"  [FAIL] {i['kind']}: {i['msg']}")
        for i in warns[:6]:
            print(f"  [warn] {i['kind']}: {i['msg']}")
        if len(warns) > 6:
            print(f"  ... +{len(warns)-6} more WARN")
        print("  worst blank:", r["worst_blank"])
    print(f"\n===== TOTAL: FAIL={tot_fail}  WARN={tot_warn}  files={len(results)} =====")


if __name__ == "__main__":
    main()
