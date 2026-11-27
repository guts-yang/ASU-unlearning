#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dump a .drawio file's mxCell inventory (id, value, style, geometry, edges).

Usage: python3 _dump_drawio.py <file.drawio> [out.txt]
"""
import sys, html
from pathlib import Path
import xml.etree.ElementTree as ET

src = Path(sys.argv[1])
out = Path(sys.argv[2]) if len(sys.argv) > 2 else None
root = ET.parse(str(src)).getroot()
model = root.find(".//mxGraphModel")
lines = ["MODEL " + " ".join(f"{k}={v}" for k, v in model.attrib.items()) if model is not None else "NO MODEL"]
cells = root.findall(".//mxCell")
lines.append(f"N_CELLS={len(cells)}")
for c in cells:
    cid = c.attrib.get("id", "")
    val = html.unescape(c.attrib.get("value", "") or "").replace("\n", " ")
    if len(val) > 110:
        val = val[:110] + "…"
    style = c.attrib.get("style", "") or ""
    if len(style) > 90:
        style = style[:90] + "…"
    kind = "EDGE" if c.attrib.get("edge") == "1" else ("VERT" if c.attrib.get("vertex") == "1" else "ROOT")
    srcid, tgtid = c.attrib.get("source", ""), c.attrib.get("target", "")
    g = c.find("mxGeometry")
    geo = ""
    if g is not None:
        geo = "x=%s y=%s w=%s h=%s" % (g.get("x"), g.get("y"), g.get("width"), g.get("height"))
        arr = g.find("Array[@as='points']")
        if arr is not None:
            plist = [(pp.get("x"), pp.get("y")) for pp in arr.findall("mxPoint")]
            geo += " pts=" + str(plist)
    lines.append(f"[{kind}] id={cid} src={srcid} tgt={tgtid} | {geo}")
    if val:
        lines.append(f"    VAL: {val}")
    if style:
        lines.append(f"    STY: {style}")
text = "\n".join(lines) + "\n"
if out:
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out} ({len(lines)} lines)")
else:
    print(text)
