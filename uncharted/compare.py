"""Compare two embedded corpora of the same field side by side.

    python -I uncharted/compare.py --a papers-clean --b papers-oa [--out data/compare-papers.md]

Reads data/embeddings/<name>.meta.jsonl (always), and <name>.space.json plus
<name>.clusters.json when space.py has run. Reports, for each corpus: size, year
median, top venues, nearest-neighbor similarity percentiles, neighborhood count and the
biggest neighborhoods, and a noise proxy: the share of items whose text names no
disability population, assistive technology or digital-accessibility phrase (the KEEP
pattern in subset.py). Between the two: DOI overlap, so you can see how much one source
contains of the other.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from subset import KEEP  # noqa: E402


def load(directory: str, name: str) -> dict:
    base = os.path.join(directory, name)
    with open(base + ".meta.jsonl", encoding="utf-8") as fh:
        meta = [json.loads(line) for line in fh]
    space = clusters = None
    if os.path.exists(base + ".space.json"):
        with open(base + ".space.json", encoding="utf-8") as fh:
            space = json.load(fh)
    if os.path.exists(base + ".clusters.json"):
        with open(base + ".clusters.json", encoding="utf-8") as fh:
            clusters = json.load(fh)
    return {"name": name, "meta": meta, "space": space, "clusters": clusters}


def doi_of(rec: dict) -> str | None:
    doi = (rec.get("extra") or {}).get("doi") or rec.get("url") or ""
    m = re.search(r"10\.\d{4,9}/\S+", doi, re.I)
    return m.group(0).lower().rstrip(".") if m else None


def summarize(c: dict) -> dict:
    meta = c["meta"]
    years = [m["year"] for m in meta if m.get("year")]
    venues = collections.Counter(m.get("venue") for m in meta if m.get("venue")).most_common(5)
    noise = sum(1 for m in meta if not KEEP.search(m["text"]))
    out = {
        "items": len(meta),
        "year_median": int(statistics.median(years)) if years else None,
        "top_venues": venues,
        "noise_share": noise / max(len(meta), 1),
        "dois": {d for d in (doi_of(m) for m in meta) if d},
    }
    if c["space"]:
        out["nn1"] = c["space"]["nn1_percentiles"]
        out["clusters"] = c["space"]["clusters"]
        out["unclustered"] = c["space"]["unclustered"]
    if c["clusters"]:
        out["biggest"] = [f"{cl['size']} {', '.join(cl['label_terms'][:3])}" for cl in sorted(c["clusters"], key=lambda x: -x["size"])[:8]]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--dir", default="data/embeddings")
    ap.add_argument("--out", help="also write the comparison as Markdown")
    args = ap.parse_args()

    A, B = load(args.dir, args.a), load(args.dir, args.b)
    sa, sb = summarize(A), summarize(B)
    both = sa["dois"] & sb["dois"]

    rows = [
        ("items", sa["items"], sb["items"]),
        ("median year", sa["year_median"], sb["year_median"]),
        ("items with a DOI", len(sa["dois"]), len(sb["dois"])),
        ("DOIs shared with the other corpus", len(both), len(both)),
        ("share of own DOIs also in the other", f"{len(both)/max(len(sa['dois']),1):.1%}", f"{len(both)/max(len(sb['dois']),1):.1%}"),
        ("noise proxy (no population/AT term)", f"{sa['noise_share']:.1%}", f"{sb['noise_share']:.1%}"),
    ]
    if "nn1" in sa and "nn1" in sb:
        for p in ("10", "50", "90", "99"):
            rows.append((f"nearest-neighbor similarity p{p}", f"{sa['nn1'][p]:.3f}", f"{sb['nn1'][p]:.3f}"))
        rows.append(("neighborhoods", sa["clusters"], sb["clusters"]))
        rows.append(("unclustered items", sa["unclustered"], sb["unclustered"]))

    lines = [f"# {args.a} vs {args.b}", "", f"| | {args.a} | {args.b} |", "|---|---|---|"]
    for label, va, vb in rows:
        lines.append(f"| {label} | {va} | {vb} |")
    for name, s in ((args.a, sa), (args.b, sb)):
        lines += ["", f"## {name}", "", "Top venues: " + "; ".join(f"{v} ({n})" for v, n in s["top_venues"])]
        if "biggest" in s:
            lines += ["", "Biggest neighborhoods:", ""] + [f"- {b}" for b in s["biggest"]]
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text)


if __name__ == "__main__":
    main()
