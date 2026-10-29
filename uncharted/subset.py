"""Subset an embedded corpus without re-embedding: keep rows whose text matches a pattern,
drop rows that match another, and collapse exact duplicates by normalized title.

    python -I uncharted/subset.py --name papers --out-name papers-clean

The defaults are tuned for an accessibility corpus pulled by keyword search, which also
catches "web-accessible" protein databases, urban transit "accessibility", and clinical
papers. KEEP requires a disability population, an assistive technology, or a digital
accessibility phrase; DROP removes the biomedical and chemistry matches.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil

import numpy as np

KEEP = re.compile(
    r"disabilit|impair|\bblind|\bdeaf|hard[- ]of[- ]hearing|low[- ]vision|visually|wheelchair|autis|dyslex|aphasi|"
    r"cerebral palsy|older adult|elderly|\bseniors\b|screen[- ]reader|assistive|braille|sign language|\bwcag\b|\baria\b|"
    r"\bcaption|audio description|alt[- ]text|alternative text|cognitive|neurodiver|a11y|universal design|inclusive design|"
    r"amputee|prosthe|paralys|spinal cord|parkinson|dementia|motor impair|speech impair|"
    r"web accessibility|digital accessibility|accessibility guideline|accessibility evaluation|accessibility of web|"
    r"website accessibility|mobile accessibility|game accessibility|accessibility barrier|accessibility issue|"
    r"accessibility testing|accessibility audit|accessible design|accessibility standard|accessibility feature",
    re.I,
)
DROP = re.compile(r"protein|genom|chemical|molecul|\bcancer\b|tumou?r|surgical|\bsurgery\b|enzyme|crystal", re.I)


def norm_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", title.lower())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out-name", required=True)
    ap.add_argument("--dir", default="data/embeddings")
    ap.add_argument("--keep", default=None, help="regex a record's text must match (default: accessibility population/AT terms)")
    ap.add_argument("--drop", default=None, help="regex that removes a record (default: biomedical and chemistry terms)")
    ap.add_argument("--no-dedupe", action="store_true")
    ap.add_argument("--no-keep", action="store_true", help="skip the KEEP test (for a topic-classified corpus)")
    ap.add_argument("--drop-types", default="", help="comma-separated extra.type values to remove, e.g. other,report,libguides")
    args = ap.parse_args()

    keep = re.compile(args.keep, re.I) if args.keep else KEEP
    drop = re.compile(args.drop, re.I) if args.drop else DROP
    drop_types = {t.strip() for t in args.drop_types.split(",") if t.strip()}
    base = os.path.join(args.dir, args.name)
    out = os.path.join(args.dir, args.out_name)
    matrix = np.load(base + ".npy")
    with open(base + ".meta.jsonl", encoding="utf-8") as fh:
        meta = [json.loads(line) for line in fh]
    assert len(meta) == matrix.shape[0]

    mask = np.zeros(len(meta), dtype=bool)
    seen_titles: set[str] = set()
    reasons = {"no keep term": 0, "drop term": 0, "dropped type": 0, "duplicate title": 0}
    for i, m in enumerate(meta):
        text = m["text"]
        if drop_types and (m.get("extra") or {}).get("type") in drop_types:
            reasons["dropped type"] += 1
            continue
        if not args.no_keep and not keep.search(text):
            reasons["no keep term"] += 1
            continue
        if drop.search(text):
            reasons["drop term"] += 1
            continue
        if not args.no_dedupe:
            key = norm_title(m["title"])
            if key in seen_titles:
                reasons["duplicate title"] += 1
                continue
            seen_titles.add(key)
        mask[i] = True

    np.save(out + ".npy", matrix[mask])
    with open(out + ".meta.jsonl", "w", encoding="utf-8") as fh:
        for i in np.where(mask)[0]:
            fh.write(json.dumps(meta[i], ensure_ascii=False) + "\n")
    if os.path.exists(base + ".model.json"):
        shutil.copyfile(base + ".model.json", out + ".model.json")
    print(f"{args.name}: {len(meta)} -> {args.out_name}: {int(mask.sum())} kept; removed {reasons}")


if __name__ == "__main__":
    main()
