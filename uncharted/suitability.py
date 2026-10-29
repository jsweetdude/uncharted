"""Check that an embedding suits a corpus before trusting its map: the same work given twice
should rank as each other's close neighbors, and clearly unrelated pairs should not.

    python -I uncharted/suitability.py data/calibration/talks.pairs.json talks talks-q

The pairs file lists "near" pairs (one talk at two conferences, a patent and its
continuation) and "unrelated" pairs, each as [id, id, note]. For every embedding named, each
pair's partner is ranked among all items by similarity, in both directions and averaged, as
a percentile: 99.9 means the partner is closer than 99.9% of the corpus. The bars are --near
(default 90) and --unrelated (default 50). The result is a report, not a verdict: pick
unrelated pairs by labels the embedding did not produce, such as conference tracks or patent
classes, or the check measures the model against itself.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np


def rank_pct(X: np.ndarray, a: int, b: int) -> tuple[float, float]:
    out = []
    for x, y in ((a, b), (b, a)):
        s = X @ X[x]
        others = np.delete(s, x)
        out.append(100.0 * float(np.mean(others < s[y])))
    return float(np.mean(out)), float(X[a] @ X[b])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pairs", help="JSON file with near and unrelated pairs")
    ap.add_argument("names", nargs="+", help="embedding basenames to compare")
    ap.add_argument("--dir", default="data/embeddings")
    ap.add_argument("--near", type=float, default=90.0)
    ap.add_argument("--unrelated", type=float, default=50.0)
    args = ap.parse_args()
    with open(args.pairs, encoding="utf-8") as fh:
        pairs = json.load(fh)
    for name in args.names:
        base = os.path.join(args.dir, name)
        X = np.load(base + ".npy").astype(np.float32)
        with open(base + ".meta.jsonl", encoding="utf-8") as fh:
            pos = {json.loads(line)["id"]: i for i, line in enumerate(fh)}
        print(f"== {name} ({X.shape[0]} items)")
        for kind, bar in (("near", args.near), ("unrelated", args.unrelated)):
            rows = []
            for a, b, note in pairs.get(kind, []):
                if a not in pos or b not in pos:
                    rows.append((None, None, note))
                    continue
                pct, sim = rank_pct(X, pos[a], pos[b])
                rows.append((pct, sim, note))
            ok = sum(1 for pct, _, _ in rows if pct is not None and (pct >= bar if kind == "near" else pct <= bar))
            ranked = [pct for pct, _, _ in rows if pct is not None]
            print(f"  {kind}: {ok} of {len(rows)} meet the bar ({'at least' if kind == 'near' else 'at most'} the {bar:g}th percentile); mean rank {np.mean(ranked):.1f}" if ranked else f"  {kind}: no pairs found")
            for pct, sim, note in rows:
                if pct is None:
                    print(f"    --  missing from this corpus: {note}")
                else:
                    met = pct >= bar if kind == "near" else pct <= bar
                    print(f"    {'ok' if met else 'NO'}  rank {pct:5.1f}  similarity {sim:.3f}  {note}")


if __name__ == "__main__":
    main()
