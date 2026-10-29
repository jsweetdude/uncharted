"""Build the viewer's random sample map: made-up points run through the real pipeline.

    python -I uncharted/demo.py --out viewer/samples/random.json.gz

No text, model or network is involved. Points are drawn on a 6-dimensional shape around twelve
neighborhoods in four groups of three, then lifted into the embedding's dimensions, so neighborhoods in a group are related. Most related pairs get a few items
between them, loose points sit everywhere, and three pairs get none, so their mixtures have empty space to land in and the
candidate search has something to find. Titles are color and object names, plainly invented.
The same seed gives the same map. Intermediate files go under data/ as demo-random.*.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
COLORS = ["Amber", "Cobalt", "Crimson", "Ivory", "Jade", "Onyx", "Saffron", "Slate", "Teal", "Umber", "Violet", "Ochre"]
THINGS = ["circle", "prism", "spiral", "lattice", "ribbon", "pebble", "comet", "harbor", "meadow", "lantern", "compass", "garnet"]
EXTRA = ["notes", "study", "sketch", "survey", "pattern", "trial", "draft", "outline"]
EMPTY_PAIRS = {(0, 1), (4, 5), (8, 10)}


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def run(*args: str) -> None:
    print("$", " ".join(args), flush=True)
    subprocess.run([sys.executable, "-I", *args], check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per", type=int, default=110, help="items per neighborhood")
    ap.add_argument("--dim", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=9)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    k = 6  # the points live on a 6-dimensional shape, as real embeddings roughly do, then get lifted
    groups = rng.normal(0, 3.8, (4, k))
    centers = np.repeat(groups, 3, axis=0) + rng.normal(0, 1.9, (12, k))
    pts, hood = [], []
    for c in range(12):
        shape = rng.normal(0, 0.45, (k, k))
        pts.append(centers[c] + rng.standard_normal((args.per, k)) @ shape)
        hood += [c] * args.per
    for g in range(4):
        for a in range(3 * g, 3 * g + 3):
            for b in range(a + 1, 3 * g + 3):
                if (a, b) in EMPTY_PAIRS:
                    continue
                w = rng.uniform(0.25, 0.75, (14, 1))
                pts.append(w * centers[a] + (1 - w) * centers[b] + rng.normal(0, 0.35, (14, k)))
                hood += [a if x > 0.5 else b for x in w[:, 0]]
    loose = 260
    pts.append(rng.normal(0, 4.2, (loose, k)))
    hood += [-1] * loose
    Z = np.vstack(pts)
    lift = rng.standard_normal((k, args.dim)) / np.sqrt(args.dim)
    base = unit(rng.standard_normal(args.dim))
    X = unit(5.0 * base + Z @ lift + rng.normal(0, 0.25 / np.sqrt(args.dim), (len(Z), args.dim))).astype(np.float32)
    d = args.dim

    meta = []
    for i, c in enumerate(hood):
        color = COLORS[c] if c >= 0 else COLORS[int(rng.integers(12))]
        thing = THINGS[c] if c >= 0 else THINGS[int(rng.integers(12))]
        title = f"{color} {thing} {EXTRA[int(rng.integers(len(EXTRA)))]} {i + 1:04d}"
        meta.append({"id": f"random:{i + 1:04d}", "corpus": "random", "title": title, "text": title,
                     "year": int(rng.integers(2010, 2027)), "venue": None, "tags": [color.lower(), thing],
                     "popularity": int(rng.pareto(1.5) * 10), "url": None, "extra": {}})

    os.chdir(os.path.join(HERE, ".."))
    base = "data/embeddings/demo-random"
    os.makedirs("data/embeddings", exist_ok=True)
    np.save(base + ".npy", X)
    with open(base + ".meta.jsonl", "w", encoding="utf-8") as fh:
        for m in meta:
            fh.write(json.dumps(m) + "\n")
    with open(base + ".model.json", "w", encoding="utf-8") as fh:
        json.dump({"provider": "local", "model": "none: random vectors", "dim": d, "records": len(meta),
                   "normalized": True, "instruction": None, "input": "random points, no text"}, fh, indent=2)
    info = "data/corpora/demo-random.corpus.json"
    os.makedirs("data/corpora", exist_ok=True)
    with open(info, "w", encoding="utf-8") as fh:
        json.dump({"kind": "invented item", "topic": "random points with made-up names",
                   "sources": [{"name": "uncharted/demo.py", "query": f"seed {args.seed}, {len(meta)} points",
                                "items": len(meta), "license": "CC0-1.0",
                                "attribution": "Invented by uncharted/demo.py; no real works."}],
                   "popularity": "a random number", "tags": "the color and object in each name"}, fh, indent=2)
    run("uncharted/space.py", "--name", "demo-random")
    run("uncharted/gaps.py", "connections", "--name", "demo-random", "--kind", "paper", "--briefs",
        "--replace", "--out-dir", "data/gaps/demo-random-connections")
    run("uncharted/export.py", "--name", "demo-random", "--id", "random", "--title", "A random map (invented)",
        "--description", "Made-up points with invented names, to show what a map looks like.",
        "--corpus-info", info, "--candidates", "data/gaps/demo-random-connections", "--out", args.out)


if __name__ == "__main__":
    main()
