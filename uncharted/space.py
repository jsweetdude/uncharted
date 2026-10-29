"""Map an embedded corpus: nearest neighbors, loneliness, neighborhoods, flat 2-D layout.

    python -I uncharted/space.py --name papers
    python -I uncharted/space.py --name talks --min-cluster-size 5

Reads data/embeddings/<name>.npy and <name>.meta.jsonl (from embed.py). Writes next to them:

    <name>.nn.npz          top-K neighbor indices and cosine similarities per item
    <name>.items.jsonl     per item: nn1 similarity, loneliness percentile, cluster, map x/y
    <name>.umap2d.npy      2-D coordinates for the flat map
    <name>.centroids.npy   one unit vector per cluster (row = cluster id)
    <name>.clusters.json   neighborhoods: size, label terms, popularity, density, examples
    <name>.space.json      corpus-level stats: nn1 percentiles, cluster count, noise count

"Loneliness percentile" of an item is the share of the corpus whose nearest neighbor is
closer than this item's: 99.9 means lonelier than 99.9% of the corpus.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # `python -I` leaves the script's directory off sys.path
from provenance import corpus_fingerprint  # noqa: E402

STOP = set(
    """a an the and or of for to in on at by with from into via as is are be been was were
    this that these those it its their them they we our you your how what why when where
    which who will can may not no new more than about over under between through toward
    towards using use used based study approach approaches paper work towards case cases
    analysis evaluation evaluating results research toward during within without""".split()
)


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def load(base: str) -> tuple[np.ndarray, list[dict]]:
    matrix = np.load(base + ".npy").astype(np.float32)
    with open(base + ".meta.jsonl", encoding="utf-8") as fh:
        meta = [json.loads(line) for line in fh]
    if len(meta) != matrix.shape[0]:
        raise SystemExit(f"{base}.npy has {matrix.shape[0]} rows but {base}.meta.jsonl has {len(meta)}: they come from different embedding runs; embed again")
    try:
        with open(base + ".model.json", encoding="utf-8") as fh:
            info = json.load(fh)
    except (OSError, json.JSONDecodeError):
        info = {}
    if info.get("fingerprint") and info["fingerprint"] != corpus_fingerprint(meta, info):
        raise SystemExit(f"{base}.model.json describes different rows than {base}.meta.jsonl (an embedding run stopped while writing them?); embed again")
    return matrix, meta


def topk_neighbors(matrix: np.ndarray, k: int = 10, chunk: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    n = matrix.shape[0]
    k = min(k, n - 1)
    idx = np.zeros((n, k), dtype=np.int32)
    sims = np.zeros((n, k), dtype=np.float32)
    for start in range(0, n, chunk):
        block = matrix[start : start + chunk] @ matrix.T
        rows = np.arange(block.shape[0])
        block[rows, start + rows] = -1.0  # never your own neighbor
        part = np.argpartition(-block, k, axis=1)[:, :k]
        part_sims = np.take_along_axis(block, part, axis=1)
        order = np.argsort(-part_sims, axis=1)
        idx[start : start + chunk] = np.take_along_axis(part, order, axis=1)
        sims[start : start + chunk] = np.take_along_axis(part_sims, order, axis=1)
    return idx, sims


def loneliness_percentile(nn1: np.ndarray) -> np.ndarray:
    """Share of the other items whose nearest neighbor is more similar than this item's, 0 to
    100. Ties count as not more similar, so the loneliest item scores 100 only when no other
    item is as lonely."""
    ranked = np.sort(nn1)
    more_crowded = len(nn1) - np.searchsorted(ranked, nn1, side="right")
    return (100.0 * more_crowded / max(len(nn1) - 1, 1)).astype(np.float32)


def title_terms(title: str) -> list[str]:
    words = re.findall(r"[a-z][a-z\-]{2,}", title.lower())
    return [w for w in words if w not in STOP]


def distinctive(counter_in: collections.Counter, counter_all: collections.Counter, size: int, n: int, top: int = 6) -> list[str]:
    scored = []
    for term, c_in in counter_in.items():
        if c_in < 2:
            continue
        rate_in = c_in / size
        rate_all = (counter_all[term] + 1) / n
        scored.append((c_in * math.log(rate_in / rate_all), term))
    scored.sort(reverse=True)
    return [t for _, t in scored[:top]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True)
    ap.add_argument("--dir", default="data/embeddings")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--min-cluster-size", type=int, default=0, help="HDBSCAN min cluster size (default: sqrt(n)/2, at least 5)")
    ap.add_argument("--min-samples", type=int, default=5, help="HDBSCAN min samples; left equal to the cluster size, as HDBSCAN's own default does, a map can collapse into one blob")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    base = os.path.join(args.dir, args.name)
    matrix, meta = load(base)
    n = matrix.shape[0]
    log(f"{args.name}: {n} items x {matrix.shape[1]} dims")

    idx, sims = topk_neighbors(matrix, args.k)
    nn1 = sims[:, 0]
    pct = loneliness_percentile(nn1)
    np.savez_compressed(base + ".nn.npz", idx=idx, sims=sims)
    log(f"nearest-neighbor similarity: p10 {np.percentile(nn1,10):.3f} median {np.median(nn1):.3f} p90 {np.percentile(nn1,90):.3f} max {nn1.max():.3f}")

    import sklearn  # noqa: E402
    import umap  # noqa: E402  (slow import, keep it after the cheap work)
    from sklearn.cluster import HDBSCAN  # noqa: E402

    t0 = time.time()
    reducer10 = umap.UMAP(n_components=10, n_neighbors=30, min_dist=0.0, metric="cosine", random_state=args.seed)
    low = reducer10.fit_transform(matrix)
    log(f"umap 10-d in {time.time()-t0:.0f}s")
    mcs = args.min_cluster_size or max(5, int(math.sqrt(n) / 2))
    labels = HDBSCAN(min_cluster_size=mcs, min_samples=args.min_samples, copy=True).fit_predict(low)
    selection = "eom"
    sizes = np.bincount(labels[labels >= 0]) if labels.max() >= 0 else np.zeros(0, dtype=int)
    if len(sizes) < 3 or sizes.max() > 0.5 * n:
        # HDBSCAN's default selection sometimes keeps one blob over most of the map; its leaf
        # selection splits that blob into the dense regions inside it
        leaf = HDBSCAN(min_cluster_size=mcs, min_samples=args.min_samples, cluster_selection_method="leaf", copy=True).fit_predict(low)
        if leaf.max() + 1 > len(sizes):
            log(f"default selection kept {len(sizes)} neighborhoods, the largest {sizes.max() if len(sizes) else 0} of {n} items; using leaf selection")
            labels, selection = leaf, "leaf"
    n_clusters = int(labels.max()) + 1
    noise = int((labels == -1).sum())
    log(f"hdbscan min_cluster_size={mcs} min_samples={args.min_samples} selection={selection}: {n_clusters} clusters, {noise} unclustered")

    t0 = time.time()
    reducer2 = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric="cosine", random_state=args.seed)
    xy = reducer2.fit_transform(matrix).astype(np.float32)
    np.save(base + ".umap2d.npy", xy)
    log(f"umap 2-d in {time.time()-t0:.0f}s")

    # Neighborhood summaries
    tag_all = collections.Counter(t for m in meta for t in set(m.get("tags") or []))
    term_all = collections.Counter(t for m in meta for t in set(title_terms(m["title"])))
    clusters = []
    centroids = np.zeros((max(n_clusters, 1), matrix.shape[1]), dtype=np.float32)
    for c in range(n_clusters):
        members = np.where(labels == c)[0]
        size = len(members)
        tag_in = collections.Counter(t for i in members for t in set(meta[i].get("tags") or []))
        term_in = collections.Counter(t for i in members for t in set(title_terms(meta[i]["title"])))
        centroid = matrix[members].mean(axis=0)
        centroid /= np.linalg.norm(centroid) + 1e-9
        centroids[c] = centroid
        central = members[np.argsort(-(matrix[members] @ centroid))[:3]]
        clusters.append(
            {
                "cluster": c,
                "size": size,
                "label_tags": distinctive(tag_in, tag_all, size, n),
                "label_terms": distinctive(term_in, term_all, size, n),
                "popularity": int(sum(meta[i].get("popularity") or 0 for i in members)),
                "density_median_nn1": float(np.median(nn1[members])),
                "x": float(xy[members, 0].mean()),
                "y": float(xy[members, 1].mean()),
                "examples": [meta[i]["title"] for i in central],
            }
        )
    np.save(base + ".centroids.npy", centroids)
    with open(base + ".clusters.json", "w", encoding="utf-8") as fh:
        json.dump(clusters, fh, indent=1, ensure_ascii=False)

    with open(base + ".items.jsonl", "w", encoding="utf-8") as fh:
        for i, m in enumerate(meta):
            fh.write(
                json.dumps(
                    {
                        "i": i,
                        "id": m["id"],
                        "title": m["title"],
                        "year": m.get("year"),
                        "venue": m.get("venue"),
                        "nn1": float(nn1[i]),
                        "lonely_pct": float(pct[i]),
                        "cluster": int(labels[i]),
                        "x": float(xy[i, 0]),
                        "y": float(xy[i, 1]),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    summary = {
        "name": args.name,
        "items": n,
        "clusters": n_clusters,
        "unclustered": noise,
        "min_cluster_size": mcs,
        "params": {
            "neighbors": args.k,
            "seed": args.seed,
            "minClusterSize": mcs,
            "minSamples": args.min_samples,
            "selection": selection,
            "clustering": "HDBSCAN on a 10-d UMAP (30 neighbors, min distance 0, cosine)",
            "layout": "2-d UMAP (15 neighbors, min distance 0.1, cosine)",
            "libraries": f"umap-learn {umap.__version__}, scikit-learn {sklearn.__version__}, numpy {np.__version__}",
        },
        "nn1_percentiles": {str(p): float(np.percentile(nn1, p)) for p in (1, 5, 10, 25, 50, 75, 90, 95, 99)},
    }
    with open(base + ".space.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    # Console report
    by_size = sorted(clusters, key=lambda c: -c["size"])
    print("\nBiggest neighborhoods:")
    for c in by_size[:10]:
        print(f"  #{c['cluster']:>3} n={c['size']:<5} {', '.join(c['label_terms'][:4]) or '-'} | tags: {', '.join(c['label_tags'][:3]) or '-'}")
    if any(c["popularity"] for c in clusters):
        print("\nMost popular neighborhoods:")
        for c in sorted(clusters, key=lambda c: -c["popularity"])[:5]:
            print(f"  #{c['cluster']:>3} pop={c['popularity']:<8} {', '.join(c['label_terms'][:4])}")
    print("\nTightest (most clone-like) and loosest neighborhoods:")
    by_density = sorted(clusters, key=lambda c: -c["density_median_nn1"])
    for c in by_density[:3] + by_density[-3:]:
        print(f"  #{c['cluster']:>3} median nn1={c['density_median_nn1']:.3f} n={c['size']:<4} {', '.join(c['label_terms'][:4])}")
    print("\nLoneliest items:")
    for i in np.argsort(nn1)[:10]:
        m = meta[i]
        print(f"  {pct[i]:5.1f}%  nn1={nn1[i]:.3f}  {m['title'][:90]} ({m.get('year')}, {m.get('venue')})")
    print("\nMost cloned items:")
    for i in np.argsort(-nn1)[:5]:
        m, j = meta[i], idx[i, 0]
        print(f"  nn1={nn1[i]:.3f}  {m['title'][:70]} ~ {meta[j]['title'][:70]}")


if __name__ == "__main__":
    main()
