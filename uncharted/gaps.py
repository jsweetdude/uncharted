"""Find empty spots in an embedded corpus and turn them into writing briefs.

    python -I uncharted/gaps.py blend --name talks --titles "annotation" "forced colors" --out data/gaps/annot-colors
    python -I uncharted/gaps.py blend --name papers --clusters 12 40 --out data/gaps/c12-c40
    python -I uncharted/gaps.py holes --name papers --top 8 --out-dir data/gaps/holes

blend: average the parents' vectors, then push the point away from every item in the
corpus, tethered to the blend so it stays specific, until it is lonelier than
--target-pct of the corpus or has drifted as far from the blend as --max-drift allows.

holes: find empty regions of the flat 2-D map with items on (nearly) every side, anchor
a point on the ring of items around each hole, and push the same way.

Both write <out>.json (the measurements) and <out>.md (the brief a writer works from).
Needs the files space.py writes (items, clusters, centroids, umap2d).
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import re
import time

import numpy as np


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-9)


def slugify(text: str, limit: int = 50) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:limit]


class Space:
    def __init__(self, directory: str, name: str):
        base = os.path.join(directory, name)
        self.name = name
        self.X = np.load(base + ".npy").astype(np.float32)
        with open(base + ".meta.jsonl", encoding="utf-8") as fh:
            self.meta = [json.loads(line) for line in fh]
        with open(base + ".items.jsonl", encoding="utf-8") as fh:
            self.items = [json.loads(line) for line in fh]
        with open(base + ".clusters.json", encoding="utf-8") as fh:
            self.clusters = json.load(fh)
        with open(base + ".space.json", encoding="utf-8") as fh:
            self.stats = json.load(fh)
        self.centroids = np.load(base + ".centroids.npy")
        self.xy = np.load(base + ".umap2d.npy")
        self.nn1 = np.array([it["nn1"] for it in self.items], dtype=np.float32)
        self.tag_freq = collections.Counter(t for m in self.meta for t in set(m.get("tags") or []))

    def n(self) -> int:
        return self.X.shape[0]

    def lonely_pct(self, sim: float) -> float:
        """Share of the corpus whose nearest neighbor is closer than `sim` (0-100)."""
        return float(100.0 * np.mean(self.nn1 > sim))

    def sim_for_pct(self, pct: float) -> float:
        """The nn1 value an item needs to be lonelier than `pct` percent of the corpus."""
        return float(np.percentile(self.nn1, 100 - pct))

    def find_by_title(self, needle: str) -> int:
        hits = [i for i, m in enumerate(self.meta) if needle.lower() in m["title"].lower()]
        if len(hits) == 1:
            return hits[0]
        if not hits:
            raise SystemExit(f"no title contains {needle!r}")
        sample = "; ".join(self.meta[i]["title"][:60] for i in hits[:5])
        raise SystemExit(f"{len(hits)} titles contain {needle!r}: {sample} ...")

    def find_by_id(self, item_id: str) -> int:
        for i, m in enumerate(self.meta):
            if m["id"] == item_id:
                return i
        raise SystemExit(f"no item with id {item_id}")

    def cluster_label(self, c: int) -> str:
        if c < 0 or c >= len(self.clusters):
            return "unclustered"
        cl = self.clusters[c]
        return f"#{c} ({', '.join(cl['label_terms'][:4]) or ', '.join(cl['label_tags'][:3]) or 'unlabeled'}, n={cl['size']})"

    def nearest(self, p: np.ndarray, k: int = 8, exclude: set[int] | None = None) -> list[tuple[int, float]]:
        sims = self.X @ p
        order = np.argsort(-sims)
        out = []
        for i in order:
            if exclude and int(i) in exclude:
                continue
            out.append((int(i), float(sims[i])))
            if len(out) == k:
                break
        return out

    def gist(self, i: int, words: int = 30) -> str:
        text = self.meta[i]["text"]
        title = self.meta[i]["title"]
        body = text[len(title) + 2 :] if text.startswith(title) else text
        return " ".join(body.split()[:words])

    def item_line(self, i: int, sim: float | None = None) -> str:
        m = self.meta[i]
        head = f"{sim:.3f}  " if sim is not None else ""
        return f"{head}{m['title']} ({m.get('year')}, {m.get('venue') or 'unknown venue'}) — {self.gist(i)}"

    def leaned_tags(self, p: np.ndarray, k: int = 30, top: int = 8) -> list[str]:
        near = self.nearest(p, k)
        score: dict[str, float] = collections.defaultdict(float)
        total_sim = sum(s for _, s in near)
        for i, s in near:
            for t in set(self.meta[i].get("tags") or []):
                score[t] += s
        n = self.n()
        ranked = sorted(score.items(), key=lambda kv: -(kv[1] - total_sim * self.tag_freq[kv[0]] / n))
        return [t for t, _ in ranked[:top]]

    def nearest_clusters(self, p: np.ndarray, k: int = 3) -> list[tuple[int, float]]:
        if len(self.centroids) == 0:
            return []
        sims = self.centroids @ p
        return [(int(c), float(sims[c])) for c in np.argsort(-sims)[:k]]


def repel(
    space: Space,
    p0: np.ndarray,
    target_sim: float,
    max_drift: float = 0.75,
    tau: float = 0.05,
    tether: float = 1.0,
    lr: float = 0.01,
    steps: int = 400,
    pool: int = 4000,
) -> tuple[np.ndarray, dict]:
    """Push p away from nearby items (soft weights, hottest on the closest) while a tether
    holds it near p0. Stops at the target loneliness, at the drift limit, or after `steps`."""
    X = space.X
    p = p0.copy()
    pool_idx = np.argsort(-(X @ p0))[: min(pool, space.n())]
    Xp = X[pool_idx]
    history = []
    reason = "max steps"
    for t in range(steps):
        if t % 50 == 0 and t:
            pool_idx = np.argsort(-(X @ p))[: min(pool, space.n())]
            Xp = X[pool_idx]
        s = Xp @ p
        nn1 = float(s.max())
        drift = float(p @ p0)
        history.append((nn1, drift))
        if nn1 <= target_sim:
            reason = "target reached"
            break
        if drift <= max_drift:
            reason = "drift limit"
            break
        w = np.exp((s - 1.0) / tau)
        push = (Xp * (w[:, None] / tau)).sum(axis=0)
        grad = unit(push) - tether * p0
        p = unit(p - lr * unit(grad))
    nn1 = float((X @ p).max())
    return p, {"steps": len(history), "stop": reason, "start_nn1": history[0][0] if history else None, "end_nn1": nn1, "drift_sim": float(p @ p0)}


def describe(space: Space, p0: np.ndarray, p: np.ndarray, parents: list[int], info: dict, target_pct: float, how: str) -> dict:
    near = space.nearest(p, 8, exclude=set(parents))
    near0 = space.nearest(p0, 5)
    pair_sims = []
    for a in range(len(parents)):
        for b in range(a + 1, len(parents)):
            pair_sims.append(float(space.X[parents[a]] @ space.X[parents[b]]))
    return {
        "corpus": space.name,
        "corpus_items": space.n(),
        "how": how,
        "target_pct": target_pct,
        "target_sim": space.sim_for_pct(target_pct),
        "corpus_nn1_median": float(np.median(space.nn1)),
        "parents": [{"i": i, "id": space.meta[i]["id"], "title": space.meta[i]["title"], "year": space.meta[i].get("year"), "venue": space.meta[i].get("venue")} for i in parents],
        "parent_pairwise_sim_min": min(pair_sims) if pair_sims else None,
        "blend_nn1": info.get("start_nn1"),
        "blend_nearest": [{"i": i, "sim": s, "title": space.meta[i]["title"]} for i, s in near0],
        "gap_nn1": info["end_nn1"],
        "gap_lonely_pct": space.lonely_pct(info["end_nn1"]),
        "drift_sim": info["drift_sim"],
        "stop": info["stop"],
        "steps": info["steps"],
        "nearest_clusters": [{"cluster": c, "sim": s, "label": space.cluster_label(c)} for c, s in space.nearest_clusters(p)],
        "leaned_tags": space.leaned_tags(p),
        "must_not_copy": [{"i": i, "id": space.meta[i]["id"], "sim": s, "title": space.meta[i]["title"], "year": space.meta[i].get("year"), "venue": space.meta[i].get("venue"), "gist": space.gist(i)} for i, s in near],
        "point": [round(float(v), 6) for v in p],
        "blend_point": [round(float(v), 6) for v in p0],
    }


KIND_INSTRUCTIONS = {
    "paper": "a research-paper title and a 150–250 word abstract in the register of the corpus: the users and the barrier, the system or method, how it was evaluated, and the main finding. Specific nouns. No survey, no manifesto.",
    "talk": "a conference-talk title and a 120–200 word abstract in the register of the corpus: who the talk is for, the concrete practice or finding it delivers, and what attendees will be able to do afterwards. Specific, first-person-practitioner tone. No keynote generalities.",
    "patent": "an invention-disclosure title and a 150–250 word abstract in the register of a US patent abstract: the problem, what existing systems do and where they stop short, the mechanism of the invention (its components, the signals or data they exchange, the order of operations), and one closing sentence on what an independent claim would cover. The title names the mechanism's own nouns, not its benefit. No marketing language, no list of advantages.",
}


def brief_markdown(d: dict, kind: str) -> str:
    lines = [f"# Gap brief — {d['corpus']} ({d['how']})", ""]
    lines.append(
        f"Corpus: {d['corpus_items']} items; typical item sits at {d['corpus_nn1_median']:.3f} similarity to its nearest neighbor. "
        f"Target for a new item: nearest-neighbor similarity ≤ {d['target_sim']:.3f} (lonelier than {d['target_pct']:.0f}% of the corpus)."
    )
    lines.append("")
    lines.append("## Where the gap sits")
    lines.append("")
    for c in d["nearest_clusters"]:
        lines.append(f"- neighborhood {c['label']} (similarity {c['sim']:.3f})")
    lines.append("")
    lines.append("## Ingredients (parents or ring items)")
    lines.append("")
    for p in d["parents"]:
        lines.append(f"- {p['title']} ({p['year']}, {p['venue'] or 'unknown venue'})")
    if d.get("parent_pairwise_sim_min") is not None:
        warn = "  ← ingredients are far apart; keep the blend specific or it turns to mush" if d["parent_pairwise_sim_min"] < d["corpus_nn1_median"] - 0.12 else ""
        lines.append(f"\nLeast-related pair of ingredients: {d['parent_pairwise_sim_min']:.3f} similarity.{warn}")
    lines.append("")
    lines.append("## Leans toward (tags of what surrounds the gap)")
    lines.append("")
    lines.append(", ".join(d["leaned_tags"]) or "(no tags in this corpus)")
    lines.append("")
    lines.append("## Must not copy (nearest existing items to the gap point)")
    lines.append("")
    for m in d["must_not_copy"]:
        lines.append(f"- {m['sim']:.3f}  **{m['title']}** ({m['year']}, {m['venue'] or 'unknown venue'}): {m['gist']}")
    lines.append("")
    lines.append("## Measurements")
    lines.append("")
    lines.append(f"- blend point nearest neighbor: {d['blend_nn1']:.3f}" if d.get("blend_nn1") is not None else "- blend point nearest neighbor: n/a")
    lines.append(f"- gap point nearest neighbor: {d['gap_nn1']:.3f} (lonelier than {d['gap_lonely_pct']:.1f}% of the corpus); stopped: {d['stop']} after {d['steps']} steps")
    lines.append(f"- gap point still {d['drift_sim']:.3f} similar to the blend (1.0 = did not move)")
    lines.append("")
    lines.append("## Write")
    lines.append("")
    lines.append(
        f"Write {KIND_INSTRUCTIONS.get(kind, KIND_INSTRUCTIONS['paper'])} "
        "It must live in this gap: build on the ingredients, lean toward the tags, and take none of the must-not-copy items' core ideas."
    )
    lines.append("")
    lines.append(
        "The embedding model weighs topic words heavily: a draft that repeats the neighborhood's standard vocabulary "
        "(the words in the must-not-copy titles) lands next to them whatever its idea. Put the mechanism's own nouns in the "
        "title and the first sentence; use the field's standard terms once, where the reader needs them. "
        "Cut framing sentences that restate what the field already knows."
    )
    lines.append("")
    lines.append(
        "End the draft with a line containing only `---`, then three lines: `Helps:` (who, specifically), `Barrier:` (what they "
        "cannot do today), `Standard:` (the guideline, law or practice it touches, or `none`). verify.py keeps that footer out of the "
        "embedding and records it as the value check. Then run verify.py; a RETRY names the item that is too close."
    )
    lines.append("")
    return "\n".join(lines)


def write_out(d: dict, out: str, kind: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    with open(out + ".json", "w", encoding="utf-8") as fh:
        json.dump(d, fh, ensure_ascii=False)
    with open(out + ".md", "w", encoding="utf-8") as fh:
        fh.write(brief_markdown(d, kind))
    log(f"wrote {out}.json and {out}.md — gap nn1 {d['gap_nn1']:.3f} (lonelier than {d['gap_lonely_pct']:.1f}%), drift {d['drift_sim']:.3f}, stop: {d['stop']}")


def cmd_blend(args: argparse.Namespace) -> None:
    space = Space(args.dir, args.name)
    parents: list[int] = []
    for t in args.titles or []:
        parents.append(space.find_by_title(t))
    for i in args.ids or []:
        parents.append(space.find_by_id(i))
    vectors = [space.X[i] for i in parents]
    how = "blend of items"
    if args.clusters:
        for c in args.clusters:
            vectors.append(space.centroids[c])
        how = "blend of neighborhoods " + ", ".join(f"#{c}" for c in args.clusters)
        if not parents:
            for c in args.clusters:
                members = [it["i"] for it in space.items if it["cluster"] == c]
                central = max(members, key=lambda i: float(space.X[i] @ space.centroids[c]))
                parents.append(central)
    if not vectors:
        raise SystemExit("give --titles, --ids or --clusters")
    p0 = unit(np.mean(vectors, axis=0))
    target_sim = space.sim_for_pct(args.target_pct)
    p, info = repel(space, p0, target_sim, max_drift=args.max_drift, tau=args.tau, tether=args.tether, lr=args.lr, steps=args.steps)
    d = describe(space, p0, p, parents, info, args.target_pct, how)
    out = args.out or os.path.join("data/gaps", slugify("-".join(space.meta[i]["title"] for i in parents[:3])))
    write_out(d, out, args.kind)


def find_holes(space: Space, grid: int = 60, sectors: int = 8, min_sectors: int = 7, reach: float = 0.12, top: int = 8, min_clusters: int = 2) -> list[dict]:
    xy = space.xy
    cluster_of = np.array([it["cluster"] for it in space.items])
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    span = float(max(hi - lo))
    radius_reach = reach * span
    gx = np.linspace(lo[0], hi[0], grid)
    gy = np.linspace(lo[1], hi[1], grid)
    cands = np.array([(x, y) for x in gx for y in gy], dtype=np.float32)
    holes = []
    for cx, cy in cands:
        d = np.hypot(xy[:, 0] - cx, xy[:, 1] - cy)
        empty_radius = float(d.min())
        if empty_radius < 0.01 * span:
            continue
        within = np.where(d < radius_reach)[0]
        within = within[cluster_of[within] >= 0]  # a hole is judged by the neighborhoods around it, not by stray items
        if len(within) < sectors:
            continue
        ang = np.arctan2(xy[within, 1] - cy, xy[within, 0] - cx)
        sector = ((ang + math.pi) / (2 * math.pi) * sectors).astype(int) % sectors
        covered = len(set(sector.tolist()))
        if covered < min_sectors:
            continue
        ring = []
        for s in range(sectors):
            members = within[sector == s]
            if len(members):
                ring.append(int(members[np.argmin(d[members])]))
        if len({int(cluster_of[i]) for i in ring}) < min_clusters:
            continue
        holes.append({"x": float(cx), "y": float(cy), "empty_radius": empty_radius, "covered_sectors": covered, "ring": ring})
    holes.sort(key=lambda h: -h["empty_radius"])
    picked: list[dict] = []
    for h in holes:
        if all(math.hypot(h["x"] - q["x"], h["y"] - q["y"]) > h["empty_radius"] + q["empty_radius"] for q in picked):
            picked.append(h)
        if len(picked) == top:
            break
    return picked


def cmd_holes(args: argparse.Namespace) -> None:
    space = Space(args.dir, args.name)
    holes = find_holes(space, grid=args.grid, reach=args.reach, top=args.top, min_clusters=args.min_clusters)
    log(f"{len(holes)} holes on the flat map of {space.name}")
    os.makedirs(args.out_dir, exist_ok=True)
    target_sim = space.sim_for_pct(args.target_pct)
    for k, h in enumerate(holes, start=1):
        p0 = unit(space.X[h["ring"]].mean(axis=0))
        p, info = repel(space, p0, target_sim, max_drift=args.max_drift, tau=args.tau, tether=args.tether, lr=args.lr, steps=args.steps)
        ring_clusters = collections.Counter(space.items[i]["cluster"] for i in h["ring"])
        how = "hole between " + ", ".join(space.cluster_label(c) for c, _ in ring_clusters.most_common(3))
        d = describe(space, p0, p, h["ring"], info, args.target_pct, how)
        d["hole"] = {"x": h["x"], "y": h["y"], "empty_radius": h["empty_radius"], "covered_sectors": h["covered_sectors"]}
        write_out(d, os.path.join(args.out_dir, f"hole-{k:02d}"), args.kind)


def cmd_pairs(args: argparse.Namespace) -> None:
    """Rank related pairs of neighborhoods by how empty the space between them is."""
    space = Space(args.dir, args.name)
    C = space.centroids
    k = len(C)
    if k < 2:
        raise SystemExit("need at least two neighborhoods; run space.py with a smaller --min-cluster-size")
    csim = C @ C.T
    upper = csim[np.triu_indices(k, 1)]
    log(f"{k} neighborhoods; centroid similarity between pairs: p10 {np.percentile(upper,10):.3f} median {np.median(upper):.3f} p90 {np.percentile(upper,90):.3f}")
    allowed = set(args.only) if args.only else None
    pairs = [
        (a, b, float(csim[a, b]))
        for a in range(k)
        for b in range(a + 1, k)
        if args.min_related <= csim[a, b] <= args.max_related and (allowed is None or (a in allowed and b in allowed))
    ]
    if not pairs:
        raise SystemExit("no pairs in the --min-related/--max-related band (or none among --only); widen it")
    M = np.array([unit(C[a] + C[b]) for a, b, _ in pairs], dtype=np.float32)
    mid_nn1 = np.zeros(len(pairs), dtype=np.float32)
    for s0 in range(0, len(pairs), 512):
        mid_nn1[s0 : s0 + 512] = (M[s0 : s0 + 512] @ space.X.T).max(axis=1)
    ranked = sorted(range(len(pairs)), key=lambda i: float(mid_nn1[i]))[: args.top]
    log(f"{len(pairs)} related pairs (centroid similarity {args.min_related}–{args.max_related}); emptiest midpoints first:")
    for i in ranked:
        a, b, s = pairs[i]
        print(f"  midpoint nn1 {mid_nn1[i]:.3f} (lonelier than {space.lonely_pct(float(mid_nn1[i])):5.1f}%)  related {s:.3f}  {space.cluster_label(a)}  +  {space.cluster_label(b)}")
    if not args.briefs:
        return
    central: dict[int, int] = {}
    for it in space.items:
        c = it["cluster"]
        if c < 0:
            continue
        score = float(space.X[it["i"]] @ C[c])
        if c not in central or score > float(space.X[central[c]] @ C[c]):
            central[c] = it["i"]
    target_sim = space.sim_for_pct(args.target_pct)
    os.makedirs(args.out_dir, exist_ok=True)
    for n, i in enumerate(ranked, start=1):
        a, b, s = pairs[i]
        p0 = M[i]
        p, info = repel(space, p0, target_sim, max_drift=args.max_drift, tau=args.tau, tether=args.tether, lr=args.lr, steps=args.steps)
        d = describe(space, p0, p, [central[a], central[b]], info, args.target_pct, f"blend of neighborhoods #{a} + #{b}")
        d["pair"] = {"a": a, "b": b, "centroid_sim": s, "midpoint_nn1": float(mid_nn1[i])}
        write_out(d, os.path.join(args.out_dir, f"pair-{n:02d}-c{a}-c{b}"), args.kind)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("blend", "holes", "pairs"):
        sp = sub.add_parser(name)
        sp.add_argument("--name", required=True)
        sp.add_argument("--dir", default="data/embeddings")
        sp.add_argument("--kind", default="paper", choices=sorted(KIND_INSTRUCTIONS))
        sp.add_argument("--target-pct", type=float, default=95.0, help="stop once lonelier than this share of the corpus")
        sp.add_argument("--max-drift", type=float, default=0.75, help="stop once this dissimilar to the blend point")
        sp.add_argument("--tau", type=float, default=0.05)
        sp.add_argument("--tether", type=float, default=1.0)
        sp.add_argument("--lr", type=float, default=0.01)
        sp.add_argument("--steps", type=int, default=400)
    b = sub.choices["blend"]
    b.add_argument("--titles", nargs="*", help="title substrings, each matching exactly one item")
    b.add_argument("--ids", nargs="*")
    b.add_argument("--clusters", nargs="*", type=int, help="blend neighborhood centroids")
    b.add_argument("--out", help="output basename (default: data/gaps/<slug>)")
    h = sub.choices["holes"]
    h.add_argument("--top", type=int, default=8)
    h.add_argument("--grid", type=int, default=60)
    h.add_argument("--reach", type=float, default=0.12, help="ring radius as a share of the map's extent")
    h.add_argument("--min-clusters", type=int, default=2, help="a hole must sit between at least this many neighborhoods")
    h.add_argument("--out-dir", default="data/gaps/holes")
    pr = sub.choices["pairs"]
    pr.add_argument("--top", type=int, default=15)
    pr.add_argument("--min-related", type=float, default=0.60, help="ignore pairs of neighborhoods less similar than this (mush)")
    pr.add_argument("--max-related", type=float, default=0.85, help="ignore pairs more similar than this (same topic)")
    pr.add_argument("--only", nargs="*", type=int, help="consider only pairs where both neighborhoods are in this list")
    pr.add_argument("--briefs", action="store_true", help="also push each midpoint into the gap and write a brief")
    pr.add_argument("--out-dir", default="data/gaps/pairs")
    args = ap.parse_args()
    {"blend": cmd_blend, "holes": cmd_holes, "pairs": cmd_pairs}[args.cmd](args)


if __name__ == "__main__":
    main()
