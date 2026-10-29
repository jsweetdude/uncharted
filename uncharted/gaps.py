"""Find empty spots in an embedded corpus and turn them into writing briefs.

    python -I uncharted/gaps.py blend --name talks --titles "annotation" "forced colors" --out data/gaps/annot-colors
    python -I uncharted/gaps.py blend --name papers --clusters 12 40 --out data/gaps/c12-c40
    python -I uncharted/gaps.py holes --name papers --top 8 --out-dir data/gaps/holes

blend: average the parents' vectors, then push the point away from every item in the
corpus, tethered to the blend so it stays specific, until it is lonelier than
--target-pct of the corpus or has drifted as far from the blend as --max-drift allows.

connections: for every pair of related neighborhoods, mix random items from the two and keep
the pairs whose mixtures are emptier than chance, against mixtures matched on how similar
their parents are and how crowded their region is; push each kept pair's emptiest mixture
the same way. This is version 1's way of finding candidates (see cmd_connections).

pairs: rank related pairs of neighborhoods by how empty the midpoint of their centroids is.

holes: find empty regions of the flat 2-D map with items on (nearly) every side, anchor
a point on the ring of items around each hole, and push the same way. Version 0 only: the
flat map can hide real gaps and invent false ones, and export.py leaves holes out.

Each writes <out>.json (the measurements) and <out>.md (the brief a writer works from).
Needs the files space.py writes (items, clusters, centroids, umap2d, nn).
"""

from __future__ import annotations

import argparse
import collections
import glob
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
    holds it near p0. Stops at the target loneliness, at the drift limit, or after `steps`.
    The drift limit is hard: a step that would cross it is not taken. The target counts as
    reached only when the whole corpus, not just the nearby pool, agrees."""
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
        history.append((nn1, float(p @ p0)))
        if nn1 <= target_sim:
            if float((X @ p).max()) <= target_sim:
                reason = "target reached"
                break
            pool_idx = np.argsort(-(X @ p))[: min(pool, space.n())]  # something outside the pool is close: look again
            Xp = X[pool_idx]
            continue
        w = np.exp((s - 1.0) / tau)
        push = (Xp * (w[:, None] / tau)).sum(axis=0)
        grad = unit(push) - tether * p0
        step = unit(p - lr * unit(grad))
        if float(step @ p0) <= max_drift:
            reason = "drift limit"
            break
        p = step
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


def effective_target(d: dict) -> tuple[float, float]:
    """(percent, similarity) a draft for this gap must reach. When the push stopped short of the
    target percentile, a draft is asked to be as lonely as the spot itself, not lonelier."""
    if d["gap_lonely_pct"] < d["target_pct"]:
        return d["gap_lonely_pct"], d["gap_nn1"]
    return d["target_pct"], d["target_sim"]


def brief_markdown(d: dict, kind: str) -> str:
    lines = [f"# Gap brief — {d['corpus']} ({d['how']})", ""]
    pct, sim = effective_target(d)
    short = " The push stopped before the {:.0f}% target, so a draft only has to be as lonely as this spot.".format(d["target_pct"]) if pct < d["target_pct"] else ""
    lines.append(
        f"Corpus: {d['corpus_items']} items; typical item sits at {d['corpus_nn1_median']:.3f} similarity to its nearest neighbor. "
        f"Target for a new item: nearest-neighbor similarity ≤ {sim:.3f} (lonelier than {pct:.0f}% of the corpus).{short}"
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


def related_band(C: np.ndarray, band: list[float], lo_abs: float | None = None, hi_abs: float | None = None) -> tuple[float, float, list[tuple[int, int, float]]]:
    """Pairs of neighborhoods whose centroid similarity lies in the related band: between the
    given percentiles of all pairs' similarities (or absolute bounds, when given). Related
    enough to combine meaningfully, not so similar that they are one topic."""
    k = len(C)
    csim = C @ C.T
    upper = csim[np.triu_indices(k, 1)]
    lo = float(np.percentile(upper, band[0])) if lo_abs is None else lo_abs
    hi = float(np.percentile(upper, band[1])) if hi_abs is None else hi_abs
    pairs = [(a, b, float(csim[a, b])) for a in range(k) for b in range(a + 1, k) if lo <= csim[a, b] <= hi]
    return lo, hi, pairs


def cmd_pairs(args: argparse.Namespace) -> None:
    """Rank related pairs of neighborhoods by how empty the space between them is."""
    space = Space(args.dir, args.name)
    C = space.centroids
    k = len(C)
    if k < 2:
        raise SystemExit("need at least two neighborhoods; run space.py with a smaller --min-cluster-size")
    lo, hi, pairs = related_band(C, args.band, args.min_related, args.max_related)
    allowed = set(args.only) if args.only else None
    pairs = [(a, b, s) for a, b, s in pairs if allowed is None or (a in allowed and b in allowed)]
    log(f"{k} neighborhoods; related band {lo:.3f}-{hi:.3f} centroid similarity")
    if not pairs:
        raise SystemExit("no pairs in the related band (or none among --only); widen --band")
    M = np.array([unit(C[a] + C[b]) for a, b, _ in pairs], dtype=np.float32)
    mid_nn1 = np.zeros(len(pairs), dtype=np.float32)
    for s0 in range(0, len(pairs), 512):
        mid_nn1[s0 : s0 + 512] = (M[s0 : s0 + 512] @ space.X.T).max(axis=1)
    ranked = sorted(range(len(pairs)), key=lambda i: float(mid_nn1[i]))[: args.top]
    log(f"{len(pairs)} related pairs (centroid similarity {lo:.3f}-{hi:.3f}); emptiest midpoints first:")
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
        d["pair"] = {"a": a, "b": b, "centroid_sim": s, "midpoint_nn1": float(mid_nn1[i]), "min_related": lo, "max_related": hi}
        write_out(d, os.path.join(args.out_dir, f"pair-{n:02d}-c{a}-c{b}"), args.kind)


def tail_test(member_groups: np.ndarray, member_tail: np.ndarray, group_size: np.ndarray, group_tail: np.ndarray) -> float:
    """Chance that a pair's mixtures hold at least this many tail places if mixtures were
    assigned to pairs at random within each matched group, the tail sets held fixed. Exact:
    within a group the pair's count is hypergeometric, and the groups are independent, so the
    pair's count is their convolution. Small groups, ties and a lone pair are handled exactly."""
    from scipy.stats import hypergeom

    observed = int(member_tail.sum())
    dist = np.array([1.0])
    for g in np.unique(member_groups):
        m = int((member_groups == g).sum())
        pmf = hypergeom.pmf(np.arange(m + 1), int(group_size[g]), int(group_tail[g]), m)
        dist = np.convolve(dist, pmf)
    return float(min(1.0, max(0.0, dist[observed:].sum())))


def quantile_bins(values: np.ndarray, k: int) -> np.ndarray:
    edges = np.quantile(values, np.linspace(0, 1, k + 1))
    return np.clip(np.searchsorted(edges, values, side="right") - 1, 0, k - 1)


def matched_ranks(scores: np.ndarray, groups: np.ndarray, valid: np.ndarray, tail_pct: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per mixture: its group's median score, the share of its group that is less empty
    (`beats`; ties are not less empty), and whether it sits in the group's emptiest tail."""
    baseline = np.full(len(scores), np.nan, dtype=np.float32)
    beats = np.zeros(len(scores), dtype=np.float32)
    for g in np.unique(groups[valid]):
        idx = np.where((groups == g) & valid)[0]
        vals = scores[idx]
        ranked = np.sort(vals)
        baseline[idx] = np.median(vals)
        higher = len(vals) - np.searchsorted(ranked, vals, side="right")
        beats[idx] = 100.0 * higher / max(len(vals) - 1, 1)
    return baseline, beats, valid & (beats >= 100 - tail_pct)


def cmd_connections(args: argparse.Namespace) -> None:
    """Candidate connections: pairs of related neighborhoods whose mixtures are emptier than
    comparable mixtures from other pairs. An exploratory, corpus-relative signal: it says where
    to look, not that an idea is new or good.

    1. For every pair of neighborhoods in the related band, draw --mixes random parent pairs
       (one item from each) and mix them with equal weights.
    2. Score each mixture by its nearest real item, leaving out its two parents and their
       --ring nearest neighbors, which a mixture sits close to by construction. A mixture with
       too few items left to compare against is not scored. The same is done leaving out the
       parents only, and reported alongside, since the rings could also hide work that
       already joins the two neighborhoods.
    3. Mixing always lands somewhere fairly empty, and how empty depends on how similar the
       parents are and how crowded their region is. Each mixture is ranked only within its
       matched group (quantile bins of parent similarity crossed with bins of the parents'
       mean nn1), pooled over every related pair, and the emptiest --tail percent of each
       group form its tail.
    4. A pair passes when its mixtures hold more tail places than random assignment of
       mixtures to pairs within groups would give (exact test, Bonferroni-corrected over the
       pairs tested), the tail mixtures draw on enough different parents from each side, and
       no single parent supplies more than --max-busiest of them.
    5. Each passing pair's emptiest tail mixture is pushed into the nearest empty space, as a
       blend is: a proposal for a writer, whose own measurements are separate from the
       pair's evidence.
    """
    for name in ("mixes", "max_mixtures", "ring", "sim_bins", "density_bins", "chunk", "top"):
        if getattr(args, name) < (0 if name == "ring" else 1):
            raise SystemExit(f"--{name.replace('_', '-')} must be {'zero or more' if name == 'ring' else 'positive'}")
    if not 0 < args.tail <= 50:
        raise SystemExit("--tail must be between 0 and 50 percent")
    space = Space(args.dir, args.name)
    if not np.isfinite(space.X).all():
        raise SystemExit("the embeddings contain values that are not finite; embed again")
    ring_all = np.load(os.path.join(args.dir, args.name + ".nn.npz"))["idx"]
    ring = ring_all[:, : min(args.ring, ring_all.shape[1])]
    labels = np.array([it["cluster"] for it in space.items])
    if len(space.centroids) < 2:
        raise SystemExit("need at least two neighborhoods; run space.py with a smaller --min-cluster-size")
    lo, hi, pairs = related_band(space.centroids, args.band)
    members = {c: np.where(labels == c)[0] for c in range(len(space.centroids))}
    pairs = [(a, b, sim) for a, b, sim in pairs if len(members[a]) and len(members[b])]
    if len(pairs) < 3:
        raise SystemExit(f"only {len(pairs)} related pairs of neighborhoods: too few to compare pairs with each other; widen --band or map a larger corpus")
    m = args.mixes if args.mixes * len(pairs) <= args.max_mixtures else args.max_mixtures // len(pairs)
    if m < 20:
        raise SystemExit(f"{len(pairs)} related pairs leave {m} mixtures each under --max-mixtures {args.max_mixtures}; raise it or narrow --band")
    rng = np.random.default_rng(args.seed)
    rows = np.concatenate([np.column_stack([np.full(m, pi), rng.choice(members[a], m), rng.choice(members[b], m)]) for pi, (a, b, _) in enumerate(pairs)])
    X = space.X
    n = space.n()
    t0 = time.time()
    with_ring = np.full(len(rows), np.nan, dtype=np.float32)
    parents_only = np.full(len(rows), np.nan, dtype=np.float32)
    mix_all = np.full(len(rows), np.nan, dtype=np.float32)
    for start in range(0, len(rows), args.chunk):
        part = rows[start : start + args.chunk]
        P = X[part[:, 1]] + X[part[:, 2]]
        norms = np.linalg.norm(P, axis=1, keepdims=True)
        ok = norms[:, 0] > 1e-6  # opposite parents cancel out: nothing to score
        P = P / np.maximum(norms, 1e-6)
        S = P @ X.T
        mix_all[start : start + args.chunk] = np.where(ok, S.max(axis=1), np.nan)
        np.put_along_axis(S, part[:, 1:3], -np.inf, axis=1)
        parents_only[start : start + args.chunk] = np.where(ok, S.max(axis=1), np.nan)
        excluded = np.concatenate([part[:, 1:3], ring[part[:, 1]], ring[part[:, 2]]], axis=1)
        np.put_along_axis(S, excluded, -np.inf, axis=1)
        left = n - np.array([len(set(e.tolist())) for e in excluded])
        with_ring[start : start + args.chunk] = np.where(ok & (left >= 10), S.max(axis=1), np.nan)
    parent_sim = np.einsum("ij,ij->i", X[rows[:, 1]], X[rows[:, 2]])
    density = (space.nn1[rows[:, 1]] + space.nn1[rows[:, 2]]) / 2
    groups = quantile_bins(parent_sim, args.sim_bins) * args.density_bins + quantile_bins(density, args.density_bins)
    n_groups = args.sim_bins * args.density_bins
    results = {}
    for variant, scores in (("ring", with_ring), ("parents", parents_only)):
        valid = np.isfinite(scores)
        baseline, beats, tail = matched_ranks(scores, groups, valid, args.tail)
        size = np.bincount(groups[valid], minlength=n_groups)
        tail_size = np.bincount(groups[tail], minlength=n_groups)
        results[variant] = (valid, baseline, beats, tail, size, tail_size)
    valid, baseline, beats, tail, size, tail_size = results["ring"]
    log(f"{len(pairs)} related pairs (centroid similarity {lo:.3f}-{hi:.3f}) x {m} mixtures = {len(rows)}, scored in {time.time() - t0:.0f}s; "
        f"{int((~valid).sum())} unscorable; mixture nn1 without parents and rings: median {np.nanmedian(with_ring):.3f}; {100 * tail.sum() / max(valid.sum(), 1):.1f}% in the tail")

    cutoff = args.alpha / len(pairs)
    table = []
    for pi, (a, b, csim) in enumerate(pairs):
        mine = np.where((rows[:, 0] == pi) & valid)[0]
        hits = mine[tail[mine]]
        p = tail_test(groups[mine], tail[mine], size, tail_size) if len(mine) else 1.0
        pv, _, _, tail_p, size_p, tail_size_p = results["parents"]
        mine_p = np.where((rows[:, 0] == pi) & pv)[0]
        p_parents = tail_test(groups[mine_p], tail_p[mine_p], size_p, tail_size_p) if len(mine_p) else 1.0
        busiest, parents_a, parents_b = 0.0, 0, 0
        if len(hits):
            ua, ca = np.unique(rows[hits, 1], return_counts=True)
            ub, cb = np.unique(rows[hits, 2], return_counts=True)
            busiest = max(ca.max(), cb.max()) / len(hits)
            parents_a, parents_b = len(ua), len(ub)
        need = (min(args.min_parents, len(members[a])), min(args.min_parents, len(members[b])))
        best = int(hits[np.argmax(baseline[hits] - with_ring[hits])]) if len(hits) else None
        table.append({
            "a": a, "b": b, "centroid_sim": csim, "tail": int(len(hits)), "mixes": int(len(mine)), "p": p, "p_parents_only": p_parents,
            "busiest": float(busiest), "parents_a": parents_a, "parents_b": parents_b, "best": best,
            "passes": bool(len(hits) and p < cutoff and busiest <= args.max_busiest and parents_a >= need[0] and parents_b >= need[1]),
            "robust": bool(p_parents < cutoff),
        })
    # passing pairs that also pass with only the parents left out come first: their emptiness
    # does not depend on the rings, which could be hiding work that already joins the two
    table.sort(key=lambda t: (not t["passes"], not t["robust"], t["p"], -t["tail"]))
    passing = [t for t in table if t["passes"]]
    print(f"\n{len(passing)} of {len(pairs)} pairs pass (exact p < {cutoff:.2g} after Bonferroni; at least {args.min_parents} different parents a side and none in more than {100 * args.max_busiest:.0f}% of the tail mixtures); "
          f"{sum(t['robust'] for t in passing)} also pass with only the parents left out (PASS), the rest only with the rings left out (pass):")
    for t in table[: len(passing) + 5]:
        mark = ("PASS" if t["robust"] else "pass") if t["passes"] else "    "
        print(f"  {mark} {space.cluster_label(t['a'])} + {space.cluster_label(t['b'])}: tail {t['tail']}/{t['mixes']} ({100 * t['tail'] / max(t['mixes'], 1):.0f}%), p {t['p']:.1e} (parents only {t['p_parents_only']:.1e}), parents {t['parents_a']}/{t['parents_b']}, busiest {100 * t['busiest']:.0f}%")

    method = {
        "method": "connections-v1",
        "band": f"{args.band[0]:g}th to {args.band[1]:g}th percentile of neighborhood similarity ({lo:.3f} to {hi:.3f})",
        "mixesPerPair": m,
        "pairsTested": len(pairs),
        "ringSize": int(ring.shape[1]),
        "simBins": args.sim_bins,
        "densityBins": args.density_bins,
        "tailPct": args.tail,
        "test": "exact: random assignment of mixtures to pairs within matched groups",
        "alpha": args.alpha,
        "pairCutoff": float(f"{cutoff:.3g}"),
        "minParents": args.min_parents,
        "maxBusiest": args.max_busiest,
        "minTargetLonely": args.min_target_lonely,
        "seed": args.seed,
    }
    os.makedirs(args.out_dir, exist_ok=True)
    if args.briefs:
        old = sorted(glob.glob(os.path.join(args.out_dir, "connection-*")))
        if old:
            if not args.replace:
                raise SystemExit(f"{args.out_dir} already holds candidates from an earlier run; pass --replace to move them aside first")
            archive = f"{args.out_dir.rstrip(os.sep)}-previous-{time.strftime('%Y%m%d-%H%M%S')}"
            os.makedirs(archive)
            for f in old + glob.glob(os.path.join(args.out_dir, "pairs.json")):
                os.replace(f, os.path.join(archive, os.path.basename(f)))
            log(f"moved {len(old)} earlier candidate files to {archive}")
    tmp = os.path.join(args.out_dir, "pairs.json.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump({"corpus": space.name, "method": method, "pairs": [{k2: v for k2, v in t.items() if k2 != "best"} for t in table]}, fh, indent=1)
    os.replace(tmp, os.path.join(args.out_dir, "pairs.json"))
    if not args.briefs:
        return
    target_sim = space.sim_for_pct(args.target_pct)
    chosen: list[np.ndarray] = []
    written = 0
    for t in passing:
        if written >= args.top:
            break
        i = t["best"]
        a_item, b_item = int(rows[i, 1]), int(rows[i, 2])
        p0 = unit(X[a_item] + X[b_item])
        if any(float(p0 @ q) > 0.95 for q in chosen):
            continue
        p, info = repel(space, p0, target_sim, max_drift=args.max_drift, tau=args.tau, tether=args.tether, lr=args.lr, steps=args.steps)
        d = describe(space, p0, p, [a_item, b_item], info, args.target_pct, f"connection between neighborhoods #{t['a']} and #{t['b']}")
        if d["gap_lonely_pct"] < args.min_target_lonely:
            log(f"skip #{t['a']} + #{t['b']}: pushed as far as allowed, its spot is still only lonelier than {d['gap_lonely_pct']:.0f}% of the corpus")
            continue
        d["connection"] = {
            "a": t["a"],
            "b": t["b"],
            "centroid_sim": t["centroid_sim"],
            "parent_sim": float(parent_sim[i]),
            "mix_nn1": float(with_ring[i]),
            "mix_nn1_all": float(mix_all[i]),
            "baseline": float(baseline[i]),
            "margin": float(baseline[i] - with_ring[i]),
            "beats": float(beats[i]),
            "pair_tail": 100.0 * t["tail"] / max(t["mixes"], 1),
            "pair_tail_count": t["tail"],
            "pair_p": t["p"],
            "pair_p_parents_only": t["p_parents_only"],
            "parents_a": t["parents_a"],
            "parents_b": t["parents_b"],
            "busiest": t["busiest"],
            "method": method,
        }
        written += 1
        write_out(d, os.path.join(args.out_dir, f"connection-{written:02d}-c{t['a']}-c{t['b']}"), args.kind)
        chosen.append(p0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("blend", "holes", "pairs", "connections"):
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
    pr.add_argument("--band", type=float, nargs=2, default=[75, 97], metavar=("LOW", "HIGH"), help="related band: percentiles of the similarity between every pair of neighborhood centroids")
    pr.add_argument("--min-related", type=float, help="absolute lower bound on centroid similarity, instead of the band's low percentile")
    pr.add_argument("--max-related", type=float, help="absolute upper bound on centroid similarity, instead of the band's high percentile")
    pr.add_argument("--only", nargs="*", type=int, help="consider only pairs where both neighborhoods are in this list")
    pr.add_argument("--briefs", action="store_true", help="also push each midpoint into the gap and write a brief")
    pr.add_argument("--out-dir", default="data/gaps/pairs")
    cn = sub.choices["connections"]
    cn.add_argument("--band", type=float, nargs=2, default=[75, 97], metavar=("LOW", "HIGH"), help="related band: percentiles of the similarity between every pair of neighborhood centroids")
    cn.add_argument("--mixes", type=int, default=200, help="random mixtures per pair of neighborhoods")
    cn.add_argument("--max-mixtures", type=int, default=200000, help="cap on all mixtures; fewer per pair when many pairs are related")
    cn.add_argument("--ring", type=int, default=10, help="leave out each parent's this many nearest neighbors when scoring a mixture")
    cn.add_argument("--sim-bins", type=int, default=10, help="quantile bins of parent similarity for matching")
    cn.add_argument("--density-bins", type=int, default=5, help="quantile bins of the parents' mean nn1 for matching")
    cn.add_argument("--tail", type=float, default=5.0, help="percent of each matched group counted as its emptiest tail")
    cn.add_argument("--alpha", type=float, default=0.05, help="family-wise error rate for the pair test, split over the pairs tested")
    cn.add_argument("--max-busiest", type=float, default=0.4, help="a pair fails when one parent supplies more than this share of its tail mixtures")
    cn.add_argument("--min-parents", type=int, default=5, help="a pair fails unless its tail mixtures draw on this many different parents from each side (or all of a smaller neighborhood)")
    cn.add_argument("--replace", action="store_true", help="move earlier candidates in --out-dir aside before writing new ones")
    cn.add_argument("--top", type=int, default=12, help="most candidates to write")
    cn.add_argument("--min-target-lonely", type=float, default=50.0, help="drop a candidate whose pushed spot is lonelier than less than this percent of the corpus")
    cn.add_argument("--seed", type=int, default=42)
    cn.add_argument("--chunk", type=int, default=1024, help="mixtures scored per block (memory: chunk x corpus size x 4 bytes)")
    cn.add_argument("--briefs", action="store_true", help="also push each passing pair's emptiest mixture into empty space and write a brief")
    cn.add_argument("--out-dir", default="data/gaps/connections")
    args = ap.parse_args()
    {"blend": cmd_blend, "holes": cmd_holes, "pairs": cmd_pairs, "connections": cmd_connections}[args.cmd](args)


if __name__ == "__main__":
    main()
