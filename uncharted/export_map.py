"""Export a mapped corpus and its gaps for the flat-map viewer in viewer/.

    python -I uncharted/export_map.py --name talks
    python -I uncharted/export_map.py --name papers-clean --gaps data/gaps/papers-holes data/gaps/papers-pairs

Reads the files space.py writes (items, meta, clusters, space) and gap JSONs from gaps.py.
Without --gaps, every gap JSON under data/gaps whose corpus matches --name is included.
Writes viewer/data/<name>.js and refreshes viewer/data/index.js, the list of exported corpora.

The data ships as a script that assigns one object, so viewer/index.html opens straight
from disk (file:// pages cannot fetch JSON). Items are stored column by column to keep the
file small: items.x[i], items.title[i] and so on, in the corpus's row order.

Gap points live in the 1,024-d embedding space and have no map position. A hole keeps the
map x/y gaps.py found it at; every other gap is placed at the similarity-weighted mean of
its must-not-copy items' map positions, which is approximate.

Drafts verified with verify.py (<draft>.verify.json under the gaps root) are attached to the
gap they were written for: the gap path verify.py recorded, else the gap file that shares the
draft's name (x.cand2.md -> x.json). Each draft keeps its latest attempt and is placed the same
approximate way, at the weighted mean of its nearest real items. Calibration baselines and
drafts whose gap was not exported are skipped, with the reason logged.
"""

from __future__ import annotations

import argparse
import collections
import datetime
import glob
import json
import os
import re
import time

import numpy as np


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def split_url(url: str) -> tuple[str, str]:
    """Prefix up to the last '/' or '#', so doi.org/10.1145/ and the like are stored once."""
    cut = max(url.rfind("/"), url.rfind("#"))
    if cut < 8:  # nothing past the scheme
        return "", url
    return url[: cut + 1], url[cut + 1 :]


def color_slots(xy: np.ndarray, labels: np.ndarray, n_clusters: int, slots: int = 8, grid: int = 80) -> list[int]:
    """Give each neighborhood one of `slots` colors so neighborhoods that touch on the map differ.

    Two neighborhoods touch when their items share a grid cell or neighboring cells; the
    weight is how many such cells they share. Greedy coloring, most-connected first."""
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    cell = np.floor((xy - lo) / (hi - lo + 1e-9) * (grid - 1)).astype(int)
    occupants: dict[tuple[int, int], set[int]] = collections.defaultdict(set)
    for (cx, cy), c in zip(cell.tolist(), labels.tolist()):
        if c >= 0:
            occupants[(cx, cy)].add(c)
    weight = np.zeros((n_clusters, n_clusters), dtype=np.int64)
    for (cx, cy) in list(occupants):
        near: set[int] = set()
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                near |= occupants.get((cx + dx, cy + dy), set())
        near_list = sorted(near)
        for a in range(len(near_list)):
            for b in range(a + 1, len(near_list)):
                weight[near_list[a], near_list[b]] += 1
                weight[near_list[b], near_list[a]] += 1
    slot = [-1] * n_clusters
    used = [0] * slots
    for c in sorted(range(n_clusters), key=lambda c: -int(weight[c].sum())):
        conflict = [0] * slots
        for other in range(n_clusters):
            if slot[other] >= 0 and weight[c, other]:
                conflict[slot[other]] += int(weight[c, other])
        slot[c] = min(range(slots), key=lambda s: (conflict[s], used[s], s))
        used[slot[c]] += 1
    return slot


def gap_files(paths: list[str] | None, corpus: str, root: str) -> list[str]:
    if not paths:
        candidates = glob.glob(os.path.join(root, "**", "*.json"), recursive=True)
    else:
        candidates = []
        for p in paths:
            candidates += glob.glob(os.path.join(p, "*.json")) if os.path.isdir(p) else [p]
    out = []
    for f in sorted(set(candidates)):
        if f.endswith(".verify.json"):
            continue
        try:
            with open(f, encoding="utf-8") as fh:
                d = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(d, dict) or "point" not in d or "must_not_copy" not in d:
            continue
        if d.get("corpus") != corpus:
            if paths:
                log(f"skip {f}: gap is for corpus {d.get('corpus')!r}, not {corpus!r}")
            continue
        out.append(f)
    return out


def attach_drafts(gap_by_path: dict[str, dict], corpus: str, root: str, index_of: dict[str, int], xy: np.ndarray) -> collections.Counter:
    """Add a "drafts" list to each gap in gap_by_path; return counts of what was skipped and why."""
    skipped: collections.Counter = collections.Counter()
    for vf in sorted(glob.glob(os.path.join(root, "**", "*.verify.json"), recursive=True)):
        try:
            with open(vf, encoding="utf-8") as fh:
                attempts = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(attempts, list) or not attempts or attempts[-1].get("corpus") != corpus:
            continue
        a = attempts[-1]
        cand = a.get("candidate") or vf[: -len(".verify.json")]
        base = os.path.basename(cand)
        stem = base.split(".cand")[0] if ".cand" in base else os.path.splitext(base)[0]
        if stem.startswith("baseline"):
            skipped["calibration baseline"] += 1
            continue
        gap_path = a.get("gap") or os.path.join(os.path.dirname(cand), stem + ".json")
        g = gap_by_path.get(os.path.normpath(gap_path))
        if g is None:
            skipped[f"no exported gap for {stem}"] += 1
            continue
        nearest = [(index_of[m["id"]], float(m["sim"])) for m in a.get("nearest", []) if m.get("id") in index_of]
        if not nearest:
            skipped[f"nearest items of {base} not in {corpus}"] += 1
            continue
        w = np.array([s for _, s in nearest])
        x, y = (xy[[i for i, _ in nearest]] * w[:, None]).sum(axis=0) / w.sum()
        m = re.search(r"\.cand(\d+)", base)
        num = int(m.group(1)) if m else len(g["drafts"]) + 1
        g["drafts"].append(
            {
                "code": f"d{num}",
                "n": num,
                "title": a.get("title") or stem,
                "file": cand,
                "verdict": a.get("verdict"),
                "nn1": round(a["nn1"], 3),
                "lonely": round(a["lonely_pct"], 1),
                "toGap": round(a["sim_to_gap_point"], 3) if a.get("sim_to_gap_point") is not None else None,
                "reasons": a.get("reasons", []),
                "x": round(float(x), 3),
                "y": round(float(y), 3),
                "nearest": [[i, round(s, 3)] for i, s in nearest],
            }
        )
    for g in gap_by_path.values():
        g["drafts"].sort(key=lambda d: d["n"])
    return skipped


def brief_headline(json_path: str, d: dict) -> str:
    """First line of the gap's brief, without the '# Gap brief — <corpus> (...)' wrapper."""
    md = json_path[: -len(".json")] + ".md"
    line = ""
    if os.path.exists(md):
        with open(md, encoding="utf-8") as fh:
            line = fh.readline().strip().lstrip("#").strip()
    m = re.match(rf"Gap brief — {re.escape(d['corpus'])} \((.*)\)$", line)
    if m:
        return m.group(1)
    return line or d.get("how", "")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--dir", default="data/embeddings")
    ap.add_argument("--gaps", nargs="*", help="gap directories or JSON files (default: every gap for this corpus under --gaps-root)")
    ap.add_argument("--gaps-root", default="data/gaps")
    ap.add_argument("--out-dir", default="viewer/data")
    args = ap.parse_args()

    base = os.path.join(args.dir, args.name)
    items = read_jsonl(base + ".items.jsonl")
    meta = read_jsonl(base + ".meta.jsonl")
    with open(base + ".clusters.json", encoding="utf-8") as fh:
        clusters = json.load(fh)
    with open(base + ".space.json", encoding="utf-8") as fh:
        stats = json.load(fh)
    assert len(items) == len(meta), "items and meta disagree on row count"
    n = len(items)
    log(f"{args.name}: {n} items, {len(clusters)} neighborhoods")

    xy = np.array([(it["x"], it["y"]) for it in items], dtype=np.float64)
    labels = np.array([it["cluster"] for it in items], dtype=np.int64)
    index_of = {m["id"]: i for i, m in enumerate(meta)}

    venues: list[str] = []
    venue_index: dict[str, int] = {}
    prefixes: list[str] = []
    prefix_index: dict[str, int] = {}
    prefix_count = collections.Counter(split_url(m.get("url") or "")[0] for m in meta)
    cols: dict[str, list] = {k: [] for k in ("x", "y", "cluster", "title", "year", "venue", "lonely", "nn1", "urlPrefix", "url")}
    for it, m in zip(items, meta):
        cols["x"].append(round(it["x"], 3))
        cols["y"].append(round(it["y"], 3))
        cols["cluster"].append(it["cluster"])
        cols["title"].append(m["title"])
        cols["year"].append(m.get("year"))
        v = m.get("venue")
        if v and v not in venue_index:
            venue_index[v] = len(venues)
            venues.append(v)
        cols["venue"].append(venue_index[v] if v else -1)
        cols["lonely"].append(round(it["lonely_pct"], 1))
        cols["nn1"].append(round(it["nn1"], 3))
        prefix, rest = split_url(m.get("url") or "")
        if prefix and prefix_count[prefix] >= 5:
            if prefix not in prefix_index:
                prefix_index[prefix] = len(prefixes)
                prefixes.append(prefix)
            cols["urlPrefix"].append(prefix_index[prefix])
            cols["url"].append(rest)
        else:
            cols["urlPrefix"].append(-1)
            cols["url"].append(m.get("url") or "")

    slots = color_slots(xy, labels, len(clusters))
    neighborhoods = []
    for c in clusters:
        cid = c["cluster"]
        members = np.where(labels == cid)[0]
        by_title = {meta[i]["title"]: int(i) for i in members}
        neighborhoods.append(
            {
                "id": cid,
                "terms": c["label_terms"][:6],
                "tags": c["label_tags"][:6],
                "size": c["size"],
                "x": round(c["x"], 3),
                "y": round(c["y"], 3),
                "density": round(c["density_median_nn1"], 3),
                "popularity": c.get("popularity", 0),
                "slot": slots[cid],
                "examples": [by_title[t] for t in c.get("examples", []) if t in by_title],
            }
        )

    gaps = []
    gap_by_path: dict[str, dict] = {}
    for f in gap_files(args.gaps, args.name, args.gaps_root):
        with open(f, encoding="utf-8") as fh:
            d = json.load(fh)
        nearest = [(index_of[m["id"]], float(m["sim"])) for m in d["must_not_copy"] if m["id"] in index_of]
        parents = [index_of[p["id"]] for p in d.get("parents", []) if p["id"] in index_of]
        if "hole" in d:
            kind, x, y = "hole", d["hole"]["x"], d["hole"]["y"]
        else:
            if not nearest:
                log(f"skip {f}: none of its must-not-copy items are in {args.name}")
                continue
            kind = "pair" if "pair" in d else "blend"
            w = np.array([s for _, s in nearest])
            pts = xy[[i for i, _ in nearest]]
            x, y = (pts * w[:, None]).sum(axis=0) / w.sum()
        rel = os.path.relpath(f, args.gaps_root) if os.path.abspath(f).startswith(os.path.abspath(args.gaps_root)) else f
        gaps.append(
            {
                "name": rel[: -len(".json")],
                "kind": kind,
                "headline": brief_headline(f, d),
                "brief": f[: -len(".json")] + ".md",
                "x": round(float(x), 3),
                "y": round(float(y), 3),
                "radius": round(d["hole"]["empty_radius"], 3) if kind == "hole" else None,
                "nn1": round(d["gap_nn1"], 3),
                "lonely": round(d["gap_lonely_pct"], 1),
                "targetPct": d.get("target_pct"),
                "stop": d.get("stop"),
                "neighborhoods": [c["cluster"] for c in d.get("nearest_clusters", [])],
                "tags": d.get("leaned_tags", [])[:8],
                "nearest": [[i, round(s, 3)] for i, s in nearest],
                "parents": parents,
                "drafts": [],
            }
        )
        gap_by_path[os.path.normpath(f)] = gaps[-1]
    skipped = attach_drafts(gap_by_path, args.name, args.gaps_root, index_of, xy)
    n_drafts = sum(len(g["drafts"]) for g in gaps)
    log(f"{n_drafts} drafts attached to {sum(1 for g in gaps if g['drafts'])} gaps" + (f"; skipped: {dict(skipped)}" if skipped else ""))
    order = {"hole": 0, "pair": 1, "blend": 2}
    gaps.sort(key=lambda g: (order[g["kind"]], g["name"]))
    counters: collections.Counter = collections.Counter()
    for g in gaps:
        counters[g["kind"]] += 1
        g["code"] = f"{g['kind'][0].upper()}{counters[g['kind']]}"

    data = {
        "format": 1,
        "corpus": args.name,
        "exported": datetime.date.today().isoformat(),
        "stats": {
            "items": n,
            "neighborhoods": len(clusters),
            "unclustered": int((labels < 0).sum()),
            "nn1Percentiles": {k: round(v, 3) for k, v in stats.get("nn1_percentiles", {}).items()},
        },
        "venues": venues,
        "urlPrefixes": prefixes,
        "items": cols,
        "neighborhoods": neighborhoods,
        "gaps": gaps,
    }
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"{args.name}.js")
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("window.UNCHARTED_DATA = window.UNCHARTED_DATA || {};\n")
        fh.write(f"window.UNCHARTED_DATA[{json.dumps(args.name)}] = {payload};\n")
    log(f"wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB): {n} items, {len(neighborhoods)} neighborhoods, {len(gaps)} gaps ({dict(counters)})")

    # index.js lists every exported corpus, so the viewer can offer a switcher
    entries = []
    for path in sorted(glob.glob(os.path.join(args.out_dir, "*.js"))):
        name = os.path.basename(path)[: -len(".js")]
        if name == "index":
            continue
        entries.append({"name": name, "bytes": os.path.getsize(path)})
    with open(os.path.join(args.out_dir, "index.js"), "w", encoding="utf-8") as fh:
        fh.write(f"window.UNCHARTED_INDEX = {json.dumps(entries)};\n")
    log(f"index lists {len(entries)} corpora: {', '.join(e['name'] for e in entries)}")


if __name__ == "__main__":
    main()
