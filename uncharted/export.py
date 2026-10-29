"""Export a mapped corpus and its candidates as an Uncharted map file (version 1).

    python -I uncharted/export.py --name talks-q --id talks --title "Accessibility conference talks" \
        --corpus-info data/corpora/talks.corpus.json \
        --candidates data/gaps/talks-q-pairs data/gaps/talks-q-blends \
        --out data/exports/talks.json

Reads what embed.py and space.py wrote for --name, the candidate JSONs gaps.py wrote under
--candidates (directories or files) and the drafts verify.py checked against them. Writes one
JSON document in the format schema/README.md describes, validated by schema.py before it is
written; an --out ending in .gz is gzip-compressed. Briefs and draft text go into the file and
no local path does, so the file can be shared as it is (--no-briefs and --no-drafts leave them
out; --snippet-words adds the first words of every item's text; --labels replaces the
generated candidate labels with names a person or a model wrote, keyed by candidate id).

--corpus-info names a JSON file holding the manifest's corpus section: what one item is, the
sources with the exact query and retrieval date, the cleaning rules, and what popularity and
tags mean. A source may carry "idPrefix", the start of its items' ids, so the file can say
which source each item came from; the key itself is not exported.

A draft is exported only when its text and the corpus are what verify.py measured: a draft
edited since, or a corpus re-embedded since, is left out until verify.py runs again.

Candidates and drafts have no true position on the flat map; each is placed at the
similarity-weighted mean of its 8 nearest real items. The 2-D layout keeps neighbors only
roughly together, so that position means something only when those items sit as tightly on
the map as a real item's own 8 nearest neighbors usually do: within the --place-pct
percentile of real items' spread. Otherwise x and y are null and it shows only in lists.
"""

from __future__ import annotations

import argparse
import collections
import datetime
import glob
import gzip
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import tomllib
import urllib.parse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # `python -I` leaves the script's directory off sys.path
import schema  # noqa: E402
from drafts import embedded_text, read_candidate  # noqa: E402
from provenance import corpus_fingerprint, text_sha256  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KIND_ORDER = {"connection": 0, "pair": 1, "blend": 2}
PLACE_NEIGHBORS = 8
URL_SAFE = "!#$%&'()*+,-./:;=?@[]_~"


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def clip(v: float, lo: float, hi: float, digits: int) -> float:
    return float(min(max(round(float(v), digits), lo), hi))


def sig3(v: float) -> float:
    """Three significant figures, for probabilities too small for fixed decimals."""
    return float(f"{float(v):.3g}")


def text(value, limit: int) -> str:
    """One line of display text: whitespace collapsed, cut to `limit` characters."""
    return " ".join(str(value or "").split())[:limit]


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9._-]+", "-", name.lower()).strip("-._")
    return slug[:72] or "candidate"


def unique(base: str, taken: set[str]) -> str:
    """`base`, or `base-2`, `base-3` … whichever is free; the result is added to `taken`."""
    name, n = base, 1
    while name in taken:
        n += 1
        name = f"{base}-{n}"
    taken.add(name)
    return name


def to_timestamp(value) -> str | None:
    """An RFC 3339 timestamp with an offset. A time recorded without one is read as this
    machine's local time on that date, which is where verify.py ran."""
    if not value:
        return None
    try:
        moment = datetime.datetime.fromisoformat(str(value).replace(" ", "T").replace("Z", "+00:00"))
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.astimezone()
    return moment.isoformat(timespec="seconds").replace("+00:00", "Z")


def generator_info() -> dict:
    with open(os.path.join(ROOT, "pyproject.toml"), "rb") as fh:
        info = {"name": "uncharted", "version": tomllib.load(fh)["project"]["version"]}
    try:
        commit = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short=12", "HEAD"], capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = ""
    if re.fullmatch(r"[0-9a-f]{7,40}", commit):
        info["commit"] = commit
    return info


def embedding_info(info: dict) -> dict:
    """The manifest's embedding section, from what embed.py recorded in <name>.model.json.
    Nothing is inferred from today's model cache, which may have changed since."""
    out = {"provider": info.get("provider", "local"), "model": info["model"], "dimensions": int(info["dim"])}
    if info.get("revision"):
        out["revision"] = str(info["revision"])
    if info.get("max_seq_length"):
        out["maxTokens"] = int(info["max_seq_length"])
    if info.get("pooling"):
        out["pooling"] = str(info["pooling"])
    if info.get("normalized") is not None:
        out["normalized"] = bool(info["normalized"])
    if "instruction" in info:
        out["instruction"] = info["instruction"]
    precision = info.get("precision") or ("float16" if info.get("fp16") else None)
    if precision:
        out["precision"] = precision
    if info.get("input"):
        out["input"] = str(info["input"])
    if info.get("truncated") is not None:
        out["truncated"] = int(info["truncated"])
    return out


def clean_link(url: str) -> str | None:
    """A link as the map file stores it (http or https, a host, no credentials, ASCII), or None."""
    url = (url or "").strip()
    if ":" not in url:
        return None
    scheme, rest = url.split(":", 1)
    url = urllib.parse.quote(scheme.lower() + ":" + rest, safe=URL_SAFE)
    try:
        parts = urllib.parse.urlsplit(url)
        host = parts.hostname
    except ValueError:
        return None
    if parts.scheme not in ("http", "https") or not host or parts.username is not None or parts.password is not None:
        return None
    return url if re.fullmatch(r"[!-~]+", url) and len(url) <= 2000 else None


def link_columns(urls: list[str]) -> tuple[list[str], list[int], list[str], int]:
    """Store each link as a shared prefix (up to its last / or #) plus the rest, so common
    prefixes are written once. Rare prefixes fall back to the bare scheme. Links that are not
    usable http or https links are dropped; the count of dropped links is returned last."""
    cleaned = [clean_link(u) for u in urls]
    parts: list[tuple[str, str] | None] = []
    counts: collections.Counter = collections.Counter()
    for u in cleaned:
        if u is None:
            parts.append(None)
            continue
        scheme = u[: u.index("://") + 3]
        cut = max(u.rfind("/"), u.rfind("#"))
        prefix = u[: cut + 1] if cut >= len(scheme) else scheme
        parts.append((prefix, scheme))
        counts[prefix] += 1
    prefixes: list[str] = []
    index: dict[str, int] = {}
    col_prefix: list[int] = []
    col_rest: list[str] = []
    for u, part in zip(cleaned, parts):
        if part is None:
            col_prefix.append(-1)
            col_rest.append("")
            continue
        prefix, scheme = part
        if counts[prefix] < 5 or len(prefix) > 500:
            prefix = scheme
        if prefix not in index:
            index[prefix] = len(prefixes)
            prefixes.append(prefix)
        col_prefix.append(index[prefix])
        col_rest.append(u[len(prefix) :])
    dropped = sum(1 for u, c in zip(urls, cleaned) if (u or "").strip() and c is None)
    return prefixes, col_prefix, col_rest, dropped


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
    for cx, cy in list(occupants):
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
        for other in np.nonzero(weight[c])[0]:
            if slot[other] >= 0:
                conflict[slot[other]] += int(weight[c, other])
        slot[c] = min(range(slots), key=lambda s: (conflict[s], used[s], s))
        used[slot[c]] += 1
    return slot


def spread_of(xy: np.ndarray, idx, sims) -> tuple[np.ndarray, float]:
    """Similarity-weighted center of some items on the map, and their RMS distance from it.
    Weights are similarities floored at 0, plus a little, so negative similarities never pull."""
    pts = xy[list(idx)]
    w = np.maximum(np.asarray(sims, dtype=np.float64), 0.0) + 1e-9
    center = (pts * w[:, None]).sum(axis=0) / w.sum()
    return center, float(np.sqrt((w * ((pts - center) ** 2).sum(axis=1)).sum() / w.sum()))


class Placer:
    """Approximate flat-map positions for points that exist only in the full embedding space."""

    def __init__(self, xy: np.ndarray, nn_idx: np.ndarray, nn_sims: np.ndarray, place_pct: float):
        self.xy = xy
        k = min(PLACE_NEIGHBORS, nn_idx.shape[1])
        real = [spread_of(xy, nn_idx[i, :k], nn_sims[i, :k])[1] for i in range(len(xy))]
        self.limit = float(np.percentile(real, place_pct))

    def place(self, nearest: list[tuple[int, float]]) -> tuple[float | None, float | None, float]:
        support = nearest[:PLACE_NEIGHBORS]
        center, spread = spread_of(self.xy, [i for i, _ in support], [s for _, s in support])
        if spread > self.limit:
            return None, None, spread
        return clip(center[0], 0, 1, 4), clip(center[1], 0, 1, 4), spread


def candidate_files(paths: list[str]) -> list[str]:
    found: list[str] = []
    for p in paths:
        found += glob.glob(os.path.join(p, "**", "*.json"), recursive=True) if os.path.isdir(p) else [p]
    out = []
    for f in sorted({os.path.realpath(f) for f in found}):
        if f.endswith(".verify.json"):
            continue
        try:
            with open(f, encoding="utf-8") as fh:
                d = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(d, dict) and "point" in d and "must_not_copy" in d:
            out.append(f)
    return out


def verify_files(paths: list[str]) -> list[str]:
    found: set[str] = set()
    for p in paths:
        root = p if os.path.isdir(p) else os.path.dirname(p) or "."
        found.update(os.path.realpath(f) for f in glob.glob(os.path.join(root, "**", "*.verify.json"), recursive=True))
    return sorted(found)


def hood_name(cluster: dict) -> str:
    return ", ".join(cluster["label_terms"][:2]) or ", ".join(cluster["label_tags"][:2]) or f"neighborhood {cluster['cluster']}"


def short_title(title: str, words: int = 5) -> str:
    parts = title.split()
    return " ".join(parts[:words]) + ("…" if len(parts) > words else "")


def snippet_of(m: dict, words: int) -> str:
    body = m["text"][len(m["title"]) + 2 :] if m["text"].startswith(m["title"]) else m["text"]
    cut = body.split()
    out = " ".join(cut[:words]) + ("…" if len(cut) > words else "")
    return out[:400]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="embedding basename under --dir")
    ap.add_argument("--dir", default="data/embeddings")
    ap.add_argument("--id", help="map id: lowercase letters, digits, dots and hyphens (default: --name)")
    ap.add_argument("--title", required=True)
    ap.add_argument("--description")
    ap.add_argument("--corpus-info", required=True, help="JSON file with the manifest's corpus section")
    ap.add_argument("--candidates", nargs="*", default=[], help="gap JSONs from gaps.py, or directories of them")
    ap.add_argument("--place-pct", type=float, default=90.0, help="draw a candidate only when its nearest items sit at least as tightly on the map as this share of real items' nearest neighbors")
    ap.add_argument("--snippet-words", type=int, default=0, help="add each item's first N words of text (0: leave them out)")
    ap.add_argument("--labels", help="JSON file mapping candidate ids to labels that replace the generated ones")
    ap.add_argument("--no-briefs", action="store_true")
    ap.add_argument("--no-drafts", action="store_true")
    ap.add_argument("--out", required=True, help="output file; .gz compresses it")
    args = ap.parse_args()

    base = os.path.join(args.dir, args.name)
    rows = read_jsonl(base + ".items.jsonl")
    meta = read_jsonl(base + ".meta.jsonl")
    with open(base + ".clusters.json", encoding="utf-8") as fh:
        clusters = json.load(fh)
    with open(base + ".space.json", encoding="utf-8") as fh:
        space = json.load(fh)
    with open(base + ".model.json", encoding="utf-8") as fh:
        model_info = json.load(fh)
    with open(args.corpus_info, encoding="utf-8") as fh:
        corpus_info = json.load(fh)
    nn = np.load(base + ".nn.npz")
    n, k = len(rows), len(clusters)
    if len(meta) != n or any(r["id"] != m["id"] for r, m in zip(rows, meta)):
        raise SystemExit(f"{base}.items.jsonl and .meta.jsonl do not list the same items in the same order; rerun space.py")
    if nn["idx"].shape[0] != n or int(nn["idx"].max()) >= n:
        raise SystemExit(f"{base}.nn.npz does not match the corpus; rerun space.py")
    if [c["cluster"] for c in clusters] != list(range(k)):
        raise SystemExit("clusters.json is not numbered 0..k-1; rerun space.py")
    fingerprint = corpus_fingerprint(meta, model_info)
    log(f"{args.name}: {n} items, {k} neighborhoods, fingerprint {fingerprint}")

    raw_xy = np.array([(r["x"], r["y"]) for r in rows], dtype=np.float64)
    lo = raw_xy.min(axis=0)
    span = float((raw_xy.max(axis=0) - lo).max()) or 1.0
    xy = (raw_xy - lo) / span  # longer side spans 0 to 1, aspect ratio kept, y up as in UMAP
    labels = np.array([r["cluster"] for r in rows], dtype=np.int64)
    members: dict[int, list[int]] = collections.defaultdict(list)
    for i, c in enumerate(labels.tolist()):
        members[c].append(i)
    index_of = {m["id"]: i for i, m in enumerate(meta)}
    placer = Placer(xy, nn["idx"], nn["sims"], args.place_pct)

    venues: list[str] = []
    venue_index: dict[str, int] = {}
    venue_col = []
    for m in meta:
        v = text(m.get("venue"), 300)
        if v and v not in venue_index:
            venue_index[v] = len(venues)
            venues.append(v)
        venue_col.append(venue_index[v] if v else -1)
    prefixes, prefix_col, url_col, dropped_links = link_columns([m.get("url") or "" for m in meta])

    def year_of(m: dict) -> int | None:
        try:
            y = int(m.get("year"))
        except (TypeError, ValueError):
            return None
        return y if 1000 <= y <= 2999 else None

    items = {
        "count": n,
        "id": [str(m["id"])[:200] for m in meta],
        "title": [text(m.get("title"), 1000) or "(untitled)" for m in meta],
        "x": [clip(v, 0, 1, 4) for v in xy[:, 0]],
        "y": [clip(v, 0, 1, 4) for v in xy[:, 1]],
        "cluster": labels.tolist(),
        "nn1": [clip(r["nn1"], -1, 1, 3) for r in rows],
        "lonely": [clip(r["lonely_pct"], 0, 100, 1) for r in rows],
        "year": [year_of(m) for m in meta],
        "venue": venue_col,
        "urlPrefix": prefix_col,
        "url": url_col,
    }
    if args.snippet_words > 0:
        items["snippet"] = [snippet_of(m, args.snippet_words) for m in meta]
    sources = [dict(s) for s in corpus_info["sources"]]
    id_prefixes = [s.pop("idPrefix", None) for s in sources]
    if len(sources) == 1:
        items["source"] = [0] * n
    elif any(id_prefixes):
        items["source"] = [next((j for j, p in enumerate(id_prefixes) if p and str(m["id"]).startswith(p)), -1) for m in meta]
    corpus_section = {**corpus_info, "sources": sources, "fingerprint": fingerprint}

    slots = color_slots(xy, labels, k)
    neighborhoods = []
    for c in clusters:
        cid = c["cluster"]
        mine = members.get(cid, [])
        by_title = {meta[i]["title"]: i for i in mine}
        neighborhoods.append(
            {
                "id": cid,
                "terms": [text(t, 80) for t in c["label_terms"][:6] if text(t, 80)],
                "tags": [text(t, 200) for t in c["label_tags"][:6] if text(t, 200)],
                "size": len(mine),
                "x": clip(xy[mine, 0].mean(), 0, 1, 4),
                "y": clip(xy[mine, 1].mean(), 0, 1, 4),
                "density": clip(c["density_median_nn1"], -1, 1, 3),
                "popularity": max(int(c.get("popularity") or 0), 0),
                "slot": slots[cid],
                "examples": [by_title[t] for t in c.get("examples", []) if t in by_title][:20],
            }
        )

    skipped: collections.Counter = collections.Counter()
    candidates: list[dict] = []
    by_gap: dict[str, dict] = {}
    taken_ids: set[str] = set()
    pair_bands: set[tuple[float, float]] = set()
    methods: list[dict] = []
    for f in candidate_files(args.candidates):
        with open(f, encoding="utf-8") as fh:
            d = json.load(fh)
        if d.get("corpus") != args.name:
            skipped[f"made for corpus {d.get('corpus')!r}"] += 1
            continue
        kind = "connection" if "connection" in d else "pair" if "pair" in d else "hole" if "hole" in d else "blend"
        if kind == "hole":
            skipped["2-D hole (version 1 does not export holes)"] += 1
            continue
        nearest = [(index_of[m["id"]], float(m["sim"])) for m in d["must_not_copy"] if m["id"] in index_of]
        if not nearest:
            skipped["no nearest items in this corpus"] += 1
            continue
        parents = [index_of[p["id"]] for p in d.get("parents", []) if p["id"] in index_of]
        if kind in ("pair", "connection"):
            ends = d[kind]
            hoods = [int(ends["a"]), int(ends["b"])]
            label = f"{hood_name(clusters[hoods[0]])} × {hood_name(clusters[hoods[1]])}"
            headline = "Between " + " and ".join(f"neighborhood {h} ({', '.join(clusters[h]['label_terms'][:4]) or 'unlabeled'})" for h in hoods)
            if "min_related" in ends:
                pair_bands.add((ends["min_related"], ends["max_related"]))
        else:
            hoods = list(dict.fromkeys(int(c["cluster"]) for c in d.get("nearest_clusters", [])))[:3]
            label = " × ".join(short_title(meta[i]["title"]) for i in parents[:3]) or "blend"
            how = d.get("how") or "blend"
            headline = how[0].upper() + how[1:]
        x, y, spread = placer.place(nearest)
        score = {"nn1": clip(d["gap_nn1"], -1, 1, 3), "lonely": clip(d["gap_lonely_pct"], 0, 100, 1)}
        if kind == "connection":
            ev = d["connection"]
            score.update(
                mixNn1=clip(ev["mix_nn1"], -1, 1, 3),
                baseline=clip(ev["baseline"], -1, 1, 3),
                margin=clip(ev["margin"], -2, 2, 3),
                beats=clip(ev["beats"], 0, 100, 1),
                pairTail=clip(ev["pair_tail"], 0, 100, 1),
                pairP=sig3(ev["pair_p"]),
            )
            if ev.get("pair_p_parents_only") is not None:
                score["pairPParentsOnly"] = sig3(ev["pair_p_parents_only"])
            if ev.get("parents_a") is not None:
                score["parentsA"], score["parentsB"] = int(ev["parents_a"]), int(ev["parents_b"])
            if ev.get("mix_nn1_all") is not None:
                score["mixNn1All"] = clip(ev["mix_nn1_all"], -1, 1, 3)
            if isinstance(ev.get("method"), dict):
                methods.append(ev["method"])
        cand = {
            "id": unique(slugify(os.path.splitext(os.path.basename(f))[0]), taken_ids),
            "code": "",
            "kind": kind,
            "label": text(label, 160),
            "headline": text(headline, 400),
            "x": x,
            "y": y,
            "spread": clip(spread, 0, 2, 4),
            "score": score,
            "neighborhoods": hoods,
            "parents": parents[:50],
            "nearest": [[i, clip(s, -1, 1, 3)] for i, s in nearest][:50],
            "tags": [text(t, 200) for t in d.get("leaned_tags", [])[:8] if text(t, 200)],
            "drafts": [],
        }
        if d.get("target_pct") is not None:
            cand["targetPct"] = clip(d["target_pct"], 0, 100, 1)
        if d.get("stop"):
            cand["stop"] = text(d["stop"], 80)
        brief_path = f[: -len(".json")] + ".md"
        if not args.no_briefs and os.path.exists(brief_path):
            with open(brief_path, encoding="utf-8") as fh:
                cand["brief"] = fh.read()[:50000]
        candidates.append(cand)
        by_gap[f] = cand

    if not args.no_drafts:
        found: dict[str, list[tuple[int | None, dict]]] = collections.defaultdict(list)
        for vf in verify_files(args.candidates):
            try:
                with open(vf, encoding="utf-8") as fh:
                    attempts = json.load(fh)
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(attempts, list) or not attempts or attempts[-1].get("corpus") != args.name:
                continue
            a = attempts[-1]  # the latest check of this draft
            md = os.path.realpath(a.get("candidate") or vf[: -len(".verify.json")] + ".md")
            stem = os.path.splitext(os.path.basename(md))[0]
            number = re.search(r"\.cand(\d+)$", stem)
            stem = re.sub(r"\.cand\d+$", "", stem)
            if stem.startswith("baseline"):
                skipped["calibration baseline"] += 1
                continue
            cand = by_gap.get(os.path.realpath(a["gap"]) if a.get("gap") else os.path.join(os.path.dirname(md), stem + ".json"))
            if cand is None:
                skipped[f"draft for {stem}, which is not exported"] += 1
                continue
            try:
                title, body, notes = read_candidate(md)
            except (OSError, ValueError):
                skipped["draft file missing or without a title"] += 1
                continue
            if a.get("text_sha256") != text_sha256(embedded_text(title, body)):
                skipped["draft edited since verify.py checked it (or checked before text hashes)"] += 1
                continue
            if a.get("corpus_fingerprint") != fingerprint:
                skipped["corpus changed since verify.py checked the draft"] += 1
                continue
            nearest = [(index_of[m["id"]], float(m["sim"])) for m in a.get("nearest", []) if m.get("id") in index_of]
            if not nearest:
                skipped["draft whose nearest items are not in this corpus"] += 1
                continue
            x, y, _ = placer.place(nearest)
            draft = {
                "title": text(title, 500),
                "text": body[:50000] or text(title, 500),
                "verdict": "pass" if a.get("verdict") == "PASS" else "retry",
                "nn1": clip(a["nn1"], -1, 1, 3),
                "lonely": clip(a["lonely_pct"], 0, 100, 1),
                "reasons": [text(r, 500) for r in a.get("reasons", []) if text(r, 500)][:20],
                "x": x,
                "y": y,
                "nearest": [[i, clip(s, -1, 1, 3)] for i, s in nearest][:50],
            }
            if draft["verdict"] == "retry" and not draft["reasons"]:
                draft["reasons"] = ["verify.py gave no reason"]
            if draft["verdict"] == "pass":
                draft["reasons"] = []
            if notes:
                draft["notes"] = notes[:5000]
            if a.get("sim_to_gap_point") is not None:
                draft["toCandidate"] = clip(a["sim_to_gap_point"], -1, 1, 3)
            checked = to_timestamp(a.get("checked_at"))
            if checked:
                draft["checked"] = checked
            found[cand["id"]].append((int(number.group(1)) if number else None, draft))
        for cand in candidates:
            numbered = found.get(cand["id"], [])
            used = {num for num, _ in numbered if num is not None}
            next_free = max(used, default=0) + 1
            for num, draft in sorted(numbered, key=lambda t: (t[0] is None, t[0] or 0)):
                if num is None or num in {int(dr["code"][1:]) for dr in cand["drafts"]}:
                    num, next_free = next_free, next_free + 1
                cand["drafts"].append({"id": f"{cand['id']}.d{num}", "code": f"d{num}", **draft})
            cand["drafts"] = cand["drafts"][:20]

    if args.labels:
        with open(args.labels, encoding="utf-8") as fh:
            names = json.load(fh)
        unknown = sorted(set(names) - {c["id"] for c in candidates})
        if unknown:
            log(f"labels for candidates not in this map, ignored: {', '.join(unknown[:5])}")
        for cand in candidates:
            if text(names.get(cand["id"]), 160):
                cand["label"] = text(names[cand["id"]], 160)
    # connections by their evidence: those that also pass with only the parents left out
    # first, then by p; the other kinds by how lonely their spot is
    cutoff = next((m.get("pairCutoff") for m in methods if m.get("pairCutoff") is not None), 0.0)
    candidates.sort(key=lambda c: (KIND_ORDER[c["kind"]], c["kind"] == "connection" and c["score"].get("pairPParentsOnly", 0.0) >= cutoff, c["score"].get("pairP", 1.0), -c["score"]["lonely"], c["id"]))
    ranks: collections.Counter = collections.Counter()
    for cand in candidates:
        ranks[cand["kind"]] += 1
        cand["code"] = f"{cand['kind'][0].upper()}{ranks[cand['kind']]}"

    kinds = collections.Counter(c["kind"] for c in candidates)
    cand_params: dict = {"connections": kinds["connection"], "pairs": kinds["pair"], "blends": kinds["blend"], "placePct": args.place_pct, "placeNeighbors": PLACE_NEIGHBORS, "placeSpread": round(placer.limit, 4)}
    targets = sorted({c["targetPct"] for c in candidates if "targetPct" in c})
    if len(targets) == 1:
        cand_params["targetPct"] = targets[0]
    if len(pair_bands) == 1:
        low, high = next(iter(pair_bands))
        cand_params["pairBand"] = f"centroid similarity {low:.3f} to {high:.3f}"
    if methods:
        if any(m != methods[0] for m in methods):
            log("connections come from runs with different settings; the manifest records the first run's")
        cand_params.update({key: value for key, value in methods[0].items() if isinstance(value, (str, int, float, bool)) or value is None})
    map_params = {key: value for key, value in (space.get("params") or {"minClusterSize": space.get("min_cluster_size")}).items() if value is not None}
    n_drafts = sum(len(c["drafts"]) for c in candidates)
    doc = {
        "format": schema.FORMAT,
        "version": schema.VERSION,
        "id": args.id or slugify(args.name),
        "title": text(args.title, 120),
        "exported": datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "manifest": {
            "generator": generator_info(),
            "corpus": corpus_section,
            "embedding": embedding_info(model_info),
            "map": map_params,
            "candidates": cand_params,
        },
        "stats": {
            "items": n,
            "neighborhoods": k,
            "unclustered": int((labels < 0).sum()),
            "candidates": len(candidates),
            "drafts": n_drafts,
            "nn1Percentiles": {p: clip(space["nn1_percentiles"][p], -1, 1, 3) for p in schema.PERCENTILES},
        },
        "lookups": {"venues": venues, "urlPrefixes": prefixes},
        "items": items,
        "neighborhoods": neighborhoods,
        "candidates": candidates,
    }
    if args.description:
        doc["description"] = text(args.description, 2000)

    errors = schema.validate(doc)
    if errors:
        for e in errors[:20]:
            print("  " + e)
        shown = f"the first {min(20, len(errors))} shown above" if len(errors) > 20 else "shown above"
        raise SystemExit(f"not written: the map fails validation ({len(errors)} problems, {shown})")
    payload = json.dumps(doc, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    handle, tmp = tempfile.mkstemp(dir=out_dir, prefix=os.path.basename(args.out) + ".", suffix=".part")
    try:
        with os.fdopen(handle, "wb") as fh:
            if args.out.endswith(".gz"):
                with gzip.GzipFile(fileobj=fh, mode="wb", compresslevel=9, mtime=0, filename="") as gz:
                    gz.write(payload)
            else:
                fh.write(payload)
        umask = os.umask(0)
        os.umask(umask)
        os.chmod(tmp, 0o666 & ~umask)  # mkstemp makes files private; a map file is meant to be read
        os.replace(tmp, args.out)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    on_map = sum(1 for c in candidates if c["x"] is not None)
    log(f"wrote {args.out} ({os.path.getsize(args.out) / 1e6:.2f} MB, {len(payload) / 1e6:.2f} MB uncompressed): {schema.summary(doc)}")
    log(f"candidates: {dict(kinds)}, {on_map} of {len(candidates)} on the map; {dropped_links} links dropped as unusable" + (f"; skipped: {dict(skipped)}" if skipped else ""))


if __name__ == "__main__":
    main()
