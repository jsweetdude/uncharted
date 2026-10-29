"""Write the example map files in schema/examples/.

    python -I schema/examples/make_examples.py

small.json: a complete, valid map of twelve invented talks: three neighborhoods, two items in
no neighborhood, one candidate of each kind (one of them left off the map) and two drafts,
one pass and one retry. Loneliness and positions follow the definitions in schema/README.md.

hostile-text.json: the same map with markup, script and control characters in every
free-text field. It is valid; a viewer must show that text literally and never run it.

invalid/: each file breaks one rule. expected-errors.json maps each file name to a phrase
that the first error from uncharted/schema.py contains.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "uncharted"))  # `python -I` leaves the script's directory off sys.path
import schema  # noqa: E402

TALKS = [
    # title, year, venue (-1 for none), neighborhood (-1 for none), x, y, nn1
    ("Live captions for town council meetings", 2021, 0, 0, 0.18, 0.72, 0.84),
    ("Caption timing in fast dialogue scenes", 2022, 0, 0, 0.24, 0.80, 0.87),
    ("Speaker labels in group captions", 2023, 0, 0, 0.12, 0.66, 0.87),
    ("Reading speeds for children's captions", 2024, 1, 0, 0.28, 0.62, 0.79),
    ("Extended audio description for lectures", 2020, 0, 1, 0.70, 0.78, 0.82),
    ("Describing on-screen text in news broadcasts", 2022, 1, 1, 0.76, 0.86, 0.85),
    ("Audio description scripts written by volunteers", 2023, 0, 1, 0.64, 0.70, 0.85),
    ("Synthetic voices for audio description", 2025, -1, 1, 0.80, 0.70, 0.80),
    ("Sound cues as visual indicators in shooters", 2024, 0, 2, 0.48, 0.22, 0.76),
    ("Subtitles for game cutscenes and menus", 2025, 0, 2, 0.56, 0.16, 0.76),
    ("Haptic alerts for doorbells and alarms", None, 1, -1, 1.00, 0.40, 0.62),
    ("Plain-language summaries of court rulings", 2019, 1, -1, 0.00, 0.00, 0.58),
]
HOODS = [
    # terms, tags, color slot, examples (closest to the center first)
    (["captions", "caption", "speaker", "timing"], ["Deaf", "captions"], 0, [1, 0, 2]),
    (["description", "audio", "scripts"], ["blind", "audio description"], 1, [6, 4, 5]),
    (["game", "subtitles", "sound"], ["games"], 2, [8, 9]),
]
NN1 = [t[6] for t in TALKS]
PLACE_SPREAD = 0.30  # the fixture's stand-in for the placement threshold an export measures
SNIPPETS = [
    "Our town council asked for live captions in its public meetings. We share what it took…",
    "Fast dialogue breaks caption timing rules. We tested three timing strategies with deaf viewers…",
    "Group conversations need speaker labels. We compared names, colors and positions…",
    "Children read captions more slowly than adults. We measured reading speeds by age…",
    "Lectures leave little room for description. Extended description pauses the video…",
    "News broadcasts put text on screen that blind viewers miss. We described it in the gaps…",
    "Volunteers wrote audio description scripts for a regional theater. Here is how we trained them…",
    "Synthetic voices can read description scripts. We compared them with human narrators…",
    "Shooters signal danger with sound. We turned the key sounds into visual indicators…",
    "Cutscenes and menus often lack subtitles. We audited thirty games…",
    "Doorbells and alarms can buzz a phone or a wristband. We tested patterns deaf households prefer…",
    "Court rulings are hard to read. We wrote plain-language summaries and tested them…",
]


def lonely_of(sim: float) -> float:
    """For a point outside the corpus: the share of all items whose nearest neighbor is closer."""
    return round(100 * sum(1 for v in NN1 if v > sim) / len(NN1), 1)


def item_lonely(i: int) -> float:
    """For item i: the share of the other items whose nearest neighbor is closer; ties are not closer."""
    return round(100 * sum(1 for j, v in enumerate(NN1) if j != i and v > NN1[i]) / (len(NN1) - 1), 1)


def place(nearest: list[list]) -> tuple[float | None, float | None, float]:
    """export.py's placement rule: similarity-weighted mean of the 8 nearest items' positions,
    left off the map when they scatter more than the threshold."""
    support = nearest[:8]
    w = [max(sim, 0.0) + 1e-9 for _, sim in support]
    cx = sum(wi * TALKS[i][4] for wi, (i, _) in zip(w, support)) / sum(w)
    cy = sum(wi * TALKS[i][5] for wi, (i, _) in zip(w, support)) / sum(w)
    spread = (sum(wi * ((TALKS[i][4] - cx) ** 2 + (TALKS[i][5] - cy) ** 2) for wi, (i, _) in zip(w, support)) / sum(w)) ** 0.5
    if spread > PLACE_SPREAD:
        return None, None, round(spread, 4)
    return round(cx, 4), round(cy, 4), round(spread, 4)


def percentile(values: list[float], q: float) -> float:
    v = sorted(values)
    pos = (len(v) - 1) * q / 100
    lo = int(pos)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)


def small() -> dict:
    n = len(TALKS)
    lonely = [item_lonely(i) for i in range(n)]
    cluster = [t[3] for t in TALKS]
    hoods = []
    for h, (terms, tags, slot, examples) in enumerate(HOODS):
        members = [i for i in range(n) if cluster[i] == h]
        hoods.append(
            {
                "id": h,
                "terms": terms,
                "tags": tags,
                "size": len(members),
                "x": round(sum(TALKS[i][4] for i in members) / len(members), 4),
                "y": round(sum(TALKS[i][5] for i in members) / len(members), 4),
                "density": round(statistics.median(NN1[i] for i in members), 3),
                "popularity": 0,
                "slot": slot,
                "examples": examples,
            }
        )
    url_prefix = [0] * 10 + [1, -1]
    url = [str(i + 1) for i in range(10)] + ["example.com/haptics", ""]
    connection = {
        "id": "captions-x-description",
        "code": "C1",
        "kind": "connection",
        "label": "captions × audio description",
        "headline": "Between neighborhood 0 (captions) and neighborhood 1 (audio description)",
        "score": {"nn1": 0.61, "lonely": lonely_of(0.61), "mixNn1": 0.58, "baseline": 0.64, "margin": 0.06, "beats": 97.0, "pairTail": 16.0, "pairP": 0.0001},
        "targetPct": 90,
        "stop": "target reached",
        "neighborhoods": [0, 1],
        "parents": [1, 6],
        "nearest": [[1, 0.61], [6, 0.6], [2, 0.57], [4, 0.55]],
        "tags": ["captions", "audio description"],
        "brief": "# Brief: captions × audio description\n\nWrite a talk title and a 120 to 200 word abstract that joins the two neighborhoods.\n\n## Must not copy\n\n- 0.610 Caption timing in fast dialogue scenes\n- 0.600 Audio description scripts written by volunteers\n",
        "drafts": [
            {
                "id": "captions-x-description.d1",
                "code": "d1",
                "title": "One script for captions and audio description",
                "text": "Caption files and audio description scripts are written by different people from the same picture, and they drift apart: a sound the captions name is a sound the description talks over. We merged them into one timed script with three kinds of line, for speech, for sound and for picture, and let each output take the lines it needs. This talk shows the format, the review workflow and what changed for viewers who use both.",
                "notes": "Helps: deafblind viewers and anyone using captions and description together\nBarrier: two scripts that contradict each other\nStandard: WCAG 2.2 success criteria 1.2.2 and 1.2.5",
                "verdict": "pass",
                "nn1": 0.6,
                "lonely": lonely_of(0.6),
                "toCandidate": 0.83,
                "reasons": [],
                "nearest": [[1, 0.6], [6, 0.59], [2, 0.56]],
                "checked": "2026-10-28T12:00:00Z",
            }
        ],
    }
    pair = {
        "id": "description-x-games",
        "code": "P1",
        "kind": "pair",
        "label": "audio description × games",
        "headline": "Between neighborhood 1 (audio description) and neighborhood 2 (games)",
        "score": {"nn1": 0.66, "lonely": lonely_of(0.66)},
        "targetPct": 90,
        "stop": "drift limit",
        "neighborhoods": [1, 2],
        "parents": [6, 9],
        "nearest": [[7, 0.66], [9, 0.64], [8, 0.61]],
        "tags": ["audio description", "games"],
        "brief": "# Brief: audio description × games\n\nWrite a talk title and abstract between the two neighborhoods.\n",
        "drafts": [],
    }
    blend = {
        "id": "haptics-x-sound-cues",
        "code": "B1",
        "kind": "blend",
        "label": "doorbell haptics × sound cues",
        "headline": "Blend of two items",
        "score": {"nn1": 0.7, "lonely": lonely_of(0.7)},
        "targetPct": 90,
        "stop": "max steps",
        "neighborhoods": [2],
        "parents": [10, 8],
        "nearest": [[8, 0.7], [10, 0.69], [11, 0.62], [9, 0.6]],
        "tags": ["games", "haptics"],
        "drafts": [
            {
                "id": "haptics-x-sound-cues.d1",
                "code": "d1",
                "title": "Doorbell haptics for game sound cues",
                "text": "Games show sound cues as icons at the edge of the screen. We tried the vibration patterns deaf households already know from doorbell and alarm alerts instead, so a player feels footsteps behind them as the pattern for the front door.",
                "verdict": "retry",
                "nn1": 0.81,
                "lonely": lonely_of(0.81),
                "toCandidate": 0.71,
                "reasons": ["too close: nearest real item at 0.810 (target ≤ 0.700)"],
                "nearest": [[8, 0.81], [9, 0.7]],
                "checked": "2026-10-28T12:05:00-06:00",
            }
        ],
    }
    candidates = [connection, pair, blend]
    for c in candidates:
        c["x"], c["y"], c["spread"] = place(c["nearest"])
        for d in c["drafts"]:
            d["x"], d["y"], _ = place(d["nearest"])
    assert connection["x"] is not None and pair["x"] is not None and blend["x"] is None, "placement no longer matches the example's design"
    return {
        "format": schema.FORMAT,
        "version": schema.VERSION,
        "id": "sample-talks",
        "title": "Sample talks (invented)",
        "description": "Twelve invented talks for testing map viewers.",
        "exported": "2026-10-28T12:00:00Z",
        "manifest": {
            "generator": {"name": "uncharted", "version": "1.0.0.dev0"},
            "corpus": {
                "kind": "talk",
                "topic": "captions and audio description (invented sample)",
                "sources": [
                    {
                        "name": "Invented sample",
                        "url": "https://example.org/talks/",
                        "query": "none: written by hand to test viewers",
                        "retrieved": "2026-10-28",
                        "items": len(TALKS),
                        "license": "CC0-1.0",
                        "attribution": "Invented for testing; these talks do not exist.",
                    }
                ],
                "cleaning": ["No cleaning: the sample was written by hand."],
                "popularity": "Not used in this sample.",
                "tags": "Invented labels.",
                "fingerprint": hashlib.sha256("".join(f"sample:{i + 1}\t{t[0]}\n" for i, t in enumerate(TALKS)).encode("utf-8")).hexdigest()[:32],
            },
            "embedding": {
                "provider": "local",
                "model": "Qwen/Qwen3-Embedding-0.6B",
                "dimensions": 1024,
                "maxTokens": 1024,
                "pooling": "last-token",
                "normalized": True,
                "instruction": None,
                "precision": "float16",
                "input": "title and abstract",
                "truncated": 0,
            },
            "map": {"neighbors": 10, "seed": 42, "layout": "2-d UMAP, cosine", "clustering": "HDBSCAN on a 10-d UMAP, min cluster size 2"},
            "candidates": {
                "connections": 1,
                "pairs": 1,
                "blends": 1,
                "targetPct": 90,
                "placePct": 90,
                "placeNeighbors": 8,
                "placeSpread": PLACE_SPREAD,
                "method": "connections-v1",
                "band": "75th to 97th percentile of neighborhood similarity",
                "mixesPerPair": 200,
                "pairsTested": 3,
                "ringSize": 10,
                "simBins": 10,
                "densityBins": 5,
                "tailPct": 5,
                "alpha": 0.05,
                "pairCutoff": 0.0167,
                "maxBusiest": 0.4,
                "seed": 42,
            },
        },
        "stats": {
            "items": n,
            "neighborhoods": len(hoods),
            "unclustered": cluster.count(-1),
            "candidates": len(candidates),
            "drafts": sum(len(c["drafts"]) for c in candidates),
            "nn1Percentiles": {p: round(percentile(NN1, float(p)), 3) for p in schema.PERCENTILES},
        },
        "lookups": {"venues": ["Sample Conference", "Sample Journal"], "urlPrefixes": ["https://example.org/talks/", "https://"]},
        "items": {
            "count": n,
            "id": [f"sample:{i + 1}" for i in range(n)],
            "title": [t[0] for t in TALKS],
            "x": [t[4] for t in TALKS],
            "y": [t[5] for t in TALKS],
            "cluster": cluster,
            "nn1": NN1,
            "lonely": lonely,
            "year": [t[1] for t in TALKS],
            "venue": [t[2] for t in TALKS],
            "urlPrefix": url_prefix,
            "url": url,
            "snippet": SNIPPETS,
            "source": [0] * n,
        },
        "neighborhoods": hoods,
        "candidates": candidates,
    }


def hostile(base: dict) -> dict:
    d = copy.deepcopy(base)
    d["id"] = "hostile-sample"
    d["title"] = "<b>Hostile</b> sample"
    d["description"] = "Bell \u0007, null \u0000, and <script>alert('description')</script>"
    titles = d["items"]["title"]
    titles[0] = "<img src=x onerror=\"alert('title')\">Live captions for town council meetings"
    titles[1] = "Caption timing</title><script>alert('title')</script>"
    titles[2] = "Speaker" + "labels" * 50
    titles[3] = "Reading speeds ‮for children's​ captions"
    titles[4] = "{{constructor.constructor('alert(1)')()}} ${alert('title')}"
    d["items"]["url"][0] = "1?q=\"><script>alert('url')</script>"
    d["lookups"]["venues"][0] = "<iframe src=javascript:alert('venue')></iframe>"
    hood = d["neighborhoods"][0]
    hood["terms"] = ["<b>captions</b>", "&lt;script&gt;", "‮reversed"]
    hood["tags"] = ["javascript:alert(1)"]
    c = d["candidates"][0]
    c["label"] = "<svg onload=alert('label')> × audio description"
    c["headline"] = "${alert('headline')}"
    c["brief"] = "# Heading <script>alert('brief')</script>\n\n[a link](javascript:alert('brief'))\n\n<img src=x onerror=alert('brief')>\n"
    c["tags"] = ["<u>captions</u>"]
    dr = c["drafts"][0]
    dr["title"] = "</textarea><script>alert('draft')</script>"
    dr["text"] = "<style>body{display:none}</style> The text of the draft."
    dr["notes"] = "<a href=\"javascript:alert('notes')\">notes</a>"
    d["candidates"][2]["drafts"][0]["reasons"] = ["<marquee>too close</marquee>"]
    corpus = d["manifest"]["corpus"]
    corpus["topic"] = "<h1>topic</h1>"
    corpus["sources"][0]["name"] = "<script>alert('source')</script>"
    corpus["sources"][0]["url"] = "https://example.org/\"onmouseover=\"alert('source')"
    d["manifest"]["map"]["layout"] = "<script>alert('params')</script>"
    return d


def invalid(base: dict) -> dict[str, tuple[str, str]]:
    """file name -> (file text, phrase the first error contains)"""

    def mutate(change) -> str:
        d = copy.deepcopy(base)
        change(d)
        return json.dumps(d, ensure_ascii=False, indent=1)

    out = {
        "version-2.json": (mutate(lambda d: d.update(version=2)), "supports version 1"),
        "wrong-format.json": (mutate(lambda d: d.update(format="uncharted-map")), "not an Uncharted map file"),
        "javascript-link.json": (mutate(lambda d: d["lookups"]["urlPrefixes"].__setitem__(0, "javascript:alert(1)//")), "does not match the pattern"),
        "short-column.json": (mutate(lambda d: d["items"]["title"].pop()), "items.title: has 11 entries"),
        "bad-reference.json": (mutate(lambda d: d["candidates"][0]["nearest"][0].__setitem__(0, 99)), "item 99 does not exist"),
        "out-of-range.json": (mutate(lambda d: d["items"]["x"].__setitem__(3, 1.5)), "above the maximum"),
        "wrong-type.json": (mutate(lambda d: d["items"]["nn1"].__setitem__(0, "0.84")), "should be number"),
        "unknown-field.json": (mutate(lambda d: d.update(script="alert(1)")), "unexpected property"),
        "missing-field.json": (mutate(lambda d: d["manifest"].pop("embedding")), 'is missing "embedding"'),
        "duplicate-id.json": (mutate(lambda d: d["items"]["id"].__setitem__(1, d["items"]["id"][0])), "is used twice"),
        "count-mismatch.json": (mutate(lambda d: d["stats"].update(items=13)), "stats.items: is 13"),
    }
    def retry_without_reason(d):
        d["candidates"][2]["drafts"][0]["reasons"] = []

    def credentials(d):
        d["lookups"]["urlPrefixes"][0] = "https://user:secret@example.org/talks/"

    def big_integer(d):
        d["neighborhoods"][0]["popularity"] = 2**60

    def no_evidence(d):
        del d["candidates"][0]["score"]["pairP"]

    def stray_example(d):
        d["neighborhoods"][0]["examples"][0] = 4

    def bad_date(d):
        d["exported"] = "2026-02-30T12:00:00Z"

    def no_offset(d):
        d["candidates"][0]["drafts"][0]["checked"] = "2026-10-28T12:00:00"

    out.update(
        {
            "retry-without-reason.json": (mutate(retry_without_reason), "must say why"),
            "link-with-credentials.json": (mutate(credentials), "carries a user name or password"),
            "huge-integer.json": (mutate(big_integer), "too large for a JavaScript reader"),
            "connection-without-evidence.json": (mutate(no_evidence), "a connection must report pairP"),
            "example-outside-neighborhood.json": (mutate(stray_example), "is not in this neighborhood"),
            "impossible-date.json": (mutate(bad_date), "is not a real date or time"),
            "timestamp-without-offset.json": (mutate(no_offset), "does not match the pattern"),
        }
    )
    text = json.dumps(base, ensure_ascii=False, indent=1)
    out["truncated.json"] = (text[:200], "not valid JSON")
    out["deep-nesting.json"] = ("[" * 20000 + "]" * 20000, "too deeply")
    d = copy.deepcopy(base)
    d["items"]["x"][0] = 0.123456789
    out["overflowing-number.json"] = (json.dumps(d, ensure_ascii=False, indent=1).replace("0.123456789", "1e400", 1), "too large to represent")
    d = copy.deepcopy(base)
    d["items"]["nn1"][0] = 12345.678
    out["nan.json"] = (json.dumps(d, ensure_ascii=False, indent=1).replace("12345.678", "NaN", 1), "NaN")
    return out


def write(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text if text.endswith("\n") or path.endswith("truncated.json") else text + "\n")


def main() -> None:
    base = small()
    os.makedirs(os.path.join(HERE, "invalid"), exist_ok=True)
    for name, doc, ascii_only in (("small.json", base, False), ("hostile-text.json", hostile(base), True)):
        errors = schema.validate(doc)
        if errors:
            raise SystemExit(f"{name} is not valid: {errors[:3]}")
        write(os.path.join(HERE, name), json.dumps(doc, ensure_ascii=ascii_only, indent=1))
        print(f"wrote {name}")
    expected = {}
    for name, (text, phrase) in invalid(base).items():
        path = os.path.join(HERE, "invalid", name)
        write(path, text)
        try:
            errors = schema.validate(schema.read_map(path))
        except schema.InvalidMap as exc:
            errors = [f"the file {exc}"]
        if not errors or phrase not in errors[0]:
            raise SystemExit(f"invalid/{name}: expected an error containing {phrase!r}, got {errors[:1]}")
        expected[name] = phrase
    write(os.path.join(HERE, "expected-errors.json"), json.dumps(expected, indent=1, ensure_ascii=False))
    print(f"wrote {len(expected)} invalid examples and expected-errors.json")


if __name__ == "__main__":
    main()
