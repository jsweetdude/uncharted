"""Estimate what a map of a research topic will cost and how long it will take, before anything runs.

    python -I uncharted/estimate.py topics "haptic feedback"
    python -I uncharted/estimate.py scope --topic T11687 --name haptics --provider openai \\
        [--years 2015-2026] [--max-items 20000] --scope-out data/runs/haptics/scope.json

topics: the OpenAlex topics that best match a search string, with ids, fields and work
counts, for the person to pick one. One OpenAlex search call ($0.001).

scope: resolves a topic (--topic), a title-and-abstract search (--search) or a raw OpenAlex
filter (--filter) to the exact filter the run will use, counts the matching works, and
fetches a random sample of up to 200 of them to measure how many survive cleaning and how
long their text is. It prints the estimate as JSON: items, requests, tokens and dollars,
each as a low-high range, and the time for the model download, the pull, the embedding, the
map and the candidate search. That costs two OpenAlex list calls ($0.0002), or search calls
for --search. --scope-out writes the scope file run.py executes: the filter and count, the
configuration, and the caps on items, tokens and dollars the person approves. Nothing else
is fetched and nothing is embedded.

Prices are dollars as of 2026-10-28: OpenAlex $0.0001 per list call and $0.001 per search
call (help.openalex.org/access/example-costs); embedding prices are in embed.py.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # `python -I` leaves the script's directory off sys.path
import openalex  # noqa: E402
from clean import DROP_TYPES, clean  # noqa: E402
from embed import CHARS_PER_TOKEN, DEFAULT_MODELS, PRICES  # noqa: E402
from provenance import hf_snapshot  # noqa: E402

LIST_CALL = 0.0001
SEARCH_CALL = 0.001
PER_PAGE = 200
FREE_DAILY = 1.00  # a free OpenAlex key's daily budget, in dollars
LOCAL_RATES = {"mps": (6.0, 14.0), "cuda": (40.0, 150.0), "cpu": (0.8, 4.0)}  # items a second, Qwen3-Embedding-0.6B on abstracts
MODEL_BYTES = {"Qwen/Qwen3-Embedding-0.6B": 1.2e9}
DUPLICATE_SHARE = (0.0, 0.08)  # titles repeated within a topic; 4% in the accessibility corpus


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% interval for a proportion from a sample."""
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def minutes(seconds: float) -> float:
    return round(seconds / 60, 1)


def map_seconds(n: int) -> float:
    """Neighbors, both UMAP layouts and clustering; fitted to runs on an M3 Pro (967 talks in 16 s,
    32,485 papers in 75 s), with a fixed start-up cost for compiling UMAP."""
    return 15 + 8.2e-4 * n**1.1


def candidate_seconds(n: int) -> float:
    """gaps.py connections: up to 200,000 mixtures scored against every item."""
    return 5 + 200_000 * n * 1024 / 1.5e11


def local_rate(model: str, device: str, benchmarks: str) -> tuple[tuple[float, float], str]:
    try:
        with open(benchmarks, encoding="utf-8") as fh:
            known = json.load(fh)
        rate = known[f"{model}|{device}"]["items_per_second"]
        return (rate * 0.8, rate * 1.2), f"measured on this machine ({rate:.1f} items a second)"
    except (OSError, KeyError, json.JSONDecodeError):
        lo, hi = LOCAL_RATES.get(device, LOCAL_RATES["cpu"])
        return (lo, hi), f"typical for {device}; run once to measure this machine"


def local_device() -> str | None:
    try:
        import torch
    except ImportError:
        return None
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def cmd_topics(args: argparse.Namespace) -> None:
    topics = openalex.search_topics(args.query, args.api_key, top=args.top)
    out = [
        {
            "id": t["id"].rsplit("/", 1)[-1],
            "name": t.get("display_name"),
            "field": (t.get("field") or {}).get("display_name"),
            "subfield": (t.get("subfield") or {}).get("display_name"),
            "works": t.get("works_count"),
            "description": (t.get("description") or "")[:300],
        }
        for t in topics
    ]
    print(json.dumps({"query": args.query, "topics": out, "cost": {"openalexDollars": SEARCH_CALL}}, indent=1))


def cmd_scope(args: argparse.Namespace) -> None:
    if sum(bool(x) for x in (args.topic, args.search, args.filter)) != 1:
        raise SystemExit("give exactly one of --topic, --search or --filter")
    parts = []
    topic = None
    if args.topic:
        tid = args.topic.rsplit("/", 1)[-1]
        parts.append(f"topics.id:{tid}")
        try:
            data, _ = openalex.fetch(f"https://api.openalex.org/topics/{tid}" + (f"?api_key={args.api_key}" if args.api_key else ""))
            topic = {"id": tid, "name": data.get("display_name"), "field": (data.get("field") or {}).get("display_name"), "works": data.get("works_count")}
        except Exception:  # noqa: BLE001  (the name is only for display)
            topic = {"id": tid}
    elif args.search:
        parts.append(f"title_and_abstract.search:{args.search}")
    else:
        parts.append(args.filter)
    if not args.filter:
        parts += ["has_abstract:true", "language:en"]
    if args.years:
        parts.append(f"publication_year:{args.years}")
    flt = ",".join(parts)
    per_call = SEARCH_CALL if ".search" in flt else LIST_CALL

    t0 = time.time()
    count = openalex.count_works(flt, args.api_key)
    sample = openalex.sample_works(flt, args.sample, 42, args.api_key)
    latency = (time.time() - t0) / 2
    records = [r for r in (openalex.normalize(w, "estimate") for w in sample) if r]
    kept, removed = clean(records, set(DROP_TYPES.split(",")), 25, dedupe=True)
    n = len(sample)
    lo_s, hi_s = wilson(len(kept), n)
    fetched = min(count, math.ceil(args.max_items / max(lo_s, 0.05))) if args.max_items else count
    requests = max(1, math.ceil(fetched / PER_PAGE))
    items_hi = min(count * hi_s, args.max_items or math.inf)
    items_lo = min(count * lo_s * (1 - DUPLICATE_SHARE[1]), args.max_items or math.inf)
    chars = [len(r["text"]) for r in kept] or [1500]
    mean_chars = sum(chars) / len(chars)
    se = (sum((c - mean_chars) ** 2 for c in chars) / max(len(chars) - 1, 1)) ** 0.5 / max(len(chars), 1) ** 0.5
    tpi_lo = (mean_chars - 2 * se) / CHARS_PER_TOKEN * 0.85
    tpi_hi = (mean_chars + 2 * se) / CHARS_PER_TOKEN * 1.15
    tokens = (int(items_lo * tpi_lo), int(math.ceil(items_hi * tpi_hi)))

    model = args.model or DEFAULT_MODELS[args.provider]
    price = PRICES.get((args.provider, model), 0.0 if args.provider == "local" else None)
    warnings = []
    if price is None:
        warnings.append(f"no price known for {args.provider} {model}; the dollar estimate leaves embedding out")
        price = 0.0
    embed_dollars = (tokens[0] * price / 1e6, tokens[1] * price / 1e6)
    openalex_dollars = requests * per_call
    times: dict[str, tuple[float, float]] = {"pull": (requests * (latency * 0.7 + 0.15), requests * (latency * 1.5 + 0.15))}
    device = None
    if args.provider == "local":
        device = local_device()
        if device is None:
            warnings.append("local embedding needs the local extra: uv sync --extra local")
            device = "cpu"
        revision, _ = hf_snapshot(model)
        size = MODEL_BYTES.get(model)
        times["download"] = (0.0, 0.0) if revision else ((size / 50e6, size / 8e6) if size else (60.0, 600.0))
        (r_lo, r_hi), rate_note = local_rate(model, device, args.benchmarks)
        times["embed"] = (items_lo / r_hi, items_hi / r_lo)
    else:
        rate_note = "API time depends on the account's rate limits"
        batches = (max(1, tokens[0] // 250_000), max(1, math.ceil(tokens[1] / 100_000)))
        times["download"] = (0.0, 0.0)
        times["embed"] = (batches[0] * 2.0, batches[1] * 10.0)
        if not any(os.environ.get(v) for v in ({"openai": ("UNCHARTED_OPENAI_API_KEY", "OPENAI_API_KEY"), "voyage": ("VOYAGE_API_KEY",)}[args.provider])):
            warnings.append(f"no {args.provider} key in the environment; the run will stop at the embedding step without one")
    times["map"] = (map_seconds(int(items_lo)) * 0.7, map_seconds(int(items_hi)) * 2)
    times["candidates"] = (candidate_seconds(int(items_lo)) * 0.5, candidate_seconds(int(items_hi)) * 3)
    if count > 100_000 and not args.max_items:
        warnings.append(f"{count:,} works match; consider --max-items (most-cited first) or --years to narrow it")
    if lo_s < 0.5:
        warnings.append(f"only {len(kept)} of {n} sampled works survive cleaning; many have no usable abstract")
    if not args.api_key and requests > 1:
        warnings.append("OpenAlex serves only the first page without an API key; set OPENALEX_API_KEY (free at openalex.org)")
    if openalex_dollars > FREE_DAILY:
        warnings.append(f"the pull costs about ${openalex_dollars:.2f}, more than a free key's ${FREE_DAILY:.2f} a day: it pauses when the day's budget runs out and resumes the next day")
    if count < 50:
        warnings.append(f"only {count} works match: too few for a map")

    total = (embed_dollars[0] + openalex_dollars, embed_dollars[1] + openalex_dollars)
    caps = {
        "items": int(min(args.max_items or math.inf, math.ceil(count * hi_s * 1.05))),
        "tokens": int(math.ceil(tokens[1] * 1.1)) if args.provider != "local" else None,
        "dollars": round(math.ceil((total[1] * 1.1 + 0.005) * 100) / 100, 2),
    }
    estimate = {
        "scope": {
            "source": "OpenAlex works",
            "topic": topic,
            "search": args.search,
            "filter": flt,
            "years": args.years,
            "order": "most-cited first" if args.max_items else "as OpenAlex returns them",
            "maxItems": args.max_items or None,
            "count": count,
            "countedAt": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        },
        "sample": {"fetched": n, "usable": len(records), "kept": len(kept), "removed": removed, "survival": [round(lo_s, 3), round(hi_s, 3)], "meanChars": round(mean_chars), "types": sorted({(w.get("type") or "") for w in sample})},
        "items": [int(items_lo), int(items_hi)],
        "requests": {"openalex": requests, "perCallDollars": per_call},
        "tokens": list(tokens),
        "dollars": {"openalex": round(openalex_dollars, 4), "embedding": [round(embed_dollars[0], 4), round(embed_dollars[1], 4)], "total": [round(total[0], 4), round(total[1], 4)]},
        "minutes": {k: [minutes(v[0]), minutes(v[1])] for k, v in times.items()},
        "minutesTotal": [minutes(sum(v[0] for v in times.values())), minutes(sum(v[1] for v in times.values()))],
        "embedding": {"provider": args.provider, "model": model, "dimensions": 1024, "maxSeq": args.max_seq, "device": device, "pricePerMillion": price, "rateNote": rate_note},
        "caps": caps,
        "warnings": warnings,
        "estimateCost": {"openalexDollars": round(2 * per_call + (LIST_CALL if args.topic else 0), 4)},
    }
    print(json.dumps(estimate, indent=1))
    if args.scope_out:
        scope = {
            "uncharted_scope": 1,
            "name": args.name,
            "title": args.title or (topic or {}).get("name") or args.search or args.name,
            "kind": "paper",
            "created": estimate["scope"]["countedAt"],
            "source": estimate["scope"],
            "config": {"provider": args.provider, "model": model, "dimensions": 1024, "maxSeq": args.max_seq},
            "caps": caps,
            "estimate": {k: estimate[k] for k in ("items", "requests", "tokens", "dollars", "minutes", "minutesTotal", "warnings")},
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.scope_out)), exist_ok=True)
        with open(args.scope_out, "w", encoding="utf-8") as fh:
            json.dump(scope, fh, indent=1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api-key", default=os.environ.get("OPENALEX_API_KEY"), help="OpenAlex key (or set OPENALEX_API_KEY)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("topics")
    t.add_argument("query")
    t.add_argument("--top", type=int, default=5)
    s = sub.add_parser("scope")
    s.add_argument("--topic", help="OpenAlex topic id, such as T11687")
    s.add_argument("--search", help="words to find in titles and abstracts (search calls cost ten times list calls)")
    s.add_argument("--filter", help="a raw OpenAlex works filter, used as given")
    s.add_argument("--years", help="publication years, such as 2015-2026")
    s.add_argument("--max-items", type=int, default=0, help="keep at most this many works, most-cited first")
    s.add_argument("--name", default="map", help="name for the run's files")
    s.add_argument("--title", help="title for the map (default: the topic's name)")
    s.add_argument("--provider", choices=sorted(DEFAULT_MODELS), default="local")
    s.add_argument("--model")
    s.add_argument("--max-seq", type=int, default=1024)
    s.add_argument("--sample", type=int, default=200)
    s.add_argument("--benchmarks", default="data/benchmarks.json", help="measured local embedding rates")
    s.add_argument("--scope-out", help="write the scope file for run.py here")
    args = ap.parse_args()
    try:
        {"topics": cmd_topics, "scope": cmd_scope}[args.cmd](args)
    except openalex.BudgetExhausted as err:
        raise SystemExit(f"OpenAlex refused the request: {err}") from None


if __name__ == "__main__":
    main()
