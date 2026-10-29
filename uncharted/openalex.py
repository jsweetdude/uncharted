"""Pull research papers from OpenAlex into the uncharted corpus format.

Standard library only, so it can run before the project venv exists:

    python3 -I uncharted/openalex.py --out data/corpora/papers-oa.jsonl          # the accessibility corpus
    python3 -I uncharted/openalex.py --out data/corpora/haptics.jsonl --corpus haptics \
        --filter haptics="topics.id:T11687,has_abstract:true,language:en" --max-items 20000
    python3 -I uncharted/openalex.py --out data/corpora/papers-oa.jsonl --resume   # next day

Without --filter it pulls the accessibility corpus: OpenAlex topic T12481 plus four
accessibility venues. The key comes from --api-key or OPENALEX_API_KEY.

OpenAlex meters its API per request: a page of up to 200 filtered results costs $0.0001
($0.10 per 1,000 list calls; a search costs ten times that), per help.openalex.org's
example costs, checked 2026-10-28. Without a key, requests share a $0.10 daily budget per
IP address and cannot cursor-page past the first page; a free key has its own $1 a day,
10,000 pages or about 2 million records. This loader writes records as they arrive, saves
its cursor after every page, and stops cleanly when OpenAlex refuses the next page;
``--resume` continues where it stopped. The output file is only renamed from `.part` once
every filter is complete; `<out>.pull.json` then records the calls made and their cost.
--max-items stops after that many records (with --order cited, the most-cited first, so a
capped corpus holds the field's core), and --max-dollars stops before a page would cross it.
Exit status: 0 complete, 3 stopped at --max-dollars, 5 OpenAlex refused a page (usually the
day's budget), 1 anything else.

Every uncharted loader writes the same JSONL record shape:

    id, corpus, title, text, year, venue, tags, popularity, url, extra

`text` is what gets embedded (title + abstract). `tags` and `popularity` are the Steam
"tags" and "review count" analogues: OpenAlex topics and keywords, and citation count.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

API = "https://api.openalex.org/works"
UA = "uncharted/0.1 (research corpus builder)"
SELECT = ",".join(
    [
        "id",
        "doi",
        "title",
        "publication_year",
        "type",
        "language",
        "cited_by_count",
        "primary_location",
        "topics",
        "keywords",
        "abstract_inverted_index",
        "authorships",
    ]
)

# ASSETS, TACCESS, W4A, Universal Access in the Information Society
VENUES = "S7407088110|S79551877|S7407088483|S30338973"

FILTERS = {
    # OpenAlex topic T12481 "Digital Accessibility for Disabilities": the core corpus.
    "digital-accessibility-topic": "topics.id:T12481,has_abstract:true,language:en",
    # Accessibility venues whose papers were not tagged with the topic.
    "accessibility-venues": f"primary_location.source.id:{VENUES},has_abstract:true,topics.id:!T12481",
}

MIN_WORDS = 25  # OpenAlex marks "No abstract available." as an abstract; drop stubs.


class BudgetExhausted(Exception):
    pass


def fetch(url: str, tries: int = 6) -> tuple[dict, dict]:
    """Return (json, headers). Raises BudgetExhausted when OpenAlex refuses for lack of credit."""
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.load(resp), dict(resp.headers)
        except urllib.error.HTTPError as err:
            body = ""
            try:
                body = err.read().decode("utf-8", errors="replace")[:300]
            except Exception:  # noqa: BLE001
                pass
            if err.code in (401, 402, 403):
                raise BudgetExhausted(f"HTTP {err.code}: {body}") from err
            if err.code == 429 and attempt == tries - 1:
                raise BudgetExhausted(f"HTTP 429 after {tries} tries: {body}") from err
            if err.code in (429, 500, 502, 503, 504):
                time.sleep(2**attempt)
                continue
            raise
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            time.sleep(2**attempt)
    raise RuntimeError(f"gave up on {url}")


def abstract_from(inverted: dict | None) -> str:
    if not inverted:
        return ""
    positions: dict[int, str] = {}
    for word, places in inverted.items():
        for p in places:
            positions[p] = word
    return " ".join(positions[i] for i in sorted(positions))


def normalize(work: dict, source_filter: str, corpus: str = "papers") -> dict | None:
    abstract = abstract_from(work.get("abstract_inverted_index"))
    if len(abstract.split()) < MIN_WORDS or abstract.lower().startswith("no abstract"):
        return None
    title = (work.get("title") or "").strip()
    if not title:
        return None
    location = work.get("primary_location") or {}
    source = (location.get("source") or {}) if location else {}
    topics = [t.get("display_name") for t in (work.get("topics") or []) if t.get("display_name")]
    keywords = [k.get("display_name") for k in (work.get("keywords") or []) if k.get("display_name")]
    authors = []
    for authorship in (work.get("authorships") or [])[:3]:
        name = (authorship.get("author") or {}).get("display_name")
        if name:
            authors.append(name)
    oid = work["id"].rsplit("/", 1)[-1]
    return {
        "id": f"openalex:{oid}",
        "corpus": corpus,
        "title": title,
        "text": f"{title}. {abstract}",
        "year": work.get("publication_year"),
        "venue": source.get("display_name"),
        "tags": topics + keywords,
        "popularity": work.get("cited_by_count") or 0,
        "url": work.get("doi") or f"https://openalex.org/{oid}",
        "extra": {
            "type": work.get("type"),
            "authors": authors,
            "topics": topics,
            "keywords": keywords,
            "source_filter": source_filter,
        },
    }


def count_works(flt: str, api_key: str | None) -> int:
    """How many works match a filter: one list call."""
    params = {"filter": flt, "per-page": 1, "select": "id"}
    if api_key:
        params["api_key"] = api_key
    data, _ = fetch(f"{API}?{urllib.parse.urlencode(params)}")
    return int(data["meta"]["count"])


def sample_works(flt: str, n: int, seed: int, api_key: str | None) -> list[dict]:
    """A random sample of up to n works (at most 200) matching a filter: one list call."""
    params = {"filter": flt, "sample": n, "seed": seed, "per-page": min(n, 200), "select": SELECT}
    if api_key:
        params["api_key"] = api_key
    data, _ = fetch(f"{API}?{urllib.parse.urlencode(params)}")
    return data.get("results", [])


def search_topics(query: str, api_key: str | None, top: int = 5) -> list[dict]:
    """OpenAlex topics matching a search string, best match first: one search call."""
    params = {"search": query, "per-page": top, "select": "id,display_name,works_count,description,subfield,field,domain"}
    if api_key:
        params["api_key"] = api_key
    data, _ = fetch(f"https://api.openalex.org/topics?{urllib.parse.urlencode(params)}")
    return data.get("results", [])


def pull(name: str, flt: str, state: dict, api_key: str | None, seen: set[str], out, log, corpus: str = "papers", max_items: int | None = None,
         per_page: int = 200, order: str | None = None, ledger: dict | None = None, max_dollars: float | None = None, per_call: float = 0.0001) -> str:
    """Pull one filter from its saved cursor. Returns "done", "budget" (OpenAlex refused the
    next page, usually because the day's budget is spent) or "cap" (the next page would cross
    --max-dollars). Every page served is charged to the ledger, which outlives the pull."""
    ledger = ledger if ledger is not None else {"calls": 0, "dollars": 0.0}
    st = state.setdefault(name, {"cursor": "*", "done": False, "pages": 0, "kept": 0, "dropped": 0, "dup": 0})
    if st["done"]:
        log(f"{name}: already complete ({st['kept']} records)")
        return "done"
    while st["cursor"]:
        if max_items is not None and len(seen) >= max_items:
            st["capped"] = True
            log(f"{name}: stopped at the cap of {max_items} records")
            break
        if max_dollars is not None and ledger["dollars"] + per_call > max_dollars + 1e-12:
            log(f"{name}: stopping before the next page: ${ledger['dollars']:.4f} spent of the ${max_dollars:.4f} allowed")
            return "cap"
        params = {"filter": flt, "per-page": per_page, "cursor": st["cursor"], "select": SELECT}
        if order == "cited":
            params["sort"] = "cited_by_count:desc"
        if api_key:
            params["api_key"] = api_key
        try:
            data, headers = fetch(f"{API}?{urllib.parse.urlencode(params)}")
        except BudgetExhausted as err:
            log(f"{name}: OpenAlex refused the next page: {redact(str(err), api_key)}")
            log(f"{name}: {st['pages']} pages, {st['kept']} records so far; rerun with --resume to continue")
            return "budget"
        try:
            cost = float(headers.get("x-ratelimit-cost-usd") or headers.get("X-RateLimit-Cost-USD") or per_call)
        except ValueError:
            cost = per_call
        ledger["calls"] += 1
        ledger["dollars"] = round(ledger["dollars"] + cost, 6)
        for work in data.get("results", []):
            if max_items is not None and len(seen) >= max_items:
                break
            rec = normalize(work, name, corpus)
            if rec is None:
                st["dropped"] += 1
                continue
            if rec["id"] in seen:
                st["dup"] += 1
                continue
            seen.add(rec["id"])
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            st["kept"] += 1
        out.flush()
        st["pages"] += 1
        st["cursor"] = data.get("meta", {}).get("next_cursor")
        state["_ledger"] = ledger
        yield_state(state)
        if st["pages"] % 10 == 0:
            remaining = headers.get("x-ratelimit-remaining-usd") or headers.get("X-RateLimit-Remaining-USD")
            budget = f", ${remaining} of today's budget left" if remaining else ""
            log(f"{name}: {st['pages']} pages, kept {st['kept']}, dropped {st['dropped']}, dup {st['dup']}{budget}")
        time.sleep(0.15)
    st["done"] = True
    state["_ledger"] = ledger
    yield_state(state)
    log(f"{name} DONE: {st['pages']} pages, kept {st['kept']}, dropped {st['dropped']}, dup {st['dup']}")
    return "done"


def redact(text: str, secret: str | None) -> str:
    return text.replace(secret, "[key]") if secret else text


_STATE_PATH = ""


def yield_state(state: dict) -> None:
    tmp = _STATE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=1)
    os.replace(tmp, _STATE_PATH)


def main() -> int:
    global _STATE_PATH  # noqa: PLW0603
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output JSONL path")
    ap.add_argument("--only", action="append", help="run only this named filter (repeatable)")
    ap.add_argument("--api-key", default=os.environ.get("OPENALEX_API_KEY"), help="OpenAlex API key (or set OPENALEX_API_KEY)")
    ap.add_argument("--resume", action="store_true", help="continue a pull that stopped when the budget ran out")
    ap.add_argument("--filter", action="append", metavar="NAME=FILTER", help="pull this OpenAlex works filter instead of the accessibility corpus (repeatable)")
    ap.add_argument("--corpus", default="papers", help="corpus name written into every record")
    ap.add_argument("--max-items", type=int, help="stop after this many records")
    ap.add_argument("--order", choices=["cited"], help="cited: most-cited works first, so a capped pull keeps the field's core")
    ap.add_argument("--max-dollars", type=float, help="stop before a page would bring the pull's spend past this")
    ap.add_argument("--per-call", type=float, help="dollars per page, for the cap (default: $0.0001, or $0.001 for a search)")
    ap.add_argument("--per-page", type=int, default=200)
    ap.add_argument("--corpus-info", help="also write the map manifest's corpus section (sources, query, retrieval date) to this JSON file")
    args = ap.parse_args()
    filters = dict(FILTERS)
    if args.filter:
        filters = {}
        for spec in args.filter:
            name, _, flt = spec.partition("=")
            if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", name) or not flt:
                raise SystemExit(f"--filter takes NAME=FILTER with a lowercase name, not {spec[:80]!r}")
            filters[name] = flt
    if args.max_items is not None and args.max_items < 1:
        raise SystemExit("--max-items must be at least 1")
    if args.max_dollars is not None and not (args.max_dollars >= 0):
        raise SystemExit("--max-dollars must be zero or more")
    per_call = args.per_call if args.per_call is not None else (0.001 if any(".search" in f for f in filters.values()) else 0.0001)

    def log(msg: str) -> None:
        print(time.strftime("%H:%M:%S"), redact(msg, args.api_key), flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    part = args.out + ".part"
    _STATE_PATH = args.out + ".state.json"
    summary_path = args.out + ".pull.json"

    state: dict = {}
    seen: set[str] = set()
    if args.resume and os.path.exists(_STATE_PATH):
        with open(_STATE_PATH, encoding="utf-8") as fh:
            state = json.load(fh)
        if os.path.exists(part):
            with open(part, encoding="utf-8") as fh:
                for line in fh:
                    seen.add(json.loads(line)["id"])
        log(f"resuming: {len(seen)} records already in {part}")
    elif os.path.exists(part) or os.path.exists(_STATE_PATH):
        raise SystemExit(f"{part} or its state file exists; pass --resume to continue it or delete both to start over")
    if not args.api_key:
        log("no API key: OpenAlex serves only the first page of each filter without one; pass --api-key or set OPENALEX_API_KEY")
    ledger = state.get("_ledger") or {"calls": 0, "dollars": 0.0}

    finished_before = os.path.exists(args.out) and not os.path.exists(part) and state and all(st.get("done") for k, st in state.items() if not k.startswith("_"))
    outcome = "done"
    if not finished_before:
        with open(part, "a", encoding="utf-8") as out:
            for name, flt in filters.items():
                if args.only and name not in args.only:
                    continue
                outcome = pull(name, flt, state, args.api_key, seen, out, log, args.corpus, args.max_items, args.per_page, args.order, ledger, args.max_dollars, per_call)
                if outcome != "done":
                    break
                time.sleep(0.15)
    if outcome != "done":
        log(f"partial pull kept in {part}; state in {_STATE_PATH}; ${ledger['dollars']:.4f} spent in {ledger['calls']} calls")
        return 3 if outcome == "cap" else 5

    total = sum(st["kept"] for k, st in state.items() if not k.startswith("_"))
    summary = {"calls": ledger["calls"], "dollars": ledger["dollars"], "records": total, "filters": filters, "maxItems": args.max_items, "order": args.order,
               "capped": any(st.get("capped") for k, st in state.items() if not k.startswith("_")), "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    if args.corpus_info:
        query = "; ".join(f"works filter {flt}" for flt in filters.values())
        if args.max_items is not None:
            query += f"; at most {args.max_items} works" + (", most-cited first" if args.order == "cited" else "")
        info = {
            "kind": "paper",
            "sources": [
                {
                    "name": "OpenAlex",
                    "url": "https://openalex.org/",
                    "query": query,
                    "retrieved": time.strftime("%Y-%m-%d"),
                    "items": total,
                    "license": "CC0-1.0",
                    "attribution": "Work metadata and abstracts from OpenAlex (openalex.org), released under CC0.",
                }
            ],
            "cleaning": [f"Dropped works whose abstract is under {MIN_WORDS} words or a placeholder, and works without a title."],
            "popularity": "citations (OpenAlex cited_by_count)",
            "tags": "OpenAlex topics and keywords",
        }
        atomic_json(args.corpus_info, info)
    atomic_json(summary_path, summary)  # written before the corpus is published, so its cost is never lost
    if not finished_before:
        os.replace(part, args.out)
    os.remove(_STATE_PATH)
    log(f"wrote {total} records to {args.out}; ${ledger['dollars']:.4f} in {ledger['calls']} calls")
    return 0


def atomic_json(path: str, data: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1)
    os.replace(tmp, path)


if __name__ == "__main__":
    sys.exit(main())
