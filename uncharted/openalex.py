"""Pull accessibility research papers from OpenAlex into the uncharted corpus format.

Standard library only, so it can run before the project venv exists:

    python3 -I uncharted/openalex.py --out data/corpora/papers-oa.jsonl --api-key KEY
    OPENALEX_API_KEY=KEY python3 -I uncharted/openalex.py --out data/corpora/papers-oa.jsonl
    python3 -I uncharted/openalex.py --out data/corpora/papers-oa.jsonl --resume   # next day

OpenAlex meters its API: about $0.0001 per record returned. Without a key, requests share
a $0.10 daily budget per IP address and cannot cursor-page past the first 200 results at
all; a free key has its own $1 a day (10,000 records). This loader therefore writes
records as they arrive, saves its cursor after every page, and stops cleanly when
OpenAlex refuses the next page; `--resume` continues where it stopped. The output file is
only renamed from `.part` once every filter is complete.

Every uncharted loader writes the same JSONL record shape:

    id, corpus, title, text, year, venue, tags, popularity, url, extra

`text` is what gets embedded (title + abstract). `tags` and `popularity` are the Steam
"tags" and "review count" analogues: OpenAlex topics and keywords, and citation count.
"""

from __future__ import annotations

import argparse
import json
import os
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


def normalize(work: dict, source_filter: str) -> dict | None:
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
        "corpus": "papers",
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


def pull(name: str, flt: str, state: dict, api_key: str | None, seen: set[str], out, log) -> bool:
    """Pull one filter from its saved cursor. Returns True when the filter is complete."""
    st = state.setdefault(name, {"cursor": "*", "done": False, "pages": 0, "kept": 0, "dropped": 0, "dup": 0})
    if st["done"]:
        log(f"{name}: already complete ({st['kept']} records)")
        return True
    while st["cursor"]:
        params = {"filter": flt, "per-page": 200, "cursor": st["cursor"], "select": SELECT}
        if api_key:
            params["api_key"] = api_key
        try:
            data, headers = fetch(f"{API}?{urllib.parse.urlencode(params)}")
        except BudgetExhausted as err:
            log(f"{name}: OpenAlex refused the next page: {err}")
            log(f"{name}: {st['pages']} pages, {st['kept']} records so far; rerun with --resume (and --api-key) to continue")
            return False
        for work in data.get("results", []):
            rec = normalize(work, name)
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
        yield_state(state)
        if st["pages"] % 10 == 0:
            remaining = headers.get("x-ratelimit-remaining-usd") or headers.get("X-RateLimit-Remaining-USD")
            budget = f", ${remaining} of today's budget left" if remaining else ""
            log(f"{name}: {st['pages']} pages, kept {st['kept']}, dropped {st['dropped']}, dup {st['dup']}{budget}")
        time.sleep(0.15)
    st["done"] = True
    yield_state(state)
    log(f"{name} DONE: {st['pages']} pages, kept {st['kept']}, dropped {st['dropped']}, dup {st['dup']}")
    return True


_STATE_PATH = ""


def yield_state(state: dict) -> None:
    tmp = _STATE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=1)
    os.replace(tmp, _STATE_PATH)


def main() -> None:
    global _STATE_PATH  # noqa: PLW0603
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output JSONL path")
    ap.add_argument("--only", action="append", help="run only this named filter (repeatable)")
    ap.add_argument("--api-key", default=os.environ.get("OPENALEX_API_KEY"), help="OpenAlex API key (or set OPENALEX_API_KEY)")
    ap.add_argument("--resume", action="store_true", help="continue a pull that stopped when the budget ran out")
    args = ap.parse_args()

    def log(msg: str) -> None:
        print(time.strftime("%H:%M:%S"), msg, flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    part = args.out + ".part"
    _STATE_PATH = args.out + ".state.json"

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

    complete = True
    with open(part, "a", encoding="utf-8") as out:
        for name, flt in FILTERS.items():
            if args.only and name not in args.only:
                continue
            if not pull(name, flt, state, args.api_key, seen, out, log):
                complete = False
                break
            time.sleep(0.15)

    if complete:
        os.replace(part, args.out)
        os.remove(_STATE_PATH)
        total = sum(st["kept"] for st in state.values())
        log(f"wrote {total} records to {args.out}")
    else:
        log(f"partial pull kept in {part}; state in {_STATE_PATH}")


if __name__ == "__main__":
    main()
