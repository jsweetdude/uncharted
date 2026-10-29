"""Pull accessibility papers from Semantic Scholar's bulk search into the uncharted corpus format.

Free and keyless, at about one request a second; each page returns up to 1,000 papers.
Semantic Scholar has abstracts for roughly half of the matching papers, and records
without an abstract are dropped, so the corpus is smaller than the match count.

    python3 -I uncharted/s2.py --out data/corpora/papers-s2.jsonl

Record shape is shared with every uncharted loader:

    id, corpus, title, text, year, venue, tags, popularity, url, extra
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

API = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
UA = "uncharted/0.1 (research corpus builder)"
FIELDS = "title,abstract,year,venue,citationCount,s2FieldsOfStudy,publicationTypes,externalIds,authors"
MIN_WORDS = 25
PAUSE = 1.2  # seconds between requests; the keyless pool allows about one a second

PHRASES = [
    "web accessibility",
    "digital accessibility",
    "accessible design",
    "accessibility evaluation",
    "accessibility guidelines",
    "inclusive design",
    "assistive technology",
    "screen reader",
    "screen readers",
    "blind users",
    "visually impaired users",
    "deaf users",
    "hard of hearing",
    "low vision",
    "people with disabilities",
    "users with disabilities",
    "cognitive accessibility",
    "alt text",
    "audio description",
    "closed captions",
    "sign language",
    "WCAG",
]
TERMS_QUERY = "(" + " | ".join(f'"{p}"' if " " in p else p for p in PHRASES) + ")"

# Broad disability vocabulary, used only inside accessibility venues.
VENUE_QUERY = "(accessib* | disabilit* | impair* | blind | deaf | assistive | inclusive | braille | caption* | dyslexia | autism | aging | older)"
VENUES = ",".join(
    [
        "Conference on Computers and Accessibility",
        "ACM Transactions on Accessible Computing",
        "International Cross-Disciplinary Conference on Web Accessibility",
        "Universal Access in the Information Society",
    ]
)

QUERIES = {
    "accessibility-terms-cs": {"query": TERMS_QUERY, "fieldsOfStudy": "Computer Science"},
    "accessibility-venues": {"query": VENUE_QUERY, "venue": VENUES},
}


def fetch(params: dict, tries: int = 7) -> dict:
    url = f"{API}?{urllib.parse.urlencode(params)}"
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=90) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as err:
            if err.code in (429, 500, 502, 503, 504):
                time.sleep(min(5 * 2**attempt, 90))
                continue
            raise
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            time.sleep(min(5 * 2**attempt, 90))
    raise RuntimeError(f"gave up on {url}")


def normalize(paper: dict, source_query: str) -> dict | None:
    abstract = (paper.get("abstract") or "").strip()
    title = (paper.get("title") or "").strip()
    if not title or len(abstract.split()) < MIN_WORDS:
        return None
    fields = []
    for f in paper.get("s2FieldsOfStudy") or []:
        cat = f.get("category")
        if cat and cat not in fields:
            fields.append(cat)
    types = [t for t in (paper.get("publicationTypes") or []) if t]
    doi = (paper.get("externalIds") or {}).get("DOI")
    authors = [a.get("name") for a in (paper.get("authors") or [])[:3] if a.get("name")]
    pid = paper["paperId"]
    return {
        "id": f"s2:{pid}",
        "corpus": "papers",
        "title": title,
        "text": f"{title}. {abstract}",
        "year": paper.get("year"),
        "venue": (paper.get("venue") or "").strip() or None,
        "tags": fields + types,
        "popularity": paper.get("citationCount") or 0,
        "url": f"https://doi.org/{doi}" if doi else f"https://www.semanticscholar.org/paper/{pid}",
        "extra": {
            "authors": authors,
            "fields": fields,
            "types": types,
            "doi": doi,
            "source_query": source_query,
        },
    }


def pull(name: str, base: dict, seen: set[str], out, log) -> int:
    params = dict(base, fields=FIELDS, limit=1000)
    token = None
    pages = kept = dropped = dup = 0
    total = None
    while True:
        if token:
            params["token"] = token
        data = fetch(params)
        if total is None:
            total = data.get("total")
            log(f"{name}: {total} matches")
        for paper in data.get("data", []):
            rec = normalize(paper, name)
            if rec is None:
                dropped += 1
                continue
            if rec["id"] in seen:
                dup += 1
                continue
            seen.add(rec["id"])
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1
        pages += 1
        if pages % 10 == 0:
            log(f"{name}: {pages} pages, kept {kept}, no-abstract {dropped}, dup {dup}")
        token = data.get("token")
        if not token or not data.get("data"):
            break
        time.sleep(PAUSE)
    log(f"{name} DONE: {pages} pages, kept {kept}, no-abstract {dropped}, dup {dup}")
    return kept


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", action="append", help="run only this named query (repeatable)")
    args = ap.parse_args()

    def log(msg: str) -> None:
        print(time.strftime("%H:%M:%S"), msg, flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tmp = args.out + ".part"
    seen: set[str] = set()
    total = 0
    with open(tmp, "w", encoding="utf-8") as out:
        for name, base in QUERIES.items():
            if args.only and name not in args.only:
                continue
            total += pull(name, base, seen, out, log)
            time.sleep(PAUSE)
    os.replace(tmp, args.out)
    log(f"wrote {total} records to {args.out}")


if __name__ == "__main__":
    main()
