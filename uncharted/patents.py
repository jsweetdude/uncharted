"""Load US patent grants in streaming-media and accessibility CPC groups into the uncharted corpus format.

Source: PatentsView's bulk tables, final release dated 2024-12-31 (grants from 1976 through
2024), deposited on Zenodo under CC BY 4.0 and downloadable without a key or account. The
same tables now also live on the USPTO Open Data Portal, which needs a USPTO.gov sign-in;
files from there drop into the same directory and load the same way.

    python -I uncharted/patents.py fetch --dir data/raw/patentsview --citations
    python -I uncharted/patents.py build --dir data/raw/patentsview --out data/corpora/patents.jsonl

fetch prints each file's size before downloading it (about 2.7 GB for the four required
tables, plus 2.2 GB for the citation table). build streams the zipped TSVs, keeps utility
patents with any current CPC code in SCOPE and an abstract of at least 25 words, collapses
patents whose abstract repeats an earlier grant's word for word (continuations) onto the
earliest, and, if more than --max-records remain, keeps the newest grant years.

Record shape is shared with every uncharted loader:

    id, corpus, title, text, year, venue, tags, popularity, url, extra

id is the patent number (US + number), venue the first assignee, tags every current CPC
code on the patent, popularity the count of later US patents citing it (0 without the
citation table), url its Google Patents page.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import sys
import time
import urllib.request
import zipfile

ZENODO_RECORD = "https://zenodo.org/api/records/15783125"
UA = "uncharted/0.1 (research corpus builder)"
TABLES = ["g_patent", "g_patent_abstract", "g_cpc_current", "g_assignee_disambiguated"]
CITATIONS = "g_us_patent_citation"
MIN_WORDS = 25

# The corpus: CPC groups for streaming and media delivery, and for accessibility. A code is
# in scope when it starts with one of the prefixes, so H04N21/430 covers H04N21/4302,
# H04N21/4307 and H04N21/43072. Children that CPC numbers as siblings (H04M1/72478 under
# H04M1/72475) are listed on their own.
SCOPE = {
    "captions": ("subtitles, captions and teletext", ["H04N21/4884", "H04N7/088"]),
    "sync": ("synchronising content streams and additional data on the client", ["H04N21/430"]),
    "rendering": ("rendering content and additional data; visual interfaces", ["H04N21/431"]),
    "client-audio": ("audio processing on the client", ["H04N21/439"]),
    "end-user-apps": ("end-user applications: requesting content, playback control, supplementary services", ["H04N21/47"]),
    "program-guides": ("program guides and content-selection interfaces", ["H04N21/482"]),
    "client-config": ("end-user configuration of audio, image, language and screen layout", ["H04N21/485"]),
    "aids-blind-deaf": ("teaching and communication aids for blind, deaf or mute people", ["G09B21"]),
    "speech-visible": (
        "speech made visible or tactile for hearing-impaired people",
        ["G10L21/06", "G10L21/10", "G10L21/12", "G10L21/14", "G10L21/16", "G10L21/18", "G10L2021/065", "G10L2021/105"],
    ),
    "phones-disabled": ("phone interfaces for disabled, hearing-impaired and visually impaired users", ["H04M1/72475", "H04M1/72478", "H04M1/72481"]),
    "disabled-operation": ("devices that let disabled people operate apparatus", ["A61F4"]),
    "sound-io": ("sound input and output in user interfaces; voice commands", ["G06F3/16"]),
    "input-arrangements": ("input arrangements: gaze, gesture, haptics, head tracking, brain-computer, VR", ["G06F3/01"]),
}
PREFIXES = sorted(((p, key) for key, (_, prefixes) in SCOPE.items() for p in prefixes), key=lambda pk: -len(pk[0]))


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def scope_of(code: str) -> str | None:
    for prefix, key in PREFIXES:
        if code.startswith(prefix):
            return key
    return None


def table_path(directory: str, table: str) -> str | None:
    for ext in (".tsv.zip", ".tsv"):
        path = os.path.join(directory, table + ext)
        if os.path.exists(path):
            return path
    return None


def rows(directory: str, table: str, columns: list[str]):
    """Yield tuples of the named columns from <table>.tsv.zip or <table>.tsv."""
    path = table_path(directory, table)
    if path is None:
        raise SystemExit(f"missing {table}.tsv.zip in {directory}; run fetch first")
    csv.field_size_limit(sys.maxsize)
    if path.endswith(".zip"):
        zf = zipfile.ZipFile(path)
        fh = io.TextIOWrapper(zf.open(zf.namelist()[0]), encoding="utf-8", newline="")
    else:
        fh = open(path, encoding="utf-8", newline="")
    with fh:
        reader = csv.reader(fh, delimiter="\t", quotechar='"')
        header = next(reader)
        try:
            idx = [header.index(c) for c in columns]
        except ValueError:
            raise SystemExit(f"{table}: expected columns {columns}, found {header}")
        width = max(idx)
        for n, row in enumerate(reader, start=1):
            if len(row) > width:
                yield tuple(row[i] for i in idx)
            if n % 10_000_000 == 0:
                log(f"  {table}: {n:,} rows")


def cmd_fetch(args: argparse.Namespace) -> None:
    req = urllib.request.Request(ZENODO_RECORD, headers={"User-Agent": UA, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=90) as resp:
        record = json.load(resp)
    files = {f["key"]: f for f in record["files"]}
    wanted = TABLES + ([CITATIONS] if args.citations else [])
    os.makedirs(args.dir, exist_ok=True)
    todo = []
    for table in wanted:
        f = files[table + ".tsv.zip"]
        dest = os.path.join(args.dir, f["key"])
        have = os.path.getsize(dest) if os.path.exists(dest) else 0
        state = "have" if have == f["size"] else "fetch"
        log(f"{f['key']}: {f['size'] / 1e6:,.0f} MB ({state})")
        if state == "fetch":
            todo.append((f, dest))
    log(f"to download: {sum(f['size'] for f, _ in todo) / 1e9:.2f} GB")
    for f, dest in todo:
        tmp = dest + ".part"
        req = urllib.request.Request(f["links"]["self"], headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as out:
            while chunk := resp.read(1 << 20):
                out.write(chunk)
        if os.path.getsize(tmp) != f["size"]:
            raise SystemExit(f"{f['key']}: size mismatch, keeping {tmp}")
        os.replace(tmp, dest)
        log(f"wrote {dest}")


def cmd_build(args: argparse.Namespace) -> None:
    d = args.dir
    log("pass 1: patents with a CPC code in scope")
    scopes: dict[str, set[str]] = {}
    for pid, code in rows(d, "g_cpc_current", ["patent_id", "cpc_group"]):
        key = scope_of(code)
        if key:
            scopes.setdefault(pid, set()).add(key)
    log(f"{len(scopes):,} patents in scope")

    log("pass 2: every current CPC code on those patents")
    codes: dict[str, list[tuple[int, str]]] = {}
    for pid, seq, code in rows(d, "g_cpc_current", ["patent_id", "cpc_sequence", "cpc_group"]):
        if pid in scopes:
            codes.setdefault(pid, []).append((int(seq or 0), code))

    log("patents: type, date, title")
    patents: dict[str, dict] = {}
    for pid, ptype, date, title, kind, claims, withdrawn in rows(
        d, "g_patent", ["patent_id", "patent_type", "patent_date", "patent_title", "wipo_kind", "num_claims", "withdrawn"]
    ):
        if pid in scopes and ptype == "utility" and withdrawn not in ("1", "True", "true"):
            patents[pid] = {"date": date, "title": " ".join(title.split()), "kind": kind, "claims": int(claims) if claims.isdigit() else None}
    log(f"{len(patents):,} utility patents, not withdrawn")

    log("abstracts")
    short = 0
    for pid, abstract in rows(d, "g_patent_abstract", ["patent_id", "patent_abstract"]):
        p = patents.get(pid)
        if p is None:
            continue
        abstract = " ".join(abstract.split())
        if len(abstract.split()) < MIN_WORDS:
            short += 1
            continue
        p["abstract"] = abstract
    patents = {pid: p for pid, p in patents.items() if p.get("abstract") and p["title"]}
    log(f"{len(patents):,} with an abstract of {MIN_WORDS}+ words ({short:,} too short)")

    # Continuations often repeat the parent's abstract word for word; on the map they are one
    # point. Keep the earliest grant and list the rest on it.
    first_with: dict[str, str] = {}
    for pid in sorted(patents, key=lambda q: (patents[q]["date"], q)):
        key = re.sub(r"[^a-z0-9]+", "", patents[pid]["abstract"].lower())
        if key in first_with:
            patents[first_with[key]].setdefault("same_abstract", []).append(f"US{pid}")
        else:
            first_with[key] = pid
    dropped = len(patents) - len(first_with)
    patents = {pid: patents[pid] for pid in first_with.values()}
    log(f"{len(patents):,} after collapsing {dropped:,} repeated abstracts")

    log("first assignees")
    for pid, seq, org, first, last in rows(
        d,
        "g_assignee_disambiguated",
        ["patent_id", "assignee_sequence", "disambig_assignee_organization", "disambig_assignee_individual_name_first", "disambig_assignee_individual_name_last"],
    ):
        p = patents.get(pid)
        if p is None or seq != "0":
            continue
        p["assignee"] = org.strip() or " ".join(x for x in (first.strip(), last.strip()) if x) or None

    cited = None
    if table_path(d, CITATIONS):
        log("forward citations")
        cited = dict.fromkeys(patents, 0)
        for (target,) in rows(d, CITATIONS, ["citation_patent_id"]):
            if target in cited:
                cited[target] += 1
    else:
        log(f"no {CITATIONS} table: popularity is 0")

    keep = sorted(patents, key=lambda pid: patents[pid]["date"], reverse=True)
    if args.since:
        keep = [pid for pid in keep if int(patents[pid]["date"][:4]) >= args.since]
    if len(keep) > args.max_records:
        cutoff = int(patents[keep[args.max_records - 1]]["date"][:4]) + 1  # whole grant years only
        keep = [pid for pid in keep if int(patents[pid]["date"][:4]) >= cutoff]
        log(f"over --max-records: kept grant years {cutoff} on")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tmp = args.out + ".part"
    per_scope = dict.fromkeys(SCOPE, 0)
    with open(tmp, "w", encoding="utf-8") as out:
        for pid in keep:
            p = patents[pid]
            cpc = []
            for _, code in sorted(codes.get(pid, [])):
                if code not in cpc:
                    cpc.append(code)
            in_scope = sorted(scopes[pid])
            for key in in_scope:
                per_scope[key] += 1
            number = f"US{pid}"
            rec = {
                "id": number,
                "corpus": "patents",
                "title": p["title"],
                "text": f"{p['title']}. {p['abstract']}",
                "year": int(p["date"][:4]),
                "venue": p.get("assignee"),
                "tags": cpc,
                "popularity": cited[pid] if cited is not None else 0,
                "url": f"https://patents.google.com/patent/{number}{p['kind'] or ''}",
                "extra": {
                    "cpc": cpc,
                    "scope": in_scope,
                    "assignee": p.get("assignee"),
                    "grant_date": p["date"],
                    "kind": p["kind"],
                    "claims": p["claims"],
                    "forward_citations": cited[pid] if cited is not None else None,
                    "same_abstract": p.get("same_abstract", []),
                },
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.replace(tmp, args.out)
    log(f"wrote {len(keep):,} records to {args.out}")
    for key, n in per_scope.items():
        log(f"  {key:20s} {n:7,}  {SCOPE[key][0]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="download the PatentsView tables from Zenodo")
    f.add_argument("--dir", default="data/raw/patentsview")
    f.add_argument("--citations", action="store_true", help=f"also fetch {CITATIONS} (2.2 GB) for forward-citation counts")
    b = sub.add_parser("build", help="stream the tables into a corpus JSONL")
    b.add_argument("--dir", default="data/raw/patentsview")
    b.add_argument("--out", required=True)
    b.add_argument("--max-records", type=int, default=100_000, help="above this, keep only the newest grant years")
    b.add_argument("--since", type=int, help="keep grants from this year on")
    args = ap.parse_args()
    {"fetch": cmd_fetch, "build": cmd_build}[args.cmd](args)


if __name__ == "__main__":
    main()
