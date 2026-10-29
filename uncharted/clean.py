"""Clean a corpus before it is embedded, so nothing is paid for that is then thrown away.

    python -I uncharted/clean.py data/corpora/haptics.jsonl --out data/corpora/haptics.clean.jsonl \\
        --corpus-info data/corpora/haptics.corpus.json

Drops records whose type is not research (OpenAlex types such as editorials, errata and
datasets), records whose text after the title is shorter than --min-words, and every repeat
of a title once punctuation and case are ignored, keeping the first. With --corpus-info, the
rules applied and what each removed are added to that file's cleaning list for the map
manifest. Standard library only.
"""

from __future__ import annotations

import argparse
import json
import os
import re

DROP_TYPES = "other,report,libguides,paratext,dataset,editorial,letter,erratum,peer-review,retraction,supplementary-materials"


def norm_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", title.lower())


def clean(records: list[dict], drop_types: set[str], min_words: int, dedupe: bool) -> tuple[list[dict], dict[str, int]]:
    kept: list[dict] = []
    seen: set[str] = set()
    removed = {"type": 0, "short": 0, "duplicate title": 0}
    for r in records:
        if (r.get("extra") or {}).get("type") in drop_types:
            removed["type"] += 1
            continue
        text, title = r.get("text") or "", r.get("title") or ""
        body = text[len(title) :] if text.startswith(title) else text
        if len(body.split()) < min_words:
            removed["short"] += 1
            continue
        if dedupe:
            key = norm_title(title)
            if key in seen:
                removed["duplicate title"] += 1
                continue
            seen.add(key)
        kept.append(r)
    return kept, removed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("corpus", help="JSONL corpus file")
    ap.add_argument("--out", required=True)
    ap.add_argument("--drop-types", default=DROP_TYPES, help="comma-separated extra.type values to remove")
    ap.add_argument("--min-words", type=int, default=25, help="drop records with fewer words than this after the title")
    ap.add_argument("--no-dedupe", action="store_true")
    ap.add_argument("--corpus-info", help="JSON file whose cleaning list gets one line per rule")
    args = ap.parse_args()

    with open(args.corpus, encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh]
    drop_types = {t.strip() for t in args.drop_types.split(",") if t.strip()}
    kept, removed = clean(records, drop_types, args.min_words, not args.no_dedupe)
    tmp = args.out + ".part"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in kept)
    os.replace(tmp, args.out)
    print(f"{args.corpus}: {len(records)} -> {len(kept)} kept; removed {removed}")
    if args.corpus_info:
        with open(args.corpus_info, encoding="utf-8") as fh:
            info = json.load(fh)
        rules = [
            f"Dropped {removed['type']} items of types other than research ({', '.join(sorted(drop_types))}).",
            f"Dropped {removed['short']} items with under {args.min_words} words of text after the title.",
        ]
        if not args.no_dedupe:
            rules.append(f"Kept the first of each repeated title, ignoring case and punctuation ({removed['duplicate title']} repeats dropped).")
        info["cleaning"] = [line for line in info.get("cleaning", []) if not line.startswith(("Dropped ", "Kept the first"))] + rules
        info["sources"][0]["items"] = len(kept) if len(info.get("sources", [])) == 1 else info["sources"][0].get("items")
        tmp = args.corpus_info + ".tmp"  # replace, never truncate: the manifest's sources live here
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(info, fh, indent=1)
        os.replace(tmp, args.corpus_info)


if __name__ == "__main__":
    main()
