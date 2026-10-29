"""Scrape accessibility conference talk abstracts into the uncharted corpus format.

Standard library only. Sources:

    id24     Inclusive Design 24, https://inclusivedesign24.org/<year>/schedule/ (2018 onward)

Usage:

    python3 -I uncharted/talks.py id24 --out data/corpora/talks-id24.jsonl
    python3 -I uncharted/talks.py id24 --years 2024 2025 --out data/corpora/talks-id24.jsonl

Record shape is shared with every uncharted loader:

    id, corpus, title, text, year, venue, tags, popularity, url, extra
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import time
import urllib.error
import urllib.request

UA = "uncharted/0.1 (research corpus builder)"
MIN_WORDS = 15
# Recurring schedule filler, not talks: the same text every year shows up as perfect clones.
NOT_A_TALK = re.compile(r"^(the )?(pre-show|after[- ]party|welcome|closing|opening|break|intermission)\b", re.I)


def get(url: str, tries: int = 4) -> str | None:
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as err:
            if err.code == 404:
                return None
            if err.code in (429, 500, 502, 503, 504):
                time.sleep(2**attempt)
                continue
            raise
        except (urllib.error.URLError, TimeoutError):
            time.sleep(2**attempt)
    return None


def clean(fragment: str) -> str:
    text = re.sub(r"<[^>]+>", " ", fragment)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def id24(years: list[int], log) -> list[dict]:
    """One record per session block on each year's schedule page."""
    records: list[dict] = []
    for year in years:
        url = f"https://inclusivedesign24.org/{year}/schedule/"
        page = get(url)
        if page is None:
            log(f"id24 {year}: no schedule page")
            continue
        blocks = re.split(r'<div class="session"', page)[1:]
        kept = 0
        for n, block in enumerate(blocks, start=1):
            title = re.search(r'<span class="session_title">(.*?)</span>', block, re.S)
            presenter = re.search(r'<span class="session_presenter">(.*?)</span>', block, re.S)
            abstract = re.search(r'<div class="session_abstract">(.*?)</div>', block, re.S)
            video = re.search(r'<a class="videostream"[^>]*href="([^"]+)"', block)
            anchor = re.search(r'<h2 id="([^"]+)"', block)
            if not title or not abstract:
                continue
            title_text = clean(title.group(1))
            abstract_text = clean(abstract.group(1))
            if len(abstract_text.split()) < MIN_WORDS or NOT_A_TALK.search(title_text):
                continue
            slug = anchor.group(1) if anchor else f"session{n}"
            records.append(
                {
                    "id": f"id24:{year}:{slug}",
                    "corpus": "talks",
                    "title": title_text,
                    "text": f"{title_text}. {abstract_text}",
                    "year": year,
                    "venue": "Inclusive Design 24",
                    "tags": [],
                    "popularity": 0,
                    "url": f"{url}#{slug}",
                    "extra": {
                        "presenters": clean(presenter.group(1)) if presenter else None,
                        "video": video.group(1) if video else None,
                        "event": "id24",
                    },
                }
            )
            kept += 1
        log(f"id24 {year}: {len(blocks)} session blocks, kept {kept}")
        time.sleep(0.5)
    return records


HUB = "https://www.deque.com/axe-con/content-hub/"
RAW_DIR = "data/raw"


def cached_get(url: str, cache_path: str) -> str | None:
    """Fetch a page once; later runs re-parse the saved HTML instead of hitting the site."""
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as fh:
            return fh.read()
    page = get(url)
    if page is not None:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as fh:
            fh.write(page)
        time.sleep(0.3)
    return page


def axecon(years: list[int], log) -> list[dict]:
    """axe-con (Deque). The session index needs a login, but the public content hub
    paginates every recorded session (9 a page) and each session page is public."""
    slugs: list[str] = []
    seen: set[str] = set()
    page_no = 0
    for page_no in range(1, 200):
        url = HUB if page_no == 1 else f"{HUB}?query-0-page={page_no}"
        page = get(url)
        if page is None:
            break
        found = []
        for slug in re.findall(r'axe-con/sessions/([^/"?#]+)/', page):
            if slug not in seen:  # each card links to its session three times
                seen.add(slug)
                found.append(slug)
        if not found:
            break
        slugs.extend(found)
        time.sleep(0.3)
    log(f"axecon: {len(slugs)} session slugs from {page_no} hub pages")

    records: list[dict] = []
    for n, slug in enumerate(slugs, start=1):
        url = f"https://www.deque.com/axe-con/sessions/{slug}/"
        page = cached_get(url, os.path.join(RAW_DIR, "axecon", f"{slug}.html"))
        if page is None:
            log(f"axecon: could not fetch {slug}")
            continue
        title = re.search(r"<h1[^>]*>(.*?)</h1>", page, re.S)
        desc = re.search(r'<div class="axecon-content-hub-session__description">(.*?)</div>', page, re.S)
        if not title or not desc:
            log(f"axecon: no title or description on {slug}")
            continue
        abstract = clean(desc.group(1))
        if len(abstract.split()) < MIN_WORDS:
            continue
        year = re.search(r"Recorded Axe-con (\d{4})", page)
        track = re.search(r"content-hub/\?track=([a-z-]+)", page)
        tags_block = re.search(r'<ul class="axecon-content-hub-session__tags">(.*?)</ul>', page, re.S)
        tags = [clean(t) for t in re.findall(r"<li[^>]*>(.*?)</li>", tags_block.group(1), re.S)] if tags_block else []
        speakers = [clean(s) for s in re.findall(r'axecon-speaker-text__name">(.*?)</p>', page, re.S)]
        orgs = [clean(s) for s in re.findall(r'axecon-speaker-text__org">(.*?)</p>', page, re.S)]
        title_text = clean(title.group(1))
        records.append(
            {
                "id": f"axecon:{slug}",
                "corpus": "talks",
                "title": title_text,
                "text": f"{title_text}. {abstract}",
                "year": int(year.group(1)) if year else None,
                "venue": "axe-con",
                "tags": [t for t in tags if t],
                "popularity": 0,
                "url": url,
                "extra": {
                    "presenters": ", ".join(speakers) or None,
                    "orgs": orgs,
                    "track": track.group(1) if track else None,
                    "event": "axecon",
                },
            }
        )
        if n % 25 == 0:
            log(f"axecon: {n}/{len(slugs)} session pages")
    return records


CSUN_BASE = "https://www.csun.edu/cod/conference/sessions/{year}/index.php/public/"
CSUN_YEARS = (2023, 2024)  # the only years csun.edu still serves; 2025+ moved to a Cvent app


def _dl_pairs(block: str) -> dict[str, str]:
    """Return {label: raw inner HTML} for every <dt>/<dd> pair in a definition list."""
    pairs: dict[str, str] = {}
    for dt, dd in re.findall(r"<dt[^>]*>(.*?)</dt>\s*<dd[^>]*>(.*?)</dd>", block, re.S):
        pairs[clean(dt)] = dd
    return pairs


def _list_items(fragment: str) -> list[str]:
    items = [clean(li) for li in re.findall(r"<li[^>]*>(.*?)</li>", fragment, re.S)]
    return [i for i in items if i]


def csun(years: list[int], log) -> list[dict]:
    """CSUN Assistive Technology Conference, from the per-year PHP session app on csun.edu."""
    records: list[dict] = []
    for year in [y for y in years if y in CSUN_YEARS]:
        base = CSUN_BASE.format(year=year)
        listing = get(base + "conf_sessions/")
        if listing is None:
            log(f"csun {year}: no session list")
            continue
        presenters: dict[str, str] = {}
        for pid, cell in re.findall(
            r'presentations/view/(\d+)\.html">.*?</a>\s*</td>\s*(?:<!--.*?-->\s*)?<td style="width: 20%;">(.*?)</td>',
            listing,
            re.S,
        ):
            presenters[pid] = clean(cell.replace("<br />", ", ").replace("<br>", ", "))
        ids = sorted(set(re.findall(r"presentations/view/(\d+)\.html", listing)))
        log(f"csun {year}: {len(ids)} presentations listed")
        kept = 0
        for n, pid in enumerate(ids, start=1):
            url = base + f"presentations/view/{pid}.html"
            page = cached_get(url, os.path.join(RAW_DIR, "csun", str(year), f"{pid}.html"))
            if page is None:
                continue
            block = re.search(r'<div class="presentations websitePagesPublic">(.*?)</dl>', page, re.S)
            if not block:
                log(f"csun {year}: no presentation block on {pid}")
                continue
            title = re.search(r"<h1>(.*?)</h1>", block.group(1), re.S)
            pairs = _dl_pairs(block.group(1))
            description = clean(pairs.get("Description", ""))
            summary = clean(pairs.get("Session Summary (Abstract)", ""))
            body = description or summary
            if not title or len(body.split()) < MIN_WORDS:
                continue
            title_text = clean(title.group(1))
            topics = [clean(pairs.get("Primary Topic", ""))] + _list_items(pairs.get("Secondary Topics", ""))
            records.append(
                {
                    "id": f"csun:{year}:{pid}",
                    "corpus": "talks",
                    "title": title_text,
                    "text": f"{title_text}. {body}",
                    "year": year,
                    "venue": "CSUN Assistive Technology Conference",
                    "tags": [t for t in topics if t],
                    "popularity": 0,
                    "url": url,
                    "extra": {
                        "presenters": presenters.get(pid),
                        "summary": summary or None,
                        "audience": _list_items(pairs.get("Audience", "")),
                        "level": clean(pairs.get("Audience Level", "")) or None,
                        "event": "csun",
                    },
                }
            )
            kept += 1
            if n % 50 == 0:
                log(f"csun {year}: {n}/{len(ids)} pages, kept {kept}")
        log(f"csun {year}: kept {kept}")
    return records


SOURCES = {"id24": id24, "axecon": axecon, "csun": csun}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", choices=sorted(SOURCES))
    ap.add_argument("--out", required=True)
    ap.add_argument("--years", nargs="*", type=int, default=list(range(2018, 2027)))
    args = ap.parse_args()

    def log(msg: str) -> None:
        print(time.strftime("%H:%M:%S"), msg, flush=True)

    records = SOURCES[args.source](args.years, log)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as out:
        for rec in records:
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    log(f"wrote {len(records)} records to {args.out}")


if __name__ == "__main__":
    main()
