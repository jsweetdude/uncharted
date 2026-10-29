# uncharted

Find the empty places in a field and write what would live there.

The method: embed every item in a corpus as a point on a map of meaning, find the regions where nothing exists yet (holes between neighborhoods, or the emptiest spot near a blend of related items), have a writer draft the item that would sit there, then embed the draft exactly like a real item and measure where it landed. If it is too close to something that exists, the measurements say what, and the writer tries again. Every surviving candidate carries a number: how similar it is to the closest real thing, and what share of the corpus is more crowded than it.

The pipeline is domain-agnostic. A domain is a loader that emits the shared record shape and a one-paragraph instruction saying what kind of artifact to write (paper abstract, talk abstract, invention disclosure, component spec). Everything runs locally; no text leaves the machine.

## Setup

Python 3.12 (arm64 on Apple silicon). Create a venv and install:

    uv venv --python 3.12 .venv
    uv pip install --python .venv/bin/python -r requirements.txt

The first embedding run downloads `BAAI/bge-large-en-v1.5` (about 1.3 GB) into the Hugging Face cache. On an M3 Pro the model embeds about 25 abstracts a second in float16.

Run scripts with `python -I` so downloaded corpus files can never shadow a module.

## Pipeline

1. **Load** a corpus into `data/corpora/<name>.jsonl`. Every loader writes the same record:

       id, corpus, title, text, year, venue, tags, popularity, url, extra

   `text` is what gets embedded (title + body). `tags` and `popularity` are optional signals used for neighborhood labels and ranking.

   - `uncharted/s2.py` — accessibility papers from Semantic Scholar's keyless bulk search (about one request a second; abstracts present for roughly half of matches).
   - `uncharted/openalex.py` — the same from OpenAlex topic T12481 and four venues. OpenAlex charges per request ($0.0001 for a filter-only page of up to 200 records), and paging past the first 200 results needs a free API key: pass `--api-key` or set `OPENALEX_API_KEY`. The loader saves its cursor after every page; `--resume` continues an interrupted pull.
   - `uncharted/talks.py id24|axecon|csun` — conference talk abstracts from Inclusive Design 24 (2018 on), axe-con (the public content hub: recorded sessions, 2025 on) and CSUN (2023 and 2024 from csun.edu). Fetched pages are cached under `data/raw/`.
   - `uncharted/patents.py fetch|build` — US patent grants in streaming-media and accessibility CPC groups, from PatentsView's bulk tables (see Data sources). `fetch --citations` downloads `g_patent`, `g_patent_abstract`, `g_cpc_current`, `g_assignee_disambiguated` and `g_us_patent_citation` (about 4.9 GB zipped, sizes printed first); `build` streams them with the standard library and keeps utility patents carrying any CPC code in `SCOPE`: captions and teletext (`H04N21/4884`, `H04N7/088`), client synchronization, rendering and audio (`H04N21/430`, `/431`, `/439`), end-user applications, program guides and client configuration (`H04N21/47`, `/482`, `/485`), aids for blind and deaf people (`G09B21`), speech made visible or tactile (`G10L21/06`–`/18`, `G10L2021/065`, `/105`), phones for disabled users (`H04M1/72475`, `/72478`, `/72481`), devices operated by disabled people (`A61F4`), sound input and output (`G06F3/16`) and input arrangements (`G06F3/01`). Record: `id` the patent number, `venue` the first assignee, `tags` every current CPC code, `popularity` forward citations, `url` the Google Patents page. Abstracts repeated word for word by continuations collapse onto the earliest grant. Zenodo serves each file over one connection at 1 to 4 MB a second, so `fetch` can take 30 to 60 minutes; `build` takes about 8 minutes and writes 72,389 records from the 2024 release.

2. **Embed**: `python -I uncharted/embed.py --name papers --fp16 data/corpora/papers-s2.jsonl`
   Writes `data/embeddings/<name>.npy` (L2-normalized rows) plus `<name>.meta.jsonl` in row order and `<name>.model.json`.

   Optional clean-up without re-embedding: `python -I uncharted/subset.py --name papers --out-name papers-clean` keeps rows whose text names a disability population, an assistive technology or a digital-accessibility phrase, drops biomedical matches, and collapses duplicate titles (keyword searches catch "web-accessible" protein databases and the like).

   Compare two embedded corpora of one field side by side (size, DOI overlap in both directions, nearest-neighbor percentiles, neighborhoods, a noise proxy): `python -I uncharted/compare.py --a papers-clean --b papers-oa --out data/compare-papers.md`.

3. **Map**: `python -I uncharted/space.py --name papers [--min-cluster-size N]`
   Top-10 neighbors per item, loneliness percentiles, HDBSCAN neighborhoods on a 10-d UMAP, a 2-d UMAP for the flat map, per-neighborhood labels (distinctive title terms and tags), and a console report: biggest and most popular neighborhoods, tightest and loosest, loneliest items, most cloned items.

4. **Find gaps**: `python -I uncharted/gaps.py blend|holes ...`
   - `blend --titles "..." "..."` or `--ids` or `--clusters 12 40`: average the ingredients, then push the point away from every item (soft weights, hottest on the closest) while a tether holds it near the blend, until it is lonelier than `--target-pct` of the corpus or has drifted past `--max-drift`.
   - `holes --top 8`: grid the flat map, keep empty cells with clustered items on at least 7 of 8 sides and at least 2 neighborhoods in the ring, anchor a point on the ring, push the same way.
   - `pairs --top 15 [--briefs]`: rank every related pair of neighborhoods (centroid similarity inside `--min-related`/`--max-related`) by how empty their midpoint is; with `--briefs`, push each midpoint into the gap and write a brief.
   Each gap writes `<out>.json` (measurements and the point) and `<out>.md`, a brief for the writer: where it sits, the ingredients, the tags it leans toward, the nearest existing items it must not copy, and the writing instruction for `--kind paper|talk|patent`. `patent` asks for an invention-disclosure title and abstract: the problem, what existing systems do, the mechanism, and one sentence on what an independent claim would cover.

5. **Write.** A person or a model drafts the candidate from the brief into a Markdown file: `# Title` on the first line, body after, then an optional `---` line and three footer lines (`Helps:`, `Barrier:`, `Standard:`) that verify.py records but does not embed.

6. **Verify**: `python -I uncharted/verify.py --name papers draft.md --gap <out>.json`
   Embeds the draft like a real item and reports nearest real items with similarities, the loneliness percentile, the neighborhood it landed in, similarity to the gap point and to the blend point, and a verdict: PASS when lonelier than the target and still on target, otherwise RETRY with what was too close. Results append to `draft.verify.json` so the attempts are kept.

7. **Report**: `python -I uncharted/report.py --name papers --out data/gaps/report-papers.md`
   One table per corpus: each idea's loneliest passing draft (its loneliest draft when none passed), drafts tried, nearest real item, similarity, loneliness percentile, verdict; calibration baselines listed separately; details and the value-check footer per idea, plus the items that earlier drafts of the idea landed near, since rewording moves a draft away from some neighbors without changing the idea. Nearest real items carry their link when the corpus has one (for patents, the nearest prior art on Google Patents).

8. **See the map**: `python -I uncharted/export_map.py --name papers [--gaps data/gaps/holes data/gaps/pairs]`
   Writes `viewer/data/<name>.js` and refreshes `viewer/data/index.js`, the list of exported corpora. Without `--gaps`, every gap JSON under `data/gaps` whose corpus matches `--name` is included. Then open `viewer/index.html` (see Map viewer).

## Map viewer

`viewer/index.html` draws every item of a corpus on the flat 2-D map, colored by neighborhood or by loneliness percentile, with the holes and gaps marked. Hover a point for its title and venue; click a point, a gap marker or a list entry to pin its details (year, venue, neighborhood, loneliness percentile, nearest-neighbor similarity, link to the source). Drag to pan, scroll or pinch to zoom; with the map focused, arrow keys pan and `+`, `-` and `0` zoom and reset. Pinning a gap dims the rest of the map, lights the neighborhoods it sits between, rings its must-not-copy items in cobalt with numbers that match the Details list, outlines a hole's ring items, and fits the view to them; opening one of those items keeps the gap in focus, and Clear or Esc restores the full map. A pinned gap also shows the drafts written for it and verified with verify.py: a filled square for a draft that passed, a hollow one for a draft that needs another try, and a Drafts list in Details; a draft opens to its verdict, the reasons it did not pass, and its nearest real items. The gap list, the neighborhood list and title search reach the same items without the canvas, so the map works by keyboard and with a screen reader. With more than one corpus exported, a switcher appears in the header; `?corpus=<name>` in the URL picks one. The legend sits in the map's bottom-left corner (under the map on phones, folded) and folds away; fitting the map keeps what it frames clear of it.

Open it straight from disk, with no server and no build step:

    open viewer/index.html

The exporter writes the data as a script that assigns one object (`window.UNCHARTED_DATA[name] = {...}`) rather than as `.json`, because a page opened from `file://` cannot `fetch` JSON but can load a script. Serving the folder works too: `python -I -m http.server 8000 --directory viewer`, then http://localhost:8000/?corpus=talks. Fonts come from Google Fonts; offline, the page falls back to system fonts.

What the marks mean:

- **Dot**: one item. Gray dots are unclustered (HDBSCAN noise). Colors repeat across the map: eight hues are assigned so that neighborhoods touching on the map differ, and the labels and the neighborhood list name them. Select a neighborhood in the list to isolate it.
- **Dashed ring with a dot** (H1, H2 …): a hole from `gaps.py holes`, drawn at the empty radius gaps.py measured.
- **Diamond** (P1 …): a gap between two related neighborhoods from `gaps.py pairs`.
- **Triangle** (B1 …): a blend pushed into a gap from `gaps.py blend`.

**Gap positions are approximate.** A gap point lives in the 1,024-dimension embedding space and has no position on the flat map. Holes are drawn where gaps.py found them on the map. Pairs and blends are drawn at the similarity-weighted mean of their must-not-copy items' map positions: that shows the neighborhood a gap sits in, not an exact spot, since UMAP would not place a new point the way it placed the corpus. Drafts are placed the same way, from their nearest real items, so a draft can sit well away from its gap when the items it resembles are spread across the map. export_map.py attaches a draft to the gap path verify.py recorded, or else to the gap file that shares its name (`x.cand2.md` → `x.json`); calibration baselines, and drafts whose gap was not exported, are left out, and the export log says which.

Colors: eight categorical hues and a seven-step blue ramp for loneliness (crowded items recede; items at the 95th percentile and up draw last, larger, with a ring, and the legend shows the same sizes), each with separate light and dark steps checked for color-vision deficiency and contrast against the page surface. The page follows the system light or dark setting. The styling is the Porcelain design system's stylesheet, copied into `viewer/porcelain.css`; refresh that file whole rather than editing it.

## Reading the numbers

Cosine similarities from this model sit in a narrow band, so always read them against the corpus's own distribution (`<name>.space.json`). For the talk corpus of 967 abstracts: a paraphrase of a real talk scores about 0.93 to its original, the median talk sits at 0.815 to its nearest neighbor, the loneliest 5% sit below 0.72, and the loneliest 2% below 0.70. A candidate that verifies at 0.70 is further from everything that exists than 98% of the talks actually given.

For the patent corpus of 72,389 abstracts the scale is different. A paraphrase of a real patent abstract scores 0.854 to its original, about the corpus median of 0.842, so distance alone cannot tell a reworded patent from an ordinary neighbor; the 90th percentile is 0.954 because continuation families whose abstracts differ by a few words survive the collapse. The loneliest 5% sit below 0.766 and the loneliest 1% below 0.733. More than half the corpus is generic interaction hardware (VR, gesture, haptics, voice assistants, from `G06F3/01` and `G06F3/16`), so `pairs` ranks gaps between unrelated hardware first; pass `--only` with the streaming and accessibility neighborhood numbers from the map for a ranking worth drafting from. Verification here is a prior-art pointer, not a novelty opinion: read the nearest patents' claims before treating a PASS as new.

Distance measures how unlike a thing is, not whether it is any good. Blends of unrelated ingredients produce text that is far from everything only because it says nothing (the brief warns when the least-related pair of ingredients is unusually far apart). Judge the survivors.

## Data sources

- **Patents**: PatentsView's final release of granted-patent tables, dated 2024-12-31 (grants 1976 through 2024), deposited on Zenodo at https://zenodo.org/records/15783125 under CC BY 4.0; credit PatentsView and the USPTO when results are shared. No key or account is needed. PatentsView's download page now redirects to the USPTO Open Data Portal, which requires a signed-in USPTO.gov account with multi-factor authentication; tables downloaded from there, or from a later release, go in the same directory and load the same way. Grants issued after 2024 are not in the Zenodo release, so prior-art checks against this corpus miss the newest two years.
- **Papers and talks**: Semantic Scholar's public API, OpenAlex, and public conference pages, as described under Load.

## Layout

    uncharted/      loaders, embed, space, gaps, verify (standalone scripts)
    data/corpora/   loader output (git-ignored)
    data/embeddings/ matrices, metadata, maps (git-ignored)
    data/gaps/      briefs, candidates, verification logs (git-ignored)
    data/raw/       cached fetched pages (git-ignored)
    viewer/         map viewer: index.html, map.js, porcelain.css
    viewer/data/    exported map data (git-ignored)
    logs/           run logs
