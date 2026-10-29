# uncharted

Find the empty places in a field and write what would live there.

The method: embed every item in a corpus as a point on a map of meaning, find the regions where nothing exists yet (holes between neighborhoods, or the emptiest spot near a blend of related items), have a writer draft the item that would sit there, then embed the draft exactly like a real item and measure where it landed. If it is too close to something that exists, the measurements say what, and the writer tries again. Every surviving candidate carries a number: how similar it is to the closest real thing, and what share of the corpus is more crowded than it.

The pipeline is domain-agnostic. A domain is a loader that emits the shared record shape and a one-paragraph instruction saying what kind of artifact to write (paper abstract, talk abstract, invention disclosure, component spec). Everything runs locally; no text leaves the machine.

## Setup

Python 3.11 or later. Dependencies are declared in `pyproject.toml` and locked in `uv.lock`. With [uv](https://docs.astral.sh/uv/):

    uv sync --extra local

The `local` extra adds PyTorch and sentence-transformers, for embedding on this machine. Run scripts as `uv run python -I uncharted/<script>.py ...`, or activate `.venv` and use `python -I` as the examples below do. Without uv, `python -m venv .venv && .venv/bin/pip install -r requirements.txt` installs the same packages, unpinned.

The first embedding run downloads the model into the Hugging Face cache: `BAAI/bge-large-en-v1.5` by default (about 1.3 GB), which embeds about 25 abstracts a second in float16 on an M3 Pro.

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

3. **Map**: `python -I uncharted/space.py --name papers [--min-cluster-size N] [--min-samples 5]`
   Top-10 neighbors per item, loneliness percentiles, HDBSCAN neighborhoods on a 10-d UMAP, a 2-d UMAP for the flat map, per-neighborhood labels (distinctive title terms and tags), and a console report: biggest and most popular neighborhoods, tightest and loosest, loneliest items, most cloned items. `--min-samples` stays small (default 5) because HDBSCAN's own default, equal to the cluster size, can fold a whole map into one neighborhood. `<name>.space.json` records the settings for the map file's manifest.

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

9. **Export a map file**: `python -I uncharted/export.py --name talks-q --title "Accessibility conference talks" --corpus-info data/corpora/talks.corpus.json --candidates data/gaps/talks-q-pairs --out data/exports/talks.json`
   Writes one JSON map file (format `uncharted-export`, version 1; an `--out` ending in `.gz` is compressed): the items, neighborhoods and candidates, each candidate's brief and drafts inline, and a manifest of how the map was made, with the sources and cleaning rules taken from `--corpus-info`. A draft goes into the file only while its text and the corpus are what verify.py measured; `--snippet-words N` adds each item's first N words. The file is validated before it is written, and `python -I uncharted/schema.py <file>` checks any map file. The format, its rules and example files are in `schema/README.md`. This is the file the version 1 viewer imports; until then, step 8 feeds the current viewer.

## Map viewer

`viewer/` is a static site with no build step and no server-side code. It opens Uncharted map files (version 1, defined in `schema/README.md`) in the browser.

- **Opening a map.** "Open a map file", or drop a file onto the page: plain `.json` or `.json.gz`. The file is read in the browser and never uploaded. A Web Worker checks it against the schema and the rules beyond it (the checks `uncharted/schema.py` runs, ported to `viewer/mapfile.js`) under the 200 MB cap, counted while decompressing. A file that fails shows its first problem and the open map stays; a second map that passes replaces the first.
- **Samples.** `viewer/samples/` holds the built-in samples, opened with "Load the sample" or `?sample=<id>` and checked the same way. Today that is `small.json`, the invented twelve-talk map from `schema/examples` (CC0), which a test keeps identical to its source.
- **What it shows.** Items are colored by neighborhood or by loneliness percentile. Candidate connections are hollow shapes (a circle with a dot for a connection, a diamond for a pair, a triangle for a blend) at their placement, gathered into a numbered circle where they overlap; a candidate whose `x` and `y` are null appears only in the list. Pinning a candidate dims the rest, lights its neighborhoods, rings its nearest items in numbered cobalt (the must-not-copy list in Details), outlines the items it was built from, shows its drafts (a filled square passed, a hollow one needs another try) and fits the view. Opening one of its items keeps it in focus, with previous, next and a way back; Esc steps back one level. The address carries the pinned candidate's code (`#C1`). A connection's Details show the evidence it was chosen on, including how its pair test passed: a pair that passes only with its parents' nearest neighbors left out (`pairPParentsOnly` at or above `manifest.candidates.pairCutoff`) reads "weaker evidence" in the list and Details. A score number the viewer has no label for shows under its own name. "About this map" shows the manifest: sources and retrieval dates, the embedding model, provider and dimensions, since maps from different models cannot be compared.
- **Without the canvas.** The candidate, draft, neighborhood and title-search lists reach everything on the map. Every section has a "?" with a short explanation. Search pages through every match, 25 at a time. With the map focused, arrow keys pan, `+` and `-` zoom, `0` fits the map, and N and P step through candidates.
- **Untrusted text.** Every string goes in as text, never markup. A link is joined from the prefix table, parsed with `URL`, and opened only when it is http or https with a host and no credentials. A content security policy (in `index.html` and `vercel.json`) allows no inline script or style.

Run it locally from a web server, since a page opened from disk cannot start the checking worker or fetch a sample:

    python -I -m http.server 8000 --directory viewer

then open http://localhost:8000/ or http://localhost:8000/?sample=small.

**Tests.** `node --test tests/viewer/*.test.mjs` checks the reader and validator against every file in `schema/examples`, with no packages. `node tests/viewer/e2e.mjs` drives a browser through every example, the hostile one, the keyboard route, the performance budgets and screenshots. It needs `playwright-core` (`PLAYWRIGHT_CORE=<folder>` when installed elsewhere; `UNCHARTED_BROWSER_CHANNEL=chrome` uses Google Chrome), `UNCHARTED_PERF_MAP=<a large map>` for the budgets, and `SCREENSHOT_DIR=<folder>` for screenshots.

**Performance budgets**, against the largest map (about 72,000 items): opening a file until the map is drawn and the lists are filled within 3 s on a desktop and 8 s on a CPU throttled four times at phone size; reading and validating within 8 s throttled; a redraw within 50 ms on a desktop and 120 ms throttled. On an Apple M3 with a 72,389-item map: 0.27 s and 0.47 s to open, 0.53 s to read and validate throttled, 17 ms and 34 ms a redraw.

**Deploying.** On Vercel, set the project's Root Directory to `viewer`, the framework to Other, and leave the build command and output directory empty. `viewer/vercel.json` adds the security headers.

Positions are approximate. The flat map is a projection; candidates and drafts are placed at the weighted mean of their nearest items (schema/README.md > Coordinates and placement), so a draft can sit well away from its candidate when the items it resembles are spread across the map. Every number in Details is measured in the full embedding space.

Colors: eight categorical hues, chosen per neighborhood by the file's `slot` so that neighborhoods touching on the map differ, and a seven-step blue ramp for loneliness (crowded items recede; items at the 95th percentile and up draw last, larger and ringed, and the legend shows the same sizes). Each has separate light and dark steps checked for color-vision deficiency and contrast against the page. The page follows the system's light or dark setting. The styling is the Porcelain design system's stylesheet, copied into `viewer/porcelain.css`; refresh that file whole rather than editing it.

## Reading the numbers

Cosine similarities from this model sit in a narrow band, so always read them against the corpus's own distribution (`<name>.space.json`). For the talk corpus of 967 abstracts: a paraphrase of a real talk scores about 0.93 to its original, the median talk sits at 0.815 to its nearest neighbor, the loneliest 5% sit below 0.72, and the loneliest 2% below 0.70. A candidate that verifies at 0.70 is further from everything that exists than 98% of the talks actually given.

For the patent corpus of 72,389 abstracts the scale is different. A paraphrase of a real patent abstract scores 0.854 to its original, about the corpus median of 0.842, so distance alone cannot tell a reworded patent from an ordinary neighbor; the 90th percentile is 0.954 because continuation families whose abstracts differ by a few words survive the collapse. The loneliest 5% sit below 0.766 and the loneliest 1% below 0.733. More than half the corpus is generic interaction hardware (VR, gesture, haptics, voice assistants, from `G06F3/01` and `G06F3/16`), so `pairs` ranks gaps between unrelated hardware first; pass `--only` with the streaming and accessibility neighborhood numbers from the map for a ranking worth drafting from. Verification here is a prior-art pointer, not a novelty opinion: read the nearest patents' claims before treating a PASS as new.

Distance measures how unlike a thing is, not whether it is any good. Blends of unrelated ingredients produce text that is far from everything only because it says nothing (the brief warns when the least-related pair of ingredients is unusually far apart). Judge the survivors.

## Data sources

- **Patents**: PatentsView's final release of granted-patent tables, dated 2024-12-31 (grants 1976 through 2024), deposited on Zenodo at https://zenodo.org/records/15783125 under CC BY 4.0; credit PatentsView and the USPTO when results are shared. No key or account is needed. PatentsView's download page now redirects to the USPTO Open Data Portal, which requires a signed-in USPTO.gov account with multi-factor authentication; tables downloaded from there, or from a later release, go in the same directory and load the same way. Grants issued after 2024 are not in the Zenodo release, so prior-art checks against this corpus miss the newest two years.
- **Papers and talks**: Semantic Scholar's public API, OpenAlex, and public conference pages, as described under Load.

## Layout

    uncharted/      loaders, embed, space, gaps, verify, export, schema (standalone scripts)
    schema/         map file format: JSON Schema, rules, example files
    tests/          python -I -m unittest discover -s tests
    data/corpora/   loader output (git-ignored)
    data/embeddings/ matrices, metadata, maps (git-ignored)
    data/gaps/      briefs, candidates, verification logs (git-ignored)
    data/raw/       cached fetched pages (git-ignored)
    data/exports/   map files (git-ignored)
    viewer/         map viewer: index.html, map.js, porcelain.css
    viewer/data/    exported map data (git-ignored)
    logs/           run logs
