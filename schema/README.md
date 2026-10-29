# Uncharted map files

A map file holds one map: every item of a corpus placed by meaning, the neighborhoods the items form, candidate connections (places in the corpus where little has been written) and any drafts written for them. `uncharted/export.py` writes map files and the viewer reads them. Version 1 is defined by `uncharted-export-v1.schema.json` together with the rules on this page.

## The file

- One JSON document in UTF-8, plain or gzip-compressed. A reader detects gzip by its first two bytes (`1f 8b`), not by the file name.
- `format` is `"uncharted-export"` and `version` is `1`. A reader rejects any other format, and rejects any other version with a message naming the version it reads.
- Standard JSON only: no `NaN`, `Infinity` or comments. Every number must still be finite once parsed (a JavaScript reader checks `Number.isFinite`, because `1e400` parses to `Infinity`), and every integer must lie within ±(2^53 − 1), the range a JavaScript reader holds exactly.
- Producers write only the properties the schema defines, and readers reject any others. The exceptions are `manifest.map`, `manifest.candidates` and a candidate's `score`, which take extra named values (short scalars, or numbers in `score`), up to 50, 50 and 20 of them.

## Items are columns

`items` holds one array per field instead of one object per item: `items.title[i]`, `items.x[i]` and the rest describe item `i`. Every column has exactly `items.count` entries, and a map has at least two items. Storing items this way keeps a map of 72,000 patents to about 10 MB.

- `id` is the source's identifier (an OpenAlex work id, a patent number, `id24:2018:talk1`), unique in the file.
- `cluster` is the item's neighborhood id, or -1 for an item in no neighborhood.
- `venue` is a position in `lookups.venues`, or -1.
- `urlPrefix` and `url` make the link: `lookups.urlPrefixes[urlPrefix] + url`. With `urlPrefix` -1 the item has no link and `url` is empty.
- `snippet` holds the first words of the item's embedded text, when the exporter was asked for them.
- `source` is a position in `manifest.corpus.sources`, or -1.
- `year`, `venue`, `urlPrefix`, `url`, `snippet` and `source` are optional columns. `url` and `urlPrefix` come together or not at all.

A map file carries titles, links and at most a snippet, never whole abstracts: they would multiply the file's size, and the sources' text is not the exporter's to redistribute. The link leads to the full item.

## References

- Items are referred to by their position in the columns, counting from 0: `neighborhoods[].examples`, `candidates[].parents`, and the `[position, similarity]` pairs in each `nearest` list.
- A neighborhood's `id` equals its position in `neighborhoods`. `items.cluster` and `candidates[].neighborhoods` refer to it.
- Item ids, candidate ids, candidate codes and draft ids are each unique in the file. Draft codes are unique within their candidate.

## Coordinates and placement

`x` and `y` place things on the flat map. The exporter rescales the 2-D layout so its longer side runs from 0 to 1, keeping the aspect ratio; both axes start at 0. `y` increases upward, as the layout algorithm produces it, so a screen drawing flips it. A reader fits the view to the data rather than assuming either range is full.

The flat map is a projection for looking at. Distances on it are approximate, and every score below is measured in the full embedding space. Candidates and drafts have no true position on the map, so the exporter places them:

1. Take the first 8 entries of the `nearest` list (all of them when there are fewer).
2. Weight each by its similarity, floored at 0, plus 10^-9 so the weights never sum to zero.
3. The position is the weighted mean of those items' `x` and `y`, so it lies within the map.
4. The spread is the weighted root-mean-square distance of those items from that position, stored as `spread` on a candidate.
5. The same spread is computed for every real item from its own 8 nearest neighbors. When a candidate's or draft's spread is larger than the `manifest.candidates.placePct` percentile of those (recorded as `placeSpread`), the position means little: `x` and `y` are null, and the reader shows it only in lists.

## Scores

Similarities are cosine similarities between embeddings, from -1 to 1. Every value is computed at full precision and rounded only when written: similarities to 3 decimals, percentages to 1, coordinates to 4, and `pairP` to 3 significant figures. A value derived from others (such as `margin`) is computed before rounding, so it can differ from the rounded operands by 0.001.

- `nn1` is the similarity to the nearest real item. For an item, that is its nearest other item.
- `lonely` is the share of items whose nearest neighbor is closer than this one's, so higher is emptier. For an item, it is the share of the other items, n − 1 of them, whose `nn1` is higher; ties are not higher, so the loneliest item scores 100 only when no other item is as lonely. For a candidate or a draft, which is not part of the corpus, it is the share of all n items.
- A candidate's `score.nn1` and `score.lonely` measure the candidate point that way, against every real item.
- `stats.nn1Percentiles` gives the corpus's `nn1` percentiles as a yardstick, computed from unrounded values with linear interpolation: a 50th percentile of 0.70 means half the items have a neighbor at 0.70 or closer.
- Similarities depend on the embedding model. Maps made with different models or providers cannot be compared; `manifest.embedding` names the configuration and `manifest.corpus.fingerprint` identifies the exact corpus.

A connection also carries the evidence it was chosen on, measured on its mixture before the push into empty space. Mixing two items always lands somewhere fairly empty, and how empty depends on how similar the two parents are and how crowded their region is, so a mixture is compared only with random mixtures matched on both:

- `mixNn1`: the mixture's similarity to its nearest real item, leaving out its two parents and their nearest neighbors (`ringSize` of them each), which a mixture is close to by construction.
- `baseline`: the median `mixNn1` of the matched random mixtures. `margin` is `baseline` minus `mixNn1`, so above 0 means emptier than chance.
- `beats`: the share of the other mixtures in its matched group whose `mixNn1` is higher; ties are not higher.
- `pairTail`: the share of the random mixtures drawn from this pair of neighborhoods that fall in the emptiest `tailPct` percent of their matched groups. Around `tailPct` is ordinary; a pair at four times that is empty in a way its parents' similarity and crowding do not explain.
- `pairP`: the chance of a tail count at least this high if mixtures were assigned to pairs at random within each matched group, with the groups' tails held fixed: an exact test (a sum of hypergeometric counts), so small groups and ties are accounted for. It is not corrected for the number of pairs tested; `manifest.candidates.pairCutoff` is the corrected bar.
- `pairPParentsOnly`: the same test when a mixture's score leaves out only its parents. A pair that passes with the rings left out but not here may owe its emptiness to the rings hiding work that already joins the two neighborhoods.
- `parentsA` and `parentsB`: how many different parents from each neighborhood the pair's tail mixtures use. A pair must reach `minParents` on each side (or all of a smaller neighborhood) and keep any single parent under `maxBusiest` of its tail mixtures, so one unusual item cannot carry it.
- `mixNn1All`: the mixture's similarity to its nearest real item with nothing left out, usually one of its parents.

This is an exploratory, corpus-relative signal. A passing pair says the combination is emptier than comparable combinations in this corpus; the cause can be untried work, work missing from the corpus, thin abstracts, or a combination that makes no sense.

## Candidate kinds

- `connection`: a mixture of one item from each of two related neighborhoods, kept when that pair of neighborhoods is emptier than comparable pairs (`pairP` below `pairCutoff`, with enough different parents and none busiest), then pushed out to the nearest empty space as a writing target. The push is a step toward a proposal: the pair's evidence describes the mixtures before it, and `score.nn1` and `score.lonely` the point after it.
- `pair`: the midpoint of two related neighborhoods, pushed out the same way.
- `blend`: a mixture of chosen items or neighborhoods, pushed out the same way.

The word "candidate" is deliberate. An empty spot says nothing about whether an idea is good, and every candidate needs a person's judgment and a search beyond the corpus.

A file with connections records how they were found in `manifest.candidates`: `method` (the scoring method's name, `connections-v1`), `mixesPerPair`, `pairsTested`, `tailPct`, `pairCutoff` and `seed` always, and `band`, `ringSize`, `simBins`, `densityBins`, `test`, `alpha`, `minParents`, `maxBusiest` and `minTargetLonely` when they apply. Matched groups are the cross of `simBins` quantile bins of parent similarity and `densityBins` quantile bins of the parents' mean `nn1`, over every mixture from every related pair.

## Rules beyond the schema

`uncharted/schema.py` checks these after the schema, and a reader applies the same checks:

1. Every items column has `items.count` entries; `url` and `urlPrefix` come together.
2. Item ids are unique.
3. Every reference points at something that exists: `items.cluster` (or -1), `items.venue`, `items.source` and `items.urlPrefix` (or -1), neighborhood `examples`, candidate `neighborhoods` and `parents`, and every `nearest` entry.
4. A neighborhood's `id` equals its position, its `size` equals the number of items in it, and each of its `examples` is one of its items.
5. An item whose `urlPrefix` is -1 has an empty `url`. Every link, item links and `manifest.corpus.sources[].url` alike, parses as an `http` or `https` URL with a host and without a user name or password.
6. `stats` matches the file: items, neighborhoods, unclustered items, candidates and drafts. `stats.nn1Percentiles` never decreases from the 1st to the 99th.
7. A file holds at most 1,000 drafts in all.
8. Candidate ids and codes are unique; draft ids are unique; draft codes are unique within a candidate.
9. `x` and `y` are both numbers or both null, on candidates and drafts.
10. A `nearest` list names each item once, with similarities that never increase down the list.
11. A candidate lists each neighborhood once. A `pair` or `connection` joins exactly two neighborhoods; a `connection` mixes exactly two parents and reports `mixNn1`, `baseline`, `margin`, `beats`, `pairTail` and `pairP`. A file with a connection records the method keys above.
12. A draft that passed has no reasons; a draft to retry has at least one.
13. Dates and timestamps are real calendar dates and times.

## Text and links are untrusted

A map file can come from anyone. Readers treat every string as plain text, inserted with `textContent` or an equivalent and never parsed as HTML. `brief` and draft `text` use light Markdown (headings, lists, bold): a reader may show them as plain text, or render that subset with any raw HTML shown as text.

One policy covers every link in the file. Join prefix and rest first, parse the result with `URL` and no base, and open it only when parsing succeeds, the scheme is `http:` or `https:`, there is a host, and there is no user name or password. Assign it through DOM properties, and open it with `rel="noopener noreferrer"`. Never fetch a link to preview it. These checks stop executable links; they do not vouch for where a link leads. `examples/hostile-text.json` puts markup and script into every free-text field to test all of this.

## Limits

The schema caps every list and string: 500,000 items, 2,000 neighborhoods, 500 candidates, 20 drafts per candidate (and 1,000 in all), 1,000 characters for a title, 50,000 for a brief or a draft. Lengths count Unicode code points, so a JavaScript reader counts with `[...text].length` rather than `text.length`.

A reader enforces a byte cap before it parses anything: it refuses a file over 200 MB on disk, and stops decompressing once the JSON passes 200 MB, so a small gzip file cannot expand without bound. `uncharted/schema.py` applies the same cap. The schema nests no deeper than eight levels, so a reader may reject anything deeper on sight. A browser reader should validate off the main thread and draw large collections incrementally.

## Dates and times

`exported` and a draft's `checked` are RFC 3339 timestamps with seconds and an offset (`2026-10-28T21:14:03Z`, `2026-10-28T15:14:03-06:00`). A source's `retrieved` is a timestamp or a calendar date (`2026-10-24`).

## Validating

    python -I uncharted/schema.py <file>...

checks a file against the schema and every rule above. The schema uses only these JSON Schema 2020-12 keywords, so a reader can interpret it directly without a library: `type`, `const`, `enum`, `required`, `properties`, `additionalProperties`, `propertyNames`, `maxProperties`, `items`, `prefixItems`, `minItems`, `maxItems`, `minimum`, `maximum`, `minLength`, `maxLength`, `pattern`, `anyOf`, and `$ref` to a local JSON Pointer (`#/$defs/...`). `title` and `description` are notes. Patterns are ECMA-262 regular expressions that use only literal characters, explicit ranges and anchors (no `\d`, `\w` or `\s`), so JavaScript and Python read them the same way; `$` means the end of the text.

## Examples

- `examples/small.json`: a complete, valid map of twelve invented talks. It has three neighborhoods, two unclustered items, one candidate of each kind (one of them off the map), two drafts (a pass and a retry), every optional column, and items without a year, a venue or a link.
- `examples/hostile-text.json`: valid, with markup, script and control characters in every free-text field.
- `examples/invalid/`: each file breaks one rule. `examples/expected-errors.json` gives a phrase that `schema.py`'s first error for each file contains; a reader's own messages may differ, but it must reject every one.

Regenerate the examples with `python -I schema/examples/make_examples.py`.

## Changing the format

Version 1 may still change until the first public release of Uncharted. From then on it is frozen. Readers reject properties they do not know, so any change an existing reader would reject, an added property included, makes a new version with its own schema file; readers keep accepting every version they know.
