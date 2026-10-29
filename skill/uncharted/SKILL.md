---
name: uncharted
description: Map a research field by meaning and find candidate connections nobody has written about yet, with a cost and time estimate the person approves before anything runs. Use when someone wants to map a topic or a field's literature, look for unexplored combinations or gaps in research, estimate what such a map would cost, or draft and verify ideas from an Uncharted map. Runs the Uncharted pipeline on this machine and hands back one map file to open in the viewer.
---

# Uncharted

Uncharted pulls a field's papers from OpenAlex, embeds every title and abstract as a point in a map of meaning, groups the points into neighborhoods, and looks for pairs of related neighborhoods whose combinations are emptier than chance: candidate connections. The result is one map file the person opens in the Uncharted viewer. Nothing is uploaded anywhere; the file stays on this machine.

An empty spot is not a good idea by itself. Say so when presenting results: every candidate needs the person's judgment and a search beyond the corpus.

## Ground rules

- Never run `run.py` before the person has seen the estimate and said yes to it. The caps in the scope file are what they approve.
- API keys come from the environment, never from the chat. If a key is missing, tell the person which variable to set and where (their shell profile, or the `env` block of their Claude Code settings), and wait. Never print a key.
- Never commit, push or upload anything, and never edit the repository's code.
- Long commands run in the background; give a one-line progress note at each stage.

## Find the repository

This file lives at `<repository>/skill/uncharted/SKILL.md`, usually reached through a symlink in `~/.claude/skills/`. Resolve the real path and run every command from the repository root:

    REPO="$(cd "$(dirname "$(realpath "<this skill's base directory>/SKILL.md")")/../.." && pwd)"
    cd "$REPO"

Check `uv --version`. Without uv, point the person to https://docs.astral.sh/uv/ and stop. The first `uv run` installs the Python packages into `$REPO/.venv`; with `--extra local` that includes PyTorch and, on the first embedding, a 1.2 GB model download. Say so before the first run.

## 1. The topic

Ask what field to map, in their words. Then list matching OpenAlex topics:

    uv run python -I uncharted/estimate.py topics "<their phrase>"

Show the topics as a short list (name, field, works) and ask which one to use. If none fits, offer a title-and-abstract search instead (`--search "<words>"` in step 3), which costs ten times as much per page. `OPENALEX_API_KEY` must be set for anything beyond a tiny map; a free key comes from https://openalex.org/settings/api.

## 2. Where the embedding runs

Ask, with these options:

- **On this computer** (`local`): free, private, slower (roughly 6 to 14 papers a second on Apple silicon, much slower on a CPU), needs the `local` extra.
- **OpenAI** (`openai`): fast, costs cents to a few dollars, needs `UNCHARTED_OPENAI_API_KEY` (or `OPENAI_API_KEY`) in the environment.
- **Voyage** (`voyage`): experimental, needs `VOYAGE_API_KEY`.

Maps made with different providers cannot be compared with each other.

## 3. The estimate

Pick a short lowercase name for the run, then:

    uv run python -I uncharted/estimate.py scope --topic <id> --name <name> --provider <provider> --scope-out data/runs/<name>/scope.json

Add `--years 2015-2026` or `--max-items 20000` (most-cited first) when the person wants a narrower map; suggest `--max-items` when more than 100,000 works match. Present the JSON as a few lines: works matched and items expected, the dollar range (OpenAlex plus embedding), the minutes per stage and in total, the caps, and every warning. Then ask plainly whether to run it with those caps. Anything but a clear yes means no.

## 4. The run

    uv run [--extra local] python -I uncharted/run.py --yes --scope data/runs/<name>/scope.json

Use `--extra local` for the local provider. Run it in the background and report each stage. Exit status:

- 0: done; the map file is `data/exports/<name>.json.gz`.
- 3: a cap would be crossed or the field's count changed. Run the estimate again, show it, and ask again.
- 4: the embedding provider refused: a missing or wrong key, or no credit. Say which, and stop.
- 5: OpenAlex's daily budget is spent. The same command resumes tomorrow.

Rerunning the same command after any stop resumes where it stopped.

## 5. What came back

Read `data/gaps/<name>-connections/pairs.json` and the candidate briefs in that folder. Give the person the map file's full path and tell them how to open it: in the Uncharted viewer (the hosted copy linked in the README, or `viewer/index.html` from this repository), choose Open a map file. Summarize the strongest candidate connections in a sentence each, naming the two neighborhoods and the nearest existing work.

Then offer to draft the top three in this session.

## 6. Drafting candidates, on request

For each candidate the person picks:

1. Read its brief, `data/gaps/<name>-connections/connection-NN-….md`. It says what to write, the target, the ingredients and what not to copy.
2. Write the draft to `connection-NN-….cand1.md` beside it: `# Title` on the first line, then the body, then a line containing only `---` and three lines starting `Helps:`, `Barrier:` and `Standard:`.
3. Verify it:

       uv run [--extra local] python -I uncharted/verify.py --name <name> --gap data/gaps/<name>-connections/connection-NN-….json data/gaps/<name>-connections/connection-NN-….cand1.md

4. On RETRY, read the reasons (the item it is too close to, or how far off target it is), revise into `.cand2.md`, and verify again. Stop after three tries and report the best.

Then write `data/gaps/<name>-connections/labels.json`, mapping each candidate's file name without `.json` to a label of at most 60 characters that names the idea, and export again with the drafts and labels:

    uv run python -I uncharted/export.py --name <name> --id <name> --title "<title>" --corpus-info data/corpora/<name>.corpus.json --candidates data/gaps/<name>-connections --labels data/gaps/<name>-connections/labels.json --out data/exports/<name>.json.gz

Show each draft's verdict, its loneliness and its nearest real item, and remind the person that a pass means far from what exists in this corpus, not new to the world.
