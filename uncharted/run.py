"""Run an approved scope end to end: pull, clean, embed, map, find candidate connections, export.

    python -I uncharted/run.py --yes --scope data/runs/haptics/scope.json

The scope file comes from estimate.py and carries the caps the person approved: items,
tokens and dollars. run.py never asks anything, and --yes says the approval happened. Before
pulling it counts the works again and stops for a new estimate when the count moved more than
--max-drift. Every stage records its completion in data/runs/<name>/state.json, so running the
same command again resumes an interrupted run at the stage it stopped in.

Exit status: 0 done; 3 a cap or the scope changed, so a new estimate and approval are needed;
4 a provider refused (key, quota or input); 5 paused because OpenAlex's daily budget is spent
(run the same command tomorrow); 2 a usage error; 1 anything else.

Everything lands under data/ (git-ignored): corpora/<name>.jsonl, <name>.clean.jsonl and
<name>.corpus.json, embeddings/<name>.*, gaps/<name>-connections/, the map file
exports/<name>.json.gz, and the run's record runs/<name>/state.json.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # `python -I` leaves the script's directory off sys.path
import openalex  # noqa: E402

STAGES = ("count", "pull", "clean", "embed", "map", "candidates", "export")
MIN_ITEMS = 100


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


class Stop(Exception):
    def __init__(self, message: str, code: int):
        super().__init__(message)
        self.code = code


def now() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def script(name: str, *args: str) -> list[str]:
    return [sys.executable, "-I", os.path.join(HERE, name), *args]


def call(cmd: list[str], what: str) -> int:
    log(f"{what}: {' '.join(os.path.basename(c) if i < 3 else c for i, c in enumerate(cmd))}")
    return subprocess.run(cmd).returncode


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scope", required=True, help="scope file written by estimate.py --scope-out")
    ap.add_argument("--yes", action="store_true", help="the person approved this scope and its caps")
    ap.add_argument("--data", default="data", help="data folder")
    ap.add_argument("--max-drift", type=float, default=0.02, help="largest change in the work count since the estimate, as a share")
    ap.add_argument("--api-key", default=os.environ.get("OPENALEX_API_KEY"), help="OpenAlex key (or set OPENALEX_API_KEY)")
    args = ap.parse_args()
    if not args.yes:
        print("run.py runs only an approved scope: show the estimate, get a yes, then pass --yes", file=sys.stderr)
        return 2
    with open(args.scope, encoding="utf-8") as fh:
        scope = json.load(fh)
    if scope.get("uncharted_scope") != 1:
        print(f"{args.scope} is not a scope file from estimate.py", file=sys.stderr)
        return 2
    name, caps, config, source = scope["name"], scope["caps"], scope["config"], scope["source"]
    d = args.data
    corpus, clean_corpus, info = (os.path.join(d, "corpora", f"{name}{ext}") for ext in (".jsonl", ".clean.jsonl", ".corpus.json"))
    gaps_dir = os.path.join(d, "gaps", f"{name}-connections")
    export = os.path.join(d, "exports", f"{name}.json.gz")
    run_dir = os.path.join(d, "runs", name)
    os.makedirs(run_dir, exist_ok=True)
    state_path = os.path.join(run_dir, "state.json")
    state = {"scope": os.path.abspath(args.scope), "approved": now(), "caps": caps, "stages": {}, "spent": {"openalex": 0.0, "embedding": 0.0, "tokens": 0}}
    if os.path.exists(state_path):
        with open(state_path, encoding="utf-8") as fh:
            state = json.load(fh)
        if state.get("caps") != caps:
            log("the scope's caps changed since this run started; the new caps apply")
            state["caps"] = caps

    def save() -> None:
        tmp = state_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=1)
        os.replace(tmp, state_path)

    def done(stage: str, **extra) -> None:
        state["stages"][stage] = {"done": now(), **extra}
        save()

    per_call = openalex_cost = 0.0
    try:
        if "count" not in state["stages"]:
            count = openalex.count_works(source["filter"], args.api_key)
            approved = int(source["count"])
            drift = abs(count - approved) / max(approved, 1)
            if drift > args.max_drift and not source.get("maxItems"):
                raise Stop(f"{count:,} works match now against {approved:,} when estimated; run estimate.py again and get a new approval", 3)
            done("count", works=count)
            log(f"{count:,} works match the approved filter (estimated {approved:,})")

        per_call = 0.001 if ".search" in source["filter"] else 0.0001
        if "pull" not in state["stages"]:
            code = call(script("openalex.py", "--out", corpus, "--corpus", name, "--filter", f"{name}={source['filter']}", "--max-items", str(caps["items"]), "--resume", "--corpus-info", info), "pull")
            pull_state = corpus + ".state.json"
            if os.path.exists(pull_state):
                with open(pull_state, encoding="utf-8") as fh:
                    pages = sum(st.get("pages", 0) for st in json.load(fh).values())
                state["spent"]["openalex"] = round(pages * per_call, 4)
                save()
                raise Stop("OpenAlex stopped serving pages, usually because today's budget is spent; run the same command tomorrow to continue", 5)
            if code != 0 or not os.path.exists(corpus):
                raise Stop(f"the pull failed (exit {code})", 1)
            with open(corpus, encoding="utf-8") as fh:
                pulled = sum(1 for _ in fh)
            state["spent"]["openalex"] = round((pulled / 200 + 1) * per_call, 4)
            done("pull", items=pulled)
        openalex_cost = state["spent"]["openalex"]

        if "clean" not in state["stages"]:
            if call(script("clean.py", corpus, "--out", clean_corpus, "--corpus-info", info), "clean") != 0:
                raise Stop("cleaning failed", 1)
            with open(clean_corpus, encoding="utf-8") as fh:
                kept = sum(1 for _ in fh)
            if kept < MIN_ITEMS:
                raise Stop(f"only {kept} items survive cleaning: too few for a map", 2)
            done("clean", items=kept)

        if "embed" not in state["stages"]:
            cmd = script("embed.py", clean_corpus, "--name", name, "--out-dir", os.path.join(d, "embeddings"), "--provider", config["provider"], "--max-seq", str(config.get("maxSeq", 1024)), "--resume")
            if config.get("model"):
                cmd += ["--model", config["model"]]
            if config["provider"] != "local":
                cmd += ["--dimensions", str(config.get("dimensions", 1024)), "--max-dollars", f"{max(caps['dollars'] - openalex_cost, 0):.4f}"]
                if caps.get("tokens"):
                    cmd += ["--max-tokens", str(caps["tokens"])]
            code = call(cmd, "embed")
            if code in (3, 4):
                raise Stop("embedding stopped at a cap" if code == 3 else "the embedding provider refused the run", code)
            if code != 0:
                raise Stop(f"embedding failed (exit {code})", 1)
            with open(os.path.join(d, "embeddings", f"{name}.model.json"), encoding="utf-8") as fh:
                model_info = json.load(fh)
            state["spent"]["embedding"] = model_info.get("dollars", 0.0)
            state["spent"]["tokens"] = model_info.get("tokens", 0)
            done("embed", seconds=model_info.get("seconds"))

        if "map" not in state["stages"]:
            if call(script("space.py", "--name", name, "--dir", os.path.join(d, "embeddings")), "map") != 0:
                raise Stop("mapping failed", 1)
            done("map")

        if "candidates" not in state["stages"]:
            if call(script("gaps.py", "connections", "--name", name, "--dir", os.path.join(d, "embeddings"), "--kind", scope.get("kind", "paper"), "--briefs", "--out-dir", gaps_dir), "candidates") != 0:
                raise Stop("the candidate search failed", 1)
            done("candidates")

        if "export" not in state["stages"]:
            cmd = script("export.py", "--name", name, "--dir", os.path.join(d, "embeddings"), "--id", name, "--title", scope["title"], "--corpus-info", info, "--candidates", gaps_dir, "--out", export)
            if call(cmd, "export") != 0:
                raise Stop("the export failed", 1)
            done("export", file=os.path.abspath(export))
    except Stop as stop:
        state["stopped"] = {"at": now(), "reason": str(stop), "code": stop.code}
        save()
        log(f"stopped: {stop}")
        return stop.code
    except openalex.BudgetExhausted as err:
        state["stopped"] = {"at": now(), "reason": f"OpenAlex refused the count: {err}", "code": 5}
        save()
        log(f"stopped: OpenAlex refused the request ({err})")
        return 5
    state.pop("stopped", None)
    save()
    spent = state["spent"]
    log(f"done: {os.path.abspath(export)}; spent ${spent['openalex'] + spent['embedding']:.4f} (OpenAlex ${spent['openalex']:.4f}, embedding ${spent['embedding']:.4f}) of the ${caps['dollars']:.2f} approved")
    return 0


if __name__ == "__main__":
    sys.exit(main())
