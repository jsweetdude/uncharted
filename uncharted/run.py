"""Run an approved scope end to end: pull, clean, embed, map, find candidate connections, export.

    python -I uncharted/run.py --yes --scope data/runs/haptics/scope.json

The scope file comes from estimate.py and carries the caps the person approved: works pulled,
tokens, and dollars for OpenAlex and for embedding. run.py never asks anything, and --yes says
the approval happened. Each stage records its completion in data/runs/<name>/state.json, so
running the same command again resumes at the stage that stopped; the state is bound to the
scope it was made for, and a lock keeps two runs from sharing it.

Until the pull completes, every start counts the works again. An uncapped scope stops for a
new estimate when the count moved more than --max-drift; a capped scope pulls its most-cited
works whatever the total, so a change in the total is reported, not stopped on.

Exit status: 0 done; 3 a cap would be crossed or the scope changed, so a new estimate and
approval are needed; 4 the embedding provider refused (key, quota or input); 5 OpenAlex
refused a page, usually because the day's budget is spent (run the same command tomorrow);
2 a usage error; 1 anything else.

Everything lands under data/ (git-ignored): corpora/<name>.jsonl, <name>.clean.jsonl,
<name>.corpus.json and <name>.jsonl.pull.json, embeddings/<name>.*, gaps/<name>-connections/,
the map file exports/<name>.json.gz, and the run's record runs/<name>/state.json.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # `python -I` leaves the script's directory off sys.path
import openalex  # noqa: E402

MIN_ITEMS = 100
NAME = re.compile(r"[a-z0-9][a-z0-9-]{0,63}")


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


class Stop(Exception):
    def __init__(self, message: str, code: int):
        super().__init__(message)
        self.code = code


def money(dollars: float) -> str:
    return f"${dollars:.2f}" if dollars >= 1 else f"${dollars:.4f}"


def now() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def script(name: str, *args: str) -> list[str]:
    return [sys.executable, "-I", os.path.join(HERE, name), *args]


def check_caps(caps: dict) -> None:
    def number(key: str, integer: bool = False, optional: bool = False) -> None:
        value = caps.get(key)
        if value is None and optional:
            return
        ok = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= (1 if integer and key == "items" else 0)
        if not ok or (integer and value != int(value)):
            raise Stop(f"the scope's {key} cap is {value!r}, which is not a valid cap", 2)

    number("items", integer=True)
    number("tokens", integer=True, optional=True)
    for key in ("openalexDollars", "embeddingDollars", "dollars"):
        number(key)


def scope_fingerprint(scope: dict) -> str:
    core = {"source": {k: scope["source"].get(k) for k in ("filter", "maxItems", "order")}, "config": scope["config"], "kind": scope.get("kind")}
    return hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest()[:16]


def take_lock(path: str) -> None:
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            with open(path, encoding="utf-8") as fh:
                pid = int(fh.read().strip() or 0)
            os.kill(pid, 0)
        except (ValueError, ProcessLookupError, PermissionError, OSError):
            os.remove(path)  # left by a run that is no longer alive
            return take_lock(path)
        raise Stop(f"another run of this scope is active (process {pid})", 2) from None
    with os.fdopen(fd, "w") as fh:
        fh.write(str(os.getpid()))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scope", required=True, help="scope file written by estimate.py --scope-out")
    ap.add_argument("--yes", action="store_true", help="the person approved this scope and its caps")
    ap.add_argument("--data", default="data", help="data folder")
    ap.add_argument("--max-drift", type=float, default=0.02, help="largest change in an uncapped scope's work count since the estimate, as a share")
    ap.add_argument("--api-key", default=os.environ.get("OPENALEX_API_KEY"), help="OpenAlex key (or set OPENALEX_API_KEY)")
    args = ap.parse_args()
    if not args.yes:
        print("run.py runs only an approved scope: show the estimate, get a yes, then pass --yes", file=sys.stderr)
        return 2
    try:
        with open(args.scope, encoding="utf-8") as fh:
            scope = json.load(fh)
        if scope.get("uncharted_scope") != 1:
            raise Stop(f"{args.scope} is not a scope file from estimate.py", 2)
        name, caps, config, source = scope["name"], scope["caps"], scope["config"], scope["source"]
        if not NAME.fullmatch(str(name)):
            raise Stop(f"the scope's name {str(name)[:40]!r} is not lowercase letters, digits and hyphens", 2)
        check_caps(caps)
    except Stop as stop:
        print(stop, file=sys.stderr)
        return stop.code
    except (OSError, json.JSONDecodeError, KeyError) as exc:
        print(f"cannot read the scope file: {exc}", file=sys.stderr)
        return 2

    d = args.data
    corpus, clean_corpus, info = (os.path.join(d, "corpora", f"{name}{ext}") for ext in (".jsonl", ".clean.jsonl", ".corpus.json"))
    embeddings = os.path.join(d, "embeddings")
    gaps_dir = os.path.join(d, "gaps", f"{name}-connections")
    export = os.path.join(d, "exports", f"{name}.json.gz")
    run_dir = os.path.join(d, "runs", name)
    os.makedirs(run_dir, exist_ok=True)
    state_path = os.path.join(run_dir, "state.json")
    lock = os.path.join(run_dir, "lock")
    env = dict(os.environ)
    if args.api_key:
        env["OPENALEX_API_KEY"] = args.api_key  # through the environment, never on a command line
    fingerprint = scope_fingerprint(scope)
    state = {"scope": os.path.abspath(args.scope), "fingerprint": fingerprint, "approved": now(), "caps": caps, "stages": {}, "spent": {"openalex": 0.0, "embedding": 0.0, "tokens": 0}}

    def save() -> None:
        tmp = state_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=1)
        os.replace(tmp, state_path)

    def done(stage: str, **extra) -> None:
        state["stages"][stage] = {"done": now(), **extra}
        save()

    def call(cmd: list[str], what: str) -> int:
        log(f"{what}: {' '.join(os.path.basename(c) if i < 3 else c for i, c in enumerate(cmd))}")
        return subprocess.run(cmd, env=env).returncode

    try:
        take_lock(lock)
    except Stop as stop:
        log(f"stopped: {stop}")
        return stop.code
    try:
        if os.path.exists(state_path):
            with open(state_path, encoding="utf-8") as fh:
                saved = json.load(fh)
            if saved.get("fingerprint") != fingerprint:
                raise Stop(f"{run_dir} belongs to a different scope; pick another name, or delete that folder to start over", 2)
            state = saved
            if state.get("caps") != caps:
                spent = state["spent"]
                if spent["openalex"] > caps["openalexDollars"] or spent["embedding"] > caps["embeddingDollars"] or (caps.get("tokens") is not None and spent["tokens"] > caps["tokens"]):
                    raise Stop("the scope's new caps are below what this run has already spent", 3)
                log("the scope's caps changed since this run started; the new caps apply")
                state["caps"] = caps

        if "pull" not in state["stages"]:
            count = openalex.count_works(source["filter"], args.api_key)
            approved = int(source["count"])
            drift = abs(count - approved) / max(approved, 1)
            if drift > args.max_drift:
                if not source.get("maxItems"):
                    raise Stop(f"{count:,} works match now against {approved:,} when estimated; run estimate.py again and get a new approval", 3)
                log(f"{count:,} works match now against {approved:,} when estimated; the cap of {source['maxItems']:,} most-cited works still applies")
            state["count"] = {"works": count, "at": now()}
            save()
            cmd = script("openalex.py", "--out", corpus, "--corpus", name, "--filter", f"{name}={source['filter']}", "--max-items", str(caps["items"]),
                         "--max-dollars", f"{max(caps['openalexDollars'] - 0.0001, 0):.6f}", "--resume", "--corpus-info", info)
            if source.get("order") == "cited":
                cmd += ["--order", "cited"]
            code = call(cmd, "pull")
            ledger_path, state_file = corpus + ".pull.json", corpus + ".state.json"
            source_of_truth = ledger_path if code == 0 else state_file
            if os.path.exists(source_of_truth):
                with open(source_of_truth, encoding="utf-8") as fh:
                    ledger = json.load(fh)
                ledger = ledger if code == 0 else ledger.get("_ledger", {})
                state["spent"]["openalex"] = round(float(ledger.get("dollars", 0.0)) + 0.0001, 6)  # plus the recount
                save()
            if code == 3:
                raise Stop("the pull reached its OpenAlex spending cap before finishing; run estimate.py again for a larger cap", 3)
            if code == 5:
                raise Stop("OpenAlex refused a page, usually because today's budget is spent; run the same command tomorrow to continue", 5)
            if code != 0 or not os.path.exists(corpus):
                raise Stop(f"the pull failed (exit {code})", 1)
            with open(corpus, encoding="utf-8") as fh:
                pulled = sum(1 for _ in fh)
            done("pull", items=pulled)

        if "clean" not in state["stages"]:
            if call(script("clean.py", corpus, "--out", clean_corpus, "--corpus-info", info), "clean") != 0:
                raise Stop("cleaning failed", 1)
            with open(clean_corpus, encoding="utf-8") as fh:
                kept = sum(1 for _ in fh)
            if kept < MIN_ITEMS:
                raise Stop(f"only {kept} items survive cleaning: too few for a map", 2)
            done("clean", items=kept)

        if "embed" not in state["stages"]:
            cmd = script("embed.py", clean_corpus, "--name", name, "--out-dir", embeddings, "--provider", config["provider"], "--max-seq", str(config.get("maxSeq", 1024)), "--resume")
            if config.get("model"):
                cmd += ["--model", config["model"]]
            if config["provider"] != "local":
                cmd += ["--dimensions", str(config.get("dimensions", 1024)), "--max-dollars", f"{caps['embeddingDollars']:.4f}"]
                if caps.get("tokens") is not None:
                    cmd += ["--max-tokens", str(caps["tokens"])]
            code = call(cmd, "embed")
            if code in (3, 4):
                raise Stop("embedding stopped before crossing a cap" if code == 3 else "the embedding provider refused the run", code)
            if code != 0:
                raise Stop(f"embedding failed (exit {code})", 1)
            with open(os.path.join(embeddings, f"{name}.model.json"), encoding="utf-8") as fh:
                model_info = json.load(fh)
            state["spent"]["embedding"] = model_info.get("dollars_possible", model_info.get("dollars", 0.0))
            state["spent"]["tokens"] = model_info.get("tokens", 0) + model_info.get("uncertain_tokens", 0)
            done("embed", seconds=model_info.get("seconds"))

        if "map" not in state["stages"]:
            if call(script("space.py", "--name", name, "--dir", embeddings), "map") != 0:
                raise Stop("mapping failed", 1)
            done("map")

        if "candidates" not in state["stages"]:
            if call(script("gaps.py", "connections", "--name", name, "--dir", embeddings, "--kind", scope.get("kind", "paper"), "--briefs", "--replace", "--out-dir", gaps_dir), "candidates") != 0:
                raise Stop("the candidate search failed", 1)
            done("candidates")

        if "export" not in state["stages"]:
            cmd = script("export.py", "--name", name, "--dir", embeddings, "--id", name, "--title", scope["title"], "--corpus-info", info, "--candidates", gaps_dir, "--out", export)
            if call(cmd, "export") != 0:
                raise Stop("the export failed", 1)
            done("export", file=os.path.abspath(export))
    except Stop as stop:
        state["stopped"] = {"at": now(), "reason": str(stop), "code": stop.code}
        save()
        log(f"stopped: {stop}")
        return stop.code
    except openalex.BudgetExhausted as err:
        state["stopped"] = {"at": now(), "reason": "OpenAlex refused the count", "code": 5}
        save()
        log("stopped: OpenAlex refused the request, usually because today's budget is spent")
        return 5
    finally:
        if os.path.exists(lock):
            os.remove(lock)
    state.pop("stopped", None)
    save()
    spent = state["spent"]
    log(f"done: {os.path.abspath(export)}; spent {money(spent['openalex'] + spent['embedding'])} (OpenAlex {money(spent['openalex'])}, embedding {money(spent['embedding'])}) of the {money(caps['dollars'])} approved")
    return 0


if __name__ == "__main__":
    sys.exit(main())
