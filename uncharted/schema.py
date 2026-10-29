"""Read and validate Uncharted map files (format "uncharted-export", version 1).

    python -I uncharted/schema.py data/exports/talks.json schema/examples/small.json

A map file is one JSON document, plain or gzip-compressed, described in schema/README.md.
`read_map` loads one defensively (size cap, UTF-8, finite numbers only) and raises only
InvalidMap or OSError. `validate` checks a parsed document against
schema/uncharted-export-v1.schema.json, then against the rules a JSON Schema cannot express
(schema/README.md > Rules beyond the schema), and returns a list of problems; it never
raises on bad data. Exit status 1 when any file fails.

The schema keeps to a small subset of JSON Schema 2020-12, listed in schema/README.md, which
this module interprets itself, so validation needs no third-party package.
"""

from __future__ import annotations

import collections
import datetime
import gzip
import json
import math
import os
import re
import sys
import urllib.parse
import zlib

FORMAT = "uncharted-export"
VERSION = 1
SCHEMA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "schema", "uncharted-export-v1.schema.json")
MAX_BYTES = 200_000_000  # uncompressed size cap for one map file
MAX_DRAFTS = 1000  # drafts in one file, all candidates together
SAFE_INTEGER = 2**53 - 1  # the largest integer a JavaScript reader holds exactly
PERCENTILES = ("1", "5", "10", "25", "50", "75", "90", "95", "99")
CONNECTION_EVIDENCE = ("mixNn1", "baseline", "margin", "beats", "pairTail", "pairP")
CONNECTION_METHOD = ("method", "mixesPerPair", "pairsTested", "tailPct", "pairCutoff", "seed")


class InvalidMap(ValueError):
    """The file cannot be read as a map at all: too large, not UTF-8, not JSON, or holding a
    number that JSON readers disagree on."""


def _reject_constant(name: str):
    raise InvalidMap(f"contains {name}, which is not valid JSON")


def _finite_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise InvalidMap(f"contains the number {text[:40]}, which is too large to represent")
    return value


def read_map(path: str, max_bytes: int = MAX_BYTES) -> object:
    """Load a map file, plain or gzip, refusing anything over max_bytes uncompressed."""
    with open(path, "rb") as fh:
        compressed = fh.read(2) == b"\x1f\x8b"
        fh.seek(0)
        if compressed:
            try:
                with gzip.GzipFile(fileobj=fh) as gz:
                    raw = gz.read(max_bytes + 1)
            except (OSError, EOFError, zlib.error) as exc:
                raise InvalidMap(f"is not a readable gzip file ({exc})") from None
        else:
            raw = fh.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise InvalidMap(f"is larger than {max_bytes // 1_000_000} MB uncompressed")
    try:
        text = raw.decode("utf-8").removeprefix("﻿")
    except UnicodeDecodeError:
        raise InvalidMap("is not UTF-8 text") from None
    try:
        return json.loads(text, parse_constant=_reject_constant, parse_float=_finite_float)
    except InvalidMap:
        raise
    except json.JSONDecodeError as exc:
        raise InvalidMap(f"is not valid JSON (line {exc.lineno}, column {exc.colno}: {exc.msg}); it may be cut short") from None
    except RecursionError:
        raise InvalidMap("nests lists or objects too deeply to be a map file") from None
    except ValueError as exc:  # an integer longer than Python's conversion limit, for one
        raise InvalidMap(f"contains a value JSON readers cannot agree on ({str(exc)[:80]})") from None


_SCHEMA: dict | None = None


def load_schema() -> dict:
    global _SCHEMA
    if _SCHEMA is None:
        with open(SCHEMA_PATH, encoding="utf-8") as fh:
            _SCHEMA = json.load(fh)
    return _SCHEMA


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


TYPE_TESTS = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: (isinstance(v, int) and not isinstance(v, bool)) or (isinstance(v, float) and v.is_integer()),
    "number": _is_number,
    "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
}

SIMPLE_KEYS = {"type", "minimum", "maximum", "minLength", "maxLength", "pattern", "description", "title"}


def _describe(v) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if _is_number(v):
        return f"the number {v}" if isinstance(v, float) or abs(v) < 10**15 else "a very large number"
    if isinstance(v, str):
        return "text"
    return "a list" if isinstance(v, list) else "an object"


def _quote(text: str, limit: int = 40) -> str:
    return json.dumps(text if len(text) <= limit else text[:limit] + "…", ensure_ascii=False)


def _same(a, b) -> bool:
    """JSON equality at every depth: true is not 1, and [true] is not [1]."""
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, list) or isinstance(b, list):
        return isinstance(a, list) and isinstance(b, list) and len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) or isinstance(b, dict):
        return isinstance(a, dict) and isinstance(b, dict) and a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    return a == b


def _compile(pattern: str) -> re.Pattern:
    # Patterns are ECMA-262 and unanchored. Python's `$` also matches before a final newline,
    # so an unescaped trailing `$` becomes `\Z`, or "abc\n" would pass ^abc$.
    if pattern.endswith("$"):
        backslashes = len(pattern) - 1 - len(pattern[:-1].rstrip("\\"))
        if backslashes % 2 == 0:
            pattern = pattern[:-1] + r"\Z"
    return re.compile(pattern, re.ASCII)


def _safe_number(v) -> bool:
    return math.isfinite(v) if isinstance(v, float) else abs(v) <= SAFE_INTEGER


class Checker:
    """Interprets the subset of JSON Schema the map schema uses; collects up to `limit` errors."""

    def __init__(self, root: dict, limit: int = 50):
        self.root = root
        self.limit = max(1, int(limit))
        self.errors: list[str] = []
        self.patterns: dict[str, re.Pattern] = {}
        self.fast: dict[int, object] = {}

    def full(self) -> bool:
        return len(self.errors) >= self.limit

    def fail(self, path: str, message: str) -> None:
        if not self.full():
            self.errors.append(f"{path or 'the file'}: {message}")

    def resolve(self, ref: str):
        """A local reference: `#`, or a JSON Pointer such as `#/$defs/sim`."""
        if ref == "#":
            return self.root
        if not ref.startswith("#/"):
            raise ValueError(f"only local references are supported, not {ref}")
        node = self.root
        for part in ref[2:].split("/"):
            part = urllib.parse.unquote(part).replace("~1", "/").replace("~0", "~")
            node = node[part] if isinstance(node, dict) else node[int(part)]
        return node

    def target(self, schema):
        """The schema a `$ref`-only schema points to, followed to the end."""
        for _ in range(32):
            if not (isinstance(schema, dict) and set(schema) <= {"$ref", "description", "title"} and "$ref" in schema):
                return schema
            schema = self.resolve(schema["$ref"])
        raise ValueError("reference loop in schema")

    def pattern(self, p: str) -> re.Pattern:
        if p not in self.patterns:
            self.patterns[p] = _compile(p)
        return self.patterns[p]

    def passes(self, schema, value) -> bool:
        trial = Checker(self.root, limit=1)
        trial.patterns = self.patterns
        trial.fast = self.fast
        trial.check(schema, value, "")
        return not trial.errors

    def check(self, schema, value, path: str, depth: int = 0) -> None:
        if self.full():
            return
        if depth > 32:
            raise ValueError("reference loop in schema")
        if schema is True:
            return
        if schema is False:
            self.fail(path, "is not allowed here")
            return
        if "$ref" in schema:
            self.check(self.resolve(schema["$ref"]), value, path, depth + 1)
            if self.full() or set(schema) <= {"$ref", "description", "title"}:
                return
        if "anyOf" in schema and not any(self.passes(sub, value) for sub in schema["anyOf"]):
            self.fail(path, f"has the wrong form ({_describe(value)})")
            return
        types = schema.get("type")
        if types is not None:
            names = [types] if isinstance(types, str) else types
            if not any(TYPE_TESTS[t](value) for t in names):
                self.fail(path, f"should be {' or '.join(names)}, not {_describe(value)}")
                return
        if "const" in schema and not _same(value, schema["const"]):
            self.fail(path, f"should be {json.dumps(schema['const'])}")
        if "enum" in schema and not any(_same(value, e) for e in schema["enum"]):
            self.fail(path, "should be one of " + ", ".join(json.dumps(e) for e in schema["enum"]))
        if isinstance(value, str):
            self.check_string(schema, value, path)
        elif _is_number(value):
            self.check_number(schema, value, path)
        elif isinstance(value, dict):
            self.check_object(schema, value, path, depth)
        elif isinstance(value, list):
            self.check_array(schema, value, path, depth)

    def check_string(self, s: dict, v: str, path: str) -> None:
        if "minLength" in s and len(v) < s["minLength"]:
            self.fail(path, "is empty" if s["minLength"] == 1 else f"is shorter than {s['minLength']} characters")
        if "maxLength" in s and len(v) > s["maxLength"]:
            self.fail(path, f"is longer than {s['maxLength']} characters")
        if "pattern" in s and not self.pattern(s["pattern"]).search(v):
            self.fail(path, f"{_quote(v)} does not match the pattern {s['pattern']}")

    def check_number(self, s: dict, v, path: str) -> None:
        if isinstance(v, float) and not math.isfinite(v):
            self.fail(path, "is not a finite number")
            return
        if isinstance(v, int) and abs(v) > SAFE_INTEGER:
            self.fail(path, "is too large for a JavaScript reader to hold exactly")
            return
        if "minimum" in s and v < s["minimum"]:
            self.fail(path, f"{v} is below the minimum {s['minimum']}")
        if "maximum" in s and v > s["maximum"]:
            self.fail(path, f"{v} is above the maximum {s['maximum']}")

    def check_object(self, s: dict, v: dict, path: str, depth: int) -> None:
        if "maxProperties" in s and len(v) > s["maxProperties"]:
            self.fail(path, f"has {len(v)} properties; at most {s['maxProperties']} are allowed")
            return
        for key in s.get("required", ()):
            if key not in v:
                self.fail(path, f"is missing {_quote(key)}")
        props = s.get("properties", {})
        names = s.get("propertyNames")
        extra = s.get("additionalProperties", True)
        for key, item in v.items():
            sub = f"{path}.{key}" if path else key
            if names is not None and not self.passes(names, key):
                self.fail(path, f"has a property named {_quote(key)}, which is not an allowed name")
            elif key in props:
                self.check(props[key], item, sub, depth)
            elif extra is False:
                self.fail(path, f"has an unexpected property {_quote(key)}")
            elif extra is not True:
                self.check(extra, item, sub, depth)
            if self.full():
                return

    def check_array(self, s: dict, v: list, path: str, depth: int) -> None:
        if "minItems" in s and len(v) < s["minItems"]:
            self.fail(path, f"has {len(v)} entries; at least {s['minItems']} are needed")
        if "maxItems" in s and len(v) > s["maxItems"]:
            self.fail(path, f"has {len(v)} entries; at most {s['maxItems']} are allowed")
            return
        prefix = s.get("prefixItems", [])
        for i, sub in enumerate(prefix[: len(v)]):
            self.check(sub, v[i], f"{path}[{i}]", depth)
        rest = s.get("items", True)
        if len(v) <= len(prefix) or rest is True:
            return
        if rest is False:
            self.fail(path, f"has {len(v)} entries; at most {len(prefix)} are allowed")
            return
        quick = self.quick_test(rest)
        for i in range(len(prefix), len(v)):
            if quick is None or not quick(v[i]):
                self.check(rest, v[i], f"{path}[{i}]", depth)
                if self.full():
                    return

    def quick_test(self, schema):
        """A fast yes/no test for simple element schemas, so long columns validate quickly.
        A no falls back to check(), which writes the message."""
        schema = self.target(schema)
        key = id(schema)
        if key in self.fast:
            return self.fast[key]
        test = None
        if isinstance(schema, dict) and set(schema) <= SIMPLE_KEYS:
            types = schema.get("type")
            tests = [TYPE_TESTS[t] for t in ([types] if isinstance(types, str) else (types or []))]
            lo, hi = schema.get("minimum"), schema.get("maximum")
            min_len, max_len = schema.get("minLength"), schema.get("maxLength")
            pat = self.pattern(schema["pattern"]) if "pattern" in schema else None

            def test(v) -> bool:
                if tests and not any(t(v) for t in tests):
                    return False
                if isinstance(v, str):
                    return (min_len is None or len(v) >= min_len) and (max_len is None or len(v) <= max_len) and (pat is None or pat.search(v) is not None)
                if _is_number(v):
                    return _safe_number(v) and (lo is None or v >= lo) and (hi is None or v <= hi)
                return True

        self.fast[key] = test
        return test


def _link_problem(url: str) -> str | None:
    """Why a link is unusable or unsafe to open, or None when it is fine."""
    try:
        parts = urllib.parse.urlsplit(url)
        host = parts.hostname
    except ValueError:
        return "is not a well-formed link"
    if parts.scheme.lower() not in ("http", "https"):
        return "is not an http or https link"
    if not host:
        return "has no host"
    if parts.username is not None or parts.password is not None:
        return "carries a user name or password"
    return None


def _date_problem(text: str) -> str | None:
    try:
        if "T" in text:
            datetime.datetime.fromisoformat(text.replace("Z", "+00:00"))
        else:
            datetime.date.fromisoformat(text)
    except ValueError:
        return f"{_quote(text)} is not a real date or time"
    return None


def cross_check(doc: dict, limit: int = 50) -> list[str]:
    """Rules the schema cannot express. Assumes the document already passed the schema."""
    errors: list[str] = []
    limit = max(1, int(limit))

    def fail(message: str) -> bool:
        """Record a problem; True once the list is full."""
        if len(errors) < limit:
            errors.append(message)
        return len(errors) >= limit

    def refs(path: str, positions, size: int, what: str) -> bool:
        for j, i in enumerate(positions):
            if i >= size and fail(f"{path}[{j}]: {what} {i} does not exist (the file has {size})"):
                return True
        return len(errors) >= limit

    def nearest_list(path: str, pairs: list, n: int) -> bool:
        seen: set = set()
        previous = None
        for j, (i, sim) in enumerate(pairs):
            if i >= n:
                if fail(f"{path}[{j}][0]: item {i} does not exist (the file has {n})"):
                    return True
            elif i in seen:
                if fail(f"{path}[{j}][0]: item {i} is listed twice"):
                    return True
            seen.add(i)
            if previous is not None and sim > previous and fail(f"{path}[{j}][1]: similarities must not increase down the list"):
                return True
            previous = sim
        return len(errors) >= limit

    items = doc["items"]
    n = items["count"]
    for key, column in items.items():
        if key != "count" and len(column) != n and fail(f"items.{key}: has {len(column)} entries, but items.count is {n}"):
            return errors
    if ("url" in items) != ("urlPrefix" in items):
        fail("items: url and urlPrefix must both be present or both be absent")
    if errors:
        return errors  # later checks index into the columns

    dupes = [x for x, c in collections.Counter(items["id"]).items() if c > 1]
    if dupes and fail(f"items.id: {_quote(dupes[0])} is used twice; item ids must be unique"):
        return errors

    hoods = doc["neighborhoods"]
    k = len(hoods)
    if refs("items.cluster", items["cluster"], k, "neighborhood"):
        return errors
    sizes = collections.Counter(items["cluster"])
    for i, h in enumerate(hoods):
        p = f"neighborhoods[{i}]"
        if h["id"] != i and fail(f"{p}.id: is {h['id']}; a neighborhood's id must equal its position, {i}"):
            return errors
        if h["size"] != sizes.get(i, 0) and fail(f"{p}.size: is {h['size']}, but {sizes.get(i, 0)} items are in it"):
            return errors
        for j, e in enumerate(h["examples"]):
            if e >= n:
                if fail(f"{p}.examples[{j}]: item {e} does not exist (the file has {n})"):
                    return errors
            elif items["cluster"][e] != i and fail(f"{p}.examples[{j}]: item {e} is not in this neighborhood"):
                return errors

    sources = doc["manifest"]["corpus"]["sources"]
    if "venue" in items and refs("items.venue", items["venue"], len(doc["lookups"]["venues"]), "venue"):
        return errors
    if "source" in items and refs("items.source", items["source"], len(sources), "source"):
        return errors
    if "urlPrefix" in items:
        prefixes = doc["lookups"]["urlPrefixes"]
        if refs("items.urlPrefix", items["urlPrefix"], len(prefixes), "link prefix"):
            return errors
        for i, (p, rest) in enumerate(zip(items["urlPrefix"], items["url"])):
            if p == -1:
                if rest and fail(f"items.url[{i}]: an item whose urlPrefix is -1 has no link, so its url must be empty"):
                    return errors
                continue
            problem = _link_problem(prefixes[p] + rest)
            if problem and fail(f"items.url[{i}]: the link {problem}"):
                return errors
    for j, src in enumerate(sources):
        problem = _link_problem(src["url"]) if "url" in src else None
        if problem and fail(f"manifest.corpus.sources[{j}].url: the link {problem}"):
            return errors
        problem = _date_problem(src["retrieved"]) if "retrieved" in src else None
        if problem and fail(f"manifest.corpus.sources[{j}].retrieved: {problem}"):
            return errors
    problem = _date_problem(doc["exported"])
    if problem and fail(f"exported: {problem}"):
        return errors

    stats = doc["stats"]
    candidates = doc["candidates"]
    n_drafts = sum(len(c["drafts"]) for c in candidates)
    for key, actual in (("items", n), ("neighborhoods", k), ("unclustered", sizes.get(-1, 0)), ("candidates", len(candidates)), ("drafts", n_drafts)):
        if stats[key] != actual and fail(f"stats.{key}: is {stats[key]}, but the file has {actual}"):
            return errors
    pct = [stats["nn1Percentiles"][p] for p in PERCENTILES]
    if any(a > b for a, b in zip(pct, pct[1:])) and fail("stats.nn1Percentiles: values must not decrease from the 1st to the 99th percentile"):
        return errors
    if n_drafts > MAX_DRAFTS and fail(f"candidates: the file has {n_drafts} drafts; at most {MAX_DRAFTS} are allowed"):
        return errors

    candidate_ids: set[str] = set()
    candidate_codes: set[str] = set()
    draft_ids: set[str] = set()
    has_connection = False
    for ci, c in enumerate(candidates):
        p = f"candidates[{ci}]"
        if c["id"] in candidate_ids and fail(f"{p}.id: {_quote(c['id'])} is used twice"):
            return errors
        if c["code"] in candidate_codes and fail(f"{p}.code: {_quote(c['code'])} is used twice"):
            return errors
        candidate_ids.add(c["id"])
        candidate_codes.add(c["code"])
        if (c["x"] is None) != (c["y"] is None) and fail(f"{p}: x and y must both be numbers or both be null"):
            return errors
        if refs(f"{p}.neighborhoods", c["neighborhoods"], k, "neighborhood") or refs(f"{p}.parents", c["parents"], n, "item"):
            return errors
        if len(set(c["neighborhoods"])) != len(c["neighborhoods"]) and fail(f"{p}.neighborhoods: a neighborhood is listed twice"):
            return errors
        if nearest_list(f"{p}.nearest", c["nearest"], n):
            return errors
        if c["kind"] in ("pair", "connection") and len(c["neighborhoods"]) != 2 and fail(f"{p}.neighborhoods: a {c['kind']} joins exactly two neighborhoods"):
            return errors
        if c["kind"] == "connection":
            has_connection = True
            if len(c["parents"]) != 2 and fail(f"{p}.parents: a connection mixes exactly two items"):
                return errors
            missing = [key for key in CONNECTION_EVIDENCE if key not in c["score"]]
            if missing and fail(f"{p}.score: a connection must report {', '.join(missing)}"):
                return errors
        codes: set[str] = set()
        for di, d in enumerate(c["drafts"]):
            dp = f"{p}.drafts[{di}]"
            if d["id"] in draft_ids and fail(f"{dp}.id: {_quote(d['id'])} is used twice"):
                return errors
            if d["code"] in codes and fail(f"{dp}.code: {_quote(d['code'])} is used twice in this candidate"):
                return errors
            draft_ids.add(d["id"])
            codes.add(d["code"])
            if (d["x"] is None) != (d["y"] is None) and fail(f"{dp}: x and y must both be numbers or both be null"):
                return errors
            if nearest_list(f"{dp}.nearest", d["nearest"], n):
                return errors
            if d["verdict"] == "pass" and d["reasons"] and fail(f"{dp}.reasons: a draft that passed has no reasons to retry"):
                return errors
            if d["verdict"] == "retry" and not d["reasons"] and fail(f"{dp}.reasons: a draft to retry must say why"):
                return errors
            problem = _date_problem(d["checked"]) if "checked" in d else None
            if problem and fail(f"{dp}.checked: {problem}"):
                return errors
    if has_connection:
        params = doc["manifest"].get("candidates", {})
        missing = [key for key in CONNECTION_METHOD if key not in params]
        if missing:
            fail(f"manifest.candidates: a file with connections must record {', '.join(missing)}")
    return errors


def validate(doc, schema: dict | None = None, limit: int = 50) -> list[str]:
    """Every problem found, up to `limit`; an empty list means the document is a valid map."""
    limit = max(1, int(limit))
    if not isinstance(doc, dict):
        return ["the file: should be a JSON object"]
    if doc.get("format") != FORMAT:
        return [f"format: should be {json.dumps(FORMAT)}; this is not an Uncharted map file"]
    if not _same(doc.get("version"), VERSION):
        version = doc.get("version")
        shown = json.dumps(version) if isinstance(version, (int, float, str, bool)) or version is None else _describe(version)
        return [f"version: this file is version {shown[:40]}; this reader supports version {VERSION}"]
    checker = Checker(schema or load_schema(), limit)
    checker.check(checker.root, doc, "")
    if checker.errors:
        return checker.errors
    return cross_check(doc, limit)


def summary(doc: dict) -> str:
    drafts = sum(len(c["drafts"]) for c in doc["candidates"])
    return f"{doc['items']['count']} items, {len(doc['neighborhoods'])} neighborhoods, {len(doc['candidates'])} candidates, {drafts} drafts"


def main() -> int:
    paths = sys.argv[1:]
    if not paths or paths[0] in ("-h", "--help"):
        print(__doc__.strip())
        return 0 if paths else 2
    failed = 0
    for path in paths:
        try:
            doc = read_map(path)
            errors = validate(doc)
        except (OSError, InvalidMap) as exc:
            errors = [f"the file {exc}"]
        if errors:
            failed += 1
            print(f"FAIL {path}")
            for e in errors[:20]:
                print(f"  {e}")
            if len(errors) > 20:
                print(f"  … and {len(errors) - 20} more")
        else:
            print(f"ok   {path}: {summary(doc)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
