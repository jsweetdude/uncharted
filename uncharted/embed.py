"""Embed uncharted corpora on this machine or through an embedding API with your own key.

    python -I uncharted/embed.py --name talks data/corpora/talks-*.jsonl
    python -I uncharted/embed.py --name talks --provider openai data/corpora/talks-*.jsonl
    python -I uncharted/embed.py --name talks --resume data/corpora/talks-*.jsonl

Writes, under --out-dir (default data/embeddings):

    <name>.npy         float32 matrix, one L2-normalized row per record
    <name>.meta.jsonl  the records, in row order (so row i of the matrix is line i)
    <name>.model.json  how the vectors were made: provider, model and revision, dimensions,
                       token limit, pooling, instruction, precision, input, how many items
                       ran past the token limit, tokens and dollars spent, timing

Providers:

    local   (default) a sentence-transformers model on this machine, on Apple's GPU, an
            NVIDIA GPU or the CPU, whichever is there. The first run downloads the model
            into the Hugging Face cache. Needs the `local` extra (PyTorch).
    openai  OpenAI's embeddings endpoint; text-embedding-3-large at 1,024 dimensions by
            default. The key comes from UNCHARTED_OPENAI_API_KEY, else OPENAI_API_KEY.
    voyage  Voyage AI's embeddings endpoint; voyage-4-large at 1,024 dimensions by default.
            Experimental: tested against recorded responses only. Key: VOYAGE_API_KEY.

Every item is embedded the same way: its `text` field (title and abstract), no instruction
prefix, cut at --max-seq tokens (approximated by characters for the APIs, whose tokenizers
are not available locally). Work is saved under <name>.parts/ as it goes, one file per API
request or per local chunk; --resume continues an interrupted run with the same
configuration, and the parts are removed once the outputs are written.

--max-tokens and --max-dollars cap an API run. Exact token counts are only known after a
request, so each request is admitted against the cap with a reserve: the characters it sends
at the highest tokens-per-character rate seen so far (at least 1 token per 4 characters),
plus 25%. Every attempt whose outcome is unknown (a timeout, a dropped connection, a garbled
reply) is charged its reserve as possibly billed, counts against the cap, and is recorded.
The cap can therefore be exceeded only by one request's estimation error. A run stops with
exit status 3 before a request would cross a cap; authentication failures, an exhausted
quota and rejected input stop it at once with exit status 4.
"""

from __future__ import annotations

import argparse
import datetime
import glob
import http.client
import json
import math
import os
import re
import shutil
import ssl
import sys
import tempfile
import time
import urllib.error
import urllib.request

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # `python -I` leaves the script's directory off sys.path
from provenance import corpus_fingerprint, hf_snapshot, pooling_of  # noqa: E402

DEFAULT_MODELS = {"local": "Qwen/Qwen3-Embedding-0.6B", "openai": "text-embedding-3-large", "voyage": "voyage-4-large"}
# Dollars per million input tokens, from each provider's pricing page, checked 2026-10-28:
# developers.openai.com/api/docs/pricing and docs.voyageai.com/docs/pricing.
PRICES = {
    ("openai", "text-embedding-3-large"): 0.13,
    ("openai", "text-embedding-3-small"): 0.02,
    ("voyage", "voyage-4-large"): 0.12,
    ("voyage", "voyage-4"): 0.06,
    ("voyage", "voyage-4-lite"): 0.02,
}
CHARS_PER_TOKEN = 4.0  # planning estimate for English; the APIs report the exact count after each request
INPUT = "title and abstract"
_urlopen = urllib.request.urlopen  # tests replace this


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


class StopRun(Exception):
    """The run must stop without retrying. `code` is the exit status."""

    def __init__(self, message: str, code: int = 4):
        super().__init__(message)
        self.code = code


def load_records(patterns: list[str]) -> list[dict]:
    records: list[dict] = []
    seen: set[str] = set()
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    rec = json.loads(line)
                    if rec["id"] in seen:
                        continue
                    seen.add(rec["id"])
                    records.append(rec)
    return records


def estimate_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN) + 8


class LocalProvider:
    name = "local"

    def __init__(self, model: str, max_seq: int, precision: str, batch_size: int):
        import torch
        from sentence_transformers import SentenceTransformer

        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        if precision == "auto":
            precision = "float32" if device == "cpu" else "float16"
        kwargs = {"torch_dtype": torch.float16} if precision == "float16" else {}
        t0 = time.time()
        self.model = SentenceTransformer(model, device=device, model_kwargs=kwargs)
        self.model.max_seq_length = max_seq
        revision, snapshot = hf_snapshot(model)
        self.info = {
            "provider": "local",
            "model": model,
            "revision": revision,
            "pooling": pooling_of(snapshot),
            "dim": int(self.model.get_embedding_dimension()),
            "max_seq_length": max_seq,
            "precision": precision,
            "device": device,
        }
        self.batch_size = batch_size
        log(f"model {model} on {device} ({precision}) in {time.time() - t0:.0f}s, dim {self.info['dim']}")

    def count_over(self, texts: list[str], limit: int) -> int:
        lengths = [len(ids) for ids in self.model.tokenizer(texts, add_special_tokens=True, truncation=False)["input_ids"]]
        return sum(1 for n in lengths if n > limit)

    def embed(self, texts: list[str], on_unknown=None) -> tuple[np.ndarray, int | None]:
        vecs = self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        return checked_unit(vecs.astype(np.float32), self.info["dim"]), 0


def checked_unit(vecs: np.ndarray, dim: int) -> np.ndarray:
    """Vectors of the expected width, finite and not degenerate, scaled to unit length."""
    if vecs.ndim != 2 or vecs.shape[1] != dim:
        raise StopRun(f"embeddings came back with shape {vecs.shape}, expected width {dim}", code=4)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    if not np.isfinite(vecs).all() or (norms < 1e-6).any():
        raise StopRun("an embedding came back empty or not finite", code=4)
    return (vecs / norms).astype(np.float32)


def redact(text: str, secret: str | None) -> str:
    return text.replace(secret, "[key]") if secret else text


class APIProvider:
    """An HTTP embeddings endpoint: one request per call, retries for failures that leave no
    doubt nothing was billed, and an immediate stop for failures that retrying cannot fix.
    Attempts whose outcome is unknown are reported through on_unknown, which may stop the run."""

    name = ""
    url = ""
    key_vars: tuple[str, ...] = ()
    max_inputs = 1000
    max_request_tokens = 100_000
    max_input_chars = 32_000

    def __init__(self, model: str, dimensions: int, max_seq: int, tries: int = 6):
        self.key = next((os.environ[v] for v in self.key_vars if os.environ.get(v)), None)
        if not self.key:
            raise StopRun(f"no API key: set {' or '.join(self.key_vars)} in the environment")
        self.model = model
        self.dimensions = dimensions
        self.max_chars = min(self.max_input_chars, int(max_seq * CHARS_PER_TOKEN))
        self.tries = tries
        self.info = {"provider": self.name, "model": model, "dim": dimensions, "max_seq_length": max_seq, "precision": "float32", "device": "api"}

    def body(self, texts: list[str]) -> dict:
        raise NotImplementedError

    def parse(self, data: dict, n: int) -> tuple[np.ndarray, int | None]:
        rows = data.get("data")
        if not isinstance(rows, list) or sorted(r.get("index") for r in rows) != list(range(n)):
            raise ValueError(f"{self.name} returned embeddings that do not match the {n} inputs one to one")
        rows = sorted(rows, key=lambda r: r["index"])
        vecs = np.array([r["embedding"] for r in rows], dtype=np.float32)
        usage = (data.get("usage") or {}).get("total_tokens")
        return vecs, int(usage) if isinstance(usage, (int, float)) else None

    def classify(self, status: int, body: str) -> None:
        """Raise StopRun for failures retrying cannot fix; return for ones worth retrying."""
        try:
            err = json.loads(body).get("error") or {}
        except (ValueError, AttributeError):
            err = {}
        code = f"{err.get('code') or ''} {err.get('type') or ''}".lower() if isinstance(err, dict) else ""
        detail = redact(str(err.get("message") if isinstance(err, dict) and err.get("message") else body)[:300], self.key)
        if status in (401, 403):
            raise StopRun(f"{self.name} refused the API key (HTTP {status}); check the key and the account")
        if status == 402 or "insufficient_quota" in code or "insufficient_quota" in body:
            raise StopRun(f"{self.name} says the account is out of credit or quota (HTTP {status}); add credit, then rerun with --resume")
        if status not in (408, 409, 425, 429) and status < 500:
            raise StopRun(f"{self.name} rejected the request (HTTP {status}): {detail}")

    def embed(self, texts: list[str], on_unknown=None) -> tuple[np.ndarray, int | None]:
        texts = [t[: self.max_chars] for t in texts]
        payload = json.dumps(self.body(texts)).encode("utf-8")
        for attempt in range(self.tries):
            req = urllib.request.Request(self.url, data=payload, method="POST", headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json", "User-Agent": "uncharted"})
            try:
                with _urlopen(req, timeout=120) as resp:
                    raw = resp.read()
            except urllib.error.HTTPError as err:
                try:
                    body = err.read().decode("utf-8", errors="replace")
                except Exception:  # noqa: BLE001
                    body = ""
                self.classify(err.code, body)
                try:
                    wait = float(err.headers.get("Retry-After") or 0) if err.headers else 0.0
                except ValueError:
                    wait = 0.0
                time.sleep(min(max(wait, 2**attempt), 120))
                continue
            except urllib.error.URLError as err:
                if isinstance(err.reason, ssl.SSLError):
                    raise StopRun(f"{self.name}: the secure connection failed ({err.reason}); check the machine's certificates or proxy") from None
                if "timed out" in str(err.reason):
                    on_unknown and on_unknown()  # the request may have been processed
                time.sleep(2**attempt)
                continue
            except (TimeoutError, ConnectionError, http.client.HTTPException):
                on_unknown and on_unknown()  # sent, reply lost: possibly billed
                time.sleep(2**attempt)
                continue
            try:
                vecs, tokens = self.parse(json.loads(raw), len(texts))
            except (ValueError, KeyError, TypeError) as exc:
                on_unknown and on_unknown()  # a reply arrived, so the request was likely billed
                log(f"{self.name} sent an unreadable reply ({redact(str(exc), self.key)[:120]}); retrying")
                time.sleep(2**attempt)
                continue
            return checked_unit(vecs, self.dimensions), tokens
        raise StopRun(f"{self.name} kept failing after {self.tries} tries; rerun with --resume later")


class OpenAIProvider(APIProvider):
    name = "openai"
    url = "https://api.openai.com/v1/embeddings"
    key_vars = ("UNCHARTED_OPENAI_API_KEY", "OPENAI_API_KEY")
    max_inputs = 2048
    max_request_tokens = 250_000  # the endpoint allows 300,000 per request
    max_input_chars = 8191 * 2  # 8,191 tokens per input; 2 characters a token leaves room for unusual text

    def body(self, texts: list[str]) -> dict:
        return {"model": self.model, "input": texts, "dimensions": self.dimensions, "encoding_format": "float"}


class VoyageProvider(APIProvider):
    name = "voyage"
    url = "https://api.voyageai.com/v1/embeddings"
    key_vars = ("VOYAGE_API_KEY",)
    max_inputs = 1000
    max_request_tokens = 100_000
    max_input_chars = 32_000 * 3

    def body(self, texts: list[str]) -> dict:
        return {"model": self.model, "input": texts, "input_type": None, "output_dimension": self.dimensions, "truncation": True}


def request_chunks(order: np.ndarray, texts: list[str], provider) -> list[np.ndarray]:
    """For an API: split the items, in length order, into requests the endpoint accepts. Each
    request is saved on its own, so a stopped run never pays twice for a finished request."""
    chunks: list[list[int]] = []
    current: list[int] = []
    tokens = 0
    for i in order:
        est = estimate_tokens(texts[i][: provider.max_chars])
        if current and (len(current) >= provider.max_inputs or tokens + est > provider.max_request_tokens):
            chunks.append(current)
            current, tokens = [], 0
        current.append(int(i))
        tokens += est
    if current:
        chunks.append(current)
    return [np.array(c) for c in chunks]


def validate_args(args: argparse.Namespace) -> None:
    for name in ("chunk", "batch_size", "dimensions", "max_seq"):
        if getattr(args, name) <= 0:
            raise StopRun(f"--{name.replace('_', '-')} must be positive", code=2)
    if args.limit < 0:
        raise StopRun("--limit cannot be negative", code=2)
    if args.max_tokens is not None and args.max_tokens < 0:
        raise StopRun("--max-tokens cannot be negative", code=2)
    for name in ("max_dollars", "price_per_million"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value < 0):
            raise StopRun(f"--{name.replace('_', '-')} must be a finite number, zero or more", code=2)


def run(args: argparse.Namespace) -> int:
    validate_args(args)
    records = load_records(args.corpora)
    if args.limit:
        records = records[: args.limit]
    if not records:
        raise StopRun("no records matched the corpus paths", code=2)
    texts = [r["text"] for r in records]
    log(f"{len(records)} records from {args.corpora}")
    model = args.model or DEFAULT_MODELS[args.provider]
    price = args.price_per_million if args.price_per_million is not None else PRICES.get((args.provider, model), 0.0 if args.provider == "local" else None)
    if price is None:
        raise StopRun(f"no price known for {args.provider} {model}; pass --price-per-million", code=2)
    api = args.provider != "local"

    if api:
        provider = (OpenAIProvider if args.provider == "openai" else VoyageProvider)(model, args.dimensions, args.max_seq)
        truncated = sum(1 for t in texts if len(t) > provider.max_chars)
    else:
        provider = LocalProvider(model, args.max_seq, "float16" if args.fp16 else args.precision, args.batch_size)
        truncated = provider.count_over(texts, args.max_seq)
    if truncated:
        log(f"{truncated} of {len(texts)} items run past {args.max_seq} tokens and are cut there")

    order = np.argsort([len(t) for t in texts], kind="stable")  # similar lengths batch together
    chunks = request_chunks(order, texts, provider) if api else [order[s : s + args.chunk] for s in range(0, len(order), args.chunk)]
    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.join(args.out_dir, args.name)
    parts = base + ".parts"
    config = {
        "provider": args.provider,
        "model": model,
        "revision": provider.info.get("revision"),
        "precision": provider.info.get("precision"),
        "dimensions": provider.info["dim"],
        "max_seq_length": args.max_seq,
        "instruction": None,
        "input": INPUT,
        "price_per_million": price,
        "chunking": "one per request" if api else args.chunk,
        "chunks": len(chunks),
        "records": corpus_fingerprint(records, {"model": model}),
    }
    state = {"tokens": 0, "chars": 0, "uncertain_tokens": 0, "seconds": 0.0, "pending": None}

    def write_json(path: str, data: dict) -> None:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=1)
        os.replace(tmp, path)

    def save_state() -> None:
        write_json(os.path.join(parts, "state.json"), state)

    done: set[int] = set()
    if os.path.isdir(parts):
        if not args.resume:
            raise StopRun(f"{parts} holds an unfinished run; pass --resume to continue it, or delete it to start over", code=2)
        try:
            with open(os.path.join(parts, "config.json"), encoding="utf-8") as fh:
                saved = json.load(fh)
        except (OSError, json.JSONDecodeError):
            raise StopRun(f"{parts} has no readable configuration; delete it to start over", code=2) from None
        if saved != config:
            changed = sorted(k for k in set(saved) | set(config) if saved.get(k) != config.get(k))
            raise StopRun(f"{parts} was made with different settings ({', '.join(changed)}); delete it to start over", code=2)
        try:
            with open(os.path.join(parts, "state.json"), encoding="utf-8") as fh:
                state.update(json.load(fh))
        except (OSError, json.JSONDecodeError):
            pass  # stopped before its first request: nothing was spent
        for path in glob.glob(os.path.join(parts, "chunk-*.npy")):
            m = re.fullmatch(r"chunk-(\d+)\.npy", os.path.basename(path))
            if m and int(m.group(1)) < len(chunks):
                done.add(int(m.group(1)))
        if state.get("pending") is not None:
            pending = state["pending"]
            if pending["chunk"] not in done:
                state["uncertain_tokens"] += pending["reserve"]  # sent before the stop; it may have been billed
            state["pending"] = None
            save_state()
        log(f"resuming: {len(done)} of {len(chunks)} parts done" + (f"; {state['tokens']:,} tokens billed, {state['uncertain_tokens']:,} possibly billed" if api else ""))
    else:
        os.makedirs(parts)
        write_json(os.path.join(parts, "config.json"), config)
        save_state()

    def reserve_for(chunk_texts: list[str]) -> int:
        chars = sum(len(t[: provider.max_chars]) for t in chunk_texts)
        ratio = max(1 / CHARS_PER_TOKEN, state["tokens"] / state["chars"] if state["chars"] else 0.0)
        return int(math.ceil(chars * ratio * 1.25)) + 8 * len(chunk_texts)

    def admit(reserve: int) -> None:
        committed = state["tokens"] + state["uncertain_tokens"]
        if args.max_tokens is not None and committed + reserve > args.max_tokens:
            save_state()
            raise StopRun(f"stopping before the token cap: {committed:,} tokens billed or possibly billed, the next request may need {reserve:,}, the cap is {args.max_tokens:,}; approve a larger cap and rerun with --resume", code=3)
        if args.max_dollars is not None and (committed + reserve) * price / 1e6 > args.max_dollars:
            save_state()
            raise StopRun(f"stopping before the spending cap: ${committed * price / 1e6:.4f} billed or possibly billed, the next request may cost ${reserve * price / 1e6:.4f}, the cap is ${args.max_dollars:.2f}; approve a larger cap and rerun with --resume", code=3)

    started = time.time()
    embedded = sum(len(chunks[c]) for c in done)
    for c, idx in enumerate(chunks):
        if c in done:
            continue
        chunk_texts = [texts[i] for i in idx]
        t0 = time.time()
        if api:
            reserve = reserve_for(chunk_texts)
            admit(reserve)
            state["pending"] = {"chunk": c, "reserve": reserve}
            save_state()

            def unknown_outcome() -> None:
                state["uncertain_tokens"] += reserve
                save_state()
                admit(reserve)  # the retry must fit under the cap too

            vecs, tokens = provider.embed(chunk_texts, on_unknown=unknown_outcome)
            if tokens is None:  # no usage in the reply: count the reserve as possibly billed
                state["uncertain_tokens"] += reserve
            else:
                state["tokens"] += tokens
                state["chars"] += sum(len(t[: provider.max_chars]) for t in chunk_texts)
        else:
            vecs, _ = provider.embed(chunk_texts)
        if vecs.shape[0] != len(idx):
            raise StopRun(f"part {c} came back with {vecs.shape[0]} rows for {len(idx)} items", code=4)
        tmp = os.path.join(parts, f"tmp-{c}.npy")  # not matched by chunk-*.npy until complete
        np.save(tmp, vecs)
        os.replace(tmp, os.path.join(parts, f"chunk-{c}.npy"))
        state["pending"] = None
        state["seconds"] += time.time() - t0
        save_state()
        embedded += len(idx)
        left = (len(texts) - embedded) / max(embedded / max(time.time() - started, 1e-6), 1e-6) / 60
        log(f"embedded {embedded}/{len(texts)} ({len(idx) / max(time.time() - t0, 1e-6):.1f}/s, ~{left:.1f} min left" + (f"; {state['tokens']:,} tokens, ${state['tokens'] * price / 1e6:.4f}" if api else "") + ")")

    dim = provider.info["dim"]
    matrix = np.zeros((len(texts), dim), dtype=np.float32)
    for c, idx in enumerate(chunks):
        vecs = np.load(os.path.join(parts, f"chunk-{c}.npy"))
        if vecs.shape != (len(idx), dim):
            raise StopRun(f"part {c} has shape {vecs.shape}, expected {(len(idx), dim)}; delete {parts} and start over", code=2)
        matrix[idx] = vecs
    if not np.isfinite(matrix).all() or not np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-3):
        raise StopRun("some embeddings are not finite or not unit length; delete the parts folder and start over", code=2)

    def staged(path: str, write) -> str:
        """Write a complete file beside its destination; the caller renames it into place."""
        handle, tmp = tempfile.mkstemp(dir=args.out_dir, prefix=os.path.basename(path) + ".", suffix=".part")
        with os.fdopen(handle, "wb") as fh:
            write(fh)
        umask = os.umask(0)
        os.umask(umask)
        os.chmod(tmp, 0o666 & ~umask)
        return tmp

    fingerprint = corpus_fingerprint(records, {**provider.info, "max_seq_length": args.max_seq})
    info = {
        **{k: v for k, v in provider.info.items() if v is not None},
        "normalized": True,
        "instruction": None,
        "input": INPUT,
        "records": len(records),
        "fingerprint": fingerprint,
        "truncated": int(truncated),
        "tokens": int(state["tokens"]),
        "uncertain_tokens": int(state["uncertain_tokens"]),
        "dollars": float(f"{state['tokens'] * price / 1e6:.6g}"),
        "dollars_possible": float(f"{(state['tokens'] + state['uncertain_tokens']) * price / 1e6:.6g}"),
        "price_per_million": price,
        "seconds": round(state["seconds"], 1),
        "embedded_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    staged_files = [
        (staged(base + ".npy", lambda fh: np.save(fh, matrix)), base + ".npy"),
        (staged(base + ".meta.jsonl", lambda fh: fh.writelines((json.dumps(r, ensure_ascii=False) + "\n").encode("utf-8") for r in records)), base + ".meta.jsonl"),
        (staged(base + ".model.json", lambda fh: fh.write(json.dumps(info, indent=2).encode("utf-8"))), base + ".model.json"),
    ]
    # model.json goes last and carries the fingerprint of the rows: a stop between the renames
    # leaves files that space.py recognizes as mismatched instead of silently pairing them
    for tmp, final in staged_files:
        os.replace(tmp, final)
    shutil.rmtree(parts)
    if (args.provider != "local") and state["uncertain_tokens"]:
        log(f"{state['uncertain_tokens']:,} tokens were sent in attempts whose outcome is unknown and may have been billed")
    log(f"wrote {base}.npy {matrix.shape}" + (f"; {state['tokens']:,} tokens, ${state['tokens'] * price / 1e6:.4f}" if api else ""))
    if not api:
        record_rate(args.out_dir, provider.info, len(texts), state["seconds"])
    return 0


def record_rate(out_dir: str, info: dict, n: int, seconds: float) -> None:
    """Keep this machine's measured local rate for estimate.py (data/benchmarks.json)."""
    if seconds <= 0 or n < 200:
        return
    path = os.path.join(os.path.dirname(os.path.abspath(out_dir)), "benchmarks.json")
    try:
        with open(path, encoding="utf-8") as fh:
            known = json.load(fh)
    except (OSError, json.JSONDecodeError):
        known = {}
    known[f"{info['model']}|{info['device']}"] = {"items_per_second": round(n / seconds, 2), "items": n, "precision": info.get("precision"), "measured": datetime.date.today().isoformat()}
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(known, fh, indent=1)
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("corpora", nargs="+", help="JSONL corpus files or globs")
    ap.add_argument("--name", required=True, help="output basename, e.g. papers")
    ap.add_argument("--provider", choices=sorted(DEFAULT_MODELS), default="local")
    ap.add_argument("--model", help="model name (default: Qwen/Qwen3-Embedding-0.6B, text-embedding-3-large or voyage-4-large)")
    ap.add_argument("--dimensions", type=int, default=1024, help="output dimensions for API models")
    ap.add_argument("--out-dir", default="data/embeddings")
    ap.add_argument("--batch-size", type=int, default=64, help="local batch size")
    ap.add_argument("--chunk", type=int, default=512, help="items per saved local chunk, the unit --resume picks up from (API runs save every request)")
    ap.add_argument("--max-seq", type=int, default=1024, help="cut each item at this many tokens")
    ap.add_argument("--precision", choices=["auto", "float16", "float32"], default="auto", help="local model precision (auto: float16 on a GPU, float32 on the CPU)")
    ap.add_argument("--fp16", action="store_true", help="same as --precision float16")
    ap.add_argument("--max-tokens", type=int, help="stop before an API run would spend more tokens than this")
    ap.add_argument("--max-dollars", type=float, help="stop before an API run would spend more than this")
    ap.add_argument("--price-per-million", type=float, help="dollars per million tokens, when the built-in price list lacks the model")
    ap.add_argument("--resume", action="store_true", help="continue an interrupted run with the same settings")
    ap.add_argument("--limit", type=int, default=0, help="embed only the first N records (smoke test)")
    args = ap.parse_args()
    try:
        return run(args)
    except StopRun as stop:
        log(f"stopped: {stop}")
        return stop.code


if __name__ == "__main__":
    sys.exit(main())
