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
prefix, cut at --max-seq tokens. Work is saved in chunks under <name>.parts/ as it goes;
--resume continues an interrupted run with the same configuration, and the parts are
removed once the matrix is written. --max-tokens and --max-dollars stop an API run before a
request would cross them (exit status 3); authentication failures, an exhausted quota and
rejected input stop it at once (exit status 4) instead of retrying.
"""

from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
import shutil
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

    def embed(self, texts: list[str]) -> tuple[np.ndarray, int]:
        vecs = self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        return vecs.astype(np.float32), 0


class APIProvider:
    """An HTTP embeddings endpoint: batched requests, retries for transient failures, and an
    immediate stop for failures that retrying cannot fix."""

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
        self.uncertain_tokens = 0  # sent, then timed out: possibly billed
        self.info = {"provider": self.name, "model": model, "dim": dimensions, "max_seq_length": max_seq, "precision": "float32", "device": "api"}

    def body(self, texts: list[str]) -> dict:
        raise NotImplementedError

    def parse(self, data: dict, n: int) -> tuple[np.ndarray, int]:
        rows = sorted(data["data"], key=lambda r: r["index"])
        if len(rows) != n:
            raise StopRun(f"{self.name} returned {len(rows)} embeddings for {n} inputs")
        vecs = np.array([r["embedding"] for r in rows], dtype=np.float32)
        return vecs, int(data.get("usage", {}).get("total_tokens") or 0)

    def classify(self, status: int, body: str) -> None:
        """Raise StopRun for failures retrying cannot fix; return for transient ones."""
        lowered = body.lower()
        if status in (401, 403):
            raise StopRun(f"{self.name} refused the API key (HTTP {status}); check the key and the account")
        if status == 402 or "insufficient_quota" in lowered or (status == 429 and "billing" in lowered):
            raise StopRun(f"{self.name} says the account is out of credit or quota (HTTP {status}); add credit, then rerun with --resume")
        if status in (400, 404, 413, 422):
            raise StopRun(f"{self.name} rejected the request (HTTP {status}): {body[:300]}")

    def embed(self, texts: list[str]) -> tuple[np.ndarray, int]:
        texts = [t[: self.max_chars] for t in texts]
        payload = json.dumps(self.body(texts)).encode("utf-8")
        for attempt in range(self.tries):
            req = urllib.request.Request(self.url, data=payload, method="POST", headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json", "User-Agent": "uncharted"})
            try:
                with _urlopen(req, timeout=120) as resp:
                    data = json.load(resp)
                vecs, tokens = self.parse(data, len(texts))
                norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                return vecs / np.maximum(norms, 1e-12), tokens
            except urllib.error.HTTPError as err:
                try:
                    body = err.read().decode("utf-8", errors="replace")
                except Exception:  # noqa: BLE001
                    body = ""
                self.classify(err.code, body)
                if err.code not in (408, 409, 425, 429) and err.code < 500:
                    raise StopRun(f"{self.name} failed with HTTP {err.code}: {body[:300]}") from None
                try:
                    wait = float(err.headers.get("Retry-After") or 0) if err.headers else 0.0
                except ValueError:
                    wait = 0.0
                time.sleep(min(max(wait, 2**attempt), 120))
            except (TimeoutError, urllib.error.URLError, ConnectionError, json.JSONDecodeError) as err:
                if isinstance(err, TimeoutError) or "timed out" in str(err):
                    self.uncertain_tokens += sum(estimate_tokens(t) for t in texts)
                time.sleep(2**attempt)
        raise StopRun(f"{self.name} kept failing after {self.tries} tries; rerun with --resume later")


class OpenAIProvider(APIProvider):
    name = "openai"
    url = "https://api.openai.com/v1/embeddings"
    key_vars = ("UNCHARTED_OPENAI_API_KEY", "OPENAI_API_KEY")
    max_inputs = 2048
    max_request_tokens = 250_000  # the endpoint allows 300,000 per request
    max_input_chars = 8191 * 3  # 8,191 tokens per input, at a conservative 3 characters a token

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


def requests_for(texts: list[str], provider) -> list[list[int]]:
    """Split a chunk's texts into requests the endpoint accepts."""
    groups: list[list[int]] = []
    current: list[int] = []
    tokens = 0
    for i, t in enumerate(texts):
        est = estimate_tokens(t)
        if current and (len(current) >= provider.max_inputs or tokens + est > provider.max_request_tokens):
            groups.append(current)
            current, tokens = [], 0
        current.append(i)
        tokens += est
    if current:
        groups.append(current)
    return groups


def run(args: argparse.Namespace) -> int:
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

    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.join(args.out_dir, args.name)
    parts = base + ".parts"
    config = {"provider": args.provider, "model": model, "dimensions": args.dimensions if args.provider != "local" else None, "max_seq_length": args.max_seq, "instruction": None, "input": INPUT,
              "chunk": args.chunk, "records": corpus_fingerprint(records, {"model": model})}
    state = {"tokens": 0, "dollars": 0.0, "uncertain_tokens": 0, "seconds": 0.0}
    done: dict[int, str] = {}
    if os.path.isdir(parts):
        if not args.resume:
            raise StopRun(f"{parts} holds an unfinished run; pass --resume to continue it, or delete it to start over", code=2)
        with open(os.path.join(parts, "config.json"), encoding="utf-8") as fh:
            saved = json.load(fh)
        if saved != config:
            raise StopRun(f"{parts} was made with different settings or records; delete it to start over", code=2)
        with open(os.path.join(parts, "state.json"), encoding="utf-8") as fh:
            state = json.load(fh)
        done = {int(os.path.basename(p)[6:11]): p for p in glob.glob(os.path.join(parts, "chunk-*.npy"))}
        log(f"resuming: {len(done)} chunks done, {state['tokens']:,} tokens and ${state['dollars']:.4f} spent so far")
    else:
        os.makedirs(parts)
        with open(os.path.join(parts, "config.json"), "w", encoding="utf-8") as fh:
            json.dump(config, fh, indent=1)

    if args.provider == "local":
        provider = LocalProvider(model, args.max_seq, "float16" if args.fp16 else args.precision, args.batch_size)
        truncated = provider.count_over(texts, args.max_seq)
    else:
        provider = (OpenAIProvider if args.provider == "openai" else VoyageProvider)(model, args.dimensions, args.max_seq)
        provider.uncertain_tokens = state.get("uncertain_tokens", 0)
        truncated = sum(1 for t in texts if len(t) > provider.max_chars)
    if truncated:
        log(f"{truncated} of {len(texts)} items run past {args.max_seq} tokens and are cut there")

    order = np.argsort([len(t) for t in texts], kind="stable")  # similar lengths batch together
    chunks = [order[s : s + args.chunk] for s in range(0, len(order), args.chunk)]

    def save_state() -> None:
        state["uncertain_tokens"] = getattr(provider, "uncertain_tokens", 0)
        tmp = os.path.join(parts, "state.json.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(state, fh)
        os.replace(tmp, os.path.join(parts, "state.json"))

    started = time.time()
    embedded = sum(len(chunks[c]) for c in done)
    for c, idx in enumerate(chunks):
        if c in done:
            continue
        chunk_texts = [texts[i] for i in idx]
        if args.provider != "local":
            est = sum(estimate_tokens(t[: provider.max_chars]) for t in chunk_texts)
            if args.max_tokens and state["tokens"] + est > args.max_tokens:
                save_state()
                raise StopRun(f"stopping before the token cap: {state['tokens']:,} spent, the next chunk needs about {est:,}, the cap is {args.max_tokens:,}; approve a larger cap and rerun with --resume", code=3)
            if args.max_dollars is not None and state["dollars"] + est * price / 1e6 > args.max_dollars:
                save_state()
                raise StopRun(f"stopping before the spending cap: ${state['dollars']:.4f} spent, the next chunk costs about ${est * price / 1e6:.4f}, the cap is ${args.max_dollars:.2f}; approve a larger cap and rerun with --resume", code=3)
        t0 = time.time()
        if args.provider == "local":
            vecs, tokens = provider.embed(chunk_texts)
        else:
            parts_out = []
            tokens = 0
            for group in requests_for(chunk_texts, provider):
                v, t = provider.embed([chunk_texts[g] for g in group])
                parts_out.append(v)
                tokens += t
                state["tokens"] += t
                state["dollars"] += t * price / 1e6
                save_state()
            vecs = np.concatenate(parts_out)
        np.save(os.path.join(parts, f"tmp-{c:05d}.npy"), vecs)  # not matched by chunk-*.npy until complete
        os.replace(os.path.join(parts, f"tmp-{c:05d}.npy"), os.path.join(parts, f"chunk-{c:05d}.npy"))
        state["seconds"] += time.time() - t0
        save_state()
        embedded += len(idx)
        rate = len(idx) / max(time.time() - t0, 1e-6)
        left = (len(texts) - embedded) / max(embedded / max(time.time() - started, 1e-6), 1e-6) / 60
        log(f"embedded {embedded}/{len(texts)} ({rate:.1f}/s, ~{left:.1f} min left" + (f"; {state['tokens']:,} tokens, ${state['dollars']:.4f}" if args.provider != "local" else "") + ")")

    dim = provider.info["dim"]
    matrix = np.zeros((len(texts), dim), dtype=np.float32)
    for c, idx in enumerate(chunks):
        vecs = np.load(os.path.join(parts, f"chunk-{c:05d}.npy"))
        if vecs.shape != (len(idx), dim):
            raise StopRun(f"chunk {c} has shape {vecs.shape}, expected {(len(idx), dim)}; delete {parts} and start over", code=2)
        matrix[idx] = vecs
    if not np.isfinite(matrix).all() or (np.linalg.norm(matrix, axis=1) < 0.5).any():
        raise StopRun("some embeddings are not finite or not unit length; delete the parts folder and start over", code=2)

    def atomic(path: str, write) -> None:
        """Write through a temporary file in the same folder, then rename it into place."""
        handle, tmp = tempfile.mkstemp(dir=args.out_dir, prefix=os.path.basename(path) + ".", suffix=".part")
        try:
            with os.fdopen(handle, "wb") as fh:
                write(fh)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    atomic(base + ".npy", lambda fh: np.save(fh, matrix))
    atomic(base + ".meta.jsonl", lambda fh: fh.writelines((json.dumps(r, ensure_ascii=False) + "\n").encode("utf-8") for r in records))
    info = {
        **{k: v for k, v in provider.info.items() if v is not None},
        "normalized": True,
        "instruction": None,
        "input": INPUT,
        "records": len(records),
        "truncated": int(truncated),
        "tokens": int(state["tokens"]),
        "dollars": float(f"{state['dollars']:.6g}"),
        "uncertain_tokens": int(getattr(provider, "uncertain_tokens", 0)),
        "price_per_million": price,
        "seconds": round(state["seconds"], 1),
        "embedded_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    atomic(base + ".model.json", lambda fh: fh.write(json.dumps(info, indent=2).encode("utf-8")))
    shutil.rmtree(parts)
    log(f"wrote {base}.npy {matrix.shape}" + (f"; {state['tokens']:,} tokens, ${state['dollars']:.4f}" if args.provider != "local" else ""))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("corpora", nargs="+", help="JSONL corpus files or globs")
    ap.add_argument("--name", required=True, help="output basename, e.g. papers")
    ap.add_argument("--provider", choices=sorted(DEFAULT_MODELS), default="local")
    ap.add_argument("--model", help="model name (default: Qwen/Qwen3-Embedding-0.6B, text-embedding-3-large or voyage-4-large)")
    ap.add_argument("--dimensions", type=int, default=1024, help="output dimensions for API models")
    ap.add_argument("--out-dir", default="data/embeddings")
    ap.add_argument("--batch-size", type=int, default=64, help="local batch size")
    ap.add_argument("--chunk", type=int, default=512, help="items saved per chunk, the unit --resume picks up from")
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
