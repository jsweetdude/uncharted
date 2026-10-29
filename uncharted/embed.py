"""Embed uncharted corpora with a local sentence-embedding model.

    python -I uncharted/embed.py --name papers data/corpora/papers.jsonl
    python -I uncharted/embed.py --name talks data/corpora/talks-*.jsonl

Writes, under --out-dir (default data/embeddings):

    <name>.npy         float32 matrix, one L2-normalized row per record
    <name>.meta.jsonl  the records, in row order (so row i of the matrix is line i)
    <name>.model.json  model name, dimensions, max sequence length, timing

Everything runs on this machine (MPS when available). First run downloads the
model into the Hugging Face cache.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

DEFAULT_MODEL = "BAAI/bge-large-en-v1.5"


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


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


def embed_texts(model: SentenceTransformer, texts: list[str], batch_size: int, chunk: int = 2048) -> np.ndarray:
    """Encode in length-sorted chunks so padding is cheap, logging throughput as it goes."""
    n = len(texts)
    order = np.argsort([len(t) for t in texts])
    out = np.zeros((n, model.get_embedding_dimension()), dtype=np.float32)
    started = time.time()
    done = 0
    for start in range(0, n, chunk):
        idx = order[start : start + chunk]
        vecs = model.encode(
            [texts[i] for i in idx],
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        out[idx] = vecs.astype(np.float32)
        done += len(idx)
        rate = done / max(time.time() - started, 1e-6)
        remaining = (n - done) / max(rate, 1e-6)
        log(f"embedded {done}/{n} ({rate:.1f}/s, ~{remaining/60:.1f} min left)")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("corpora", nargs="+", help="JSONL corpus files or globs")
    ap.add_argument("--name", required=True, help="output basename, e.g. papers")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out-dir", default="data/embeddings")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-seq", type=int, default=512)
    ap.add_argument("--fp16", action="store_true", help="load the model in float16 (faster on MPS)")
    ap.add_argument("--limit", type=int, default=0, help="embed only the first N records (smoke test)")
    args = ap.parse_args()

    records = load_records(args.corpora)
    if args.limit:
        records = records[: args.limit]
    log(f"{len(records)} records from {args.corpora}")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model_kwargs = {"torch_dtype": torch.float16} if args.fp16 else {}
    t0 = time.time()
    model = SentenceTransformer(args.model, device=device, model_kwargs=model_kwargs)
    model.max_seq_length = args.max_seq
    log(f"model {args.model} on {device} in {time.time()-t0:.0f}s, dim {model.get_embedding_dimension()}")

    t1 = time.time()
    matrix = embed_texts(model, [r["text"] for r in records], args.batch_size)
    elapsed = time.time() - t1

    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.join(args.out_dir, args.name)
    np.save(base + ".npy", matrix)
    with open(base + ".meta.jsonl", "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with open(base + ".model.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "model": args.model,
                "dim": int(matrix.shape[1]),
                "max_seq_length": args.max_seq,
                "fp16": args.fp16,
                "device": device,
                "records": len(records),
                "seconds": round(elapsed, 1),
            },
            fh,
            indent=2,
        )
    log(f"wrote {base}.npy {matrix.shape} in {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
