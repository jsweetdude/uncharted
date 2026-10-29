"""Provenance shared by embed.py, verify.py and export.py: what produced a corpus's vectors,
and fingerprints that tell whether two artifacts describe the same items.

Standard library only, so it imports without the embedding stack.
"""

from __future__ import annotations

import hashlib
import json
import os

FINGERPRINT_KEYS = ("provider", "model", "revision", "dim", "max_seq_length", "instruction", "precision", "fp16")


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def corpus_fingerprint(meta: list[dict], model_info: dict) -> str:
    """32 hex characters over the embedding configuration and every item's id and embedded
    text, in row order: equal fingerprints mean the same corpus embedded the same way."""
    h = hashlib.sha256(json.dumps({k: model_info.get(k) for k in FINGERPRINT_KEYS}, sort_keys=True).encode("utf-8"))
    for m in meta:
        h.update(b"\n" + str(m["id"]).encode("utf-8") + b"\t" + hashlib.sha256(m["text"].encode("utf-8")).digest())
    return h.hexdigest()[:32]


def hf_snapshot(model: str) -> tuple[str | None, str | None]:
    """The revision of a Hugging Face model in the local cache, and that snapshot's folder.
    Only meaningful at embedding time: the cache can move on afterwards."""
    if "/" not in model or os.path.isdir(model):
        return None, None
    home = os.environ.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    repo = os.path.join(os.environ.get("HF_HUB_CACHE") or os.path.join(home, "hub"), "models--" + model.replace("/", "--"))
    try:
        with open(os.path.join(repo, "refs", "main"), encoding="utf-8") as fh:
            revision = fh.read().strip()
    except OSError:
        return None, None
    snapshot = os.path.join(repo, "snapshots", revision)
    return revision, snapshot if os.path.isdir(snapshot) else None


def pooling_of(snapshot: str | None) -> str | None:
    """The pooling a sentence-transformers snapshot declares, in the map file's words."""
    if not snapshot:
        return None
    try:
        with open(os.path.join(snapshot, "1_Pooling", "config.json"), encoding="utf-8") as fh:
            cfg = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    modes = (("lasttoken", "last-token"), ("cls_token", "cls"), ("mean_tokens", "mean"), ("max_tokens", "max"), ("mean_sqrt_len_tokens", "mean-sqrt-len"), ("weightedmean_tokens", "weighted-mean"))
    return next((name for key, name in modes if cfg.get("pooling_mode_" + key)), None)
