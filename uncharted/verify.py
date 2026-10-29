"""Verify candidates: embed them exactly like real items and measure where they land.

    python -I uncharted/verify.py --name papers data/gaps/x.cand1.md data/gaps/x.cand2.md --gap data/gaps/x.json

A candidate file starts with a `# Title` line; the rest is the body. Each candidate is
embedded (title + body, same model and settings as the corpus), then reported:

    nearest real items and their similarity, loneliness percentile against the corpus,
    the neighborhood it landed in, similarity to the gap point and to the blend point
    (when --gap is given), and a verdict: PASS when lonelier than --target-pct of the
    corpus and still on target, else RETRY with what is too close.

Results are appended to <candidate>.verify.json so the attempts are kept.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import sys  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # `python -I` leaves the script's directory off sys.path
from drafts import embedded_text, read_candidate  # noqa: E402
from gaps import Space  # noqa: E402
from provenance import corpus_fingerprint, text_sha256  # noqa: E402


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("candidates", nargs="+")
    ap.add_argument("--name", required=True)
    ap.add_argument("--dir", default="data/embeddings")
    ap.add_argument("--gap", help="gap json from gaps.py, to check the candidate landed where intended")
    ap.add_argument("--target-pct", type=float, default=95.0)
    ap.add_argument("--min-on-target", type=float, default=0.65, help="minimum similarity to the gap point to count as on target (the gap point is a synthetic vector; real text rarely exceeds ~0.75 to it)")
    ap.add_argument("--k", type=int, default=6)
    args = ap.parse_args()

    space = Space(args.dir, args.name)
    with open(os.path.join(args.dir, args.name + ".model.json"), encoding="utf-8") as fh:
        model_info = json.load(fh)
    gap = None
    if args.gap:
        with open(args.gap, encoding="utf-8") as fh:
            gap = json.load(fh)
        target_pct = gap.get("target_pct", args.target_pct)
    else:
        target_pct = args.target_pct
    target_sim = space.sim_for_pct(target_pct)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model_kwargs = {"torch_dtype": torch.float16} if model_info.get("fp16") else {}
    model = SentenceTransformer(model_info["model"], device=device, model_kwargs=model_kwargs)
    model.max_seq_length = model_info.get("max_seq_length", 512)

    try:
        parsed = [read_candidate(p) for p in args.candidates]
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    fingerprint = corpus_fingerprint(space.meta, model_info)
    texts = [embedded_text(t, b) for t, b, _ in parsed]
    vecs = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False).astype(np.float32)

    print(f"\ncorpus {space.name}: {space.n()} items; target nn1 ≤ {target_sim:.3f} (lonelier than {target_pct:.0f}%)\n")
    for path, (title, body, notes), text, v in zip(args.candidates, parsed, texts, vecs):
        near = space.nearest(v, args.k)
        nn1 = near[0][1]
        pct = space.lonely_pct(nn1)
        clusters = space.nearest_clusters(v, 2)
        result = {
            "candidate": path,
            "corpus": space.name,
            "corpus_fingerprint": fingerprint,
            "text_sha256": text_sha256(text),
            "gap": args.gap,
            "title": title,
            "notes": notes,
            "words": len(body.split()),
            "nn1": nn1,
            "lonely_pct": pct,
            "target_sim": target_sim,
            "target_pct": target_pct,
            "nearest": [{"id": space.meta[i]["id"], "sim": s, "title": space.meta[i]["title"], "year": space.meta[i].get("year"), "venue": space.meta[i].get("venue"), "url": space.meta[i].get("url")} for i, s in near],
            "neighborhoods": [{"cluster": c, "sim": s, "label": space.cluster_label(c)} for c, s in clusters],
            "checked_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        on_target = True
        if gap is not None:
            gp = np.array(gap["point"], dtype=np.float32)
            bp = np.array(gap["blend_point"], dtype=np.float32)
            result["sim_to_gap_point"] = float(v @ gp)
            result["sim_to_blend_point"] = float(v @ bp)
            on_target = result["sim_to_gap_point"] >= args.min_on_target
        lonely_enough = nn1 <= target_sim
        verdict = "PASS" if (lonely_enough and on_target) else "RETRY"
        result["verdict"] = verdict
        reasons = []
        if not lonely_enough:
            reasons.append(f"too close: nearest real item at {nn1:.3f} (target ≤ {target_sim:.3f})")
        if not on_target:
            reasons.append(f"off target: only {result['sim_to_gap_point']:.3f} similar to the gap point (want ≥ {args.min_on_target:.2f})")
        result["reasons"] = reasons

        print(f"== {title}  [{verdict}]")
        print(f"   nearest {nn1:.3f} → lonelier than {pct:.1f}% of the corpus; {result['words']} words")
        if gap is not None:
            print(f"   on-target: {result['sim_to_gap_point']:.3f} to gap point, {result['sim_to_blend_point']:.3f} to blend point")
        for c in result["neighborhoods"]:
            print(f"   lands in {c['label']} ({c['sim']:.3f})")
        for m in result["nearest"]:
            print(f"   {m['sim']:.3f}  {m['title'][:85]} ({m['year']}, {m['venue'] or 'unknown venue'})")
        for r in reasons:
            print(f"   ! {r}")
        print()

        log_path = os.path.splitext(path)[0] + ".verify.json"
        history = []
        if os.path.exists(log_path):
            with open(log_path, encoding="utf-8") as fh:
                history = json.load(fh)
        history.append(result)
        with open(log_path, "w", encoding="utf-8") as fh:
            json.dump(history, fh, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
