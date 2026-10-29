"""Verify candidates: embed them exactly like real items and measure where they land.

    python -I uncharted/verify.py --name papers data/gaps/x.cand1.md data/gaps/x.cand2.md --gap data/gaps/x.json

A candidate file starts with a `# Title` line; the rest is the body. Each candidate is
embedded (title + body) through the same provider, model and settings as the corpus, so a
corpus embedded through an API verifies through it too, at a few hundred tokens a draft.
Each is then reported:

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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # `python -I` leaves the script's directory off sys.path
from drafts import embedded_text, read_candidate  # noqa: E402
from embed import LocalProvider, OpenAIProvider, StopRun, VoyageProvider  # noqa: E402
from gaps import Space, effective_target  # noqa: E402
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
    ap.add_argument("--on-target-rank", type=int, default=8, help="on target when the draft is closer to the gap point than the point's Nth nearest real item")
    ap.add_argument("--k", type=int, default=6)
    args = ap.parse_args()

    space = Space(args.dir, args.name)
    with open(os.path.join(args.dir, args.name + ".model.json"), encoding="utf-8") as fh:
        model_info = json.load(fh)
    gap = None
    if args.gap:
        with open(args.gap, encoding="utf-8") as fh:
            gap = json.load(fh)
        target_pct = effective_target(gap)[0]
    else:
        target_pct = args.target_pct
    target_sim = space.sim_for_pct(target_pct)

    # embed drafts exactly as the corpus was embedded: same provider, model, dimensions and limit
    max_seq = int(model_info.get("max_seq_length", 512))
    provider_name = model_info.get("provider", "local")
    try:
        if provider_name == "local":
            provider = LocalProvider(model_info["model"], max_seq, model_info.get("precision") or ("float16" if model_info.get("fp16") else "auto"), 16)
        else:
            provider = {"openai": OpenAIProvider, "voyage": VoyageProvider}[provider_name](model_info["model"], int(model_info["dim"]), max_seq)
    except StopRun as stop:
        raise SystemExit(str(stop)) from None

    try:
        parsed = [read_candidate(p) for p in args.candidates]
    except (OSError, ValueError) as exc:
        raise SystemExit(f"cannot read the draft: {exc}") from None
    fingerprint = corpus_fingerprint(space.meta, model_info)
    texts = [embedded_text(t, b) for t, b, _ in parsed]
    try:
        vecs, _ = provider.embed(texts)
    except StopRun as stop:
        raise SystemExit(str(stop)) from None

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
            rank_sim = float(np.sort(space.X @ gp)[-args.on_target_rank])
            result["on_target_sim"] = rank_sim
            on_target = result["sim_to_gap_point"] >= rank_sim
        lonely_enough = nn1 <= target_sim
        verdict = "PASS" if (lonely_enough and on_target) else "RETRY"
        result["verdict"] = verdict
        reasons = []
        if not lonely_enough:
            reasons.append(f"too close: nearest real item at {nn1:.3f} (target ≤ {target_sim:.3f})")
        if not on_target:
            reasons.append(f"off target: {result['sim_to_gap_point']:.3f} similar to the gap point, below its {args.on_target_rank}th nearest real item ({result['on_target_sim']:.3f})")
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
