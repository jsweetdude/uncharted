"""Compile candidates and their verification history into one Markdown report.

    python -I uncharted/report.py --name talks --out data/gaps/report-talks.md

Scans data/gaps for *.verify.json written by verify.py, keeps the ones verified against
the named corpus, and reports each candidate's latest attempt: nearest real item and
similarity, loneliness percentile, verdict, attempts so far, and the value-check notes.
Candidates whose file name starts with "baseline" are listed separately as calibration.
"""

from __future__ import annotations

import argparse
import glob
import json
import os


def load_results(root: str, corpus: str) -> list[dict]:
    rows = []
    for path in sorted(glob.glob(os.path.join(root, "**", "*.verify.json"), recursive=True)):
        with open(path, encoding="utf-8") as fh:
            history = json.load(fh)
        history = [h for h in history if h.get("corpus") == corpus]  # results without a corpus field are skipped
        if not history:
            continue
        last = dict(history[-1])
        last["attempts"] = len(history)
        last["first_nn1"] = history[0]["nn1"]
        rows.append(last)
    return rows


def group_key(row: dict) -> str:
    """Attempts of one idea share a stem: x.cand1.md, x.cand2.md -> x."""
    base = os.path.basename(row["candidate"])
    stem = base.split(".cand")[0] if ".cand" in base else os.path.splitext(base)[0]
    return stem


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True, help="corpus name the candidates were verified against")
    ap.add_argument("--root", default="data/gaps")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = load_results(args.root, args.name)
    if not rows:
        raise SystemExit(f"no verification results for corpus {args.name!r} under {args.root}")

    # One line per idea: the best attempt across its candidate files.
    ideas: dict[str, list[dict]] = {}
    for r in rows:
        ideas.setdefault(group_key(r), []).append(r)
    baselines = {k: v for k, v in ideas.items() if k.startswith("baseline")}
    real = {k: v for k, v in ideas.items() if not k.startswith("baseline")}

    def best(attempts: list[dict]) -> dict:
        """The loneliest draft that passed, or the loneliest draft when none did."""
        passed = [r for r in attempts if r["verdict"] == "PASS"]
        return min(passed or attempts, key=lambda r: r["nn1"])

    lines = [f"# Candidates verified against the `{args.name}` corpus", ""]
    sample = rows[0]
    lines.append(
        f"Target: nearest-neighbor similarity ≤ {sample['target_sim']:.3f} (lonelier than {sample['target_pct']:.0f}% of the corpus). "
        "Each idea shows its loneliest passing draft, or its loneliest draft when none passed; drafts counts every verified draft of that idea."
    )
    lines.append("")
    lines.append("| Idea (best draft) | Drafts | Nearest real item | Sim | Lonelier than | Verdict |")
    lines.append("|---|---|---|---|---|---|")
    ordered = sorted(real.items(), key=lambda kv: best(kv[1])["nn1"])
    for key, attempts in ordered:
        b = best(attempts)
        near = b["nearest"][0]
        lines.append(
            f"| {b['title']} | {sum(a['attempts'] for a in attempts)} | {near['title']} ({near['year']}) | {b['nn1']:.3f} | {b['lonely_pct']:.1f}% | {b['verdict']} |"
        )
    if baselines:
        lines.append("")
        lines.append("Calibration:")
        lines.append("")
        for key, attempts in baselines.items():
            b = best(attempts)
            near = b["nearest"][0]
            lines.append(f"- {b['title']}: {b['nn1']:.3f} to \"{near['title']}\" ({near['year']}), lonelier than {b['lonely_pct']:.1f}%.")
    lines.append("")
    lines.append("## Details")
    for key, attempts in ordered:
        b = best(attempts)
        lines.append("")
        lines.append(f"### {b['title']}")
        lines.append("")
        lines.append(f"File: `{b['candidate']}` ({b['words']} words). Gap: `{b.get('gap') or 'none'}`.")
        trail = " → ".join(f"{a['nn1']:.3f}" for a in sorted(attempts, key=lambda r: r["candidate"]))
        lines.append(f"Nearest-item similarity by draft: {trail}. Landed in {', '.join(c['label'] for c in b['neighborhoods'])}.")
        if "sim_to_gap_point" in b:
            lines.append(f"On target: {b['sim_to_gap_point']:.3f} to the gap point, {b['sim_to_blend_point']:.3f} to the blend point.")
        lines.append("")
        lines.append("Nearest real items:")
        lines.append("")
        for m in b["nearest"][:5]:
            link = f" [{m['id']}]({m['url']})" if m.get("url") else ""
            lines.append(f"- {m['sim']:.3f} {m['title']} ({m['year']}, {m['venue'] or 'unknown venue'}){link}")
        if len(attempts) > 1:
            # Rewording moves a draft away from some neighbors without changing the idea, so the
            # items any draft landed near are all candidates for prior art.
            named: dict[str, dict] = {}
            for a in attempts:
                for m in a["nearest"]:
                    if m["id"] not in named or m["sim"] > named[m["id"]]["sim"]:
                        named[m["id"]] = m
            shown = {m["id"] for m in b["nearest"][:5]}
            others = sorted((m for m in named.values() if m["id"] not in shown), key=lambda m: -m["sim"])[:6]
            if others:
                lines.append("")
                lines.append("Also near earlier drafts:")
                lines.append("")
                for m in others:
                    link = f" [{m['id']}]({m['url']})" if m.get("url") else ""
                    lines.append(f"- {m['sim']:.3f} {m['title']} ({m['year']}, {m['venue'] or 'unknown venue'}){link}")
        if b.get("notes"):
            lines.append("")
            lines.append("Value check:")
            lines.append("")
            for ln in b["notes"].splitlines():
                lines.append(f"- {ln.lstrip('- ').strip()}")
        if b.get("reasons"):
            lines.append("")
            lines.append("Why not PASS: " + "; ".join(b["reasons"]))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {args.out}: {len(real)} ideas, {len(baselines)} baselines")


if __name__ == "__main__":
    main()
