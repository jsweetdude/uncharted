"""Draft files: the text a writer produces for a candidate.

A draft file's first line is `# Title` and the body follows. Everything after the first line
containing only `---` is notes (who it helps, the barrier it removes, the standard it
touches), kept with the draft and never embedded.
"""

from __future__ import annotations


def read_candidate(path: str) -> tuple[str, str, str]:
    """Return (title, body, notes). The body keeps its paragraph breaks, one blank line between
    paragraphs; callers that embed it collapse the whitespace first."""
    with open(path, encoding="utf-8") as fh:
        lines = [ln.rstrip("\r\n") for ln in fh]
    first = next((i for i, ln in enumerate(lines) if ln.strip()), None)
    if first is None or not lines[first].startswith("# ") or not lines[first][2:].strip():
        raise ValueError(f"{path}: the first line must be '# Title'")
    title = lines[first][2:].strip()
    body_lines: list[str] = []
    notes_lines: list[str] = []
    in_notes = False
    for ln in lines[first + 1 :]:
        if not in_notes and ln.strip() == "---":
            in_notes = True
            continue
        (notes_lines if in_notes else body_lines).append(ln)
    paragraphs: list[str] = []
    current: list[str] = []
    for ln in body_lines:
        if ln.strip():
            current.append(ln.strip())
        elif current:
            paragraphs.append(" ".join(current))
            current = []
    if current:
        paragraphs.append(" ".join(current))
    notes = "\n".join(ln.strip() for ln in notes_lines if ln.strip())
    return title, "\n\n".join(paragraphs), notes


def embedded_text(title: str, body: str) -> str:
    """Exactly what verify.py embeds for a draft, and what its text hash covers."""
    return f"{title}. {' '.join(body.split())}"
