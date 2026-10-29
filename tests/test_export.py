"""Tests for uncharted/export.py on a small synthetic corpus built in a temporary folder.

    python -I -m unittest discover -s tests
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "uncharted"))
import schema  # noqa: E402
from drafts import embedded_text, read_candidate  # noqa: E402
from provenance import corpus_fingerprint, text_sha256  # noqa: E402

N = 30


def write_json(path: str, data) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh)


class ExportSmallCorpus(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        t = cls.tmp.name
        rng = np.random.default_rng(7)
        emb = os.path.join(t, "embeddings")
        gaps = os.path.join(t, "gaps", "run")
        os.makedirs(emb)
        os.makedirs(gaps)
        os.makedirs(os.path.join(t, "gaps", "other"))
        urls = [f"https://example.org/talks/{i}" for i in range(N)]
        urls[3] = "javascript:alert(1)"
        urls[4] = "HTTPS://Example.org/Talks/café"
        urls[5] = "https://user:pw@example.org/x"
        meta = [{"id": f"t:{i}", "title": f"Talk number {i}", "text": f"Talk number {i}. Body of talk {i} about topic {i % 3}.", "year": 2020 + i % 5, "venue": "Sample Conf" if i % 2 else "", "url": urls[i]} for i in range(N)]
        labels = [i % 3 if i < 27 else -1 for i in range(N)]
        nn1 = rng.uniform(0.5, 0.9, N)
        pct = [100 * sum(1 for j in range(N) if j != i and nn1[j] > nn1[i]) / (N - 1) for i in range(N)]
        xy = rng.uniform(-5, 10, (N, 2))
        with open(os.path.join(emb, "c.meta.jsonl"), "w", encoding="utf-8") as fh:
            fh.writelines(json.dumps(m) + "\n" for m in meta)
        with open(os.path.join(emb, "c.items.jsonl"), "w", encoding="utf-8") as fh:
            for i, m in enumerate(meta):
                fh.write(json.dumps({"i": i, "id": m["id"], "title": m["title"], "nn1": float(nn1[i]), "lonely_pct": pct[i], "cluster": labels[i], "x": float(xy[i, 0]), "y": float(xy[i, 1])}) + "\n")
        clusters = [{"cluster": c, "size": 9, "label_terms": ["topic", str(c)], "label_tags": [], "popularity": 0, "density_median_nn1": 0.7, "x": 0.0, "y": 0.0, "examples": [f"Talk number {c}"]} for c in range(3)]
        write_json(os.path.join(emb, "c.clusters.json"), clusters)
        write_json(os.path.join(emb, "c.space.json"), {"name": "c", "items": N, "clusters": 3, "unclustered": 3, "min_cluster_size": 5, "nn1_percentiles": {p: float(np.percentile(nn1, float(p))) for p in schema.PERCENTILES}, "params": {"neighbors": 10, "seed": 42}})
        model = {"model": "test/model", "dim": 8, "max_seq_length": 512, "provider": "local", "revision": "abc", "pooling": "mean", "normalized": True, "instruction": None, "precision": "float32", "input": "title and abstract"}
        write_json(os.path.join(emb, "c.model.json"), model)
        idx = np.array([[(i + d) % N for d in range(1, 11)] for i in range(N)])
        np.savez(os.path.join(emb, "c.nn.npz"), idx=idx, sims=np.full((N, 10), 0.8, dtype=np.float32))
        info = {"kind": "talk", "sources": [{"name": "Sample", "url": "https://example.org/", "retrieved": "2026-10-28"}]}
        write_json(os.path.join(t, "corpus.json"), info)

        def gap(name: str, folder: str, extra: dict) -> str:
            d = {"corpus": "c", "how": "blend of items", "target_pct": 95, "gap_nn1": 0.6, "gap_lonely_pct": 90.0, "stop": "target reached", "parents": [{"id": "t:1"}, {"id": "t:2"}],
                 "nearest_clusters": [{"cluster": 1}, {"cluster": 2}], "leaned_tags": [], "must_not_copy": [{"id": f"t:{i}", "sim": 0.7 - i / 100} for i in (1, 2, 7, 8)], "point": [0.0] * 8, "blend_point": [0.0] * 8, **extra}
            path = os.path.join(t, "gaps", folder, name + ".json")
            write_json(path, d)
            return path

        g1 = gap("x", "run", {"pair": {"a": 1, "b": 2}})
        gap("x", "other", {})  # same file name in another folder: ids must not collide
        fingerprint = corpus_fingerprint(meta, model)
        for n, body, edit in ((1, "A draft that stays as verified.", False), (2, "A draft edited after it was verified.", True)):
            md = os.path.join(gaps, f"x.cand{n}.md")
            with open(md, "w", encoding="utf-8") as fh:
                fh.write(f"# Draft {n}\n\n{body}\n---\nHelps: someone\n")
            title, parsed_body, _ = read_candidate(md)
            record = {"candidate": md, "corpus": "c", "corpus_fingerprint": fingerprint, "text_sha256": text_sha256(embedded_text(title, parsed_body)), "gap": g1,
                      "nn1": 0.72, "lonely_pct": 60.0, "verdict": "RETRY", "reasons": ["too close"], "nearest": [{"id": "t:7", "sim": 0.72}, {"id": "t:8", "sim": 0.7}], "checked_at": "2026-10-28 12:00:00"}
            write_json(md[: -len(".md")] + ".verify.json", [record])
            if edit:  # a change to the body, which is embedded; notes after --- are not
                with open(md, "w", encoding="utf-8") as fh:
                    fh.write(f"# Draft {n}\n\n{body} Now with another sentence.\n---\nHelps: someone\n")
        cls.out = os.path.join(t, "map.json")
        cls.proc = subprocess.run(
            [sys.executable, "-I", os.path.join(ROOT, "uncharted", "export.py"), "--name", "c", "--dir", emb, "--title", "Test map", "--corpus-info", os.path.join(t, "corpus.json"), "--candidates", os.path.join(t, "gaps"), "--snippet-words", "4", "--out", cls.out],
            capture_output=True, text=True, cwd=t,
        )
        cls.doc = schema.read_map(cls.out) if cls.proc.returncode == 0 else None

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_export_runs_and_validates(self):
        self.assertEqual(self.proc.returncode, 0, self.proc.stdout + self.proc.stderr)
        self.assertEqual(schema.validate(self.doc), [])

    def test_unsafe_links_are_dropped_and_good_ones_normalized(self):
        items, prefixes = self.doc["items"], self.doc["lookups"]["urlPrefixes"]
        link = lambda i: prefixes[items["urlPrefix"][i]] + items["url"][i] if items["urlPrefix"][i] >= 0 else None  # noqa: E731
        self.assertIsNone(link(3))
        self.assertIsNone(link(5))
        self.assertEqual(link(4), "https://Example.org/Talks/caf%C3%A9")
        self.assertEqual(link(0), "https://example.org/talks/0")

    def test_edited_draft_is_left_out(self):
        drafts = [d for c in self.doc["candidates"] for d in c["drafts"]]
        self.assertEqual([d["title"] for d in drafts], ["Draft 1"])
        self.assertIn("edited since verify.py checked it", self.proc.stdout)
        self.assertTrue(drafts[0]["checked"].startswith("2026-10-28T12:00:00"))
        self.assertNotEqual(drafts[0]["checked"], "2026-10-28T12:00:00")

    def test_candidate_ids_do_not_collide(self):
        ids = [c["id"] for c in self.doc["candidates"]]
        self.assertEqual(len(ids), len(set(ids)))

    def test_manifest_carries_fingerprint_and_no_local_paths(self):
        self.assertRegex(self.doc["manifest"]["corpus"]["fingerprint"], r"^[0-9a-f]{32}$")
        with open(self.out, encoding="utf-8") as fh:
            raw = fh.read()
        self.assertNotIn(self.tmp.name, raw)
        self.assertEqual(self.doc["items"]["snippet"][0], "Body of talk 0…")


if __name__ == "__main__":
    unittest.main()
