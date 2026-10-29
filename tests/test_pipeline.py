"""Tests for the pipeline's small pieces: cleaning, the estimate's interval, run.py's refusal.

    python -I -m unittest discover -s tests
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "uncharted"))
from clean import clean  # noqa: E402
from estimate import wilson  # noqa: E402


def record(i: int, title: str, words: int, kind: str = "article") -> dict:
    return {"id": f"r{i}", "title": title, "text": f"{title}. " + "word " * words, "extra": {"type": kind}}


class Cleaning(unittest.TestCase):
    def test_types_short_texts_and_repeated_titles_go(self):
        records = [record(0, "A study", 40), record(1, "An editorial", 40, "editorial"), record(2, "Short", 5), record(3, "A Study!", 40), record(4, "Another study", 40)]
        kept, removed = clean(records, {"editorial"}, 25, dedupe=True)
        self.assertEqual([r["id"] for r in kept], ["r0", "r4"])
        self.assertEqual(removed, {"type": 1, "short": 1, "duplicate title": 1})


class Interval(unittest.TestCase):
    def test_wilson_interval_contains_the_proportion_and_stays_in_bounds(self):
        lo, hi = wilson(191, 200)
        self.assertLess(lo, 191 / 200)
        self.assertGreater(hi, 191 / 200)
        self.assertEqual(wilson(0, 0), (0.0, 1.0))
        self.assertGreaterEqual(wilson(0, 10)[0], 0.0)
        self.assertLessEqual(wilson(10, 10)[1], 1.0)


class RunNeedsApproval(unittest.TestCase):
    def test_run_without_yes_refuses(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            fh.write('{"uncharted_scope": 1}')
        try:
            proc = subprocess.run([sys.executable, "-I", os.path.join(ROOT, "uncharted", "run.py"), "--scope", fh.name], capture_output=True, text=True)
        finally:
            os.remove(fh.name)
        self.assertEqual(proc.returncode, 2)
        self.assertIn("approved", proc.stderr)


if __name__ == "__main__":
    unittest.main()
