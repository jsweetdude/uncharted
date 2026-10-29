"""Tests for uncharted/schema.py, run against the example files in schema/examples/.

    python -I -m unittest discover -s tests
"""

from __future__ import annotations

import gzip
import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "uncharted"))
import schema  # noqa: E402

EXAMPLES = os.path.join(ROOT, "schema", "examples")


def small() -> dict:
    return schema.read_map(os.path.join(EXAMPLES, "small.json"))


class ExampleFiles(unittest.TestCase):
    def test_valid_examples_pass(self):
        for name in ("small.json", "hostile-text.json"):
            with self.subTest(name=name):
                self.assertEqual(schema.validate(schema.read_map(os.path.join(EXAMPLES, name))), [])

    def test_each_invalid_example_fails_with_its_expected_error(self):
        with open(os.path.join(EXAMPLES, "expected-errors.json"), encoding="utf-8") as fh:
            expected = json.load(fh)
        files = sorted(f for f in os.listdir(os.path.join(EXAMPLES, "invalid")) if f.endswith(".json"))
        self.assertEqual(sorted(expected), files)
        for name in files:
            with self.subTest(name=name):
                try:
                    errors = schema.validate(schema.read_map(os.path.join(EXAMPLES, "invalid", name)))
                except schema.InvalidMap as exc:
                    errors = [f"the file {exc}"]
                self.assertTrue(errors)
                self.assertIn(expected[name], errors[0])


class Reading(unittest.TestCase):
    def test_gzip_reads_and_size_cap_holds(self):
        with open(os.path.join(EXAMPLES, "small.json"), "rb") as fh:
            raw = fh.read()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "small.json.gz")
            with gzip.open(path, "wb") as fh:
                fh.write(raw)
            self.assertEqual(schema.validate(schema.read_map(path)), [])
            with self.assertRaises(schema.InvalidMap):
                schema.read_map(path, max_bytes=len(raw) - 1)


class Rules(unittest.TestCase):
    def test_true_is_not_version_1(self):
        doc = small()
        doc["version"] = True
        self.assertIn("supports version 1", schema.validate(doc)[0])

    def test_trailing_newline_does_not_satisfy_an_anchored_pattern(self):
        doc = small()
        doc["id"] = "sample-talks\n"
        self.assertIn("does not match the pattern", schema.validate(doc)[0])

    def test_item_without_a_prefix_cannot_carry_a_link(self):
        doc = small()
        doc["items"]["url"][11] = "javascript:alert(1)"
        self.assertIn("has no link", schema.validate(doc)[0])

    def test_neighborhood_size_must_match_its_items(self):
        doc = small()
        doc["neighborhoods"][2]["size"] = 3
        self.assertIn("neighborhoods[2].size", schema.validate(doc)[0])

    def test_x_and_y_are_null_together(self):
        doc = small()
        doc["candidates"][2]["x"] = 0.5
        self.assertIn("both be null", schema.validate(doc)[0])

    def test_error_count_is_capped(self):
        doc = small()
        doc["items"]["x"] = [2.0] * doc["items"]["count"]
        self.assertEqual(len(schema.validate(doc, limit=5)), 5)

    def test_non_finite_number_in_memory_is_rejected(self):
        doc = small()
        doc["items"]["nn1"][0] = float("nan")
        self.assertIn("finite", schema.validate(doc)[0])


if __name__ == "__main__":
    unittest.main()
