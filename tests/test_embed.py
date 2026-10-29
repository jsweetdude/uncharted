"""Tests for uncharted/embed.py's API providers, resume and spending caps, with the HTTP layer
replaced by recorded responses: no network, no key, no PyTorch.

    python -I -m unittest discover -s tests
"""

from __future__ import annotations

import email.message
import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
import urllib.error
from unittest import mock

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "uncharted"))
import embed  # noqa: E402

DIM = 8


def vector(text: str) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return [b / 255 - 0.5 for b in digest[:DIM]]


class FakeAPI:
    """Answers embedding requests from a script: each entry is 'ok' or an HTTP status to fail with."""

    def __init__(self, script=None):
        self.script = list(script or [])
        self.requests: list[dict] = []

    def __call__(self, req, timeout=None):
        body = json.loads(req.data)
        self.requests.append({"url": req.full_url, "body": body, "auth": req.headers.get("Authorization")})
        step = self.script.pop(0) if self.script else "ok"
        if step == "timeout":
            raise TimeoutError("timed out")
        if step == "crash":
            raise RuntimeError("process killed")
        if step not in ("ok", "no-usage"):
            headers = email.message.Message()
            headers["Retry-After"] = "0"
            message = json.dumps({"error": {"code": "insufficient_quota" if step == "quota" else "x", "message": "failure"}}).encode()
            raise urllib.error.HTTPError(req.full_url, 429 if step == "quota" else int(step), "fail", headers, io.BytesIO(message))
        texts = body["input"]
        data = {"data": [{"index": i, "embedding": vector(t)} for i, t in reversed(list(enumerate(texts)))]}
        if step != "no-usage":
            data["usage"] = {"total_tokens": sum(len(t) // 4 for t in texts)}
        return io.BytesIO(json.dumps(data).encode())


class EmbedAPI(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        self.corpus = os.path.join(self.dir, "c.jsonl")
        with open(self.corpus, "w", encoding="utf-8") as fh:
            for i in range(6):
                fh.write(json.dumps({"id": f"r{i}", "title": f"T{i}", "text": f"T{i}. " + "word " * (5 + 3 * i)}) + "\n")
        self.env = mock.patch.dict(os.environ, {"UNCHARTED_OPENAI_API_KEY": "test-key", "VOYAGE_API_KEY": "voyage-key"}, clear=False)
        self.env.start()
        self.nosleep = mock.patch.object(embed.time, "sleep", lambda s: None)
        self.nosleep.start()
        self.small = [mock.patch.object(cls, "max_inputs", 2) for cls in (embed.OpenAIProvider, embed.VoyageProvider)]
        for patch in self.small:  # two inputs a request, so six records take three requests
            patch.start()

    def tearDown(self):
        for patch in self.small:
            patch.stop()
        self.nosleep.stop()
        self.env.stop()
        self.tmp.cleanup()

    def run_embed(self, fake: FakeAPI, *extra: str, name: str = "c") -> int:
        argv = ["embed.py", self.corpus, "--name", name, "--out-dir", os.path.join(self.dir, "emb"), "--dimensions", str(DIM), "--chunk", "2", *extra]
        with mock.patch.object(embed, "_urlopen", fake), mock.patch.object(sys, "argv", argv):
            return embed.main()

    def outputs(self, name: str = "c"):
        base = os.path.join(self.dir, "emb", name)
        with open(base + ".model.json", encoding="utf-8") as fh:
            return np.load(base + ".npy"), json.load(fh)

    def test_openai_run_writes_unit_vectors_and_costs(self):
        fake = FakeAPI()
        self.assertEqual(self.run_embed(fake, "--provider", "openai"), 0)
        matrix, info = self.outputs()
        self.assertEqual(matrix.shape, (6, DIM))
        self.assertTrue(np.allclose(np.linalg.norm(matrix, axis=1), 1, atol=1e-5))
        expected = np.array(vector("T0. " + "word " * 5))
        self.assertTrue(np.allclose(matrix[0], expected / np.linalg.norm(expected), atol=1e-5))
        self.assertEqual(info["provider"], "openai")
        self.assertGreater(info["tokens"], 0)
        self.assertAlmostEqual(info["dollars"], info["tokens"] * 0.13 / 1e6, places=9)
        self.assertEqual(fake.requests[0]["auth"], "Bearer test-key")
        self.assertEqual(fake.requests[0]["body"]["dimensions"], DIM)
        self.assertFalse(os.path.exists(os.path.join(self.dir, "emb", "c.parts")))

    def test_rate_limit_is_retried(self):
        fake = FakeAPI(["429", "503"])
        self.assertEqual(self.run_embed(fake, "--provider", "openai"), 0)
        self.assertEqual(len(fake.requests), 3 + 2)

    def test_bad_key_and_exhausted_quota_stop_at_once(self):
        for step in ("401", "quota"):
            with self.subTest(step=step):
                fake = FakeAPI([step])
                self.assertEqual(self.run_embed(fake, "--provider", "openai", name=f"stop-{step}"), 4)
                self.assertEqual(len(fake.requests), 1)

    def test_resume_after_a_failure_finishes_only_the_rest(self):
        first = FakeAPI(["ok", "401"])
        self.assertEqual(self.run_embed(first, "--provider", "openai", name="r"), 4)
        second = FakeAPI()
        self.assertEqual(self.run_embed(second, "--provider", "openai", "--resume", name="r"), 0)
        self.assertEqual(len(second.requests), 2)
        resumed, _ = self.outputs("r")
        self.assertEqual(self.run_embed(FakeAPI(), "--provider", "openai", name="fresh"), 0)
        fresh, _ = self.outputs("fresh")
        self.assertTrue(np.allclose(resumed, fresh))

    def test_unfinished_run_needs_resume(self):
        self.assertEqual(self.run_embed(FakeAPI(["ok", "401"]), "--provider", "openai", name="u"), 4)
        self.assertEqual(self.run_embed(FakeAPI(), "--provider", "openai", name="u"), 2)

    def test_spending_cap_stops_before_the_request(self):
        fake = FakeAPI()
        self.assertEqual(self.run_embed(fake, "--provider", "openai", "--max-tokens", "3", name="cap"), 3)
        self.assertEqual(fake.requests, [])
        self.assertEqual(self.run_embed(FakeAPI(), "--provider", "openai", "--resume", "--max-tokens", "100000", name="cap"), 0)

    def test_zero_cap_is_a_cap(self):
        fake = FakeAPI()
        self.assertEqual(self.run_embed(fake, "--provider", "openai", "--max-tokens", "0", name="zero"), 3)
        self.assertEqual(fake.requests, [])

    def test_unknown_outcome_counts_against_the_cap(self):
        fake = FakeAPI(["timeout"])
        self.assertEqual(self.run_embed(fake, "--provider", "openai", name="timeout"), 0)
        _, info = self.outputs("timeout")
        self.assertGreater(info["uncertain_tokens"], 0)
        self.assertGreater(info["dollars_possible"], info["dollars"])
        tight = FakeAPI(["timeout"])  # room for the first try but not for a retry
        self.assertEqual(self.run_embed(tight, "--provider", "openai", "--max-tokens", "40", name="tight"), 3)
        self.assertEqual(len(tight.requests), 1)

    def test_crash_after_sending_is_counted_on_resume(self):
        with self.assertRaises(RuntimeError):
            self.run_embed(FakeAPI(["ok", "crash"]), "--provider", "openai", name="crash")
        self.assertEqual(self.run_embed(FakeAPI(), "--provider", "openai", "--resume", name="crash"), 0)
        _, info = self.outputs("crash")
        self.assertGreater(info["uncertain_tokens"], 0)

    def test_reply_without_usage_is_counted_as_possible(self):
        self.assertEqual(self.run_embed(FakeAPI(["no-usage"]), "--provider", "openai", name="nousage"), 0)
        _, info = self.outputs("nousage")
        self.assertGreater(info["uncertain_tokens"], 0)

    def test_voyage_request_shape(self):
        fake = FakeAPI()
        self.assertEqual(self.run_embed(fake, "--provider", "voyage", name="v"), 0)
        body = fake.requests[0]["body"]
        self.assertEqual(body["model"], "voyage-4-large")
        self.assertEqual(body["output_dimension"], DIM)
        self.assertIsNone(body["input_type"])
        self.assertEqual(fake.requests[0]["auth"], "Bearer voyage-key")

    def test_missing_key_stops(self):
        with mock.patch.dict(os.environ, {"UNCHARTED_OPENAI_API_KEY": "", "OPENAI_API_KEY": ""}):
            self.assertEqual(self.run_embed(FakeAPI(), "--provider", "openai", name="nokey"), 4)


if __name__ == "__main__":
    unittest.main()
