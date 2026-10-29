// Tests for viewer/mapfile.js, the browser's map-file reader and validator.
//
//     node --test tests/viewer/
//
// Every file in schema/examples is a case: the valid ones must load, every file in
// examples/invalid must be rejected, and gzip, the size cap and the copies the viewer serves
// are checked too. No packages needed: Node 22's Blob, DecompressionStream and zlib suffice.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import { gzipSync } from "node:zlib";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const require = createRequire(import.meta.url);
const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..", "..");
const M = require(join(ROOT, "viewer", "mapfile.js"));
const SCHEMA = JSON.parse(readFileSync(join(ROOT, "schema", "uncharted-export-v1.schema.json"), "utf8"));
const EXAMPLES = join(ROOT, "schema", "examples");
const EXPECTED = JSON.parse(readFileSync(join(EXAMPLES, "expected-errors.json"), "utf8"));

const blobOf = (bytes) => new Blob([bytes]);
const read = (path) => M.readMapFile(blobOf(readFileSync(path)), SCHEMA);
const small = () => JSON.parse(readFileSync(join(EXAMPLES, "small.json"), "utf8"));
const firstError = (doc) => M.validate(doc, SCHEMA)[0];

test("valid examples load", async () => {
  for (const name of ["small.json", "hostile-text.json"]) {
    const r = await read(join(EXAMPLES, name));
    assert.equal(r.error, undefined, `${name}: ${r.error}`);
    assert.equal(r.doc.format, "uncharted-export");
  }
});

test("every invalid example is rejected", async () => {
  const files = readdirSync(join(EXAMPLES, "invalid")).filter((f) => f.endsWith(".json"));
  assert.equal(files.length, Object.keys(EXPECTED).length, "expected-errors.json lists every invalid file");
  for (const name of files) {
    const r = await read(join(EXAMPLES, "invalid", name));
    assert.ok(r.error, `${name} was accepted`);
    // Messages may differ from schema.py's, but this reader keeps the same key phrases.
    assert.ok(r.error.includes(EXPECTED[name]), `${name}: "${r.error}" lacks "${EXPECTED[name]}"`);
  }
});

test("gzip files load, and the format is detected by content, not name", async () => {
  const raw = readFileSync(join(EXAMPLES, "small.json"));
  const r = await M.readMapFile(blobOf(gzipSync(raw)), SCHEMA);
  assert.equal(r.error, undefined, r.error);
  assert.equal(r.doc.items.count, 12);
});

test("the size cap applies before reading and while decompressing", async () => {
  const big = new Blob([new Uint8Array(2048)]);
  assert.match((await M.readMapFile(big, SCHEMA, { maxBytes: 1024 })).error, /larger than/);
  // 4 MB of spaces compresses to a few KB; the cap stops it at 1 MB of output.
  const bomb = gzipSync(Buffer.alloc(4_000_000, 32));
  assert.ok(bomb.length < 10_000);
  assert.match((await M.readMapFile(blobOf(bomb), SCHEMA, { maxBytes: 1_000_000 })).error, /larger than .* uncompressed/);
});

test("bytes that are not UTF-8 are refused", async () => {
  const r = await M.readMapFile(blobOf(Buffer.from([0x7b, 0xff, 0xfe, 0x7d])), SCHEMA);
  assert.match(r.error, /not UTF-8/);
});

test("a broken gzip stream is refused with a plain message", async () => {
  const bytes = gzipSync(readFileSync(join(EXAMPLES, "small.json"))).subarray(0, 200);
  assert.match((await M.readMapFile(blobOf(bytes), SCHEMA)).error, /gzip|not valid JSON/);
});

test("rules beyond the schema", () => {
  let doc = small();
  doc.candidates[0].x = null; // y stays a number
  assert.match(firstError(doc), /both be numbers or both be null/);

  doc = small();
  const pass = doc.candidates.flatMap((c) => c.drafts).find((d) => d.verdict === "pass");
  pass.reasons = ["too close"];
  assert.match(firstError(doc), /passed has no reasons/);

  doc = small();
  doc.candidates[0].nearest = [doc.candidates[0].nearest[0], doc.candidates[0].nearest[0]];
  assert.match(firstError(doc), /listed twice/);

  doc = small();
  doc.stats.nn1Percentiles["99"] = -1;
  assert.match(firstError(doc), /must not decrease/);

  doc = small();
  doc.title = "x".repeat(121);
  assert.match(firstError(doc), /longer than 120/);

  // Lengths count code points: 120 emoji are 240 UTF-16 units but within the limit.
  doc = small();
  doc.title = "🗺".repeat(120);
  assert.equal(firstError(doc), undefined);
});

test("only http and https links with a host and no credentials are opened", () => {
  assert.ok(M.safeLink("https://example.org/a"));
  assert.ok(M.safeLink("http://example.org"));
  for (const bad of ["javascript:alert(1)", "data:text/html,x", "https://user:pw@example.org/", "https://u@example.org", "//example.org", "not a url", "file:///etc/passwd"]) {
    assert.equal(M.safeLink(bad), null, bad);
  }
});

test("edge cases the review found", async () => {
  const raw = readFileSync(join(EXAMPLES, "small.json"));
  const bom = Buffer.from([0xef, 0xbb, 0xbf]);
  assert.equal((await M.readMapFile(blobOf(Buffer.concat([bom, raw])), SCHEMA)).error, undefined, "one byte-order mark is allowed");
  assert.ok((await M.readMapFile(blobOf(Buffer.concat([bom, bom, raw])), SCHEMA)).error, "two are not");
  // options can lower the cap but not raise it, and bad options cannot make the reader throw
  assert.equal((await M.readMapFile(blobOf(raw), SCHEMA, null)).error, undefined);
  assert.equal((await M.readMapFile(blobOf(raw), SCHEMA, { maxBytes: Infinity })).error, undefined);
  assert.match((await M.readMapFile(blobOf(raw), SCHEMA, { maxBytes: 100 })).error, /larger than/);
  // a number that overflows is refused wherever it sits, here in a score's extra values
  const doc = small();
  const text = JSON.stringify(doc).replace('"lonely":', '"extra":1e400,"lonely":');
  assert.match((await M.readMapFile(blobOf(Buffer.from(text)), SCHEMA)).error, /too large to represent/);
  // year 0000 is not a date, as in Python
  const zero = small();
  zero.exported = "0000-01-01T00:00:00Z";
  assert.match(firstError(zero), /not a real date/);
  // an @ in the authority is a user name, as schema.py reads it, whatever URL would make of it
  assert.equal(M.safeLink("https://example.org\\@evil.example/"), null);
  // a very long title is measured without copying it, and the message quotes only its start
  const long = small();
  long.title = "x".repeat(3_000_000);
  assert.match(firstError(long), /longer than 120/);
  const key = small();
  key["k".repeat(1_000_000)] = 1;
  const keyError = firstError(key);
  assert.ok(keyError.length < 200 && /unexpected property/.test(keyError), keyError.slice(0, 120));
  // property names that mean something to JavaScript are just names here
  const proto = JSON.parse(JSON.stringify(small()).replace('{"format"', '{"__proto__":{"x":1},"format"'));
  assert.match(firstError(proto), /unexpected property "__proto__"/);
});

test("the viewer's copies of the schema and the sample match their sources", () => {
  assert.deepEqual(readFileSync(join(ROOT, "viewer", "uncharted-export-v1.schema.json")), readFileSync(join(ROOT, "schema", "uncharted-export-v1.schema.json")));
  assert.deepEqual(readFileSync(join(ROOT, "viewer", "samples", "small.json")), readFileSync(join(EXAMPLES, "small.json")));
});

test("every export in data/exports validates when there are any", async (t) => {
  let names;
  try { names = readdirSync(join(ROOT, "data", "exports")).filter((f) => /\.json(\.gz)?$/.test(f)); } catch { names = []; }
  if (!names.length) { t.skip("data/exports holds no map files"); return; }
  for (const name of names) {
    const r = await M.readMapFile(blobOf(readFileSync(join(ROOT, "data", "exports", name))), SCHEMA);
    assert.equal(r.error, undefined, `${name}: ${r.error}`);
  }
});
