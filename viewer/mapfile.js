/* Read and validate Uncharted map files (format "uncharted-export", version 1).

   The browser counterpart of uncharted/schema.py, following schema/README.md:
   - readMapFile(blob, schema) reads a File or Blob, plain or gzip, under a byte cap counted
     while decompressing, decodes strict UTF-8, refuses NaN, Infinity and deep nesting before
     parsing, parses, and validates. It resolves to { doc } or { error }, never throws.
   - validate(doc, schema) checks a parsed document against the schema (interpreting the small
     JSON Schema subset the README lists) and then against the rules beyond the schema, and
     returns a list of problems, empty for a valid map.
   - safeLink(text) returns a URL object only for an http(s) link with a host and no
     credentials, else null.

   Runs on the page, in a Web Worker and under Node, so the same code is tested in all three. */
(function (root) {
  "use strict";

  const FORMAT = "uncharted-export";
  const VERSION = 1;
  const MAX_BYTES = 200000000; // uncompressed size cap for one map file
  const MAX_DRAFTS = 1000;
  const MAX_DEPTH = 64; // the schema nests eight levels; anything far deeper is not a map file
  const SAFE_INTEGER = Number.MAX_SAFE_INTEGER;
  const PERCENTILES = ["1", "5", "10", "25", "50", "75", "90", "95", "99"];
  const CONNECTION_EVIDENCE = ["mixNn1", "baseline", "margin", "beats", "pairTail", "pairP"];
  const CONNECTION_METHOD = ["method", "mixesPerPair", "pairsTested", "tailPct", "pairCutoff", "seed"];

  class InvalidMap extends Error {}

  // ---------- reading ----------

  async function readBytes(blob, maxBytes) {
    if (blob.size > maxBytes) throw new InvalidMap(`is larger than ${Math.floor(maxBytes / 1e6)} MB`);
    const head = new Uint8Array(await blob.slice(0, 2).arrayBuffer());
    if (!(head.length === 2 && head[0] === 0x1f && head[1] === 0x8b)) return new Uint8Array(await blob.arrayBuffer());
    if (typeof DecompressionStream === "undefined") throw new InvalidMap("is gzip-compressed, and this browser cannot decompress it; open the uncompressed .json instead");
    const reader = blob.stream().pipeThrough(new DecompressionStream("gzip")).getReader();
    const chunks = [];
    let total = 0;
    try {
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        total += value.length;
        if (total > maxBytes) {
          await reader.cancel().catch(() => {});
          throw new InvalidMap(`is larger than ${Math.floor(maxBytes / 1e6)} MB uncompressed`);
        }
        chunks.push(value);
      }
    } catch (err) {
      if (err instanceof InvalidMap) throw err;
      throw new InvalidMap(`is not a readable gzip file (${String(err && err.message ? err.message : err).slice(0, 80)})`);
    }
    const out = new Uint8Array(total);
    let at = 0;
    for (const c of chunks) { out.set(c, at); at += c.length; }
    return out;
  }

  // TextDecoder drops one leading byte-order mark, as schema.py does; a second one is not JSON.
  function decode(bytes) {
    try {
      return new TextDecoder("utf-8", { fatal: true }).decode(bytes);
    } catch (err) {
      throw new InvalidMap("is not UTF-8 text");
    }
  }

  // One pass outside strings: nesting depth, and the NaN and Infinity tokens JSON does not allow.
  function prescan(text) {
    let depth = 0;
    let inString = false;
    for (let i = 0; i < text.length; i++) {
      const c = text.charCodeAt(i);
      if (inString) {
        if (c === 92) i++; // backslash: skip the escaped character
        else if (c === 34) inString = false;
        continue;
      }
      if (c === 34) inString = true;
      else if (c === 123 || c === 91) {
        if (++depth > MAX_DEPTH) throw new InvalidMap("nests lists or objects too deeply to be a map file");
      } else if (c === 125 || c === 93) depth--;
      else if (c === 78 && text.startsWith("NaN", i)) throw new InvalidMap("contains NaN, which is not valid JSON");
      else if (c === 73 && text.startsWith("Infinity", i)) throw new InvalidMap("contains Infinity, which is not valid JSON");
    }
  }

  function parse(text) {
    prescan(text);
    let doc;
    try {
      doc = JSON.parse(text);
    } catch (err) {
      const why = String(err && err.message ? err.message : err).slice(0, 120);
      throw new InvalidMap(`is not valid JSON (${why}); it may be cut short`);
    }
    // JSON.parse turns a number like 1e400 into Infinity; schema.py refuses it while parsing,
    // wherever it sits, so this walks every value once.
    const stack = [doc];
    while (stack.length) {
      const v = stack.pop();
      if (typeof v === "number") { if (!Number.isFinite(v)) throw new InvalidMap("contains a number too large to represent"); }
      else if (Array.isArray(v)) { for (let i = 0; i < v.length; i++) if (typeof v[i] !== "string") stack.push(v[i]); }
      else if (v !== null && typeof v === "object") for (const k in v) stack.push(v[k]);
    }
    return doc;
  }

  async function readMapFile(blob, schema, options) {
    try {
      // a caller may lower the cap (tests do) but never raise it
      const asked = options && options.maxBytes;
      const maxBytes = Number.isInteger(asked) && asked > 0 ? Math.min(asked, MAX_BYTES) : MAX_BYTES;
      const limit = options && Number.isInteger(options.limit) && options.limit > 0 ? options.limit : 20;
      const doc = parse(decode(await readBytes(blob, maxBytes)));
      const errors = validate(doc, schema, limit);
      return errors.length ? { error: errors[0], errors } : { doc };
    } catch (err) {
      if (err instanceof InvalidMap) return { error: `The file ${err.message}.`, errors: [`the file ${err.message}`] };
      return { error: `The file could not be read (${String(err && err.message ? err.message : err).slice(0, 120)}).`, errors: [] };
    }
  }

  // ---------- schema ----------

  function describe(v) {
    if (v === null) return "null";
    if (typeof v === "boolean") return v ? "true" : "false";
    if (typeof v === "number") return Math.abs(v) < 1e15 ? `the number ${v}` : "a very large number";
    if (typeof v === "string") return "text";
    return Array.isArray(v) ? "a list" : "an object";
  }

  // at most `limit` code points of untrusted text in a message, however long the text is
  function quote(text, limit = 40) {
    let out = "", n = 0;
    for (const ch of text) {
      if (n === limit) return JSON.stringify(out + "…");
      out += ch;
      n++;
    }
    return JSON.stringify(out);
  }

  const isObject = (v) => v !== null && typeof v === "object" && !Array.isArray(v);
  const TYPE_TESTS = {
    object: isObject,
    array: Array.isArray,
    string: (v) => typeof v === "string",
    integer: (v) => typeof v === "number" && Number.isInteger(v),
    number: (v) => typeof v === "number",
    boolean: (v) => typeof v === "boolean",
    null: (v) => v === null,
  };
  const SIMPLE_KEYS = new Set(["type", "minimum", "maximum", "minLength", "maxLength", "pattern", "description", "title"]);
  const NOTE_KEYS = new Set(["$ref", "description", "title"]);

  // Code points, counted without copying the string and only as far as the comparison needs:
  // a code point is one or two UTF-16 units, so the length settles most cases on its own.
  function cpCountUpTo(s, stop) {
    let n = 0;
    for (const ch of s) { if (++n >= stop) return n; } // eslint-disable-line no-unused-vars
    return n;
  }
  function cpLengthAtLeast(s, n) { return s.length >= 2 * n || (s.length >= n && cpCountUpTo(s, n) >= n); }
  function cpLengthAtMost(s, n) { return s.length <= n || (s.length <= 2 * n && cpCountUpTo(s, n + 1) <= n); }

  function same(a, b) {
    if (Array.isArray(a) || Array.isArray(b)) return Array.isArray(a) && Array.isArray(b) && a.length === b.length && a.every((x, i) => same(x, b[i]));
    if (isObject(a) || isObject(b)) {
      if (!isObject(a) || !isObject(b)) return false;
      const ka = Object.keys(a), kb = Object.keys(b);
      return ka.length === kb.length && ka.every((k) => Object.prototype.hasOwnProperty.call(b, k) && same(a[k], b[k]));
    }
    return a === b;
  }

  const own = (o, k) => Object.prototype.hasOwnProperty.call(o, k);

  class Checker {
    constructor(root, limit = 50) {
      this.root = root;
      this.limit = Math.max(1, limit | 0);
      this.errors = [];
      this.patterns = new Map();
      this.fast = new Map();
    }

    full() { return this.errors.length >= this.limit; }
    fail(path, message) { if (!this.full()) this.errors.push(`${path || "the file"}: ${message}`); }

    resolve(ref) {
      if (ref === "#") return this.root;
      if (!ref.startsWith("#/")) throw new Error(`only local references are supported, not ${ref}`);
      let node = this.root;
      for (let part of ref.slice(2).split("/")) {
        part = decodeURIComponent(part).replace(/~1/g, "/").replace(/~0/g, "~");
        node = Array.isArray(node) ? node[Number(part)] : node[part];
      }
      return node;
    }

    target(schema) {
      for (let n = 0; n < 32; n++) {
        if (!(isObject(schema) && own(schema, "$ref") && Object.keys(schema).every((k) => NOTE_KEYS.has(k)))) return schema;
        schema = this.resolve(schema.$ref);
      }
      throw new Error("reference loop in schema");
    }

    pattern(p) {
      if (!this.patterns.has(p)) this.patterns.set(p, new RegExp(p, "u"));
      return this.patterns.get(p);
    }

    passes(schema, value) {
      const trial = new Checker(this.root, 1);
      trial.patterns = this.patterns;
      trial.fast = this.fast;
      trial.check(schema, value, "");
      return trial.errors.length === 0;
    }

    check(schema, value, path, depth = 0) {
      if (this.full()) return;
      if (depth > 32) throw new Error("reference loop in schema");
      if (schema === true) return;
      if (schema === false) { this.fail(path, "is not allowed here"); return; }
      if (own(schema, "$ref")) {
        this.check(this.resolve(schema.$ref), value, path, depth + 1);
        if (this.full() || Object.keys(schema).every((k) => NOTE_KEYS.has(k))) return;
      }
      if (own(schema, "anyOf") && !schema.anyOf.some((sub) => this.passes(sub, value))) {
        this.fail(path, `has the wrong form (${describe(value)})`);
        return;
      }
      if (own(schema, "type")) {
        const names = Array.isArray(schema.type) ? schema.type : [schema.type];
        if (!names.some((t) => TYPE_TESTS[t](value))) {
          this.fail(path, `should be ${names.join(" or ")}, not ${describe(value)}`);
          return;
        }
      }
      if (own(schema, "const") && !same(value, schema.const)) this.fail(path, `should be ${JSON.stringify(schema.const)}`);
      if (own(schema, "enum") && !schema.enum.some((e) => same(value, e))) this.fail(path, "should be one of " + schema.enum.map((e) => JSON.stringify(e)).join(", "));
      if (typeof value === "string") this.checkString(schema, value, path);
      else if (typeof value === "number") this.checkNumber(schema, value, path);
      else if (isObject(value)) this.checkObject(schema, value, path, depth);
      else if (Array.isArray(value)) this.checkArray(schema, value, path, depth);
    }

    checkString(s, v, path) {
      if (own(s, "minLength") && !cpLengthAtLeast(v, s.minLength)) this.fail(path, s.minLength === 1 ? "is empty" : `is shorter than ${s.minLength} characters`);
      if (own(s, "maxLength") && !cpLengthAtMost(v, s.maxLength)) this.fail(path, `is longer than ${s.maxLength} characters`);
      if (own(s, "pattern") && !this.pattern(s.pattern).test(v)) this.fail(path, `${quote(v)} does not match the pattern ${s.pattern}`);
    }

    checkNumber(s, v, path) {
      if (!Number.isFinite(v)) { this.fail(path, "is a number too large to represent"); return; }
      if (Number.isInteger(v) && Math.abs(v) > SAFE_INTEGER) { this.fail(path, "is too large for a JavaScript reader to hold exactly"); return; }
      if (own(s, "minimum") && v < s.minimum) this.fail(path, `${v} is below the minimum ${s.minimum}`);
      if (own(s, "maximum") && v > s.maximum) this.fail(path, `${v} is above the maximum ${s.maximum}`);
    }

    checkObject(s, v, path, depth) {
      const keys = Object.keys(v);
      if (own(s, "maxProperties") && keys.length > s.maxProperties) {
        this.fail(path, `has ${keys.length} properties; at most ${s.maxProperties} are allowed`);
        return;
      }
      for (const key of s.required || []) if (!own(v, key)) this.fail(path, `is missing ${quote(key)}`);
      const props = s.properties || {};
      const names = s.propertyNames;
      const extra = own(s, "additionalProperties") ? s.additionalProperties : true;
      for (const key of keys) {
        const name = key.length > 80 ? quote(key) : key; // file text in a path stays short
        const sub = path ? `${path}.${name}` : name;
        if (names !== undefined && !this.passes(names, key)) this.fail(path, `has a property named ${quote(key)}, which is not an allowed name`);
        else if (own(props, key)) this.check(props[key], v[key], sub, depth);
        else if (extra === false) this.fail(path, `has an unexpected property ${quote(key)}`);
        else if (extra !== true) this.check(extra, v[key], sub, depth);
        if (this.full()) return;
      }
    }

    checkArray(s, v, path, depth) {
      if (own(s, "minItems") && v.length < s.minItems) this.fail(path, `has ${v.length} entries; at least ${s.minItems} are needed`);
      if (own(s, "maxItems") && v.length > s.maxItems) {
        this.fail(path, `has ${v.length} entries; at most ${s.maxItems} are allowed`);
        return;
      }
      const prefix = s.prefixItems || [];
      for (let i = 0; i < Math.min(prefix.length, v.length); i++) this.check(prefix[i], v[i], `${path}[${i}]`, depth);
      const rest = own(s, "items") ? s.items : true;
      if (v.length <= prefix.length || rest === true) return;
      if (rest === false) {
        this.fail(path, `has ${v.length} entries; at most ${prefix.length} are allowed`);
        return;
      }
      const quick = this.quickTest(rest);
      for (let i = prefix.length; i < v.length; i++) {
        if (quick === null || !quick(v[i])) {
          this.check(rest, v[i], `${path}[${i}]`, depth);
          if (this.full()) return;
        }
      }
    }

    // A fast yes/no for simple element schemas, so long columns validate quickly. A no falls
    // back to check(), which writes the message.
    quickTest(schema) {
      schema = this.target(schema);
      if (this.fast.has(schema)) return this.fast.get(schema);
      let test = null;
      if (isObject(schema) && Object.keys(schema).every((k) => SIMPLE_KEYS.has(k))) {
        const types = schema.type === undefined ? [] : Array.isArray(schema.type) ? schema.type : [schema.type];
        const tests = types.map((t) => TYPE_TESTS[t]);
        const lo = schema.minimum, hi = schema.maximum, minLen = schema.minLength, maxLen = schema.maxLength;
        const pat = own(schema, "pattern") ? this.pattern(schema.pattern) : null;
        test = (v) => {
          if (tests.length && !tests.some((t) => t(v))) return false;
          if (typeof v === "string") {
            return (minLen === undefined || cpLengthAtLeast(v, minLen)) && (maxLen === undefined || cpLengthAtMost(v, maxLen)) && (pat === null || pat.test(v));
          }
          if (typeof v === "number") {
            return Number.isFinite(v) && !(Number.isInteger(v) && Math.abs(v) > SAFE_INTEGER) && (lo === undefined || v >= lo) && (hi === undefined || v <= hi);
          }
          return true;
        };
      }
      this.fast.set(schema, test);
      return test;
    }
  }

  // ---------- rules beyond the schema ----------

  function safeLink(text) {
    let url;
    try { url = new URL(text); } catch (err) { return null; }
    if (url.protocol !== "http:" && url.protocol !== "https:") return null;
    if (!url.hostname || url.username || url.password || authority(text).includes("@")) return null;
    return url;
  }

  // The authority as schema.py's urlsplit sees it: up to the first /, ? or #. An @ in it is a
  // user name there, even where the browser's parser would treat a backslash as a separator.
  function authority(text) {
    const start = text.indexOf("://");
    return start < 0 ? "" : text.slice(start + 3).split(/[/?#]/)[0];
  }
  function linkProblem(text) {
    let url;
    try { url = new URL(text); } catch (err) { return "is not a well-formed link"; }
    if (url.protocol !== "http:" && url.protocol !== "https:") return "is not an http or https link";
    if (!url.hostname) return "has no host";
    if (url.username || url.password || authority(text).includes("@")) return "carries a user name or password";
    return null;
  }

  const DAYS = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
  function realDate(y, m, d) {
    if (y < 1 || m < 1 || m > 12 || d < 1 || d > DAYS[m - 1]) return false;
    return !(m === 2 && d === 29 && !((y % 4 === 0 && y % 100 !== 0) || y % 400 === 0));
  }
  function dateProblem(text) {
    const m = /^([0-9]{4})-([0-9]{2})-([0-9]{2})(?:T([0-9]{2}):([0-9]{2}):([0-9]{2})(?:\.[0-9]+)?(?:Z|[+-]([0-9]{2}):([0-9]{2}))?)?$/.exec(text);
    const ok = m && realDate(+m[1], +m[2], +m[3]) && (m[4] === undefined || (+m[4] < 24 && +m[5] < 60 && +m[6] < 60 && (m[7] === undefined || (+m[7] < 24 && +m[8] < 60))));
    return ok ? null : `${quote(text)} is not a real date or time`;
  }

  function crossCheck(doc, limit = 50) {
    const errors = [];
    limit = Math.max(1, limit | 0);
    const fail = (message) => { if (errors.length < limit) errors.push(message); return errors.length >= limit; };
    const refs = (path, positions, size, what) => {
      for (let j = 0; j < positions.length; j++) {
        if (positions[j] >= size && fail(`${path}[${j}]: ${what} ${positions[j]} does not exist (the file has ${size})`)) return true;
      }
      return errors.length >= limit;
    };
    const nearestList = (path, pairs, n) => {
      const seen = new Set();
      let previous = null;
      for (let j = 0; j < pairs.length; j++) {
        const [i, sim] = pairs[j];
        if (i >= n) { if (fail(`${path}[${j}][0]: item ${i} does not exist (the file has ${n})`)) return true; }
        else if (seen.has(i)) { if (fail(`${path}[${j}][0]: item ${i} is listed twice`)) return true; }
        seen.add(i);
        if (previous !== null && sim > previous && fail(`${path}[${j}][1]: similarities must not increase down the list`)) return true;
        previous = sim;
      }
      return errors.length >= limit;
    };

    const items = doc.items;
    const n = items.count;
    for (const key of Object.keys(items)) {
      if (key !== "count" && items[key].length !== n && fail(`items.${key}: has ${items[key].length} entries, but items.count is ${n}`)) return errors;
    }
    if (own(items, "url") !== own(items, "urlPrefix")) fail("items: url and urlPrefix must both be present or both be absent");
    if (errors.length) return errors;

    const ids = new Set();
    for (const id of items.id) {
      if (ids.has(id)) { fail(`items.id: ${quote(id)} is used twice; item ids must be unique`); return errors; }
      ids.add(id);
    }

    const hoods = doc.neighborhoods;
    const k = hoods.length;
    if (refs("items.cluster", items.cluster, k, "neighborhood")) return errors;
    const sizes = new Map();
    for (const c of items.cluster) sizes.set(c, (sizes.get(c) || 0) + 1);
    for (let i = 0; i < k; i++) {
      const h = hoods[i], p = `neighborhoods[${i}]`;
      if (h.id !== i && fail(`${p}.id: is ${h.id}; a neighborhood's id must equal its position, ${i}`)) return errors;
      if (h.size !== (sizes.get(i) || 0) && fail(`${p}.size: is ${h.size}, but ${sizes.get(i) || 0} items are in it`)) return errors;
      for (let j = 0; j < h.examples.length; j++) {
        const e = h.examples[j];
        if (e >= n) { if (fail(`${p}.examples[${j}]: item ${e} does not exist (the file has ${n})`)) return errors; }
        else if (items.cluster[e] !== i && fail(`${p}.examples[${j}]: item ${e} is not in this neighborhood`)) return errors;
      }
    }

    const sources = doc.manifest.corpus.sources;
    if (own(items, "venue") && refs("items.venue", items.venue, doc.lookups.venues.length, "venue")) return errors;
    if (own(items, "source") && refs("items.source", items.source, sources.length, "source")) return errors;
    if (own(items, "urlPrefix")) {
      const prefixes = doc.lookups.urlPrefixes;
      if (refs("items.urlPrefix", items.urlPrefix, prefixes.length, "link prefix")) return errors;
      for (let i = 0; i < n; i++) {
        const p = items.urlPrefix[i], rest = items.url[i];
        if (p === -1) {
          if (rest && fail(`items.url[${i}]: an item whose urlPrefix is -1 has no link, so its url must be empty`)) return errors;
          continue;
        }
        const problem = linkProblem(prefixes[p] + rest);
        if (problem && fail(`items.url[${i}]: the link ${problem}`)) return errors;
      }
    }
    for (let j = 0; j < sources.length; j++) {
      const src = sources[j];
      let problem = own(src, "url") ? linkProblem(src.url) : null;
      if (problem && fail(`manifest.corpus.sources[${j}].url: the link ${problem}`)) return errors;
      problem = own(src, "retrieved") ? dateProblem(src.retrieved) : null;
      if (problem && fail(`manifest.corpus.sources[${j}].retrieved: ${problem}`)) return errors;
    }
    const exportedProblem = dateProblem(doc.exported);
    if (exportedProblem && fail(`exported: ${exportedProblem}`)) return errors;

    const stats = doc.stats;
    const candidates = doc.candidates;
    const nDrafts = candidates.reduce((sum, c) => sum + c.drafts.length, 0);
    for (const [key, actual] of [["items", n], ["neighborhoods", k], ["unclustered", sizes.get(-1) || 0], ["candidates", candidates.length], ["drafts", nDrafts]]) {
      if (stats[key] !== actual && fail(`stats.${key}: is ${stats[key]}, but the file has ${actual}`)) return errors;
    }
    const pct = PERCENTILES.map((p) => stats.nn1Percentiles[p]);
    if (pct.some((a, i) => i > 0 && pct[i - 1] > a) && fail("stats.nn1Percentiles: values must not decrease from the 1st to the 99th percentile")) return errors;
    if (nDrafts > MAX_DRAFTS && fail(`candidates: the file has ${nDrafts} drafts; at most ${MAX_DRAFTS} are allowed`)) return errors;

    const candidateIds = new Set(), candidateCodes = new Set(), draftIds = new Set();
    let hasConnection = false;
    for (let ci = 0; ci < candidates.length; ci++) {
      const c = candidates[ci], p = `candidates[${ci}]`;
      if (candidateIds.has(c.id) && fail(`${p}.id: ${quote(c.id)} is used twice`)) return errors;
      if (candidateCodes.has(c.code) && fail(`${p}.code: ${quote(c.code)} is used twice`)) return errors;
      candidateIds.add(c.id);
      candidateCodes.add(c.code);
      if ((c.x === null) !== (c.y === null) && fail(`${p}: x and y must both be numbers or both be null`)) return errors;
      if (refs(`${p}.neighborhoods`, c.neighborhoods, k, "neighborhood") || refs(`${p}.parents`, c.parents, n, "item")) return errors;
      if (new Set(c.neighborhoods).size !== c.neighborhoods.length && fail(`${p}.neighborhoods: a neighborhood is listed twice`)) return errors;
      if (nearestList(`${p}.nearest`, c.nearest, n)) return errors;
      if ((c.kind === "pair" || c.kind === "connection") && c.neighborhoods.length !== 2 && fail(`${p}.neighborhoods: a ${c.kind} joins exactly two neighborhoods`)) return errors;
      if (c.kind === "connection") {
        hasConnection = true;
        if (c.parents.length !== 2 && fail(`${p}.parents: a connection mixes exactly two items`)) return errors;
        const missing = CONNECTION_EVIDENCE.filter((key) => !own(c.score, key));
        if (missing.length && fail(`${p}.score: a connection must report ${missing.join(", ")}`)) return errors;
      }
      const codes = new Set();
      for (let di = 0; di < c.drafts.length; di++) {
        const d = c.drafts[di], dp = `${p}.drafts[${di}]`;
        if (draftIds.has(d.id) && fail(`${dp}.id: ${quote(d.id)} is used twice`)) return errors;
        if (codes.has(d.code) && fail(`${dp}.code: ${quote(d.code)} is used twice in this candidate`)) return errors;
        draftIds.add(d.id);
        codes.add(d.code);
        if ((d.x === null) !== (d.y === null) && fail(`${dp}: x and y must both be numbers or both be null`)) return errors;
        if (nearestList(`${dp}.nearest`, d.nearest, n)) return errors;
        if (d.verdict === "pass" && d.reasons.length && fail(`${dp}.reasons: a draft that passed has no reasons to retry`)) return errors;
        if (d.verdict === "retry" && !d.reasons.length && fail(`${dp}.reasons: a draft to retry must say why`)) return errors;
        const problem = own(d, "checked") ? dateProblem(d.checked) : null;
        if (problem && fail(`${dp}.checked: ${problem}`)) return errors;
      }
    }
    if (hasConnection) {
      const params = doc.manifest.candidates || {};
      const missing = CONNECTION_METHOD.filter((key) => !own(params, key));
      if (missing.length) fail(`manifest.candidates: a file with connections must record ${missing.join(", ")}`);
    }
    return errors;
  }

  function validate(doc, schema, limit = 50) {
    limit = Math.max(1, limit | 0);
    if (!isObject(doc)) return ["the file: should be a JSON object"];
    if (doc.format !== FORMAT) return [`format: should be ${JSON.stringify(FORMAT)}; this is not an Uncharted map file`];
    if (!same(doc.version, VERSION)) {
      const v = doc.version;
      const shown = v === undefined ? "missing" : (v === null || ["number", "string", "boolean"].includes(typeof v)) ? JSON.stringify(v) : describe(v);
      return [`version: this file is version ${String(shown).slice(0, 40)}; this reader supports version ${VERSION}`];
    }
    const checker = new Checker(schema, limit);
    checker.check(checker.root, doc, "");
    if (checker.errors.length) return checker.errors;
    return crossCheck(doc, limit);
  }

  const api = { FORMAT, VERSION, MAX_BYTES, InvalidMap, readMapFile, validate, crossCheck, safeLink, parse, decode };
  root.UnchartedMapFile = api;
  if (typeof module === "object" && module.exports) module.exports = api;
})(typeof self !== "undefined" ? self : globalThis);
