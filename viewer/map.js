/* Uncharted viewer. Opens one map file at a time (see schema/README.md), from the file picker,
   a drop, or a built-in sample, always through mapfile.js's checks, and draws it on a canvas
   with HTML lists beside it, so everything on the map can also be reached without the canvas.
   Everything in a map file is untrusted: it reaches the page only as text (textContent), and
   links only through UnchartedMapFile.safeLink. */
(() => {
  "use strict";

  const MF = window.UnchartedMapFile;
  const TAU = Math.PI * 2;
  const BIN_EDGES = [50, 75, 90, 95, 98, 99];
  const BIN_LABELS = ["Below 50 · most crowded", "50–75", "75–90", "90–95", "95–98", "98–99", "99 and up · loneliest"];
  // loneliness mode: bins from LOUD_FROM up (95th percentile and lonelier) draw last, larger,
  // ringed; the crowded bins recede. Strength, not new color, carries the emphasis.
  const LOUD_FROM = 4;
  const BIN_SCALE = [1, 1, 1, 1, 1.6, 1.9, 2.2];
  const QUIET_ALPHA = { dense: [0.3, 0.42, 0.58, 0.75], sparse: [0.55, 0.68, 0.82, 0.9] };
  const KIND_NAMES = { connection: "Connection", pair: "Pair", blend: "Blend" };
  const KIND_HINTS = { connection: "mix of two related neighborhoods, emptier than chance", pair: "between two related neighborhoods", blend: "mix of chosen items" };
  const VERDICT_NAMES = { pass: "Passed", retry: "Retry" };
  const SAMPLES = [{ id: "small", file: "samples/small.json", name: "the sample: 12 invented talks" }];
  const EMPTY_DETAILS = "Pick a candidate connection to see what surrounds it, or select any point.";
  const CLUMP_PX = 14;

  const $ = (id) => document.getElementById(id);
  const canvas = $("map");
  const ctx = canvas.getContext("2d");
  const wrap = $("map-wrap");
  const tooltip = $("tooltip");
  // the stacked layout: one scrolling page rather than a fixed map beside two scrolling regions
  const narrow = window.matchMedia("(max-width: 959px), (max-height: 639px)");
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

  let S = freshState();
  function freshState() {
    return {
      d: null, n: 0, name: "",
      x: null, y: null, cluster: null, lonely: null, bin: null,
      lower: null, nbMembers: new Map(), nbBox: new Map(), byBin: [], unclustered: null, nbShort: [], nbLong: [],
      candsByItem: new Map(), drafts: [], mapCands: [], grid: null, focusCache: new WeakMap(),
      bounds: null, W: 0, H: 0, dpr: 1, cx: 0.5, cy: 0.5, k: 1, fitK: 1,
      mode: "neighborhood", hover: null, detail: null, selectedNb: null, focusCand: null,
      clumps: [], legendClumps: false, pal: null, raf: 0,
    };
  }

  // ---------- small DOM helpers ----------

  function el(tag, attrs, ...children) {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(attrs || {})) {
      if (value === null || value === undefined || value === false) continue;
      if (key === "class") node.className = value;
      else if (key === "text") node.textContent = value;
      else if (key === "style") node.style.cssText = value; // CSSOM, which the content security policy allows
      else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
      else node.setAttribute(key, value === true ? "" : String(value));
    }
    for (const child of children.flat(Infinity)) {
      if (child === null || child === undefined || child === false) continue;
      node.append(child instanceof Node ? child : document.createTextNode(String(child)));
    }
    return node;
  }
  // replaceChildren turns a null argument into the text "null", so absent children are dropped first
  function fill(node, ...children) {
    node.replaceChildren(...children.flat(Infinity).filter((c) => c !== null && c !== undefined && c !== false));
  }
  // untrusted text, isolated so a direction override inside it cannot reorder what surrounds it
  const t = (text) => el("bdi", { text });
  const hidden = (text) => el("span", { class: "pc-visually-hidden", text });
  function announce(text) { $("announce").textContent = ""; requestAnimationFrame(() => { $("announce").textContent = text; }); }

  function showMessage(title, detail) {
    fill($("message-text"), el("strong", { text: title }), detail ? el("p", { text: detail }) : null);
    $("message").hidden = false;
  }
  function hideMessage() { $("message").hidden = true; }

  const SHAPES = {
    connection: '<circle cx="8" cy="8" r="5.5" fill="none" stroke="currentColor" stroke-width="2"/><circle cx="8" cy="8" r="1.6" fill="currentColor"/>',
    pair: '<path d="M8 2.2 13.8 8 8 13.8 2.2 8Z" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/>',
    blend: '<path d="M8 2.8 13.8 13H2.2Z" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/>',
    clump: '<circle cx="8" cy="8" r="6.5" fill="none" stroke="currentColor" stroke-width="1.5"/><text x="8" y="11" text-anchor="middle" font-size="8" font-weight="600" fill="currentColor">3</text>',
    nearest: '<circle cx="8" cy="8" r="5.5" fill="none" stroke="currentColor" stroke-width="1.75"/><text x="8" y="11" text-anchor="middle" font-size="8" font-weight="600" fill="currentColor">1</text>',
    parent: '<circle cx="8" cy="8" r="5.5" fill="none" stroke="currentColor" stroke-width="1.5" stroke-dasharray="2 2"/>',
    draftPass: '<rect x="3.5" y="3.5" width="9" height="9" fill="currentColor"/>',
    draftRetry: '<rect x="4.25" y="4.25" width="7.5" height="7.5" fill="none" stroke="currentColor" stroke-width="1.5"/>',
  };
  function markerIcon(kind) {
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", "0 0 16 16");
    svg.setAttribute("class", kind === "nearest" ? "marker-icon focus-icon" : "marker-icon");
    svg.setAttribute("aria-hidden", "true");
    svg.innerHTML = SHAPES[kind]; // fixed markup above, never data
    return svg;
  }

  // ---------- palette ----------

  function readPalette() {
    const cs = getComputedStyle(document.documentElement);
    const v = (name) => cs.getPropertyValue(name).trim();
    S.pal = {
      surface: v("--pc-surface"), text: v("--pc-text"), ink: v("--pc-ink"), muted: v("--pc-muted"),
      cobalt: v("--pc-cobalt"), unclustered: v("--pc-control-border"),
      cat: [1, 2, 3, 4, 5, 6, 7, 8].map((i) => v(`--map-c${i}`)),
      seq: [1, 2, 3, 4, 5, 6, 7].map((i) => v(`--map-seq-${i}`)),
    };
  }

  // ---------- reading the map ----------

  const col = (name) => S.d.items[name];
  const median = () => S.d.stats.nn1Percentiles["50"];
  function itemLink(i) {
    const prefixes = col("urlPrefix");
    if (!prefixes || prefixes[i] < 0) return null;
    return MF.safeLink(S.d.lookups.urlPrefixes[prefixes[i]] + col("url")[i]);
  }
  function itemVenue(i) { const v = col("venue"); return v && v[i] >= 0 ? S.d.lookups.venues[v[i]] : ""; }
  function itemYear(i) { const y = col("year"); return y && y[i] !== null ? y[i] : null; }
  function itemSource(i) { const s = col("source"); return s && s[i] >= 0 ? S.d.manifest.corpus.sources[s[i]].name : ""; }
  const slotOf = (nb) => (Number.isInteger(nb.slot) ? nb.slot : nb.id) % 8;
  const nbShort = (id) => (id >= 0 ? S.nbShort[id] : "Unclustered");
  const nbLong = (id) => (id >= 0 ? S.nbLong[id] : "Unclustered");

  // Neighborhood names from their distinctive terms, skipping a term that shares a stem with an
  // earlier one ("games gaming"), and taking a third term when two names would read the same.
  const SUFFIXES = ["ibility", "ability", "ically", "ations", "ation", "ible", "able", "ical", "ings", "ing", "ers", "ies", "er", "es", "al", "ic", "s"];
  function stem(word) {
    const w = word.toLowerCase();
    for (const s of SUFFIXES) if (w.length - s.length >= 3 && w.endsWith(s)) return w.slice(0, -s.length);
    return w;
  }
  function distinctTerms(terms) {
    const out = [], stems = [];
    for (const term of terms) {
      const s = stem(term);
      if (stems.some((p) => p === s || p.startsWith(s) || s.startsWith(p))) continue;
      stems.push(s);
      out.push(term);
    }
    return out;
  }
  function nameNeighborhoods(hoods) {
    const terms = hoods.map((nb) => {
      const d = distinctTerms(nb.terms);
      return d.length ? d : nb.tags.length ? [nb.tags[0]] : [`neighborhood ${nb.id}`];
    });
    const short = terms.map((d) => d.slice(0, 2).join(" · "));
    const count = new Map();
    for (const s of short) count.set(s, (count.get(s) || 0) + 1);
    S.nbShort = short.map((s, i) => (count.get(s) > 1 && terms[i].length > 2 ? terms[i].slice(0, 3).join(" · ") : s));
    S.nbLong = terms.map((d) => d.slice(0, 4).join(" · "));
  }

  function prepare(doc) {
    const it = doc.items;
    const n = it.count;
    S.d = doc; S.n = n;
    S.x = Float32Array.from(it.x);
    S.y = Float32Array.from(it.y);
    S.cluster = Int32Array.from(it.cluster);
    S.lonely = Float32Array.from(it.lonely);
    S.bin = new Uint8Array(n);
    const bins = BIN_EDGES.map(() => []).concat([[]]);
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    const members = new Map();
    const unclustered = [];
    for (let i = 0; i < n; i++) {
      const xi = S.x[i], yi = S.y[i];
      if (xi < minX) minX = xi; if (xi > maxX) maxX = xi;
      if (yi < minY) minY = yi; if (yi > maxY) maxY = yi;
      let b = 0;
      while (b < BIN_EDGES.length && S.lonely[i] >= BIN_EDGES[b]) b++;
      S.bin[i] = b;
      bins[b].push(i);
      const c = S.cluster[i];
      if (c < 0) unclustered.push(i);
      else { if (!members.has(c)) members.set(c, []); members.get(c).push(i); }
    }
    S.bounds = { minX, maxX, minY, maxY };
    S.unclustered = Int32Array.from(unclustered);
    S.byBin = bins.map((a) => Int32Array.from(a));
    S.nbMembers = new Map([...members].map(([c, a]) => [c, Int32Array.from(a)]));
    nameNeighborhoods(doc.neighborhoods);
    S.bySize = [...doc.neighborhoods].sort((a, b) => b.size - a.size);
    S.mapCands = doc.candidates.filter((c) => c.x !== null);
    S.drafts = doc.candidates.flatMap((c) => c.drafts.map((draft) => ({ draft, cand: c })));
    S.candsByItem = new Map();
    for (const c of doc.candidates) {
      for (const [i] of c.nearest) {
        if (!S.candsByItem.has(i)) S.candsByItem.set(i, []);
        S.candsByItem.get(i).push(c);
      }
    }
    buildGrid();
  }

  // Uniform grid over the map for hover and click hit tests (counting sort into cells).
  function buildGrid() {
    const G = Math.max(16, Math.min(256, Math.round(Math.sqrt(S.n) * 1.5)));
    const { minX, maxX, minY, maxY } = S.bounds;
    const sx = (G - 1) / (maxX - minX || 1), sy = (G - 1) / (maxY - minY || 1);
    const cellOf = new Int32Array(S.n);
    const start = new Int32Array(G * G + 1);
    for (let i = 0; i < S.n; i++) {
      const c = Math.floor((S.y[i] - minY) * sy) * G + Math.floor((S.x[i] - minX) * sx);
      cellOf[i] = c;
      start[c + 1]++;
    }
    for (let c = 0; c < G * G; c++) start[c + 1] += start[c];
    const next = start.slice(0, G * G);
    const items = new Int32Array(S.n);
    for (let i = 0; i < S.n; i++) items[next[cellOf[i]]++] = i;
    S.grid = { G, sx, sy, start, items };
  }

  // What a pinned candidate lights: its neighborhoods, its nearest items in list order, the items
  // it was built from, and its drafts; items near any of its drafts keep the focus too.
  function candFocus(c) {
    let f = S.focusCache.get(c);
    if (!f) {
      const nearest = c.nearest.map(([i]) => i);
      const parents = c.parents.filter((i) => !nearest.includes(i));
      const draftNearest = c.drafts.flatMap((d) => d.nearest.map(([i]) => i));
      f = {
        nbs: new Set(c.neighborhoods), nearest, parents, drafts: c.drafts,
        items: new Set([...nearest, ...parents]),
        related: new Set([...nearest, ...parents, ...draftNearest]),
      };
      S.focusCache.set(c, f);
    }
    return f;
  }
  function candSubline(c) {
    const parts = [KIND_NAMES[c.kind]];
    if (c.x === null) parts.push("not on the map");
    if (c.drafts.length) parts.push(`${c.drafts.length} ${c.drafts.length === 1 ? "draft" : "drafts"}`);
    return parts.join(" · ");
  }
  function verdictBadge(d) {
    return el("span", { class: d.verdict === "pass" ? "pc-badge pc-badge--success" : "pc-badge", text: VERDICT_NAMES[d.verdict] });
  }
  function similarity(v) {
    return el("span", {}, el("span", { class: "mono", text: v.toFixed(3) }), el("span", { class: "pc-muted", text: ` · corpus median ${median().toFixed(3)}` }));
  }
  const mono = (text) => el("span", { class: "mono", text });
  function headRow(left, right) {
    return el("div", { class: "list-head pc-label sub-label" }, el("span", { text: left }), el("span", { text: right }));
  }

  // ---------- view transform ----------

  const toSX = (x) => (x - S.cx) * S.k + S.W / 2;
  const toSY = (y) => S.H / 2 - (y - S.cy) * S.k;
  const toDX = (sx) => (sx - S.W / 2) / S.k + S.cx;
  const toDY = (sy) => S.cy - (sy - S.H / 2) / S.k;

  // the corner the open legend covers (with its margins), or null when it sits below the map or is closed
  function legendArea() {
    const box = $("legend-box");
    if (!box || !box.open || getComputedStyle(box).position !== "absolute") return null;
    return { w: Math.min(box.offsetWidth + 20, S.W * 0.5), h: Math.min(box.offsetHeight + 20, S.H * 0.5) };
  }
  // fit a data box into the map, beside the legend rather than under it: above it or to its
  // right, whichever leaves the bigger map
  function fitTo(minX, maxX, minY, maxY, pad = 0.9, maxZoom = Infinity) {
    const w = Math.max(maxX - minX, 1e-6), h = Math.max(maxY - minY, 1e-6);
    const L = legendArea();
    let k = Math.min(S.W / w, S.H / h), offX = 0, offY = 0;
    if (L) {
      const above = Math.min(S.W / w, (S.H - L.h) / h);
      const right = Math.min((S.W - L.w) / w, S.H / h);
      if (right >= above) { k = right; offX = L.w / 2; } else { k = above; offY = -L.h / 2; }
    }
    S.k = Math.min(k * pad, S.fitK * maxZoom);
    S.cx = (minX + maxX) / 2 - offX / S.k;
    S.cy = (minY + maxY) / 2 + offY / S.k;
  }
  function fitAll() {
    const b = S.bounds;
    if (!b || S.W < 4 || S.H < 4) return;
    S.fitK = Math.min(S.W / Math.max(b.maxX - b.minX, 1e-6), S.H / Math.max(b.maxY - b.minY, 1e-6)) * 0.92;
    fitTo(b.minX, b.maxX, b.minY, b.maxY, 0.92);
  }
  function zoomAt(factor, sx, sy) {
    const k = Math.max(S.fitK * 0.5, Math.min(S.fitK * 80, S.k * factor));
    const dx = toDX(sx), dy = toDY(sy);
    S.k = k;
    S.cx = dx - (sx - S.W / 2) / k;
    S.cy = dy + (sy - S.H / 2) / k;
    requestDraw();
  }
  function panBy(px, py) { S.cx -= px / S.k; S.cy += py / S.k; requestDraw(); }
  function centerOn(x, y, minZoom) {
    S.cx = x; S.cy = y;
    if (minZoom) S.k = Math.max(S.k, S.fitK * minZoom);
    requestDraw();
  }
  // A candidate with the items that define it, leaving out far outliers so one distant item
  // cannot shrink the frame to nothing.
  function fitToCandidate(c) {
    const f = candFocus(c);
    if (S.W < 4 || S.H < 4) return;
    const pts = [...f.nearest.slice(0, 8), ...f.parents].map((i) => [S.x[i], S.y[i]]);
    for (const d of f.drafts) if (d.x !== null) pts.push([d.x, d.y]);
    const center = c.x !== null ? [c.x, c.y] : pts.reduce((a, p) => [a[0] + p[0] / pts.length, a[1] + p[1] / pts.length], [0, 0]);
    const dist = pts.map((p) => Math.hypot(p[0] - center[0], p[1] - center[1])).sort((a, b) => a - b);
    const cap = Math.max(dist[Math.floor(dist.length / 2)] * 2.5, 0.02);
    const keep = pts.filter((p) => Math.hypot(p[0] - center[0], p[1] - center[1]) <= cap).concat([center]);
    const xs = keep.map((p) => p[0]), ys = keep.map((p) => p[1]);
    fitTo(Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys), 0.8, 6);
    requestDraw();
  }

  function dotRadius() {
    const base = Math.max(1.1, Math.min(3, 2.6 * Math.sqrt(1000 / S.n)));
    return Math.min(base * 2.2, base * Math.pow(S.k / S.fitK, 0.4));
  }

  // ---------- drawing ----------

  // Animation frames pause in background or occluded pages; the timer keeps the canvas current
  // there. Frame and timer are one request: whichever runs first cancels the other.
  let rafTimer = 0;
  function requestDraw() {
    if (S.raf) return;
    S.raf = requestAnimationFrame(draw);
    rafTimer = setTimeout(() => { rafTimer = 0; if (S.raf) { cancelAnimationFrame(S.raf); draw(); } }, 120);
  }

  function drawPoints(indices, color, alpha, r, filter) {
    ctx.globalAlpha = alpha;
    ctx.fillStyle = color;
    ctx.beginPath();
    const W = S.W, H = S.H, round = r >= 1.5;
    for (let j = 0; j < indices.length; j++) {
      const i = indices[j];
      if (filter && !filter(i)) continue;
      const sx = toSX(S.x[i]), sy = toSY(S.y[i]);
      if (sx < -r || sy < -r || sx > W + r || sy > H + r) continue;
      if (round) { ctx.moveTo(sx + r, sy); ctx.arc(sx, sy, r, 0, TAU); }
      else ctx.rect(sx - r, sy - r, r * 2, r * 2);
    }
    ctx.fill();
    ctx.globalAlpha = 1;
  }

  // a ringed dot: surface halo first, then the fill, so a lonely item separates from the haze under it
  function drawLoud(indices, color, r, filter) {
    const W = S.W, H = S.H, halo = r + 1.5;
    for (const [paint, rad] of [[S.pal.surface, halo], [color, r]]) {
      ctx.fillStyle = paint;
      ctx.beginPath();
      for (let j = 0; j < indices.length; j++) {
        const i = indices[j];
        if (filter && !filter(i)) continue;
        const sx = toSX(S.x[i]), sy = toSY(S.y[i]);
        if (sx < -halo || sy < -halo || sx > W + halo || sy > H + halo) continue;
        ctx.moveTo(sx + rad, sy); ctx.arc(sx, sy, rad, 0, TAU);
      }
      ctx.fill();
    }
  }

  // Candidate shapes are hollow (a spot where little exists), drawn in ink over a surface halo.
  function markerPath(kind, sx, sy, s) {
    ctx.beginPath();
    if (kind === "pair") { ctx.moveTo(sx, sy - s); ctx.lineTo(sx + s, sy); ctx.lineTo(sx, sy + s); ctx.lineTo(sx - s, sy); ctx.closePath(); }
    else if (kind === "blend") { ctx.moveTo(sx, sy - s); ctx.lineTo(sx + s * 0.95, sy + s * 0.75); ctx.lineTo(sx - s * 0.95, sy + s * 0.75); ctx.closePath(); }
    else ctx.arc(sx, sy, s * 0.8, 0, TAU);
  }
  function drawMarker(c, sx, sy, s, alpha) {
    ctx.globalAlpha = alpha;
    markerPath(c.kind, sx, sy, s);
    ctx.lineWidth = 5; ctx.strokeStyle = S.pal.surface; ctx.stroke();
    ctx.fillStyle = S.pal.surface; ctx.fill();
    ctx.lineWidth = 2; ctx.strokeStyle = S.pal.text; ctx.stroke();
    if (c.kind === "connection") { ctx.beginPath(); ctx.arc(sx, sy, 1.8, 0, TAU); ctx.fillStyle = S.pal.text; ctx.fill(); }
    ctx.globalAlpha = 1;
  }

  function boxesOverlap(a, b) { return a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h; }

  function haloText(text, x, y, color, placed, align = "center") {
    const w = ctx.measureText(text).width;
    const box = { x: align === "center" ? x - w / 2 - 3 : x - 2, y: y - 9, w: w + 6, h: 17 };
    if (placed) {
      if (box.x < 2 || box.y < 2 || box.x + box.w > S.W - 2 || box.y + box.h > S.H - 2) return false;
      if (placed.some((p) => boxesOverlap(p, box))) return false;
      placed.push(box);
    }
    ctx.textAlign = align;
    ctx.lineWidth = 3.5;
    ctx.lineJoin = "round";
    ctx.strokeStyle = S.pal.surface;
    ctx.strokeText(text, x, y);
    ctx.fillStyle = color;
    ctx.fillText(text, x, y);
    return true;
  }

  function colorOf(i) {
    if (S.mode === "loneliness") return S.pal.seq[S.bin[i]];
    const nb = S.d.neighborhoods[S.cluster[i]];
    return nb ? S.pal.cat[slotOf(nb)] : S.pal.unclustered;
  }

  // Unpinned candidate markers that would overlap gather into one numbered clump; pick it to zoom in.
  function clumpMarkers(except) {
    const clumps = [];
    for (const c of S.mapCands) {
      if (c === except) continue;
      const sx = toSX(c.x), sy = toSY(c.y);
      if (sx < -20 || sy < -20 || sx > S.W + 20 || sy > S.H + 20) continue;
      const near = clumps.find((g) => Math.hypot(g.sx - sx, g.sy - sy) < CLUMP_PX);
      if (near) { near.members.push(c); near.sx = (near.sx * (near.members.length - 1) + sx) / near.members.length; near.sy = (near.sy * (near.members.length - 1) + sy) / near.members.length; }
      else clumps.push({ sx, sy, members: [c] });
    }
    for (const g of clumps) g.key = g.members.map((c) => c.code).join(",");
    return clumps;
  }

  function draw() {
    S.raf = 0;
    if (rafTimer) { clearTimeout(rafTimer); rafTimer = 0; }
    if (!S.d || !S.W) return;
    const P = S.pal;
    ctx.setTransform(S.dpr, 0, 0, S.dpr, 0, 0);
    ctx.fillStyle = P.surface;
    ctx.fillRect(0, 0, S.W, S.H);

    const r = dotRadius();
    const s = 7;
    const dense = S.n > 5000;
    const dim = 0.12;
    const focusCand = S.focusCand;
    const focus = focusCand ? candFocus(focusCand) : null;
    // lit neighborhoods: the pinned candidate's, or the one isolated from the list; everything else dims
    const lit = focus ? focus.nbs : S.selectedNb !== null ? new Set([S.selectedNb]) : null;
    const isLit = lit ? (i) => lit.has(S.cluster[i]) : null;
    const isUnlit = lit ? (i) => !lit.has(S.cluster[i]) : null;
    const sizeOf = (i) => (S.mode === "loneliness" && S.bin[i] >= LOUD_FROM ? r * BIN_SCALE[S.bin[i]] : r);
    const ring = (sx, sy, rad, color, width) => {
      ctx.beginPath(); ctx.arc(sx, sy, rad, 0, TAU);
      ctx.lineWidth = width + 3; ctx.strokeStyle = P.surface; ctx.stroke();
      ctx.lineWidth = width; ctx.strokeStyle = color; ctx.stroke();
    };

    if (S.mode === "neighborhood") {
      const a = dense ? 0.7 : 0.9;
      drawPoints(S.unclustered, P.unclustered, (lit ? dim : 1) * (dense ? 0.4 : 0.55), r);
      for (const nb of S.d.neighborhoods) {
        if (lit && lit.has(nb.id)) continue;
        drawPoints(S.nbMembers.get(nb.id) || [], P.cat[slotOf(nb)], lit ? dim : a, r);
      }
      if (lit) for (const id of lit) { const nb = S.d.neighborhoods[id]; if (nb) drawPoints(S.nbMembers.get(id) || [], P.cat[slotOf(nb)], 1, r * 1.15); }
    } else {
      const quiet = QUIET_ALPHA[dense ? "dense" : "sparse"];
      for (let b = 0; b < S.byBin.length; b++) {
        if (lit) drawPoints(S.byBin[b], P.seq[b], dim, r, isUnlit);
        else if (b >= LOUD_FROM) drawLoud(S.byBin[b], P.seq[b], r * BIN_SCALE[b], null);
        else drawPoints(S.byBin[b], P.seq[b], quiet[b], r, null);
      }
      if (lit) {
        for (let b = 0; b < S.byBin.length; b++) {
          if (b >= LOUD_FROM) drawLoud(S.byBin[b], P.seq[b], r * 1.15 * BIN_SCALE[b], isLit);
          else drawPoints(S.byBin[b], P.seq[b], 1, r * 1.15, isLit);
        }
      }
    }
    // a pinned candidate's own items at full strength, wherever they sit
    if (focus) {
      for (const i of focus.items) {
        if (S.mode === "loneliness" && S.bin[i] >= LOUD_FROM) drawLoud([i], colorOf(i), r * 1.15 * BIN_SCALE[S.bin[i]], null);
        else drawPoints([i], colorOf(i), 1, r * 1.15);
      }
    }

    // Candidate markers: the pinned one alone; the rest faded while one is pinned, clumped where they overlap
    S.clumps = clumpMarkers(focusCand);
    const hasClumps = S.clumps.some((g) => g.members.length > 1);
    if (hasClumps !== S.legendClumps) { S.legendClumps = hasClumps; renderLegend(); }
    for (const g of S.clumps) {
      if (g.members.length === 1) { drawMarker(g.members[0], g.sx, g.sy, s, focus ? 0.35 : 1); continue; }
      ctx.globalAlpha = focus ? 0.35 : 1;
      ctx.beginPath(); ctx.arc(g.sx, g.sy, 10, 0, TAU);
      ctx.lineWidth = 5; ctx.strokeStyle = P.surface; ctx.stroke();
      ctx.fillStyle = P.surface; ctx.fill();
      ctx.lineWidth = 1.75; ctx.strokeStyle = P.text; ctx.stroke();
      ctx.font = '600 11px "Atkinson Hyperlegible Mono", ui-monospace, monospace';
      ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillStyle = P.text;
      ctx.fillText(String(g.members.length), g.sx, g.sy + 0.5);
      ctx.globalAlpha = 1;
    }
    if (focusCand && focusCand.x !== null) drawMarker(focusCand, toSX(focusCand.x), toSY(focusCand.y), s, 1);

    // A pinned candidate's marks: dotted outlines on the items it was built from, cobalt rings on
    // its must-not-copy items, and its drafts (a filled square passed, a hollow one needs another try)
    if (focus) {
      ctx.setLineDash([2, 2]);
      for (const i of focus.parents) {
        ctx.beginPath(); ctx.arc(toSX(S.x[i]), toSY(S.y[i]), sizeOf(i) + 3.5, 0, TAU);
        ctx.lineWidth = 1.5; ctx.strokeStyle = P.text; ctx.stroke();
      }
      ctx.setLineDash([]);
      for (const i of focus.nearest) ring(toSX(S.x[i]), toSY(S.y[i]), sizeOf(i) + 4, P.cobalt, 1.75);
      for (const d of focus.drafts) {
        if (d.x === null) continue;
        const sx = toSX(d.x), sy = toSY(d.y), h = 5;
        ctx.beginPath(); ctx.rect(sx - h, sy - h, 2 * h, 2 * h);
        ctx.lineWidth = 4; ctx.strokeStyle = P.surface; ctx.stroke();
        if (d.verdict === "pass") { ctx.fillStyle = P.text; ctx.fill(); }
        else { ctx.fillStyle = P.surface; ctx.fill(); ctx.lineWidth = 1.75; ctx.strokeStyle = P.text; ctx.stroke(); }
      }
    }

    // Labels, most important first so they win the space: the pinned candidate's code, the lit
    // neighborhoods, its drafts and item numbers, other codes, then (with nothing focused) every neighborhood
    const placed = [];
    ctx.textBaseline = "middle";
    for (const g of S.clumps) placed.push({ x: g.sx - s - 3, y: g.sy - s - 3, w: 2 * s + 6, h: 2 * s + 6 });
    const monoFont = '500 11px "Atkinson Hyperlegible Mono", ui-monospace, monospace';
    ctx.font = monoFont;
    if (focusCand && focusCand.x !== null) haloText(focusCand.code, toSX(focusCand.x) + s + 12, toSY(focusCand.y), P.text, placed, "left");
    const nbLabel = (nb) => haloText(S.nbShort[nb.id], toSX(nb.x), toSY(nb.y), P.ink, placed);
    const bySize = S.bySize;
    ctx.font = '600 12px "Atkinson Hyperlegible Next", Verdana, sans-serif';
    if (lit) for (const nb of bySize) if (lit.has(nb.id)) nbLabel(nb);
    ctx.font = monoFont;
    if (focus) {
      for (const d of focus.drafts) if (d.x !== null) placed.push({ x: toSX(d.x) - 7, y: toSY(d.y) - 7, w: 14, h: 14 });
      for (const d of focus.drafts) if (d.x !== null) haloText(d.code, toSX(d.x) + 9, toSY(d.y), P.text, placed, "left");
      ctx.font = '600 11px "Atkinson Hyperlegible Mono", ui-monospace, monospace';
      focus.nearest.forEach((i, k) => {
        const sx = toSX(S.x[i]), sy = toSY(S.y[i]), off = r + 6;
        if (!haloText(String(k + 1), sx + off, sy - off, P.cobalt, placed, "left")) haloText(String(k + 1), sx - off - 8, sy - off, P.cobalt, placed, "left");
      });
      ctx.font = monoFont;
    } else {
      for (const g of S.clumps) if (g.members.length === 1) haloText(g.members[0].code, g.sx + s + 6, g.sy, P.text, placed, "left");
      ctx.font = '600 12px "Atkinson Hyperlegible Next", Verdana, sans-serif';
      if (!lit) for (const nb of bySize) nbLabel(nb);
    }

    // Selection and hover rings (selection is cobalt, hover is text color)
    if (S.detail && S.detail.type === "item") ring(toSX(S.x[S.detail.i]), toSY(S.y[S.detail.i]), sizeOf(S.detail.i) + 7, P.cobalt, 2.5);
    if (focusCand && focusCand.x !== null) ring(toSX(focusCand.x), toSY(focusCand.y), s + 6, P.cobalt, 2.5);
    if (S.detail && S.detail.type === "draft" && S.detail.draft.x !== null) ring(toSX(S.detail.draft.x), toSY(S.detail.draft.y), 10, P.cobalt, 2.5);
    const h = S.hover;
    if (h && h.type === "item") ring(toSX(S.x[h.i]), toSY(S.y[h.i]), sizeOf(h.i) + 3.5, P.text, 1.5);
    if (h && h.type === "cand" && h.cand.x !== null) ring(toSX(h.cand.x), toSY(h.cand.y), s + 4, P.text, 1.5);
    if (h && h.type === "clump") ring(h.clump.sx, h.clump.sy, 14, P.text, 1.5);
    if (h && h.type === "draft" && h.draft.x !== null) ring(toSX(h.draft.x), toSY(h.draft.y), 9, P.text, 1.5);
  }

  // ---------- hit testing ----------

  function hitTest(px, py, touch = false) {
    const reachMarker = touch ? 18 : 12;
    if (S.focusCand) {
      for (const d of candFocus(S.focusCand).drafts) {
        if (d.x !== null && Math.hypot(toSX(d.x) - px, toSY(d.y) - py) <= (touch ? 16 : 10)) return { type: "draft", cand: S.focusCand, draft: d };
      }
      const c = S.focusCand;
      if (c.x !== null && Math.hypot(toSX(c.x) - px, toSY(c.y) - py) <= reachMarker) return { type: "cand", cand: c };
    }
    for (const g of S.clumps) {
      if (Math.hypot(g.sx - px, g.sy - py) <= reachMarker) return g.members.length === 1 ? { type: "cand", cand: g.members[0] } : { type: "clump", clump: g };
    }
    const r = dotRadius();
    const reach = touch ? Math.max(18, r + 10) : Math.max(10, r + 6);
    const { G, sx, sy, start, items } = S.grid;
    const { minX, minY } = S.bounds;
    const dx = toDX(px), dy = toDY(py), rd = reach / S.k;
    // dimmed points are not targets: only the pinned candidate's neighborhoods and items, or the isolated neighborhood
    const f = S.focusCand ? candFocus(S.focusCand) : null;
    const sel = S.selectedNb;
    const allowed = f ? (i) => f.nbs.has(S.cluster[i]) || f.items.has(i) : sel !== null ? (i) => S.cluster[i] === sel : null;
    const c0 = Math.max(0, Math.floor((dx - rd - minX) * sx)), c1 = Math.min(G - 1, Math.floor((dx + rd - minX) * sx));
    const r0 = Math.max(0, Math.floor((dy - rd - minY) * sy)), r1 = Math.min(G - 1, Math.floor((dy + rd - minY) * sy));
    let best = -1, bestD = reach;
    for (let row = r0; row <= r1; row++) {
      for (let c = c0; c <= c1; c++) {
        const cell = row * G + c;
        for (let j = start[cell]; j < start[cell + 1]; j++) {
          const i = items[j];
          if (allowed && !allowed(i)) continue;
          const d = Math.hypot(toSX(S.x[i]) - px, toSY(S.y[i]) - py);
          if (d <= bestD) { bestD = d; best = i; }
        }
      }
    }
    return best >= 0 ? { type: "item", i: best } : null;
  }
  const sameHit = (a, b) => (!a && !b) || (a && b && a.type === b.type && a.i === b.i && a.cand === b.cand && a.draft === b.draft && (a.clump && a.clump.key) === (b.clump && b.clump.key));

  function showTooltip(hit, px, py) {
    if (!hit) { tooltip.hidden = true; return; }
    if (hit.type === "item") {
      const i = hit.i;
      const meta = [itemVenue(i), itemYear(i)].filter(Boolean).join(", ");
      const extra = S.mode === "loneliness" ? `loneliness percentile ${S.lonely[i].toFixed(1)}` : nbShort(S.cluster[i]);
      fill(tooltip, el("strong", {}, t(S.d.items.title[i])), meta ? el("span", {}, t(meta)) : null, el("span", {}, t(extra)));
    } else if (hit.type === "cand") {
      const c = hit.cand;
      fill(tooltip, el("strong", {}, `${c.code} · `, t(c.label)), el("span", { text: `${KIND_NAMES[c.kind]} · loneliness percentile ${c.score.lonely.toFixed(1)}` }));
    } else if (hit.type === "draft") {
      const d = hit.draft;
      fill(tooltip, el("strong", {}, `${hit.cand.code} ${d.code} · `, t(d.title)), el("span", { text: `${VERDICT_NAMES[d.verdict]} · nearest real item ${d.nn1.toFixed(3)} · loneliness percentile ${d.lonely.toFixed(1)}` }));
    } else {
      const codes = hit.clump.members.map((c) => c.code);
      fill(tooltip, el("strong", { text: `${codes.length} candidates: ${codes.slice(0, 8).join(", ")}${codes.length > 8 ? " …" : ""}` }), el("span", { text: "Select to zoom in" }));
    }
    tooltip.hidden = false;
    const tw = tooltip.offsetWidth, th = tooltip.offsetHeight;
    let left = px + 14, top = py + 14;
    if (left + tw > S.W - 8) left = Math.max(8, px - tw - 14);
    if (top + th > S.H - 8) top = Math.max(8, py - th - 14);
    tooltip.style.left = `${left}px`;
    tooltip.style.top = `${top}px`;
  }

  // ---------- details panel ----------

  function linkButton(label, onclick, extra) {
    return el("button", { type: "button", class: "row-btn two-col", onclick }, el("span", { class: "main" }, t(label)), extra || null);
  }
  function facts(rows) {
    return el("dl", { class: "facts" }, rows.filter(Boolean).flatMap(([k, v]) => [el("dt", { text: k }), el("dd", {}, v)]));
  }

  // Keep a row under the pointer while the panel above it changes height.
  function keepInPlace(node, change) {
    const lists = document.querySelector(".side-lists");
    const before = node && node.isConnected ? node.getBoundingClientRect().top : null;
    change();
    if (before !== null && node.isConnected && !narrow.matches) lists.scrollTop += node.getBoundingClientRect().top - before;
  }

  // detail: null | {type: "item", i, via?} | {type: "cand", cand} | {type: "draft", cand, draft} | {type: "nb", id}
  // via: {cand, draft?, list: "nearest" | "parents" | "draftNearest", index} — how an item was reached from a candidate
  // from: "map" (canvas), "list" (a list outside the panel) or "details" (a button inside the
  // panel, which the re-render removes, so focus moves to the new heading instead of the body).
  function setDetail(detail, { center = false, say = true, from = "map", anchor = null } = {}) {
    const previousFocus = S.focusCand;
    S.detail = detail;
    // a pinned candidate stays in focus while its own items and drafts are opened; anything else releases it
    if (!detail || detail.type === "nb") S.focusCand = null;
    else if (detail.type === "cand" || detail.type === "draft") { S.focusCand = detail.cand; S.selectedNb = null; }
    else if (detail.type === "item" && S.focusCand && !candFocus(S.focusCand).related.has(detail.i)) S.focusCand = null;
    const box = $("details");
    $("details-clear").hidden = !detail;
    keepInPlace(anchor, () => {
      $("details-card").classList.toggle("is-pinned", !!detail);
      if (!detail) fill(box, el("p", { class: "empty", text: EMPTY_DETAILS }));
      else if (detail.type === "item") renderItem(box, detail);
      else if (detail.type === "cand") renderCandidate(box, detail.cand);
      else if (detail.type === "draft") renderDraft(box, detail.cand, detail.draft);
      else if (detail.type === "nb") renderNeighborhood(box, detail.id);
    });
    if (detail && detail.type === "item") {
      if (center) {
        const sx = toSX(S.x[detail.i]), sy = toSY(S.y[detail.i]);
        const inView = sx > 0 && sy > 0 && sx < S.W && sy < S.H;
        if (!(S.focusCand && inView)) centerOn(S.x[detail.i], S.y[detail.i], S.focusCand ? 0 : 4);
      }
      if (say) announce(`Pinned: ${S.d.items.title[detail.i]}`);
    } else if (detail && detail.type === "cand") {
      fitToCandidate(detail.cand);
      const c = detail.cand, nd = c.drafts.length;
      if (say) announce(`Showing ${c.code}, ${c.label}, with its ${c.nearest.length} must-not-copy items numbered on the map${nd ? ` and ${nd} ${nd === 1 ? "draft" : "drafts"}` : ""}`);
    } else if (detail && detail.type === "draft") {
      const d = detail.draft;
      if (d.x !== null) {
        const sx = toSX(d.x), sy = toSY(d.y);
        if (!(sx > 0 && sy > 0 && sx < S.W && sy < S.H)) centerOn(d.x, d.y, 0);
      } else if (previousFocus !== detail.cand || from === "list") fitToCandidate(detail.cand);
      if (say) announce(`Showing draft ${d.code} for ${detail.cand.code}: ${VERDICT_NAMES[d.verdict]}`);
    } else if (detail && detail.type === "nb" && say) announce(`Showing ${nbLong(detail.id)}`);
    syncPressed();
    renderLegend();
    syncHash();
    requestDraw();
    const heading = box.querySelector("h3");
    if (from === "details" && heading) heading.focus();
    else if (from === "list" && detail && narrow.matches && heading) { heading.focus({ preventScroll: true }); $("details-card").scrollIntoView({ block: "start", behavior: reducedMotion.matches ? "auto" : "smooth" }); }
    else if (from === "list" && detail) $("details-card").scrollTop = 0;
  }

  // One step back: an item reached from a candidate returns to it (or to its draft), a draft to its
  // candidate, anything else to nothing.
  function stepBack() {
    const d = S.detail;
    if (!d && S.selectedNb === null) return false;
    if (d && d.type === "item" && d.via) {
      if (d.via.draft) setDetail({ type: "draft", cand: d.via.cand, draft: d.via.draft }, { from: "details" });
      else setDetail({ type: "cand", cand: d.via.cand }, { from: "details" });
    } else if (d && d.type === "draft") setDetail({ type: "cand", cand: d.cand }, { from: "details" });
    else {
      S.selectedNb = null;
      setDetail(null, { say: false });
      announce("Details cleared");
    }
    return true;
  }

  function viaList(via) {
    if (via.list === "nearest") return via.cand.nearest.map(([i]) => i);
    if (via.list === "parents") return via.cand.parents;
    return via.draft.nearest.map(([i]) => i);
  }
  const VIA_NAMES = { nearest: "Must not copy", parents: "Built from", draftNearest: "Nearest real item" };

  function renderTrail(via) {
    const list = viaList(via);
    const back = via.draft
      ? () => setDetail({ type: "draft", cand: via.cand, draft: via.draft }, { from: "details" })
      : () => setDetail({ type: "cand", cand: via.cand }, { from: "details" });
    const go = (k) => () => setDetail({ type: "item", i: list[k], via: { ...via, index: k } }, { center: true, from: "details" });
    return el("div", { class: "trail" },
      el("button", { type: "button", class: "link-btn", onclick: back }, "← ", via.draft ? `${via.cand.code} ${via.draft.code}` : via.cand.code, " ", t(via.draft ? via.draft.title : via.cand.label)),
      el("span", { class: "step" },
        el("span", { class: "pc-muted", text: `${VIA_NAMES[via.list]} ${via.index + 1} of ${list.length}` }),
        el("button", { type: "button", class: "pc-btn", disabled: via.index === 0, "aria-label": "Previous", onclick: go(via.index - 1) }, "‹"),
        el("button", { type: "button", class: "pc-btn", disabled: via.index === list.length - 1, "aria-label": "Next", onclick: go(via.index + 1) }, "›")));
  }

  function renderItem(box, detail) {
    const i = detail.i;
    const url = itemLink(i);
    const c = S.cluster[i];
    const nbCell = c >= 0 ? el("button", { type: "button", class: "link-btn", onclick: () => selectNeighborhood(c, "details") }, t(nbLong(c))) : "Unclustered";
    const cands = S.candsByItem.get(i) || [];
    const snippet = col("snippet") ? col("snippet")[i] : "";
    fill(box,
      detail.via ? renderTrail(detail.via) : null,
      el("h3", { tabindex: "-1" }, t(S.d.items.title[i])),
      snippet ? el("p", { class: "snippet" }, t(snippet)) : null,
      facts([
        itemYear(i) !== null ? ["Year", mono(String(itemYear(i)))] : null,
        itemVenue(i) ? ["Venue", t(itemVenue(i))] : null,
        itemSource(i) ? ["Source", t(itemSource(i))] : null,
        ["Neighborhood", nbCell],
        ["Loneliness percentile", mono(S.lonely[i].toFixed(1))],
        ["Nearest-neighbor similarity", similarity(S.d.items.nn1[i])],
        cands.length ? ["Must not copy for", el("span", {}, cands.map((cand, k) => [k ? ", " : "", el("button", { type: "button", class: "link-btn mono", onclick: () => setDetail({ type: "cand", cand }, { from: "details" }) }, cand.code)]))] : null,
      ]),
      el("div", { class: "pc-btn-row" },
        url ? el("a", { class: "pc-btn", href: url.href, target: "_blank", rel: "noopener noreferrer" }, "Open the source page") : null,
        el("button", { type: "button", class: "pc-btn", onclick: () => centerOn(S.x[i], S.y[i], 4) }, "Center on item"),
      ),
    );
  }

  function itemRows(pairs, via) {
    const open = (i, k) => () => setDetail({ type: "item", i, via: via ? { ...via, index: k } : undefined }, { center: true, from: "details" });
    return el("ul", { class: "row-list inline-list" }, pairs.map(([i, sim], k) => el("li", {},
      via && via.list === "nearest"
        ? el("button", { type: "button", class: "row-btn", onclick: open(i, k) },
            el("span", { class: "num rank", text: String(k + 1) }), el("span", { class: "main" }, t(S.d.items.title[i])), el("span", { class: "num" }, hidden("similarity "), sim.toFixed(3)))
        : linkButton(S.d.items.title[i], open(i, k), sim === undefined ? null : el("span", { class: "num" }, hidden("similarity "), sim.toFixed(3))),
    )));
  }

  const EVIDENCE = [
    ["mixNn1", "Mixture's nearest similarity", (v) => v.toFixed(3)],
    ["baseline", "Typical for matched mixtures", (v) => v.toFixed(3)],
    ["margin", "Margin over chance", (v) => v.toFixed(3)],
    ["beats", "Emptier than matched mixtures", (v) => `${v.toFixed(1)}%`],
    ["pairTail", "Pair's share in the emptiest tail", (v) => `${v.toFixed(1)}%`],
    ["pairP", "Probability by chance", (v) => (v < 0.001 ? v.toExponential(1) : v.toPrecision(2))],
  ];

  function renderCandidate(box, c) {
    const nbs = c.neighborhoods;
    fill(box,
      el("h3", { tabindex: "-1" }, el("span", { class: "mono", text: c.code }), " ", t(c.label)),
      c.headline ? el("p", { class: "pc-muted" }, t(c.headline)) : null,
      nbs.length ? el("p", { class: "pc-label sub-label", text: "Neighborhoods" }) : null,
      nbs.length ? el("ul", { class: "row-list inline-list" }, nbs.map((id) => el("li", {}, linkButton(nbLong(id), () => selectNeighborhood(id, "details"))))) : null,
      c.drafts.length ? headRow("Drafts", "Nearest similarity") : null,
      c.drafts.length ? el("ul", { class: "row-list inline-list" }, c.drafts.map((d) => el("li", {},
        el("button", { type: "button", class: "row-btn", onclick: () => setDetail({ type: "draft", cand: c, draft: d }, { from: "details" }) },
          el("span", { class: "mono", text: d.code }),
          el("span", { class: "main" }, t(d.title), el("span", { class: "sub" }, verdictBadge(d), ` loneliness percentile ${d.lonely.toFixed(1)}`)),
          el("span", { class: "num" }, hidden("nearest similarity "), d.nn1.toFixed(3)))))) : null,
      headRow("Must not copy", "Similarity"),
      itemRows(c.nearest, { cand: c, list: "nearest" }),
      c.parents.length ? el("p", { class: "pc-label sub-label", text: "Built from" }) : null,
      c.parents.length ? itemRows(c.parents.map((i) => [i]), { cand: c, list: "parents" }) : null,
      el("p", { class: "pc-label sub-label", text: "Measurements" }),
      facts([
        ["Kind", KIND_NAMES[c.kind]],
        ["Loneliness percentile", mono(c.score.lonely.toFixed(1))],
        ["Nearest-neighbor similarity", similarity(c.score.nn1)],
        ...EVIDENCE.filter(([key]) => typeof c.score[key] === "number").map(([key, label, fmt]) => [label, mono(fmt(c.score[key]))]),
        c.x === null ? ["On the map", "Not shown"] : null,
      ]),
      c.tags.length ? el("p", { class: "pc-label sub-label", text: "Leans toward" }) : null,
      c.tags.length ? el("p", { class: "tag-list" }, t(c.tags.join(", "))) : null,
      c.brief ? el("details", { class: "disclosure" }, el("summary", { text: "Writing brief" }), el("p", { class: "prose" }, t(c.brief))) : null,
      el("div", { class: "pc-btn-row" }, el("button", { type: "button", class: "pc-btn", onclick: () => fitToCandidate(c) }, `Fit to ${c.code}`)),
    );
  }

  function renderDraft(box, c, d) {
    const back = () => setDetail({ type: "cand", cand: c }, { from: "details" });
    fill(box,
      el("div", { class: "trail" }, el("button", { type: "button", class: "link-btn", onclick: back }, "← ", c.code, " ", t(c.label))),
      el("h3", { tabindex: "-1" }, el("span", { class: "mono", text: `${c.code} ${d.code}` }), " ", t(d.title)),
      facts([
        ["Verdict", verdictBadge(d)],
        ["Nearest-neighbor similarity", similarity(d.nn1)],
        ["Loneliness percentile", mono(d.lonely.toFixed(1))],
        typeof d.toCandidate === "number" ? ["Similarity to the candidate", mono(d.toCandidate.toFixed(3))] : null,
        d.checked ? ["Checked", mono(d.checked.slice(0, 10))] : null,
        d.x === null ? ["On the map", "Not shown"] : null,
      ]),
      d.reasons.length ? el("p", { class: "pc-label sub-label", text: "Why it needs another try" }) : null,
      d.reasons.map((r) => el("p", { class: "tag-list" }, t(r))),
      el("details", { class: "disclosure", open: true }, el("summary", { text: "Text" }), el("p", { class: "prose" }, t(d.text))),
      d.notes ? el("details", { class: "disclosure" }, el("summary", { text: "Notes" }), el("p", { class: "prose" }, t(d.notes))) : null,
      headRow("Nearest real items", "Similarity"),
      itemRows(d.nearest, { cand: c, draft: d, list: "draftNearest" }),
      el("div", { class: "pc-btn-row" },
        el("button", { type: "button", class: "pc-btn", onclick: back }, `Back to ${c.code}`),
        d.x !== null ? el("button", { type: "button", class: "pc-btn", onclick: () => centerOn(d.x, d.y, 0) }, "Center on draft") : null,
      ),
    );
  }

  function renderNeighborhood(box, id) {
    const nb = S.d.neighborhoods[id];
    fill(box,
      el("h3", { tabindex: "-1" }, t(nbLong(id))),
      facts([
        ["Items", mono(String(nb.size))],
        ["Median nearest-neighbor similarity", mono(nb.density.toFixed(3))],
        typeof nb.popularity === "number" && nb.popularity > 0 ? ["Popularity", mono(nb.popularity.toLocaleString())] : null,
        ["Neighborhood number", mono(String(id))],
      ]),
      nb.tags.length ? el("p", { class: "pc-label sub-label", text: "Tags" }) : null,
      nb.tags.length ? el("p", { class: "tag-list" }, t(nb.tags.join(", "))) : null,
      nb.examples.length ? el("p", { class: "pc-label sub-label", text: "Central items" }) : null,
      nb.examples.length ? itemRows(nb.examples.map((i) => [i])) : null,
      el("div", { class: "pc-btn-row" }, el("button", { type: "button", class: "pc-btn", onclick: () => zoomToNeighborhood(id) }, "Fit to neighborhood")),
    );
  }

  function selectNeighborhood(id, from = "list", anchor = null) {
    if (S.selectedNb === id && S.detail && S.detail.type === "nb" && from === "list") {
      S.selectedNb = null;
      setDetail(null, { say: false, anchor });
      announce("Showing all neighborhoods");
      return;
    }
    S.selectedNb = id;
    setDetail({ type: "nb", id }, { from, anchor });
    zoomToNeighborhood(id);
  }

  function zoomToNeighborhood(id) {
    const m = S.nbMembers.get(id);
    if (!m || !m.length) return;
    let box = S.nbBox.get(id);
    if (!box) {
      const xs = Float32Array.from(m, (i) => S.x[i]).sort();
      const ys = Float32Array.from(m, (i) => S.y[i]).sort();
      const q = (a, p) => a[Math.min(a.length - 1, Math.max(0, Math.round(p * (a.length - 1))))];
      box = [q(xs, 0.03), q(xs, 0.97), q(ys, 0.03), q(ys, 0.97)];
      S.nbBox.set(id, box);
    }
    fitTo(box[0], box[1], box[2], box[3], 0.75, 6);
    requestDraw();
  }

  // ---------- lists, legend and the map's provenance ----------

  function syncPressed() {
    for (const b of document.querySelectorAll("#nb-list .row-btn")) b.setAttribute("aria-pressed", String(Number(b.dataset.id) === S.selectedNb));
    const code = S.focusCand ? S.focusCand.code : null;
    for (const b of document.querySelectorAll("#cand-list .row-btn")) b.setAttribute("aria-pressed", String(b.dataset.code === code));
    const draftId = S.detail && S.detail.type === "draft" ? S.detail.draft.id : null;
    for (const b of document.querySelectorAll("#draft-list .row-btn")) b.setAttribute("aria-pressed", String(b.dataset.id === draftId));
  }

  function renderLists() {
    const cands = S.d.candidates;
    fill($("cand-list"), cands.length ? cands.map((c) => el("li", {},
      el("button", { type: "button", class: "row-btn", "data-code": c.code, "aria-pressed": "false", onclick: (e) => setDetail({ type: "cand", cand: c }, { from: "list", anchor: e.currentTarget }) },
        el("span", { class: "lead" }, markerIcon(c.kind), el("span", { class: "mono", text: c.code })),
        el("span", { class: "main" }, t(c.label), el("span", { class: "sub", text: candSubline(c) })),
        el("span", { class: "num" }, hidden("loneliness percentile "), c.score.lonely.toFixed(1)),
      ))) : el("li", { class: "list-empty", text: "This map has no candidate connections yet." }));

    $("drafts-card").hidden = S.drafts.length === 0;
    fill($("draft-list"), S.drafts.map(({ draft: d, cand: c }) => el("li", {},
      el("button", { type: "button", class: "row-btn", "data-id": d.id, "aria-pressed": "false", onclick: (e) => setDetail({ type: "draft", cand: c, draft: d }, { from: "list", anchor: e.currentTarget }) },
        el("span", { class: "lead" }, markerIcon(d.verdict === "pass" ? "draftPass" : "draftRetry"), el("span", { class: "mono", text: `${c.code} ${d.code}` })),
        el("span", { class: "main" }, t(d.title), el("span", { class: "sub" }, verdictBadge(d), ` loneliness percentile ${d.lonely.toFixed(1)}`)),
        el("span", { class: "num" }, hidden("nearest similarity "), d.nn1.toFixed(3)),
      ))));

    const nbs = [...S.d.neighborhoods].sort((a, b) => b.size - a.size);
    fill($("nb-list"), nbs.length ? nbs.map((nb) => el("li", {},
      el("button", { type: "button", class: "row-btn", "data-id": nb.id, "aria-pressed": "false", onclick: (e) => selectNeighborhood(nb.id, "list", e.currentTarget) },
        el("span", { class: "swatch", style: `background: var(--map-c${slotOf(nb) + 1})`, "aria-hidden": "true" }),
        el("span", { class: "main" }, t(S.nbLong[nb.id]), nb.tags.length ? el("span", { class: "sub" }, t(nb.tags.slice(0, 3).join(" · "))) : null),
        el("span", { class: "num" }, String(nb.size), hidden(" items")),
      ))) : el("li", { class: "list-empty", text: "This map has no neighborhoods." }));
  }

  function renderLegend() {
    if (!S.d) return;
    const rows = [];
    for (const kind of ["connection", "pair", "blend"]) {
      const count = S.mapCands.filter((c) => c.kind === kind).length;
      if (count) rows.push(el("li", {}, markerIcon(kind), el("span", {}, el("strong", { text: KIND_NAMES[kind] }), el("span", { class: "pc-muted", text: ` · ${KIND_HINTS[kind]}` })), el("span", { class: "num", text: String(count) })));
    }
    if (S.clumps.some((g) => g.members.length > 1)) rows.push(el("li", {}, markerIcon("clump"), el("span", { text: "Several candidates; select to zoom in" })));
    if (S.focusCand) {
      const c = S.focusCand;
      rows.push(el("li", {}, markerIcon("nearest"), el("span", { text: `Must not copy for ${c.code} · numbers match Details` })));
      if (candFocus(c).parents.length) rows.push(el("li", {}, markerIcon("parent"), el("span", { text: `Built from, for ${c.code}` })));
      const drawn = c.drafts.filter((d) => d.x !== null);
      if (drawn.some((d) => d.verdict === "pass")) rows.push(el("li", {}, markerIcon("draftPass"), el("span", { text: `Draft for ${c.code} · passed` })));
      if (drawn.some((d) => d.verdict === "retry")) rows.push(el("li", {}, markerIcon("draftRetry"), el("span", { text: `Draft for ${c.code} · needs another try` })));
    }
    const box = $("legend");
    if (S.mode === "neighborhood") {
      fill(box,
        el("ul", { class: "legend-rows" }, rows,
          el("li", {}, el("span", { class: "swatch unclustered", "aria-hidden": "true" }), el("span", { text: "Item in no neighborhood" }), el("span", { class: "num", text: S.d.stats.unclustered.toLocaleString() }))),
        el("p", { class: "pc-hint", text: "Colors repeat across the map. Select a neighborhood in the list to isolate it." }),
      );
    } else {
      const counts = S.byBin.map((a) => a.length);
      fill(box,
        el("ul", { class: "legend-rows" }, rows),
        el("p", { class: "pc-label sub-label", text: "Loneliness percentile" }),
        el("ul", { class: "legend-rows" }, BIN_LABELS.map((label, b) => el("li", {},
          el("span", { class: "legend-dot", "aria-hidden": "true" }, el("span", {
            style: `background: var(--map-seq-${b + 1}); width: ${Math.round(7 * BIN_SCALE[b])}px; height: ${Math.round(7 * BIN_SCALE[b])}px;` +
              (b < LOUD_FROM ? ` opacity: ${QUIET_ALPHA[S.n > 5000 ? "dense" : "sparse"][b]};` : ""),
          })),
          el("span", { text: label }),
          el("span", { class: "num" }, counts[b].toLocaleString(), hidden(" items")),
        )).reverse()),
      );
    }
  }

  function sourceLine(src) {
    const link = src.url ? MF.safeLink(src.url) : null;
    const name = link ? el("a", { href: link.href, target: "_blank", rel: "noopener noreferrer", class: "source-name" }, t(src.name)) : el("span", { class: "source-name" }, t(src.name));
    const bits = [src.items !== undefined ? `${src.items.toLocaleString()} items` : "", src.retrieved ? `retrieved ${src.retrieved.slice(0, 10)}` : "", src.license || ""].filter(Boolean).join(" · ");
    return el("li", {}, name, bits ? el("span", { class: "pc-muted" }, ` · ${bits}`) : null,
      src.attribution ? el("div", {}, t(src.attribution)) : null,
      src.query ? el("div", { class: "pc-muted" }, t(src.query)) : null);
  }
  function paramList(params) {
    return facts(Object.entries(params || {}).map(([k, v]) => [k, mono(v === null ? "none" : String(v))]));
  }

  function renderAbout() {
    const d = S.d, m = d.manifest, e = m.embedding, corpus = m.corpus;
    fill($("about"),
      d.description ? el("p", { class: "tag-list" }, t(d.description)) : null,
      facts([
        ["Embedding model", el("span", {}, t(e.model), e.revision ? el("span", { class: "pc-muted" }, " @ ", t(e.revision.slice(0, 12))) : null)],
        ["Provider", t(e.provider)],
        ["Dimensions", mono(e.dimensions.toLocaleString())],
        e.maxTokens ? ["Text cut after", mono(`${e.maxTokens.toLocaleString()} tokens`)] : null,
        e.input ? ["Embedded", t(e.input)] : null,
        ["Items are", t(corpus.kind)],
        corpus.topic ? ["Topic", t(corpus.topic)] : null,
        ["Exported", mono(d.exported.slice(0, 10))],
        ["Made with", el("span", {}, t(`${m.generator.name} ${m.generator.version}`), m.generator.commit ? el("span", { class: "pc-muted" }, " · ", t(m.generator.commit.slice(0, 7))) : null)],
        corpus.fingerprint ? ["Corpus fingerprint", mono(corpus.fingerprint)] : null,
      ]),
      el("p", { class: "pc-label sub-label", text: "Sources" }),
      el("ul", {}, corpus.sources.map(sourceLine)),
      corpus.cleaning && corpus.cleaning.length ? el("details", { class: "disclosure" }, el("summary", { text: "Cleaning" }), el("ul", {}, corpus.cleaning.map((c) => el("li", {}, t(c))))) : null,
      m.map ? el("details", { class: "disclosure" }, el("summary", { text: "Map settings" }), paramList(m.map)) : null,
      m.candidates ? el("details", { class: "disclosure" }, el("summary", { text: "Candidate settings" }), paramList(m.candidates)) : null,
    );
  }

  // ---------- search ----------

  // Results come 25 at a time; "Show 25 more" pages on until every match is reachable.
  const PAGE = 25;
  let findTimer = 0;
  let findShown = PAGE;
  function runFind(keepPage = false) {
    const raw = $("find").value.trim();
    const q = raw.toLowerCase();
    const list = $("find-results");
    if (!keepPage) findShown = PAGE;
    const focused = list.contains(document.activeElement) ? document.activeElement.closest("[data-i]") : null;
    const focusedItem = focused ? Number(focused.dataset.i) : null;
    if (!S.d || !q) {
      if (focused) $("find").focus();
      list.hidden = true; list.replaceChildren(); $("find-status").textContent = ""; return;
    }
    if (!S.lower) S.lower = S.d.items.title.map((x) => x.toLowerCase());
    const hits = [];
    let total = 0;
    for (let i = 0; i < S.n; i++) if (S.lower[i].includes(q)) { total++; if (hits.length < findShown) hits.push(i); }
    fill(list,
      hits.map((i) => el("li", {}, el("button", { type: "button", class: "row-btn two-col", "data-i": i, onclick: () => setDetail({ type: "item", i }, { center: true, from: "list" }) },
        el("span", { class: "main" }, t(S.d.items.title[i])), itemYear(i) !== null ? el("span", { class: "num", text: String(itemYear(i)) }) : null))),
      total > hits.length ? el("li", {}, el("button", { type: "button", class: "row-btn two-col more", onclick: () => { const at = hits.length; findShown += PAGE; runFind(true); const next = list.querySelectorAll("[data-i]")[at]; if (next) next.focus(); } },
        el("span", { class: "main", text: `Show ${Math.min(PAGE, total - hits.length)} more` }), el("span", { class: "num", text: `${total - hits.length} left` }))) : null);
    list.hidden = !hits.length;
    $("find-status").textContent = total === 0 ? `No titles contain “${raw}”. Try a single word.` : total > hits.length ? `Showing ${hits.length} of ${total} matches.` : `${total} ${total === 1 ? "match" : "matches"}.`;
    // a refresh never strands keyboard focus: back to the same item, or to the search field
    if (focused) {
      const same = focusedItem !== null && list.querySelector(`[data-i="${focusedItem}"]`);
      (same || $("find")).focus();
    }
  }

  // ---------- "?" help ----------

  function closeHelp(btn, refocus) {
    btn.setAttribute("aria-expanded", "false");
    $(btn.getAttribute("aria-controls")).hidden = true;
    if (refocus) btn.focus();
  }
  function openHelp() { return [...document.querySelectorAll('.help-btn[aria-expanded="true"]')]; }
  for (const btn of document.querySelectorAll(".help-btn")) {
    btn.addEventListener("click", () => {
      const open = btn.getAttribute("aria-expanded") === "true";
      for (const other of openHelp()) closeHelp(other, false);
      if (!open) { btn.setAttribute("aria-expanded", "true"); $(btn.getAttribute("aria-controls")).hidden = false; }
    });
  }
  // A tap elsewhere closes an open "?" after that tap has done its own work: closing on
  // pointerdown would move the content under the pointer and send the click somewhere else.
  document.addEventListener("click", (e) => {
    for (const btn of openHelp()) {
      const pop = $(btn.getAttribute("aria-controls"));
      if (!btn.contains(e.target) && !pop.contains(e.target)) closeHelp(btn, false);
    }
  });

  // ---------- opening a map ----------

  // Each request to open a map takes a ticket; only the latest ticket may show its result, so a
  // slow sample download can never replace a file chosen after it.
  let ticket = 0;
  function busy(on, name) {
    for (const b of document.querySelectorAll("#open-file, #open-sample, #start-open, #start-sample")) b.disabled = on;
    const status = $("map-status");
    status.textContent = on ? `Reading ${name}…` : "";
    status.hidden = !on || S.d === null;
  }
  async function openBlob(blob, name, mine = ++ticket) {
    hideMessage();
    busy(true, name);
    announce(`Reading ${name}`);
    let result;
    try {
      try { result = await checkInWorker(blob); } catch (err) { result = await checkHere(blob); }
    } catch (err) {
      result = { error: "The file could not be checked." };
    } finally {
      if (mine === ticket) busy(false);
    }
    if (mine !== ticket) return;
    if (result.error) {
      showMessage(`${name} can't be opened.`, result.error);
      announce(`${name} can't be opened. ${result.error}`);
      return;
    }
    show(result.doc, name);
  }

  // Validation runs in a fresh worker so a large file never freezes the page; the worker ends with it.
  function checkInWorker(blob) {
    return new Promise((resolve, reject) => {
      let worker;
      try { worker = new Worker("mapfile-worker.js"); } catch (err) { reject(err); return; }
      const fail = (e) => { worker.terminate(); reject(e); };
      worker.onmessage = (e) => { worker.terminate(); resolve(e.data); };
      worker.onerror = fail;
      worker.onmessageerror = fail;
      try { worker.postMessage({ blob }); } catch (err) { fail(err); }
    });
  }
  async function checkHere(blob) {
    try {
      const response = await fetch("uncharted-export-v1.schema.json");
      return await MF.readMapFile(blob, await response.json());
    } catch (err) {
      return { error: "This page needs to be opened from a web address to check map files." };
    }
  }

  async function openSample(sample) {
    const mine = ++ticket;
    busy(true, sample.name);
    let blob;
    try {
      const response = await fetch(sample.file);
      if (!response.ok) throw new Error(String(response.status));
      blob = await response.blob();
    } catch (err) {
      if (mine !== ticket) return;
      busy(false);
      showMessage("The sample can't be loaded.", "Open this page from a web address, not from a file on disk.");
      return;
    }
    if (mine === ticket) await openBlob(blob, sample.name, mine);
  }

  // A new map replaces the old one entirely: nothing from the previous file is kept.
  let linkedCode = location.hash.slice(1); // a candidate code is capitals and digits, so no decoding
  function show(doc, name) {
    const wantedCode = linkedCode;
    linkedCode = "";
    pointers.clear();
    drag = null;
    canvas.classList.remove("is-dragging");
    tooltip.hidden = true;
    const keepMode = S.mode;
    if (S.raf) cancelAnimationFrame(S.raf);
    S = freshState();
    S.mode = keepMode;
    S.name = name;
    readPalette();
    prepare(doc);
    $("start").hidden = true;
    $("layout").hidden = false;
    $("import").hidden = false;
    const kind = doc.manifest.corpus.kind;
    $("map-title").textContent = doc.title;
    $("map-sub").textContent = `${doc.items.count.toLocaleString()} ${kind}${doc.items.count === 1 ? "" : "s"} · ${doc.manifest.embedding.model} (${doc.manifest.embedding.provider}, ${doc.manifest.embedding.dimensions.toLocaleString()} dimensions) · exported ${doc.exported.slice(0, 10)}`;
    $("map-sub").hidden = false;
    document.title = `${doc.title} · Uncharted`;
    $("stats").hidden = false;
    $("stat-items").textContent = doc.stats.items.toLocaleString();
    $("stat-nb").textContent = doc.stats.neighborhoods.toLocaleString();
    $("stat-candidates").textContent = doc.stats.candidates.toLocaleString();
    $("stat-drafts").textContent = doc.stats.drafts.toLocaleString();
    $("stat-drafts-box").hidden = doc.stats.drafts === 0;
    canvas.setAttribute("aria-label", `Map of ${doc.items.count.toLocaleString()} ${kind}s in ${doc.stats.neighborhoods} neighborhoods, with ${S.mapCands.length} candidate connections drawn. The candidate, draft, neighborhood and search lists reach the same items by keyboard.`);
    $("find").value = "";
    runFind();
    renderLists();
    renderAbout();
    setDetail(null, { say: false });
    resize();
    if (S.W < 640) $("legend-box").open = false;
    fitAll();
    renderLegend();
    requestDraw();
    $("map-title").focus();
    announce(`Opened ${doc.title}: ${doc.items.count.toLocaleString()} items, ${doc.stats.candidates} candidate connections${doc.stats.drafts ? `, ${doc.stats.drafts} drafts` : ""}.`);
    const cand = wantedCode && doc.candidates.find((c) => c.code === wantedCode);
    if (cand) setDetail({ type: "cand", cand }, { say: false });
  }

  // The pinned candidate's code rides in the address, so a link can open the same view.
  function syncHash() {
    const code = S.focusCand ? S.focusCand.code : "";
    const url = new URL(location.href);
    url.hash = code;
    if (url.href !== location.href) history.replaceState(null, "", code ? url.href : url.href.replace(/#$/, ""));
  }

  const fileInput = $("file-input");
  for (const id of ["open-file", "start-open"]) $(id).addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", () => {
    const file = fileInput.files && fileInput.files[0];
    if (file) openBlob(file, file.name);
    fileInput.value = "";
  });
  for (const id of ["open-sample", "start-sample"]) $(id).addEventListener("click", () => openSample(SAMPLES[0]));

  let dragDepth = 0;
  const hasFiles = (e) => e.dataTransfer && [...e.dataTransfer.types].includes("Files");
  window.addEventListener("dragenter", (e) => { if (!hasFiles(e)) return; e.preventDefault(); dragDepth++; $("drop-overlay").hidden = false; });
  window.addEventListener("dragover", (e) => { if (hasFiles(e)) e.preventDefault(); });
  window.addEventListener("dragleave", () => { if (--dragDepth <= 0) { dragDepth = 0; $("drop-overlay").hidden = true; } });
  window.addEventListener("drop", (e) => {
    if (!hasFiles(e)) return;
    e.preventDefault();
    dragDepth = 0;
    $("drop-overlay").hidden = true;
    const file = e.dataTransfer.files[0];
    if (file) openBlob(file, file.name);
  });

  // ---------- input ----------

  const pointers = new Map();
  let drag = null;

  canvas.addEventListener("pointerdown", (e) => {
    canvas.setPointerCapture(e.pointerId);
    pointers.set(e.pointerId, { x: e.offsetX, y: e.offsetY });
    if (pointers.size === 1) drag = { x: e.offsetX, y: e.offsetY, moved: false };
    else drag = { pinch: pinchState(), moved: true };
  });
  function pinchState() {
    const [a, b] = [...pointers.values()];
    return { d: Math.hypot(a.x - b.x, a.y - b.y) || 1, mx: (a.x + b.x) / 2, my: (a.y + b.y) / 2 };
  }
  canvas.addEventListener("pointermove", (e) => {
    if (pointers.has(e.pointerId)) pointers.set(e.pointerId, { x: e.offsetX, y: e.offsetY });
    if (drag && drag.pinch && pointers.size === 2) {
      const now = pinchState();
      panBy(now.mx - drag.pinch.mx, now.my - drag.pinch.my);
      zoomAt(now.d / drag.pinch.d, now.mx, now.my);
      drag.pinch = now;
      return;
    }
    if (drag && pointers.size === 1) {
      const dx = e.offsetX - drag.x, dy = e.offsetY - drag.y;
      if (!drag.moved && Math.hypot(dx, dy) > 4) { drag.moved = true; canvas.classList.add("is-dragging"); tooltip.hidden = true; }
      if (drag.moved) { panBy(dx, dy); drag.x = e.offsetX; drag.y = e.offsetY; }
      return;
    }
    if (e.pointerType === "mouse" && S.d) {
      const hit = hitTest(e.offsetX, e.offsetY);
      if (!sameHit(hit, S.hover)) { S.hover = hit; requestDraw(); }
      showTooltip(hit, e.offsetX, e.offsetY);
    }
  });
  function endPointer(e) {
    const wasClick = drag && !drag.moved && pointers.size === 1;
    pointers.delete(e.pointerId);
    if (wasClick && S.d && e.type === "pointerup") {
      const hit = hitTest(e.offsetX, e.offsetY, e.pointerType !== "mouse");
      if (hit && hit.type === "item") setDetail({ type: "item", i: hit.i });
      else if (hit && hit.type === "cand") setDetail({ type: "cand", cand: hit.cand });
      else if (hit && hit.type === "draft") setDetail({ type: "draft", cand: hit.cand, draft: hit.draft });
      else if (hit && hit.type === "clump") zoomAt(2.5, hit.clump.sx, hit.clump.sy);
    }
    if (pointers.size === 0) { drag = null; canvas.classList.remove("is-dragging"); }
    else if (pointers.size === 1) { const [p] = pointers.values(); drag = { x: p.x, y: p.y, moved: true }; }
  }
  canvas.addEventListener("pointerup", endPointer);
  canvas.addEventListener("pointercancel", endPointer);
  canvas.addEventListener("pointerleave", (e) => { if (e.pointerType === "mouse") { tooltip.hidden = true; if (S.hover) { S.hover = null; requestDraw(); } } });
  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    const unit = e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? S.H : 1;
    zoomAt(Math.exp(-e.deltaY * unit * 0.0015), e.offsetX, e.offsetY);
  }, { passive: false });
  canvas.addEventListener("dblclick", (e) => zoomAt(2, e.offsetX, e.offsetY));

  // N and P step through candidate connections in list order.
  function stepCandidate(delta) {
    const list = S.d.candidates;
    if (!list.length) return;
    const at = S.focusCand ? list.indexOf(S.focusCand) : delta > 0 ? -1 : 0;
    const next = list[(at + delta + list.length) % list.length];
    setDetail({ type: "cand", cand: next });
  }
  canvas.addEventListener("keydown", (e) => {
    if (!S.d || e.metaKey || e.ctrlKey || e.altKey) return;
    const step = 60;
    const keys = {
      ArrowLeft: () => panBy(step, 0), ArrowRight: () => panBy(-step, 0),
      ArrowUp: () => panBy(0, step), ArrowDown: () => panBy(0, -step),
      "+": () => zoomAt(1.4, S.W / 2, S.H / 2), "=": () => zoomAt(1.4, S.W / 2, S.H / 2),
      "-": () => zoomAt(1 / 1.4, S.W / 2, S.H / 2), _: () => zoomAt(1 / 1.4, S.W / 2, S.H / 2),
      0: () => { fitAll(); requestDraw(); },
      n: () => stepCandidate(1), N: () => stepCandidate(1), p: () => stepCandidate(-1), P: () => stepCandidate(-1),
    };
    if (keys[e.key]) { e.preventDefault(); keys[e.key](); }
  });

  // Escape closes an open "?" first, then steps back one level. In the search field it first
  // clears the text, as search fields do; a list box keeps it.
  const typing = (node) => node && node.closest && (node.closest("textarea, select") || (node.matches("input") && !["radio", "checkbox", "button"].includes(node.type) && !(node.id === "find" && !node.value)));
  document.addEventListener("keydown", (e) => {
    if (e.key !== "Escape") return;
    const open = openHelp();
    if (open.length) { e.preventDefault(); for (const btn of open) closeHelp(btn, true); return; }
    if (e.defaultPrevented || !S.d || typing(e.target)) return;
    const inDetails = $("details-card").contains(document.activeElement);
    if (stepBack()) {
      e.preventDefault();
      if (inDetails && !$("details").querySelector("h3")) $("details-heading").focus();
    }
  });

  $("zoom-in").addEventListener("click", () => zoomAt(1.5, S.W / 2, S.H / 2));
  $("zoom-out").addEventListener("click", () => zoomAt(1 / 1.5, S.W / 2, S.H / 2));
  $("zoom-reset").addEventListener("click", () => { fitAll(); requestDraw(); });
  $("details-clear").addEventListener("click", () => { S.selectedNb = null; setDetail(null, { say: false }); announce("Details cleared"); $("details-heading").focus(); });
  for (const radio of document.querySelectorAll('input[name="color-mode"]')) {
    radio.addEventListener("change", () => { S.mode = radio.value; renderLegend(); requestDraw(); });
  }
  $("find").addEventListener("input", () => { clearTimeout(findTimer); findTimer = setTimeout(runFind, 120); });

  function resize() {
    const rect = wrap.getBoundingClientRect();
    const W = Math.max(1, Math.round(rect.width)), H = Math.max(1, Math.round(rect.height));
    S.dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(W * S.dpr);
    canvas.height = Math.round(H * S.dpr);
    const oldFit = S.fitK, ratio = S.k / oldFit;
    S.W = W; S.H = H;
    if (S.d && S.bounds) {
      const b = S.bounds;
      S.fitK = Math.min(W / Math.max(b.maxX - b.minX, 1e-6), H / Math.max(b.maxY - b.minY, 1e-6)) * 0.92;
      if (Number.isFinite(ratio) && ratio > 0) S.k = S.fitK * ratio;
    }
    requestDraw();
  }
  new ResizeObserver(resize).observe(wrap);
  (function watchDensity() {
    window.matchMedia(`(resolution: ${window.devicePixelRatio || 1}dppx)`).addEventListener("change", () => { resize(); watchDensity(); }, { once: true });
  })();

  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { readPalette(); requestDraw(); });
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(requestDraw);

  // ?sample=<id> opens a built-in sample; otherwise the page waits for a file.
  readPalette();
  const wanted = new URLSearchParams(location.search).get("sample");
  const sample = SAMPLES.find((s) => s.id === wanted);
  if (sample) openSample(sample);
})();
