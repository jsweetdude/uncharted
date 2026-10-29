/* Uncharted flat-map viewer. Reads window.UNCHARTED_INDEX (data/index.js) and one
   window.UNCHARTED_DATA[name] (data/<name>.js), both written by uncharted/export_map.py.
   Draws every item on a canvas; the neighborhood, gap and search lists are real HTML lists
   so the map can be explored without the canvas. Corpus text is untrusted: it only ever
   reaches the page through textContent. */
(() => {
  "use strict";

  const TAU = Math.PI * 2;
  const BIN_EDGES = [50, 75, 90, 95, 98, 99];
  const BIN_LABELS = ["Below 50 · most crowded", "50–75", "75–90", "90–95", "95–98", "98–99", "99 and up · loneliest"];
  const KIND_NAMES = { hole: "Hole", pair: "Pair", blend: "Blend" };
  // loneliness mode: bins from LOUD_FROM up (95th percentile and lonelier) draw last, larger, ringed;
  // the crowded bins recede. Strength, not new color, carries the emphasis.
  const LOUD_FROM = 4;
  const BIN_SCALE = [1, 1, 1, 1, 1.6, 1.9, 2.2];
  const QUIET_ALPHA = { dense: [0.3, 0.42, 0.58, 0.75], sparse: [0.55, 0.68, 0.82, 0.9] };
  const KIND_HINTS = { hole: "empty space ringed by neighborhoods", pair: "between two related neighborhoods", blend: "mix of chosen items or neighborhoods" };
  const EMPTY_DETAILS = "Pick a gap from the Gaps list or the map to see what surrounds it. Select any point for its details.";

  const $ = (id) => document.getElementById(id);
  const canvas = $("map");
  const ctx = canvas.getContext("2d");
  const wrap = $("map-wrap");
  const tooltip = $("tooltip");

  const S = {
    name: null, d: null, n: 0,
    x: null, y: null, cluster: null, lonely: null, bin: null,
    lower: null, nbById: new Map(), nbMembers: new Map(), byBin: [], unclustered: null,
    gapsByItem: new Map(), grid: null,
    bounds: null, W: 0, H: 0, dpr: 1,
    cx: 0, cy: 0, k: 1, fitK: 1,
    mode: "neighborhood", hover: null, detail: null, selectedNb: null, focusGap: null,
    pal: null, raf: 0, sized: false,
  };

  // ---------- small DOM helpers ----------

  function el(tag, attrs, ...children) {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(attrs || {})) {
      if (value === null || value === undefined || value === false) continue;
      if (key === "class") node.className = value;
      else if (key === "text") node.textContent = value;
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

  function hidden(text) { return el("span", { class: "pc-visually-hidden", text }); }
  function announce(text) { $("announce").textContent = ""; requestAnimationFrame(() => { $("announce").textContent = text; }); }

  function showMessage(lines) {
    const box = $("message-text");
    box.replaceChildren(...lines.map((line, i) => el(i === 0 ? "strong" : "p", { text: line })));
    $("message").hidden = false;
  }

  const SHAPES = {
    hole: '<circle cx="8" cy="8" r="6" fill="none" stroke="currentColor" stroke-width="1.5" stroke-dasharray="3 2"/><circle cx="8" cy="8" r="2.2" fill="currentColor"/>',
    pair: '<path d="M8 2 14 8 8 14 2 8Z" fill="currentColor"/>',
    blend: '<path d="M8 2.5 14 13.5H2Z" fill="currentColor"/>',
    nearest: '<circle cx="8" cy="8" r="5.5" fill="none" stroke="currentColor" stroke-width="1.75"/><text x="8" y="11" text-anchor="middle" font-size="8" font-weight="600" fill="currentColor">1</text>',
    ring: '<circle cx="8" cy="8" r="5.5" fill="none" stroke="currentColor" stroke-width="1.5" stroke-dasharray="2 2"/>',
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

  // ---------- data ----------

  function itemUrl(i) {
    const it = S.d.items;
    const p = it.urlPrefix[i];
    return p >= 0 ? S.d.urlPrefixes[p] + it.url[i] : it.url[i];
  }
  function itemVenue(i) { const v = S.d.items.venue[i]; return v >= 0 ? S.d.venues[v] : ""; }
  // non-breaking space before each dot, so a wrapped name never starts a line with "·"
  const joinTerms = (terms) => terms.join("\u00a0· ");
  function nbName(id, terms = 3) {
    const nb = S.nbById.get(id);
    if (!nb) return "Unclustered";
    return `#${id} ${joinTerms(nb.terms.slice(0, terms))}`;
  }
  function englishList(a) { return a.length < 2 ? a[0] || "" : `${a.slice(0, -1).join(", ")} and ${a[a.length - 1]}`; }
  function shortHeadline(g) { return g.headline.replace(/\s*\([^()]*\)/g, ""); }
  // the neighborhoods a gap is named after: the ones its brief headline cites, else the nearest ones
  function gapNeighborhoods(g) {
    const cited = [...g.headline.matchAll(/#(\d+)/g)].map((m) => Number(m[1]));
    return [...new Set(cited.length ? cited : g.neighborhoods)].filter((id) => S.nbById.has(id));
  }
  function gapTitle(g) {
    const ids = gapNeighborhoods(g);
    if (!ids.length) return shortHeadline(g);
    const names = englishList(ids.map((id) => S.nbById.get(id).terms.slice(0, 2).join(" ")));
    const cited = /#\d/.test(g.headline);
    return g.kind === "blend" ? `${cited ? "Blend of" : "Blend near"} ${names}` : `Between ${names}`;
  }
  function gapSubline(g) {
    const ids = gapNeighborhoods(g);
    const n = (g.drafts || []).length;
    return [KIND_NAMES[g.kind], ids.map((id) => `#${id}`).join(", "), n ? `${n} ${n === 1 ? "draft" : "drafts"}` : ""].filter(Boolean).join(" · ");
  }
  const VERDICT_NAMES = { PASS: "Passed", RETRY: "Retry" };
  function verdictBadge(d) {
    return el("span", { class: d.verdict === "PASS" ? "pc-badge pc-badge--success" : "pc-badge", text: VERDICT_NAMES[d.verdict] || d.verdict || "Unchecked" });
  }
  // what a pinned gap lights: its named neighborhoods, its must-not-copy items (in Details order)
  // and, for a hole, the ring items around it
  function gapFocus(g) {
    if (!g._focus) {
      const nearest = g.nearest.map(([i]) => i);
      const ring = g.kind === "hole" ? g.parents : [];
      const drafts = g.drafts || [];
      const draftNearest = drafts.flatMap((d) => d.nearest.map(([i]) => i));
      g._focus = {
        nbs: new Set(gapNeighborhoods(g)), nearest, ring, drafts,
        items: new Set([...nearest, ...ring]),
        related: new Set([...nearest, ...ring, ...draftNearest]),
      };
    }
    return g._focus;
  }
  function similarity(v) {
    const median = S.d.stats.nn1Percentiles && S.d.stats.nn1Percentiles["50"];
    return el("span", {}, el("span", { class: "mono", text: v.toFixed(3) }), median ? el("span", { class: "pc-muted", text: ` · corpus median ${median.toFixed(3)}` }) : null);
  }
  function headRow(left, right) {
    return el("div", { class: "list-head pc-label sub-label" }, el("span", { text: left }), el("span", { text: right }));
  }

  function prepare(d) {
    const it = d.items;
    const n = it.x.length;
    S.d = d; S.n = n;
    S.x = Float32Array.from(it.x);
    S.y = Float32Array.from(it.y);
    S.cluster = Int32Array.from(it.cluster);
    S.lonely = Float32Array.from(it.lonely);
    S.bin = new Uint8Array(n);
    S.lower = null;
    S.byBin = BIN_EDGES.map(() => []).concat([[]]);
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
      S.byBin[b].push(i);
      const c = S.cluster[i];
      if (c < 0) unclustered.push(i);
      else { if (!members.has(c)) members.set(c, []); members.get(c).push(i); }
    }
    S.bounds = { minX, maxX, minY, maxY };
    S.unclustered = Int32Array.from(unclustered);
    S.byBin = S.byBin.map((a) => Int32Array.from(a));
    S.nbById = new Map(d.neighborhoods.map((nb) => [nb.id, nb]));
    S.nbMembers = new Map([...members].map(([c, a]) => [c, Int32Array.from(a)]));
    S.gapsByItem = new Map();
    for (const g of d.gaps) {
      for (const [i] of g.nearest) {
        if (!S.gapsByItem.has(i)) S.gapsByItem.set(i, []);
        S.gapsByItem.get(i).push(g);
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
    const fill = start.slice(0, G * G);
    const items = new Int32Array(S.n);
    for (let i = 0; i < S.n; i++) items[fill[cellOf[i]]++] = i;
    S.grid = { G, sx, sy, start, items };
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
  // fit a data box into the map, beside the legend rather than under it: above it or to its right,
  // whichever leaves the bigger map
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
    const w = Math.max(b.maxX - b.minX, 1e-6), h = Math.max(b.maxY - b.minY, 1e-6);
    S.fitK = Math.min(S.W / w, S.H / h) * 0.92;
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

  function fitToGap(g) {
    const f = gapFocus(g);
    const rad = g.radius || 0;
    let minX = g.x - rad, maxX = g.x + rad, minY = g.y - rad, maxY = g.y + rad;
    const pts = [...f.nearest, ...f.ring].map((i) => [S.x[i], S.y[i]]).concat(f.drafts.map((d) => [d.x, d.y]));
    for (const [x, y] of pts) {
      minX = Math.min(minX, x); maxX = Math.max(maxX, x);
      minY = Math.min(minY, y); maxY = Math.max(maxY, y);
    }
    fitTo(minX, maxX, minY, maxY, 0.8, 6);
    requestDraw();
  }

  function dotRadius() {
    const base = Math.max(1.1, Math.min(3, 2.6 * Math.sqrt(1000 / S.n)));
    return Math.min(base * 2.2, base * Math.pow(S.k / S.fitK, 0.4));
  }

  // ---------- drawing ----------

  // Animation frames pause in background or occluded pages; the timer keeps the canvas current there.
  function requestDraw() {
    if (S.raf) return;
    S.raf = requestAnimationFrame(draw);
    setTimeout(() => { if (S.raf) { cancelAnimationFrame(S.raf); draw(); } }, 120);
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

  function drawLoud(indices, color, r, filter) {
    const W = S.W, H = S.H, halo = r + 1.5;
    for (const [fill, rad] of [[S.pal.surface, halo], [color, r]]) {
      ctx.fillStyle = fill;
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

  function markerPath(kind, sx, sy, s) {
    ctx.beginPath();
    if (kind === "pair") { ctx.moveTo(sx, sy - s); ctx.lineTo(sx + s, sy); ctx.lineTo(sx, sy + s); ctx.lineTo(sx - s, sy); ctx.closePath(); }
    else if (kind === "blend") { ctx.moveTo(sx, sy - s); ctx.lineTo(sx + s * 0.95, sy + s * 0.75); ctx.lineTo(sx - s * 0.95, sy + s * 0.75); ctx.closePath(); }
    else ctx.arc(sx, sy, s * 0.55, 0, TAU);
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
    const nb = S.nbById.get(S.cluster[i]);
    return nb ? S.pal.cat[nb.slot % 8] : S.pal.unclustered;
  }

  function draw() {
    S.raf = 0;
    if (!S.d || !S.W) return;
    const P = S.pal;
    ctx.setTransform(S.dpr, 0, 0, S.dpr, 0, 0);
    ctx.fillStyle = P.surface;
    ctx.fillRect(0, 0, S.W, S.H);

    const r = dotRadius();
    const s = 7;
    const dense = S.n > 5000;
    const dim = 0.12;
    const focusGap = S.focusGap;
    const focus = focusGap ? gapFocus(focusGap) : null;
    // lit neighborhoods: the pinned gap's, or the one isolated from the list; everything else dims
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
        drawPoints(S.nbMembers.get(nb.id) || [], P.cat[nb.slot % 8], lit ? dim : a, r);
      }
      if (lit) {
        for (const id of lit) {
          const nb = S.nbById.get(id);
          if (nb) drawPoints(S.nbMembers.get(id) || [], P.cat[nb.slot % 8], 1, r * 1.15);
        }
      }
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
    // a pinned gap's own items at full strength, wherever they sit
    if (focus) {
      for (const i of focus.items) {
        if (S.mode === "loneliness" && S.bin[i] >= LOUD_FROM) drawLoud([i], colorOf(i), r * 1.15 * BIN_SCALE[S.bin[i]], null);
        else drawPoints([i], colorOf(i), 1, r * 1.15);
      }
    }

    // Holes: the empty radius gaps.py measured, drawn to scale; others fade while a gap is pinned
    ctx.setLineDash([4, 3]);
    for (const g of S.d.gaps) {
      if (g.kind !== "hole") continue;
      const fade = focus && g !== focusGap ? 0.3 : 1;
      const sx = toSX(g.x), sy = toSY(g.y), rr = Math.max(g.radius * S.k, 7);
      ctx.beginPath(); ctx.arc(sx, sy, rr, 0, TAU);
      ctx.lineWidth = 4; ctx.strokeStyle = P.surface; ctx.globalAlpha = 0.8 * fade; ctx.stroke();
      ctx.globalAlpha = fade; ctx.lineWidth = 1.5; ctx.strokeStyle = P.text; ctx.stroke();
      ctx.globalAlpha = 1;
    }
    ctx.setLineDash([]);

    // Gap markers on top, with a surface ring so they read over any color
    for (const g of S.d.gaps) {
      const sx = toSX(g.x), sy = toSY(g.y);
      if (sx < -20 || sy < -20 || sx > S.W + 20 || sy > S.H + 20) continue;
      ctx.globalAlpha = focus && g !== focusGap ? 0.35 : 1;
      markerPath(g.kind, sx, sy, s);
      ctx.lineWidth = 4; ctx.strokeStyle = P.surface; ctx.stroke();
      ctx.fillStyle = P.text; ctx.fill();
      if (g.kind === "hole") { ctx.lineWidth = 1.5; ctx.strokeStyle = P.text; ctx.stroke(); }
      ctx.globalAlpha = 1;
    }

    // A pinned gap's marks: dotted outlines on a hole's ring items, cobalt rings on must-not-copy items
    if (focus) {
      ctx.setLineDash([2, 2]);
      for (const i of focus.ring) {
        ctx.beginPath(); ctx.arc(toSX(S.x[i]), toSY(S.y[i]), sizeOf(i) + 3.5, 0, TAU);
        ctx.lineWidth = 1.5; ctx.strokeStyle = P.text; ctx.stroke();
      }
      ctx.setLineDash([]);
      for (const i of focus.nearest) ring(toSX(S.x[i]), toSY(S.y[i]), sizeOf(i) + 4, P.cobalt, 1.75);
      // drafts written for this gap: filled square passed, hollow square needs another try
      for (const d of focus.drafts) {
        const sx = toSX(d.x), sy = toSY(d.y), h = 5;
        ctx.beginPath(); ctx.rect(sx - h, sy - h, 2 * h, 2 * h);
        ctx.lineWidth = 4; ctx.strokeStyle = P.surface; ctx.stroke();
        if (d.verdict === "PASS") { ctx.fillStyle = P.text; ctx.fill(); }
        else { ctx.fillStyle = P.surface; ctx.fill(); ctx.lineWidth = 1.75; ctx.strokeStyle = P.text; ctx.stroke(); }
      }
    }

    // Labels, most important first so they win the space: the pinned gap's code, the lit neighborhoods,
    // the pinned gap's item numbers, other gap codes, then (only when nothing is focused) every neighborhood
    const placed = [];
    ctx.textBaseline = "middle";
    for (const g of S.d.gaps) placed.push({ x: toSX(g.x) - s - 2, y: toSY(g.y) - s - 2, w: 2 * s + 4, h: 2 * s + 4 });
    ctx.font = '500 11px "Atkinson Hyperlegible Mono", ui-monospace, monospace';
    if (focusGap) haloText(focusGap.code, toSX(focusGap.x) + s + 12, toSY(focusGap.y), P.text, placed, "left");
    const nbLabel = (nb) => haloText(nb.terms.slice(0, 2).join(" · "), toSX(nb.x), toSY(nb.y), P.ink, placed);
    const bySize = [...S.d.neighborhoods].sort((a, b) => b.size - a.size);
    ctx.font = '600 12px "Atkinson Hyperlegible Next", Verdana, sans-serif';
    if (lit) for (const nb of bySize) if (lit.has(nb.id)) nbLabel(nb);
    ctx.font = '500 11px "Atkinson Hyperlegible Mono", ui-monospace, monospace';
    if (focus) for (const d of focus.drafts) placed.push({ x: toSX(d.x) - 7, y: toSY(d.y) - 7, w: 14, h: 14 });
    if (focus) for (const d of focus.drafts) haloText(d.code, toSX(d.x) + 9, toSY(d.y), P.text, placed, "left");
    if (focus) {
      ctx.font = '600 11px "Atkinson Hyperlegible Mono", ui-monospace, monospace';
      focus.nearest.forEach((i, k) => {
        const sx = toSX(S.x[i]), sy = toSY(S.y[i]), off = r + 6;
        if (!haloText(String(k + 1), sx + off, sy - off, P.cobalt, placed, "left")) haloText(String(k + 1), sx - off - 8, sy - off, P.cobalt, placed, "left");
      });
      ctx.font = '500 11px "Atkinson Hyperlegible Mono", ui-monospace, monospace';
    }
    for (const g of S.d.gaps) if (g !== focusGap) haloText(g.code, toSX(g.x) + s + 6, toSY(g.y), focus ? P.muted : P.text, placed, "left");
    ctx.font = '600 12px "Atkinson Hyperlegible Next", Verdana, sans-serif';
    if (!lit) for (const nb of bySize) nbLabel(nb);

    // Selection and hover rings (selection is cobalt, hover is text color)
    if (S.detail && S.detail.type === "item") ring(toSX(S.x[S.detail.i]), toSY(S.y[S.detail.i]), sizeOf(S.detail.i) + 7, P.cobalt, 2.5);
    if (focusGap) ring(toSX(focusGap.x), toSY(focusGap.y), s + 6, P.cobalt, 2.5);
    if (S.hover && S.hover.type === "item") ring(toSX(S.x[S.hover.i]), toSY(S.y[S.hover.i]), sizeOf(S.hover.i) + 3.5, P.text, 1.5);
    if (S.hover && S.hover.type === "gap") ring(toSX(S.hover.gap.x), toSY(S.hover.gap.y), s + 4, P.text, 1.5);
    if (S.detail && S.detail.type === "draft") ring(toSX(S.detail.draft.x), toSY(S.detail.draft.y), 10, P.cobalt, 2.5);
    if (S.hover && S.hover.type === "draft") ring(toSX(S.hover.draft.x), toSY(S.hover.draft.y), 9, P.text, 1.5);
  }

  // ---------- hit testing ----------

  function hitTest(px, py, touch = false) {
    if (S.focusGap) {
      for (const d of gapFocus(S.focusGap).drafts) {
        if (Math.hypot(toSX(d.x) - px, toSY(d.y) - py) <= (touch ? 16 : 10)) return { type: "draft", gap: S.focusGap, draft: d };
      }
    }
    for (let j = S.d.gaps.length - 1; j >= 0; j--) {
      const g = S.d.gaps[j];
      if (Math.hypot(toSX(g.x) - px, toSY(g.y) - py) <= (touch ? 18 : 12)) return { type: "gap", gap: g };
    }
    const r = dotRadius();
    const reach = touch ? Math.max(18, r + 10) : Math.max(10, r + 6);
    const { G, sx, sy, start, items } = S.grid;
    const { minX, minY } = S.bounds;
    const dx = toDX(px), dy = toDY(py), rd = reach / S.k;
    // dimmed points are not targets: only the pinned gap's neighborhoods and items, or the isolated neighborhood
    const f = S.focusGap ? gapFocus(S.focusGap) : null;
    const sel = S.selectedNb;
    const allowed = f ? (i) => f.nbs.has(S.cluster[i]) || f.related.has(i) : sel !== null ? (i) => S.cluster[i] === sel : null;
    const c0 = Math.max(0, Math.floor((dx - rd - minX) * sx)), c1 = Math.min(G - 1, Math.floor((dx + rd - minX) * sx));
    const r0 = Math.max(0, Math.floor((dy - rd - minY) * sy)), r1 = Math.min(G - 1, Math.floor((dy + rd - minY) * sy));
    let best = -1, bestD = reach;
    for (let row = r0; row <= r1; row++) {
      for (let col = c0; col <= c1; col++) {
        const cell = row * G + col;
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

  function showTooltip(hit, px, py) {
    if (!hit) { tooltip.hidden = true; return; }
    if (hit.type === "item") {
      const i = hit.i;
      const meta = [itemVenue(i), S.d.items.year[i]].filter(Boolean).join(", ");
      tooltip.replaceChildren(el("strong", { text: S.d.items.title[i] }), el("span", { text: meta || "No venue" }), el("span", { text: nbName(S.cluster[i]) }));
    } else if (hit.type === "draft") {
      const d = hit.draft;
      tooltip.replaceChildren(el("strong", { text: `${hit.gap.code} ${d.code} · ${d.title}` }), el("span", { text: `${VERDICT_NAMES[d.verdict] || d.verdict} · nearest real item ${d.nn1.toFixed(3)} · loneliness percentile ${d.lonely.toFixed(1)}` }));
    } else {
      const g = hit.gap;
      tooltip.replaceChildren(el("strong", { text: `${g.code} · ${gapTitle(g)}` }), el("span", { text: `${KIND_NAMES[g.kind]} · loneliness percentile ${g.lonely.toFixed(1)}` }));
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
    return el("button", { type: "button", class: "row-btn two-col", onclick }, el("span", { class: "main" }, el("span", { text: label })), extra || null);
  }

  function facts(rows) {
    return el("dl", { class: "facts" }, rows.filter(Boolean).flatMap(([k, v]) => [el("dt", { text: k }), el("dd", {}, v)]));
  }

  const narrow = window.matchMedia("(max-width: 959px)");
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

  // from: "map" (canvas), "list" (a list outside the panel) or "details" (a button inside the
  // panel, which the re-render removes, so focus moves to the new heading instead of the body).
  function setDetail(detail, { center = false, say = true, from = "map" } = {}) {
    S.detail = detail;
    // a pinned gap stays in focus while its own items are opened; anything else releases it
    if (!detail || detail.type === "nb") S.focusGap = null;
    else if (detail.type === "gap" || detail.type === "draft") { S.focusGap = detail.gap; S.selectedNb = null; }
    else if (detail.type === "item" && S.focusGap && !gapFocus(S.focusGap).related.has(detail.i)) S.focusGap = null;
    const box = $("details");
    $("details-clear").hidden = !detail;
    if (!detail) {
      box.replaceChildren(el("p", { class: "pc-muted", text: EMPTY_DETAILS }));
    } else if (detail.type === "item") {
      renderItem(box, detail.i);
      if (center) {
        const sx = toSX(S.x[detail.i]), sy = toSY(S.y[detail.i]);
        const inView = sx > 0 && sy > 0 && sx < S.W && sy < S.H;
        if (!(S.focusGap && inView)) centerOn(S.x[detail.i], S.y[detail.i], S.focusGap ? 0 : 4);
      }
      if (say) announce(`Pinned: ${S.d.items.title[detail.i]}`);
    } else if (detail.type === "gap") {
      renderGap(box, detail.gap);
      fitToGap(detail.gap);
      const nd = (detail.gap.drafts || []).length;
      if (say) announce(`Showing gap ${detail.gap.code}, ${gapTitle(detail.gap)}, with its ${detail.gap.nearest.length} must-not-copy items numbered on the map${nd ? ` and ${nd} ${nd === 1 ? "draft" : "drafts"}` : ""}`);
    } else if (detail.type === "draft") {
      renderDraft(box, detail.gap, detail.draft);
      const sx = toSX(detail.draft.x), sy = toSY(detail.draft.y);
      if (!(sx > 0 && sy > 0 && sx < S.W && sy < S.H)) centerOn(detail.draft.x, detail.draft.y, 0);
      if (say) announce(`Showing draft ${detail.draft.code} for gap ${detail.gap.code}: ${VERDICT_NAMES[detail.draft.verdict] || detail.draft.verdict}`);
    } else if (detail.type === "nb") {
      renderNeighborhood(box, detail.id);
      if (say) announce(`Showing ${nbName(detail.id)}`);
    }
    syncPressed();
    renderLegend();
    requestDraw();
    const heading = box.querySelector("h3");
    if (from === "details" && heading) heading.focus();
    else if (from === "list" && detail && narrow.matches) $("details-card").scrollIntoView({ block: "start", behavior: reducedMotion.matches ? "auto" : "smooth" });
    else if (from === "list" && detail) $("details-card").scrollTop = 0;
  }

  function renderItem(box, i) {
    const it = S.d.items;
    const url = itemUrl(i);
    const c = S.cluster[i];
    const nbCell = c >= 0 ? el("button", { type: "button", class: "link-btn", onclick: () => selectNeighborhood(c, "details") }, nbName(c)) : "Unclustered";
    const gaps = S.gapsByItem.get(i) || [];
    fill(box,
      el("h3", { tabindex: "-1", text: it.title[i] }),
      facts([
        ["Year", it.year[i] ?? "–"],
        ["Venue", itemVenue(i) || "–"],
        ["Neighborhood", nbCell],
        ["Loneliness percentile", el("span", { class: "mono", text: it.lonely[i].toFixed(1) })],
        ["Nearest-neighbor similarity", similarity(it.nn1[i])],
        gaps.length ? ["Must-not-copy for", el("span", {}, gaps.map((g, k) => [k ? ", " : "", el("button", { type: "button", class: "link-btn mono", onclick: () => setDetail({ type: "gap", gap: g }, { center: true, from: "details" }) }, g.code)]))] : null,
      ]),
      el("div", { class: "pc-btn-row" },
        url ? el("a", { class: "pc-btn", href: url, target: "_blank", rel: "noopener noreferrer" }, "Open source page") : null,
        el("button", { type: "button", class: "pc-btn", onclick: () => centerOn(S.x[i], S.y[i], 4) }, "Center on item"),
      ),
    );
  }

  function itemRows(pairs, numbered = false) {
    const open = (i) => () => setDetail({ type: "item", i }, { center: true, from: "details" });
    return el("ul", { class: "row-list inline-list" }, pairs.map(([i, sim], k) => el("li", {},
      numbered
        ? el("button", { type: "button", class: "row-btn", onclick: open(i) },
            el("span", { class: "num rank", text: String(k + 1) }),
            el("span", { class: "main" }, el("span", { text: S.d.items.title[i] })),
            el("span", { class: "num", text: sim.toFixed(3) }))
        : linkButton(S.d.items.title[i], open(i), sim === undefined ? null : el("span", { class: "num", text: sim.toFixed(3) })),
    )));
  }

  function renderGap(box, g) {
    const nbs = gapNeighborhoods(g);
    fill(box,
      el("h3", { tabindex: "-1" }, el("span", { class: "mono", text: g.code }), " ", gapTitle(g)),
      nbs.length ? el("p", { class: "pc-label sub-label", text: "Neighborhoods" }) : null,
      nbs.length ? el("ul", { class: "row-list inline-list" }, nbs.map((id) => el("li", {}, linkButton(nbName(id), () => selectNeighborhood(id, "details"))))) : null,
      g.drafts && g.drafts.length ? headRow("Drafts", "Nearest similarity") : null,
      g.drafts && g.drafts.length ? el("ul", { class: "row-list inline-list" }, g.drafts.map((d) => el("li", {},
        el("button", { type: "button", class: "row-btn", onclick: () => setDetail({ type: "draft", gap: g, draft: d }, { from: "details" }) },
          el("span", { class: "mono draft-code", text: d.code }),
          el("span", { class: "main" }, el("span", { text: d.title }), el("span", { class: "sub" }, verdictBadge(d), ` loneliness percentile ${d.lonely.toFixed(1)}`)),
          el("span", { class: "num", text: d.nn1.toFixed(3) }))))) : null,
      headRow("Must not copy", "Similarity"),
      itemRows(g.nearest, true),
      g.parents.length ? el("p", { class: "pc-label sub-label", text: g.kind === "hole" ? "Ring items" : "Ingredients" }) : null,
      g.parents.length ? itemRows(g.parents.map((i) => [i])) : null,
      el("p", { class: "pc-label sub-label", text: "Measurements" }),
      facts([
        ["Kind", KIND_NAMES[g.kind]],
        ["Loneliness percentile", el("span", { class: "mono", text: g.lonely.toFixed(1) })],
        ["Nearest-neighbor similarity", similarity(g.nn1)],
        ["Brief", el("code", { text: g.brief })],
      ]),
      g.tags.length ? el("p", { class: "pc-label sub-label", text: "Leans toward" }) : null,
      g.tags.length ? el("p", { class: "tag-list", text: g.tags.join(", ") }) : null,
      el("div", { class: "pc-btn-row" }, el("button", { type: "button", class: "pc-btn", onclick: () => fitToGap(g) }, "Fit to gap")),
    );
  }

  function renderDraft(box, g, d) {
    const back = () => setDetail({ type: "gap", gap: g }, { from: "details" });
    fill(box,
      el("h3", { tabindex: "-1" }, el("span", { class: "mono", text: `${g.code} ${d.code}` }), " ", d.title),
      facts([
        ["Verdict", verdictBadge(d)],
        ["Nearest-neighbor similarity", similarity(d.nn1)],
        ["Loneliness percentile", el("span", { class: "mono", text: d.lonely.toFixed(1) })],
        d.toGap !== null && d.toGap !== undefined ? ["Similarity to the gap", el("span", { class: "mono", text: d.toGap.toFixed(3) })] : null,
        ["Gap", el("button", { type: "button", class: "link-btn", onclick: back }, `${g.code} ${gapTitle(g)}`)],
        ["Draft", el("code", { text: d.file })],
      ]),
      d.verdict !== "PASS" && d.reasons.length ? el("p", { class: "pc-label sub-label", text: "Why it needs another try" }) : null,
      d.verdict !== "PASS" ? d.reasons.map((t) => el("p", { class: "tag-list", text: t })) : null,
      headRow("Nearest real items", "Similarity"),
      itemRows(d.nearest),
      el("div", { class: "pc-btn-row" },
        el("button", { type: "button", class: "pc-btn", onclick: back }, `Back to ${g.code}`),
        el("button", { type: "button", class: "pc-btn", onclick: () => centerOn(d.x, d.y, 0) }, "Center on draft"),
      ),
    );
  }

  function renderNeighborhood(box, id) {
    const nb = S.nbById.get(id);
    fill(box,
      el("h3", { tabindex: "-1", text: nbName(id, 6) }),
      facts([
        ["Items", el("span", { class: "mono", text: String(nb.size) })],
        ["Median nearest-neighbor similarity", el("span", { class: "mono", text: nb.density.toFixed(3) })],
        nb.popularity ? ["Popularity", el("span", { class: "mono", text: String(nb.popularity) })] : null,
      ]),
      nb.tags.length ? el("p", { class: "pc-label sub-label", text: "Tags" }) : null,
      nb.tags.length ? el("p", { class: "tag-list", text: nb.tags.join(", ") }) : null,
      nb.examples.length ? el("p", { class: "pc-label sub-label", text: "Central items" }) : null,
      nb.examples.length ? itemRows(nb.examples.map((i) => [i])) : null,
      el("div", { class: "pc-btn-row" }, el("button", { type: "button", class: "pc-btn", onclick: () => zoomToNeighborhood(id) }, "Fit to neighborhood")),
    );
  }

  function selectNeighborhood(id, from = "list") {
    if (S.selectedNb === id && S.detail && S.detail.type === "nb" && from === "list") {
      S.selectedNb = null;
      setDetail(null, { say: false });
      announce("Showing all neighborhoods");
      return;
    }
    S.selectedNb = id;
    setDetail({ type: "nb", id }, { from });
    zoomToNeighborhood(id);
  }

  function zoomToNeighborhood(id) {
    const m = S.nbMembers.get(id);
    if (!m || !m.length) return;
    const xs = Array.from(m, (i) => S.x[i]).sort((a, b) => a - b);
    const ys = Array.from(m, (i) => S.y[i]).sort((a, b) => a - b);
    const q = (a, p) => a[Math.min(a.length - 1, Math.max(0, Math.round(p * (a.length - 1))))];
    fitTo(q(xs, 0.03), q(xs, 0.97), q(ys, 0.03), q(ys, 0.97), 0.75, 6);
    requestDraw();
  }

  // ---------- lists and legend ----------

  function syncPressed() {
    for (const b of document.querySelectorAll("#nb-list .row-btn")) b.setAttribute("aria-pressed", String(Number(b.dataset.id) === S.selectedNb));
    const gap = S.focusGap ? S.focusGap.code : null;
    for (const b of document.querySelectorAll("#gap-list .row-btn")) b.setAttribute("aria-pressed", String(b.dataset.code === gap));
  }

  function renderLists() {
    const gapList = $("gap-list");
    gapList.replaceChildren(...S.d.gaps.map((g) => el("li", {},
      el("button", { type: "button", class: "row-btn", "data-code": g.code, "aria-pressed": "false", onclick: () => setDetail({ type: "gap", gap: g }, { center: true, from: "list" }) },
        el("span", { style: "display:flex;align-items:center;gap:6px;" }, markerIcon(g.kind), el("span", { class: "mono", text: g.code })),
        el("span", { class: "main" }, el("span", { text: gapTitle(g) }), el("span", { class: "sub", text: gapSubline(g) })),
        el("span", { class: "num" }, hidden("loneliness percentile "), g.lonely.toFixed(1)),
      ))));
    if (!S.d.gaps.length) gapList.replaceChildren(el("li", { style: "padding: 8px 14px;" }, el("span", { class: "pc-muted", text: "Run gaps.py, then export again to see gaps here." })));

    const nbs = [...S.d.neighborhoods].sort((a, b) => b.size - a.size);
    $("nb-list").replaceChildren(...nbs.map((nb) => el("li", {},
      el("button", { type: "button", class: "row-btn", "data-id": nb.id, "aria-pressed": "false", onclick: () => selectNeighborhood(nb.id, "list") },
        el("span", { class: "swatch", style: `background: var(--map-c${(nb.slot % 8) + 1})`, "aria-hidden": "true" }),
        el("span", { class: "main" }, el("span", { text: nbName(nb.id) }), el("span", { class: "sub", text: nb.tags.slice(0, 3).join(" · ") || nb.terms.slice(3, 6).join(" · ") })),
        el("span", { class: "num" }, String(nb.size), hidden(" items")),
      ))));
  }

  function renderLegend() {
    const rows = [];
    for (const kind of ["hole", "pair", "blend"]) {
      const count = S.d.gaps.filter((g) => g.kind === kind).length;
      if (count) rows.push(el("li", {}, markerIcon(kind), el("span", {}, el("strong", { text: KIND_NAMES[kind] }), el("span", { class: "pc-muted", text: ` · ${KIND_HINTS[kind]}` })), el("span", { class: "num", text: String(count) })));
    }
    if (S.focusGap) {
      rows.push(el("li", {}, markerIcon("nearest"), el("span", { text: `Must not copy for ${S.focusGap.code} · numbers match Details` })));
      if (S.focusGap.kind === "hole") rows.push(el("li", {}, markerIcon("ring"), el("span", { text: `Ring item around ${S.focusGap.code}` })));
      const drafts = S.focusGap.drafts || [];
      if (drafts.some((d) => d.verdict === "PASS")) rows.push(el("li", {}, markerIcon("draftPass"), el("span", { text: `Draft for ${S.focusGap.code} · passed` })));
      if (drafts.some((d) => d.verdict !== "PASS")) rows.push(el("li", {}, markerIcon("draftRetry"), el("span", { text: `Draft for ${S.focusGap.code} · needs another try` })));
    }
    const box = $("legend");
    if (S.mode === "neighborhood") {
      box.replaceChildren(
        el("ul", { class: "legend-rows" }, rows,
          el("li", {}, el("span", { class: "swatch", style: "background: var(--pc-control-border); opacity: .6", "aria-hidden": "true" }), el("span", { text: "Unclustered item" }), el("span", { class: "num", text: String(S.d.stats.unclustered) }))),
        el("p", { class: "pc-hint", style: "margin-top: 8px;", text: "Colors repeat across the map. Select a neighborhood in the list to isolate it." }),
      );
    } else {
      const counts = S.byBin.map((a) => a.length);
      box.replaceChildren(
        el("ul", { class: "legend-rows" }, rows),
        el("p", { class: "pc-label sub-label", text: "Loneliness percentile" }),
        el("ul", { class: "legend-rows" }, BIN_LABELS.map((label, b) => el("li", {},
          el("span", { class: "legend-dot", "aria-hidden": "true" }, el("span", {
            style: `background: var(--map-seq-${b + 1}); width: ${Math.round(7 * BIN_SCALE[b])}px; height: ${Math.round(7 * BIN_SCALE[b])}px;` +
              (b < LOUD_FROM ? ` opacity: ${QUIET_ALPHA[S.n > 5000 ? "dense" : "sparse"][b]};` : ""),
          })),
          el("span", { text: label }),
          el("span", { class: "num" }, String(counts[b]), hidden(" items")),
        )).reverse()),
      );
    }
  }

  // ---------- search ----------

  let findTimer = 0;
  function runFind() {
    const raw = $("find").value.trim();
    const q = raw.toLowerCase();
    const list = $("find-results");
    if (q.length < 2) { list.hidden = true; list.replaceChildren(); $("find-status").textContent = ""; return; }
    if (!S.lower) S.lower = S.d.items.title.map((t) => t.toLowerCase());
    const hits = [];
    let total = 0;
    for (let i = 0; i < S.n; i++) {
      if (S.lower[i].includes(q)) { total++; if (hits.length < 25) hits.push(i); }
    }
    list.replaceChildren(...hits.map((i) => el("li", {}, linkButton(S.d.items.title[i], () => setDetail({ type: "item", i }, { center: true, from: "list" }), el("span", { class: "num", text: String(S.d.items.year[i] ?? "") })))));
    list.hidden = !hits.length;
    $("find-status").textContent = total === 0 ? `No titles contain “${raw}”. Try a single word.` : total > hits.length ? `Showing ${hits.length} of ${total} matches.` : `${total} ${total === 1 ? "match" : "matches"}.`;
  }

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
      const same = (hit && S.hover && hit.type === S.hover.type && hit.i === S.hover.i && hit.gap === S.hover.gap && hit.draft === S.hover.draft) || (!hit && !S.hover);
      if (!same) { S.hover = hit; requestDraw(); }
      showTooltip(hit, e.offsetX, e.offsetY);
    }
  });
  function endPointer(e) {
    const wasClick = drag && !drag.moved && pointers.size === 1;
    pointers.delete(e.pointerId);
    if (wasClick && S.d && e.type === "pointerup") {
      const hit = hitTest(e.offsetX, e.offsetY, e.pointerType !== "mouse");
      if (hit && hit.type === "item") setDetail({ type: "item", i: hit.i });
      else if (hit && hit.type === "gap") setDetail({ type: "gap", gap: hit.gap });
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
  canvas.addEventListener("keydown", (e) => {
    const step = 60;
    const keys = {
      ArrowLeft: () => panBy(step, 0), ArrowRight: () => panBy(-step, 0),
      ArrowUp: () => panBy(0, step), ArrowDown: () => panBy(0, -step),
      "+": () => zoomAt(1.4, S.W / 2, S.H / 2), "=": () => zoomAt(1.4, S.W / 2, S.H / 2),
      "-": () => zoomAt(1 / 1.4, S.W / 2, S.H / 2), _: () => zoomAt(1 / 1.4, S.W / 2, S.H / 2),
      0: () => { fitAll(); requestDraw(); },
      Escape: () => { S.selectedNb = null; setDetail(null, { say: false }); },
    };
    if (keys[e.key]) { e.preventDefault(); keys[e.key](); }
  });

  document.addEventListener("keydown", (e) => {
    if (e.key !== "Escape" || e.defaultPrevented || !S.d) return;
    if (e.target.closest && e.target.closest("input, select, textarea")) return;
    if (!S.detail && S.selectedNb === null) return;
    const inDetails = $("details-card").contains(document.activeElement);
    S.selectedNb = null;
    setDetail(null, { say: false });
    announce("Details cleared");
    if (inDetails) $("details-heading").focus();
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
    const first = !S.sized || !S.W;
    const oldFit = S.fitK;
    S.W = W; S.H = H;
    if (S.d) {
      if (first) { fitAll(); S.sized = true; }
      else { const ratio = S.k / oldFit; const b = S.bounds; S.fitK = Math.min(W / Math.max(b.maxX - b.minX, 1e-6), H / Math.max(b.maxY - b.minY, 1e-6)) * 0.92; S.k = S.fitK * ratio; }
    }
    requestDraw();
  }
  new ResizeObserver(resize).observe(wrap);

  const darkQuery = window.matchMedia("(prefers-color-scheme: dark)");
  darkQuery.addEventListener("change", () => { readPalette(); requestDraw(); });
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(requestDraw);

  // ---------- corpus loading ----------

  function loadScript(src) {
    return new Promise((resolve, reject) => {
      const s = document.createElement("script");
      s.src = src;
      s.onload = resolve;
      s.onerror = () => reject(new Error(src));
      document.head.append(s);
    });
  }

  async function loadCorpus(name) {
    window.UNCHARTED_DATA = window.UNCHARTED_DATA || {};
    if (!window.UNCHARTED_DATA[name]) {
      const status = $("map-status");
      status.textContent = `Loading ${name}…`;
      status.hidden = false;
      $("corpus").disabled = true;
      try { await loadScript(`data/${encodeURIComponent(name)}.js`); } catch (err) { /* reported below */ }
      status.hidden = true;
      $("corpus").disabled = false;
    }
    const d = window.UNCHARTED_DATA[name];
    if (!d) {
      showMessage([`No map data for “${name}”.`, `From the repository root, run: python -I uncharted/export_map.py --name ${name}`, "Then reload this page."]);
      return;
    }
    $("message").hidden = true;
    S.name = name;
    S.detail = null; S.hover = null; S.selectedNb = null; S.focusGap = null;
    prepare(d);
    $("stats").hidden = false;
    $("stat-items").textContent = d.stats.items.toLocaleString();
    $("stat-nb").textContent = String(d.stats.neighborhoods);
    $("stat-unclustered").textContent = d.stats.unclustered.toLocaleString();
    $("stat-gaps").textContent = String(d.gaps.length);
    document.title = `${name} · Uncharted map`;
    canvas.setAttribute("aria-label", `Map of ${d.stats.items.toLocaleString()} ${name} items in ${d.stats.neighborhoods} neighborhoods, with ${d.gaps.length} gaps marked. The gap, neighborhood and search lists reach the same items by keyboard.`);
    $("find").value = "";
    runFind();
    renderLists();
    renderLegend();
    setDetail(null, { say: false });
    if (S.W) { fitAll(); S.sized = true; }
    requestDraw();
  }

  function init() {
    readPalette();
    if (narrow.matches) $("legend-box").open = false;
    const index = Array.isArray(window.UNCHARTED_INDEX) ? window.UNCHARTED_INDEX : [];
    if (!index.length) {
      showMessage(["No exported corpora yet.", "From the repository root, run: python -I uncharted/export_map.py --name <corpus>", "Then reload this page."]);
      return;
    }
    const params = new URLSearchParams(location.search);
    const wanted = params.get("corpus");
    const name = index.some((e) => e.name === wanted) ? wanted : [...index].sort((a, b) => a.bytes - b.bytes)[0].name;
    if (index.length > 1) {
      const select = $("corpus");
      select.replaceChildren(...index.map((e) => el("option", { value: e.name, text: e.name })));
      select.value = name;
      select.addEventListener("change", () => {
        const url = new URL(location.href);
        url.searchParams.set("corpus", select.value);
        history.replaceState(null, "", url);
        loadCorpus(select.value);
      });
      $("corpus-switch").hidden = false;
    }
    loadCorpus(name).then(() => {
      if (wanted && wanted !== name) showMessage([`No map data for “${wanted}”.`, `Showing ${name} instead. To add it, run python -I uncharted/export_map.py --name ${wanted} from the repository root, then reload.`]);
    });
  }

  init();
})();
