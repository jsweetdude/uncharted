// Browser checks for the viewer: import every example, hostile text, keyboard workflow,
// performance budgets and screenshots.
//
//     node tests/viewer/e2e.mjs
//
// Needs playwright-core and a Chromium (Playwright's, or Google Chrome via
// UNCHARTED_BROWSER_CHANNEL=chrome). Set PLAYWRIGHT_CORE to the package's folder when it is not
// installed here. Optional: UNCHARTED_PERF_MAP=<a large map file> to measure the budgets, and
// SCREENSHOT_DIR=<folder> to save screenshots. The page is served from viewer/ by a small server
// in this script, as a static host would serve it.
import { createServer } from "node:http";
import { readFileSync, readdirSync, existsSync, mkdirSync } from "node:fs";
import { extname, join, dirname, normalize } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { gunzipSync, gzipSync } from "node:zlib";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..", "..");
const SITE = join(ROOT, "viewer");
const EXAMPLES = join(ROOT, "schema", "examples");
const EXPECTED = JSON.parse(readFileSync(join(EXAMPLES, "expected-errors.json"), "utf8"));
// The newest talks export, as a gzip upload; the plain file is the current one when both are present.
const TALKS_JSON = join(ROOT, "data", "exports", "talks.json");
const TALKS_GZ = join(ROOT, "data", "exports", "talks.json.gz");
const TALKS_BYTES = existsSync(TALKS_JSON) ? readFileSync(TALKS_JSON) : existsSync(TALKS_GZ) ? gunzipSync(readFileSync(TALKS_GZ)) : null;
const TALKS = TALKS_BYTES && { name: "talks.json.gz", mimeType: "application/gzip", buffer: gzipSync(TALKS_BYTES) };
const SHOTS = process.env.SCREENSHOT_DIR;
const PERF = process.env.UNCHARTED_PERF_MAP;

// Budgets, measured on the largest map (about 72,000 items). "Throttled" is a 4x slower CPU and a
// 375 x 812 viewport; open = choose a file until the map is drawn and the lists are filled.
const BUDGET = { openDesktopMs: 3000, openThrottledMs: 8000, frameDesktopMs: 50, frameThrottledMs: 120 };

let playwright;
try {
  playwright = process.env.PLAYWRIGHT_CORE ? await import(pathToFileURL(join(process.env.PLAYWRIGHT_CORE, "index.mjs")).href) : await import("playwright-core");
} catch {
  console.log("skip: playwright-core is not installed (npm i -D playwright-core, or set PLAYWRIGHT_CORE)");
  process.exit(0);
}
const { chromium } = playwright.default || playwright;

const TYPES = { ".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".json": "application/json", ".gz": "application/gzip" };
const server = createServer((req, res) => {
  const path = normalize(decodeURIComponent(new URL(req.url, "http://x").pathname)).replace(/^([/\\])+/, "");
  const file = join(SITE, path || "index.html");
  if (!file.startsWith(SITE) || !existsSync(file)) { res.writeHead(404); res.end(); return; }
  res.writeHead(200, { "content-type": TYPES[extname(file)] || "application/octet-stream" });
  res.end(readFileSync(file));
}).listen(0);
const BASE = `http://127.0.0.1:${server.address().port}/`;

const results = [];
const check = (name, ok, detail = "") => { results.push({ name, ok, detail }); console.log(`${ok ? "ok  " : "FAIL"} ${name}${detail ? ` — ${detail}` : ""}`); };
const browser = await chromium.launch(process.env.UNCHARTED_BROWSER_CHANNEL ? { channel: process.env.UNCHARTED_BROWSER_CHANNEL } : {});

async function newPage(viewport = { width: 1440, height: 900 }, colorScheme = "light") {
  const context = await browser.newContext({ viewport, colorScheme });
  const page = await context.newPage();
  page.errors = [];
  page.on("pageerror", (e) => page.errors.push(e.message));
  page.on("console", (m) => { if (m.type() === "error") page.errors.push(m.text()); });
  page.on("dialog", (d) => { page.errors.push(`dialog: ${d.message()}`); d.dismiss(); });
  await page.goto(BASE);
  return page;
}
async function openFile(page, path) {
  await page.setInputFiles("#file-input", path);
  await page.waitForFunction(() => !document.querySelector("#open-file").disabled, null, { timeout: 30000 });
}
const message = (page) => page.$eval("#message", (m) => (m.hidden ? "" : m.innerText));
const title = (page) => page.$eval("#map-title", (h) => h.textContent);

// 1. Start, sample, every example
{
  const page = await newPage();
  check("start state asks for a file", await page.isVisible("#start") && !(await page.isVisible("#layout")));
  await page.click("#start-sample");
  await page.waitForFunction(() => !document.querySelector("#layout").hidden);
  check("the built-in sample opens", (await title(page)) === "Sample talks (invented)");

  for (const name of readdirSync(join(EXAMPLES, "invalid")).filter((f) => f.endsWith(".json"))) {
    await openFile(page, join(EXAMPLES, "invalid", name));
    const msg = await message(page);
    check(`rejects invalid/${name}`, msg.includes(EXPECTED[name]) && (await title(page)) === "Sample talks (invented)", msg.split("\n").slice(-1)[0].slice(0, 90));
  }

  await openFile(page, join(EXAMPLES, "hostile-text.json"));
  const safe = await page.evaluate(() => ({
    title: document.querySelector("#map-title").textContent,
    imgs: document.querySelectorAll("img, iframe, object, embed").length,
    scripts: document.querySelectorAll("script").length,
    handlers: [...document.querySelectorAll("*")].filter((e) => [...e.attributes].some((a) => a.name.startsWith("on"))).length,
    links: [...document.querySelectorAll("a[href]")].map((a) => a.href).filter((h) => !/^https?:/.test(h)).length,
  }));
  check("hostile text renders as text and runs nothing", safe.title === "<b>Hostile</b> sample" && safe.imgs === 0 && safe.scripts === 2 && safe.handlers === 0 && safe.links === 0 && page.errors.length === 0, JSON.stringify(safe));
  await page.click("#cand-list button >> nth=0");
  await page.click("#about-heading");
  const afterPin = await page.evaluate(() => document.querySelectorAll("img, iframe, script").length);
  check("hostile candidate details stay inert", afterPin === 2 && page.errors.length === 0);

  await openFile(page, join(EXAMPLES, "small.json"));
  check("a valid file opens after a hostile one", (await title(page)) === "Sample talks (invented)" && !(await message(page)));
  if (TALKS) {
    const talks = JSON.parse(TALKS_BYTES.toString("utf8"));
    await openFile(page, TALKS);
    const counts = await page.evaluate(() => [document.querySelector("#stat-items").textContent, document.querySelectorAll("#cand-list button").length, document.querySelector("#map-title").textContent]);
    check("a gzip export opens and replaces the previous map", counts[0] === talks.items.count.toLocaleString("en-US") && counts[1] === talks.candidates.length && counts[2] === talks.title, counts.join(" / "));

    // Connections tested both ways say how they passed; one-way passes are marked weaker in the list and the details.
    const cut = talks.manifest.candidates ? talks.manifest.candidates.pairCutoff : undefined;
    const tested = typeof cut === "number" ? talks.candidates.filter((c) => c.kind === "connection" && typeof c.score.pairPParentsOnly === "number") : [];
    if (tested.length) {
      const isWeak = (c) => !(c.score.pairPParentsOnly < cut);
      const rows = await page.evaluate(() => Object.fromEntries([...document.querySelectorAll("#cand-list .row-btn")].map((b) => [b.dataset.code, b.textContent.includes("weaker evidence")])));
      const listOk = talks.candidates.every((c) => rows[c.code] === (tested.includes(c) && isWeak(c)));
      const pick = async (c) => { await page.click(`#cand-list .row-btn[data-code="${c.code}"]`); return page.$eval("#details", (d) => d.textContent); };
      const both = tested.find((c) => !isWeak(c)), weak = tested.find(isWeak);
      const bothText = both ? await pick(both) : "Passed both waysDifferent parents used";
      const weakText = weak ? await pick(weak) : "Passed one way only";
      if (weak && SHOTS) { mkdirSync(SHOTS, { recursive: true }); await page.screenshot({ path: join(SHOTS, "weaker-evidence-1440-light.png") }); }
      check("connections say how their pair test passed, and one-way passes read as weaker", listOk && bothText.includes("Passed both ways") && bothText.includes("Different parents used") && weakText.includes("Passed one way only"),
        `${tested.length} tested, ${tested.filter(isWeak).length} one way only`);
    }
  }

  // A score number this viewer has no label for still shows, under its own name.
  const extra = JSON.parse(readFileSync(join(EXAMPLES, "small.json"), "utf8"));
  extra.candidates[0].score.newMeasure = 0.4321;
  await openFile(page, { name: "extra.json", mimeType: "application/json", buffer: Buffer.from(JSON.stringify(extra)) });
  await page.click(`#cand-list .row-btn[data-code="${extra.candidates[0].code}"]`);
  const extraText = await page.$eval("#details", (d) => d.textContent);
  check("an unknown score number shows under its own name", extraText.includes("newMeasure") && extraText.includes("0.4321"));
  // A file dropped anywhere on the page opens the same way.
  const dropped = await page.evaluate(async () => {
    const blob = await (await fetch("samples/small.json")).blob();
    const dt = new DataTransfer();
    dt.items.add(new File([blob], "dropped.json", { type: "application/json" }));
    for (const type of ["dragenter", "dragover", "drop"]) window.dispatchEvent(new DragEvent(type, { dataTransfer: dt, bubbles: true, cancelable: true }));
    await new Promise((r) => setTimeout(r, 1500));
    return { title: document.querySelector("#map-title").textContent, overlay: document.querySelector("#drop-overlay").hidden };
  });
  check("a dropped file opens", dropped.title === "Sample talks (invented)" && dropped.overlay, JSON.stringify(dropped));
  check("no page errors while importing", page.errors.length === 0, page.errors.join("; "));
  await page.context().close();
}

// 2. Keyboard: from the start page to an item reached through a candidate, and back
{
  const page = await newPage();
  await page.click("#start-sample");
  await page.waitForFunction(() => !document.querySelector("#layout").hidden);
  check("focus lands on the map's name after opening", await page.evaluate(() => document.activeElement.id === "map-title"));
  await page.focus("#cand-list button >> nth=0");
  await page.keyboard.press("Enter");
  await page.keyboard.press("Tab"); // to Clear, then the "?" and into the details
  const firstNearest = page.locator("#details .row-btn:has(.rank)").first();
  await firstNearest.focus();
  await page.keyboard.press("Enter");
  const trail = await page.$eval("#details .trail", (t) => t.innerText.replace(/\s+/g, " "));
  const focusOnHeading = await page.evaluate(() => document.activeElement.tagName === "H3");
  await page.keyboard.press("Escape");
  const back = await page.$eval("#details h3", (h) => h.textContent);
  await page.keyboard.press("Escape");
  const cleared = await page.$eval("#details", (d) => d.innerText);
  check("candidate → must-not-copy item → back by Escape", /Must not copy 1 of/.test(trail) && focusOnHeading && back.startsWith("C1") && cleared.startsWith("Pick"), trail);
  await page.focus(".help-btn[aria-controls=help-map]");
  await page.keyboard.press("Enter");
  const opened = await page.$eval("#help-map", (p) => !p.hidden);
  await page.keyboard.press("Escape");
  const closed = await page.evaluate(() => document.querySelector("#help-map").hidden && document.activeElement.getAttribute("aria-controls") === "help-map");
  check("a ? opens by keyboard and Escape closes it, returning focus", opened && closed);
  await page.click("#cand-list button >> nth=0");
  await page.focus("#find");
  await page.keyboard.press("Escape");
  check("Escape in an empty search field steps back", await page.$eval("#details", (d) => d.innerText.startsWith("Pick")));
  // A tap outside an open "?" closes it without stealing the tap: the row picked is the row pinned.
  await page.click(".help-btn[aria-controls=help-cand]");
  await page.click("#cand-list button >> nth=1");
  const tapped = await page.evaluate(() => ({ pinned: document.querySelector("#details h3").textContent, closed: document.querySelector("#help-cand").hidden }));
  check("a tap outside an open ? closes it and still lands", tapped.pinned.startsWith("P1") && tapped.closed, JSON.stringify(tapped));
  const unnamed = await page.evaluate(() => [...document.querySelectorAll("button, a[href], input, select, canvas[tabindex]")].filter((e) => !e.hidden && e.offsetParent !== null).filter((e) => {
    const name = (e.getAttribute("aria-label") || e.textContent || (e.labels && e.labels[0] && e.labels[0].textContent) || e.getAttribute("placeholder") || "").trim();
    return !name;
  }).map((e) => e.outerHTML.slice(0, 60)));
  check("every visible control has a name", unnamed.length === 0, unnamed.join(" | "));
  await page.context().close();
}

// 2b. Search reaches every match; the address's candidate opens only with the first map
if (TALKS) {
  const page = await newPage();
  await page.goto(`${BASE}#%`);
  await openFile(page, TALKS);
  check("a malformed address fragment does not stop a map opening", !(await page.isHidden("#layout")));
  await page.fill("#find", "a");
  await page.waitForTimeout(400);
  const first = await page.$$eval("#find-results [data-i]", (b) => b.length);
  await page.focus("#find-results .more");
  await page.keyboard.press("Enter");
  const after = await page.evaluate(() => ({ rows: document.querySelectorAll("#find-results [data-i]").length, focus: document.activeElement.dataset.i !== undefined }));
  check("search shows 25 and pages on with focus on the next match", first === 25 && after.rows === 50 && after.focus, `${first} → ${after.rows}`);
  await page.goto(`${BASE}?link=1#C1`); // a full load, as following a shared link would be
  await openFile(page, TALKS);
  const pinned = await page.$eval("#details", (d) => (d.querySelector("h3") || {}).textContent || "");
  await openFile(page, join(EXAMPLES, "small.json"));
  const second = await page.$eval("#details", (d) => d.innerText);
  check("#C1 opens the first map at C1 and does not carry to the next map", pinned.startsWith("C1") && second.startsWith("Pick"), pinned.slice(0, 30));
  await page.context().close();
}

// 3. Performance budgets on a large map
if (PERF && existsSync(PERF)) {
  for (const [label, viewport, rate, openBudget, frameBudget] of [["desktop", { width: 1440, height: 900 }, 1, BUDGET.openDesktopMs, BUDGET.frameDesktopMs], ["throttled mobile", { width: 375, height: 812 }, 4, BUDGET.openThrottledMs, BUDGET.frameThrottledMs]]) {
    const page = await newPage(viewport);
    const cdp = await page.context().newCDPSession(page);
    await cdp.send("Emulation.setCPUThrottlingRate", { rate });
    const t0 = Date.now();
    await page.setInputFiles("#file-input", PERF);
    await page.waitForFunction(() => !document.querySelector("#layout").hidden && document.querySelectorAll("#nb-list li").length > 0, null, { timeout: 60000 });
    await page.evaluate(() => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r))));
    const open = Date.now() - t0;
    const frames = await page.evaluate(async () => {
      const out = [];
      for (const id of ["zoom-in", "zoom-out", "zoom-in", "zoom-reset"]) {
        const t = performance.now();
        document.getElementById(id).click();
        await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
        out.push((performance.now() - t) / 2);
      }
      return out;
    });
    const frame = Math.max(...frames);
    check(`${label}: open the largest map within ${openBudget} ms`, open <= openBudget, `${open} ms`);
    if (rate > 1) {
      // Chrome's CPU throttle reaches the page but not its workers, so read and validate once on
      // the throttled main thread too: that is what a slow phone's worker would take.
      await page.route("**/perf-map", (route) => route.fulfill({ path: PERF, contentType: "application/gzip" }));
      const validate = await page.evaluate(async () => {
        const blob = await (await fetch("perf-map")).blob();
        const schema = await (await fetch("uncharted-export-v1.schema.json")).json();
        const t = performance.now();
        const r = await window.UnchartedMapFile.readMapFile(blob, schema);
        return { ms: Math.round(performance.now() - t), error: r.error || "" };
      });
      check(`${label}: read and validate on a 4x slower CPU within ${openBudget} ms`, !validate.error && validate.ms <= openBudget, `${validate.ms} ms ${validate.error}`);
    }
    check(`${label}: a redraw within ${frameBudget} ms`, frame <= frameBudget, `${Math.round(frame)} ms`);
    await page.context().close();
  }
}

// 4. Screenshots
if (SHOTS) {
  mkdirSync(SHOTS, { recursive: true });
  const map = TALKS || join(EXAMPLES, "small.json");
  for (const [w, h] of [[1440, 900], [375, 812]]) {
    for (const scheme of ["light", "dark"]) {
      const page = await newPage({ width: w, height: h }, scheme);
      await page.screenshot({ path: join(SHOTS, `start-${w}-${scheme}.png`) });
      await openFile(page, map);
      await page.screenshot({ path: join(SHOTS, `map-${w}-${scheme}.png`) });
      await page.click("#cand-list button >> nth=0");
      await page.waitForTimeout(250);
      if (w < 960) await page.evaluate(() => window.scrollTo(0, document.querySelector(".map-stage").getBoundingClientRect().top + scrollY - 8));
      await page.screenshot({ path: join(SHOTS, `candidate-${w}-${scheme}.png`) });
      const scroll = await page.evaluate(() => document.documentElement.scrollWidth - innerWidth);
      check(`${w} ${scheme}: no sideways scroll`, scroll <= 0, `${scroll}px`);
      await page.context().close();
    }
  }
}

await browser.close();
server.close();
const failed = results.filter((r) => !r.ok);
console.log(`\n${results.length - failed.length} passed, ${failed.length} failed`);
process.exit(failed.length ? 1 : 0);
