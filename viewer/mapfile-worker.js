/* Reads and validates one map file off the page's main thread. Receives { blob }, replies with
   { doc } or { error }. The page starts a fresh worker for each file and ends it afterwards. */
importScripts("mapfile.js");

self.onmessage = async (event) => {
  let result;
  try {
    const response = await fetch("uncharted-export-v1.schema.json");
    if (!response.ok) throw new Error(`the schema could not be loaded (${response.status})`);
    result = await self.UnchartedMapFile.readMapFile(event.data.blob, await response.json());
  } catch (err) {
    result = { error: `The file could not be checked (${String(err && err.message ? err.message : err).slice(0, 120)}).` };
  }
  self.postMessage(result);
};
