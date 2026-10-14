// Service worker: runs the value search (search.js) for the page, off the game's main thread.
// - bot_game: compete mode, the bot's game and the analysis (assistant.js), then the optional analytics event.
// - suggest: suggest mode, the plan for the current position.
// Requests come from content.js through bridge.js. No server is needed.
importScripts("policy.js", "search.js", "assistant.js");

const ready = (async () => {
  const fetchBytes = async (file) => (await fetch(chrome.runtime.getURL(file))).arrayBuffer();
  PPSearch.init(await fetchBytes("mlp.wasm"));
  const meta = await (await fetch(chrome.runtime.getURL("nets/nets.json"))).json();
  const load = async (m) => PPSearch.loadNet(await fetchBytes("nets/" + m.file), m);
  return PPSearch.makeNets(await load(meta.value), await load(meta.inner));
})();

// Optional anonymous analytics (only if the user set an endpoint in the popup): the scores, no cards.
async function sendAnalytics(data, version) {
  const { analyticsUrl } = await chrome.storage.sync.get({ analyticsUrl: "" });
  if (!analyticsUrl) return;
  const event = { version, player_score: data.player.final_score, bot_score: data.bot.final_score, total_loss: data.analysis.total_loss };
  fetch(analyticsUrl.replace(/\/+$/, "") + "/analytics", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(event),
  }).catch(() => {}); // analytics never affect the game
}

chrome.runtime.onMessage.addListener((msg, sender, reply) => {
  if (msg.type !== "bot_game" && msg.type !== "suggest") return;
  ready.then((nets) => {
    const t0 = performance.now();
    const data = msg.type === "bot_game" ? PPAssistant.botGame(nets, msg.body) : PPAssistant.suggest(nets, msg.body);
    data.ms = Math.round(performance.now() - t0);
    reply({ data });
    if (msg.type === "bot_game") sendAnalytics(data, chrome.runtime.getManifest().version);
  }).catch((err) => reply({ error: err.message }));
  return true; // reply asynchronously
});
