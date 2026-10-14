// Isolated-world helper for content.js (which runs in the page's world and cannot use chrome.* APIs).
// It passes the mode setting to content.js and relays bot_game and suggest requests to the background
// service worker, which runs the search. Messages use window.postMessage with a "ppExt" field.
(function () {
  "use strict";
  const toPage = (msg) => window.postMessage({ ppExt: msg }, location.origin);

  function sendSettings() {
    chrome.storage.sync.get({ mode: "compete" }, (s) => toPage({ type: "settings", mode: s.mode }));
  }
  chrome.storage.onChanged.addListener(sendSettings);

  window.addEventListener("message", (e) => {
    const msg = e.source === window && e.data && e.data.ppExt;
    if (!msg) return;
    if (msg.type === "hello") sendSettings();
    if (msg.type === "bot_game" || msg.type === "suggest") {
      chrome.runtime.sendMessage({ type: msg.type, body: msg.body }, (res) =>
        toPage({ type: msg.type + "_result", id: msg.id, ...(res || { error: String(chrome.runtime.lastError?.message) }) }));
    }
  });
  sendSettings();
})();
