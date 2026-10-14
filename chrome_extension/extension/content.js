// PokerGrid 4x4: reads the visible game state of Pile-Up Poker on Puzzmo.
// - Suggest mode: shows the value search's plan for the current position.
// - Compete mode: no suggestions; records each hand when it is dealt, and when the game is finished
//   asks for the bot's game on the same hands and shows the replay of "You vs Bot" (replay/replay.js)
//   in a full-screen overlay.
// The search runs in the extension's service worker (background.js, reached through bridge.js), so the
// game stays responsive; nothing is sent to a server.
// Runs in the game's iframe (runtime.html) in the page's own JavaScript world, because the game's
// Redux store is only reachable from there (a prop of the react-redux Provider, found by walking up
// the React fiber tree from a card element). Only the board, the hand and the discard pile are read;
// the face-down draw piles never are. It never plays.
(function () {
  "use strict";
  const SUIT = { S: "\u2660", H: "\u2665", D: "\u2666", C: "\u2663" };
  let store = null;
  let roundHand = null; // the round's cards in deal order, to keep hand slots stable
  let last = "";
  let mode = null; // "suggest" or "compete", sent by bridge.js
  let hands = []; // compete mode: the hands dealt in this game (Puzzmo card ids)
  let sentBoard = null, requestId = 0;
  let suggestId = 0; // the last suggest request (answers to older ones are dropped)

  function findStore() {
    const el = document.querySelector('[id^="card-"]');
    if (!el) return null;
    const key = Object.keys(el).find((k) => k.startsWith("__reactFiber$"));
    for (let f = key && el[key]; f; f = f.return) {
      const p = f.memoizedProps;
      if (p && p.store && typeof p.store.getState === "function") return p.store;
    }
    return null;
  }

  // Rows top to bottom as seen on screen: pile tableau-3 is the top row, tableau-0 the bottom.
  const unsupported = (s) => s.tableauSize !== 4 || s.isFantasyLand || s.isMoonGame;

  function readBoard(piles) {
    const board = [];
    for (let r = 3; r >= 0; r--) for (let c = 0; c < 4; c++) board.push(piles["tableau-" + r].cards[c] || null);
    return board;
  }

  function readState(s) {
    if (unsupported(s)) return { unsupported: "only the normal 4x4 game is supported" };
    const piles = s.deck.piles;
    const cards = s.deck.cards;
    const board = readBoard(piles);
    const thisRound = (id) => id && cards[id] && !cards[id].locked;
    const discard = piles.discard.cards.filter(Boolean);
    const visible = piles.visibleDrawPile.cards.map((id) => id || null);
    if (visible.filter(Boolean).length === 5) roundHand = { round: s.round, cards: visible };
    const known = roundHand && roundHand.round === s.round ? roundHand.cards : null;
    const placed = board.filter(thisRound);
    return {
      round: s.round,
      board,
      hand: known ? known.map((id) => (visible.includes(id) ? id : null)) : visible,
      placed_this_round: known ? known.filter((id) => placed.includes(id)) : placed,
      discard,
      discarded_this_round: discard.find(thisRound) || null,
    };
  }

  function label(puzzmoId) {
    const [suit, rank] = puzzmoId.split("-");
    return ({ 1: "A", 11: "J", 12: "Q", 13: "K" }[rank] || rank) + SUIT[suit];
  }

  function panel(text) {
    let el = document.getElementById("pp-assistant");
    if (!el) {
      el = document.createElement("div");
      el.id = "pp-assistant";
      el.style.cssText =
        "position:fixed;top:8px;right:8px;z-index:100000;max-width:250px;padding:6px 8px;" +
        "background:rgba(20,20,30,.85);color:#fff;font:12px/1.35 monospace;border-radius:8px;pointer-events:none;white-space:pre-wrap";
      document.body.appendChild(el);
    }
    el.textContent = text;
  }

  function clearHighlights() {
    document.querySelectorAll("[data-pp-hl]").forEach((e) => {
      e.style.outline = "";
      e.removeAttribute("data-pp-hl");
    });
  }

  function highlight(id) {
    const e = document.getElementById("card-" + id);
    if (!e) return;
    e.style.outline = "4px solid #ff2bd6";
    e.style.outlineOffset = "2px";
    e.setAttribute("data-pp-hl", "1");
  }

  function show(res) {
    clearHighlights();
    const top = res.moves[0];
    highlight(top.puzzmo);
    highlight("PLACEHOLDER_tableau-" + (3 - top.row) + "_" + top.col);
    const where = (m) => label(m.puzzmo) + " -> row " + (m.row + 1) + " col " + (m.col + 1);
    panel(
      [
        "Round " + res.round + "  (value search)",
        "Best: " + where(top),
        "Plan: " + res.plan.map((m) => label(m.puzzmo) + "@r" + (m.row + 1) + "c" + (m.col + 1)).join(" "),
        "Discard: " + label(res.discard.puzzmo),
        "Expected final: " + Math.round(res.expected_final_score),
        "Other moves (expected final):",
        ...res.moves.filter((m) => !res.plan.some((p) => p.action === m.action)).slice(0, 3).map((m) => "  " + where(m) + "  " + Math.round(m.score)),
      ].join("\n")
    );
  }

  function update() {
    if (!store && (store = findStore())) {
      // Compete mode also checks every store change, so a hand is recorded even if it is played at once.
      store.subscribe(() => mode === "compete" && record(store.getState()));
    }
    if (!store || !mode) return;
    const s = store.getState();
    if (mode === "compete") return record(s);
    const state = readState(s);
    const key = JSON.stringify(state);
    if (key === last) return;
    last = key;
    if (state.unsupported) return panel(state.unsupported);
    const left = state.hand.filter(Boolean).length;
    const roundDone = state.placed_this_round.length === 4 || state.board.every(Boolean);
    if (state.round < 0 || left === 0 || roundDone) {
      clearHighlights();
      return panel("PokerGrid 4x4: waiting for the next hand");
    }
    suggestId++;
    window.postMessage({ ppExt: { type: "suggest", id: suggestId, body: state } }, location.origin);
  }

  function showSuggestion(msg) {
    if (msg.id !== suggestId || mode !== "suggest") return; // the position changed meanwhile
    if (msg.error) {
      clearHighlights();
      return panel("PokerGrid 4x4: " + msg.error);
    }
    show(msg.data);
  }

  // ------------------------------------------------------------ compete mode

  function record(s) {
    if (unsupported(s)) return panel("Compete mode: only the normal 4x4 game is supported");
    const piles = s.deck.piles;
    const board = readBoard(piles);
    const discard = piles.discard.cards.filter(Boolean);
    const visible = piles.visibleDrawPile.cards.filter(Boolean);
    if (!board.some(Boolean) && !discard.length) hands = []; // a new game
    // A hand is recorded when its 5 cards are dealt (each card belongs to one hand only).
    if (visible.length === 5 && !hands.some((h) => h.includes(visible[0]))) hands.push(visible.slice());
    const finished = board.every(Boolean) && discard.length === 4;
    if (finished) document.getElementById("pp-assistant")?.remove();
    else panel(`Compete mode: no suggestions. Hands seen: ${hands.length}/4. The bot replay opens when you finish.`);
    const key = JSON.stringify(board);
    if (!finished || key === sentBoard) return;
    sentBoard = key;
    if (hands.length !== 4) return showMessage(`Could not record all 4 hands (saw ${hands.length}), so there is no replay for this game.`);
    requestBotGame({ hands, board, discard });
  }

  function requestBotGame(body) {
    requestId++;
    showMessage("Bot is playing your hands\u2026", { spinner: true });
    window.postMessage({ ppExt: { type: "bot_game", id: requestId, body } }, location.origin);
    overlayEl().dataset.retry = JSON.stringify(body);
  }

  function overlayEl() {
    let el = document.getElementById("pp-overlay");
    if (!el) {
      el = document.createElement("div");
      el.id = "pp-overlay";
      el.className = "ppr-overlay";
      el.setAttribute("role", "dialog");
      el.setAttribute("aria-label", "You vs Bot replay");
      const close = document.createElement("button");
      close.className = "ppr-close";
      close.textContent = "Close \u2715";
      close.addEventListener("click", () => el.remove());
      el.addEventListener("keydown", (e) => e.key === "Escape" && el.remove());
      const content = document.createElement("div");
      content.className = "pp-overlay-content";
      el.append(close, content);
      document.body.appendChild(el);
    }
    return el;
  }

  function showMessage(text, { spinner = false, retry = false } = {}) {
    const box = document.createElement("div");
    box.className = "ppr-message";
    if (spinner) box.append(Object.assign(document.createElement("div"), { className: "ppr-spinner" }));
    box.append(document.createTextNode(text));
    if (retry) {
      const b = Object.assign(document.createElement("button"), { className: "ppr-btn", textContent: "Try again" });
      b.addEventListener("click", () => requestBotGame(JSON.parse(overlayEl().dataset.retry)));
      box.append(document.createElement("br"), b);
    }
    overlayEl().querySelector(".pp-overlay-content").replaceChildren(box);
  }

  function showResult(msg) {
    const el = document.getElementById("pp-overlay");
    if (!el || msg.id !== requestId) return;
    if (msg.error) return showMessage("The bot could not play this game: " + msg.error, { retry: true });
    PPReplay.render(el.querySelector(".pp-overlay-content"), msg.data);
    el.querySelector(".ppr-app").focus();
  }

  function setMode(next) {
    if (next === mode) return;
    mode = next;
    last = "";
    clearHighlights();
    const el = document.getElementById("pp-assistant");
    if (el) el.remove();
  }

  window.addEventListener("message", (e) => {
    const msg = e.source === window && e.data && e.data.ppExt;
    if (msg && msg.type === "settings") setMode(msg.mode);
    if (msg && msg.type === "bot_game_result") showResult(msg);
    if (msg && msg.type === "suggest_result") showSuggestion(msg);
  });
  window.postMessage({ ppExt: { type: "hello" } }, location.origin);
  setInterval(update, 300);
})();
