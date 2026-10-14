// Post-game replay and analysis for Pile-Up Poker: "You" vs "Bot" on the same hands.
// PPReplay.render(container, data, options) draws a bot game answer (assistant.js botGame).
// Plain JavaScript, no external assets; scoring comes from policy.js (PPPolicy.scoreBreakdown),
// which must be loaded first. Styles: replay.css (all classes start with "ppr-").
(function (root) {
  "use strict";
  const SUIT = { S: "\u2660", H: "\u2665", D: "\u2666", C: "\u2663" };
  const HAND_NAME = {
    pair: "Pair", two_pairs: "Two pair", flush: "Flush", three: "Three of a kind",
    straight: "Straight", four: "Four of a kind", straight_flush: "Straight flush",
  };
  const STEPS = ["Round 1", "Round 2", "Round 3", "Round 4", "Final"];
  const FINAL = 4;
  const fmt = (n) => Math.round(n).toLocaleString("en-US");

  function el(tag, cls, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined) e.textContent = text;
    return e;
  }

  // Card names are "8H", "TS", "AD" (pileup_poker.DECK).
  function cardEl(name, cls) {
    const rank = name[0] === "T" ? "10" : name[0], suit = name[1];
    const c = el("div", "ppr-card" + (suit === "H" || suit === "D" ? " ppr-red" : "") + (cls ? " " + cls : ""));
    c.append(el("span", "ppr-rank", rank), el("span", "ppr-suit", SUIT[suit]));
    c.setAttribute("aria-label", rank + SUIT[suit]);
    return c;
  }
  const cardText = (name) => (name[0] === "T" ? "10" : name[0]) + SUIT[name[1]];
  const cellText = (cell) => "row " + (Math.floor(cell / 4) + 1) + " col " + ((cell % 4) + 1);

  // Board (16 card names or null) and discard pile after `rounds` rounds of a game.
  function position(game, rounds) {
    const board = Array(16).fill(null), discard = [];
    game.rounds.slice(0, rounds).forEach((r) => {
      r.placements.forEach((p) => (board[p.cell] = p.card));
      discard.push(r.discard);
    });
    return { board, discard };
  }

  function breakdown(pos) {
    const ids = (cards) => cards.map((c) => (c ? PPPolicy.parseCard(c) : -1));
    const discard = ids(pos.discard).concat(Array(4 - pos.discard.length).fill(-1));
    return PPPolicy.scoreBreakdown(ids(pos.board), discard);
  }

  function handLabel(h) {
    const where = h.kind === "row" ? "Row " + (h.index + 1) : h.kind === "col" ? "Column " + (h.index + 1)
      : h.kind === "corners" ? "Corners" : "Discard pile";
    const pts = h.factor > 1 ? `${h.base} \u00d7${h.factor} = +${h.points}` : "+" + h.points;
    return { where, name: HAND_NAME[h.hand], pts };
  }

  function animateNumber(node, to, reduced) {
    const from = Number(node.dataset.value || 0);
    node.dataset.value = to;
    if (reduced || from === to) return void (node.textContent = fmt(to));
    const t0 = performance.now(), ms = 700;
    const tick = (t) => {
      const k = Math.min(1, (t - t0) / ms), e = 1 - (1 - k) ** 3;
      node.textContent = fmt(from + (to - from) * e);
      if (k < 1 && node.dataset.value == to) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }

  // One side ("You" or "Bot"): board with line labels, discard pile, list of scoring hands.
  function drawSide(side, title, game, step, prevStep) {
    const rounds = Math.min(step + 1, 4);
    const pos = position(game, rounds);
    // Cards not shown at the previous step slide in (none when stepping back; all at the first render).
    const prev = position(game, prevStep === null ? 0 : Math.min(prevStep + 1, 4));
    const isNew = (card) => card && !prev.board.includes(card) && !prev.discard.includes(card);
    const bd = breakdown(pos);
    const final = step === FINAL;
    side.replaceChildren();
    side.classList.toggle("ppr-final", final);

    const head = el("header", "ppr-side-head");
    const score = el("span", "ppr-side-score");
    score.dataset.value = side.dataset.score || 0;
    head.append(el("h3", "", title), score);
    side.append(head);

    const board = el("div", "ppr-board");
    const scoredCells = new Set();
    bd.hands.forEach((h) => h.cells.forEach((c) => scoredCells.add(c)));
    const lineOf = (kind, index) => bd.hands.find((h) => h.kind === kind && h.index === index);
    let delay = 0;
    for (let row = 0; row < 4; row++) {
      for (let col = 0; col < 4; col++) {
        const i = 4 * row + col, card = pos.board[i];
        const cell = el("div", "ppr-cell" + ([0, 3, 12, 15].includes(i) ? " ppr-corner" : ""));
        cell.dataset.cell = i;
        if (card) {
          const c = cardEl(card, isNew(card) ? "ppr-enter" : "");
          if (isNew(card)) c.style.animationDelay = (delay++ * 90) + "ms";
          if (final && !scoredCells.has(i)) c.classList.add("ppr-dim");
          if (!final && game.rounds[rounds - 1].placements.some((p) => p.cell === i)) c.classList.add("ppr-current");
          cell.append(c);
        }
        board.append(cell);
      }
      board.append(lineTag(lineOf("row", row), "ppr-row-tag", final));
    }
    for (let col = 0; col < 4; col++) board.append(lineTag(lineOf("col", col), "ppr-col-tag", final));
    side.append(board);

    const pile = el("div", "ppr-pile");
    const discardHand = lineOf("discard", 0);
    pile.append(el("span", "ppr-pile-label", "Discard"));
    for (let k = 0; k < 4; k++) {
      const slot = el("div", "ppr-slot");
      const card = pos.discard[k];
      if (card) slot.append(cardEl(card, (isNew(card) ? "ppr-enter " : "") + (final && !discardHand ? "ppr-dim" : "")));
      pile.append(slot);
    }
    side.append(pile);

    const list = el("ul", "ppr-hands");
    bd.hands.forEach((h, k) => {
      const { where, name, pts } = handLabel(h);
      const li = el("li", final ? "ppr-pop" : "");
      li.style.animationDelay = k * 70 + "ms";
      li.append(el("span", "ppr-where", where), el("span", "ppr-name", name), el("span", "ppr-pts", pts));
      li.addEventListener("mouseenter", () => highlight(board, h.cells, true));
      li.addEventListener("mouseleave", () => highlight(board, h.cells, false));
      list.append(li);
    });
    if (!bd.hands.length) list.append(el("li", "ppr-none", "No scoring hands yet"));
    const sum = bd.hands.reduce((a, h) => a + h.points, 0);
    const total = el("li", "ppr-total");
    total.append(el("span", "ppr-where", `${bd.hands.length} hand${bd.hands.length === 1 ? "" : "s"}`),
      el("span", "ppr-name", `${fmt(sum)} \u00d7${bd.multiplier}`), el("span", "ppr-pts", "= " + fmt(bd.total)));
    list.append(total);
    side.append(list);
    return { score, total: bd.total };
  }

  function lineTag(h, cls, final) {
    const tag = el("div", cls + (h ? "" : " ppr-empty") + (h && final ? " ppr-pop" : ""));
    if (h) tag.append(el("span", "ppr-tag-name", HAND_NAME[h.hand]), el("span", "ppr-tag-pts", "+" + h.points));
    return tag;
  }

  function highlight(board, cells, on) {
    cells.forEach((c) => board.querySelector(`[data-cell="${c}"]`).classList.toggle("ppr-hl", on));
  }

  // Analysis of one round from the player's position: text and a mini board with the bot's choice.
  function drawRound(a, data, active, onPick) {
    const r = a.round - 1;
    const item = el("li", "ppr-round" + (active ? " ppr-active" : ""));
    item.tabIndex = 0;
    item.addEventListener("click", () => onPick(r));
    item.addEventListener("keydown", (e) => (e.key === "Enter" || e.key === " ") && onPick(r));
    const approx = r < 3 ? "~" : "";
    let text;
    if (a.same_as_bot) text = "you played the bot's choice.";
    else if (a.loss < 0.5) text = "as good as the bot's choice.";
    else text = `you lost ${approx}${fmt(a.loss)} ${r < 3 ? "expected " : ""}points.`;
    const head = el("div", "ppr-round-text");
    head.append(el("strong", "", `Round ${a.round}: `), document.createTextNode(text));
    item.append(head);
    if (!a.same_as_bot) {
      const b = a.bot_choice;
      item.append(el("div", "ppr-round-detail",
        "Bot would have played " + b.placements.map((p) => `${cardText(p.card)} at ${cellText(p.cell)}`).join(", ") +
        ` and discarded ${cardText(b.discard)}.`));
      const start = position(data.player, r).board;
      const mine = data.player.rounds[r].placements;
      const mini = el("div", "ppr-mini");
      for (let i = 0; i < 16; i++) {
        const cell = el("div", "ppr-mini-cell");
        const bot = b.placements.find((p) => p.cell === i), me = mine.find((p) => p.cell === i);
        if (start[i]) cell.append(el("span", "ppr-mini-old", cardText(start[i])));
        else if (bot) cell.append(el("span", "ppr-mini-bot" + (bot.card[1] === "H" || bot.card[1] === "D" ? " ppr-red" : ""), cardText(bot.card)));
        if (me) cell.classList.add("ppr-mini-you");
        mini.append(cell);
      }
      const legend = el("div", "ppr-mini-legend", "Bot's cards in color; your cells this round outlined.");
      const box = el("div", "ppr-mini-box");
      box.append(mini, legend);
      item.append(box);
    }
    return item;
  }

  function render(container, data, options = {}) {
    const reduced = options.reducedMotion ?? (root.matchMedia && matchMedia("(prefers-reduced-motion: reduce)").matches);
    let step = options.step ?? 0, prevStep = null, timer = null;

    container.replaceChildren();
    const app = el("div", "ppr-app" + (reduced ? " ppr-reduced" : ""));
    const summary = el("div", "ppr-summary");
    const youScore = el("span", "ppr-big"), botScore = el("span", "ppr-big");
    const you = el("span", "ppr-who"), bot = el("span", "ppr-who");
    you.append(el("span", "ppr-label", "You "), youScore);
    bot.append(el("span", "ppr-label", "Bot "), botScore);
    const verdict = el("div", "ppr-verdict");
    const diff = data.player.final_score - data.bot.final_score;
    verdict.textContent = diff > 0 ? `You beat the bot by ${fmt(diff)}!` : diff < 0 ? `The bot wins by ${fmt(-diff)}.` : "A tie!";
    summary.append(you, el("span", "ppr-vs", "vs"), bot);
    const header = el("header", "ppr-header");
    header.append(summary, verdict);

    const controls = el("nav", "ppr-controls");
    const prev = el("button", "ppr-btn", "\u25c0 Prev"), play = el("button", "ppr-btn ppr-play"), next = el("button", "ppr-btn", "Next \u25b6");
    prev.setAttribute("aria-label", "Previous round");
    next.setAttribute("aria-label", "Next round");
    const tabs = el("div", "ppr-tabs");
    const tabButtons = STEPS.map((name, k) => {
      const b = el("button", "ppr-tab", name);
      b.addEventListener("click", () => { stop(); go(k); });
      tabs.append(b);
      return b;
    });
    controls.append(prev, play, next, tabs);

    const sides = el("div", "ppr-sides");
    const youSide = el("section", "ppr-side ppr-you"), botSide = el("section", "ppr-side ppr-bot");
    sides.append(youSide, botSide);

    const analysis = el("section", "ppr-analysis");
    const lost = data.analysis.total_loss;
    analysis.append(el("h3", "", "Your rounds, checked by the bot"),
      el("p", "ppr-accuracy", lost < 0.5 ? "You matched the bot's judgement in every round."
        : `In total you gave away about ${fmt(lost)} expected points.`));
    const roundList = el("ol", "ppr-rounds");
    analysis.append(roundList,
      el("p", "ppr-note", `Bot: value search in your browser (beam search of width ${data.settings.beam_width}, up to ` +
        `${data.settings.num_candidates} candidates and ${data.settings.num_samples} sampled next hands per round). ` +
        `"Expected points" are the search's estimate of the final score from your position; round 4 is exact.`));

    app.append(header, controls, sides, analysis);
    container.append(app);
    youScore.dataset.value = 0;
    botScore.dataset.value = 0;

    function go(k) {
      prevStep = youSide.childElementCount ? step : null;
      step = Math.max(0, Math.min(FINAL, k));
      const y = drawSide(youSide, "You", data.player, step, prevStep);
      const b = drawSide(botSide, "Bot", data.bot, step, prevStep);
      for (const [s, side] of [[y, youSide], [b, botSide]]) {
        animateNumber(s.score, s.total, reduced);
        side.dataset.score = s.total;
      }
      animateNumber(youScore, y.total, reduced);
      animateNumber(botScore, b.total, reduced);
      verdict.classList.toggle("ppr-hidden", step !== FINAL);
      tabButtons.forEach((t, i) => t.setAttribute("aria-current", i === step ? "step" : "false"));
      prev.disabled = step === 0;
      next.disabled = step === FINAL;
      roundList.replaceChildren(...data.analysis.rounds.map((a) =>
        drawRound(a, data, step < FINAL && a.round - 1 === step, (r) => { stop(); go(r); })));
      app.dataset.step = step;
    }
    function stop() {
      clearInterval(timer);
      timer = null;
      play.textContent = "\u25b6 Play";
      play.setAttribute("aria-label", "Play");
    }
    function start() {
      if (step === FINAL) go(0);
      play.textContent = "\u275a\u275a Pause";
      play.setAttribute("aria-label", "Pause");
      timer = setInterval(() => (step < FINAL ? go(step + 1) : stop()), options.interval || 1800);
    }
    prev.addEventListener("click", () => { stop(); go(step - 1); });
    next.addEventListener("click", () => { stop(); go(step + 1); });
    play.addEventListener("click", () => (timer ? stop() : start()));
    app.tabIndex = 0;
    app.addEventListener("keydown", (e) => {
      if (e.key === "ArrowLeft") { stop(); go(step - 1); }
      if (e.key === "ArrowRight") { stop(); go(step + 1); }
    });

    stop();
    go(step);
    if (options.autoplay ?? !reduced) start();
    return { go, stop, destroy: () => { stop(); container.replaceChildren(); } };
  }

  root.PPReplay = { render };
})(globalThis);
