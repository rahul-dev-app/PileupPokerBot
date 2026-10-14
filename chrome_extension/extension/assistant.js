// The assistant's answers, computed with the cheap value search (search.js):
// - botGame: compete mode, the bot's game on the player's hands and an analysis of the player's rounds,
//   in the format that replay/replay.js draws (see docs/chrome-extension.md, "Replay data").
// - suggest: suggest mode, the plan for the rest of the round: the round is planned once and planned
//   again when the player leaves the plan.
// Runs in the extension's service worker (background.js); also loads in Node for the tests.
(function (root) {
  "use strict";
  const P = root.PPPolicy || require("./policy.js");
  const S = root.PPSearch || require("./search.js");
  const CELLS = 16, HAND = 5, ROUNDS = 4, PLAYS = 4;
  const SEED = 0;

  const random = (r) => S.rng(SEED, r); // the same next-hand samples for every search of round r

  // ------------------------------------------------------------ compete mode

  // req: {hands: 4 hands of 5 cards as dealt, board: the player's final 16 cards (row-major), discard (optional)}.
  function parseGame(req) {
    const hands = req.hands.map((h) => h.map(P.parseCard));
    if (hands.length !== ROUNDS || hands.some((h) => h.length !== HAND) || new Set(hands.flat()).size !== 20)
      throw new Error("hands needs 4 hands of 5 different cards");
    if (req.board.length !== CELLS || !req.board.every(Boolean)) throw new Error("board needs 16 cards");
    const board = req.board.map(P.parseCard);
    hands.forEach((h, r) => {
      if (h.filter((c) => board.includes(c)).length !== PLAYS) throw new Error(`the board needs exactly 4 cards of hand ${r + 1}`);
    });
    if (!board.every((c) => hands.flat().includes(c))) throw new Error("a board card is not in the hands");
    const discard = hands.map((h) => h.find((c) => !board.includes(c)));
    if (req.discard && JSON.stringify(req.discard.map(P.parseCard).sort((a, b) => a - b)) !== JSON.stringify(discard.slice().sort((a, b) => a - b)))
      throw new Error("discard does not match the hand cards that are not on the board");
    return { hands, board };
  }

  const code = (slots, cells) => slots.map((s, k) => s * CELLS + cells[k]).sort((a, b) => a - b).join(",");

  function describe(hand, r, dec) {
    const order = dec.cells.map((_, k) => k).sort((a, b) => dec.cells[a] - dec.cells[b]);
    return {
      placements: order.map((k) => ({ card: P.cardName(hand[dec.slots[k]]), cell: dec.cells[k] })),
      discard: P.cardName(dec.discard[r]),
      score: S.totalScore(dec.grid, 0, dec.discard, 0),
    };
  }

  function botGame(nets, req) {
    const { hands, board } = parseGame(req);
    let bot = { grid: new Int8Array(16).fill(-1), discard: new Int8Array(4).fill(-1) }, me = bot;
    const botRounds = [], myRounds = [], analysis = [];
    hands.forEach((hand, r) => {
      const search = (pos, forced) => S.searchRound(nets, { ...pos, hand, round: r }, { random: random(r), forced });
      const botDec = search(bot);
      const samePosition = bot.grid.every((c, i) => c === me.grid[i]) && bot.discard.every((c, i) => c === me.discard[i]);
      const best = samePosition ? botDec : search(me);
      const mySlots = [0, 1, 2, 3, 4].filter((s) => board.includes(hand[s]));
      const myCells = mySlots.map((s) => board.indexOf(hand[s]));
      const same = code(best.slots, best.cells) === code(mySlots, myCells);
      const myDec = same ? best : search(me, { slots: mySlots, cells: myCells });
      botRounds.push(describe(hand, r, botDec));
      myRounds.push(describe(hand, r, myDec));
      const round1 = (x) => Math.round(x * 10) / 10;
      analysis.push({ round: r + 1, bot_choice: describe(hand, r, best), bot_value: round1(best.value),
        player_value: round1(myDec.value), loss: round1(best.value - myDec.value), same_as_bot: same });
      bot = botDec;
      me = myDec;
    });
    return {
      hands: hands.map((h) => h.map(P.cardName)),
      settings: { model: "value_search", where: "browser", num_candidates: 16, num_samples: 64, beam_width: S.DEFAULT_CONFIG.beamWidth,
        halving: S.DEFAULT_CONFIG.halving, inner: "card", exact_last: true, seed: SEED },
      player: { rounds: myRounds, final_score: myRounds[ROUNDS - 1].score },
      bot: { rounds: botRounds, final_score: botRounds[ROUNDS - 1].score },
      analysis: { rounds: analysis, total_loss: Math.round(analysis.reduce((a, x) => a + Math.max(x.loss, 0), 0) * 10) / 10 },
    };
  }

  // ------------------------------------------------------------ suggest mode

  const plans = new Map(); // round start position -> the round plan (a searchRound result)

  // req: the visible state (see policy.js buildState). The answer is what content.js shows in the panel.
  function suggest(nets, req) {
    const st = P.buildState(req);
    const hand = st.hand, r = st.round;
    const start = st.grid.map((c) => (hand.includes(c) ? -1 : c)); // the board at the round start
    const fixed = [];
    hand.forEach((c, s) => { if (st.grid.includes(c)) fixed.push([s, st.grid.indexOf(c)]); });
    const key = JSON.stringify([r, start, st.discard, hand]);
    let plan = plans.get(key);
    const fits = (p) => fixed.every(([s, c]) => p.cells[p.slots.indexOf(s)] === c) && !(st.banned >= 0 && p.slots.includes(st.banned));
    if (!plan || !fits(plan)) {
      plan = S.searchRound(nets, { grid: start, discard: st.discard, hand, round: r }, { fixed, banned: st.banned, random: random(r) });
      plans.set(key, plan);
    }
    const actions = plan.slots.map((s, k) => s * CELLS + plan.cells[k]).filter((a) => st.grid[a % CELLS] === -1);
    const left = hand.map((c) => (st.grid.includes(c) ? -1 : c));
    const scores = S.placementValues(nets, { ...st, hand: left }, hand);
    const legal = Object.keys(scores).map(Number);
    const order = actions.concat(legal.filter((a) => !actions.includes(a)).sort((a, b) => scores[b] - scores[a]));
    const info = (a, score) => {
      const slot = Math.floor(a / CELLS), cell = a % CELLS, c = hand[slot];
      const o = { card: P.cardName(c), puzzmo: P.puzzmoId(c), slot, cell, row: cell >> 2, col: cell & 3, action: a };
      if (score !== undefined) o.score = score;
      return o;
    };
    const leftover = plan.discard[r];
    return {
      model: "value_search",
      score_name: "expected_score",
      round: r + 1,
      current_score: S.totalScore(st.grid, 0, st.discard, 0),
      expected_final_score: plan.value,
      moves: order.map((a) => info(a, scores[a])),
      plan: actions.map((a) => info(a)),
      discard: { card: P.cardName(leftover), puzzmo: P.puzzmoId(leftover) },
      final_score: r === ROUNDS - 1 ? S.totalScore(plan.grid, 0, plan.discard, 0) : null,
    };
  }

  const api = { botGame, suggest, parseGame };
  root.PPAssistant = api;
  if (typeof module !== "undefined") module.exports = api;
})(globalThis);
