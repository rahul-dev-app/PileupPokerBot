// Pile-Up Poker rules in plain JavaScript (a port of jax_env.py): cards, building the game state from the
// visible cards, the action mask, round stepping and scoring with a breakdown.
// Card id = rank * 4 + suit with ranks 6..A and suits S H D C (pileup_poker.DECK).
(function (root) {
  "use strict";
  const CELLS = 16, HAND = 5, ROUNDS = 4, PLAYS = 4, DECK = 36, ACTIONS = HAND * CELLS;
  const RANKS = "6789TJQKA", SUITS = "SHDC";
  const PUZZMO_RANK = { 1: "A", 10: "T", 11: "J", 12: "Q", 13: "K" };
  const HAND_SCORES = { pair: 5, two_pairs: 60, flush: 80, three: 125, straight: 180, four: 325, straight_flush: 450 };

  // ------------------------------------------------------------ cards

  function parseCard(card) {
    card = card.trim().toUpperCase();
    let rank, suit;
    if (card.includes("-")) {
      [suit, rank] = card.split("-");
      rank = PUZZMO_RANK[rank] || rank;
    } else {
      rank = card.slice(0, -1);
      suit = card.slice(-1);
      if (rank === "10") rank = "T";
    }
    const r = RANKS.indexOf(rank), s = SUITS.indexOf(suit);
    if (r < 0 || s < 0) throw new Error("bad card " + card);
    return r * 4 + s;
  }
  function puzzmoId(card) {
    const rank = RANKS[card >> 2];
    const back = { A: 1, T: 10, J: 11, Q: 12, K: 13 };
    return SUITS[card & 3] + "-" + (back[rank] || rank);
  }
  const cardName = (card) => RANKS[card >> 2] + SUITS[card & 3];

  // ------------------------------------------------------------ state

  // Returns {grid, used, round, numMoves, discard, hand, banned}; hand = the round's 5 cards by slot.
  function buildState(req) {
    const board = req.board.map((c) => (c ? parseCard(c) : -1));
    if (board.length !== CELLS) throw new Error("board needs 16 entries");
    const hand = (req.hand || []).map((c) => (c ? parseCard(c) : -1));
    const placed = (req.placed_this_round || []).map(parseCard);
    const discard = (req.discard || []).map(parseCard);
    const early = req.discarded_this_round ? parseCard(req.discarded_this_round) : -1;

    const onBoard = board.filter((c) => c >= 0);
    if (placed.some((c) => !onBoard.includes(c))) throw new Error("placed_this_round cards must be on the board");
    const placedBefore = onBoard.length - placed.length;
    const round = Math.floor(placedBefore / PLAYS);
    if (placedBefore % PLAYS || round >= ROUNDS) throw new Error("board card count does not match placed_this_round");
    const oldDiscard = discard.filter((c) => c !== early);
    if (oldDiscard.length !== round || (early >= 0 && !discard.includes(early)))
      throw new Error(`round ${round + 1} needs ${round} discards from earlier rounds`);

    const slots = hand.concat(Array(Math.max(0, HAND - hand.length)).fill(-1));
    for (const card of placed.concat(early >= 0 ? [early] : [])) slots[slots.indexOf(-1)] = card;
    if (slots.includes(-1) || slots.length !== HAND)
      throw new Error("the round's cards (hand + placed this round + early discard) must be exactly 5");
    const seen = onBoard.concat(oldDiscard, slots.filter((c) => !onBoard.includes(c)));
    if (new Set(seen).size !== seen.length) throw new Error("a card appears twice");

    const used = Array(DECK).fill(false);
    onBoard.forEach((c) => (used[c] = true));
    return {
      grid: board,
      used,
      round,
      numMoves: placed.length,
      discard: oldDiscard.concat(Array(ROUNDS - round).fill(-1)),
      hand: slots,
      banned: early >= 0 ? slots.indexOf(early) : -1,
    };
  }

  function legalMask(st) {
    const mask = new Array(ACTIONS).fill(false);
    if (st.round >= ROUNDS) return mask;
    for (let s = 0; s < HAND; s++) {
      if (s === st.banned || st.used[st.hand[s]]) continue;
      for (let c = 0; c < CELLS; c++) if (st.grid[c] === -1) mask[s * CELLS + c] = true;
    }
    return mask;
  }

  // jax_env.step for a legal action, within one round: after the 4th placement the first unplaced
  // hand card is discarded and the round ends (the next hand is unknown, so the state stops there).
  function place(st, action) {
    const slot = Math.floor(action / CELLS), cell = action % CELLS;
    const next = { ...st, grid: st.grid.slice(), used: st.used.slice(), discard: st.discard.slice() };
    next.grid[cell] = st.hand[slot];
    next.used[st.hand[slot]] = true;
    next.numMoves = st.numMoves + 1;
    if (next.numMoves === PLAYS) {
      next.discard[st.round] = st.hand.find((c) => !next.used[c]);
      next.round = st.round + 1;
      next.numMoves = 0;
    }
    return next;
  }

  // ------------------------------------------------------------ scoring (jax_env.calculate_total_score)

  // Hand name of up to 4 cards (null = no hand); generic = jax_env._generic_hand_score.
  function genericHand(cards) {
    const valid = cards.filter((c) => c >= 0);
    if (!valid.length) return null;
    const ranks = valid.map((c) => c >> 2), suits = valid.map((c) => c & 3);
    const counts = ranks.map((r) => ranks.filter((x) => x === r).length);
    const unique = new Set(ranks).size;
    const flush = suits.every((s) => s === suits[0]);
    const sorted = ranks.slice().sort((a, b) => a - b);
    const straight = valid.length === 4 && sorted.every((r, i) => i === 0 || r === sorted[i - 1] + 1);
    if (unique === 1) return "four";
    if (unique === 2) return counts.includes(3) ? "three" : "two_pairs";
    if (unique === 3) return "pair";
    if (straight && flush) return "straight_flush";
    if (flush) return "flush";
    if (straight) return "straight";
    return null;
  }

  // A row or column: a full line scores any hand, a partial line (2-3 cards) only a pair or three of a kind.
  function lineHand(cards) {
    const valid = cards.filter((c) => c >= 0);
    if (valid.length === 4) return genericHand(cards);
    if (valid.length < 2) return null;
    const ranks = valid.map((c) => c >> 2);
    const maxCount = Math.max(...ranks.map((r) => ranks.filter((x) => x === r).length));
    if (maxCount === 3) return "three";
    return maxCount === 2 ? "pair" : null;
  }

  // jax_env.calculate_total_score with every scoring hand: {hands, multiplier, total}. Each hand is
  // {kind: "row" | "col" | "corners" | "discard", index, cells (board cells, none for the discard pile),
  // hand (a HAND_SCORES key), base, factor (corners x2, discard x3), points}.
  function scoreBreakdown(grid, discard) {
    const lines = [];
    for (let i = 0; i < 4; i++) lines.push({ kind: "row", index: i, cells: [4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3] });
    for (let j = 0; j < 4; j++) lines.push({ kind: "col", index: j, cells: [j, 4 + j, 8 + j, 12 + j] });
    const hands = lines.map((l) => ({ ...l, hand: lineHand(l.cells.map((c) => grid[c])), factor: 1 }));
    const corners = [0, 3, 12, 15];
    const cornerCards = corners.map((c) => grid[c]);
    hands.push({ kind: "corners", index: 0, cells: corners, hand: cornerCards.every((c) => c >= 0) ? genericHand(cornerCards) : null, factor: 2 });
    const scoring = hands.filter((h) => h.hand).length;
    hands.push({ kind: "discard", index: 0, cells: [], hand: scoring === 9 ? genericHand(discard) : null, factor: 3 });
    const scored = hands.filter((h) => h.hand);
    scored.forEach((h) => { h.base = HAND_SCORES[h.hand]; h.points = h.base * h.factor; });
    const n = scored.length;
    const multiplier = n === 10 ? 6 : n >= 8 ? 5 : n >= 6 ? 4 : n >= 4 ? 3 : n >= 2 ? 2 : 1;
    return { hands: scored, multiplier, total: scored.reduce((a, h) => a + h.points, 0) * multiplier };
  }

  const totalScore = (grid, discard) => scoreBreakdown(grid, discard).total;

  const api = { parseCard, puzzmoId, cardName, buildState, legalMask, place, totalScore, scoreBreakdown, HAND_SCORES };
  root.PPPolicy = api;
  if (typeof module !== "undefined") module.exports = api;
})(globalThis);
