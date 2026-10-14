// Tests of extension/assistant.js (compete-mode bot game and suggest mode) in Node, with timings.
//   node chrome_extension/test_assistant.js   (needs /tmp/pp_decks.json from bench_decks.py)
const fs = require("fs");
const P = require("./extension/policy.js");
const S = require("./extension/search.js");
const A = require("./extension/assistant.js");
const { loadNets } = require("./node_nets.js");

const nets = loadNets();
const decks = JSON.parse(fs.readFileSync("/tmp/pp_decks.json"));
const sample = JSON.parse(fs.readFileSync(__dirname + "/replay_sample.json"));
let failed = false;
const check = (ok, text) => { if (!ok) { failed = true; console.log("FAIL " + text); } };
const keys = (o) => (o && typeof o === "object" && !Array.isArray(o) ? Object.keys(o).sort().map((k) => k + ":" + keys(o[k])).join(",") : "");

// Suggest mode: play a game by following the panel; in round 2 discard the plan's discard early, in round 3
// play the second-ranked move first (the plan must be redone around it).
function suggestGame(deck) {
  const board = Array(16).fill(null), discard = [];
  let times = [], lastRes = null;
  for (let r = 0; r < 4; r++) {
    const hand = deck.slice(5 * r, 5 * r + 5).map(P.puzzmoId);
    const placed = [];
    let early = null;
    for (let k = 0; k < 4; k++) {
      const req = { board, hand: hand.map((c) => (placed.includes(c) || c === early ? null : c)), placed_this_round: placed.slice(), discard: discard.slice(), discarded_this_round: early };
      const t0 = performance.now();
      const res = A.suggest(nets, req);
      times.push(performance.now() - t0);
      check(res.plan.length === 4 - k && res.plan[0].action === res.moves[0].action, `round ${r + 1} move ${k + 1}: plan ${res.plan.length}`);
      if (r === 1 && k === 0) { early = res.discard.puzzmo; discard.push(early); }
      const move = r === 2 && k === 0 ? res.moves[1] : res.moves[0];
      if (early) check(move.puzzmo !== early, "suggested the discarded card");
      board[move.cell] = move.puzzmo;
      placed.push(move.puzzmo);
      lastRes = res;
    }
    if (!early) discard.push(hand.find((c) => !placed.includes(c)));
  }
  const final = P.totalScore(board.map(P.parseCard), discard.map(P.parseCard));
  check(lastRes.final_score === final, `suggest: final score ${final}, panel said ${lastRes.final_score}`);
  return { final, times };
}

const botTimes = [], suggestTimes = [], scores = [];
for (let i = 0; i < 10; i++) {
  const deck = decks[i];
  // A player who places the first 4 cards of each hand into the first empty cells.
  const hands = [0, 1, 2, 3].map((r) => deck.slice(5 * r, 5 * r + 5));
  const board = [];
  hands.forEach((h) => board.push(...h.slice(0, 4)));
  const req = { hands: hands.map((h) => h.map(P.puzzmoId)), board: board.map(P.puzzmoId), discard: hands.map((h) => P.puzzmoId(h[4])) };
  let t0 = performance.now();
  const res = A.botGame(nets, req);
  botTimes.push(performance.now() - t0);
  const noSettings = (o) => keys({ ...o, settings: {} });
  check(noSettings(res) === noSettings(sample), "answer keys differ from replay_sample.json: " + keys(res));
  const myScore = P.totalScore(board, hands.map((h) => h[4]));
  check(res.player.final_score === myScore, `deck ${i}: player score ${res.player.final_score} vs ${myScore}`);
  const botBoard = Array(16).fill(-1), botDiscard = [];
  res.bot.rounds.forEach((rd) => { rd.placements.forEach((p) => (botBoard[p.cell] = P.parseCard(p.card))); botDiscard.push(P.parseCard(rd.discard)); });
  check(res.bot.final_score === P.totalScore(botBoard, botDiscard), `deck ${i}: bot score does not match its board`);
  check(res.analysis.rounds[3].loss >= 0, "last-round loss is exact, never negative");
  // The bot's own game replayed as the player: every round the same as the bot, no loss.
  const botReq = { hands: req.hands, board: botBoard.map(P.puzzmoId) };
  const again = A.botGame(nets, botReq);
  check(again.analysis.rounds.every((a) => a.same_as_bot && a.loss === 0) && again.bot.final_score === res.bot.final_score,
    `deck ${i}: replaying the bot's game is not loss-free`);
  const sg = suggestGame(deck);
  suggestTimes.push(...sg.times);
  scores.push([myScore, res.bot.final_score, res.analysis.total_loss, sg.final]);
}
const stat = (xs) => `mean ${(xs.reduce((a, b) => a + b) / xs.length).toFixed(0)} ms, max ${Math.max(...xs).toFixed(0)} ms`;
console.log("scores [player, bot, total loss, suggest-mode game]:", JSON.stringify(scores));
console.log(`bot game + analysis: ${stat(botTimes)}; suggestion: ${stat(suggestTimes)} (first move of a round = a search)`);
console.log(failed ? "RESULT FAIL" : "RESULT PASS");
process.exit(failed ? 1 : 0);
