// Benchmark of the in-browser value search on the compare_agents decks (compare_agents.make_decks(1000, 0),
// written to /tmp/pp_decks.json by `python -m chrome_extension.bench_decks`).
//   node chrome_extension/bench_search.js [--decks 1000] [--workers 16] [--seed 0] [--out results.json]
// Each game is played as dealt; the search only sees the current hand and samples next hands with
// rng(seed * 100003 + deck, round). Prints mean ± SE, time per game and per round (one thread per worker).
const fs = require("fs");
const os = require("os");
const { Worker, isMainThread, parentPort, workerData } = require("worker_threads");
const S = require("./extension/search.js");
const { loadNets } = require("./node_nets.js");

function playGame(nets, deck, seed) {
  let grid = new Int8Array(16).fill(-1), discard = new Int8Array(4).fill(-1);
  const times = [], calls = [];
  for (let r = 0; r < 4; r++) {
    const hand = deck.slice(5 * r, 5 * r + 5);
    const t0 = performance.now();
    const res = S.searchRound(nets, { grid, discard, hand, round: r }, { random: S.rng(seed, r) });
    times.push(performance.now() - t0);
    calls.push([res.calls.value, res.calls.inner]);
    grid = res.grid; discard = res.discard;
  }
  return { score: S.totalScore(grid, 0, discard, 0), times, calls };
}

function arg(name, fallback) {
  const i = process.argv.indexOf("--" + name);
  return i >= 0 ? process.argv[i + 1] : fallback;
}

if (!isMainThread) {
  const nets = loadNets();
  const { decks, indices, seed } = workerData;
  for (const i of indices) parentPort.postMessage({ deck: i, ...playGame(nets, decks[i], seed * 100003 + i) });
} else {
  const decks = JSON.parse(fs.readFileSync("/tmp/pp_decks.json")).slice(0, +arg("decks", 1000));
  const workers = +arg("workers", os.cpus().length), seed = +arg("seed", 0), out = arg("out", null);
  const results = [], t0 = performance.now();
  let running = workers;
  for (let w = 0; w < workers; w++) {
    const indices = decks.map((_, i) => i).filter((i) => i % workers === w);
    new Worker(__filename, { workerData: { decks, indices, seed }, argv: process.argv.slice(2) })
      .on("message", (m) => results.push(m))
      .on("exit", () => {
        if (--running) return;
        results.sort((a, b) => a.deck - b.deck);
        const n = results.length, scores = results.map((r) => r.score);
        const mean = scores.reduce((a, b) => a + b, 0) / n;
        const se = Math.sqrt(scores.reduce((a, b) => a + (b - mean) ** 2, 0) / (n - 1) / n);
        const avg = (f) => results.reduce((a, r) => a + f(r), 0) / n;
        const rounds = [0, 1, 2, 3].map((k) => avg((r) => r.times[k]).toFixed(1)).join(" / ");
        const worst = Math.max(...results.map((r) => r.times[0]));
        console.log(`${n} decks, seed ${seed}: mean ${mean.toFixed(1)} ± ${se.toFixed(1)}; ` +
          `${avg((r) => r.times.reduce((a, b) => a + b)).toFixed(0)} ms/game per thread (rounds ${rounds} ms, worst round 1 ${worst.toFixed(0)} ms); ` +
          `calls/game ${avg((r) => r.calls.reduce((a, c) => a + c[0], 0)).toFixed(0)} value + ${avg((r) => r.calls.reduce((a, c) => a + c[1], 0)).toFixed(0)} inner; ` +
          `wall ${((performance.now() - t0) / 1000).toFixed(1)} s with ${workers} workers`);
        if (out) fs.writeFileSync(out, JSON.stringify(results));
      });
  }
}
