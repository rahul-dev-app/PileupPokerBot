// Compares the in-browser value search (extension/search.js) with JAX on /tmp/pp_search_cases.json
// (written by test_search_parity.py). Exit code 1 if a check fails.
const fs = require("fs");
const S = require("./extension/search.js");
const P = require("./extension/policy.js");
const { loadNets } = require("./node_nets.js");

const nets = loadNets();
const cases = JSON.parse(fs.readFileSync(process.argv[2] || "/tmp/pp_search_cases.json"));
const code = (slots, cells) => slots.map((s, k) => s * 16 + cells[k]).sort((a, b) => a - b).join(",");
let failed = false;
const report = (name, ok, text) => { console.log(`${ok ? "ok  " : "FAIL"} ${name}: ${text}`); if (!ok) failed = true; };

// L1: features, scores and V of single afterstates.
{
  const X = S.featureBuffer();
  let maxF = 0, maxV = 0, maxI = 0, badScore = 0;
  for (const a of cases.afterstates) {
    const unseen = new Uint8Array(36);
    a.unseen.forEach((c) => (unseen[c] = 1));
    const phi = S.writeFeatures(X, 0, a.grid, 0, a.discard, 0, a.hand, 0, a.placed, S.unseenContext(unseen));
    if (phi !== a.score || S.totalScore(a.grid, 0, a.discard, 0) !== a.score || P.totalScore(a.grid, a.discard) !== a.score) badScore++;
    for (let i = 0; i < S.NF; i++) maxF = Math.max(maxF, Math.abs(X[i] - a.features[i]));
    const v = new Float64Array(1);
    S.values(nets.value, X, 1, [phi], [a.placed], v);
    maxV = Math.max(maxV, Math.abs(v[0] - a.value) * 1000);
    S.values(nets.inner, X, 1, [phi], [a.placed], v);
    maxI = Math.max(maxI, Math.abs(v[0] - a.inner) * 1000);
  }
  const n = cases.afterstates.length, ends = cases.afterstates.filter((a) => a.placed % 4 === 0).length;
  report("L1 afterstates", badScore === 0 && maxF < 1e-5 && maxV < 0.05 && maxI < 0.05,
    `${n} afterstates (${ends} end-of-round), score mismatches ${badScore} (search.js and policy.js), max |feature diff| ${maxF.toExponential(1)}, ` +
    `max |V diff| ${maxV.toFixed(4)} points (value net), ${maxI.toFixed(4)} (inner net)`);
}

// L1: beam children, distinct candidates, leaf values. L3: decisions with the same next hands.
{
  let beamBad = 0, beamMaxDiff = 0, candBad = 0, leafMax = 0, decBad = 0, valMax = 0, n = 0, nLast = 0, lastBad = 0;
  const callsJs = [0, 0], callsJax = [0, 0];
  const mismatches = [];
  for (const c of cases.searches) {
    const pos = { grid: c.grid, discard: c.discard, hand: c.hand, round: c.round };
    const res = S.searchRound(nets, pos, { samples: c.samples });
    if (c.round === 3) {
      nLast++;
      if (code(res.slots, res.cells) !== code(c.slots, c.cells) || Math.abs(res.value - c.value) > 0.01) lastBad++;
      continue;
    }
    n++;
    callsJs[0] += res.calls.value; callsJs[1] += res.calls.inner; callsJax[0] += c.calls[0]; callsJax[1] += c.calls[1];
    const beam = S.beamDecisions(nets, c.grid, c.discard, c.hand, c.round, 16);
    let ok = beam.length === c.beam_count;
    c.beam.forEach(([v, codes], k) => {
      beamMaxDiff = Math.max(beamMaxDiff, Math.abs(beam[k].value - v) * 1000);
      if (code(beam[k].slots, beam[k].cells) !== codes.join(",")) ok = false;
    });
    if (!ok) beamBad++;
    const stages = S.DEFAULT_CONFIG.halving[c.round];
    const cands = S.distinctTop(beam, stages[0][0], 512);
    if (cands.map((d) => code(d.slots, d.cells)).join("|") !== c.candidates.map((x) => x.join(",")).join("|")) candBad++;
    const items = [];
    for (const k of [0, 1]) for (let s = 0; s < 8; s++) items.push([cands[Math.min(k, cands.length - 1)], c.samples[s]]);
    const leaves = S.leafValues(nets, items, c.round + 1);
    leaves.forEach((v, i) => (leafMax = Math.max(leafMax, Math.abs(v - c.leaves[i]) * 1000)));
    const same = code(res.slots, res.cells) === code(c.slots, c.cells);
    if (!same) { decBad++; mismatches.push(`round ${c.round + 1}: JS ${res.value.toFixed(2)} vs JAX ${c.value.toFixed(2)}`); }
    else valMax = Math.max(valMax, Math.abs(res.value - c.value));
  }
  report("L1 beam children", beamBad === 0 && beamMaxDiff < 0.05, `${beamBad}/${n} positions differ in the top-24 children or their count; max |V diff| ${beamMaxDiff.toFixed(4)} points`);
  report("L1 distinct candidates", candBad === 0, `${candBad}/${n} positions differ`);
  report("L1 leaf values", leafMax < 0.05, `max |leaf diff| ${leafMax.toFixed(4)} points (2 candidates x 8 next hands per position)`);
  report("L3 decisions (injected next hands)", decBad <= Math.ceil(0.02 * n),
    `${n - decBad}/${n} same decision; same-decision max |value diff| ${valMax.toFixed(4)} points` + (mismatches.length ? `; differ: ${mismatches.join("; ")}` : ""));
  report("last round (exact)", lastBad === 0, `${nLast - lastBad}/${nLast} same decision and value`);
  report("model calls", callsJs[0] === callsJax[0] && callsJs[1] === callsJax[1], `JS ${callsJs}, JAX ${callsJax} (value net, inner net)`);
}
process.exit(failed ? 1 : 0);
