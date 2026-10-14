// Cheap value search for Pile-Up Poker in JavaScript, with the MLP layers in a WebAssembly SIMD kernel (mlp.wasm).
// A port of pileup_poker/value_search.py with the options outer="beam", beam_width=16, successive halving per
// round, inner="card" with a small inner network, exact_last=True (see "Making value search cheap" in docs/value-search.md), and the afterstate
// features and MLP value of pileup_poker/afterstate.py.
// Card id = rank * 4 + suit (ranks 6..A, suits S H D C); -1 = empty. Values are in value units
// (expected final score / value_scale) inside the search and in points in the answers.
// The search only gets the visible cards: next hands are sampled from the unseen cards with a seeded PRNG.
(function (root) {
  "use strict";
  const CELLS = 16, HAND = 5, ROUNDS = 4, PLAYS = 4, DECK = 36, STEPS = 16;
  const NF = 250, LINE_F = 14, HAND_OFF = 140, UNSEEN_OFF = 176, PLACED_OFF = 212, HAND_RANK_OFF = 228, CTX_OFF = 237;
  const LINES = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15],
    [0, 4, 8, 12], [1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15], [0, 3, 12, 15]]; // the discard pile is line 10
  const MULT = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6]; // by the number of scoring hands
  const MAX_ROWS = 512; // afterstates per network call

  // Successive halving schedules per round, ((candidates, more samples), ...): fast-search's b16_P1.
  const R1 = [[16, 8], [8, 8], [4, 16], [2, 32]], R2 = [[8, 8], [4, 8], [2, 16]];
  const DEFAULT_CONFIG = { beamWidth: 16, halving: [R1, R2, R1], poolSize: 512 };

  // ------------------------------------------------------------ scoring (jax_env.calculate_total_score)

  // Stats of the last line passed to lineStats (module-level so the hot loops do not allocate).
  let sN = 0, sUnique = 0, sMax = 0, sFlush = true, sSuit = -1, sLo = 9, sHi = -1;
  const rankCount = new Int8Array(9), four = new Int8Array(4);

  // Stats of four[0..3]; adds the rank counts to X[off..off+9) if X is given.
  function lineStats(X, off) {
    sN = 0; sUnique = 0; sMax = 0; sFlush = true; sSuit = -1; sLo = 9; sHi = -1;
    rankCount.fill(0);
    for (let k = 0; k < 4; k++) {
      const c = four[k];
      if (c < 0) continue;
      const r = c >> 2, s = c & 3;
      sN++;
      if (sSuit < 0) sSuit = s; else if (s !== sSuit) sFlush = false;
      const count = ++rankCount[r];
      if (count === 1) sUnique++;
      if (count > sMax) sMax = count;
      if (r < sLo) sLo = r;
      if (r > sHi) sHi = r;
      if (X) X[off + r] += 1;
    }
  }

  // jax_env._generic_hand_score of the last lineStats (also for fewer than 4 cards, like jax_env).
  function genericScore() {
    switch (sUnique) {
      case 0: return 0;
      case 1: return 325;
      case 2: return sMax === 3 ? 125 : 60;
      case 3: return 5;
      default: {
        const straight = sHi - sLo === 3;
        return sFlush && straight ? 450 : sFlush ? 80 : straight ? 180 : 0;
      }
    }
  }

  // jax_env._line_score: a full line scores any hand, 2-3 cards only a pair or three of a kind.
  function lineScore() {
    if (sN === 4) return genericScore();
    if (sN < 2) return 0;
    return sMax === 3 ? 125 : sMax === 2 ? 5 : 0;
  }

  function loadLine(grid, go, l) {
    const L = LINES[l];
    four[0] = grid[go + L[0]]; four[1] = grid[go + L[1]]; four[2] = grid[go + L[2]]; four[3] = grid[go + L[3]];
  }
  function loadDiscard(disc, dO) {
    four[0] = disc[dO]; four[1] = disc[dO + 1]; four[2] = disc[dO + 2]; four[3] = disc[dO + 3];
  }

  function totalScore(grid, go = 0, disc = grid, dO = 16) {
    let sum = 0, scoring = 0;
    for (let l = 0; l < 9; l++) {
      loadLine(grid, go, l);
      lineStats(null, 0);
      const s = l < 8 ? lineScore() : sN === 4 ? 2 * genericScore() : 0;
      if (s) { sum += s; scoring++; }
    }
    if (scoring === 9) {
      loadDiscard(disc, dO);
      lineStats(null, 0);
      const s = 3 * genericScore();
      if (s) { sum += s; scoring++; }
    }
    return sum * MULT[scoring];
  }

  // ------------------------------------------------------------ afterstate features (afterstate.mlp_features)

  // The features that depend only on the unseen cards.
  function unseenContext(unseen) { // unseen: (36) 0/1
    const hot = new Float32Array(DECK), bySuit = new Int8Array(4), ctx = new Float32Array(13);
    for (let c = 0; c < DECK; c++) {
      if (!unseen[c]) continue;
      hot[c] = 1;
      bySuit[c & 3]++;
      ctx[c >> 2] += 0.25;
    }
    const sorted = Array.from(bySuit).sort((a, b) => a - b);
    for (let s = 0; s < 4; s++) ctx[9 + s] = sorted[s] / 9;
    return { hot, bySuit, ctx };
  }

  function unseenOf(...cardLists) {
    const unseen = new Uint8Array(DECK).fill(1);
    for (const list of cardLists) for (const c of list) if (c >= 0) unseen[c] = 0;
    return unseen;
  }

  // Writes the 250 features of an afterstate to X[row]; returns the afterstate's current score (phi).
  function writeFeatures(X, row, grid, go, disc, dO, hand, hO, placed, U) {
    const base = row * NF;
    X.fill(0, base, base + NF);
    let sum = 0, scoring = 0, discardGeneric = 0;
    for (let l = 0; l < 10; l++) {
      if (l < 9) loadLine(grid, go, l); else loadDiscard(disc, dO);
      const o = base + l * LINE_F;
      lineStats(X, o);
      const score = lineScore();
      X[o + 9] = sN / 4;
      X[o + 10] = sFlush ? 1 : 0;
      X[o + 11] = sN === 0 || (sMax <= 1 && sHi - sLo <= 3) ? 1 : 0;
      X[o + 12] = sFlush && sSuit >= 0 ? U.bySuit[sSuit] / 9 : 0;
      X[o + 13] = score / 450;
      const s = l < 8 ? score : l === 8 ? (sN === 4 ? 2 * score : 0) : 0;
      if (s) { sum += s; scoring++; }
      if (l === 9) discardGeneric = 3 * genericScore();
    }
    if (scoring === 9 && discardGeneric) { sum += discardGeneric; scoring++; }
    for (let k = 0; k < HAND; k++) {
      const c = hand[hO + k];
      if (c < 0) continue;
      X[base + HAND_OFF + c] = 1;
      X[base + HAND_RANK_OFF + (c >> 2)] += 0.5;
    }
    X.set(U.hot, base + UNSEEN_OFF);
    X[base + PLACED_OFF + placed - 1] = 1;
    X.set(U.ctx, base + CTX_OFF);
    return sum * MULT[scoring];
  }

  // ------------------------------------------------------------ MLP (afterstate._residual)

  // The layers run in WebAssembly SIMD kernels (mlp.wasm, built from chrome_extension/wasm). Their inputs,
  // outputs and weights live in its memory, which is sized once so that the Float32Array views stay valid.
  const WASM_BYTES = 16 << 20;
  let kernel = null, heap = 0, nzScratch = 0, X_BUF = null, LEAF_X = null;

  function allocF32(n) {
    const ptr = (heap + 15) & ~15;
    heap = ptr + 4 * n;
    if (heap > kernel.memory.buffer.byteLength) throw new Error("search.js: WebAssembly memory budget exceeded");
    return new Float32Array(kernel.memory.buffer, ptr, n);
  }

  // wasmBytes: the contents of mlp.wasm. Call once before loadNet.
  function init(wasmBytes) {
    kernel = new WebAssembly.Instance(new WebAssembly.Module(wasmBytes)).exports;
    heap = kernel.heap_base();
    kernel.memory.grow(Math.ceil((heap + WASM_BYTES - kernel.memory.buffer.byteLength) / 65536));
    nzScratch = allocF32(1024).byteOffset;
    X_BUF = allocF32(MAX_ROWS * NF);
    LEAF_X = allocF32(MAX_ROWS * NF);
  }

  // buffer: an ArrayBuffer of float32 w1, b1, w2, b2, ...; meta: {sizes: [250, h, ..., 1], value_scale}.
  function loadNet(buffer, meta) {
    const all = new Float32Array(buffer), layers = [];
    let off = 0;
    for (let l = 0; l + 1 < meta.sizes.length; l++) {
      const n = meta.sizes[l], m = meta.sizes[l + 1];
      const W = allocF32(n * m), b = allocF32(m);
      W.set(all.subarray(off, off + n * m)); off += n * m;
      b.set(all.subarray(off, off + m)); off += m;
      layers.push({ W, b, n, m, out: m > 1 ? allocF32(MAX_ROWS * m) : null });
    }
    if (off !== all.length) throw new Error("weights do not match the layer sizes");
    return { layers, scale: meta.value_scale, out: allocF32(MAX_ROWS) };
  }

  // value: the network for beam values and leaves; inner: the one that plays sampled rounds card by card.
  const makeNets = (value, inner) => ({ value, inner, counts: { value: 0, inner: 0 } });

  // out[r] = f(X[r]) for rows 0..n-1 (n <= MAX_ROWS); X is a buffer in the kernel's memory. ReLU hidden layers.
  function forward(net, X, n, out) {
    let inp = X;
    const L = net.layers, last = L[L.length - 1];
    for (const { W, b, n: dim, m, out: H } of L.slice(0, -1)) {
      kernel.dense(inp.byteOffset, n, dim, W.byteOffset, b.byteOffset, m, H.byteOffset, 1, nzScratch);
      inp = H;
    }
    kernel.dot(inp.byteOffset, n, last.n, last.W.byteOffset, last.b[0], net.out.byteOffset);
    for (let r = 0; r < n; r++) out[r] = net.out[r];
  }

  // V = phi + f (afterstate.value) of the rows of X with their scores phi; final boards: exact score.
  const F_BUF = new Float64Array(MAX_ROWS);
  function values(net, X, n, phi, placed, out, outOff = 0) {
    forward(net, X, n, F_BUF);
    for (let r = 0; r < n; r++) out[outOff + r] = phi[r] / net.scale + (placed[r] === STEPS ? 0 : F_BUF[r]);
  }

  // ------------------------------------------------------------ symmetries (search_agent.canonical_hash)

  const LINE_MAPS = [[0, 1, 2, 3], [3, 1, 2, 0], [0, 2, 1, 3], [3, 2, 1, 0]];
  const CELL_PERMS = [];
  for (const transpose of [false, true])
    for (const rmap of LINE_MAPS)
      for (const cmap of LINE_MAPS) {
        const perm = [];
        for (let r = 0; r < 4; r++)
          for (let c = 0; c < 4; c++) perm.push(transpose ? cmap[c] * 4 + rmap[r] : rmap[r] * 4 + cmap[c]);
        CELL_PERMS.push(perm);
      }
  const SUIT_PERMS = [];
  (function permute(prefix, rest) {
    if (!rest.length) return SUIT_PERMS.push(prefix);
    rest.forEach((s, i) => permute(prefix.concat([s]), rest.filter((_, k) => k !== i)));
  })([], [0, 1, 2, 3]);

  // The same string for boards that are equal up to the 32 board symmetries and the 24 suit relabelings
  // (the discard pile as a set): the smallest key over all of them.
  function canonicalKey(grid, disc) {
    let best = null;
    const key = new Int8Array(20);
    for (const sp of SUIT_PERMS) {
      const relabel = (c) => (c < 0 ? -1 : (c & ~3) | sp[c & 3]);
      const d = Array.from(disc, relabel).sort((a, b) => a - b);
      let cmpDisc = 0;
      if (best) for (let k = 0; k < 4 && !cmpDisc; k++) cmpDisc = d[k] - best[k];
      if (cmpDisc > 0) continue;
      for (let k = 0; k < 4; k++) key[k] = d[k];
      for (const cp of CELL_PERMS) {
        let cmp = best ? cmpDisc : -1;
        for (let i = 0; i < 16; i++) {
          key[4 + i] = relabel(grid[cp[i]]);
          if (cmp === 0) cmp = key[4 + i] - best[4 + i];
          if (cmp > 0) break;
        }
        if (cmp < 0) { best = key.slice(); cmpDisc = 0; }
      }
    }
    return String.fromCharCode(...best.map((x) => x + 1));
  }

  // ------------------------------------------------------------ round decisions

  // Round decision: slots (4) and cells (4) in placement order; grid and discard after the round.
  function applyDecision(grid, discard, hand, round, slots, cells) {
    const g = Int8Array.from(grid), d = Int8Array.from(discard);
    for (let k = 0; k < PLAYS; k++) g[cells[k]] = hand[slots[k]];
    for (let s = 0; s < HAND; s++) if (!slots.includes(s)) d[round] = hand[s];
    return { slots: Array.from(slots), cells: Array.from(cells), grid: g, discard: d };
  }

  const PERMS4 = [];
  (function permute(prefix, rest) {
    if (!rest.length) return PERMS4.push(prefix);
    rest.forEach((s, i) => permute(prefix.concat([s]), rest.filter((_, k) => k !== i)));
  })([], [0, 1, 2, 3]);
  const HAND_COMBOS = [[0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 3, 4], [0, 2, 3, 4], [1, 2, 3, 4]];

  // The 120 decisions of the last round in jax_greedy.ROUND_CANDIDATES order (ties: the first one wins).
  function lastRoundDecisions(grid, discard, hand) {
    const empty = [];
    for (let c = 0; c < CELLS; c++) if (grid[c] < 0) empty.push(c);
    const out = [];
    for (const combo of HAND_COMBOS)
      for (const p of PERMS4) out.push(applyDecision(grid, discard, hand, ROUNDS - 1, p.map((i) => combo[i]), empty));
    return out;
  }

  // Best final score over the last round's decisions (no network calls): value_search._exact_last_round_values.
  const g16 = new Int8Array(16), d4 = new Int8Array(4);
  function exactLastBest(grid, go, disc, dO, hand, hO) {
    const empty = [];
    for (let c = 0; c < CELLS; c++) if (grid[go + c] < 0) empty.push(c);
    let best = -1;
    for (let k = 0; k < 16; k++) g16[k] = grid[go + k];
    for (let k = 0; k < 4; k++) d4[k] = disc[dO + k];
    for (let ds = 0; ds < HAND; ds++) {
      d4[ROUNDS - 1] = hand[hO + ds];
      const cards = [];
      for (let s = 0; s < HAND; s++) if (s !== ds) cards.push(hand[hO + s]);
      for (const p of PERMS4) {
        for (let k = 0; k < 4; k++) g16[empty[k]] = cards[p[k]];
        const s = totalScore(g16, 0, d4, 0);
        if (s > best) best = s;
      }
    }
    return best;
  }

  // ------------------------------------------------------------ beam search (value_search._beam_candidates)

  // Round decisions found by beam search over the round's remaining placements, best first, without
  // repeats (same cells for the same cards). fixed: [[slot, cell], ...] already placed this round;
  // banned: a hand slot that may not be placed (the player discarded it early), or -1.
  function beamDecisions(nets, grid, discard, hand, round, width, fixed = [], banned = -1) {
    const net = nets.value;
    const U = unseenContext(unseenOf(grid, discard, hand));
    const start = Int8Array.from(grid);
    for (const [s, c] of fixed) start[c] = -1;
    let beam = [Int8Array.from({ length: HAND }, (_, s) => (fixed.find((f) => f[0] === s) || [0, -1])[1])];
    const X = X_BUF, phi = new Float64Array(MAX_ROWS), placedArr = new Int8Array(MAX_ROWS).fill(0);
    const g = new Int8Array(16), d = Int8Array.from(discard), h = new Int8Array(HAND);
    let children = [];
    for (let t = fixed.length; t < PLAYS; t++) {
      const placed = PLAYS * round + t + 1, last = t === PLAYS - 1;
      const seen = new Set();
      children = [];
      for (const where of beam) {
        g.set(start);
        for (let s = 0; s < HAND; s++) if (where[s] >= 0) g[where[s]] = hand[s];
        for (let slot = 0; slot < HAND; slot++) {
          if (where[slot] >= 0 || slot === banned) continue;
          for (let cell = 0; cell < CELLS; cell++) {
            if (g[cell] >= 0 || (round === 0 && t === 0 && cell !== 0 && cell !== 1 && cell !== 5)) continue;
            const child = Int8Array.from(where);
            child[slot] = cell;
            let key = 0;
            for (let s = HAND - 1; s >= 0; s--) key = key * 17 + child[s] + 1;
            if (seen.has(key)) continue;
            seen.add(key);
            children.push({ where: child, value: 0 });
          }
        }
      }
      for (let i0 = 0; i0 < children.length; i0 += MAX_ROWS) {
        const n = Math.min(MAX_ROWS, children.length - i0);
        for (let r = 0; r < n; r++) {
          const where = children[i0 + r].where;
          g.set(start);
          let leftover = -1;
          for (let s = 0; s < HAND; s++) {
            if (where[s] >= 0) { g[where[s]] = hand[s]; h[s] = -1; } else { h[s] = hand[s]; leftover = hand[s]; }
          }
          d[round] = last ? leftover : discard[round];
          if (last) h.fill(-1);
          phi[r] = writeFeatures(X, r, g, 0, d, 0, h, 0, placed, U);
          placedArr[r] = placed;
        }
        const v = new Float64Array(n);
        values(net, X, n, phi, placedArr, v);
        for (let r = 0; r < n; r++) children[i0 + r].value = v[r];
        if (placed < STEPS) nets.counts.value += n;
      }
      children.forEach((c, i) => (c.index = i));
      children.sort((a, b) => b.value - a.value || a.index - b.index); // lax.top_k order
      beam = children.slice(0, width).map((c) => c.where);
    }
    return children.map(({ where, value }) => {
      const slots = [], cells = [];
      for (let s = 0; s < HAND; s++) if (where[s] >= 0) { slots.push(s); cells.push(where[s]); }
      return { ...applyDecision(start, discard, hand, round, slots, cells), value };
    });
  }

  // The best k decisions that are not symmetric copies of a better one (value_search._distinct_top).
  function distinctTop(decisions, k, poolSize) {
    const keys = new Set(), out = [];
    for (const dec of decisions.slice(0, poolSize)) {
      const key = canonicalKey(dec.grid, dec.discard);
      if (keys.has(key)) continue;
      keys.add(key);
      out.push(dec);
      if (out.length === k) break;
    }
    return out;
  }

  // ------------------------------------------------------------ leaves (value_search._card_leaf_values)

  // For each (candidate, next hand): play the next round card by card with the inner net, then V of the
  // end-of-round afterstate by the value net; or, if the next round is the last, its exact best score.
  // Returns the leaf values in value units, in items order.
  function leafValues(nets, items, nextRound) {
    const out = new Float64Array(items.length);
    if (nextRound === ROUNDS - 1) {
      items.forEach(([dec, hand], i) => (out[i] = exactLastBest(dec.grid, 0, dec.discard, 0, hand, 0) / nets.value.scale));
      return out;
    }
    const chunk = Math.floor(MAX_ROWS / (HAND * (CELLS - PLAYS * nextRound)));
    for (let i0 = 0; i0 < items.length; i0 += chunk) playChunk(nets, items.slice(i0, i0 + chunk), nextRound, out, i0);
    return out;
  }

  function playChunk(nets, items, nextRound, out, outOff) {
    const n = items.length, inner = nets.inner, X = X_BUF;
    const grids = new Int8Array(n * 16), discs = new Int8Array(n * 4), hands = new Int8Array(n * HAND);
    const Us = items.map(([dec, hand], i) => {
      grids.set(dec.grid, i * 16);
      discs.set(dec.discard, i * 4);
      hands.set(hand, i * HAND);
      return unseenContext(unseenOf(dec.grid, dec.discard, hand));
    });
    const phi = new Float64Array(MAX_ROWS), placedArr = new Int8Array(MAX_ROWS), v = new Float64Array(MAX_ROWS);
    const rowItem = new Int16Array(MAX_ROWS), rowAction = new Int16Array(MAX_ROWS);
    const g = new Int8Array(16), d = new Int8Array(4), h = new Int8Array(HAND);
    const leafPhi = new Float64Array(n), leafPlaced = new Int8Array(n);
    for (let t = 0; t < PLAYS; t++) {
      const placed = PLAYS * nextRound + t + 1, last = t === PLAYS - 1;
      let rows = 0;
      for (let i = 0; i < n; i++) {
        for (let slot = 0; slot < HAND; slot++) {
          const card = hands[i * HAND + slot];
          if (card < 0) continue;
          for (let cell = 0; cell < CELLS; cell++) {
            if (grids[i * 16 + cell] >= 0) continue;
            for (let k = 0; k < 16; k++) g[k] = grids[i * 16 + k];
            for (let k = 0; k < 4; k++) d[k] = discs[i * 4 + k];
            for (let k = 0; k < HAND; k++) h[k] = hands[i * HAND + k];
            g[cell] = card;
            h[slot] = -1;
            if (last) {
              for (let k = 0; k < HAND; k++) if (h[k] >= 0) d[nextRound] = h[k];
              h.fill(-1);
            }
            phi[rows] = writeFeatures(X, rows, g, 0, d, 0, h, 0, placed, Us[i]);
            placedArr[rows] = placed;
            rowItem[rows] = i;
            rowAction[rows] = slot * CELLS + cell;
            rows++;
          }
        }
      }
      values(inner, X, rows, phi, placedArr, v);
      if (placed < STEPS) nets.counts.inner += rows;
      // argmax per item (rows are in action order: the first maximum wins, like jnp.argmax)
      let r = 0;
      while (r < rows) {
        const i = rowItem[r];
        let best = r;
        for (r++; r < rows && rowItem[r] === i; r++) if (v[r] > v[best]) best = r;
        const slot = Math.floor(rowAction[best] / CELLS), cell = rowAction[best] % CELLS;
        grids[i * 16 + cell] = hands[i * HAND + slot];
        hands[i * HAND + slot] = -1;
        if (last) {
          for (let k = 0; k < HAND; k++) if (hands[i * HAND + k] >= 0) discs[i * 4 + nextRound] = hands[i * HAND + k];
          LEAF_X.set(X.subarray(best * NF, (best + 1) * NF), i * NF);
          leafPhi[i] = phi[best];
          leafPlaced[i] = placed;
        }
      }
    }
    values(nets.value, LEAF_X, n, leafPhi, leafPlaced, out, outOff);
    nets.counts.value += n;
  }

  // ------------------------------------------------------------ the search (value_search._game_plan)

  function mulberry32(a) {
    return function () {
      a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  // A PRNG seeded by a list of integers (e.g. seed, round).
  function rng(...ints) {
    let h = 0x9e3779b9;
    for (const x of ints) {
      h = Math.imul(h ^ (x >>> 0), 0x85ebca6b);
      h ^= h >>> 13;
      h = Math.imul(h, 0xc2b2ae35);
      h ^= h >>> 16;
    }
    return mulberry32(h);
  }

  // n random next hands (5 cards each) from the cards that are not on the board, in the discard pile or
  // in the current hand (the only cards a player has not seen).
  function sampleHands(grid, discard, hand, n, random) {
    const unseen = unseenOf(grid, discard, hand), pool = [];
    for (let c = 0; c < DECK; c++) if (unseen[c]) pool.push(c);
    const hands = [];
    for (let i = 0; i < n; i++) {
      const p = pool.slice();
      for (let k = 0; k < HAND; k++) {
        const j = k + Math.floor(random() * (p.length - k));
        [p[k], p[j]] = [p[j], p[k]];
      }
      hands.push(p.slice(0, HAND));
    }
    return hands;
  }

  // Value search for one game at the start of a round, or with some of the round's cards already placed.
  // pos: {grid (16, the board at the round start), discard (4), hand (5), round}.
  // opts: fixed [[slot, cell], ...] placed this round, banned slot (-1), forced {slots, cells} (evaluate only
  // that decision), seed (or random), samples (inject next hands instead of sampling), config.
  // Returns {slots, cells, grid, discard, value (points), candidates, calls: {value, inner}}.
  function searchRound(nets, pos, opts = {}) {
    const { grid, discard, hand, round } = pos;
    const config = opts.config || DEFAULT_CONFIG, fixed = opts.fixed || [], banned = opts.banned ?? -1;
    nets.counts = { value: 0, inner: 0 };
    const sameCells = (dec, f) => f.every(([s, c]) => dec.cells[dec.slots.indexOf(s)] === c);
    const isForced = (dec) => dec.slots.every((s, k) => opts.forced.cells[opts.forced.slots.indexOf(s)] === dec.cells[k]);
    const finish = (dec, value, candidates) =>
      ({ ...dec, value: value * nets.value.scale, candidates, calls: nets.counts });

    if (round === ROUNDS - 1) { // exact: V of a full board is its score
      let best = null, bestScore = -1;
      for (const dec of lastRoundDecisions(grid, discard, hand)) {
        const ok = opts.forced ? isForced(dec) : sameCells(dec, fixed) && (banned < 0 || !dec.slots.includes(banned));
        if (!ok) continue;
        const s = totalScore(dec.grid, 0, dec.discard, 0);
        if (s > bestScore) { best = dec; bestScore = s; }
      }
      return finish(best, bestScore / nets.value.scale, 1);
    }

    const stages = config.halving[Math.min(round, config.halving.length - 1)];
    const cands = opts.forced
      ? [applyDecision(grid, discard, hand, round, opts.forced.slots, opts.forced.cells)]
      : distinctTop(beamDecisions(nets, grid, discard, hand, round, config.beamWidth, fixed, banned), stages[0][0], config.poolSize);
    const total = stages.reduce((a, [, s]) => a + s, 0);
    const samples = opts.samples || sampleHands(grid, discard, hand, total, opts.random || rng(opts.seed || 0, round));
    const sums = new Float64Array(cands.length);
    let alive = cands.map((_, i) => i), offset = 0;
    for (const [n, s] of stages) {
      if (n < alive.length) alive = alive.map((c, k) => [c, k]).sort((a, b) => sums[b[0]] - sums[a[0]] || a[1] - b[1]).slice(0, n).map((x) => x[0]);
      const items = [];
      for (const c of alive) for (let k = 0; k < s; k++) items.push([cands[c], samples[offset + k]]);
      const leaves = leafValues(nets, items, round + 1);
      alive.forEach((c, a) => { for (let k = 0; k < s; k++) sums[c] += leaves[a * s + k]; });
      offset += s;
    }
    let best = alive[0];
    for (const c of alive) if (sums[c] > sums[best]) best = c;
    return finish(cands[best], sums[best] / total, cands.length);
  }

  // V (points) of the afterstate after each legal single placement: move scores for the suggestion panel.
  // st: {grid, discard, hand (the round's 5 cards, -1 = placed), round, numMoves, banned (slot or -1)}.
  function placementValues(nets, st, hand5) {
    const U = unseenContext(unseenOf(st.grid, st.discard, hand5));
    const placed = PLAYS * st.round + st.numMoves + 1, last = st.numMoves === PLAYS - 1;
    const rows = [], phi = new Float64Array(MAX_ROWS), placedArr = new Int8Array(MAX_ROWS).fill(placed);
    const g = new Int8Array(16), d = new Int8Array(4), h = new Int8Array(HAND);
    for (let slot = 0; slot < HAND; slot++) {
      if (st.hand[slot] < 0 || slot === st.banned) continue;
      for (let cell = 0; cell < CELLS; cell++) {
        if (st.grid[cell] >= 0) continue;
        g.set(st.grid); d.set(st.discard); h.set(st.hand);
        g[cell] = st.hand[slot];
        h[slot] = -1;
        if (last) { for (let k = 0; k < HAND; k++) if (h[k] >= 0) d[st.round] = h[k]; h.fill(-1); }
        phi[rows.length] = writeFeatures(X_BUF, rows.length, g, 0, d, 0, h, 0, placed, U);
        rows.push(slot * CELLS + cell);
      }
    }
    const v = new Float64Array(rows.length);
    values(nets.value, X_BUF, rows.length, phi, placedArr, v);
    const scores = {};
    rows.forEach((a, r) => (scores[a] = v[r] * nets.value.scale));
    return scores;
  }

  const api = {
    DEFAULT_CONFIG, init, loadNet, makeNets, totalScore, writeFeatures, unseenContext, unseenOf, values, canonicalKey, beamDecisions,
    distinctTop, leafValues, exactLastBest, searchRound, sampleHands, rng, placementValues, applyDecision, NF,
    featureBuffer: () => X_BUF,
  };
  root.PPSearch = api;
  if (typeof module !== "undefined") module.exports = api;
})(globalThis);
