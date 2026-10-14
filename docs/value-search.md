# Value search (round-level expectimax with the afterstate MLP)

This is the strongest agent in the repo: **6881 ± 73** on the 1000-deck benchmark (K=8, S=32). A cheap version with beam search, successive halving and a small network for the inner moves plays about as well (6897 ± 72, the difference is within noise) with 9.5 GFLOP instead of 13.8 TFLOP per game: see [Making value search cheap](#making-value-search-cheap). Its JavaScript/WebAssembly port runs in the Chrome extension in about 0.16 s per game on one CPU thread: see [Browser port](#browser-port).

An end-to-end description of the algorithm the extension runs is in [algorithm.md](algorithm.md).

## Idea

Plan a whole round at once, and judge positions with the trained [afterstate MLP](afterstate-td.md) instead of greedy rollouts. The algorithm is explained in [algorithm.md, section 3](algorithm.md#3-value-search-the-full-version): rank every round decision by V of its end-of-round afterstate, keep the best K distinct ones, average V over S sampled next hands (the next round played by round-level V-argmax), and play the candidate with the best mean. With `lookahead` > 1, more rounds are played on each sample before V is taken.

## Implementation

`pileup_poker/value_search.py`:

- `round_values(model, params, value_scale, grid, discard, hand, round_idx)`: V of the end-of-round afterstate of every round decision for one game (218,400 in round 1, 59,400 in round 2, 8,400 in round 3, 120 in round 4), computed in chunks with `lax.map`.
- `apply_decision`: board and discard after a decision.
- `value_search_plans(model, params, value_scale, state, round_idx, keys, ...)`: the decision for a batch of games at the start of a round. Jitted; one PRNG key per game. Options (defaults in parentheses):
  - `num_candidates` (8): K. `num_samples` (0): S; 0 is the plain round-level V-argmax (no chance node).
  - `pool_size` (512): top decisions checked for symmetric copies (`search_agent.canonical_hash`) before keeping K.
  - All candidates of a game use the same S determinized next-hand copies (common random numbers).
  - `lookahead` (1): rounds played by V-argmax on each copy before taking V.
  - `inner` (`"round"`): `"card"` plays the lookahead rounds card by card like the afterstate agent (cheaper, so more samples fit in the same time).
  - Cheaper options (off by default): `outer="beam"` with `beam_width`, `round1_symmetry`, `halving`, `exact_last`, `inner_model`/`inner_params`; see [Making value search cheap](#making-value-search-cheap).
- `value_search_plans_with_calls`: the same, and also returns the number of model calls.
- `evaluate_on_decks(model_path, decks, num_candidates, num_samples, lookahead, ...)`: final scores on fixed decks. Results depend only on the decks, the options and `seed`. With `return_calls=True` it also returns the model calls per deck and round.

It never reads the real order of undealt cards (see [determinization](symmetries-and-determinization.md)).

There is no extra training: it uses the `mlp_long` afterstate model (`models/afterstate_mlp/afterstate_params.pkl`).

## Settings sweep

First 200 decks of the benchmark (seed 0), afterstate MLP, `inner="round"`, RTX 3090 shared with training runs. Time per game includes compilation.

| K (candidates) | S (next hands) | lookahead | mean ± SE | paired diff vs S=0 | s/game |
|---|---|---|---|---|---|
| 8 | 0 (V-argmax) | – | 6457 ± 153 | 0 | 0.10 |
| 8 | 8 | 1 | 6549 ± 163 | +92 ± 118 | 0.50 |
| 8 | 8 | 3 | 6626 ± 155 | +169 ± 136 | 1.11 |
| 4 | 16 | 1 | 6726 ± 149 | +269 ± 103 | 0.55 |
| 8 | 32 | 1 | **6759 ± 152** | +302 ± 107 | 2.09 |
| 16 | 16 | 1 | 6629 ± 158 | +172 ± 132 | 4.05 |

The card-by-card afterstate MLP agent scores 5918 on the same 200 decks.

Notes (200 decks are few, so only large differences are clear):

- Planning the whole round with V (S=0) already beats the card-by-card afterstate agent: +539 ± 110 on the same decks, and it is still cheap (0.1 s/game including compilation).
- Sampling next hands adds about +270 to +300 (K=4, S=16 and K=8, S=32).
- More samples looked better than more candidates (K=4, S=16 vs K=16, S=16: +97 ± 97, not significant), and a 3-round lookahead did not clearly beat 1 round (+77 ± 132). So K=8, S=32, lookahead 1 was used for the 1000-deck runs.

## Results

1000 decks (`compare_agents.py --num-decks 1000 --seed 0`), lookahead 1:

| setting | mean ± SE | paired diff vs K=8 S=32 | time for 1000 games (shared RTX 3090) |
|---|---|---|---|
| K=8, S=32 | **6881 ± 73** | 0 | 1528 s (1.5 s/game) |
| K=4, S=16 | 6812 ± 71 | -69 ± 46 | 668 s (0.67 s/game) |

All agents on the same decks: [results.md](results.md).

## Cost

Almost all the cost is MLP evaluations. Per game with K=8, S=32, lookahead 1:

- Round 1: 218,400 round decisions are ranked, then 8 x 32 next-round searches over 59,400 decisions each (about 15 million afterstate evaluations).
- Rounds 2 and 3 are much cheaper (59,400 and 8,400 decisions, and next rounds with 8,400 and 120 decisions).

In total this is about **14 TFLOP** of MLP evaluations per game (the MLP has 250 inputs and two hidden layers of 512, about 0.8 MFLOP per evaluation). On an RTX 3090 a single game's round 1 takes about 3 s, and batched evaluation takes about 1.5 s per game.

## What worked, what did not

- Using the learned value at the end of the round is much better than greedy rollouts (6881 vs 4973 for MC search at a similar cost).
- The round-level argmax alone (S=0) is a cheap improvement over the card-by-card afterstate agent, and sampling the next hand (a chance node) adds about +300 more ([settings sweep](#settings-sweep)).
- With the default settings it is too expensive to run in a browser. The cheap settings below need about 1/1,450 of the compute for the same strength (within noise).

## Making value search cheap

Goal: make value search cheap enough for one CPU core (in the browser) without losing strength. The cost is counted in **value-model calls per game**: afterstates a sequential implementation would evaluate with the MLP (legal, not repeated, not final; a full board's value is the exact score). All results are on the 1000-deck benchmark (`compare_agents.py --num-decks 1000 --seed 0`, search seed 0), with paired differences against the per-deck scores of value search K=8 S=32 (**6881**). Scores and options of every run are in `benchmarks/results/fast_search.json`, per-deck scores and calls in `benchmarks/results/fast_search/`.

### What was changed

New options of `value_search_plans` / `evaluate_on_decks` (all off by default):

1. `inner="card"` (existed): play the sampled next round card by card (the afterstate agent's policy, about 150 calls) instead of ranking all its decisions (59,400 calls).
2. `exact_last=True`: in the round before the last, play the sampled last round by its exact best decision (120 decisions, scoring only, no MLP calls). Card-by-card play in the last round was a weak spot.
3. `round1_symmetry=True`: on the empty board, rank one decision per orbit of the 32 board symmetries (7,905 of 218,400).
4. `outer="beam", beam_width=B`: find the candidate round decisions by a beam search of width B over the 4 placements of the round, scored by V of the partial afterstates (described in [algorithm.md, section 4a](algorithm.md#4-the-cheap-version-what-the-extension-runs)). The last step's end-of-round afterstates are ranked, symmetric copies removed, and the best K kept as before.
5. `halving=((n1, s1), (n2, s2), ...)`: successive halving. The best n_i candidates so far (by mean over the copies seen so far) get s_i more determinized copies each; all candidates of a stage use the same copies. `halving_per_round` (in `evaluate_on_decks`) sets one schedule per round.
6. `inner_model`/`inner_params` (`inner_model_path` in `evaluate_on_decks`): a **small value network plays the sampled next round card by card**, and the 512-wide MLP only scores the resulting end-of-round afterstate (1 call per copy).

Small networks (`train_distill.py`, then `train_afterstate.py --init-params`): the 512-wide MLP is distilled into smaller MLPs (same 250 features) on afterstates from games of the afterstate agent with 10% random moves (16 random legal afterstates per position, including the end-of-round ones with an empty hand, plus 16 random whole-round decisions per round start), with random board symmetries and suit relabelings, then fine-tuned with TD(lambda) exactly like the original training (lr 3e-4).

### Results

MLP sizes: the 512-wide MLP has 390,656 multiply-adds (MACs) per call (250 -> 512 -> 512 -> 1), the 256-wide one 129,792, 128-wide 48,512, 64-wide 20,160 (features not counted; they are a few hundred cheap operations). GFLOP = 2 x MACs x calls. GPU time: seconds per game after compilation, batches of 40 games, RTX 3090 (times marked * were measured while training jobs shared the GPU; the reference time is from the original 1000-deck run, also on a shared GPU). CPU time: one decision at a time for one game on one core of a Ryzen 7 7700X (`JAX_PLATFORMS=cpu`, single-threaded XLA), mean over 10 games unless noted, a rough stand-in for WebAssembly in a browser.

`b16` = `outer="beam", beam_width=16`; all `b16` rows use `inner="card", exact_last=True`. Halving schedules: `hA` = ((16,8),(8,8),(4,16),(2,32)) (16 candidates, finalists get 64 copies), `hE` = ((8,8),(4,8),(2,16)), `hF` = ((32,8),(16,8),(8,16),(4,32)), `hC` = ((32,4),(16,4),(8,8),(4,16),(2,32)), `hD` = ((16,4),(8,4),(4,8),(2,16)). `P1` = `hA` in rounds 1 and 3, `hE` in round 2. In `fast_search.json` these runs are named `b16_hA16`, `b16_hC32`, `b16_hD16`, `b16_hE8`, `b16_hF32` and `b16_P1*`.

| config | 512-MLP calls / game | small-MLP calls / game | GFLOP / game | mean ± SE | paired diff vs 6881 | W/T/L | GPU s/game | CPU s/game |
|---|---|---|---|---|---|---|---|---|
| value search K=8 S=32, `inner="round"` (reference) | 17,643,000 | – | 13,800 | 6881 ± 73 | 0 | – | 1.53* | 161 (1 game) |
| round-level V-argmax (S=0) | 286,200 | – | 224 | 6655 ± 72 | -226 ± 55 | 326/256/418 | 0.017 | |
| `b16`, V-argmax (S=0) | 4,105 | – | 3.2 | 6656 ± 72 | -225 ± 56 | 328/260/412 | 0.003* | 0.087 |
| card K=8 S=32 | 359,416 | – | 281 | 6813 ± 73 | -69 ± 37 | 168/645/187 | 0.031 | 5.0 (3 games) |
| card K=16 S=64 | 579,064 | – | 452 | 6858 ± 73 | -23 ± 47 | 275/454/271 | 0.069 | |
| card K=32 S=64 | 871,928 | – | 681 | 6913 ± 74 | +32 ± 50 | 296/429/275 | 0.126 | |
| card K=8 S=32, exact last round | 349,688 | – | 273 | 6843 ± 73 | -38 ± 30 | 113/757/130 | 0.029 | |
| card K=8 S=32, round-1 symmetry | 148,921 | – | 116 | 6847 ± 74 | -34 ± 46 | 264/450/286 | 0.020 | |
| `b16` K=8 S=32 | 67,590 | – | 53 | 6859 ± 73 | -22 ± 40 | 191/598/211 | 0.013 | |
| `b16` K=8 S=16 | 35,846 | – | 28 | 6815 ± 73 | -66 ± 47 | 283/407/310 | 0.013* | |
| `b16` K=16 S=32 | 131,078 | – | 102 | 6835 ± 72 | -46 ± 46 | 238/498/264 | 0.054* | |
| `b16` `hA` | 83,463 | – | 65 | 6898 ± 72 | +17 ± 49 | 278/443/279 | 0.021* | |
| `b16` `hC` | 99,344 | – | 78 | 6904 ± 73 | +23 ± 48 | 296/419/285 | 0.057* | |
| `b16` `hF` | 162,834 | – | 127 | 6916 ± 73 | +35 ± 52 | 307/398/295 | 0.078* | |
| `b16` `hD` | 43,782 | – | 34 | 6823 ± 73 | -58 ± 47 | 284/407/309 | 0.033* | |
| `b16` `hE` | 35,846 | – | 28 | 6818 ± 73 | -63 ± 44 | 234/495/271 | 0.028* | |
| **`b16` `P1`** (512 MLP only) | **65,031** | – | **51** | **6902 ± 71** | **+21 ± 47** | 272/452/276 | 0.014 | 1.47 |
| **`b16` `P1`, 128-wide inner policy** | **4,550** | **60,929** | **9.5** | **6897 ± 72** | **+16 ± 49** | 299/402/299 | 0.011 | 0.65 |
| `b16` `P1`, 64-wide inner policy | 4,550 | 60,929 | 6.0 | 6859 ± 73 | -22 ± 51 | 306/370/324 | 0.011 | 0.59 |
| `b16` `P1`, n-tuple inner policy | 4,551 | 60,929 (n-tuple) | 3.6 | 6689 ± 71 | -193 ± 59 | 342/230/428 | 0.034* | |
| beam width 8, `P1`, 64-wide inner policy | 2,674 | 60,933 | 4.6 | 6787 ± 73 | -94 ± 54 | 315/333/352 | 0.011* | |

Beam width (V-argmax, S=0, and K=8 S=32):

| beam width | calls / game (S=0) | mean (S=0) | calls / game (K=8 S=32) | mean (K=8 S=32) |
|---|---|---|---|---|
| all decisions | 286,200 | 6655 | 349,688 | 6843 |
| 8 | 2,226 | 6645 | | |
| 16 | 4,105 | 6656 | 67,590 | 6859 |
| 32 | 6,740 | 6674 | 70,224 | 6867 |
| 64 | 10,838 | 6682 | 74,317 | 6851 |
| 128 | 17,812 | 6684 | | |

The 1000-deck numbers are noisy: the paired SE against the reference is about 50, mostly from the random next-hand samples of both searches. So the cheap settings were also compared with each other on **4000 decks** (`--num-decks 4000`, the first 1000 are the benchmark decks), paired:

| config (4000 decks) | mean ± SE | vs `b16` `P1` | vs card K=32 S=64 |
|---|---|---|---|
| card K=32 S=64 (871,928 calls) | 6898 ± 37 | +41 ± 24 | 0 |
| `b16` `P1` (65,031 calls) | 6857 ± 36 | 0 | -41 ± 24 |
| `b16` `P1`, 128-wide inner policy | 6841 ± 36 | -16 ± 18 | -57 ± 25 |
| `b16` `P1`, 64-wide inner policy | 6837 ± 36 | -20 ± 20 | -61 ± 26 |
| `b16` `P6` (`hA` without its last stage in rounds 1 and 3) | 6848 ± 36 | -9 ± 13 | -50 ± 25 |
| `b16` `hA` | 6830 ± 37 | -27 ± 13 | -68 ± 23 |
| `b16` K=8 S=32 | 6830 ± 36 | -27 ± 18 | -68 ± 24 |
| `b16` `P3` (`hF`, `hA`, `hA`), 128-wide inner policy | 6827 ± 36 | -30 ± 20 | -71 ± 24 |
| beam width 8, `P1`, 64-wide inner policy | 6806 ± 36 | -51 ± 22 | -92 ± 26 |
| `b16` K=8 S=16 | 6782 ± 36 | -75 ± 22 | -116 ± 26 |

Search seeds 1 and 2 on the 1000 benchmark decks (mean of the 3 seeds vs 6881): `b16` `P1` 6888 (+6 ± 42), 128-wide inner policy 6857 (-24 ± 45), 64-wide inner policy 6837 (-44 ± 45).

Small networks, as the plain card-by-card afterstate agent (1000 decks, paired vs the 512-wide MLP, 6133 ± 72) and as the only value model inside `b16` `P1`:

| network | MACs / call | distilled only | distilled + TD fine-tune | as afterstate agent: diff vs 512 | `b16` `P1` with it as the value model | vs `b16` `P1` (512) |
|---|---|---|---|---|---|---|
| 512 (original) | 390,656 | – | 6133 ± 72 | 0 | 6902 ± 71 | 0 |
| 256 | 129,792 | 5839 ± 68 (8k iterations) | 6041 ± 72 (+6k TD iterations) | -92 ± 58 | 6805 ± 72 | -97 ± 49 |
| 128 | 48,512 | 5603 ± 66 (8k) | 5963 ± 68 (+10k TD) | -169 ± 59 | 6753 ± 71 | -149 ± 49 |
| 64 | 20,160 | 4631 ± 64 (8k) | 5694 ± 68 (+8k TD) | -439 ± 60 | | |

Distillation iterations: 256 new games each (about 1 s per 25 iterations); TD iterations: 1024 games each (about 6,000 games/s). The models are in `models/distill_h{64,128,256}/` and `models/td_h{64,128,256}/`.

### Findings

- **`inner="card"` costs about 40 to 70 points** at K=8 S=32 (-69 ± 37 on 1000 decks; it looked free on the first 200). Most of the loss is in the round before the last: playing the sampled last round exactly (`exact_last`, free in MLP calls) recovers about half of it (+31 ± 24).
- **Beam search finds the candidates as well as ranking every decision**: with width 16, V-argmax play is identical in strength to the exhaustive ranking (6656 vs 6655) at 4,105 instead of 286,200 calls per game. Width 8 is slightly worse inside the search (-31 ± 17 on 4000 decks, both with the 64-wide inner policy).
- **Round-1 symmetry dedupe** cuts the exhaustive round-1 ranking 28x (218,400 -> 7,905 decisions), but beam search makes it unnecessary.
- **Too few copies hurt; the gain from successive halving is small.** Halving the copies (K=8 S=16) loses 48 ± 21 (4000 decks). Successive halving (`hA`: 16 candidates, the last 2 get 64 copies; 320 next-round plays in round 1 instead of 256) was +40 ± 41 better than K=8 S=32 on 1000 decks but +0 ± 20 on 4000. `P1`, which uses fewer copies in round 2 (`hE`), was +27 ± 18 better than K=8 S=32 and +27 ± 13 better than `hA` on 4000 decks; round 2 clearly does not need more copies, but this gain is probably partly chance.
- **The value of the leaf must come from the big MLP, but the moves inside the sampled round need not.** A 128- or 64-wide network as the inner policy costs at most about 20 points (not significant on 4000 decks), while the 256- and 128-wide networks as the only value model lose 100 to 150 points. The n-tuple model is too weak as inner policy (-193).
- **Distillation alone gives weak small networks** (the targets of a 512-wide net are hard to fit; RMSE about 300 points, of which about 120 is the teacher's own asymmetry under board/suit symmetries). TD fine-tuning from the distilled weights is needed and helps a lot (128-wide: 5603 -> 5963).

### Recommended settings

```python
VS.evaluate_on_decks("models/afterstate_mlp/afterstate_params.pkl", decks,
                     inner="card", exact_last=True, outer="beam", beam_width=16,
                     halving_per_round=(((16, 8), (8, 8), (4, 16), (2, 32)),   # round 1
                                        ((8, 8), (4, 8), (2, 16)),             # round 2
                                        ((16, 8), (8, 8), (4, 16), (2, 32))),  # round 3
                     inner_model_path="models/td_h128/afterstate_params.pkl")
```

`b16` `P1` with the 128-wide inner policy: **6897 ± 72** (+16 ± 49 vs 6881; mean of 3 search seeds -24 ± 45), 4,550 calls of the 512-wide MLP and 60,929 of the 128-wide one per game, **9.5 GFLOP per game** instead of 13.8 TFLOP (1,450x less), **0.65 s per game on one CPU core** (round 1: 0.37 s, round 2: 0.19 s) instead of 161 s.

- If 51 GFLOP per game is affordable, `b16` `P1` with only the 512-wide MLP is the safest choice (6902 ± 71, 3-seed mean +6 ± 42; 1.5 s per game on one CPU core).
- The 64-wide inner policy saves another third (6.0 GFLOP, 0.59 s) for about 20 points (not significant).
- Round 1 is the most expensive decision (68% of the FLOPs of the recommended setting, 78% for `b16` `P1` with only the 512-wide MLP).

Reproduce. `benchmarks/fast_search.py` compares against per-deck reference scores from a `compare_agents.py --out-csv` file (`--reference-csv`, default `logs/reference_1000.csv`); `--references` names its columns (the agent specs; by default the two agents below), and the first one is the reference for the printed paired difference. Each config on 1000 decks takes a few seconds to a minute on an RTX 3090 after compilation.

```bash
python train_distill.py --hidden-dim 128 --num-iterations 8000 --eval-every 1000 --save-dir models/distill_h128
python train_afterstate.py --model mlp --hidden-dim 128 --init-params models/distill_h128/afterstate_params.pkl \
  --learning-rate 3e-4 --num-games 1024 --num-iterations 10000 --eval-every 500 --run-name td_h128 --save-dir models/td_h128

mkdir -p logs
python compare_agents.py --num-decks 1000 --seed 0 --sv-samples 32 \
  --agents search_value=models/afterstate_mlp/afterstate_params.pkl afterstate=models/afterstate_mlp/afterstate_params.pkl \
  --out-csv logs/reference_1000.csv
python -m benchmarks.fast_search --configs b16_P1 b16_P1_h128inner card_K8S32 beam16_argmax --num-decks 1000

JAX_PLATFORMS=cpu XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \
  taskset -c 0 python -m benchmarks.fast_search_latency --configs b16_P1_h128inner --num-decks 10
```

`--configs` takes the names in `benchmarks/fast_search.py` (`CONFIGS`) or `NAME=JSON` options, for example `b16_hC32='{"inner": "card", "exact_last": true, "outer": "beam", "beam_width": 16, "halving": [[32, 4], [16, 4], [8, 8], [4, 16], [2, 32]]}'`. `--search-seed` sets the seed of the next-hand samples, `--model` the value model (for the rows with a small network as the only value model; the `afterstate` config plays the plain card-by-card agent with it), and `--tag` a suffix for the result names (e.g. `_n4000` for the 4000-deck runs). The 128-wide TD fine-tune was run as 3,000 iterations and then resumed to 10,000. Results on the GPU can differ on a few decks between runs (float rounding in ties; 9 of 1000 decks for card K=32 S=64).

## Browser port

The recommended cheap setting (`b16` `P1` with the 128-wide inner policy) runs in the [Chrome extension](chrome-extension.md), in plain JavaScript with a 2 KB WebAssembly kernel for the MLP layers: `chrome_extension/extension/search.js`.

### Port details

`search.js` implements `value_search.py` with `outer="beam", beam_width=16, inner="card", exact_last=True`, the `P1` halving schedules and `inner_model` = `models/td_h128`. The algorithm is described in [algorithm.md, section 4](algorithm.md#4-the-cheap-version-what-the-extension-runs). Details that make it match JAX:

- V is `afterstate.value`: current score / 1000 + MLP. Features (`writeFeatures`) are computed exactly as `afterstate.mlp_features` (250 values) and also return the current score, so the scoring code is shared.
- Beam search: ties keep the first child in expansion order, like `lax.top_k`. On the empty board the first card goes only to cells 0, 1 and 5.
- Symmetric copies are found with the smallest key over the 32 board symmetries and 24 suit relabelings, with the discard pile as a set. This is exact; JAX uses a 64-bit hash.
- Successive halving ranks candidates by their sum so far (all survivors have seen the same hands, so this is the same as the mean).
- The 128-wide inner policy takes the first maximum of V. Round 4 takes the first best decision in `jax_greedy.ROUND_CANDIDATES` order.
- Rows of up to 512 afterstates are evaluated per network call. The kernel (`chrome_extension/wasm/src/lib.rs`) skips zero inputs (most features and about half of the ReLU outputs are 0) and accumulates 32 outputs at a time in SIMD registers.
- Next hands are drawn from the unseen cards (not on the board, in the discard pile or in the current hand) with a seeded PRNG (`rng(seed, round)`, mulberry32), so results are reproducible. No deck order is used.

Options of `searchRound`: `fixed` (cards already placed this round; the beam starts from them), `banned` (a card discarded early is never placed), `forced` (evaluate one given decision on the same samples; used by the compete-mode analysis).

### Parity with JAX

`chrome_extension/test_search_parity.py` writes JAX reference cases from 100 games of the round-level greedy agent (decks of seed 123, 4 round starts each); `chrome_extension/test_search.js` compares the JS search with them (commands: [chrome-extension.md, Tests](chrome-extension.md#tests)):

| check | result |
|---|---|
| features of 9,200 random afterstates (2,000 end-of-round with an empty hand) | max abs difference 4.5e-7; scores identical (also `policy.js`) |
| V of the same afterstates | max difference 0.0014 points (512-wide), 0.0012 (128-wide) |
| beam search, 300 positions (rounds 1-3) | same number of end-of-round children, same top 24 in the same order; max V difference 0.0019 points |
| distinct candidates | identical in 300/300 positions |
| leaf values (2 candidates x 8 next hands per position) | max difference 0.0021 points |
| decisions with JAX's sampled next hands injected | **300/300 identical**, values within 0.001 points |
| round 4 (exact) | 100/100 identical decisions and values |
| model calls | identical: 459,677 (512-wide) + 6,092,800 (128-wide) |

### Strength (1000 decks)

`chrome_extension/bench_search.js` plays the benchmark decks (`compare_agents.make_decks(1000, 0)`) with the JS search and its own PRNG (search seeds 0, 1, 2), in Node with `worker_threads`. Paired differences against the JAX runs of the same setting (`benchmarks/results/fast_search/b16_P1_h128inner*.npz`, other random samples) and against value search K=8 S=32 (6881). Results: `benchmarks/results/browser_search.json`, per-deck scores in `benchmarks/results/browser_search/`.

| search seed | JS mean ± SE | JAX mean ± SE | JS - JAX (paired) | JS - 6881 (paired) |
|---|---|---|---|---|
| 0 | 6847 ± 74 | 6897 ± 72 | -51 ± 52 | -34 ± 56 |
| 1 | 6854 ± 72 | 6867 ± 74 | -13 ± 48 | -27 ± 52 |
| 2 | 6886 ± 72 | 6808 ± 72 | +78 ± 49 | +5 ± 55 |
| mean of 3 seeds | **6862 ± 67** | 6857 ± 67 | **+5 ± 30** | -19 ± 46 |

The JS search is as strong as the JAX one: single-seed differences of ±50-80 are the noise of the random samples (JAX's own seeds range from 6808 to 6897). Calls per game: 4,550 (512-wide) + 60,919 (128-wide), about the same as JAX (4,550 + 60,929).

```bash
python -m chrome_extension.bench_decks   # writes the decks to /tmp/pp_decks.json
node chrome_extension/bench_search.js --seed 0 --out benchmarks/results/browser_search/seed0.json
```

### Speed

One thread, Ryzen 7 7700X, times per game and per round of a full game (mean over the decks; round 1 is the worst case):

| context | game | round 1 (max) | round 2 | round 3 | round 4 | bot game + analysis (compete mode) |
|---|---|---|---|---|---|---|
| Node 22 (`bench_search.js --decks 50 --workers 1`) | 166 ms | 114 ms (141) | 35 ms | 17 ms | 0.1 ms | 250 ms (`test_assistant.js`) |
| extension service worker, headless Chrome for Testing 153 (`bench_browser.py --games 20`) | 156 ms | 106 ms (108) | 33 ms | 18 ms | 0.1 ms | 231 ms (max 235) |
| JAX on one CPU core (`fast_search_latency.py`, 10 decks) | 650 ms | 370 ms | 190 ms | 90 ms | 3 ms | |
| plain JavaScript MLP (first version, no WebAssembly) | about 1.8 s | about 1.35 s | 0.4 s | 0.08 s | | |

Compete mode runs up to 3 searches per round (the bot's position, the player's position, the player's decision); the live tests measured 285-295 ms in the service worker. In suggest mode a round's plan takes about 0.1 s and is cached; following it needs only the move scores (under 2 ms). The plain JavaScript loops were about 12x slower than the WebAssembly SIMD kernel (about 1 multiply-add per ns), so the kernel is used for all network calls.

## Reproduce

Needs a trained afterstate MLP (see [afterstate TD](afterstate-td.md)).

```bash
python compare_agents.py --num-decks 1000 --seed 0 --agents greedy \
  search_value=models/afterstate_mlp/afterstate_params.pkl \
  --sv-candidates 8 --sv-samples 32 --sv-lookahead 1
```

Use `--sv-candidates 4 --sv-samples 16` for the K=4, S=16 row. `--sv-samples` defaults to 8, so set it explicitly. The sweep above used `value_search.evaluate_on_decks(..., game_batch=40, item_batch=8)` on the first 200 of the 1000 decks.

```python
from compare_agents import make_decks
from pileup_poker import value_search as VS
decks = make_decks(1000, 0)[:200]
scores = VS.evaluate_on_decks("models/afterstate_mlp/afterstate_params.pkl", decks,
                              num_candidates=4, num_samples=16, lookahead=1, game_batch=40, item_batch=8)
```
