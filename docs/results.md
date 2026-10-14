# Results

All agents played the same 1000 decks: `compare_agents.py --num-decks 1000 --seed 0`. Deck `i` is a seeded permutation of the 36 cards, and every agent gets the same 4 hands of 5 cards in the same order. Scores are final Puzzmo scores.

`vs best` is the paired difference to value search K=8 S=32, with its standard error and wins/ties/losses of the agent against it.

| agent | page | mean | SE | median | max | vs greedy | vs best | vs best SE | W/T/L vs best |
|---|---|---|---|---|---|---|---|---|---|
| value search, afterstate MLP, K=8 S=32 | [value search](value-search.md) | **6881** | 73 | 6690 | 15990 | +3778 | 0 | 0 | – |
| value search, cheap version (beam search, successive halving, 128-wide inner network) \* | [algorithm](algorithm.md) | 6897 | 72 | 6690 | 15810 | +3795 | +16 | 49 | 299/402/299 |
| value search, afterstate MLP, K=4 S=16 | [value search](value-search.md) | 6812 | 71 | 6690 | 15090 | +3709 | -69 | 46 | 249/455/296 |
| PPO attention, +60M steps (`ppo_cont_control`) | [PPO](ppo.md) | 6294 | 67 | 6090 | 14850 | +3191 | -587 | 74 | 358/57/585 |
| PPO attention + BC, 200M steps (`ppo_attn_bc`) | [PPO](ppo.md) | 6248 | 68 | 6075 | 17760 | +3145 | -633 | 76 | 361/55/584 |
| PPO continuation, residual afterstate critic | [PPO](ppo.md#afterstate-critic-experiments) | 6235 | 68 | 6075 | 14850 | +3132 | -646 | 76 | 354/53/593 |
| PPO continuation, frozen afterstate critic | [PPO](ppo.md#afterstate-critic-experiments) | 6182 | 66 | 6075 | 14850 | +3079 | -699 | 65 | 306/103/591 |
| afterstate MLP (512) | [afterstate TD](afterstate-td.md) | 6133 | 72 | 5970 | 17040 | +3030 | -748 | 61 | 262/144/594 |
| PPO from BC, 30M, frozen afterstate critic | [PPO](ppo.md#afterstate-critic-experiments) | 5723 | 66 | 5550 | 14100 | +2620 | -1159 | 68 | 242/59/699 |
| PPO from BC, 30M, normal critic | [PPO](ppo.md#afterstate-critic-experiments) | 5559 | 61 | 5475 | 15090 | +2456 | -1323 | 73 | 260/30/710 |
| afterstate n-tuple | [afterstate TD](afterstate-td.md) | 5160 | 64 | 4925 | 14490 | +2058 | -1721 | 81 | 225/7/768 |
| MC search, greedy rollouts, K=16 S=16 | [MC search](mc-search.md) | 4973 | 57 | 4710 | 16290 | +1870 | -1909 | 88 | 234/5/761 |
| AlphaZero, 64 simulations | [AlphaZero](alphazero.md) | 4551 | 56 | 4325 | 13080 | +1448 | -2331 | 82 | 172/5/823 |
| PPO MLP, 300M steps | [PPO](ppo.md) | 4270 | 52 | 4250 | 10680 | +1167 | -2611 | 69 | 87/4/909 |
| greedy | [greedy](greedy.md) | 3103 | 42 | 2850 | 10830 | 0 | -3778 | 81 | 67/0/933 |
| random | | 370 | 14 | 160 | 3125 | -2732 | -6511 | 73 | 0/0/1000 |

\* `benchmarks/fast_search.py` (`b16_P1_h128inner`, JAX, search seed 0). Other search seeds and the browser port: [value search, strength](value-search.md#strength-1000-decks).

Training cost of the learned models (one RTX 3090, shared between parallel runs):

| model | training |
|---|---|
| afterstate MLP | 10.2M self-play games, 55 min |
| afterstate n-tuple | 15.4M self-play games, 55 min |
| AlphaZero | 909k self-play games, 2.6 h |
| PPO MLP | 300M steps, 1.1 h |
| PPO attention + BC | 30k greedy games for BC + 200M steps, 2.6 h |
| PPO continuations | +60M steps, 1.3-1.5 h each |
| PPO from BC (critic experiments) | 30M steps, 25-30 min each |

Value search needs no training beyond the afterstate MLP (and, for the cheap version, the 128-wide network: [value search](value-search.md#making-value-search-cheap)). MC search needs no training.

## Main findings

- A learned afterstate value function is the most useful component: alone it beats MC search, AlphaZero and the PPO MLP, and as the leaf value of a round-level expectimax it gives the best agent.
- For PPO, the network structure (attention over cells and hand cards, factored (slot, cell) head), symmetry augmentation and a greedy warm start mattered much more than training length.
- The AlphaZero-style agent was correct but inefficient here: its network alone stayed far below greedy, and the search plateaued at about 4,550.
- Using the afterstate value as PPO critic speeds up early training but does not help a strong policy.

## Reproduce

`compare_agents.py --help` lists all options. The table above merges several runs on the same decks (same `--num-decks 1000 --seed 0`), for example:

```bash
mkdir -p logs
python compare_agents.py --num-decks 1000 --seed 0 --agents greedy random search \
  afterstate=models/afterstate_ntuple/afterstate_params.pkl \
  afterstate=models/afterstate_mlp/afterstate_params.pkl \
  ppo=models/ppo_mlp/jax_ppo_params.pkl ppo=models/ppo_attn_bc/jax_ppo_params.pkl \
  ppo=models/ppo_cont_control/jax_ppo_params.pkl \
  alphazero=models/alphazero/alphazero_params.pkl \
  --out-csv logs/compare_1000.csv
```

Value search rows: [value-search.md, Reproduce](value-search.md#reproduce).

- An agent is `NAME` or `NAME=MODEL_PATH`: `random`, `greedy`, `search` (no model), `afterstate=PATH`, `ppo=PATH`, `alphazero=PATH`, `search_value=PATH` (an afterstate params file). The same agent can be listed several times with different models.
- `--num-decks` (default 300), `--seed` (default 0): the decks. The seed also seeds the agents that sample.
- `--search-candidates`, `--search-samples`: MC search K and S (default 16, 16). `--az-simulations`: AlphaZero simulations per move (default 64). `--sv-candidates`, `--sv-samples`, `--sv-lookahead`: value search K, S and lookahead (default 8, 8, 1).
- The output shows mean ± SE, median, max, and the paired difference and wins/ties/losses against the first agent. `--out-csv` writes per-deck scores.
