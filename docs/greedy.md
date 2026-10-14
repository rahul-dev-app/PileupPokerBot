# Greedy baseline

## Idea

At the start of each round, try every way to play the round (which card to discard and where the other 4 cards go) and pick the one with the highest board score right after the round. It does not look at future rounds.

With `e` empty cells there are `5 * C(e, 4) * 4!` round decisions: 218,400 in round 1, 59,400 in round 2, 8,400 in round 3 and 120 in round 4.

## Implementation

- `pileup_poker/greedy_agent.py`: `GreedyPileupPokerAgent`, the original Python version on top of the Gymnasium env. It scores the board before the round's discard is added (the discard pile counts only if all 9 other hands score, which rarely matters before the end).
- `pileup_poker/jax_greedy.py`: a JAX port that gives exactly the same moves and scores (same candidate order, first maximum wins), batched over games. It is 17x faster than the Python version on the CPU and 402x faster on the GPU (see [JAX port](jax-port.md#greedy-agent)).
  - `ROUND_CANDIDATES[r]`: all round-`r` decisions as (slots, positions, discard slot) arrays.
  - `candidate_boards`, `candidate_scores`: boards and scores of all decisions for one game.
  - `greedy_plans`, `play_round`, `play_greedy(decks)`: play whole games.
  - `include_discard=True` also adds the round's discarded card to the discard pile before scoring. MC search and the PPO behavior cloning use this variant; `compare_agents.py` uses the original one (`include_discard=False`).

There is no training.

## Results

1000 decks (`compare_agents.py --num-decks 1000 --seed 0`): **3103 ± 42** (random play: 370 ± 14). `compare_agents.py` takes about 11 s for the 1000 games on an RTX 3090, most of it compilation.

## What worked, what did not

- It is a strong and cheap baseline: 8x random play.
- It is short-sighted. It often fills lines early in a way that blocks later hands, and it does not plan for the 10-hand multiplier or the discard pile.
- It is the base of the MC rollout search and the starting point (behavior cloning) for the best PPO models.

## Reproduce

```bash
python compare_agents.py --num-decks 1000 --seed 0 --agents greedy random
```

Python version with the Gymnasium env (prints the board after each round):

```python
from pileup_poker.pileup_poker import PileupPokerEnv
from pileup_poker.greedy_agent import GreedyPileupPokerAgent

env = PileupPokerEnv()
deck = ["9D", "9C", "6C", "7D", "KS", "7C", "6S", "9H", "KD", "6H",
        "9S", "QD", "QC", "AS", "TH", "KH", "KC", "7S", "8C", "AD"]  # 4 hands of 5
GreedyPileupPokerAgent(env).play(custom_deck=deck)  # custom_deck=None: random deck
```
