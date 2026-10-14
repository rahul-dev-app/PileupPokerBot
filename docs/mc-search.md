# Monte Carlo rollout search

## Idea

Improve the greedy agent with lookahead, without any learning. For the current round, take the best few round decisions by immediate score, play each of them on several random copies of the game, finish every copy with the greedy agent, and play the decision with the best average final score.

## Implementation

`pileup_poker/search_agent.py` (no model file).

Each round, for every game:

1. Score every round decision (which card to discard, where the other 4 go) with the round's discard added to the pile (`jax_greedy.candidate_boards(..., include_discard=True)`).
2. Take the best `pool_size` decisions by immediate score (random tie-break). Drop decisions whose board + discard pile is a symmetric copy of a better one (`canonical_hash`, 32 cell symmetries and optionally the 24 suit relabelings). Keep the top `num_candidates` (K).
3. Play each candidate on `num_samples` (S) copies of the game where the unseen cards are shuffled (`jax_env.determinize`). Finish each copy with the greedy agent (`include_discard=True`). All candidates of a game use the same copies (common random numbers), which reduces the noise of the comparison.
4. Play the candidate with the best mean final score. In the last round nothing is random, so the agent plays the exact best decision.

Functions:

- `search_plans(state, round_idx, keys, ...)`: the decision (slots, cells, value) for a batch of games at the start of a round, one PRNG key per game.
- `evaluate_on_decks(None, decks, num_candidates=16, num_samples=16, seed=0)`: final scores on `(N, 36)` decks.

Options: `num_candidates` (K, default 16), `num_samples` (S, default 16), `pool_size` (default 512), `use_suit_symmetry` (default True), `seed`, `game_batch` (default 8, games searched at once; does not change results), `batch_size` (default 32, `lax.map` batch of the greedy rollouts).

## Compute

About K x S greedy rollouts per game in rounds 1-3. Round 1 rollouts are the most expensive, because the greedy agent's round 2 checks 59,400 decisions per rollout.

- RTX 3090 (shared with other jobs): K=16, S=16 took 240 s for 1000 games (0.24 s/game).
- Heavily loaded 8-core CPU: K=4, S=4 about 0.6 s/game; K=16, S=16 about 6.7 s/game; K=32, S=8 about 10 s/game.
- Compilation takes 35-140 s once per setting.

## Results

1000 decks, K=16, S=16: **4973 ± 57** (greedy: 3103 ± 42).

## What worked, what did not

- Lookahead on determinized copies gives +60% over greedy with no training.
- The rollout policy is the weak point: greedy rollouts underestimate how good a position is, and they all make the same short-sighted mistakes. Replacing the rollouts with a learned value function ([value search](value-search.md)) gives a much stronger agent at a similar cost.

## Reproduce

```bash
python compare_agents.py --num-decks 1000 --seed 0 --agents greedy search \
  --search-candidates 16 --search-samples 16
```
