# Game environment and scoring

This page describes the game rules as implemented in this repo, the two environments (Gymnasium and JAX), and how the scoring was checked against the real game on Puzzmo.

## Rules

[Pile-Up Poker](https://www.puzzmo.com/game/pile-up-poker) is a single-player card game on Puzzmo. This repo implements the normal 4x4 mode.

- Deck: 36 cards, ranks `6 7 8 9 T J Q K A`, suits `S H D C`. Card id = `rank * 4 + suit` (`pileup_poker.pileup_poker.DECK`).
- Board: 4x4 grid, empty at the start.
- 4 rounds. Each round you get 5 cards, place 4 of them on empty cells, and the 5th card goes to the discard pile.
- At the end the board holds 16 cards and the discard pile holds 4.
- Cards never move after they are placed. You see only your current hand; the order of the remaining cards is unknown.

Scored hands (10 in total):

1. the 4 rows,
2. the 4 columns,
3. the 4 corner cards, worth x2,
4. the discard pile, worth x3, but only if all first 9 hands score.

Hand values:

| hand | points |
|---|---|
| pair | 5 |
| two pairs | 60 |
| flush | 80 |
| three of a kind | 125 |
| straight | 180 |
| four of a kind | 325 |
| straight flush | 450 |

- A straight needs 4 consecutive ranks. The ace is high only (no wrap-around).
- There is no royal flush bonus and no three-card straight in this mode.

The sum of all hand points is multiplied by a factor that depends on how many of the 10 hands score:

| scoring hands | multiplier |
|---|---|
| 2-3 | x2 |
| 4-5 | x3 |
| 6-7 | x4 |
| 8-9 | x5 |
| 10 | x6 |

Because of the multiplier, scoring all 10 hands matters a lot. Random play scores about 370 on average, and the best agent in this repo scores about 6,900 (see [results](results.md)).

During the game, incomplete lines also get a partial score: 2 or 3 cards with a pair score 5, and 3 cards with three of a kind score 125. This partial score is used for reward shaping and by the greedy agent. It does not change the final score, because all lines are complete at the end.

## Environments

### Gymnasium environment (`pileup_poker/pileup_poker.py`)

`PileupPokerEnv` is the original Python environment.

- `action_space = Discrete(80)`: action = `slot * 16 + cell`, with hand slot `0..4` and board cell `0..15` (row-major).
- Dict observation:
  - `grid`: `(4, 4, 36)` one-hot cards on the board,
  - `hand`: `(5, 36)` one-hot cards of the current round,
  - `discard`: `(36,)` multi-hot cards in the discard pile,
  - `round`: `(4,)` one-hot current round (all zeros once the game is over).
- `flatten_observation(obs)` gives the same 796-dim vector as `jax_env.build_observation`.
- `get_action_mask()` returns the legal actions. With `enforce_valid_actions=True` (default) an illegal action is replaced by the first legal action.
- Reward: a non-terminal action returns 1 + the current board score (with partial lines); the last action returns the final score.
- After the 4th placement of a round the remaining hand card goes to the discard pile automatically.

### JAX environment (`pileup_poker/jax_env.py`)

A batched, jit-compiled version of the same game. All training and evaluation code uses it.

- `EnvState(deck, grid, used, current_round, num_moves, discard, done)` holds many games at once.
- `init_env_state_from_decks(decks)` starts games from fixed decks: hand `r` is `decks[:, 5r:5r+5]`.
- `step(state, action)` places one card per game. Illegal actions are replaced by the first legal action. Reward: 0 for non-terminal moves, the final score for the last move.
- `get_action_mask`, `build_observation` (796-dim), `calculate_total_score(grid, discard)` (vmapped scoring), `reset_done` (auto-reset for training).
- `determinize(state, key)` shuffles the undealt cards; see [symmetries and determinization](symmetries-and-determinization.md).

The JAX env mirrors the Gym env: the same action encoding, discard rule, hand scoring and multipliers. [JAX port](jax-port.md) shows that both give identical games and measures the speedup.

## Agreement with the Puzzmo game

The scoring was checked in two ways (Oct 2026):

1. **Game code.** The rules above were compared with the game's JavaScript bundle: royal flush and three-card straights are disabled in this mode, the ace is high only, corners count x2, the discard pile counts x3 only when the first 9 hands score, and the multiplier table is the same as in `jax_env`.
2. **Live games.** The live tests `chrome_extension/verify_live_extension.py` (suggest mode) and `chrome_extension/verify_live_compete.py` (compete mode) play full games on puzzmo.com in headless Chrome. They compare the "Total score" shown by the game with `jax_env.calculate_total_score` of the final board. The two scores were equal in every game; the games and scores are listed in [Chrome extension: tests](chrome-extension.md#tests). Three more live games, with total scores 4,980, 7,470 and 1,700, also matched.

The discard rule on the site: after 4 placements the 5th card stays in the hand and the "Next round" button moves it to the discard pile. A player can also discard any hand card earlier in the round. Both cases give the same game as in `jax_env`, because only the set of cards matters.

## Reward used for training

The environments return the final score only at the end. Training scripts change this:

- `train_jax_ppo.py --reward-shaping` (default on): after every move the reward is `r + gamma * score(next) - score(now)`, where `r` is the env reward and `score` is the current board score (with partial lines; 0 after the last move). With `--gamma 1.0` the shaped return of a game equals its final score.
- `--reward-scale` (default 1000): training rewards are divided by this, so value targets stay near 1.
- Afterstate TD and AlphaZero also divide values by 1000. See their pages.
- All logged evaluation numbers (`eval/mean_final_reward`) are raw final scores.

## Play as a human

```bash
python play_human_terminal.py            # random deck
python play_human_terminal.py --seed 42  # fixed deck order
```

- Enter `slot cell`, for example `2 11`. `slot` is the hand index `0..4`, `cell` is the board index `0..15` (row-major).
- Type `v` to list the legal moves and `q` to quit. Illegal moves are rejected.
