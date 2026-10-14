# Afterstate TD learning (n-tuple and MLP)

## Idea

Learn a value function of **afterstates** with temporal-difference learning, in the style of TD-Gammon and the n-tuple agents for 2048. An afterstate is the position right after a card is placed and before any new card is dealt. The game is random only when a new hand is dealt, so the value of an afterstate is a clean target: the expected final score from there.

The agent places each card on the legal action whose afterstate has the highest value. There is no policy network and no search.

## Implementation

Files: `pileup_poker/afterstate.py` (afterstates, features, models, self-play, TD targets, evaluation) and `train_afterstate.py` (training loop).

- One action is one card placement (80 actions, masked).
- `expand_afterstates(state)` builds the afterstate of every legal action: the board, the discard pile (on the 4th placement of a round the unplaced hand card is already added), the hand cards still to place, the number of cards on the board, and the set of unseen cards. It never reads the order of undealt cards.
- `value(model, params, afterstates, value_scale)`: `V = phi + f(afterstate)`, where `phi` is the current board score / `value_scale` (default 1000). With 16 cards placed, `V` is the exact final score.

Value models (`--model`):

- `ntuple`: one lookup table per line class (outer rows/columns, inner rows/columns, corners, discard pile) and per round. A line's table index is its rank multiset (empty cells included) plus one bit for "all suits equal so far". So the model is suit-invariant and shares weights across board symmetries. A linear term on small context features is added (cards placed, hand and unseen rank counts, sorted unseen suit counts). 22,880 table entries + 38 linear weights.
- `mlp`: a ReLU MLP with `--num-layers` hidden layers (default 2) of width `--hidden-dim` (default 256) on 250 features: per line (4 rows, 4 columns, corners, discard pile) the rank counts, card count, "flush still possible", "straight still possible", unseen cards of the flush suit and the current line score; plus hand and unseen multi-hots and the context features. With `--hidden-dim 512`: 392k parameters.

Training (`train_afterstate.py`):

- Each iteration plays `--num-games` new games in parallel, all in one jitted `lax.scan` (`play_games`). Self-play is epsilon-greedy (`--epsilon`, default 0.05), optionally with softmax sampling (`--temperature`).
- `lambda_returns` builds TD(lambda) targets (`--td-lambda`, default 0.8). The target bootstraps from the best legal value at the next move (after the new card is dealt). The last target is the final score. The trace is cut after a non-greedy move (Watkins), so targets estimate the value of the greedy policy. The 16th afterstate is not trained on, because its value is exact.
- The value is fit with MSE for `--update-epochs` (default 2) passes over `--num-minibatches` (default 4) minibatches, with Adam and gradient clipping (`--max-grad-norm 1.0`). `num_games * 15` must be divisible by `--num-minibatches`.
- `--augment` (default on) applies a random board symmetry and suit relabeling to each training afterstate.
- Evaluation every `--eval-every` iterations: greedy play (no exploration) on `--eval-episodes` games, logged as `eval/mean_final_reward`.
- `--save-every`, `--save-dir`: writes `afterstate_params.pkl` (model type, hidden size, value scale, weights) and `afterstate_checkpoint.pkl` (also optimizer state, iteration, RNG key, run name). `--resume <checkpoint>` continues a run; `--num-iterations` is the total for the whole run. `--init-params <params>` starts from saved weights, for example a distilled small MLP (see [value search](value-search.md#making-value-search-cheap)).

Evaluation from Python:

```python
from pileup_poker.afterstate import evaluate_on_decks
scores = evaluate_on_decks("models/afterstate_mlp/afterstate_params.pkl", decks)  # decks: (N, 36) int
```

The policy is deterministic (`seed` has no effect).

## Training setup of the reported models

Both runs ran on one RTX 3090 that was shared with other training runs, so the times are upper bounds.

| | n-tuple (`ntuple_long`) | MLP (`mlp_long`) |
|---|---|---|
| command flags | `--model ntuple --learning-rate 0.01` | `--model mlp --hidden-dim 512 --learning-rate 1e-3` |
| games per iteration | 1024 | 1024 |
| iterations | 15,000 (15.4M games) | 10,000 (10.2M games) |
| other settings | defaults: TD(0.8), epsilon 0.05, 2 epochs x 4 minibatches, Adam, augmentation on | same |
| throughput | ~4,900 games/s | ~3,400 games/s |
| wall time | 55 min | 55 min |

Evaluation curve (`eval/mean_final_reward`, greedy play):

| iteration | n-tuple (4000 games) | MLP (2000 games) |
|---|---|---|
| 250 | 5024 | 5372 |
| 1,500 | | 5828 |
| 2,000 | 5098 | |
| 5,250 | | 5998 |
| 7,750 | 5159 | 6100 |
| 10,000 | | 6155 |
| 15,000 | 5069 | |

The n-tuple model reached about 5,000 within 250 iterations and then stayed between 5,000 and 5,230. The MLP was still improving slowly at the end.

## Results

1000 decks (`compare_agents.py --num-decks 1000 --seed 0`):

- afterstate MLP: **6133 ± 72**
- afterstate n-tuple: **5160 ± 64**

Both need one batched value evaluation of the legal afterstates per move: 1000 games take about 8 s on the GPU.

Short runs on a CPU (500 decks, where greedy scores 3153): n-tuple 4586 after 150 iterations x 256 games (85 s), MLP 3958 after 120 iterations x 128 games (115 s).

## What worked, what did not

- Afterstate TD is simple, fast and strong. The MLP alone beats [MC search](mc-search.md) with greedy rollouts and AlphaZero, and it is close to the best PPO models.
- The n-tuple model learns fastest but plateaus at about 5,100. Its features (rank multiset + one suit bit per line) cannot express interactions between lines or what the unseen cards allow.
- The MLP's hand-made line features (flush/straight still possible, unseen cards of the flush suit) helped it go beyond the n-tuple model.
- The learned MLP is the leaf evaluator of [value search](value-search.md), the strongest agent. It was also tried as a PPO critic ([PPO](ppo.md#afterstate-critic-experiments)).

## Reproduce

```bash
# n-tuple (55 min on a shared RTX 3090)
python train_afterstate.py --model ntuple --learning-rate 0.01 --num-games 1024 \
  --num-iterations 15000 --eval-every 250 --eval-episodes 4000 \
  --run-name ntuple_long --save-dir models/afterstate_ntuple

# MLP (55 min on a shared RTX 3090)
python train_afterstate.py --model mlp --hidden-dim 512 --learning-rate 1e-3 --num-games 1024 \
  --num-iterations 10000 --eval-every 250 \
  --run-name mlp_long --save-dir models/afterstate_mlp

tensorboard --logdir runs/afterstate

python compare_agents.py --num-decks 1000 --seed 0 --agents greedy \
  afterstate=models/afterstate_mlp/afterstate_params.pkl \
  afterstate=models/afterstate_ntuple/afterstate_params.pkl
```
