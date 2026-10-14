# AlphaZero-style agent (mctx Gumbel MuZero)

## Idea

Train a policy/value network by self-play with tree search, as in AlphaZero. The search uses DeepMind's [`mctx`](https://github.com/google-deepmind/mctx) library (`gumbel_muzero_policy`) with the real JAX environment as the model. The network learns to imitate the search policy and to predict the final score.

The main difficulty is chance: a new random hand is dealt every 4 moves, and the search must not see the real future cards.

## Implementation

Files: `pileup_poker/alphazero.py` (network, search, self-play, loss, evaluation) and `train_alphazero.py` (training loop, replay buffer, checkpoints).

Network (`init_params`, `policy_value`): a pure JAX MLP (`--hidden-dim`, `--num-layers`, ReLU) maps the 796-dim observation to 80 policy logits and a value. With 512 x 3: 975k parameters.

Search (`search`, `_recurrent_fn`):

- Every move runs `mctx.gumbel_muzero_policy`. Illegal actions are masked at the root (`invalid_actions`) and inside the tree (prior logits set to -1e9).
- The tree extends only to the end of the current round. Inside a round the game is deterministic once the hand is known.
  - A move that completes a round is a leaf with discount 0. Its reward is the board-score change plus the network value of the **afterstate**: board, discard pile (with this round's discard) and the next round, with an **empty hand** (`afterstate_observation`).
  - A move that ends the game gets only the score change.
  - The search runs on `jax_env.determinize` copies, so even nodes expanded past a leaf never see the real undealt cards.
- Values are reward-to-go with the board score as potential: `V(s) = (final_score - board_score(s)) / 1000`. In-tree rewards are board-score changes / 1000.

Training:

- `self_play` plays `--games-per-iteration` games in parallel with search. Each move gives one sample (policy target = the search's `action_weights`, value target = the game's reward-to-go). Each round start also gives an afterstate sample (hand hidden, value target only), which trains the leaf evaluation. So each game adds 20 samples (16 moves + 4 afterstates).
- Samples go to a replay buffer of the most recent `--replay-size` samples. Each minibatch sample gets a random board symmetry and suit relabeling (`augment`).
- Loss (`loss_fn`): cross-entropy to the search policy + `--value-coef` x value MSE + `--l2-coef` x L2. Adam with global-norm clipping. `--train-steps` steps per iteration.
- Evaluation on a fixed deck set: `eval/mean_final_reward` (with search, `--eval-simulations`, `gumbel_scale=0`) and `eval/network_mean_final_reward` (network argmax only).
- `--save-every`, `--save-dir` (default `models/alphazero`): `alphazero_params.pkl` and `alphazero_checkpoint.pkl` (with optimizer state, RNG key and replay buffer; a resumed run gives exactly the same params as an uninterrupted one). `--resume`, `--init-params`.

Evaluation from Python:

```python
from pileup_poker.alphazero import evaluate_on_decks
scores = evaluate_on_decks("models/alphazero/alphazero_params.pkl", decks, num_simulations=64, seed=0)
```

`num_simulations=0` plays the network argmax.

## Training setup of the reported model (`az_long`)

| setting | value |
|---|---|
| games per iteration | 512 |
| simulations per move (self-play and evaluation) | 64 |
| max considered actions, Gumbel scale | 16, 1.0 (defaults) |
| train steps per iteration, batch size | 64, 1024 |
| replay size | 500,000 samples (about 25,000 games) |
| network | MLP 512 x 3 |
| learning rate, value coef, L2, grad clip | 1e-3, 1.0, 1e-6, 1.0 (defaults) |
| evaluation | every 25 iterations, 1024 decks |
| length | stopped at iteration 1,775 of 3,000 (909k games) |
| wall time | 2.6 h on a shared RTX 3090 (~96 games/s) |

Evaluation curve:

| iteration | with search (64 sims) | network only |
|---|---|---|
| 25 | 3210 | 760 |
| 225 | 3719 | 1189 |
| 450 | 4254 | 1451 |
| 900 | 4557 | 1656 |
| 1,000 | 4602 (best) | 1639 |
| 1,775 | 4531 | 1544 |

After about iteration 575 the search score stayed near 4,530 (best 4,602 at iteration 1,000), so the run was stopped.

## Results

1000 decks, 64 simulations: **4551 ± 56**.

## What worked, what did not

- The round-limited tree with an afterstate leaf value handles chance without peeking, and it works: search + network beats greedy by 1,448 points.
- But it is weaker than the much simpler afterstate MLP (6133). The network alone stays at about 1,600, far below greedy, so the search does almost all the work.
- Likely reasons: the policy target from 64 simulations over up to 80 actions is noisy; the value target (one game's reward-to-go) has high variance; and the plain MLP on the 796-dim one-hot observation has no useful structure for lines. Self-play is also slow (~96 games/s vs ~3,400 for afterstate TD).

## Reproduce

```bash
python train_alphazero.py --games-per-iteration 512 --num-simulations 64 --train-steps 64 \
  --batch-size 1024 --replay-size 500000 --hidden-dim 512 --num-layers 3 \
  --eval-every 25 --eval-episodes 1024 --eval-simulations 64 \
  --iterations 3000 --run-name az_long --save-dir models/alphazero
# the reported model was stopped (Ctrl+C) at iteration 1775; params are saved every 10 iterations (--save-every)
tensorboard --logdir runs/alphazero

python compare_agents.py --num-decks 1000 --seed 0 --agents greedy \
  alphazero=models/alphazero/alphazero_params.pkl --az-simulations 64
```
