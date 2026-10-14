# PPO

## Idea

Train a policy network with Proximal Policy Optimization, using thousands of games in parallel in the JAX environment. Two models are reported:

1. **PPO MLP**: the original baseline, a 2-layer MLP on the one-hot observation.
2. **PPO attention + greedy BC**: a small transformer over board cells and hand cards, a factored action head, symmetry augmentation, and a warm start by behavior cloning (BC) of the greedy agent. This model was then trained further (continuation runs), and the [afterstate value model](afterstate-td.md) was tried as the critic.

## Implementation

Files: `train_jax_ppo.py` (PPO, BC, evaluation, checkpoints) and `pileup_poker/networks.py` (networks, pure JAX).

### PPO core (`train_jax_ppo.py`)

- Rollouts: `--num-envs` games in parallel for `--num-steps` steps per update, on the JAX env (`--env-backend jax`, default) or the Python Gym env (`--env-backend gym`, slow, for the speed comparison in [jax-port.md](jax-port.md); needs `--no-reward-shaping`). Reward shaping, `--symmetry-aug` and `--afterstate-critic` need the JAX env.
- Masked categorical policy; illegal actions are never sampled.
- GAE (`--gamma`, `--gae-lambda`), clipped PPO objective (`--clip-epsilon`), value loss (`--value-coef`), entropy bonus (`--entropy-coef`), `--update-epochs` epochs of `--num-minibatches` minibatches, global gradient clipping (`--max-grad-norm`), Adam with `--lr-schedule constant|linear`.
- Rewards: `--reward-shaping` (default on) and `--reward-scale` (default 1000), see [environment](environment.md#reward-used-for-training). Use `--gamma 1.0` so the shaped return equals the final score.
- Evaluation every `--eval-every` updates: deterministic (argmax) play on `--eval-episodes` games (default 2000), logged as `eval/mean_final_reward`.
- Checkpoints in `--save-dir`: `jax_ppo_params.pkl` (`{"model_config", "params"}`) and `jax_ppo_checkpoint.pkl` (also optimizer state, update count, RNG key, run name), every `--save-every` updates and at the end.
  - `--resume <checkpoint>` continues a run. `--total-timesteps` is the total for the whole run. Games in progress are not saved.
  - `--init-params <params>` starts from saved weights with a fresh optimizer and update count.
  - Both rebuild the network from the file, so architecture flags are ignored with them.
- TensorBoard logs in `--tensorboard-logdir` (default `runs/jax_ppo`) / `--run-name`.

### Networks (`pileup_poker/networks.py`)

- `--arch mlp` (default): 2-layer tanh MLP (`--hidden-dim`, default 256) on the 796-dim observation, with policy and value heads. 291k parameters.
- `--arch attention`: a small transformer with 26 tokens: 16 board cells, 5 hand slots, 4 discard slots and 1 round token.
  - Card embedding = rank embedding + suit embedding; empty cells use an "empty" embedding.
  - Cells get position embeddings. Hand slots get slot embeddings plus an "already placed" flag.
  - `--num-layers` (3) pre-LayerNorm blocks with `--embed-dim` (64) and `--num-heads` (4). 165k parameters.
  - The policy logit of (slot, cell) is the scaled dot product of the hand-slot token and the cell token (80 logits). The value comes from mean-pooled tokens.
- `--action-head joint` (default): one masked softmax over 80 actions. `--action-head factored`: p(slot) x p(cell | slot), masked at both levels. Both give exact log-probabilities and entropy over the 80 actions, so PPO is unchanged.
- `--separate-critic`: the value network gets its own trunk (not used for the reported models).

### Symmetry augmentation (`--symmetry-aug`)

At every rollout step each game is shown to the agent under a fresh random board symmetry and suit relabeling (see [symmetries](symmetries-and-determinization.md)). The agent's action is mapped back to the real board with `ACTION_PERMS`. The stored observations, masks and actions are the transformed ones, so PPO stays consistent. (Keeping one symmetry for a whole game would add nothing, because games start from an empty board and decks are uniformly random.)

`evaluate_on_decks(..., num_symmetries=k)` can average the policy over k symmetric views at test time. Use it only for models trained with `--symmetry-aug`: an MLP model trained without it dropped from 2953 to 1854 with k=4.

### Behavior cloning from greedy (`--bc-games N --bc-epochs E`)

Before PPO starts, the policy is fit to the greedy agent:

- Greedy plans come from `jax_greedy` with each round's discard included in the score (`greedy_demonstrations`).
- Many round plans tie for the best score (round 1 has about 35 equally good first moves). So the target is the set of all actions that start a best greedy completion of the current round (`greedy_round_targets`), and the loss is -log P(target set).
- The value head is fit to the scaled, shaped greedy returns.
- `--bc-learning-rate` (1e-3), `--bc-batch-size` (256). `--bc-data file.npz` caches the greedy games (loaded if it exists, written otherwise; it must match the settings). BC is skipped with `--resume`.

### Afterstate critic (`--afterstate-critic PATH`)

Uses a trained afterstate value model as the PPO critic. Needs `--reward-shaping` (default) and `--gamma 1.0`.

- The critic value of a state is `V(s) = (max over legal a of V_after(afterstate(s, a)) - board_score(s)) / reward_scale` (`afterstate_state_values`), the expected reward-to-go. It is computed on the real state, not the symmetric view.
- `--afterstate-critic-mode frozen` (default): GAE uses this value instead of the value head. The value head is still trained on the returns as an auxiliary loss.
- `--afterstate-critic-mode residual`: value = afterstate critic + value head; the head learns the correction and its output layer starts at 0.
- `train/advantage_sq` logs 0.5 x mean squared GAE advantage before normalization, to compare critics.

## Training setup of the reported models

All runs: `--num-envs 1024 --num-steps 64` (65,536 steps per update), `--gamma 1.0`, `--lr-schedule linear`, reward shaping on, reward scale 1000, and PPO defaults: 4 epochs, 8 minibatches, GAE lambda 0.95, clip 0.2, entropy 0.01, value coef 0.5, grad clip 0.5. Times are on one RTX 3090 shared with other runs.

| run | model | start | LR | steps | time |
|---|---|---|---|---|---|
| `ppo_mlp_long` | MLP, joint head | scratch | 3e-4 | 300M | 1.1 h (~78k steps/s) |
| `ppo_attn_bc_long` | attention, factored head, symmetry aug | BC on 30k greedy games, 10 epochs | 3e-4 | 200M | 2.6 h (incl. 8 min to make the greedy games; ~27k steps/s) |
| `ppo_cont_control` | same | `ppo_attn_bc` weights | 1e-4 | 60M | ~1.3 h |
| `ppo_cont_ascritic` | same + frozen afterstate critic | `ppo_attn_bc` weights | 1e-4 | 60M | ~1.5 h |
| `ppo_cont_asresid` | same + residual afterstate critic | `ppo_attn_bc` weights | 1e-4 | 60M | ~1.5 h |
| `ppo_bc_control` | attention, factored head, symmetry aug | BC (cached games) | 3e-4 | 30M | ~25 min |
| `ppo_bc_ascritic` | same + frozen afterstate critic | BC (cached games) | 3e-4 | 30M | ~30 min |

The results below name the models by their directory in `models/`: `ppo_mlp_long` is `ppo_mlp` and `ppo_attn_bc_long` is `ppo_attn_bc`.

Evaluation curves (`eval/mean_final_reward`, 2000 games, argmax play):

| steps | PPO MLP | PPO attention + BC |
|---|---|---|
| 0 (after BC) | – | 1615 |
| 3.3M | 751 | 4054 |
| 26M | | 5678 |
| 39M | 2405 | |
| 52M | | 5914 |
| 75M | 3112 | |
| 102M | | 6071 |
| 151M | 4174 | 6232 |
| 200M | | 6266 |
| 298M | 4337 | |

BC reached 93% accuracy (argmax in the greedy target set), but the BC policy alone scored only 1615: with 16 moves per game, a few non-greedy moves per game are enough to break the greedy plans. PPO fixed this within the first few million steps.

## Results

1000 decks (`compare_agents.py --num-decks 1000 --seed 0`; all agents: [results](results.md)):

| model | mean ± SE |
|---|---|
| `ppo_cont_control` (+60M steps, normal critic) | **6294 ± 67** |
| `ppo_attn_bc` (200M steps) | 6248 ± 68 |
| `ppo_cont_asresid` (+60M, residual afterstate critic) | 6235 ± 68 |
| `ppo_cont_ascritic` (+60M, frozen afterstate critic) | 6182 ± 66 |
| `ppo_bc_ascritic` (30M from BC, frozen afterstate critic) | 5723 ± 66 |
| `ppo_bc_control` (30M from BC, normal critic) | 5559 ± 61 |
| `ppo_mlp` (300M steps) | 4270 ± 52 |

A PPO policy needs one network call per move: 1000 games take a few seconds on the GPU.

### Afterstate critic experiments

Pre-check: on 4096 games played by `ppo_attn_bc`, the afterstate critic had a larger error against the true reward-to-go than PPO's own value head (MSE 3.77 vs 2.86 over all 16 steps). It is biased for this policy: in round 1 it predicts 6.07 -> 5.71 while the true value is 6.18, because PPO's moves differ from the afterstate agent's moves. It is better only at the last move.

The 1000-deck differences are within noise, so the models were also compared on 20,000 decks (seed 1), paired against `ppo_attn_bc`:

| model | mean ± SE | paired diff vs `ppo_attn_bc` |
|---|---|---|
| `ppo_attn_bc` | 6226 ± 15 | 0 |
| `ppo_cont_control` | 6254 ± 15 | +28 ± 10 |
| `ppo_cont_asresid` | 6225 ± 15 | -1 ± 11 |
| `ppo_cont_ascritic` | 6130 ± 15 | -97 ± 13 |
| `ppo_bc_ascritic` | 5654 ± 14 | -573 ± 14 |
| `ppo_bc_control` | 5391 ± 13 | -835 ± 15 |

From-scratch training curves (eval, 2000 games):

| steps | frozen afterstate critic | normal critic |
|---|---|---|
| 6.6M | 4413 | 4450 |
| 13M | 5386 | 4857 |
| 22M | 5697 | 5257 |

Conclusions:

- **Continuing a strong policy:** the afterstate critic does not help. The frozen critic hurts (-97 ± 13): it estimates the value of the afterstate-greedy policy (about 6,130), not of the PPO policy (about 6,230), so its advantages push the policy toward afterstate-greedy moves; entropy rose from 0.41 to 0.48 during training. The residual critic does no harm and gives no gain. The normal critic gives a small real gain (+28 ± 10).
- **From BC:** the frozen afterstate critic speeds up learning a lot: about +500 at 6-20M steps and +262 at 30M steps (5654 vs 5391 on 20,000 decks). Caveat: the 200M-step `ppo_attn_bc_long` run already had eval ~5,710 at 30M steps, so the short 30M linear schedule limits both runs.
- The critic helps while the policy is weaker than the afterstate agent. Once the policy is stronger, the frozen critic probably holds it near the afterstate agent's level.

## What worked, what did not

- The plain MLP learns slowly (751 after 3M steps, 4,337 after 300M) and plateaus far below the afterstate MLP.
- The attention network with the factored (slot, cell) head, symmetry augmentation and greedy BC was the biggest gain: 4,054 after only 3.3M steps, 6,266 after 200M.
- More training of the best model gives small gains (+28 ± 10 on 20,000 decks for 60M more steps).
- PPO is the best agent without search.

## Reproduce

```bash
# PPO MLP baseline (1.1 h)
python train_jax_ppo.py --arch mlp --gamma 1.0 --num-envs 1024 --num-steps 64 \
  --total-timesteps 300000000 --lr-schedule linear --eval-every 50 \
  --run-name ppo_mlp_long --save-dir models/ppo_mlp

# PPO attention + factored head + symmetry aug + greedy BC (2.6 h)
python train_jax_ppo.py --arch attention --action-head factored --symmetry-aug --gamma 1.0 \
  --bc-games 30000 --bc-epochs 10 --bc-data models/bc_greedy_30k.npz \
  --num-envs 1024 --num-steps 64 --total-timesteps 200000000 --lr-schedule linear --eval-every 50 \
  --run-name ppo_attn_bc_long --save-dir models/ppo_attn_bc

# continuation (the best PPO model); add --afterstate-critic models/afterstate_mlp/afterstate_params.pkl
# [--afterstate-critic-mode residual] for the critic experiments
python train_jax_ppo.py --init-params models/ppo_attn_bc/jax_ppo_params.pkl \
  --arch attention --action-head factored --symmetry-aug --gamma 1.0 \
  --num-envs 1024 --num-steps 64 --learning-rate 1e-4 --total-timesteps 60000000 --lr-schedule linear \
  --run-name ppo_cont_control --save-dir models/ppo_cont_control

# from BC, 30M steps; add --afterstate-critic ... for ppo_bc_ascritic
python train_jax_ppo.py --arch attention --action-head factored --symmetry-aug --gamma 1.0 \
  --bc-games 30000 --bc-epochs 10 --bc-data models/bc_greedy_30k.npz \
  --num-envs 1024 --num-steps 64 --learning-rate 3e-4 --total-timesteps 30000000 --lr-schedule linear \
  --run-name ppo_bc_control --save-dir models/ppo_bc_control

tensorboard --logdir runs/jax_ppo

python compare_agents.py --num-decks 1000 --seed 0 --agents greedy \
  ppo=models/ppo_mlp/jax_ppo_params.pkl ppo=models/ppo_attn_bc/jax_ppo_params.pkl \
  ppo=models/ppo_cont_control/jax_ppo_params.pkl

# watch a policy play in the Gym env renderer
python play_jax_policy.py --model-path models/ppo_cont_control/jax_ppo_params.pkl --episodes 3 --seed 0
```

Evaluation from Python:

```python
from train_jax_ppo import evaluate_on_decks, load_policy
scores = evaluate_on_decks("models/ppo_cont_control/jax_ppo_params.pkl", decks)  # argmax play, decks (N, 36)
params, model_cfg = load_policy("models/ppo_cont_control/jax_ppo_params.pkl")
```
