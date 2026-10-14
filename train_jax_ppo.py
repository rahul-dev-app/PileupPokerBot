from __future__ import annotations

import argparse
import os
import pickle
import time
from functools import partial
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from pileup_poker import afterstate, jax_env, jax_greedy, networks, symmetry
from pileup_poker.networks import ModelConfig

try:
    from tensorboardX import SummaryWriter
except ImportError:  # pragma: no cover - optional dependency
    SummaryWriter = None



@dataclass
class PPOConfig:
    seed: int = 0
    total_timesteps: int = 2_000_000
    num_envs: int = 256
    num_steps: int = 64
    update_epochs: int = 4
    num_minibatches: int = 8
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden_dim: int = 256
    log_every: int = 10
    eval_every: int = 20
    eval_episodes: int = 2000
    save_dir: str = "models/jax_ppo"
    tensorboard_logdir: str = "runs/jax_ppo"
    run_name: str = ""
    env_backend: str = "jax"
    reward_scale: float = 1000.0
    reward_shaping: bool = True
    lr_schedule: str = "constant"
    save_every: int = 100
    resume: str = ""
    init_params: str = ""
    arch: str = "mlp"
    action_head: str = "joint"
    separate_critic: bool = False
    embed_dim: int = 64
    num_layers: int = 3
    num_heads: int = 4
    symmetry_aug: bool = False
    bc_games: int = 0
    bc_epochs: int = 10
    bc_batch_size: int = 256
    bc_learning_rate: float = 1e-3
    bc_data: str = ""
    afterstate_critic: str = ""
    afterstate_critic_mode: str = "frozen"

    def model_config(self) -> ModelConfig:
        return ModelConfig(
            arch=self.arch,
            action_head=self.action_head,
            hidden_dim=self.hidden_dim,
            separate_critic=self.separate_critic,
            embed_dim=self.embed_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
        )


@partial(jax.jit, static_argnums=1)
def policy_log_probs(params, model_cfg: ModelConfig, obs: jnp.ndarray, action_mask: jnp.ndarray):
    """Masked log-probabilities (B, 80) of every action, and values (B,)."""
    return networks.action_log_probs(params, model_cfg, obs, action_mask)


@partial(jax.jit, static_argnums=2)
def sample_actions(
    key: jax.Array,
    params,
    model_cfg: ModelConfig,
    obs: jnp.ndarray,
    action_mask: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    log_probs_all, values = networks.action_log_probs(params, model_cfg, obs, action_mask)
    actions = jax.random.categorical(key, log_probs_all, axis=-1)
    chosen_log_probs = jnp.take_along_axis(log_probs_all, actions[:, None], axis=1).squeeze(1)
    return actions, chosen_log_probs, values


# Symmetry augmentation: the agent sees a relabeled game (cells permuted, suits permuted) that
# scores exactly like the real one; its actions are mapped back with ACTION_PERMS.
_CELL_PERMS = jnp.asarray(symmetry.CELL_PERMS)
_SUIT_PERMS = jnp.asarray(symmetry.SUIT_PERMS)
_INV_SUIT_PERMS = jnp.asarray(symmetry.inverse_perms(symmetry.SUIT_PERMS))
_ACTION_PERMS = jnp.asarray(symmetry.ACTION_PERMS)
_INV_ACTION_PERMS = jnp.asarray(symmetry.inverse_perms(symmetry.ACTION_PERMS))


@jax.jit
def symmetric_view(state: jax_env.EnvState, cell_sym: jnp.ndarray, suit_sym: jnp.ndarray) -> jax_env.EnvState:
    """The game as seen under per-game cell symmetry `cell_sym` (B,) and suit symmetry `suit_sym` (B,)."""
    suit = _SUIT_PERMS[suit_sym]

    def relabel(cards):
        return jnp.where(cards >= 0, jnp.take_along_axis(suit, jnp.maximum(cards, 0), axis=1), -1)

    grid = jnp.take_along_axis(state.grid, _CELL_PERMS[cell_sym], axis=1)
    return state._replace(
        deck=relabel(state.deck),
        grid=relabel(grid),
        discard=relabel(state.discard),
        used=jnp.take_along_axis(state.used, _INV_SUIT_PERMS[suit_sym], axis=1),
    )


def env_actions(view_actions: jnp.ndarray, cell_sym: jnp.ndarray) -> jnp.ndarray:
    """Actions chosen in the symmetric view -> actions in the real game."""
    return _ACTION_PERMS[cell_sym, view_actions]


def random_symmetries(key: jax.Array, num_envs: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    cell_key, suit_key = jax.random.split(key)
    return (
        jax.random.randint(cell_key, (num_envs,), 0, symmetry.NUM_CELL_SYMMETRIES),
        jax.random.randint(suit_key, (num_envs,), 0, symmetry.NUM_SUIT_SYMMETRIES),
    )


def compute_gae(
    rewards: jnp.ndarray,
    values: jnp.ndarray,
    dones: jnp.ndarray,
    last_values: jnp.ndarray,
    gamma: float,
    gae_lambda: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    num_steps = rewards.shape[0]
    advantages = jnp.zeros_like(rewards)
    gae = jnp.zeros_like(last_values)

    for t in reversed(range(num_steps)):
        next_values = jnp.where(t == num_steps - 1, last_values, values[t + 1])
        nonterminal = 1.0 - dones[t].astype(jnp.float32)
        delta = rewards[t] + gamma * next_values * nonterminal - values[t]
        gae = delta + gamma * gae_lambda * nonterminal * gae
        advantages = advantages.at[t].set(gae)

    returns = advantages + values
    return advantages, returns


def make_update_step(
    model_cfg: ModelConfig,
    optimizer: optax.GradientTransformation,
    clip_epsilon: float,
    value_coef: float,
    entropy_coef: float,
):
    def loss_fn(params, obs, action_mask, actions, old_log_probs, advantages, returns):
        log_probs_all, values = networks.action_log_probs(params, model_cfg, obs, action_mask)
        new_log_probs = jnp.take_along_axis(log_probs_all, actions[:, None], axis=1).squeeze(1)

        log_ratio = new_log_probs - old_log_probs
        ratio = jnp.exp(log_ratio)

        unclipped = ratio * advantages
        clipped = jnp.clip(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
        policy_loss = -jnp.mean(jnp.minimum(unclipped, clipped))

        value_loss = 0.5 * jnp.mean((returns - values) ** 2)
        entropy = jnp.mean(networks.entropy(log_probs_all))

        total_loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
        approx_kl = 0.5 * jnp.mean(log_ratio**2)
        clip_frac = jnp.mean((jnp.abs(ratio - 1.0) > clip_epsilon).astype(jnp.float32))

        metrics = {
            "loss": total_loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "approx_kl": approx_kl,
            "clip_frac": clip_frac,
        }
        return total_loss, metrics

    @jax.jit
    def update_step(params, opt_state, obs, action_mask, actions, old_log_probs, advantages, returns):
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, obs, action_mask, actions, old_log_probs, advantages, returns
        )
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, metrics

    return update_step


@partial(jax.jit, static_argnames=("round_idx",))
def greedy_round_targets(state: jax_env.EnvState, round_idx: int):
    """Greedy plan for the round and the set of greedy-optimal actions at each of its 4 steps.

    All games must be at the start of round `round_idx`. The plan is played in hand-slot order.
    Many round plans tie for the best score, so the target at step k is every action that
    starts some best-scoring completion of the placements already made (not just the first
    maximum). Returns plan actions (B, 4) and targets (B, 4, 80) bool.
    """

    def per_game(args):
        grid, discard, hand = args
        scores, slots, cells = jax_greedy.candidate_scores(grid, discard, hand, round_idx, True)
        best = jnp.argmax(scores)
        order = jnp.argsort(slots[best])
        plan = (slots[best] * jax_env.GRID_CELLS + cells[best])[order]
        cand = slots * jax_env.GRID_CELLS + cells  # (N, 4) actions of each candidate
        is_best = scores == scores[best]
        targets = []
        for k in range(jax_env.PLAYS_PER_ROUND):
            placed = plan[:k]
            consistent = jnp.all(jnp.any(cand[:, :, None] == placed[None, None, :], axis=1), axis=1)
            remaining = jnp.all(cand[:, :, None] != placed[None, None, :], axis=2)  # (N, 4)
            good = (is_best & consistent)[:, None] & remaining
            targets.append(jnp.zeros(jax_env.ACTION_SIZE, dtype=jnp.bool_).at[cand.ravel()].max(good.ravel()))
        return plan, jnp.stack(targets)

    return jax.lax.map(per_game, (state.grid, state.discard, jax_greedy.hand_cards(state)), batch_size=8)


def greedy_demonstrations(
    key: jax.Array, num_games: int, gamma: float, reward_shaping: bool, reward_scale: float, symmetry_aug: bool
):
    """Greedy games for behavior cloning: obs, masks, target action sets (B*16, 80) and scaled discounted returns.

    The greedy plan scores each round with its discard included (jax_greedy include_discard=True).
    With symmetry_aug every sample gets a random symmetry.
    """
    deck_key, sym_key = jax.random.split(key)
    state, _, _ = jax_env.reset(deck_key, num_games)
    cell_sym = suit_sym = jnp.zeros((num_games,), dtype=jnp.int32)

    obs_buf, mask_buf, target_buf, reward_buf = [], [], [], []
    for round_idx in range(jax_env.ROUNDS):
        actions, targets = greedy_round_targets(state, round_idx)
        for k in range(jax_env.PLAYS_PER_ROUND):
            if symmetry_aug:
                sym_key, step_key = jax.random.split(sym_key)
                cell_sym, suit_sym = random_symmetries(step_key, num_games)
            view = symmetric_view(state, cell_sym, suit_sym)
            obs_buf.append(jax_env.build_observation(view))
            mask_buf.append(jax_env.get_action_mask(view))
            target_buf.append(jnp.take_along_axis(targets[:, k], _ACTION_PERMS[cell_sym], axis=1))
            next_state, rewards, dones = jax_env.step(state, actions[:, k])
            if reward_shaping:
                rewards = shaped_rewards(state, next_state, rewards, dones, gamma)
            reward_buf.append(rewards / reward_scale)
            state = next_state

    returns = []
    ret = jnp.zeros((num_games,), dtype=jnp.float32)
    for rewards in reversed(reward_buf):
        ret = rewards + gamma * ret
        returns.append(ret)
    returns.reverse()

    def flat(xs):
        return jnp.concatenate(xs, axis=0)

    return flat(obs_buf), flat(mask_buf), flat(target_buf), flat(returns)


def behavior_cloning(params, model_cfg: ModelConfig, config: PPOConfig, key: jax.Array, writer):
    """Fit the policy to greedy actions and the value to greedy returns.

    Policy loss: -log of the total probability of the greedy-optimal action set.
    Accuracy: how often the argmax action is in that set.
    """
    key, data_key = jax.random.split(key)
    # Greedy games are slow to make (~0.1 s each on CPU), so --bc-data caches them in a file.
    settings = {
        "bc_games": config.bc_games,
        "seed": config.seed,
        "gamma": config.gamma,
        "reward_shaping": config.reward_shaping,
        "reward_scale": config.reward_scale,
        "symmetry_aug": config.symmetry_aug,
    }
    if config.bc_data and os.path.exists(config.bc_data):
        with np.load(config.bc_data) as data:
            saved = {k: data[k].item() for k in settings}
            if saved != settings:
                raise ValueError(f"{config.bc_data} was made with {saved}, not {settings}")
            obs = jnp.asarray(data["obs"], dtype=jnp.float32)
            masks, targets, returns = (jnp.asarray(data[k]) for k in ("masks", "targets", "returns"))
        print(f"[bc] loaded {config.bc_games} greedy games from {config.bc_data}")
    else:
        start = time.time()
        obs, masks, targets, returns = greedy_demonstrations(
            data_key,
            config.bc_games,
            config.gamma,
            config.reward_shaping,
            config.reward_scale,
            config.symmetry_aug,
        )
        print(f"[bc] {config.bc_games} greedy games in {time.time() - start:.1f}s")
        if config.bc_data:
            tmp_path = config.bc_data + ".tmp.npz"
            np.savez(tmp_path, obs=np.asarray(obs, dtype=np.uint8), masks=masks, targets=targets, returns=returns, **settings)
            os.replace(tmp_path, config.bc_data)
            print(f"[bc] saved greedy games to {config.bc_data}")
    num_samples = obs.shape[0]

    optimizer = optax.chain(optax.clip_by_global_norm(config.max_grad_norm), optax.adam(config.bc_learning_rate))
    opt_state = optimizer.init(params)

    def loss_fn(params, obs, masks, targets, returns):
        log_probs_all, values = networks.action_log_probs(params, model_cfg, obs, masks)
        policy_loss = -jnp.mean(jax.nn.logsumexp(jnp.where(targets, log_probs_all, networks.NEG_INF), axis=-1))
        value_loss = 0.5 * jnp.mean((returns - values) ** 2)
        best = jnp.argmax(log_probs_all, axis=-1)
        accuracy = jnp.mean(jnp.take_along_axis(targets, best[:, None], axis=1))
        return policy_loss + config.value_coef * value_loss, (policy_loss, value_loss, accuracy)

    @jax.jit
    def bc_step(params, opt_state, obs, masks, targets, returns):
        grads, metrics = jax.grad(loss_fn, has_aux=True)(params, obs, masks, targets, returns)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, metrics

    batch = min(config.bc_batch_size, num_samples)
    for epoch in range(1, config.bc_epochs + 1):
        key, perm_key = jax.random.split(key)
        permutation = jax.random.permutation(perm_key, num_samples)
        metrics = []
        for start in range(0, num_samples - batch + 1, batch):
            idx = permutation[start : start + batch]
            params, opt_state, m = bc_step(params, opt_state, obs[idx], masks[idx], targets[idx], returns[idx])
            metrics.append(m)
        policy_loss, value_loss, accuracy = (float(jnp.mean(jnp.stack(x))) for x in zip(*metrics))
        print(f"[bc epoch {epoch}/{config.bc_epochs}] policy={policy_loss:.4f} value={value_loss:.4f} acc={accuracy:.3f}")
        if writer is not None:
            writer.add_scalar("bc/policy_loss", policy_loss, epoch)
            writer.add_scalar("bc/value_loss", value_loss, epoch)
            writer.add_scalar("bc/accuracy", accuracy, epoch)

    eval_score = evaluate_policy(params, model_cfg, config.seed, config.eval_episodes, config.env_backend)
    print(f"[bc eval] mean_final_reward={eval_score:.2f}")
    if writer is not None:
        writer.add_scalar("eval/mean_final_reward", eval_score, 0)
    return params, key


def _flatten_gym_obs(obs: dict[str, np.ndarray]) -> np.ndarray:
    # Imported lazily: gymnasium is only needed for the gym backend.
    from pileup_poker.pileup_poker import flatten_observation

    return flatten_observation(obs)


def _init_gym_backend(num_envs: int, seed: int):
    try:
        from pileup_poker.pileup_poker import PileupPokerEnv
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Gym backend selected, but Gym dependencies are missing. "
            "Install requirements (including gymnasium) and retry."
        ) from exc

    envs: list[Any] = []
    obs = np.zeros((num_envs, jax_env.OBS_SIZE), dtype=np.float32)
    masks = np.zeros((num_envs, jax_env.ACTION_SIZE), dtype=bool)
    for env_idx in range(num_envs):
        env = PileupPokerEnv(enforce_valid_actions=True)
        env_obs, _ = env.reset(seed=seed + env_idx)
        envs.append(env)
        obs[env_idx] = _flatten_gym_obs(env_obs)
        masks[env_idx] = np.asarray(env.get_action_mask(), dtype=bool)
    return envs, obs, masks


def _step_gym_backend(envs: list[Any], actions: np.ndarray):
    num_envs = len(envs)
    next_obs = np.zeros((num_envs, jax_env.OBS_SIZE), dtype=np.float32)
    next_masks = np.zeros((num_envs, jax_env.ACTION_SIZE), dtype=bool)
    rewards = np.zeros((num_envs,), dtype=np.float32)
    dones = np.zeros((num_envs,), dtype=bool)

    for env_idx, env in enumerate(envs):
        obs, reward, done, _, _ = env.step(int(actions[env_idx]))
        rewards[env_idx] = float(reward)
        dones[env_idx] = bool(done)
        if done:
            obs, _ = env.reset()
        next_obs[env_idx] = _flatten_gym_obs(obs)
        next_masks[env_idx] = np.asarray(env.get_action_mask(), dtype=bool)

    return next_obs, next_masks, rewards, dones


@partial(jax.jit, static_argnames=("model",))
def afterstate_state_values(model: str, as_params, state: jax_env.EnvState, value_scale: float, reward_scale: float):
    """Critic from a trained afterstate value model, in PPO value units (B,).

    V(s) = (max over legal a of V_after(afterstate(s, a)) - board_score(s)) / reward_scale: the expected
    reward-to-go with --reward-shaping and gamma=1. Uses the real (not symmetric-view) state; the
    afterstates never read the order of undealt cards.
    """
    num_envs = state.grid.shape[0]
    afterstates, mask = afterstate.expand_afterstates(state)
    flat = jax.tree_util.tree_map(lambda x: x.reshape((num_envs * jax_env.ACTION_SIZE,) + x.shape[2:]), afterstates)
    v = afterstate.value(model, as_params, flat, value_scale).reshape(num_envs, jax_env.ACTION_SIZE)
    best = jnp.max(jnp.where(mask, v, -jnp.inf), axis=1) * value_scale
    return (best - jax_env.calculate_total_score(state.grid, state.discard)) / reward_scale


@jax.jit
def shaped_rewards(state, next_state, rewards, dones, gamma: float) -> jnp.ndarray:
    # Potential-based shaping with the current board score as potential: r + gamma * phi(s') - phi(s).
    # phi is 0 at terminal states, so with gamma=1 an episode's shaped return equals its final score.
    potential = jax_env.calculate_total_score(state.grid, state.discard)
    next_potential = jax_env.calculate_total_score(next_state.grid, next_state.discard)
    next_potential = jnp.where(dones, 0.0, next_potential)
    return rewards + gamma * next_potential - potential


def play_policy_greedy_actions(
    params, state: jax_env.EnvState, model_cfg: ModelConfig = ModelConfig(), num_symmetries: int = 1, seed: int = 0
) -> jnp.ndarray:
    """Play every game in `state` to the end with argmax actions; returns final scores.

    With num_symmetries > 1 the policy's log-probabilities are averaged over that many
    symmetric views of each position (the identity plus random ones drawn from `seed`).
    Only useful for policies trained with --symmetry-aug: a policy trained on its own games
    only knows its own board orientation, and other views make it much worse.
    """
    num_envs = state.grid.shape[0]
    cell_syms, suit_syms = random_symmetries(jax.random.PRNGKey(seed), num_symmetries)
    cell_syms, suit_syms = cell_syms.at[0].set(0), suit_syms.at[0].set(0)
    returns = jnp.zeros((num_envs,), dtype=jnp.float32)
    # Every game lasts exactly ROUNDS * PLAYS_PER_ROUND steps.
    for _ in range(jax_env.ROUNDS * jax_env.PLAYS_PER_ROUND):
        if num_symmetries == 1:
            log_probs, _ = policy_log_probs(params, model_cfg, jax_env.build_observation(state), jax_env.get_action_mask(state))
        else:
            log_probs = 0.0
            for k in range(num_symmetries):
                cell_sym = jnp.full((num_envs,), cell_syms[k])
                view = symmetric_view(state, cell_sym, jnp.full((num_envs,), suit_syms[k]))
                view_lp, _ = policy_log_probs(params, model_cfg, jax_env.build_observation(view), jax_env.get_action_mask(view))
                log_probs = log_probs + view_lp[:, _INV_ACTION_PERMS[cell_syms[k]]]
        action = jnp.argmax(log_probs, axis=-1)
        state, reward, _ = jax_env.step(state, action)
        returns = returns + reward
    return returns


def _evaluate_policy_jax(params, model_cfg: ModelConfig, seed: int, episodes: int) -> float:
    state, _, _ = jax_env.reset(jax.random.PRNGKey(seed), episodes)
    return float(jnp.mean(play_policy_greedy_actions(params, state, model_cfg)))


def _evaluate_policy_gym(params, model_cfg: ModelConfig, seed: int, episodes: int) -> float:
    try:
        from pileup_poker.pileup_poker import PileupPokerEnv
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Gym backend evaluation requires gymnasium/PileupPokerEnv dependencies."
        ) from exc

    env = PileupPokerEnv(enforce_valid_actions=True)
    returns = []
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        done = False
        episode_return = 0.0

        while not done:
            obs_batch = jnp.asarray(_flatten_gym_obs(obs))[None, :]
            mask_batch = jnp.asarray(np.asarray(env.get_action_mask(), dtype=np.bool_))[None, :]
            log_probs, _ = policy_log_probs(params, model_cfg, obs_batch, mask_batch)
            action = int(jnp.argmax(log_probs, axis=-1)[0])
            obs, reward, done, _, _ = env.step(action)
            episode_return += float(reward)

        returns.append(episode_return)

    env.close()
    return float(np.mean(returns))


def evaluate_policy(params, model_cfg: ModelConfig, seed: int, episodes: int, env_backend: str) -> float:
    if env_backend == "jax":
        return _evaluate_policy_jax(params, model_cfg, seed, episodes)
    return _evaluate_policy_gym(params, model_cfg, seed, episodes)


PARAMS_FILE = "jax_ppo_params.pkl"
CHECKPOINT_FILE = "jax_ppo_checkpoint.pkl"


def _pickle_atomic(obj, path: str) -> None:
    # Write then rename, so a run killed mid-save never leaves a broken file.
    tmp_path = path + ".tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(jax.device_get(obj), f)
    os.replace(tmp_path, path)


def _zero_value_output(params):
    head = params["critic"] if "critic" in params else params
    head = {**head, "w_v": jnp.zeros_like(head["w_v"]), "b_v": jnp.zeros_like(head["b_v"])}
    return {**params, "critic": head} if "critic" in params else head


def save_checkpoint(
    save_dir: str, params, opt_state, update: int, key: jax.Array, run_name: str, model_cfg: ModelConfig
) -> None:
    model_config = asdict(model_cfg)
    _pickle_atomic({"model_config": model_config, "params": params}, os.path.join(save_dir, PARAMS_FILE))
    checkpoint = {
        "params": params,
        "opt_state": opt_state,
        "update": update,
        "key": key,
        "run_name": run_name,
        "model_config": model_config,
    }
    _pickle_atomic(checkpoint, os.path.join(save_dir, CHECKPOINT_FILE))


def load_policy(path: str) -> tuple[Any, ModelConfig]:
    """Params and model config from a jax_ppo_params.pkl (or a checkpoint)."""
    with open(path, "rb") as f:
        raw = pickle.load(f)
    return jax.tree_util.tree_map(jnp.asarray, raw["params"]), ModelConfig(**raw["model_config"])


def evaluate_on_decks(model_path: str, decks: np.ndarray, seed: int = 0, num_symmetries: int = 1) -> np.ndarray:
    """Final scores (N,) of the argmax policy on (N, 36) decks, played exactly as dealt.

    num_symmetries > 1 averages the policy over symmetric views (see play_policy_greedy_actions).
    """
    params, model_cfg = load_policy(model_path)
    state = jax_env.init_env_state_from_decks(jnp.asarray(decks))
    return np.asarray(play_policy_greedy_actions(params, state, model_cfg, num_symmetries, seed))


def train(config: PPOConfig):
    if config.num_envs <= 0:
        raise ValueError("num_envs must be positive")
    if config.num_steps <= 0:
        raise ValueError("num_steps must be positive")
    if config.num_minibatches <= 0:
        raise ValueError("num_minibatches must be positive")
    if (config.num_envs * config.num_steps) % config.num_minibatches != 0:
        raise ValueError("num_envs * num_steps must be divisible by num_minibatches")
    if config.reward_shaping and config.env_backend != "jax":
        raise ValueError("--reward-shaping requires --env-backend jax; pass --no-reward-shaping for gym")
    if config.resume and config.init_params:
        raise ValueError("use either --resume or --init-params, not both")
    if config.symmetry_aug and config.env_backend != "jax":
        raise ValueError("--symmetry-aug requires --env-backend jax")
    if config.afterstate_critic and not (config.reward_shaping and config.gamma == 1.0 and config.env_backend == "jax"):
        raise ValueError("--afterstate-critic requires --reward-shaping, --gamma 1.0 and --env-backend jax")

    as_critic = None
    if config.afterstate_critic:
        payload = afterstate.load_params(config.afterstate_critic)
        as_critic = partial(
            afterstate_state_values, payload["model"], payload["params"], value_scale=payload["value_scale"],
            reward_scale=config.reward_scale,
        )
        print(f"Afterstate critic ({config.afterstate_critic_mode}): {config.afterstate_critic}")

    checkpoint = None
    if config.resume:
        with open(config.resume, "rb") as f:
            checkpoint = pickle.load(f)
        print(f"Resuming from {config.resume} after update {checkpoint['update']}")
        model_cfg = ModelConfig(**checkpoint["model_config"])
    elif config.init_params:
        loaded_params, model_cfg = load_policy(config.init_params)
    else:
        model_cfg = config.model_config()
    print(f"Model: {model_cfg}")

    os.makedirs(config.save_dir, exist_ok=True)
    devices = jax.devices()
    print(f"JAX backend: {jax.default_backend()} | devices: {[str(d) for d in devices]}")
    print(f"Rollout environment backend: {config.env_backend}")

    if config.run_name:
        run_name = config.run_name
    elif checkpoint is not None:
        run_name = checkpoint["run_name"]
    else:
        run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = None
    if SummaryWriter is None:
        print("TensorBoard logging disabled: install tensorboardX to enable it.")
    elif config.tensorboard_logdir:
        tb_dir = os.path.join(config.tensorboard_logdir, run_name)
        writer = SummaryWriter(logdir=tb_dir)
        print(f"TensorBoard run dir: {tb_dir}")

    key = jax.random.PRNGKey(config.seed)
    key, reset_key, net_key = jax.random.split(key, 3)

    state = None
    gym_envs = None
    if config.env_backend == "jax":
        state, obs, action_mask = jax_env.reset(reset_key, config.num_envs)
        if config.symmetry_aug:
            # Every step each game is shown under a fresh random symmetry. (A symmetry fixed for a
            # whole episode would change nothing: games start empty and decks are uniformly random.)
            key, sym_key = jax.random.split(key)
            cell_sym, suit_sym = random_symmetries(sym_key, config.num_envs)
            view = symmetric_view(state, cell_sym, suit_sym)
            obs, action_mask = jax_env.build_observation(view), jax_env.get_action_mask(view)
    else:
        gym_envs, obs_np, mask_np = _init_gym_backend(config.num_envs, config.seed)
        obs = jnp.asarray(obs_np)
        action_mask = jnp.asarray(mask_np)

    batch_size = config.num_envs * config.num_steps
    minibatch_size = batch_size // config.num_minibatches
    num_updates = config.total_timesteps // batch_size
    if num_updates < 1:
        raise ValueError("total_timesteps is too small for one PPO update")

    # One optimizer step per minibatch. Both schedules keep the same optimizer state layout,
    # so a checkpoint can be resumed with either one.
    steps_per_update = config.update_epochs * config.num_minibatches
    if config.lr_schedule == "linear":
        lr_schedule = optax.linear_schedule(config.learning_rate, 0.0, num_updates * steps_per_update)
    else:
        lr_schedule = optax.constant_schedule(config.learning_rate)

    optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(lr_schedule),
    )

    start_update = 1
    if checkpoint is not None:
        params = jax.tree_util.tree_map(jnp.asarray, checkpoint["params"])
        opt_state = jax.tree_util.tree_map(jnp.asarray, checkpoint["opt_state"])
        key = jnp.asarray(checkpoint["key"])
        start_update = checkpoint["update"] + 1
        if start_update > num_updates:
            raise ValueError(
                f"checkpoint is already at update {checkpoint['update']} of {num_updates}; raise --total-timesteps"
            )
    else:
        if config.init_params:
            params = loaded_params
            print(f"Initialized parameters from {config.init_params}")
        else:
            params = networks.init_params(net_key, model_cfg)
        if config.bc_games > 0:
            # Own key, so the PPO random stream is the same with or without behavior cloning.
            params, _ = behavior_cloning(
                params, model_cfg, config, jax.random.fold_in(jax.random.PRNGKey(config.seed), 1), writer
            )
        if as_critic is not None and config.afterstate_critic_mode == "residual":
            # The value head now predicts a correction to the afterstate critic, so it starts at 0.
            params = _zero_value_output(params)
        opt_state = optimizer.init(params)

    update_step = make_update_step(
        model_cfg,
        optimizer=optimizer,
        clip_epsilon=config.clip_epsilon,
        value_coef=config.value_coef,
        entropy_coef=config.entropy_coef,
    )

    running_returns = np.zeros((config.num_envs,), dtype=np.float32)
    finished_returns: list[float] = []  # raw final scores of episodes finished since the last log

    start_time = time.time()
    for update in range(start_update, num_updates + 1):
        obs_buf = []
        mask_buf = []
        action_buf = []
        logprob_buf = []
        reward_buf = []
        done_buf = []
        value_buf = []
        baseline_buf = []

        for _ in range(config.num_steps):
            key, action_key = jax.random.split(key)
            actions, log_probs, values = sample_actions(action_key, params, model_cfg, obs, action_mask)
            if as_critic is not None:
                # frozen: the afterstate critic replaces the value head; residual: the head adds a correction.
                baseline = as_critic(state)
                baseline_buf.append(baseline)
                values = baseline if config.afterstate_critic_mode == "frozen" else values + baseline

            obs_buf.append(obs)
            mask_buf.append(action_mask)
            action_buf.append(actions)
            logprob_buf.append(log_probs)
            value_buf.append(values)

            if config.env_backend == "jax":
                step_actions = env_actions(actions, cell_sym) if config.symmetry_aug else actions
                next_state, rewards, dones = jax_env.step(state, step_actions)
                running_returns += np.array(rewards)
                if config.reward_shaping:
                    train_rewards = shaped_rewards(state, next_state, rewards, dones, config.gamma)
                else:
                    train_rewards = rewards
                done_np = np.array(dones)
                if np.any(done_np):
                    finished_returns.extend(running_returns[done_np].tolist())
                    running_returns[done_np] = 0.0

                key, reset_done_key = jax.random.split(key)
                state = jax_env.reset_done(next_state, dones, reset_done_key)
                view = state
                if config.symmetry_aug:
                    key, sym_key = jax.random.split(key)
                    cell_sym, suit_sym = random_symmetries(sym_key, config.num_envs)
                    view = symmetric_view(state, cell_sym, suit_sym)
                obs = jax_env.build_observation(view)
                action_mask = jax_env.get_action_mask(view)
            else:
                next_obs_np, next_mask_np, rewards_np, dones_np = _step_gym_backend(
                    gym_envs, np.asarray(actions)
                )
                running_returns += rewards_np
                if np.any(dones_np):
                    finished_returns.extend(running_returns[dones_np].tolist())
                    running_returns[dones_np] = 0.0

                train_rewards = jnp.asarray(rewards_np)
                dones = jnp.asarray(dones_np)
                obs = jnp.asarray(next_obs_np)
                action_mask = jnp.asarray(next_mask_np)

            reward_buf.append(train_rewards / config.reward_scale)
            done_buf.append(dones)

        obs_arr = jnp.stack(obs_buf)
        mask_arr = jnp.stack(mask_buf)
        actions_arr = jnp.stack(action_buf)
        old_log_probs_arr = jnp.stack(logprob_buf)
        rewards_arr = jnp.stack(reward_buf)
        dones_arr = jnp.stack(done_buf)
        values_arr = jnp.stack(value_buf)

        _, last_values = policy_log_probs(params, model_cfg, obs, action_mask)
        if as_critic is not None:
            last_baseline = as_critic(state)
            last_values = last_baseline if config.afterstate_critic_mode == "frozen" else last_values + last_baseline
        advantages, returns = compute_gae(
            rewards=rewards_arr,
            values=values_arr,
            dones=dones_arr,
            last_values=last_values,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
        )

        flat_obs = obs_arr.reshape((batch_size, -1))
        flat_mask = mask_arr.reshape((batch_size, jax_env.ACTION_SIZE))
        flat_actions = actions_arr.reshape((batch_size,))
        flat_old_log_probs = old_log_probs_arr.reshape((batch_size,))
        flat_advantages = advantages.reshape((batch_size,))
        flat_returns = returns.reshape((batch_size,))
        if as_critic is not None and config.afterstate_critic_mode == "residual":
            flat_returns = flat_returns - jnp.stack(baseline_buf).reshape((batch_size,))

        # 0.5 * mean squared GAE advantage (before normalization), comparable across critics.
        raw_advantage_sq = float(0.5 * jnp.mean(flat_advantages**2))
        flat_advantages = (flat_advantages - jnp.mean(flat_advantages)) / (jnp.std(flat_advantages) + 1e-8)

        update_metrics = []
        for _ in range(config.update_epochs):
            key, perm_key = jax.random.split(key)
            permutation = jax.random.permutation(perm_key, batch_size)

            for start in range(0, batch_size, minibatch_size):
                idx = permutation[start : start + minibatch_size]
                mb_obs = flat_obs[idx]
                mb_mask = flat_mask[idx]
                mb_actions = flat_actions[idx]
                mb_old_log_probs = flat_old_log_probs[idx]
                mb_advantages = flat_advantages[idx]
                mb_returns = flat_returns[idx]

                params, opt_state, _, metrics = update_step(
                    params,
                    opt_state,
                    mb_obs,
                    mb_mask,
                    mb_actions,
                    mb_old_log_probs,
                    mb_advantages,
                    mb_returns,
                )
                update_metrics.append(metrics)

        if update % config.log_every == 0:
            # Averaged over every minibatch of this update.
            last_metrics = {k: float(jnp.mean(jnp.stack([m[k] for m in update_metrics]))) for k in update_metrics[0]}
            elapsed = time.time() - start_time
            timesteps_done = update * batch_size
            sps = int((update - start_update + 1) * batch_size / max(elapsed, 1e-6))
            current_lr = float(lr_schedule(update * steps_per_update))
            avg_return = float(np.mean(finished_returns)) if finished_returns else 0.0
            finished_returns.clear()
            print(
                f"[update {update}/{num_updates}] "
                f"timesteps={timesteps_done} sps={sps} "
                f"loss={float(last_metrics['loss']):.4f} "
                f"policy={float(last_metrics['policy_loss']):.4f} "
                f"value={float(last_metrics['value_loss']):.4f} "
                f"entropy={float(last_metrics['entropy']):.4f} "
                f"adv_sq={raw_advantage_sq:.4f} "
                f"lr={current_lr:.2e} "
                f"avg_return={avg_return:.2f}"
            )
            if writer is not None:
                writer.add_scalar("train/loss", float(last_metrics["loss"]), timesteps_done)
                writer.add_scalar("train/policy_loss", float(last_metrics["policy_loss"]), timesteps_done)
                writer.add_scalar("train/value_loss", float(last_metrics["value_loss"]), timesteps_done)
                writer.add_scalar("train/entropy", float(last_metrics["entropy"]), timesteps_done)
                writer.add_scalar("train/approx_kl", float(last_metrics["approx_kl"]), timesteps_done)
                writer.add_scalar("train/clip_frac", float(last_metrics["clip_frac"]), timesteps_done)
                writer.add_scalar("train/advantage_sq", raw_advantage_sq, timesteps_done)
                writer.add_scalar("train/sps", sps, timesteps_done)
                writer.add_scalar("train/learning_rate", current_lr, timesteps_done)
                writer.add_scalar("train/avg_return", avg_return, timesteps_done)

        if update % config.eval_every == 0:
            eval_score = evaluate_policy(
                params,
                model_cfg,
                seed=config.seed + update,
                episodes=config.eval_episodes,
                env_backend=config.env_backend,
            )
            print(f"[eval {update}] mean_final_reward={eval_score:.2f}")
            if writer is not None:
                writer.add_scalar("eval/mean_final_reward", eval_score, update * batch_size)

        if update % config.save_every == 0 or update == num_updates:
            save_checkpoint(config.save_dir, params, opt_state, update, key, run_name, model_cfg)

    print(f"Saved {PARAMS_FILE} and {CHECKPOINT_FILE} to {config.save_dir}")
    if gym_envs is not None:
        for env in gym_envs:
            env.close()
    if writer is not None:
        writer.close()


def parse_args() -> PPOConfig:
    parser = argparse.ArgumentParser(description="Train a custom JAX PPO agent for Pileup Poker")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-timesteps", type=int, default=2_000_000)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--num-steps", type=int, default=64)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--num-minibatches", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount. With --reward-shaping and gamma=1 the shaped return of an episode equals its final score.",
    )
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--eval-episodes", type=int, default=2000)
    parser.add_argument("--save-dir", type=str, default="models/jax_ppo")
    parser.add_argument("--tensorboard-logdir", type=str, default="runs/jax_ppo")
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument(
        "--env-backend",
        type=str,
        default="jax",
        choices=["jax", "gym"],
        help="Use 'jax' for vectorized JAX env rollouts or 'gym' for Python/Gym env rollouts.",
    )
    parser.add_argument(
        "--reward-scale", type=float, default=1000.0, help="Training rewards are divided by this value."
    )
    parser.add_argument(
        "--lr-schedule",
        type=str,
        default="constant",
        choices=["constant", "linear"],
        help="'linear' lowers the learning rate to 0 by the end of --total-timesteps.",
    )
    parser.add_argument("--save-every", type=int, default=100, help="Save a checkpoint every N updates.")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Continue from a jax_ppo_checkpoint.pkl (params, optimizer state, update count, RNG key).",
    )
    parser.add_argument(
        "--init-params",
        type=str,
        default="",
        help="Start from a jax_ppo_params.pkl with a fresh optimizer and update count.",
    )
    parser.add_argument(
        "--reward-shaping",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reward the change in board score after every move (jax backend only).",
    )

    parser.add_argument("--arch", type=str, default="mlp", choices=["mlp", "attention"])
    parser.add_argument(
        "--action-head",
        type=str,
        default="joint",
        choices=["joint", "factored"],
        help="'joint': one softmax over 80 actions. 'factored': p(slot) * p(cell | slot).",
    )
    parser.add_argument("--separate-critic", action="store_true", help="Value network with its own trunk.")
    parser.add_argument("--embed-dim", type=int, default=64, help="Attention token width.")
    parser.add_argument("--num-layers", type=int, default=3, help="Attention transformer blocks.")
    parser.add_argument("--num-heads", type=int, default=4, help="Attention heads per block.")
    parser.add_argument(
        "--symmetry-aug",
        action="store_true",
        help="Show every rollout step (and BC sample) under a random cell and suit symmetry (jax backend only).",
    )
    parser.add_argument(
        "--bc-games", type=int, default=0, help="Behavior-clone the greedy agent on N games before PPO (0 = off)."
    )
    parser.add_argument("--bc-epochs", type=int, default=10)
    parser.add_argument("--bc-batch-size", type=int, default=256)
    parser.add_argument("--bc-learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--bc-data",
        type=str,
        default="",
        help="Cache file (.npz) for the greedy games: loaded if it exists, otherwise written.",
    )
    parser.add_argument(
        "--afterstate-critic",
        type=str,
        default="",
        help="afterstate_params.pkl used as the critic (needs --reward-shaping and --gamma 1.0).",
    )
    parser.add_argument(
        "--afterstate-critic-mode",
        type=str,
        default="frozen",
        choices=["frozen", "residual"],
        help="'frozen': the afterstate critic replaces the value head. 'residual': value = afterstate critic + value head.",
    )

    args = parser.parse_args()
    return PPOConfig(
        seed=args.seed,
        total_timesteps=args.total_timesteps,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        update_epochs=args.update_epochs,
        num_minibatches=args.num_minibatches,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        max_grad_norm=args.max_grad_norm,
        hidden_dim=args.hidden_dim,
        log_every=args.log_every,
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
        save_dir=args.save_dir,
        tensorboard_logdir=args.tensorboard_logdir,
        run_name=args.run_name,
        env_backend=args.env_backend,
        reward_scale=args.reward_scale,
        reward_shaping=args.reward_shaping,
        lr_schedule=args.lr_schedule,
        save_every=args.save_every,
        resume=args.resume,
        init_params=args.init_params,
        arch=args.arch,
        action_head=args.action_head,
        separate_critic=args.separate_critic,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        symmetry_aug=args.symmetry_aug,
        bc_games=args.bc_games,
        bc_epochs=args.bc_epochs,
        bc_batch_size=args.bc_batch_size,
        bc_learning_rate=args.bc_learning_rate,
        bc_data=args.bc_data,
        afterstate_critic=args.afterstate_critic,
        afterstate_critic_mode=args.afterstate_critic_mode,
    )


if __name__ == "__main__":
    train(parse_args())
