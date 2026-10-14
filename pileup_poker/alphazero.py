"""AlphaZero-style agent: Gumbel MuZero search (DeepMind mctx) with the real env as the model.

Chance is handled without peeking at the real deck order:
- The search tree only extends to the end of the current round. Inside a round the
  game is deterministic given the known hand, so the tree steps copies of the env state.
- A move that completes a round is a leaf (discount 0). Its reward is the change in board
  score plus the network value of the afterstate: board + discard + next round, with an
  empty hand. A move that ends the game gets only the score change.
- The search runs on `jax_env.determinize` copies, so even nodes expanded past a leaf
  (whose values never reach the root) never see the real undealt cards.

Values are reward-to-go with the board score as potential: V(s) = (final - score(s)) / VALUE_SCALE.
"""
from __future__ import annotations

import pickle
from functools import partial

import jax
import jax.numpy as jnp
import mctx
import numpy as np

from pileup_poker import jax_env as J
from pileup_poker.symmetry import ACTION_PERMS, CELL_PERMS, SUIT_PERMS

VALUE_SCALE = 1000.0
NEG_INF = -1e9
HAND_OBS = slice(J.GRID_CELLS * J.DECK_SIZE, (J.GRID_CELLS + J.HAND_SIZE) * J.DECK_SIZE)


# ---------------------------------------------------------------- network

def init_params(key: jax.Array, hidden_dim: int = 512, num_layers: int = 3):
    dims = [J.OBS_SIZE] + [hidden_dim] * num_layers
    keys = jax.random.split(key, num_layers + 2)

    def linear(k, fan_in, fan_out, scale):
        w = jax.random.normal(k, (fan_in, fan_out), jnp.float32) * (scale / np.sqrt(fan_in))
        return {"w": w, "b": jnp.zeros((fan_out,), jnp.float32)}

    return {
        "hidden": [linear(keys[i], dims[i], dims[i + 1], np.sqrt(2.0)) for i in range(num_layers)],
        "policy": linear(keys[-2], hidden_dim, J.ACTION_SIZE, 0.01),
        "value": linear(keys[-1], hidden_dim, 1, 0.01),
    }


def policy_value(params, obs: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    h = obs
    for layer in params["hidden"]:
        h = jax.nn.relu(h @ layer["w"] + layer["b"])
    logits = h @ params["policy"]["w"] + params["policy"]["b"]
    value = (h @ params["value"]["w"] + params["value"]["b"]).squeeze(-1)
    return logits, value


def afterstate_observation(obs: jnp.ndarray) -> jnp.ndarray:
    return obs.at[..., HAND_OBS].set(0.0)


# ---------------------------------------------------------------- search

def _board_score(state: J.EnvState) -> jnp.ndarray:
    return J.calculate_total_score(state.grid, state.discard)


def _recurrent_fn(params, rng_key, action, embedding):
    del rng_key
    state, score = embedding
    next_state, _, _ = J.step(state, action)
    next_score = _board_score(next_state)
    round_done = next_state.current_round != state.current_round
    game_over = next_state.done

    obs = J.build_observation(next_state)
    obs = jnp.where(round_done[:, None], afterstate_observation(obs), obs)
    logits, value = policy_value(params, obs)
    logits = jnp.where(J.get_action_mask(next_state), logits, NEG_INF)

    reward = (next_score - score) / VALUE_SCALE + jnp.where(round_done & ~game_over, value, 0.0)
    output = mctx.RecurrentFnOutput(
        reward=reward,
        discount=jnp.where(round_done, 0.0, 1.0),
        prior_logits=logits,
        value=jnp.where(round_done, 0.0, value),
    )
    return output, (next_state, next_score)


@partial(jax.jit, static_argnames=("num_simulations", "max_num_considered_actions"))
def search(params, state: J.EnvState, key: jax.Array, num_simulations: int,
           max_num_considered_actions: int = 16, gumbel_scale: float = 1.0):
    """Gumbel MuZero search from every game in `state` (none may be done). Returns mctx PolicyOutput."""
    det_key, search_key = jax.random.split(key)
    mask = J.get_action_mask(state)
    logits, value = policy_value(params, J.build_observation(state))
    search_state = J.determinize(state, det_key)
    root = mctx.RootFnOutput(
        prior_logits=jnp.where(mask, logits, NEG_INF),
        value=value,
        embedding=(search_state, _board_score(state)),
    )
    return mctx.gumbel_muzero_policy(
        params,
        search_key,
        root,
        _recurrent_fn,
        num_simulations=num_simulations,
        invalid_actions=~mask,
        max_num_considered_actions=max_num_considered_actions,
        gumbel_scale=gumbel_scale,
    )


@jax.jit
def network_actions(params, state: J.EnvState) -> jnp.ndarray:
    logits, _ = policy_value(params, J.build_observation(state))
    return jnp.argmax(jnp.where(J.get_action_mask(state), logits, NEG_INF), axis=-1)


# ---------------------------------------------------------------- self-play and training samples

def self_play(params, decks: jnp.ndarray, key: jax.Array, num_simulations: int,
              max_num_considered_actions: int, gumbel_scale: float) -> tuple[dict, np.ndarray]:
    """Play every deck to the end with search. Returns samples and final scores.

    Samples are compact numpy arrays: grid (16,), hand (5,), discard (4,) card ids or -1;
    round; afterstate (hand hidden, no policy target); policy (80,) search weights; value target.
    """
    state = J.init_env_state_from_decks(decks)
    steps = []
    for _ in range(J.ROUNDS * J.PLAYS_PER_ROUND):
        key, step_key = jax.random.split(key)
        out = search(params, state, step_key, num_simulations, max_num_considered_actions, gumbel_scale)
        steps.append((state, out.action_weights, _board_score(state)))
        state, _, _ = J.step(state, out.action)
    final = _board_score(state)

    grid = np.concatenate([np.asarray(s.grid) for s, _, _ in steps])
    hand = np.concatenate([np.asarray(J._get_hand_cards(s.deck, s.current_round)) for s, _, _ in steps])
    discard = np.concatenate([np.asarray(s.discard) for s, _, _ in steps])
    rnd = np.concatenate([np.asarray(s.current_round) for s, _, _ in steps])
    round_start = np.concatenate([np.asarray(s.num_moves) == 0 for s, _, _ in steps])
    policy = np.concatenate([np.asarray(w) for _, w, _ in steps])
    value = np.concatenate([np.asarray((final - sc) / VALUE_SCALE) for _, _, sc in steps])

    # Every round-start state also yields an afterstate sample (same board, hand hidden).
    def stack(normal, after):
        return np.concatenate([normal, after[round_start]])

    samples = dict(
        grid=stack(grid, grid).astype(np.int8),
        hand=stack(hand, hand).astype(np.int8),
        discard=stack(discard, discard).astype(np.int8),
        round=stack(rnd, rnd).astype(np.int8),
        afterstate=np.concatenate([np.zeros(len(grid), bool), np.ones(round_start.sum(), bool)]),
        policy=stack(policy, policy).astype(np.float32),
        value=stack(value, value).astype(np.float32),
    )
    return samples, np.asarray(final)


def _relabel(cards: jnp.ndarray, suit_perm: jnp.ndarray) -> jnp.ndarray:
    return jnp.where(cards >= 0, suit_perm[jnp.maximum(cards, 0)], -1)


def augment(batch: dict, key: jax.Array) -> dict:
    """Apply a random cell symmetry and suit relabeling to each sample."""
    n = batch["grid"].shape[0]
    cell_key, suit_key = jax.random.split(key)
    k = jax.random.randint(cell_key, (n,), 0, CELL_PERMS.shape[0])
    s = jax.random.randint(suit_key, (n,), 0, SUIT_PERMS.shape[0])
    suit_perm = jnp.asarray(SUIT_PERMS)[s]
    relabel = jax.vmap(_relabel)
    grid = jnp.take_along_axis(batch["grid"], jnp.asarray(CELL_PERMS)[k], axis=1)
    return {
        **batch,
        "grid": relabel(grid, suit_perm),
        "hand": relabel(batch["hand"], suit_perm),
        "discard": relabel(batch["discard"], suit_perm),
        "policy": jnp.take_along_axis(batch["policy"], jnp.asarray(ACTION_PERMS)[k], axis=1),
    }


def observation_from_parts(grid, hand, discard, rnd, afterstate):
    """Same layout as jax_env.build_observation; the hand is zeroed for afterstates."""
    n = grid.shape[0]

    def one_hot(cards):
        return jax.nn.one_hot(jnp.maximum(cards, 0), J.DECK_SIZE) * (cards >= 0)[..., None]

    hand_obs = one_hot(hand).reshape(n, -1) * ~afterstate[:, None]
    return jnp.concatenate(
        [one_hot(grid).reshape(n, -1), hand_obs, one_hot(discard).sum(1), jax.nn.one_hot(rnd, J.ROUNDS)], axis=-1
    )


def action_mask_from_parts(grid, hand):
    card_free = ~jnp.any(hand[:, :, None] == grid[:, None, :], axis=-1)
    return (card_free[:, :, None] & (grid == -1)[:, None, :]).reshape(grid.shape[0], J.ACTION_SIZE)


def loss_fn(params, batch: dict, value_coef: float, l2_coef: float):
    grid, hand, discard = (batch[k].astype(jnp.int32) for k in ("grid", "hand", "discard"))
    afterstate = batch["afterstate"]
    obs = observation_from_parts(grid, hand, discard, batch["round"].astype(jnp.int32), afterstate)
    logits, value = policy_value(params, obs)
    logits = jnp.where(action_mask_from_parts(grid, hand), logits, NEG_INF)
    log_probs = jax.nn.log_softmax(logits)
    cross_entropy = -jnp.sum(batch["policy"] * log_probs, axis=-1)
    policy_loss = jnp.sum(jnp.where(afterstate, 0.0, cross_entropy)) / jnp.maximum(jnp.sum(~afterstate), 1)
    value_err = (value - batch["value"]) ** 2
    value_loss = jnp.mean(value_err)
    l2 = sum(jnp.sum(layer["w"] ** 2) for layer in params["hidden"] + [params["policy"], params["value"]])
    loss = policy_loss + value_coef * value_loss + l2_coef * l2
    metrics = {
        "loss": loss,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "afterstate_value_loss": jnp.sum(jnp.where(afterstate, value_err, 0.0)) / jnp.maximum(jnp.sum(afterstate), 1),
    }
    return loss, metrics


# ---------------------------------------------------------------- evaluation

def play(params, decks: np.ndarray, num_simulations: int, max_num_considered_actions: int = 16,
         gumbel_scale: float = 0.0, seed: int = 0) -> np.ndarray:
    """Final scores on (N, 36) decks. num_simulations=0 plays the raw network argmax."""
    state = J.init_env_state_from_decks(jnp.asarray(decks, jnp.int32))
    key = jax.random.PRNGKey(seed)
    for _ in range(J.ROUNDS * J.PLAYS_PER_ROUND):
        if num_simulations > 0:
            key, step_key = jax.random.split(key)
            action = search(params, state, step_key, num_simulations, max_num_considered_actions, gumbel_scale).action
        else:
            action = network_actions(params, state)
        state, _, _ = J.step(state, action)
    return np.asarray(_board_score(state))


def load_params(path: str):
    with open(path, "rb") as f:
        return jax.tree_util.tree_map(jnp.asarray, pickle.load(f))


def evaluate_on_decks(model_path: str, decks: np.ndarray, num_simulations: int = 64,
                      max_num_considered_actions: int = 16, gumbel_scale: float = 0.0,
                      seed: int = 0, batch_size: int = 256) -> np.ndarray:
    """Final scores (N,) of the AlphaZero agent on (N, 36) decks, deterministic for a given seed.

    gumbel_scale=0 makes the search pick the best action by sequential halving without noise.
    """
    params = load_params(model_path)
    decks = np.asarray(decks, np.int32)
    n = len(decks)
    batch_size = min(batch_size, n)
    # Pad to whole batches so the search compiles once.
    padded = np.concatenate([decks, decks[: (-n) % batch_size]])
    scores = [
        play(params, padded[i:i + batch_size], num_simulations, max_num_considered_actions, gumbel_scale, seed + i)
        for i in range(0, len(padded), batch_size)
    ]
    return np.concatenate(scores)[:n]
