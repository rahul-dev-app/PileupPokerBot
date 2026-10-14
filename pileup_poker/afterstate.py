"""Afterstate value agent (TD-Gammon / 2048 n-tuple style) for Pile-Up Poker.

An afterstate is the position right after a card is placed and before any chance event:
board, discard pile (on the 4th placement of a round the unplaced hand card is already
added), the hand cards still to place, the number of cards on the board and the set of
unseen cards. The agent places each card on the legal action whose afterstate has the
highest value V, an estimate of the expected final score divided by `value_scale`.

V = phi + f(afterstate), with phi = current board score / value_scale. The last afterstate
of a game (16 cards placed) is the final position, so its value is the exact final score.

Afterstates are built from the state arrays only (board, discard, current hand, used
cards). The order of undealt cards in `state.deck` is never read.

Two models for f:
- `ntuple`: one lookup table per line class (outer rows/cols, inner rows/cols, corners,
  discard) and per round, indexed by the line's rank multiset (empties included) and
  whether its suits are all equal, so it is suit-invariant and shared across the board
  symmetries; plus a linear term on small context features.
- `mlp`: a 2-layer MLP on per-line features, hand and unseen multi-hots and context.
"""
from __future__ import annotations

import os
import pickle
from functools import partial
from itertools import combinations_with_replacement
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from pileup_poker import jax_env as J
from pileup_poker.symmetry import CELL_PERMS, SUIT_PERMS, inverse_perms

NUM_RANKS = 9
NUM_SUITS = 4
NUM_STEPS = J.ROUNDS * J.PLAYS_PER_ROUND
MODELS = ("ntuple", "mlp")

_grid = np.arange(J.GRID_CELLS).reshape(J.GRID_SIZE, J.GRID_SIZE)
# 4 rows, 4 columns, corners. The discard pile is treated as a 10th line.
LINE_CELLS = np.concatenate([_grid, _grid.T, [[0, 3, 12, 15]]]).astype(np.int32)
# Line classes: 0 = outer row/column (holds corners), 1 = inner row/column, 2 = corners, 3 = discard.
LINE_CLASS = np.array([0, 1, 1, 0, 0, 1, 1, 0, 2, 3], dtype=np.int32)
NUM_LINES = len(LINE_CLASS)
NUM_LINE_CLASSES = 4

# Sorted 4-tuples of ranks 0..8 or 9 (empty) -> dense multiset id.
_multisets = list(combinations_with_replacement(range(NUM_RANKS + 1), 4))
MULTISET_ID = np.zeros(10**4, dtype=np.int32)
for _i, (_a, _b, _c, _d) in enumerate(_multisets):
    MULTISET_ID[((_a * 10 + _b) * 10 + _c) * 10 + _d] = _i
NUM_LINE_KEYS = 2 * len(_multisets)  # x flush-alive bit

_SUIT_INV = inverse_perms(SUIT_PERMS)


class Afterstate(NamedTuple):
    grid: jnp.ndarray  # (..., 16) card id or -1
    discard: jnp.ndarray  # (..., 4) card id or -1
    hand: jnp.ndarray  # (..., 5) hand cards still to place this round, -1 otherwise
    placed: jnp.ndarray  # (...,) cards on the board, 1..16
    unseen: jnp.ndarray  # (..., 36) bool, cards not on the board, in the discard or in the current hand


def _multi_hot(cards: jnp.ndarray) -> jnp.ndarray:
    """(..., k) card ids or -1 -> (..., 36) bool."""
    return jnp.any((cards[..., None] == jnp.arange(J.DECK_SIZE)) & (cards[..., None] >= 0), axis=-2)


def expand_afterstates(state: J.EnvState) -> tuple[Afterstate, jnp.ndarray]:
    """Afterstates of all 80 actions for every game: Afterstate with leading dims (B, 80), and the legal mask."""
    num_envs = state.grid.shape[0]
    hand = J._get_hand_cards(state.deck, state.current_round)
    hand_left = jnp.where(jnp.take_along_axis(state.used, hand, axis=1), -1, hand)
    unseen = ~(state.used | _multi_hot(state.discard) | _multi_hot(hand))

    slot = jnp.arange(J.ACTION_SIZE) // J.GRID_CELLS
    cell = jnp.arange(J.ACTION_SIZE) % J.GRID_CELLS
    card = hand[:, slot]  # (B, 80)
    cell_hot = cell[:, None] == jnp.arange(J.GRID_CELLS)
    grid = jnp.where(cell_hot[None], card[..., None], state.grid[:, None, :])

    slot_hot = slot[:, None] == jnp.arange(J.HAND_SIZE)
    hand_after = jnp.where(slot_hot[None], -1, hand_left[:, None, :])
    round_end = (state.num_moves + 1 == J.PLAYS_PER_ROUND)[:, None, None]
    leftover = jnp.max(hand_after, axis=-1)  # the single card left at the round end
    round_hot = (jnp.arange(J.ROUNDS) == state.current_round[:, None])[:, None, :]
    discard = jnp.where(round_end & round_hot, leftover[..., None], state.discard[:, None, :])
    hand_after = jnp.where(round_end, -1, hand_after)

    placed = J.PLAYS_PER_ROUND * state.current_round + state.num_moves + 1
    afterstates = Afterstate(
        grid=grid,
        discard=discard,
        hand=hand_after,
        placed=jnp.broadcast_to(placed[:, None], (num_envs, J.ACTION_SIZE)),
        unseen=jnp.broadcast_to(unseen[:, None, :], (num_envs, J.ACTION_SIZE, J.DECK_SIZE)),
    )
    return afterstates, J.get_action_mask(state)


def augment(a: Afterstate, key: jax.Array) -> Afterstate:
    """Random board symmetry and suit relabeling per afterstate (all have leading dim N)."""
    k_cell, k_suit = jax.random.split(key)
    n = a.grid.shape[0]
    cell_perm = jnp.asarray(CELL_PERMS)[jax.random.randint(k_cell, (n,), 0, len(CELL_PERMS))]
    suit_idx = jax.random.randint(k_suit, (n,), 0, len(SUIT_PERMS))
    suit_perm = jnp.asarray(SUIT_PERMS)[suit_idx]

    def relabel(cards):
        return jnp.where(cards >= 0, jnp.take_along_axis(suit_perm, jnp.maximum(cards, 0), axis=1), -1)

    return Afterstate(
        grid=relabel(jnp.take_along_axis(a.grid, cell_perm, axis=1)),
        discard=relabel(a.discard),
        hand=relabel(a.hand),
        placed=a.placed,
        unseen=jnp.take_along_axis(a.unseen, jnp.asarray(_SUIT_INV)[suit_idx], axis=1),
    )


def _lines(a: Afterstate) -> jnp.ndarray:
    return jnp.concatenate([a.grid[:, LINE_CELLS], a.discard[:, None, :]], axis=1)  # (N, 10, 4)


def _rank_counts(cards: jnp.ndarray) -> jnp.ndarray:
    """(..., k) card ids or -1 -> (..., 9) rank counts."""
    hot = (cards[..., None] // NUM_SUITS == jnp.arange(NUM_RANKS)) & (cards[..., None] >= 0)
    return jnp.sum(hot, axis=-2).astype(jnp.float32)


def _flush_suit(lines: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Whether all cards of each line share a suit (true for empty lines), and that suit (-1 if empty)."""
    valid = lines >= 0
    suits = lines % NUM_SUITS
    smax = jnp.max(jnp.where(valid, suits, -1), axis=-1)
    smin = jnp.min(jnp.where(valid, suits, NUM_SUITS), axis=-1)
    return smax <= smin, smax


def context_features(a: Afterstate) -> jnp.ndarray:
    """Cards placed one-hot, hand and unseen rank counts, unseen suit counts (sorted: suit-invariant)."""
    unseen = a.unseen.astype(jnp.float32)
    unseen_suits = jnp.sort(unseen.reshape(-1, NUM_RANKS, NUM_SUITS).sum(axis=1), axis=-1)
    return jnp.concatenate(
        [
            jax.nn.one_hot(a.placed - 1, NUM_STEPS),
            _rank_counts(a.hand) / 2.0,
            unseen.reshape(-1, NUM_RANKS, NUM_SUITS).sum(axis=-1) / NUM_SUITS,
            unseen_suits / NUM_RANKS,
        ],
        axis=-1,
    )


NUM_CONTEXT = NUM_STEPS + NUM_RANKS + NUM_RANKS + NUM_SUITS


def ntuple_keys(a: Afterstate) -> jnp.ndarray:
    """(N, 10) table index per line: rank multiset with empties, times the flush-alive bit."""
    lines = _lines(a)
    ranks = jnp.sort(jnp.where(lines >= 0, lines // NUM_SUITS, NUM_RANKS), axis=-1)
    code = ((ranks[..., 0] * 10 + ranks[..., 1]) * 10 + ranks[..., 2]) * 10 + ranks[..., 3]
    flush, _ = _flush_suit(lines)
    return jnp.asarray(MULTISET_ID)[code] * 2 + flush.astype(jnp.int32)


def mlp_features(a: Afterstate) -> jnp.ndarray:
    lines = _lines(a)
    valid = lines >= 0
    n_cards = valid.sum(-1)
    counts = _rank_counts(lines)
    flush, suit = _flush_suit(lines)
    present = counts > 0
    lo = jnp.argmax(present, axis=-1)
    hi = NUM_RANKS - 1 - jnp.argmax(present[..., ::-1], axis=-1)
    straight = (n_cards == 0) | ((counts.max(-1) <= 1) & (hi - lo <= 3))
    unseen_by_suit = a.unseen.reshape(-1, NUM_RANKS, NUM_SUITS).sum(axis=1)  # (N, 4)
    flush_left = jnp.where(
        flush & (suit >= 0), jnp.take_along_axis(unseen_by_suit, jnp.maximum(suit, 0), axis=1), 0
    )
    line_score = jax.vmap(jax.vmap(J._line_score))(lines)
    per_line = jnp.concatenate(
        [
            counts,
            (n_cards / 4.0)[..., None],
            flush[..., None],
            straight[..., None],
            (flush_left / NUM_RANKS)[..., None],
            (line_score / 450.0)[..., None],
        ],
        axis=-1,
    ).astype(jnp.float32)
    return jnp.concatenate(
        [
            per_line.reshape(per_line.shape[0], -1),
            _multi_hot(a.hand).astype(jnp.float32),
            a.unseen.astype(jnp.float32),
            context_features(a),
        ],
        axis=-1,
    )


NUM_MLP_FEATURES = NUM_LINES * (NUM_RANKS + 5) + 2 * J.DECK_SIZE + NUM_CONTEXT


def init_params(model: str, key: jax.Array, hidden_dim: int = 256, num_layers: int = 2):
    """num_layers: hidden layers of the MLP (weights w1..w{num_layers + 1})."""
    if model == "ntuple":
        return {
            "tables": jnp.zeros((NUM_LINE_CLASSES, J.ROUNDS, NUM_LINE_KEYS), jnp.float32),
            "w": jnp.zeros((NUM_CONTEXT,), jnp.float32),
            "b": jnp.zeros((), jnp.float32),
        }
    keys = jax.random.split(key, num_layers + 1)

    def dense(k, fan_in, fan_out, scale):
        return jax.random.normal(k, (fan_in, fan_out), jnp.float32) * (scale / np.sqrt(fan_in)), jnp.zeros(fan_out)

    sizes = [NUM_MLP_FEATURES] + [hidden_dim] * num_layers
    params = {}
    for i in range(num_layers):
        params[f"w{i + 1}"], params[f"b{i + 1}"] = dense(keys[i], sizes[i], sizes[i + 1], np.sqrt(2.0))
    w, b = dense(keys[-1], hidden_dim, 1, 0.01)
    params[f"w{num_layers + 1}"], params[f"b{num_layers + 1}"] = w, b[0]
    return params


def _residual(model: str, params, a: Afterstate) -> jnp.ndarray:
    if model == "ntuple":
        stage = (a.placed - 1) // J.PLAYS_PER_ROUND
        table_vals = params["tables"][LINE_CLASS[None, :], stage[:, None], ntuple_keys(a)]
        return table_vals.sum(-1) + context_features(a) @ params["w"] + params["b"]
    out = sum(k.startswith("w") for k in params)  # hidden layers + 1
    h = mlp_features(a)
    for i in range(1, out):
        h = jax.nn.relu(h @ params[f"w{i}"] + params[f"b{i}"])
    return (h @ params[f"w{out}"])[:, 0] + params[f"b{out}"]


def value(model: str, params, a: Afterstate, value_scale: float) -> jnp.ndarray:
    """V(afterstate) in units of score / value_scale; afterstates have leading dim N."""
    phi = J.calculate_total_score(a.grid, a.discard) / value_scale
    return jnp.where(a.placed == NUM_STEPS, phi, phi + _residual(model, params, a))


@partial(jax.jit, static_argnames=("model",))
def play_games(model: str, params, state: J.EnvState, key: jax.Array, epsilon, temperature, value_scale: float):
    """Play every game in `state` from its start to the end.

    Each move takes the max-value afterstate; with temperature > 0 it samples softmax(V / temperature)
    instead, and with probability epsilon it plays a uniform random legal action.
    Returns the chosen afterstates (16, B, ...), the best legal value per step (16, B), whether the
    played action was the greedy one (16, B), and the final scores (B,).
    """
    num_envs = state.grid.shape[0]

    def one_step(state, key):
        k_temp, k_eps, k_rand = jax.random.split(key, 3)
        afterstates, mask = expand_afterstates(state)
        flat = jax.tree_util.tree_map(lambda x: x.reshape((num_envs * J.ACTION_SIZE,) + x.shape[2:]), afterstates)
        v = jnp.where(mask, value(model, params, flat, value_scale).reshape(num_envs, J.ACTION_SIZE), -jnp.inf)
        greedy = jnp.argmax(v, axis=1)
        sampled = jax.random.categorical(k_temp, v / jnp.maximum(temperature, 1e-6), axis=1)
        action = jnp.where(temperature > 0, sampled, greedy)
        random_action = jax.random.categorical(k_rand, jnp.where(mask, 0.0, -jnp.inf), axis=1)
        action = jnp.where(jax.random.uniform(k_eps, (num_envs,)) < epsilon, random_action, action)
        chosen = jax.tree_util.tree_map(
            lambda x: jnp.take_along_axis(x, action.reshape((num_envs, 1) + (1,) * (x.ndim - 2)), axis=1)[:, 0],
            afterstates,
        )
        state, reward, _ = J.step(state, action)
        return state, (chosen, jnp.max(v, axis=1), action == greedy, reward)

    _, (chosen, best_values, is_greedy, rewards) = jax.lax.scan(one_step, state, jax.random.split(key, NUM_STEPS))
    return chosen, best_values, is_greedy, rewards.sum(0)


def lambda_returns(best_values: jnp.ndarray, is_greedy: jnp.ndarray, final: jnp.ndarray, td_lambda: float) -> jnp.ndarray:
    """TD(lambda) targets for the chosen afterstates of steps 0..14, shape (15, B).

    Bootstraps from the best legal value at the next step and cuts the trace after a
    non-greedy action (Watkins), so the targets estimate the greedy policy's value.
    `final` is the final score in value units.
    """

    def backward(g, xs):
        best_next, greedy_next = xs
        g = jnp.where(greedy_next, (1.0 - td_lambda) * best_next + td_lambda * g, best_next)
        return g, g

    _, targets = jax.lax.scan(backward, final, (best_values[1:], is_greedy[1:]), reverse=True)
    return targets


def save_params(path: str, model: str, params, hidden_dim: int, value_scale: float) -> None:
    payload = {"model": model, "hidden_dim": hidden_dim, "value_scale": value_scale, "params": jax.device_get(params)}
    tmp_path = path + ".tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(payload, f)
    os.replace(tmp_path, path)


def load_params(path: str) -> dict:
    with open(path, "rb") as f:
        payload = pickle.load(f)
    payload["params"] = jax.tree_util.tree_map(jnp.asarray, payload["params"])
    return payload


def evaluate_on_decks(model_path: str, decks: np.ndarray, batch_size: int = 512, seed: int = 0) -> np.ndarray:
    """Final scores (N,) of the greedy afterstate policy on (N, 36) decks.

    The policy is deterministic, so `seed` changes nothing; it is accepted for a uniform interface.
    """
    payload = load_params(model_path)
    key = jax.random.PRNGKey(seed)
    scores = []
    for start in range(0, len(decks), batch_size):
        state = J.init_env_state_from_decks(jnp.asarray(decks[start : start + batch_size]))
        *_, final = play_games(payload["model"], payload["params"], state, key, 0.0, 0.0, payload["value_scale"])
        scores.append(np.asarray(final))
    return np.concatenate(scores)
