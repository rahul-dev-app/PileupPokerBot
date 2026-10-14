from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


GRID_SIZE = 4
GRID_CELLS = GRID_SIZE * GRID_SIZE
HAND_SIZE = 5
PLAYS_PER_ROUND = 4
ROUNDS = 4
DECK_SIZE = 36
ACTION_SIZE = HAND_SIZE * GRID_CELLS
# grid one-hot, hand one-hot, discard multi-hot, round one-hot
OBS_SIZE = GRID_CELLS * DECK_SIZE + HAND_SIZE * DECK_SIZE + DECK_SIZE + ROUNDS


POKER_HAND_SCORES = {
    "no_hand": 0.0,
    "pair": 5.0,
    "two_pairs": 60.0,
    "flush": 80.0,
    "three_of_a_kind": 125.0,
    "straight": 180.0,
    "four_of_a_kind": 325.0,
    "straight_flush": 450.0,
}


class EnvState(NamedTuple):
    deck: jnp.ndarray
    grid: jnp.ndarray
    used: jnp.ndarray
    current_round: jnp.ndarray
    num_moves: jnp.ndarray
    discard: jnp.ndarray
    done: jnp.ndarray


def _mask_select(mask: jnp.ndarray, new_value: jnp.ndarray, old_value: jnp.ndarray) -> jnp.ndarray:
    expand_shape = (mask.shape[0],) + (1,) * (new_value.ndim - 1)
    return jnp.where(mask.reshape(expand_shape), new_value, old_value)


def _shuffle_decks(keys: jnp.ndarray) -> jnp.ndarray:
    return jax.vmap(lambda k: jax.random.permutation(k, DECK_SIZE))(keys).astype(jnp.int32)


def init_env_state(keys: jnp.ndarray) -> EnvState:
    return init_env_state_from_decks(_shuffle_decks(keys))


def init_env_state_from_decks(decks: jnp.ndarray) -> EnvState:
    """decks: (num_envs, DECK_SIZE) card ids in draw order; hand r is decks[:, 5r:5r+5]."""
    num_envs = decks.shape[0]
    return EnvState(
        deck=jnp.asarray(decks, dtype=jnp.int32),
        grid=jnp.full((num_envs, GRID_CELLS), -1, dtype=jnp.int32),
        used=jnp.zeros((num_envs, DECK_SIZE), dtype=jnp.bool_),
        current_round=jnp.zeros((num_envs,), dtype=jnp.int32),
        num_moves=jnp.zeros((num_envs,), dtype=jnp.int32),
        discard=jnp.full((num_envs, ROUNDS), -1, dtype=jnp.int32),
        done=jnp.zeros((num_envs,), dtype=jnp.bool_),
    )


@jax.jit
def determinize(state: EnvState, key: jax.Array) -> EnvState:
    """Shuffle each game's undealt cards, keeping every dealt card (board, discard, current hand).

    Search agents run on determinized copies so they never see the real future draws.
    """
    num_envs = state.deck.shape[0]
    dealt = (state.current_round[:, None] + 1) * HAND_SIZE
    positions = jnp.arange(DECK_SIZE)[None, :]
    # Noise is drawn per card id, not per position, so the result depends only on which cards
    # are undealt, never on their real order.
    card_noise = jax.random.uniform(key, (num_envs, DECK_SIZE))
    noise = jnp.take_along_axis(card_noise, state.deck, axis=1)
    # Dealt positions sort first in their own order; undealt positions sort randomly after them.
    sort_keys = jnp.where(positions < dealt, positions - DECK_SIZE, noise)
    order = jnp.argsort(sort_keys, axis=1)
    return state._replace(deck=jnp.take_along_axis(state.deck, order, axis=1))


def reset(key: jax.Array, num_envs: int) -> tuple[EnvState, jnp.ndarray, jnp.ndarray]:
    keys = jax.random.split(key, num_envs)
    state = init_env_state(keys)
    obs = build_observation(state)
    mask = get_action_mask(state)
    return state, obs, mask


def _get_hand_cards(deck: jnp.ndarray, current_round: jnp.ndarray) -> jnp.ndarray:
    offsets = current_round[:, None] * HAND_SIZE + jnp.arange(HAND_SIZE, dtype=jnp.int32)[None, :]
    return jnp.take_along_axis(deck, offsets, axis=1)


@jax.jit
def build_observation(state: EnvState) -> jnp.ndarray:
    num_envs = state.grid.shape[0]

    grid_indices = jnp.where(state.grid >= 0, state.grid, 0)
    grid_one_hot = jax.nn.one_hot(grid_indices, DECK_SIZE, dtype=jnp.float32)
    grid_one_hot = grid_one_hot * (state.grid[..., None] >= 0)
    grid_obs = grid_one_hot.reshape(num_envs, GRID_CELLS * DECK_SIZE)

    hand_cards = _get_hand_cards(state.deck, state.current_round)
    hand_one_hot = jax.nn.one_hot(hand_cards, DECK_SIZE, dtype=jnp.float32)
    hand_obs = hand_one_hot.reshape(num_envs, HAND_SIZE * DECK_SIZE)

    discard_one_hot = jax.nn.one_hot(jnp.where(state.discard >= 0, state.discard, 0), DECK_SIZE, dtype=jnp.float32)
    discard_obs = jnp.sum(discard_one_hot * (state.discard[..., None] >= 0), axis=1)

    # All zeros once the game is over (current_round == ROUNDS).
    round_obs = jax.nn.one_hot(state.current_round, ROUNDS, dtype=jnp.float32)

    return jnp.concatenate([grid_obs, hand_obs, discard_obs, round_obs], axis=-1)


@jax.jit
def get_action_mask(state: EnvState) -> jnp.ndarray:
    num_envs = state.grid.shape[0]
    env_ids = jnp.arange(num_envs, dtype=jnp.int32)[:, None]

    hand_cards = _get_hand_cards(state.deck, state.current_round)
    card_unused = ~state.used[env_ids, hand_cards]
    empty_positions = state.grid == -1

    action_mask = (card_unused[:, :, None] & empty_positions[:, None, :]).reshape(num_envs, ACTION_SIZE)
    return jnp.where(state.done[:, None], jnp.zeros_like(action_mask), action_mask)


def _generic_hand_score(cards4: jnp.ndarray, valid_mask: jnp.ndarray) -> jnp.ndarray:
    n_valid = jnp.sum(valid_mask)
    ranks = jnp.where(valid_mask, cards4 // 4, -99)
    suits = jnp.where(valid_mask, cards4 % 4, -99)

    eq = (ranks[:, None] == ranks[None, :]) & valid_mask[:, None] & valid_mask[None, :]
    rank_counts = jnp.sum(eq, axis=1)

    is_new_0 = valid_mask[0]
    is_new_1 = valid_mask[1] & ~(valid_mask[0] & (ranks[1] == ranks[0]))
    is_new_2 = valid_mask[2] & ~(
        (valid_mask[0] & (ranks[2] == ranks[0])) | (valid_mask[1] & (ranks[2] == ranks[1]))
    )
    is_new_3 = valid_mask[3] & ~(
        (valid_mask[0] & (ranks[3] == ranks[0]))
        | (valid_mask[1] & (ranks[3] == ranks[1]))
        | (valid_mask[2] & (ranks[3] == ranks[2]))
    )
    unique_ranks = is_new_0.astype(jnp.int32)
    unique_ranks = unique_ranks + is_new_1.astype(jnp.int32)
    unique_ranks = unique_ranks + is_new_2.astype(jnp.int32)
    unique_ranks = unique_ranks + is_new_3.astype(jnp.int32)

    has_three = jnp.any(rank_counts == 3)

    first_suit = jnp.where(
        valid_mask[0],
        suits[0],
        jnp.where(valid_mask[1], suits[1], jnp.where(valid_mask[2], suits[2], suits[3])),
    )
    flush = (n_valid > 0) & jnp.all((~valid_mask) | (suits == first_suit))

    sorted_ranks = jnp.sort(jnp.where(valid_mask, ranks, 99))
    straight = (
        (n_valid == 4)
        & (sorted_ranks[1] == sorted_ranks[0] + 1)
        & (sorted_ranks[2] == sorted_ranks[1] + 1)
        & (sorted_ranks[3] == sorted_ranks[2] + 1)
    )

    score = jnp.where(unique_ranks == 1, POKER_HAND_SCORES["four_of_a_kind"], POKER_HAND_SCORES["no_hand"])
    score = jnp.where((unique_ranks == 2) & has_three, POKER_HAND_SCORES["three_of_a_kind"], score)
    score = jnp.where((unique_ranks == 2) & (~has_three), POKER_HAND_SCORES["two_pairs"], score)
    score = jnp.where(unique_ranks == 3, POKER_HAND_SCORES["pair"], score)
    score = jnp.where(
        (unique_ranks > 3) & straight & flush, POKER_HAND_SCORES["straight_flush"], score
    )
    score = jnp.where(
        (unique_ranks > 3) & (~straight) & flush, POKER_HAND_SCORES["flush"], score
    )
    score = jnp.where(
        (unique_ranks > 3) & (~flush) & straight, POKER_HAND_SCORES["straight"], score
    )
    return score


def _partial_pair_or_trip_score(cards4: jnp.ndarray, valid_mask: jnp.ndarray) -> jnp.ndarray:
    n_valid = jnp.sum(valid_mask)
    ranks = jnp.where(valid_mask, cards4 // 4, -99)
    eq = (ranks[:, None] == ranks[None, :]) & valid_mask[:, None] & valid_mask[None, :]
    rank_counts = jnp.sum(eq, axis=1)
    max_count = jnp.max(jnp.where(valid_mask, rank_counts, 0))

    score_3 = jnp.where(
        max_count == 3, POKER_HAND_SCORES["three_of_a_kind"], jnp.where(max_count == 2, POKER_HAND_SCORES["pair"], 0.0)
    )
    score_2 = jnp.where(max_count == 2, POKER_HAND_SCORES["pair"], 0.0)
    return jnp.where(n_valid == 3, score_3, jnp.where(n_valid == 2, score_2, 0.0))


def _line_score(cards4: jnp.ndarray) -> jnp.ndarray:
    valid_mask = cards4 >= 0
    n_valid = jnp.sum(valid_mask)
    full_score = _generic_hand_score(cards4, valid_mask)
    partial_score = _partial_pair_or_trip_score(cards4, valid_mask)
    return jnp.where(n_valid == 4, full_score, jnp.where((n_valid > 1) & (n_valid < 4), partial_score, 0.0))


def _calculate_total_score_single(grid16: jnp.ndarray, discard4: jnp.ndarray) -> jnp.ndarray:
    grid = grid16.reshape(GRID_SIZE, GRID_SIZE)

    row_scores = jnp.array([_line_score(grid[i, :]) for i in range(GRID_SIZE)])
    col_scores = jnp.array([_line_score(grid[:, j]) for j in range(GRID_SIZE)])

    corner_cards = jnp.array([grid[0, 0], grid[0, GRID_SIZE - 1], grid[GRID_SIZE - 1, 0], grid[GRID_SIZE - 1, GRID_SIZE - 1]])
    corner_valid = corner_cards >= 0
    corner_score = jnp.where(jnp.all(corner_valid), _generic_hand_score(corner_cards, corner_valid) * 2.0, 0.0)

    nine_hands = jnp.concatenate([row_scores, col_scores, jnp.array([corner_score])], axis=0)
    num_scoring_hands = jnp.sum(nine_hands != 0.0)

    discard_valid = discard4 >= 0
    discard_score = jnp.where(
        num_scoring_hands == 9, _generic_hand_score(discard4, discard_valid) * 3.0, 0.0
    )

    all_hands = jnp.concatenate([nine_hands, jnp.array([discard_score])], axis=0)
    scored_count = jnp.sum(all_hands != 0.0)

    multiplier = jnp.where(
        (scored_count >= 2) & (scored_count <= 3),
        2.0,
        jnp.where(
            (scored_count >= 4) & (scored_count <= 5),
            3.0,
            jnp.where(
                (scored_count >= 6) & (scored_count <= 7),
                4.0,
                jnp.where((scored_count >= 8) & (scored_count <= 9), 5.0, jnp.where(scored_count == 10, 6.0, 1.0)),
            ),
        ),
    )
    return jnp.sum(all_hands) * multiplier


calculate_total_score = jax.jit(jax.vmap(_calculate_total_score_single, in_axes=(0, 0)))


@jax.jit
def step(state: EnvState, action: jnp.ndarray) -> tuple[EnvState, jnp.ndarray, jnp.ndarray]:
    num_envs = state.grid.shape[0]
    env_ids = jnp.arange(num_envs, dtype=jnp.int32)

    action = jnp.clip(action.astype(jnp.int32), 0, ACTION_SIZE - 1)
    action_mask = get_action_mask(state)
    has_any_valid_action = jnp.any(action_mask, axis=1)
    proposed_is_valid = action_mask[env_ids, action]
    fallback_action = jnp.argmax(action_mask.astype(jnp.int32), axis=1)
    action = jnp.where(proposed_is_valid, action, fallback_action)

    card_slot = action // GRID_CELLS
    position = action % GRID_CELLS

    hand_cards = _get_hand_cards(state.deck, state.current_round)
    selected_cards = hand_cards[env_ids, card_slot]

    pos_empty = state.grid[env_ids, position] == -1
    card_unused = ~state.used[env_ids, selected_cards]
    active = ~state.done & (state.current_round < ROUNDS)
    valid = active & has_any_valid_action & pos_empty & card_unused

    placed_grid = state.grid.at[env_ids, position].set(selected_cards)
    next_grid = _mask_select(valid, placed_grid, state.grid)

    used_after_place = state.used.at[env_ids, selected_cards].set(True)
    next_used = _mask_select(valid, used_after_place, state.used)

    moves_after = jnp.where(valid, state.num_moves + 1, state.num_moves)
    round_finished = valid & (moves_after == PLAYS_PER_ROUND)

    current_hand_used = next_used[env_ids[:, None], hand_cards]
    unused_in_hand = ~current_hand_used
    discard_slot = jnp.argmax(unused_in_hand, axis=1)
    discard_card = hand_cards[env_ids, discard_slot]

    discard_round_index = jnp.clip(state.current_round, 0, ROUNDS - 1)
    discard_after = state.discard.at[env_ids, discard_round_index].set(discard_card)
    next_discard = _mask_select(round_finished, discard_after, state.discard)

    next_round = jnp.where(round_finished, state.current_round + 1, state.current_round)
    next_moves = jnp.where(round_finished, 0, moves_after)
    next_done = state.done | (next_round >= ROUNDS)

    score = calculate_total_score(next_grid, next_discard)
    # Invalid proposals are auto-corrected to valid actions; reward stays terminal-focused.
    reward = jnp.zeros((num_envs,), dtype=jnp.float32)
    reward = jnp.where(valid & next_done, score, reward)
    reward = jnp.where(state.done, 0.0, reward)

    next_state = EnvState(
        deck=state.deck,
        grid=next_grid,
        used=next_used,
        current_round=next_round,
        num_moves=next_moves,
        discard=next_discard,
        done=next_done,
    )
    return next_state, reward, next_done


def reset_done(state: EnvState, done_mask: jnp.ndarray, key: jax.Array) -> EnvState:
    num_envs = state.grid.shape[0]
    keys = jax.random.split(key, num_envs)
    fresh_state = init_env_state(keys)
    return EnvState(
        deck=_mask_select(done_mask, fresh_state.deck, state.deck),
        grid=_mask_select(done_mask, fresh_state.grid, state.grid),
        used=_mask_select(done_mask, fresh_state.used, state.used),
        current_round=_mask_select(done_mask, fresh_state.current_round, state.current_round),
        num_moves=_mask_select(done_mask, fresh_state.num_moves, state.num_moves),
        discard=_mask_select(done_mask, fresh_state.discard, state.discard),
        done=_mask_select(done_mask, fresh_state.done, state.done),
    )
