"""Fast JAX version of the round-level greedy agent, plus round enumeration helpers.

A round decision = which hand card to discard and where the other 4 go. With e empty
cells there are 5 * C(e, 4) * 4! candidates (218,400 in round 1, 120 in round 4).

`greedy_plans` reproduces `GreedyPileupPokerAgent` exactly (same candidate order, first
maximum wins, same scoring of the board before the round's discard is added), so
`play_greedy` gives the same scores as the Python greedy on the same decks.
"""
from __future__ import annotations

from functools import partial
from itertools import combinations, permutations

import jax
import jax.numpy as jnp
import numpy as np

from pileup_poker import jax_env as J

NUM_STEPS = J.ROUNDS * J.PLAYS_PER_ROUND


def _round_candidates(num_empty: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Candidates in the Python greedy's loop order: hand combo, then cell combo, then card order."""
    slots, positions, discards = [], [], []
    for hand_combo in combinations(range(J.HAND_SIZE), J.PLAYS_PER_ROUND):
        discard = next(s for s in range(J.HAND_SIZE) if s not in hand_combo)
        for pos_combo in combinations(range(num_empty), J.PLAYS_PER_ROUND):
            for perm in permutations(range(J.PLAYS_PER_ROUND)):
                slots.append([hand_combo[i] for i in perm])
                positions.append(pos_combo)
                discards.append(discard)
    return np.array(slots, np.int32), np.array(positions, np.int32), np.array(discards, np.int32)


# Per round r: (slots (N, 4), positions into the sorted empty-cell list (N, 4), discard slot (N,)).
ROUND_CANDIDATES = [_round_candidates(J.GRID_CELLS - J.PLAYS_PER_ROUND * r) for r in range(J.ROUNDS)]


def candidate_boards(grid: jnp.ndarray, discard: jnp.ndarray, hand: jnp.ndarray, round_idx: int, include_discard: bool):
    """All round-`round_idx` candidates for one game at the start of that round.

    Returns grids (N, 16), discards (N, 4), slots (N, 4), cells (N, 4). With include_discard
    the unplaced card is added to the discard pile, giving the true board after the round.
    """
    slots, positions, discard_slots = ROUND_CANDIDATES[round_idx]
    num_empty = J.GRID_CELLS - J.PLAYS_PER_ROUND * round_idx
    empty_cells = jnp.nonzero(grid == -1, size=num_empty)[0].astype(jnp.int32)
    cells = empty_cells[positions]
    cards = hand[slots]
    rows = jnp.arange(slots.shape[0])[:, None]
    grids = jnp.broadcast_to(grid, (slots.shape[0], J.GRID_CELLS)).at[rows, cells].set(cards)
    discards = jnp.broadcast_to(discard, (slots.shape[0], J.ROUNDS))
    if include_discard:
        discards = discards.at[:, round_idx].set(hand[discard_slots])
    return grids, discards, jnp.asarray(slots), cells


def candidate_scores(grid, discard, hand, round_idx: int, include_discard: bool):
    grids, discards, slots, cells = candidate_boards(grid, discard, hand, round_idx, include_discard)
    return J.calculate_total_score(grids, discards), slots, cells


def hand_cards(state: J.EnvState) -> jnp.ndarray:
    return J._get_hand_cards(state.deck, state.current_round)


@partial(jax.jit, static_argnames=("round_idx", "include_discard", "batch_size"))
def greedy_plans(state: J.EnvState, round_idx: int, include_discard: bool = False, batch_size: int = 8):
    """Best candidate per game. All games must be at the start of round `round_idx`.

    Returns slots (B, 4), cells (B, 4), best score (B,).
    """

    def best(args):
        grid, discard, hand = args
        scores, slots, cells = candidate_scores(grid, discard, hand, round_idx, include_discard)
        i = jnp.argmax(scores)  # first maximum, like the Python greedy's strict ">"
        return slots[i], cells[i], scores[i]

    return jax.lax.map(best, (state.grid, state.discard, hand_cards(state)), batch_size=batch_size)


def plan_actions(slots: jnp.ndarray, cells: jnp.ndarray) -> jnp.ndarray:
    """(B, 4) slots and cells -> (B, 4) env actions, played in order."""
    return slots * J.GRID_CELLS + cells


def play_round(state: J.EnvState, slots: jnp.ndarray, cells: jnp.ndarray):
    actions = plan_actions(slots, cells)
    total = jnp.zeros(state.grid.shape[0], jnp.float32)
    for k in range(J.PLAYS_PER_ROUND):
        state, reward, _ = J.step(state, actions[:, k])
        total = total + reward
    return state, total


def play_greedy(decks: np.ndarray, include_discard: bool = False, batch_size: int = 8) -> np.ndarray:
    """Final scores of the greedy agent on (N, 36) decks.

    include_discard=False matches the Python greedy exactly. True scores candidates with the
    round's discard added (a slightly better greedy, used by the search agent).
    """
    state = J.init_env_state_from_decks(jnp.asarray(decks))
    total = jnp.zeros(state.grid.shape[0], jnp.float32)
    for r in range(J.ROUNDS):
        slots, cells, _ = greedy_plans(state, r, include_discard, batch_size)
        state, reward = play_round(state, slots, cells)
        total = total + reward
    return np.asarray(total)
