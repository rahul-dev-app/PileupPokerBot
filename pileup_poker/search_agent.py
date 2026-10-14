"""Monte Carlo rollout search on top of the greedy agent (no learning).

Each round, for every game:
1. Score all round placements with the round's discard added (the true board after the round).
   Take the best `pool_size` by immediate score (random tie-break), drop placements whose
   board + discard is a symmetric copy (cells, optionally suits) of a higher-ranked one, and
   keep the top `num_candidates`.
2. Play every kept candidate on `num_samples` determinized copies of the game (undealt cards
   shuffled by `jax_env.determinize`) and finish each copy with the greedy agent
   (include_discard=True). All candidates of a game share the same copies (common random numbers).
3. Play the candidate with the best mean final score. The last round has no chance left, so it
   plays the exact best placement.

The search sees only the board, the discard pile, the current hand and the set of unseen cards;
the real deck order is used only by the real game to deal the next hand.
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from pileup_poker import jax_env as J
from pileup_poker import jax_greedy as G
from pileup_poker.symmetry import CELL_PERMS, SUIT_PERMS

# Two independent additive hash words; row 16 hashes discard cards. Index = card + 1 (0 = empty).
_HASH_TABLES = np.random.default_rng(0).integers(0, 2**32, size=(2, J.GRID_CELLS + 1, J.DECK_SIZE + 1), dtype=np.uint32)
_SUIT_PERMS_SHIFTED = np.concatenate([np.zeros((len(SUIT_PERMS), 1), np.int32), SUIT_PERMS + 1], axis=1)


def canonical_hash(grids: jnp.ndarray, discards: jnp.ndarray, use_suit_symmetry: bool) -> jnp.ndarray:
    """(N, 16) grids, (N, 4) discards -> (N, 2) uint32, equal for boards equivalent under symmetry.

    Each word is the minimum over all symmetries of an additive hash (the discard pile is hashed
    as a set). Different classes collide with probability about 2^-64.
    """
    suit_perms = jnp.asarray(_SUIT_PERMS_SHIFTED if use_suit_symmetry else _SUIT_PERMS_SHIFTED[:1])
    grid_cards = suit_perms[:, (grids + 1)[:, CELL_PERMS]]  # (suits, N, 32 cell perms, 16)
    discard_cards = suit_perms[:, discards + 1]  # (suits, N, 4)
    words = []
    for table in jnp.asarray(_HASH_TABLES):
        h = table[jnp.arange(J.GRID_CELLS), grid_cards].sum(-1, dtype=jnp.uint32)
        h = h + table[J.GRID_CELLS, discard_cards].sum(-1, dtype=jnp.uint32)[..., None]
        words.append(h.min(axis=(0, 2)))
    return jnp.stack(words, axis=-1)


def _top_candidates(grid, discard, hand, key, round_idx, num_candidates, pool_size, use_suit_symmetry):
    """One game at round start -> slots (K, 4), cells (K, 4) of the top distinct candidates.

    If fewer than K distinct candidates exist, the rest repeat the best one.
    """
    grids, discards, slots, cells = G.candidate_boards(grid, discard, hand, round_idx, include_discard=True)
    scores = J.calculate_total_score(grids, discards)
    pool_size = min(pool_size, scores.shape[0])
    # Scores are multiples of 5, so noise in [0, 1) only breaks ties.
    _, pool = jax.lax.top_k(scores + jax.random.uniform(key, scores.shape), pool_size)
    hashes = canonical_hash(grids[pool], discards[pool], use_suit_symmetry)
    same = jnp.all(hashes[:, None] == hashes[None, :], axis=-1)
    earlier = jnp.tril(jnp.ones((pool_size, pool_size), bool), k=-1)
    distinct = ~jnp.any(same & earlier, axis=1)
    # Stable sort: distinct candidates first, in pool (score) order.
    order = jnp.argsort(~distinct, stable=True)[:num_candidates]
    order = jnp.where(distinct[order], order, 0)
    keep = pool[order]
    return slots[keep], cells[keep]


def _greedy_finish(state: J.EnvState, first_round: int, batch_size: int) -> jnp.ndarray:
    """Play greedy (include_discard=True) from the start of `first_round` to the end; final scores."""
    for r in range(first_round, J.ROUNDS):
        slots, cells, _ = G.greedy_plans(state, r, True, batch_size)
        state, _ = G.play_round(state, slots, cells)
    return J.calculate_total_score(state.grid, state.discard)


def _repeat(tree, n: int, axis: int):
    return jax.tree_util.tree_map(lambda x: jnp.repeat(jnp.expand_dims(x, axis), n, axis=axis), tree)


@partial(
    jax.jit,
    static_argnames=("round_idx", "num_candidates", "num_samples", "pool_size", "use_suit_symmetry", "batch_size"),
)
def search_plans(
    state: J.EnvState,
    round_idx: int,
    keys: jax.Array,
    num_candidates: int = 16,
    num_samples: int = 16,
    pool_size: int = 512,
    use_suit_symmetry: bool = True,
    batch_size: int = 32,
):
    """Search decision for every game; all games must be at the start of round `round_idx`.

    keys: one PRNG key per game (B, 2). Returns slots (B, 4), cells (B, 4) and the mean rollout
    score of the chosen candidate (B,) (exact final score in the last round).
    """
    if round_idx == J.ROUNDS - 1:
        return G.greedy_plans(state, round_idx, True, batch_size)

    num_games, k, s = state.grid.shape[0], num_candidates, num_samples
    tie_keys, sample_keys = jnp.moveaxis(jax.vmap(jax.random.split)(keys), 1, 0)
    hands = G.hand_cards(state)
    slots, cells = jax.vmap(
        partial(
            _top_candidates,
            round_idx=round_idx,
            num_candidates=k,
            pool_size=pool_size,
            use_suit_symmetry=use_suit_symmetry,
        )
    )(state.grid, state.discard, hands, tie_keys)  # (B, K, 4)

    # S determinized copies per game, shared by all its candidates.
    copies = jax.vmap(J.determinize)(_repeat(state, s, 1), sample_keys)  # leaves (B, S, ...)
    copies = _repeat(copies, k, 1)  # (B, K, S, ...)
    flat = jax.tree_util.tree_map(lambda x: x.reshape((num_games * k * s,) + x.shape[3:]), copies)
    flat_slots = jnp.repeat(slots, s, axis=1).reshape(-1, J.PLAYS_PER_ROUND)
    flat_cells = jnp.repeat(cells, s, axis=1).reshape(-1, J.PLAYS_PER_ROUND)
    after, _ = G.play_round(flat, flat_slots, flat_cells)
    values = _greedy_finish(after, round_idx + 1, batch_size).reshape(num_games, k, s).mean(axis=2)

    best = jnp.argmax(values, axis=1)
    games = jnp.arange(num_games)
    return slots[games, best], cells[games, best], values[games, best]


def evaluate_on_decks(
    model_path: str | None,
    decks: np.ndarray,
    num_candidates: int = 16,
    num_samples: int = 16,
    pool_size: int = 512,
    use_suit_symmetry: bool = True,
    seed: int = 0,
    game_batch: int = 8,
    batch_size: int = 32,
) -> np.ndarray:
    """Final scores of the search agent on (N, 36) decks, played exactly as dealt.

    model_path is unused (no model). Results depend only on the decks, the options and `seed`
    (each game gets its own key, so `game_batch` does not change them).
    """
    del model_path
    decks = np.asarray(decks, dtype=np.int32)
    num_decks = len(decks)
    game_keys = jax.vmap(partial(jax.random.fold_in, jax.random.PRNGKey(seed)))(jnp.arange(num_decks))
    scores = []
    for start in range(0, num_decks, game_batch):
        # Pad the last batch to a full one to reuse the compiled search.
        idx = np.minimum(np.arange(start, start + game_batch), num_decks - 1)
        state = J.init_env_state_from_decks(jnp.asarray(decks[idx]))
        total = jnp.zeros(game_batch, jnp.float32)
        for r in range(J.ROUNDS):
            round_keys = jax.vmap(jax.random.fold_in, in_axes=(0, None))(game_keys[idx], r)
            slots, cells, _ = search_plans(
                state, r, round_keys, num_candidates, num_samples, pool_size, use_suit_symmetry, batch_size
            )
            state, reward = G.play_round(state, slots, cells)
            total = total + reward
        scores.append(np.asarray(total)[: min(game_batch, num_decks - start)])
    return np.concatenate(scores)
