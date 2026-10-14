"""Round-level search with the afterstate value model (no rollouts with the greedy agent).

A round decision = which hand card to discard and where the other 4 go. The end-of-round
afterstate of a decision is the board, the discard pile with this round's discard, an empty
hand, 4 * (round + 1) cards placed and the unseen cards. This is exactly the afterstate that
`afterstate.py` builds (and trains on) after the 4th placement of a round. In the last round
V is the exact final score.

Each round, for every game:
1. Score every round decision by V(end-of-round afterstate).
2. `num_samples == 0`: play the decision with the best V (round-level argmax).
3. Otherwise (expectimax): take the best `pool_size` decisions by V, drop symmetric copies
   (`search_agent.canonical_hash`) and keep the top `num_candidates`. Deal each candidate
   `num_samples` next hands from `jax_env.determinize` copies (the same copies for every
   candidate of a game). On each copy play the next `lookahead` rounds by round-level V-argmax
   and take V of the last afterstate (exact once the game is over). Play the candidate with
   the best mean. The round before the last one is exact expectimax.
   `inner="card"` plays the lookahead rounds card by card like the afterstate agent instead
   (much cheaper, so more samples fit in the same time).

Cheaper options (default off; see "Making value search cheap" in docs/value-search.md):
- `outer="beam"`: find the candidates by a beam search over the 4 placements of the round
  (`beam_width` entries, scored by V of the partial afterstates) instead of scoring every decision.
- `round1_symmetry`: in round 1 score only one decision per board-symmetry orbit.
- `halving`: successive halving, ((n1, s1), (n2, s2), ...): the best n_i candidates so far get s_i
  more copies each (common random numbers); replaces num_candidates/num_samples.
- `exact_last`: with `inner="card"`, play the last round by its exact best decision (no V calls).
- `inner_model`/`inner_params`: a smaller value model plays the lookahead rounds; the main model
  only scores the leaf.

Like the other search agents, it never reads the real order of undealt cards.
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from pileup_poker import afterstate as A
from pileup_poker import jax_env as J
from pileup_poker import jax_greedy as G
from pileup_poker.search_agent import canonical_hash
from pileup_poker.symmetry import CELL_PERMS, inverse_perms


def _round1_orbit_reps() -> np.ndarray:
    """Indices of round-1 decisions, one per orbit of the 32 board symmetries.

    On the empty board a symmetry maps every round-1 decision to another one with the same score,
    so ranking one decision per orbit is enough (about 1/32 of the 218,400 decisions).
    """
    slots, positions, discard_slots = G.ROUND_CANDIDATES[0]
    cell_of_slot = np.full((len(slots), J.HAND_SIZE), J.GRID_CELLS, np.int64)  # GRID_CELLS = discarded
    np.put_along_axis(cell_of_slot, slots, positions, axis=1)
    moved = np.concatenate([inverse_perms(CELL_PERMS), [[J.GRID_CELLS]] * len(CELL_PERMS)], axis=1)[:, cell_of_slot]
    keys = (moved * (J.GRID_CELLS + 1) ** np.arange(J.HAND_SIZE)).sum(-1).min(0)
    return np.sort(np.unique(keys, return_index=True)[1]).astype(np.int32)


ROUND1_REPS = _round1_orbit_reps()


def _padded_candidates(round_idx: int, chunk: int, idx=None):
    """Round candidates (all, or the subset `idx`) padded to a multiple of `chunk`, reshaped to chunks."""
    slots, positions, discard_slots = G.ROUND_CANDIDATES[round_idx]
    if idx is None:
        idx = np.arange(len(slots))
    n = len(idx)
    chunk = min(chunk, n)
    pad = np.arange(-(-n // chunk) * chunk)
    pad = idx[np.where(pad < n, pad, 0)]
    return n, [x[pad].reshape((-1, chunk) + x.shape[1:]) for x in (slots, positions, discard_slots)]


def round_values(model: str, params, value_scale: float, grid, discard, hand, round_idx: int, chunk: int = 16384,
                 idx=None):
    """V of the end-of-round afterstate of every round-`round_idx` decision for one game.

    grid (16,), discard (4,), hand (5,) at the start of the round. Returns values (N,) in
    `G.ROUND_CANDIDATES[round_idx]` order, or for the decisions `idx` (numpy indices) only.
    """
    n, (slots, positions, discard_slots) = _padded_candidates(round_idx, chunk, idx)
    empty_cells = jnp.nonzero(grid == -1, size=J.GRID_CELLS - J.PLAYS_PER_ROUND * round_idx)[0].astype(jnp.int32)
    unseen = ~(A._multi_hot(grid) | A._multi_hot(discard) | A._multi_hot(hand))

    def chunk_values(xs):
        s, p, d = xs
        m = s.shape[0]
        grids = jnp.broadcast_to(grid, (m, J.GRID_CELLS)).at[jnp.arange(m)[:, None], empty_cells[p]].set(hand[s])
        discards = jnp.broadcast_to(discard, (m, J.ROUNDS)).at[:, round_idx].set(hand[d])
        a = A.Afterstate(
            grid=grids,
            discard=discards,
            hand=jnp.full((m, J.HAND_SIZE), -1, jnp.int32),
            placed=jnp.full((m,), J.PLAYS_PER_ROUND * (round_idx + 1), jnp.int32),
            unseen=jnp.broadcast_to(unseen, (m, J.DECK_SIZE)),
        )
        return A.value(model, params, a, value_scale)

    return jax.lax.map(chunk_values, (slots, positions, discard_slots)).reshape(-1)[:n]


def apply_decision(grid, discard, hand, round_idx: int, i):
    """Board and discard after round decision i (one game); also its slots and cells."""
    slots, positions, discard_slots = (jnp.asarray(x)[i] for x in G.ROUND_CANDIDATES[round_idx])
    empty_cells = jnp.nonzero(grid == -1, size=J.GRID_CELLS - J.PLAYS_PER_ROUND * round_idx)[0].astype(jnp.int32)
    cells = empty_cells[positions]
    return grid.at[cells].set(hand[slots]), discard.at[round_idx].set(hand[discard_slots]), slots, cells


def _hand(deck, round_idx: int):
    return deck[J.HAND_SIZE * round_idx : J.HAND_SIZE * (round_idx + 1)]


def _leaf_value(model, params, value_scale, grid, discard, deck, first_round: int, lookahead: int, chunk: int):
    """Play rounds first_round.. (at most `lookahead`) by V-argmax on one determinized copy; V of the last afterstate."""
    best = jnp.zeros((), jnp.float32)
    for r in range(first_round, min(first_round + lookahead, J.ROUNDS)):
        hand = _hand(deck, r)
        values = round_values(model, params, value_scale, grid, discard, hand, r, chunk)
        i = jnp.argmax(values)
        best = values[i]
        grid, discard, _, _ = apply_decision(grid, discard, hand, r, i)
    return best


def _exact_last_round_values(grids, discards, decks, value_scale: float):
    """Best final score / value_scale over the 120 last-round decisions (no value-model calls)."""
    r = J.ROUNDS - 1
    scores = jax.vmap(lambda g, d, deck: G.candidate_scores(g, d, _hand(deck, r), r, True)[0])(grids, discards, decks)
    return scores.max(axis=1) / value_scale


def _card_leaf_values(model, params, value_scale, grids, discards, decks, first_round: int, lookahead: int,
                      exact_last: bool = False, policy=None):
    """Like `_leaf_value` for a batch of copies, but each round is played card by card (the
    afterstate agent's policy: 4 x 80 afterstates per round instead of every round decision).
    exact_last: the last round is played by its exact best decision instead.
    policy: None, or (model, params, value_scale) of another (cheaper) value model that chooses the
    moves; the leaf is then one call of the main model on the last afterstate.

    Returns the leaf values (n,) and the number of calls (2,) of the main and the policy model
    (legal, not final afterstates).
    """
    n = grids.shape[0]
    state = J.EnvState(
        deck=decks,
        grid=grids,
        used=A._multi_hot(grids),
        current_round=jnp.full((n,), first_round, jnp.int32),
        num_moves=jnp.zeros((n,), jnp.int32),
        discard=discards,
        done=jnp.zeros((n,), bool),
    )
    best = jnp.zeros((n,), jnp.float32)
    calls = jnp.zeros((2,), jnp.int32)
    counter = 0 if policy is None else 1
    policy = policy or (model, params, value_scale)
    for r in range(first_round, min(first_round + lookahead, J.ROUNDS)):
        if exact_last and r == J.ROUNDS - 1:
            return _exact_last_round_values(state.grid, state.discard, decks, value_scale), calls
        for _ in range(J.PLAYS_PER_ROUND):
            afterstates, mask = A.expand_afterstates(state)
            flat = jax.tree_util.tree_map(lambda x: x.reshape((n * J.ACTION_SIZE,) + x.shape[2:]), afterstates)
            v = jnp.where(mask, A.value(*policy[:2], flat, policy[2]).reshape(n, J.ACTION_SIZE), -jnp.inf)
            calls = calls.at[counter].add(jnp.sum(mask & (afterstates.placed < A.NUM_STEPS)))
            action = jnp.argmax(v, axis=1)
            best = jnp.max(v, axis=1)
            last = jax.tree_util.tree_map(lambda x: x[jnp.arange(n), action], afterstates)
            state, _, _ = J.step(state, action)
    if counter:
        best = A.value(model, params, last, value_scale)
        calls = calls.at[0].add(jnp.sum(last.placed < A.NUM_STEPS))
    return best, calls


_ACTION_SLOT = np.arange(J.ACTION_SIZE) // J.GRID_CELLS
_ACTION_CELL = np.arange(J.ACTION_SIZE) % J.GRID_CELLS
_ORBIT_CELLS = np.isin(np.arange(J.GRID_CELLS), [0, 1, 5])  # one corner, edge and inner cell


def _beam_candidates(model, params, value_scale, grid, discard, hand, round_idx: int, width: int):
    """Round decisions found by beam search over the 4 placements of the round (one game).

    Each step expands every beam entry by every legal (hand card, empty cell), drops afterstates
    already reached in another order, scores them with V and keeps the best `width`. On the empty
    board the first card only goes to one cell per symmetry class (a symmetric copy of every
    decision is still reachable). Returns the children of the last step: grids (P, 16), discards
    (P, 4), slots (P, 4), cells (P, 4), values (P,) (-inf for illegal or repeated ones) and the
    number of value-model calls.
    """
    unseen = ~(A._multi_hot(grid) | A._multi_hot(discard) | A._multi_hot(hand))
    where = jnp.full((1, J.HAND_SIZE), -1, jnp.int32)  # cell of each hand card, -1 = still in hand
    valid = jnp.ones((1,), bool)
    calls = jnp.zeros((), jnp.int32)
    slot_hot = _ACTION_SLOT[:, None] == np.arange(J.HAND_SIZE)  # (80, 5)
    for t in range(J.PLAYS_PER_ROUND):
        w = where.shape[0]
        grids = jnp.broadcast_to(grid, (w, J.GRID_CELLS))
        for s in range(J.HAND_SIZE):
            grids = jnp.where(where[:, s : s + 1] == jnp.arange(J.GRID_CELLS), hand[s], grids)
        legal = valid[:, None] & (where[:, _ACTION_SLOT] < 0) & (grids[:, _ACTION_CELL] == -1)  # (w, 80)
        if round_idx == 0 and t == 0:
            legal &= _ORBIT_CELLS[_ACTION_CELL]
        child = jnp.where(slot_hot, _ACTION_CELL[:, None], where[:, None, :]).reshape(-1, J.HAND_SIZE)
        legal = legal.reshape(-1)
        # Same cells for the same hand cards = same afterstate: keep the first one.
        key = jnp.where(legal, ((child + 1) * (J.GRID_CELLS + 1) ** np.arange(J.HAND_SIZE)).sum(-1), -1)
        order = jnp.argsort(key, stable=True)
        sorted_key = key[order]
        repeat = jnp.zeros_like(legal).at[order[1:]].set(sorted_key[1:] == sorted_key[:-1])
        new = legal & ~repeat

        child_grids = jnp.repeat(grids, J.ACTION_SIZE, axis=0).at[
            jnp.arange(w * J.ACTION_SIZE), jnp.tile(_ACTION_CELL, w)].set(hand[jnp.tile(_ACTION_SLOT, w)])
        hand_left = jnp.where(child >= 0, -1, hand)
        child_discards = jnp.broadcast_to(discard, (w * J.ACTION_SIZE, J.ROUNDS))
        if t == J.PLAYS_PER_ROUND - 1:
            child_discards = child_discards.at[:, round_idx].set(hand_left.max(-1))
            hand_left = jnp.full_like(hand_left, -1)
        a = A.Afterstate(
            grid=child_grids,
            discard=child_discards,
            hand=hand_left,
            placed=jnp.full((w * J.ACTION_SIZE,), J.PLAYS_PER_ROUND * round_idx + t + 1, jnp.int32),
            unseen=jnp.broadcast_to(unseen, (w * J.ACTION_SIZE, J.DECK_SIZE)),
        )
        values = jnp.where(new, A.value(model, params, a, value_scale), -jnp.inf)
        if J.PLAYS_PER_ROUND * round_idx + t + 1 < A.NUM_STEPS:  # final boards have exact values
            calls += jnp.sum(new)
        if t < J.PLAYS_PER_ROUND - 1:
            top_values, top = jax.lax.top_k(values, min(width, values.shape[0]))
            where, valid = child[top], top_values > -jnp.inf
    slots = jnp.argsort(child < 0, axis=1, stable=True)[:, : J.PLAYS_PER_ROUND]
    return child_grids, child_discards, slots, jnp.take_along_axis(child, slots, axis=1), values, calls


def _beam_leaf_value(model, params, value_scale, grid, discard, deck, first_round: int, lookahead: int, width: int,
                     exact_last: bool):
    """Like `_leaf_value`, but each round is played by the best decision of a beam search of width
    `width` (width 1 = the afterstate agent's card-by-card policy). Returns the value and the calls."""
    best, calls = jnp.zeros((), jnp.float32), jnp.zeros((2,), jnp.int32)
    for r in range(first_round, min(first_round + lookahead, J.ROUNDS)):
        if exact_last and r == J.ROUNDS - 1:
            return _exact_last_round_values(grid[None], discard[None], deck[None], value_scale)[0], calls
        grids, discards, _, _, values, c = _beam_candidates(model, params, value_scale, grid, discard, _hand(deck, r),
                                                            r, width)
        i = jnp.argmax(values)
        best, grid, discard, calls = values[i], grids[i], discards[i], calls.at[0].add(c)
    return best, calls


def _distinct_top(values, grids, discards, num_candidates: int, pool_size: int, use_suit_symmetry: bool):
    """Indices (K,) of the best `num_candidates` decisions that are not symmetric copies of a better one."""
    pool_size = min(pool_size, values.shape[0])
    _, pool = jax.lax.top_k(values, pool_size)
    hashes = canonical_hash(grids[pool], discards[pool], use_suit_symmetry)
    same = jnp.all(hashes[:, None] == hashes[None, :], axis=-1)
    distinct = ~jnp.any(same & jnp.tril(jnp.ones((pool_size, pool_size), bool), k=-1), axis=1)
    order = jnp.argsort(~distinct, stable=True)[:num_candidates]
    order = jnp.where(distinct[order], order, 0)  # fewer distinct candidates than K: repeat the best
    return pool[order]


def _game_plan(model, params, value_scale, grid, discard, deck, key, round_idx, num_candidates, num_samples,
               lookahead, inner, pool_size, use_suit_symmetry, item_batch, chunk, outer, beam_width, round1_symmetry,
               halving, exact_last, inner_width, inner_model, inner_params):
    """Decision for one game at the start of round `round_idx` -> slots (4,), cells (4,), value (),
    calls (2,) of the value model and of the inner policy model.
    """
    hand = _hand(deck, round_idx)
    last = round_idx == J.ROUNDS - 1
    policy = None if inner_model is None else (inner_model, *inner_params)
    stages = halving or ((num_candidates, num_samples),)

    # Candidate decisions with their V: (values, grids, discards, slots, cells) or lazily from indices.
    if outer == "beam" and not last:
        grids, discards, slots, cells, values, calls = _beam_candidates(
            model, params, value_scale, grid, discard, hand, round_idx, beam_width)
        calls = jnp.array([calls, 0])
        decision = lambda i: (grids[i], discards[i], slots[i], cells[i])
    else:
        idx = ROUND1_REPS if round1_symmetry and round_idx == 0 else np.arange(len(G.ROUND_CANDIDATES[round_idx][0]))
        values = round_values(model, params, value_scale, grid, discard, hand, round_idx, chunk, idx)
        calls = jnp.array([0 if last else len(idx), 0], jnp.int32)  # last round: V is the exact final score
        decision = lambda i: apply_decision(grid, discard, hand, round_idx, jnp.asarray(idx)[i])
    if stages[0][1] == 0 or last:
        i = jnp.argmax(values)
        _, _, slots, cells = decision(i)
        return slots, cells, values[i], calls

    if outer == "beam" and not last:
        cand = _distinct_top(values, grids, discards, stages[0][0], pool_size, use_suit_symmetry)
    else:
        pool_size = min(pool_size, values.shape[0])
        _, pool = jax.lax.top_k(values, pool_size)
        pool_grids, pool_discards, _, _ = jax.vmap(decision)(pool)
        cand = pool[_distinct_top(values[pool], pool_grids, pool_discards, stages[0][0], pool_size, use_suit_symmetry)]
    cand_grids, cand_discards, cand_slots, cand_cells = jax.vmap(decision)(cand)  # (K, ...)

    # Determinized copies (shared by all candidates): only the undealt part of the deck changes.
    state = J.EnvState(
        deck=deck[None],
        grid=grid[None],
        used=jnp.zeros((1, J.DECK_SIZE), bool),
        current_round=jnp.full((1,), round_idx, jnp.int32),
        num_moves=jnp.zeros((1,), jnp.int32),
        discard=discard[None],
        done=jnp.zeros((1,), bool),
    )
    total_samples = sum(s for _, s in stages)
    decks = jax.vmap(lambda k: J.determinize(state, k).deck[0])(jax.random.split(key, total_samples))  # (S, 36)

    # Successive halving: each stage keeps the best n candidates so far and gives each s more copies.
    sums = jnp.zeros(cand.shape[0], jnp.float32)
    alive = jnp.arange(cand.shape[0])
    offset = 0
    for n, s in stages:
        if n < alive.shape[0]:
            alive = alive[jax.lax.top_k(sums[alive], n)[1]]
        items = (
            jnp.repeat(cand_grids[alive], s, axis=0),
            jnp.repeat(cand_discards[alive], s, axis=0),
            jnp.tile(decks[offset : offset + s], (n, 1)),
        )
        offset += s
        if inner == "card":
            leaves, item_calls = _card_leaf_values(model, params, value_scale, *items, first_round=round_idx + 1,
                                                   lookahead=lookahead, exact_last=exact_last, policy=policy)
        elif inner == "beam":
            leaf = partial(_beam_leaf_value, model, params, value_scale, first_round=round_idx + 1, lookahead=lookahead,
                           width=inner_width, exact_last=exact_last)
            leaves, item_calls = jax.vmap(leaf)(*items)
            item_calls = item_calls.sum(0)
        else:
            leaf = partial(_leaf_value, model, params, value_scale, first_round=round_idx + 1, lookahead=lookahead,
                           chunk=chunk)
            leaves = jax.lax.map(lambda x: leaf(*x), items, batch_size=item_batch)
            rounds = range(round_idx + 1, min(round_idx + 1 + lookahead, J.ROUNDS - 1))
            item_calls = jnp.array([n * s * sum(len(G.ROUND_CANDIDATES[r][0]) for r in rounds), 0])
        sums = sums.at[alive].add(leaves.reshape(n, s).sum(axis=1))
        calls += item_calls
    best = alive[jnp.argmax(sums[alive])]
    return cand_slots[best], cand_cells[best], sums[best] / total_samples, calls


_STATIC = ("model", "round_idx", "num_candidates", "num_samples", "lookahead", "inner", "pool_size", "use_suit_symmetry",
           "item_batch", "chunk", "outer", "beam_width", "round1_symmetry", "halving", "exact_last", "inner_width",
           "inner_model", "game_batch")


@partial(jax.jit, static_argnames=_STATIC)
def value_search_plans_with_calls(
    model: str,
    params,
    value_scale: float,
    state: J.EnvState,
    round_idx: int,
    keys: jax.Array,
    num_candidates: int = 8,
    num_samples: int = 0,
    lookahead: int = 1,
    inner: str = "round",
    pool_size: int = 512,
    use_suit_symmetry: bool = True,
    item_batch: int = 4,
    chunk: int = 16384,
    outer: str = "all",
    beam_width: int = 64,
    round1_symmetry: bool = False,
    halving: tuple | None = None,
    exact_last: bool = False,
    inner_width: int = 1,
    inner_model: str | None = None,
    inner_params=None,
    game_batch: int | None = None,
):
    """`value_search_plans` that also returns the number of model calls per game (B, 2): value model, and
    the inner policy model (`inner_model` with `inner_params` = (params, value_scale) plays the
    lookahead rounds card by card; the value model then only evaluates the leaf).

    Calls count the afterstates a sequential implementation would evaluate with the model:
    legal, not repeated, and not final (16 cards placed: V is the exact score).
    """
    plan = partial(
        _game_plan, model, params, value_scale,
        round_idx=round_idx, num_candidates=num_candidates, num_samples=num_samples, lookahead=lookahead,
        inner=inner, pool_size=pool_size, use_suit_symmetry=use_suit_symmetry, item_batch=item_batch, chunk=chunk,
        outer=outer, beam_width=beam_width, round1_symmetry=round1_symmetry, halving=halving, exact_last=exact_last,
        inner_width=inner_width, inner_model=inner_model, inner_params=inner_params,
    )
    return jax.lax.map(lambda x: plan(*x), (state.grid, state.discard, state.deck, keys), batch_size=game_batch)


def value_search_plans(model: str, params, value_scale: float, state: J.EnvState, round_idx: int, keys: jax.Array,
                       *args, **kwargs):
    """Decision for every game (all at the start of round `round_idx`); keys: one key per game.

    Options (after keys): num_candidates (8), num_samples (0), lookahead (1), inner ("round"),
    pool_size (512), use_suit_symmetry (True), item_batch (4), chunk (16384),
    and the cheaper-search options outer, beam_width, round1_symmetry, halving, exact_last,
    inner_width (inner="beam"), inner_model, inner_params (see the module docstring).
    Returns slots (B, 4), cells (B, 4) and the chosen decision's value (B,) in value units.
    """
    return value_search_plans_with_calls(model, params, value_scale, state, round_idx, keys, *args, **kwargs)[:3]


def evaluate_on_decks(
    model_path: str,
    decks: np.ndarray,
    num_candidates: int = 8,
    num_samples: int = 0,
    lookahead: int = 1,
    inner: str = "round",
    pool_size: int = 512,
    use_suit_symmetry: bool = True,
    seed: int = 0,
    game_batch: int = 32,
    item_batch: int = 4,
    chunk: int = 16384,
    return_calls: bool = False,
    **options,
):
    """Final scores (N,) on (N, 36) decks played exactly as dealt.

    num_samples=0 is the round-level V-argmax agent (deterministic). Results depend only on the
    decks, the options and `seed`; `game_batch`, `item_batch` and `chunk` only trade memory for speed.
    options: outer, beam_width, round1_symmetry, halving, exact_last, inner_width, halving_per_round
    (one halving schedule per round), inner_model_path (a cheaper model that plays the lookahead
    rounds), game_vmap (games evaluated in parallel inside a batch).
    return_calls: also return the model calls per deck and round (N, 4, 2): value model, inner model.
    """
    payload = A.load_params(model_path)
    options["game_batch"] = options.pop("game_vmap", None)
    per_round = options.pop("halving_per_round", None)
    inner_path = options.pop("inner_model_path", None)
    if inner_path:
        policy = A.load_params(inner_path)
        options.update(inner_model=policy["model"], inner_params=(policy["params"], policy["value_scale"]))
    decks = np.asarray(decks, dtype=np.int32)
    num_decks = len(decks)
    game_keys = jax.vmap(partial(jax.random.fold_in, jax.random.PRNGKey(seed)))(jnp.arange(num_decks))
    scores, calls = [], []
    for start in range(0, num_decks, game_batch):
        idx = np.minimum(np.arange(start, start + game_batch), num_decks - 1)  # pad the last batch
        state = J.init_env_state_from_decks(jnp.asarray(decks[idx]))
        total = jnp.zeros(game_batch, jnp.float32)
        round_calls = []
        for r in range(J.ROUNDS):
            round_keys = jax.vmap(jax.random.fold_in, in_axes=(0, None))(game_keys[idx], r)
            if per_round is not None:
                options["halving"] = per_round[min(r, len(per_round) - 1)]
            slots, cells, _, c = value_search_plans_with_calls(
                payload["model"], payload["params"], payload["value_scale"], state, r, round_keys,
                num_candidates, num_samples, lookahead, inner, pool_size, use_suit_symmetry, item_batch, chunk,
                **options,
            )
            state, reward = G.play_round(state, slots, cells)
            total = total + reward
            round_calls.append(c)
        keep = min(game_batch, num_decks - start)
        scores.append(np.asarray(total)[:keep])
        calls.append(np.stack([np.asarray(c) for c in round_calls], axis=1)[:keep])
    scores = np.concatenate(scores)
    return (scores, np.concatenate(calls)) if return_calls else scores
