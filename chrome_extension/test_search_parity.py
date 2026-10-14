"""Parity cases for the extension's in-browser value search (chrome_extension/extension/search.js).

    JAX_PLATFORMS=cpu python -m chrome_extension.test_search_parity [--positions 100]   # writes /tmp/pp_search_cases.json
    node chrome_extension/test_search.js                              # compares the JS search with them

L1: features and V (both networks) of random afterstates (all placements of random positions,
    including end-of-round afterstates with an empty hand); beam-search children values, the distinct
    candidates, and leaf values for given next hands.
L3: the decision and its value at round starts with the next hands JAX samples (injected into the JS).
Positions are round starts of games played by the round-level greedy on decks of seed 123.
Run JAX on the CPU: GPU matrix multiplications (TF32 on recent NVIDIA GPUs) are not precise enough for these checks.
"""
import argparse
import json
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from compare_agents import make_decks
from pileup_poker import afterstate as A
from pileup_poker import jax_env as J
from pileup_poker import jax_greedy as G
from pileup_poker import value_search as VS

VALUE = "models/afterstate_mlp/afterstate_params.pkl"
INNER = "models/td_h128/afterstate_params.pkl"
R1 = ((16, 8), (8, 8), (4, 16), (2, 32))
R2 = ((8, 8), (4, 8), (2, 16))
HALVING = (R1, R2, R1)
OUT = "/tmp/pp_search_cases.json"


def round_start_states(decks):
    """One-game states at the start of every round of greedy games."""
    state = J.init_env_state_from_decks(jnp.asarray(decks))
    out = []
    for r in range(J.ROUNDS):
        out.append(state)
        slots, cells, _ = G.greedy_plans(state, r, True)
        state, _ = G.play_round(state, slots, cells)
    return out


def take(state, i):
    return jax.tree_util.tree_map(lambda x: x[i : i + 1], state)


def afterstate_cases(states, value, inner, rng, per_state=6):
    """Random legal afterstates of mid-round positions (random placements within each round)."""
    cases = []
    for r, state in enumerate(states):
        n = state.grid.shape[0]
        s = state
        for k in range(J.PLAYS_PER_ROUND):
            afterstates, mask = A.expand_afterstates(s)
            flat = jax.tree_util.tree_map(lambda x: x.reshape((n * J.ACTION_SIZE,) + x.shape[2:]), afterstates)
            feats = np.asarray(A.mlp_features(flat)).reshape(n, J.ACTION_SIZE, -1)
            vals = [np.asarray(A.value(m["model"], m["params"], flat, m["value_scale"])).reshape(n, J.ACTION_SIZE)
                    for m in (value, inner)]
            mask = np.asarray(mask)
            for i in range(n):
                for a in rng.choice(np.flatnonzero(mask[i]), size=min(per_state, mask[i].sum()), replace=False):
                    cases.append({
                        "grid": np.asarray(afterstates.grid[i, a]).tolist(),
                        "discard": np.asarray(afterstates.discard[i, a]).tolist(),
                        "hand": np.asarray(afterstates.hand[i, a]).tolist(),
                        "placed": int(afterstates.placed[i, a]),
                        "unseen": np.flatnonzero(np.asarray(afterstates.unseen[i, a])).tolist(),
                        "score": float(J.calculate_total_score(afterstates.grid[i, a][None], afterstates.discard[i, a][None])[0]),
                        "features": np.round(feats[i, a], 6).tolist(),
                        "value": float(vals[0][i, a]),
                        "inner": float(vals[1][i, a]),
                    })
            actions = np.array([rng.choice(np.flatnonzero(m)) for m in mask])
            s, _, _ = J.step(s, jnp.asarray(actions))
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--positions", type=int, default=100, help="games (4 round starts each)")
    args = parser.parse_args()
    value, inner = A.load_params(VALUE), A.load_params(INNER)
    vargs = (value["model"], value["params"], value["value_scale"])
    policy = (inner["model"], inner["params"], inner["value_scale"])
    decks = make_decks(args.positions, 123)
    states = round_start_states(decks)
    rng = np.random.default_rng(0)
    after = afterstate_cases(states, value, inner, rng)

    beam = jax.jit(partial(VS._beam_candidates, *vargs), static_argnames=("round_idx", "width"))
    distinct = jax.jit(VS._distinct_top, static_argnames=("num_candidates", "pool_size", "use_suit_symmetry"))
    leaves = jax.jit(partial(VS._card_leaf_values, *vargs, lookahead=1, exact_last=True, policy=policy),
                     static_argnames=("first_round",))
    searches = []
    for r, state in enumerate(states):
        keys = jax.vmap(lambda i: jax.random.fold_in(jax.random.PRNGKey(7), i))(jnp.arange(args.positions) + 1000 * r)
        slots, cells, vals, calls = VS.value_search_plans_with_calls(
            *vargs, state, r, keys, inner="card", outer="beam", beam_width=16, halving=HALVING[min(r, 2)],
            exact_last=True, inner_model=inner["model"], inner_params=(inner["params"], inner["value_scale"]))
        stages = HALVING[min(r, 2)]
        total = sum(s for _, s in stages)
        for i in range(args.positions):
            one = take(state, i)
            grid, discard, deck = one.grid[0], one.discard[0], one.deck[0]
            hand = deck[5 * r : 5 * r + 5]
            case = {"round": r, "grid": np.asarray(grid).tolist(), "discard": np.asarray(discard).tolist(),
                    "hand": np.asarray(hand).tolist(), "slots": np.asarray(slots[i]).tolist(),
                    "cells": np.asarray(cells[i]).tolist(), "value": float(vals[i]) * value["value_scale"],
                    "calls": np.asarray(calls[i]).tolist()}
            if r < J.ROUNDS - 1:
                d = jax.vmap(lambda k: J.determinize(one._replace(used=jnp.zeros_like(one.used)), k).deck[0])(
                    jax.random.split(keys[i], total))
                case["samples"] = np.asarray(d[:, 5 * (r + 1) : 5 * (r + 2)]).tolist()
                g, dc, sl, ce, v, _ = beam(grid, discard, hand, round_idx=r, width=16)
                order = np.argsort(-np.asarray(v), kind="stable")
                fin = order[np.isfinite(np.asarray(v)[order])]
                case["beam"] = [[float(v[j]), sorted((np.asarray(sl[j]) * 16 + np.asarray(ce[j])).tolist())] for j in fin[:24]]
                case["beam_count"] = int(len(fin))
                cand = distinct(v, g, dc, num_candidates=stages[0][0], pool_size=512, use_suit_symmetry=True)
                cand = np.asarray(cand)
                case["candidates"] = [sorted((np.asarray(sl[j]) * 16 + np.asarray(ce[j])).tolist()) for j in cand]
                lv, _ = leaves(jnp.repeat(g[cand[:2]], 8, axis=0), jnp.repeat(dc[cand[:2]], 8, axis=0),
                               jnp.tile(d[:8], (2, 1)), first_round=r + 1)
                case["leaves"] = np.asarray(lv).tolist()  # candidates 0, 1 x samples 0..7
            searches.append(case)
        print(f"round {r + 1}: {args.positions} searches", flush=True)
    with open(OUT, "w") as f:
        json.dump({"afterstates": after, "searches": searches}, f)
    print(f"wrote {OUT}: {len(after)} afterstates, {len(searches)} searches")


if __name__ == "__main__":
    main()
