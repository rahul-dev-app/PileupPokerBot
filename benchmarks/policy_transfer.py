"""Cross-backend policy transfer (level 4 of the verification): agents trained in the JAX env play the
original Python/Gymnasium env, on the same decks as in the JAX env.

In the Python env the agents only see what the Gym env gives them: the flattened observation and
get_action_mask(). All games run in lockstep, so the network is called once per step for all games.

    python benchmarks/policy_transfer.py --games 1000

Writes benchmarks/results/policy_transfer.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from pileup_poker import afterstate  # noqa: E402
from pileup_poker import jax_env as J  # noqa: E402
from pileup_poker.pileup_poker import flatten_observation  # noqa: E402

from benchmarks.env_speed import greedy_decks, python_env_with_deck, system_info  # noqa: E402
from train_jax_ppo import load_policy, policy_log_probs  # noqa: E402
from train_jax_ppo import evaluate_on_decks as ppo_on_decks  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "policy_transfer.json")
G, H, D = J.GRID_CELLS * J.DECK_SIZE, J.HAND_SIZE * J.DECK_SIZE, J.DECK_SIZE


def state_from_observation(obs: np.ndarray) -> J.EnvState:
    """EnvState that the afterstate agent needs, built only from flattened Gym observations (B, 796).

    The deck holds the current hand at positions 5r..5r+4; the other positions are never read
    (the afterstate agent treats all unseen cards as one set)."""
    b = obs.shape[0]
    grid_hot = obs[:, :G].reshape(b, J.GRID_CELLS, J.DECK_SIZE)
    grid = np.where(grid_hot.any(-1), grid_hot.argmax(-1), -1)
    hand = obs[:, G:G + H].reshape(b, J.HAND_SIZE, J.DECK_SIZE).argmax(-1)
    discard_hot = obs[:, G + H:G + H + D] > 0
    rnd = obs[:, G + H + D:].argmax(-1)
    deck = np.zeros((b, J.DECK_SIZE), np.int64)
    discard = np.full((b, J.ROUNDS), -1)
    for i in range(b):
        deck[i, 5 * rnd[i]:5 * rnd[i] + 5] = hand[i]
        d = np.flatnonzero(discard_hot[i])
        discard[i, :len(d)] = d
    used = np.zeros((b, J.DECK_SIZE), bool)
    rows, cells = np.nonzero(grid >= 0)
    used[rows, grid[rows, cells]] = True
    placed = (grid >= 0).sum(1)
    return J.EnvState(deck=jnp.asarray(deck, jnp.int32), grid=jnp.asarray(grid, jnp.int32), used=jnp.asarray(used),
                      current_round=jnp.asarray(rnd, jnp.int32), num_moves=jnp.asarray(placed - 4 * rnd, jnp.int32),
                      discard=jnp.asarray(discard, jnp.int32), done=jnp.zeros(b, bool))


def afterstate_policy(path: str):
    payload = afterstate.load_params(path)

    @jax.jit
    def act(state: J.EnvState, mask: jnp.ndarray) -> jnp.ndarray:
        b = state.grid.shape[0]
        afterstates, _ = afterstate.expand_afterstates(state)
        flat = jax.tree_util.tree_map(lambda x: x.reshape((b * J.ACTION_SIZE,) + x.shape[2:]), afterstates)
        v = afterstate.value(payload["model"], payload["params"], flat, payload["value_scale"]).reshape(b, J.ACTION_SIZE)
        return jnp.argmax(jnp.where(mask, v, -jnp.inf), axis=1)

    return lambda obs, mask: act(state_from_observation(obs), jnp.asarray(mask))


def ppo_policy(path: str):
    params, cfg = load_policy(path)
    act = jax.jit(lambda o, m: jnp.argmax(policy_log_probs(params, cfg, o, m)[0], axis=-1))
    return lambda obs, mask: act(jnp.asarray(obs), jnp.asarray(mask))


def play_python_env(policy, decks: np.ndarray) -> np.ndarray:
    envs = [python_env_with_deck(d)[0] for d in decks]
    obs = [env._get_observation() for env in envs]
    scores = np.zeros(len(envs))
    for _ in range(J.ROUNDS * J.PLAYS_PER_ROUND):
        flat = np.stack([flatten_observation(o) for o in obs])
        mask = np.stack([np.asarray(env.get_action_mask(), bool) for env in envs])
        actions = np.asarray(policy(flat, mask))
        for i, env in enumerate(envs):
            obs[i], reward, done, _, _ = env.step(int(actions[i]))
            scores[i] = reward  # the last reward of a game is its final score
    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ppo", default="models/ppo_cont_control/jax_ppo_params.pkl")
    parser.add_argument("--afterstate", default="models/afterstate_mlp/afterstate_params.pkl")
    args = parser.parse_args()
    decks = greedy_decks(args.games, args.seed)  # full draw order of the Gym env after set_deck

    agents = {
        f"ppo ({args.ppo})": (lambda: ppo_on_decks(args.ppo, decks), ppo_policy(args.ppo)),
        f"afterstate ({args.afterstate})": (lambda: afterstate.evaluate_on_decks(args.afterstate, decks),
                                            afterstate_policy(args.afterstate)),
    }
    rows = {}
    for name, (in_jax, policy) in agents.items():
        jax_scores = np.asarray(in_jax(), np.float64)
        t0 = time.perf_counter()
        py_scores = play_python_env(policy, decks)
        diff = py_scores - jax_scores
        rows[name] = {
            "games": len(decks), "jax_env_mean": jax_scores.mean(), "python_env_mean": py_scores.mean(),
            "se": float(jax_scores.std(ddof=1) / np.sqrt(len(decks))),
            "paired_diff_mean": diff.mean(), "games_with_different_score": int(np.sum(diff != 0)),
            "python_env_seconds": time.perf_counter() - t0,
        }
        print(name, json.dumps(rows[name]))
    with open(RESULTS, "w") as f:
        json.dump({"system": system_info(), "seed": args.seed, "agents": rows}, f, indent=2)
    print(f"wrote {RESULTS}")


if __name__ == "__main__":
    main()
