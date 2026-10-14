"""Score several agents on the same decks.

Every agent gets identical card draws: deck i is a seeded permutation of the 36 cards,
and every agent plays its first 20 cards as the same 4 hands of 5 in the same order.

Agents are given as NAME or NAME=MODEL_PATH, e.g.
    python compare_agents.py --agents greedy search ppo=models/ppo_attn_bc/jax_ppo_params.pkl \
        afterstate=models/afterstate_mlp/afterstate_params.pkl alphazero=models/alphazero/alphazero_params.pkl
The first agent is the baseline for the paired differences.
"""
from __future__ import annotations

import argparse
import csv
import time

import jax
import jax.numpy as jnp
import numpy as np

from pileup_poker import jax_env

CARDS_DRAWN = jax_env.ROUNDS * jax_env.HAND_SIZE
NEEDS_MODEL = {"ppo", "afterstate", "alphazero", "search_value"}


def make_decks(num_decks: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.stack([rng.permutation(jax_env.DECK_SIZE) for _ in range(num_decks)]).astype(np.int32)


def random_scores(decks: np.ndarray, seed: int) -> np.ndarray:
    state = jax_env.init_env_state_from_decks(jnp.asarray(decks))
    key = jax.random.PRNGKey(seed)
    total = jnp.zeros(len(decks), jnp.float32)
    for _ in range(jax_env.ROUNDS * jax_env.PLAYS_PER_ROUND):
        key, k = jax.random.split(key)
        action = jax.random.categorical(k, jnp.where(jax_env.get_action_mask(state), 0.0, -jnp.inf))
        state, reward, _ = jax_env.step(state, action)
        total = total + reward
    return np.asarray(total)


def run_agent(name: str, model_path: str | None, decks: np.ndarray, args) -> np.ndarray:
    # Imports are local so each agent only loads what it needs.
    if name == "random":
        return random_scores(decks, args.seed)
    if name == "greedy":
        from pileup_poker.jax_greedy import play_greedy  # exact port of greedy_agent.py

        return play_greedy(decks)
    if name == "search":
        from pileup_poker.search_agent import evaluate_on_decks

        return evaluate_on_decks(
            None, decks, num_candidates=args.search_candidates, num_samples=args.search_samples, seed=args.seed
        )
    if name == "search_value":
        from pileup_poker.value_search import evaluate_on_decks

        return evaluate_on_decks(
            model_path,
            decks,
            num_candidates=args.sv_candidates,
            num_samples=args.sv_samples,
            lookahead=args.sv_lookahead,
            seed=args.seed,
        )
    if name == "ppo":
        from train_jax_ppo import evaluate_on_decks

        return evaluate_on_decks(model_path, decks)
    if name == "afterstate":
        from pileup_poker.afterstate import evaluate_on_decks

        return evaluate_on_decks(model_path, decks)
    if name == "alphazero":
        from pileup_poker.alphazero import evaluate_on_decks

        return evaluate_on_decks(model_path, decks, num_simulations=args.az_simulations, seed=args.seed)
    raise ValueError(f"unknown agent {name!r}")


def parse_agent(spec: str) -> tuple[str, str, str | None]:
    """'name' or 'name=path' -> (label, name, path)."""
    name, _, path = spec.partition("=")
    if name in NEEDS_MODEL and not path:
        raise ValueError(f"agent {name!r} needs a model: {name}=path/to/params.pkl")
    return spec if path else name, name, path or None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--agents", nargs="+", default=["greedy", "random"], help="NAME or NAME=MODEL_PATH")
    parser.add_argument("--num-decks", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0, help="Deck seed (also seeds agents that sample).")
    parser.add_argument("--search-candidates", type=int, default=16)
    parser.add_argument("--search-samples", type=int, default=16)
    parser.add_argument("--az-simulations", type=int, default=64)
    parser.add_argument("--sv-candidates", type=int, default=8, help="search_value: candidates per round (K).")
    parser.add_argument("--sv-samples", type=int, default=8, help="search_value: next hands per candidate (0 = V-argmax).")
    parser.add_argument("--sv-lookahead", type=int, default=1, help="search_value: rounds played before the V leaf.")
    parser.add_argument("--out-csv", type=str, default="", help="Optional per-deck scores file.")
    args = parser.parse_args()

    agents = [parse_agent(spec) for spec in args.agents]
    decks = make_decks(args.num_decks, args.seed)
    results = {}
    for label, name, path in agents:
        start = time.time()
        results[label] = np.asarray(run_agent(name, path, decks, args), dtype=np.float64)
        print(f"{label}: done in {time.time() - start:.1f}s")

    n = len(decks)
    base_label = agents[0][0]
    base = results[base_label]
    print(f"\n{n} decks (seed {args.seed}); paired differences are vs {base_label}")
    print(f"{'agent':40s} {'mean':>8s} {'SE':>6s} {'median':>7s} {'max':>7s} {'diff':>8s} {'diff SE':>8s} {'W/T/L':>12s}")
    for label, scores in results.items():
        diff = scores - base
        print(
            f"{label[:40]:40s} {scores.mean():8.1f} {scores.std(ddof=1) / np.sqrt(n):6.1f} "
            f"{np.median(scores):7.0f} {scores.max():7.0f} {diff.mean():8.1f} {diff.std(ddof=1) / np.sqrt(n):8.1f} "
            f"{f'{(diff > 0).sum()}/{(diff == 0).sum()}/{(diff < 0).sum()}':>12s}"
        )

    if args.out_csv:
        from pileup_poker.pileup_poker import DECK

        with open(args.out_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["deck", *results])
            for i, deck in enumerate(decks):
                writer.writerow([" ".join(DECK[c] for c in deck[:CARDS_DRAWN]), *(results[k][i] for k in results)])
        print(f"Wrote per-deck scores to {args.out_csv}")


if __name__ == "__main__":
    main()
