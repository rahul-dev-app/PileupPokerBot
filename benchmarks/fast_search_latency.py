"""Latency of one value-search decision per round for a single game (batch of 1), after compilation.

A rough stand-in for a phone or browser: run it on the CPU with one thread, e.g.
    JAX_PLATFORMS=cpu XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \
        python -m benchmarks.fast_search_latency --configs b16_P1_h128inner --num-decks 20
Results are added to benchmarks/results/fast_search_latency.json under "<config>/<backend>".
"""
from __future__ import annotations

import argparse
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks.fast_search import CONFIGS
from compare_agents import make_decks
from pileup_poker import afterstate as A
from pileup_poker import jax_env as J
from pileup_poker import jax_greedy as G
from pileup_poker import value_search as VS

OUT = "benchmarks/results/fast_search_latency.json"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--configs", nargs="+", required=True)
    p.add_argument("--model", default="models/afterstate_mlp/afterstate_params.pkl")
    p.add_argument("--num-decks", type=int, default=20)
    args = p.parse_args()
    payload = A.load_params(args.model)
    decks = make_decks(args.num_decks, 0)
    backend = jax.default_backend()
    results = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for name in args.configs:
        options = dict(CONFIGS[name])
        per_round = options.pop("halving_per_round", None)
        inner_path = options.pop("inner_model_path", None)
        if inner_path:
            policy = A.load_params(inner_path)
            options.update(inner_model=policy["model"], inner_params=(policy["params"], policy["value_scale"]))
        times = np.zeros((args.num_decks + 1, J.ROUNDS))
        for i, deck in enumerate(np.concatenate([decks[:1], decks])):  # the first game compiles
            state = J.init_env_state_from_decks(jnp.asarray(deck[None]))
            for r in range(J.ROUNDS):
                if per_round:
                    options["halving"] = per_round[min(r, len(per_round) - 1)]
                start = time.time()
                slots, cells, _, _ = VS.value_search_plans_with_calls(
                    payload["model"], payload["params"], payload["value_scale"], state, r,
                    jax.random.PRNGKey(i)[None], **options)
                slots.block_until_ready()
                times[i, r] = time.time() - start
                state, _ = G.play_round(state, slots, cells)
        mean = times[1:].mean(0)
        print(f"{name} [{backend}]: seconds per decision by round {np.round(mean, 3).tolist()}, "
              f"per game {mean.sum():.3f}", flush=True)
        results[f"{name}/{backend}"] = dict(seconds_per_round=mean.tolist(), seconds_per_game=float(mean.sum()),
                                            num_decks=args.num_decks, xla_flags=os.environ.get("XLA_FLAGS", ""))
        with open(OUT, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
