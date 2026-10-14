"""Speed and parity benchmarks: original Python/Gymnasium env vs the JAX env.

Run from the repo root. Each part writes JSON to benchmarks/results/. JAX picks the device at
import time, so run CPU and GPU parts in separate processes (JAX_PLATFORMS=cpu for the CPU):

    python benchmarks/env_speed.py parity --games 1000
    python benchmarks/env_speed.py python-env
    JAX_PLATFORMS=cpu python benchmarks/env_speed.py jax-env
    python benchmarks/env_speed.py jax-env --batch-sizes 1 16 256 4096 65536 262144   # GPU
    JAX_PLATFORMS=cpu python benchmarks/env_speed.py python-greedy --games 100
    JAX_PLATFORMS=cpu python benchmarks/env_speed.py jax-greedy --games 100
    python benchmarks/env_speed.py jax-greedy --games 1024
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import platform
import random
import statistics
import sys
import time
from functools import partial

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pileup_poker.pileup_poker import DECK, PileupPokerEnv, flatten_observation  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
STEPS_PER_GAME = 16


def system_info() -> dict:
    info = {"python": platform.python_version(), "machine": platform.machine(), "cpu_count": os.cpu_count()}
    try:
        with open("/proc/cpuinfo") as f:
            info["cpu"] = next(line.split(":", 1)[1].strip() for line in f if line.startswith("model name"))
    except (OSError, StopIteration):
        info["cpu"] = platform.processor()
    if "jax" in sys.modules:
        import jax
        import jaxlib

        info.update(jax=jax.__version__, jaxlib=jaxlib.__version__, backend=jax.default_backend(),
                    devices=[d.device_kind for d in jax.devices()])
    return info


def save(name: str, payload: dict) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    payload = {"system": system_info(), **payload}
    path = os.path.join(RESULTS_DIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {path}")


def python_env_with_deck(deck_ids: np.ndarray) -> tuple[PileupPokerEnv, list[int]]:
    """Gym env set to a deck; returns the env and its full 36-card draw order as card ids."""
    env = PileupPokerEnv(enforce_valid_actions=True)
    env.reset()
    env.set_deck([DECK[c] for c in deck_ids[:20]])
    return env, [DECK.index(c) for c in env.deck]


# ---------------------------------------------------------------- parity


def parity(args) -> None:
    import jax.numpy as jnp

    from pileup_poker import jax_env as J

    rng = np.random.default_rng(args.seed)
    n = args.games
    decks = np.zeros((n, J.DECK_SIZE), np.int32)
    actions = np.zeros((n, STEPS_PER_GAME), np.int32)
    py_obs = np.zeros((n, STEPS_PER_GAME + 1, J.OBS_SIZE), np.float32)
    py_mask = np.zeros((n, STEPS_PER_GAME + 1, J.ACTION_SIZE), bool)
    py_reward = np.zeros((n, STEPS_PER_GAME), np.float32)
    py_done = np.zeros((n, STEPS_PER_GAME), bool)
    num_invalid = 0
    for g in range(n):
        env, full_deck = python_env_with_deck(rng.permutation(J.DECK_SIZE))
        decks[g] = full_deck
        obs = env._get_observation()
        for t in range(STEPS_PER_GAME):
            mask = np.asarray(env.get_action_mask(), bool)
            py_obs[g, t], py_mask[g, t] = flatten_observation(obs), mask
            # Some illegal actions (in [0, 80)) to exercise the auto-correction of both envs.
            if rng.random() < args.invalid_prob and not mask.all():
                a = int(rng.choice(np.flatnonzero(~mask)))
                num_invalid += 1
            else:
                a = int(rng.choice(np.flatnonzero(mask)))
            actions[g, t] = a
            obs, reward, done, _, _ = env.step(a)
            py_reward[g, t], py_done[g, t] = reward, done
        py_obs[g, -1], py_mask[g, -1] = flatten_observation(obs), np.asarray(env.get_action_mask(), bool)

    state = J.init_env_state_from_decks(jnp.asarray(decks))
    mismatch = {"obs": 0, "mask": 0, "reward": 0, "done": 0, "final_score": 0}
    max_abs_reward_diff = 0.0
    final_scores = None

    def check(t):
        mismatch["obs"] += int(np.sum(np.any(np.asarray(J.build_observation(state)) != py_obs[:, t], axis=1)))
        mismatch["mask"] += int(np.sum(np.any(np.asarray(J.get_action_mask(state)) != py_mask[:, t], axis=1)))

    check(0)
    for t in range(STEPS_PER_GAME):
        state, reward, done = J.step(state, jnp.asarray(actions[:, t]))
        done = np.asarray(done)
        # The Gym env returns 1 + current board score (with partial lines) for non-terminal moves;
        # the JAX env returns 0 there. Terminal moves return the final score in both.
        board = np.asarray(J.calculate_total_score(state.grid, state.discard))
        expected_py = np.where(done, np.asarray(reward), 1.0 + board)
        diff = np.abs(expected_py - py_reward[:, t])
        max_abs_reward_diff = max(max_abs_reward_diff, float(diff.max()))
        mismatch["reward"] += int(np.sum(diff > 0))
        mismatch["done"] += int(np.sum(done != py_done[:, t]))
        check(t + 1)
        if t == STEPS_PER_GAME - 1:
            final_scores = np.asarray(reward)
            mismatch["final_score"] = int(np.sum(final_scores != py_reward[:, t]))

    result = {
        "games": n,
        "steps": n * STEPS_PER_GAME,
        "invalid_actions": num_invalid,
        "invalid_prob": args.invalid_prob,
        "seed": args.seed,
        "mismatches": mismatch,
        "max_abs_reward_diff": max_abs_reward_diff,
        "all_match": all(v == 0 for v in mismatch.values()),
        "mean_final_score": float(final_scores.mean()),
        "comparisons": "observation (796-dim) and action mask before every step and after the last; "
        "per-step reward (Gym non-terminal reward vs 1 + JAX board score); done flags; final score",
    }
    print(json.dumps(result, indent=2))
    save("parity", result)


# ---------------------------------------------------------------- Python env throughput


def _python_games(seconds: float, seed: int) -> tuple[int, float]:
    """Full games with random legal actions for about `seconds`; returns (games, elapsed)."""
    rng = random.Random(seed)
    env = PileupPokerEnv(enforce_valid_actions=True)
    games = 0
    start = time.perf_counter()
    while time.perf_counter() - start < seconds:
        obs, _ = env.reset()
        flatten_observation(obs)
        done = False
        while not done:
            mask = env.get_action_mask()
            action = rng.choice([i for i, ok in enumerate(mask) if ok])
            obs, _, done, _, _ = env.step(action)
            flatten_observation(obs)
        games += 1
    return games, time.perf_counter() - start


def python_env(args) -> None:
    runs = []
    for rep in range(args.reps):
        games, elapsed = _python_games(args.seconds, rep)
        runs.append(games / elapsed)
        print(f"single process rep {rep}: {games / elapsed:.1f} games/s")
    result = {"single_process": {"games_per_s": runs, "median_games_per_s": statistics.median(runs)}}
    if args.processes > 1:
        multi = []
        with mp.Pool(args.processes) as pool:
            for rep in range(args.reps):
                out = pool.starmap(_python_games, [(args.seconds, 1000 * rep + i) for i in range(args.processes)])
                multi.append(sum(g / e for g, e in out))
                print(f"{args.processes} processes rep {rep}: {multi[-1]:.1f} games/s")
        result["multi_process"] = {"processes": args.processes, "games_per_s": multi,
                                   "median_games_per_s": statistics.median(multi)}
    result["per_step_work"] = "get_action_mask, random legal action, step (builds the Dict observation), flatten_observation"
    save("python_env", result)


# ---------------------------------------------------------------- JAX env throughput


def jax_env_bench(args) -> None:
    import jax
    import jax.numpy as jnp

    from pileup_poker import jax_env as J

    def play(key, batch):
        """batch full games with random legal actions; one lax.scan over the 16 steps."""
        key, deck_key = jax.random.split(key)
        state = J.init_env_state(jax.random.split(deck_key, batch))

        def body(carry, step_key):
            state, total, checksum = carry
            mask = J.get_action_mask(state)
            obs = J.build_observation(state)
            action = jax.random.categorical(step_key, jnp.where(mask, 0.0, -jnp.inf))
            state, reward, _ = J.step(state, action)
            # Use the observation so XLA cannot drop it.
            return (state, total + reward, checksum + obs.sum()), None

        init = (state, jnp.zeros(batch), jnp.zeros(()))
        (state, total, checksum), _ = jax.lax.scan(body, init, jax.random.split(key, STEPS_PER_GAME))
        return total, checksum

    def loop_play(key, batch, step_fn):
        """Same games, but one jitted call per step from a Python loop (as in train_jax_ppo.py)."""
        key, deck_key = jax.random.split(key)
        state = J.init_env_state(jax.random.split(deck_key, batch))
        total = jnp.zeros(batch)
        for step_key in jax.random.split(key, STEPS_PER_GAME):
            state, reward = step_fn(state, step_key)
            total = total + reward
        return total

    @jax.jit
    def loop_step(state, step_key):
        mask = J.get_action_mask(state)
        obs = J.build_observation(state)
        action = jax.random.categorical(step_key, jnp.where(mask, 0.0, -jnp.inf))
        state, reward, _ = J.step(state, action)
        return state, reward + 0.0 * obs.sum()

    backend = jax.default_backend()
    results = []
    for mode in args.modes:
        for batch in args.batch_sizes:
            key = jax.random.PRNGKey(0)
            if mode == "scan":
                t0 = time.perf_counter()
                fn = jax.jit(partial(play, batch=batch)).lower(key).compile()
                compile_s = time.perf_counter() - t0
                run = lambda k: fn(k)  # noqa: E731
            else:
                t0 = time.perf_counter()
                jax.block_until_ready(loop_play(key, batch, loop_step))
                compile_s = time.perf_counter() - t0
                run = lambda k: loop_play(k, batch, loop_step)  # noqa: E731
            jax.block_until_ready(run(jax.random.PRNGKey(1)))  # warm-up
            # Calls per repetition so that one repetition takes about args.seconds.
            t0 = time.perf_counter()
            jax.block_until_ready(run(jax.random.PRNGKey(2)))
            one = time.perf_counter() - t0
            calls = max(1, int(args.seconds / max(one, 1e-6)))
            rates = []
            for rep in range(args.reps):
                t0 = time.perf_counter()
                for c in range(calls):
                    out = run(jax.random.PRNGKey(100 * rep + c))
                jax.block_until_ready(out)
                rates.append(calls * batch / (time.perf_counter() - t0))
            row = {"backend": backend, "mode": mode, "batch": batch, "compile_s": compile_s,
                   "calls_per_rep": calls, "games_per_s": rates, "median_games_per_s": statistics.median(rates),
                   "median_steps_per_s": STEPS_PER_GAME * statistics.median(rates)}
            print(f"{backend} {mode} batch={batch}: {row['median_games_per_s']:.0f} games/s "
                  f"(compile {compile_s:.2f} s)")
            results.append(row)
    save(f"jax_env_{backend}", {"results": results,
                                "per_step_work": "get_action_mask, build_observation, random legal action, step"})


# ---------------------------------------------------------------- greedy


def greedy_decks(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.array([python_env_with_deck(rng.permutation(36))[1] for _ in range(n)], np.int32)


def python_greedy(args) -> None:
    from pileup_poker.greedy_agent import GreedyPileupPokerAgent

    decks = greedy_decks(args.games, args.seed)
    scores, times = [], []
    for g, deck in enumerate(decks):
        env, _ = python_env_with_deck(deck)
        agent = GreedyPileupPokerAgent(env)
        t0 = time.perf_counter()
        done = False
        while not done:  # GreedyPileupPokerAgent.play without rendering
            placement, _ = agent.select_best_placement(None)
            for card, (row, col) in zip(*placement):
                action = env.hand.index(card) * 16 + row * 4 + col
                _, reward, done, _, _ = env.step(action)
        times.append(time.perf_counter() - t0)
        scores.append(float(reward))
        print(f"game {g}: score {reward} in {times[-1]:.2f} s")
    save("python_greedy", {"games": args.games, "seed": args.seed, "scores": scores, "seconds_per_game": times,
                           "games_per_s": len(times) / sum(times)})


def jax_greedy(args) -> None:
    import jax

    from pileup_poker.jax_greedy import play_greedy

    decks = greedy_decks(args.games, args.seed)
    t0 = time.perf_counter()
    scores = play_greedy(decks)
    first_s = time.perf_counter() - t0
    rates = []
    for rep in range(args.reps):
        t0 = time.perf_counter()
        scores = play_greedy(decks)  # returns a numpy array, so the work is finished
        rates.append(len(decks) / (time.perf_counter() - t0))
        print(f"rep {rep}: {rates[-1]:.1f} games/s")
    match = None
    ref_path = os.path.join(RESULTS_DIR, "python_greedy.json")
    if os.path.exists(ref_path):
        with open(ref_path) as f:
            ref = json.load(f)
        if ref["seed"] == args.seed:
            k = min(len(ref["scores"]), len(scores))
            match = {"games_compared": k, "identical": bool(np.array_equal(np.array(ref["scores"][:k]), scores[:k]))}
            print(f"scores vs python greedy: {match}")
    save(f"jax_greedy_{jax.default_backend()}", {
        "games": args.games, "seed": args.seed, "first_call_s_incl_compile": first_s, "games_per_s": rates,
        "median_games_per_s": statistics.median(rates), "mean_score": float(np.mean(scores)),
        "scores": scores.tolist(), "match_python_greedy": match})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="part", required=True)
    p = sub.add_parser("parity")
    p.add_argument("--games", type=int, default=1000)
    p.add_argument("--invalid-prob", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("python-env")
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--processes", type=int, default=os.cpu_count())
    p = sub.add_parser("jax-env")
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 16, 256, 4096, 65536])
    p.add_argument("--modes", nargs="+", default=["scan", "loop"], choices=["scan", "loop"])
    p.add_argument("--seconds", type=float, default=1.0, help="Target duration of one repetition.")
    p.add_argument("--reps", type=int, default=5)
    for name in ("python-greedy", "jax-greedy"):
        p = sub.add_parser(name)
        p.add_argument("--games", type=int, default=20)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--reps", type=int, default=3)
    args = parser.parse_args()
    {"parity": parity, "python-env": python_env, "jax-env": jax_env_bench,
     "python-greedy": python_greedy, "jax-greedy": jax_greedy}[args.part](args)


if __name__ == "__main__":
    main()
