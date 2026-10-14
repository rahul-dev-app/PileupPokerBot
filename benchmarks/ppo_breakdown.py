"""Where PPO training time goes: environment vs policy inference vs PPO update, per iteration.

Re-creates the rollout/update loop of train_jax_ppo.py (MLP, joint head, no reward shaping, 64 steps,
4 epochs x 8 minibatches) with a timer around each phase. Each phase ends with block_until_ready,
so the split is exact but slightly slower than the real loop, which overlaps some work.

    python benchmarks/ppo_breakdown.py --backend gym --num-envs 64
    python benchmarks/ppo_breakdown.py --backend jax --num-envs 64
    python benchmarks/ppo_breakdown.py --backend jax --num-envs 1024

Results are added to benchmarks/results/ppo_breakdown.json under "<backend>_<platform>_<num_envs>".
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
import optax  # noqa: E402

import train_jax_ppo as T  # noqa: E402
from pileup_poker import jax_env, networks  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "ppo_breakdown.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["gym", "jax"], required=True)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--num-steps", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()
    cfg = T.PPOConfig(num_envs=args.num_envs, num_steps=args.num_steps, env_backend=args.backend, reward_shaping=False)
    model_cfg = cfg.model_config()
    key = jax.random.PRNGKey(0)
    key, reset_key, net_key = jax.random.split(key, 3)
    params = networks.init_params(net_key, model_cfg)
    optimizer = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), optax.adam(cfg.learning_rate))
    opt_state = optimizer.init(params)
    update_step = T.make_update_step(model_cfg, optimizer, cfg.clip_epsilon, cfg.value_coef, cfg.entropy_coef)
    if args.backend == "jax":
        state, obs, mask = jax_env.reset(reset_key, cfg.num_envs)
    else:
        envs, obs_np, mask_np = T._init_gym_backend(cfg.num_envs, 0)
        obs, mask = jnp.asarray(obs_np), jnp.asarray(mask_np)
    batch = cfg.num_envs * cfg.num_steps
    mb = batch // cfg.num_minibatches
    times = []
    for it in range(args.warmup + args.iterations):
        t = {"policy": 0.0, "env": 0.0, "update": 0.0}
        bufs = {k: [] for k in ("obs", "mask", "act", "logp", "val", "rew", "done")}
        for _ in range(cfg.num_steps):
            t0 = time.perf_counter()
            key, k = jax.random.split(key)
            act, logp, val = jax.block_until_ready(T.sample_actions(k, params, model_cfg, obs, mask))
            t1 = time.perf_counter()
            for name, x in zip(("obs", "mask", "act", "logp", "val"), (obs, mask, act, logp, val)):
                bufs[name].append(x)
            if args.backend == "jax":
                state, rew, done = jax_env.step(state, act)
                np.array(rew), np.array(done)  # host copies, as in train_jax_ppo.py (episode returns)
                key, k = jax.random.split(key)
                state = jax_env.reset_done(state, done, k)
                obs, mask = jax.block_until_ready((jax_env.build_observation(state), jax_env.get_action_mask(state)))
            else:
                obs_np, mask_np, rew_np, done_np = T._step_gym_backend(envs, np.asarray(act))
                rew, done, obs, mask = (jnp.asarray(x) for x in (rew_np, done_np, obs_np, mask_np))
                jax.block_until_ready((rew, done, obs, mask))
            t2 = time.perf_counter()
            bufs["rew"].append(rew / cfg.reward_scale)
            bufs["done"].append(done)
            t["policy"] += t1 - t0
            t["env"] += t2 - t1
        t0 = time.perf_counter()
        _, last_val = T.policy_log_probs(params, model_cfg, obs, mask)
        st = {k: jnp.stack(v) for k, v in bufs.items()}
        adv, ret = T.compute_gae(st["rew"], st["val"], st["done"], last_val, cfg.gamma, cfg.gae_lambda)
        flat = [st["obs"].reshape(batch, -1), st["mask"].reshape(batch, -1), st["act"].reshape(batch),
                st["logp"].reshape(batch), adv.reshape(batch), ret.reshape(batch)]
        flat[4] = (flat[4] - flat[4].mean()) / (flat[4].std() + 1e-8)
        for _ in range(cfg.update_epochs):
            key, k = jax.random.split(key)
            perm = jax.random.permutation(k, batch)
            for s in range(0, batch, mb):
                idx = perm[s:s + mb]
                params, opt_state, _, _ = update_step(params, opt_state, *(x[idx] for x in flat))
        jax.block_until_ready(params)
        t["update"] = time.perf_counter() - t0
        t["total"] = t["policy"] + t["env"] + t["update"]
        if it >= args.warmup:
            times.append(t)
        print(it, {k: round(v, 3) for k, v in t.items()}, flush=True)

    mean = {k: float(np.mean([x[k] for x in times])) for k in times[0]}
    row = {"num_envs": cfg.num_envs, "num_steps": cfg.num_steps, "iterations": args.iterations,
           "mean_seconds_per_iteration": mean, "fraction": {k: mean[k] / mean["total"] for k in ("policy", "env", "update")},
           "steps_per_s": batch / mean["total"], "backend_jax": jax.default_backend()}
    print(json.dumps(row, indent=2))
    results = {}
    if os.path.exists(RESULTS):
        with open(RESULTS) as f:
            results = json.load(f)
    results[f"{args.backend}_{jax.default_backend()}_{cfg.num_envs}"] = row
    with open(RESULTS, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
