"""Watch a trained PPO policy play in the original Gym environment (text rendering).

    python play_jax_policy.py --model-path models/ppo_attn_bc/jax_ppo_params.pkl --episodes 3
"""
from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp
import numpy as np

from pileup_poker.networks import ModelConfig
from pileup_poker.pileup_poker import PileupPokerEnv, flatten_observation
from train_jax_ppo import load_policy, policy_log_probs


def pick_action(
    params,
    model_cfg: ModelConfig,
    obs: dict[str, np.ndarray],
    action_mask: list[bool],
    deterministic: bool,
    key: jax.Array,
) -> tuple[int, jax.Array]:
    obs_vec = jnp.asarray(flatten_observation(obs))[None, :]
    mask = jnp.asarray(np.asarray(action_mask, dtype=np.bool_))[None, :]
    log_probs, _ = policy_log_probs(params, model_cfg, obs_vec, mask)

    if deterministic:
        action = int(jnp.argmax(log_probs, axis=-1)[0])
        return action, key

    key, sample_key = jax.random.split(key)
    action = int(jax.random.categorical(sample_key, log_probs, axis=-1)[0])
    return action, key


def run_episode(params, model_cfg: ModelConfig, seed: int, deterministic: bool, render: bool) -> float:
    env = PileupPokerEnv()
    obs, _ = env.reset(seed=seed)
    done = False
    reward = 0.0
    key = jax.random.PRNGKey(seed)

    if render:
        env.render()

    while not done:
        action_mask = env.get_action_mask()
        action, key = pick_action(params, model_cfg, obs, action_mask, deterministic, key)
        obs, reward, done, _, _ = env.step(action)
        if render:
            env.render()

    env.close()
    return float(reward)  # the last step's reward is the final score


def main():
    parser = argparse.ArgumentParser(description="Run a trained JAX PPO policy in the original PileupPokerEnv renderer")
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/ppo_attn_bc/jax_ppo_params.pkl",
        help="Path to pickle file saved by train_jax_ppo.py",
    )
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stochastic", action="store_true", help="Sample actions instead of argmax")
    parser.add_argument("--no-render", action="store_true", help="Disable env.render() prints")
    args = parser.parse_args()

    params, model_cfg = load_policy(args.model_path)
    print(f"Model: {model_cfg}")
    deterministic = not args.stochastic
    render = not args.no_render

    scores = []
    for episode in range(args.episodes):
        episode_seed = args.seed + episode
        score = run_episode(params, model_cfg, seed=episode_seed, deterministic=deterministic, render=render)
        scores.append(score)
        print(f"Episode {episode + 1}/{args.episodes} final score {score:.0f}")

    print(f"Mean final score over {args.episodes} episodes: {float(np.mean(scores)):.1f}")


if __name__ == "__main__":
    main()
