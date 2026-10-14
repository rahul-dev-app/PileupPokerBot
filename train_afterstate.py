"""Train an afterstate value agent for Pile-Up Poker with TD(lambda) from batched self-play.

See pileup_poker/afterstate.py for the afterstate definition and the two value models.
"""
from __future__ import annotations

import argparse
import os
import pickle
import time
from dataclasses import dataclass
from datetime import datetime

import jax
import jax.numpy as jnp
import numpy as np
import optax

from pileup_poker import afterstate as A
from pileup_poker import jax_env

try:
    from tensorboardX import SummaryWriter
except ImportError:  # pragma: no cover - optional dependency
    SummaryWriter = None


PARAMS_FILE = "afterstate_params.pkl"
CHECKPOINT_FILE = "afterstate_checkpoint.pkl"


@dataclass
class AfterstateConfig:
    seed: int = 0
    model: str = "ntuple"
    num_iterations: int = 2000
    num_games: int = 256
    update_epochs: int = 2
    num_minibatches: int = 4
    optimizer: str = "adam"
    learning_rate: float = 1e-3
    max_grad_norm: float = 1.0
    hidden_dim: int = 256
    num_layers: int = 2
    init_params: str = ""
    td_lambda: float = 0.8
    epsilon: float = 0.05
    temperature: float = 0.0
    value_scale: float = 1000.0
    augment: bool = True
    log_every: int = 10
    eval_every: int = 50
    eval_episodes: int = 2000
    save_dir: str = "models/afterstate"
    tensorboard_logdir: str = "runs/afterstate"
    run_name: str = ""
    save_every: int = 100
    resume: str = ""


def make_update_step(model: str, optimizer: optax.GradientTransformation, value_scale: float, augment: bool):
    def loss_fn(params, afterstates, targets):
        return jnp.mean((A.value(model, params, afterstates, value_scale) - targets) ** 2)

    @jax.jit
    def update_step(params, opt_state, afterstates, targets, key):
        if augment:
            afterstates = A.augment(afterstates, key)
        loss, grads = jax.value_and_grad(loss_fn)(params, afterstates, targets)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    return update_step


def evaluate(config: AfterstateConfig, params, seed: int) -> float:
    state, _, _ = jax_env.reset(jax.random.PRNGKey(seed), config.eval_episodes)
    *_, final = A.play_games(config.model, params, state, jax.random.PRNGKey(seed), 0.0, 0.0, config.value_scale)
    return float(jnp.mean(final))


def _pickle_atomic(obj, path: str) -> None:
    tmp_path = path + ".tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(jax.device_get(obj), f)
    os.replace(tmp_path, path)


def save_checkpoint(config: AfterstateConfig, params, opt_state, iteration: int, key: jax.Array, run_name: str) -> None:
    A.save_params(os.path.join(config.save_dir, PARAMS_FILE), config.model, params, config.hidden_dim, config.value_scale)
    checkpoint = {
        "params": params,
        "opt_state": opt_state,
        "iteration": iteration,
        "key": key,
        "run_name": run_name,
        "model": config.model,
        "hidden_dim": config.hidden_dim,
        "value_scale": config.value_scale,
    }
    _pickle_atomic(checkpoint, os.path.join(config.save_dir, CHECKPOINT_FILE))


def make_optimizer(config: AfterstateConfig) -> optax.GradientTransformation:
    base = optax.adam(config.learning_rate) if config.optimizer == "adam" else optax.sgd(config.learning_rate)
    return optax.chain(optax.clip_by_global_norm(config.max_grad_norm), base)


def train(config: AfterstateConfig):
    samples = config.num_games * (A.NUM_STEPS - 1)
    if samples % config.num_minibatches != 0:
        raise ValueError("num_games * 15 must be divisible by num_minibatches")
    minibatch_size = samples // config.num_minibatches

    checkpoint = None
    if config.resume:
        with open(config.resume, "rb") as f:
            checkpoint = pickle.load(f)
        for name in ("model", "hidden_dim", "value_scale"):
            if checkpoint[name] != getattr(config, name):
                raise ValueError(f"--{name.replace('_', '-')} {getattr(config, name)} does not match checkpoint ({checkpoint[name]})")
        print(f"Resuming from {config.resume} after iteration {checkpoint['iteration']}")

    os.makedirs(config.save_dir, exist_ok=True)
    print(f"JAX backend: {jax.default_backend()} | devices: {[str(d) for d in jax.devices()]}")

    if config.run_name:
        run_name = config.run_name
    elif checkpoint is not None:
        run_name = checkpoint["run_name"]
    else:
        run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = None
    if SummaryWriter is None:
        print("TensorBoard logging disabled: install tensorboardX to enable it.")
    elif config.tensorboard_logdir:
        tb_dir = os.path.join(config.tensorboard_logdir, run_name)
        writer = SummaryWriter(logdir=tb_dir)
        print(f"TensorBoard run dir: {tb_dir}")

    optimizer = make_optimizer(config)
    key = jax.random.PRNGKey(config.seed)
    key, net_key = jax.random.split(key)
    start_iteration = 1
    if checkpoint is not None:
        params = jax.tree_util.tree_map(jnp.asarray, checkpoint["params"])
        opt_state = jax.tree_util.tree_map(jnp.asarray, checkpoint["opt_state"])
        key = jnp.asarray(checkpoint["key"])
        start_iteration = checkpoint["iteration"] + 1
        if start_iteration > config.num_iterations:
            raise ValueError(
                f"checkpoint is already at iteration {checkpoint['iteration']}; raise --num-iterations"
            )
    else:
        params = A.init_params(config.model, net_key, config.hidden_dim, config.num_layers)
        if config.init_params:
            params = A.load_params(config.init_params)["params"]
            print(f"Initialized from {config.init_params}")
        opt_state = optimizer.init(params)

    update_step = make_update_step(config.model, optimizer, config.value_scale, config.augment)
    returns_since_log: list[float] = []
    losses_since_log: list[float] = []
    start_time = time.time()

    for iteration in range(start_iteration, config.num_iterations + 1):
        key, reset_key, play_key = jax.random.split(key, 3)
        state, _, _ = jax_env.reset(reset_key, config.num_games)
        chosen, best_values, is_greedy, final = A.play_games(
            config.model, params, state, play_key, config.epsilon, config.temperature, config.value_scale
        )
        targets = A.lambda_returns(best_values, is_greedy, final / config.value_scale, config.td_lambda)
        # The 16th afterstate is the final board with an exact value, so it is not trained on.
        afterstates = jax.tree_util.tree_map(lambda x: x[:-1].reshape((samples,) + x.shape[2:]), chosen)
        targets = targets.reshape(samples)
        returns_since_log.extend(np.asarray(final).tolist())

        for _ in range(config.update_epochs):
            key, perm_key = jax.random.split(key)
            permutation = jax.random.permutation(perm_key, samples)
            for start in range(0, samples, minibatch_size):
                idx = permutation[start : start + minibatch_size]
                key, aug_key = jax.random.split(key)
                batch = jax.tree_util.tree_map(lambda x: x[idx], afterstates)
                params, opt_state, loss = update_step(params, opt_state, batch, targets[idx], aug_key)
                losses_since_log.append(float(loss))

        games_done = iteration * config.num_games
        if iteration % config.log_every == 0:
            elapsed = time.time() - start_time
            games_per_sec = (iteration - start_iteration + 1) * config.num_games / max(elapsed, 1e-6)
            avg_return = float(np.mean(returns_since_log))
            loss = float(np.mean(losses_since_log))
            returns_since_log.clear()
            losses_since_log.clear()
            print(
                f"[iter {iteration}/{config.num_iterations}] games={games_done} games/s={games_per_sec:.0f} "
                f"loss={loss:.5f} avg_return={avg_return:.1f}"
            )
            if writer is not None:
                writer.add_scalar("train/loss", loss, games_done)
                writer.add_scalar("train/avg_return", avg_return, games_done)
                writer.add_scalar("train/games_per_sec", games_per_sec, games_done)

        if iteration % config.eval_every == 0:
            eval_score = evaluate(config, params, seed=config.seed + iteration)
            print(f"[eval {iteration}] mean_final_reward={eval_score:.2f}")
            if writer is not None:
                writer.add_scalar("eval/mean_final_reward", eval_score, games_done)

        if iteration % config.save_every == 0 or iteration == config.num_iterations:
            save_checkpoint(config, params, opt_state, iteration, key, run_name)

    print(f"Saved {PARAMS_FILE} and {CHECKPOINT_FILE} to {config.save_dir}")
    if writer is not None:
        writer.close()


def parse_args() -> AfterstateConfig:
    d = AfterstateConfig()
    parser = argparse.ArgumentParser(description="Train an afterstate value agent for Pileup Poker with TD(lambda)")
    parser.add_argument("--seed", type=int, default=d.seed)
    parser.add_argument("--model", type=str, default=d.model, choices=A.MODELS)
    parser.add_argument("--num-iterations", type=int, default=d.num_iterations, help="Total iterations for the whole run.")
    parser.add_argument("--num-games", type=int, default=d.num_games, help="Self-play games per iteration.")
    parser.add_argument("--update-epochs", type=int, default=d.update_epochs)
    parser.add_argument("--num-minibatches", type=int, default=d.num_minibatches)
    parser.add_argument("--optimizer", type=str, default=d.optimizer, choices=["adam", "sgd"])
    parser.add_argument("--learning-rate", type=float, default=d.learning_rate)
    parser.add_argument("--max-grad-norm", type=float, default=d.max_grad_norm)
    parser.add_argument("--hidden-dim", type=int, default=d.hidden_dim, help="MLP width (mlp model only).")
    parser.add_argument("--num-layers", type=int, default=d.num_layers, help="MLP hidden layers (mlp model only).")
    parser.add_argument(
        "--init-params", type=str, default=d.init_params, help="Start from these weights (e.g. a distilled model)."
    )
    parser.add_argument("--td-lambda", type=float, default=d.td_lambda, help="1 = Monte Carlo final-score targets.")
    parser.add_argument("--epsilon", type=float, default=d.epsilon, help="Chance of a random legal move in self-play.")
    parser.add_argument(
        "--temperature", type=float, default=d.temperature, help="Softmax temperature on values (0 = argmax)."
    )
    parser.add_argument("--value-scale", type=float, default=d.value_scale, help="Values are final score / this.")
    parser.add_argument(
        "--augment",
        action=argparse.BooleanOptionalAction,
        default=d.augment,
        help="Random board symmetry and suit relabeling of training afterstates.",
    )
    parser.add_argument("--log-every", type=int, default=d.log_every)
    parser.add_argument("--eval-every", type=int, default=d.eval_every)
    parser.add_argument("--eval-episodes", type=int, default=d.eval_episodes)
    parser.add_argument("--save-dir", type=str, default=d.save_dir)
    parser.add_argument("--tensorboard-logdir", type=str, default=d.tensorboard_logdir)
    parser.add_argument("--run-name", type=str, default=d.run_name)
    parser.add_argument("--save-every", type=int, default=d.save_every, help="Save a checkpoint every N iterations.")
    parser.add_argument(
        "--resume",
        type=str,
        default=d.resume,
        help="Continue from an afterstate_checkpoint.pkl (params, optimizer state, iteration, RNG key, run name).",
    )
    return AfterstateConfig(**vars(parser.parse_args()))


if __name__ == "__main__":
    train(parse_args())
