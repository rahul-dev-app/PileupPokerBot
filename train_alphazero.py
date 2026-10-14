"""Train the AlphaZero-style agent (pileup_poker/alphazero.py): search self-play + replay + SGD."""
from __future__ import annotations

import argparse
import os
import pickle
import time
from dataclasses import dataclass, fields
from datetime import datetime

import jax
import jax.numpy as jnp
import numpy as np
import optax

from pileup_poker import alphazero as AZ
from pileup_poker import jax_env

try:
    from tensorboardX import SummaryWriter
except ImportError:  # pragma: no cover - optional dependency
    SummaryWriter = None


@dataclass
class AlphaZeroConfig:
    seed: int = 0
    iterations: int = 1000
    games_per_iteration: int = 128
    num_simulations: int = 32
    max_num_considered_actions: int = 16
    gumbel_scale: float = 1.0
    replay_size: int = 200_000
    batch_size: int = 512
    train_steps: int = 32
    learning_rate: float = 1e-3
    value_coef: float = 1.0
    l2_coef: float = 1e-6
    max_grad_norm: float = 1.0
    hidden_dim: int = 512
    num_layers: int = 3
    log_every: int = 1
    eval_every: int = 10
    eval_episodes: int = 512
    eval_simulations: int = 32
    save_every: int = 10
    save_dir: str = "models/alphazero"
    tensorboard_logdir: str = "runs/alphazero"
    run_name: str = ""
    resume: str = ""
    init_params: str = ""


PARAMS_FILE = "alphazero_params.pkl"
CHECKPOINT_FILE = "alphazero_checkpoint.pkl"
SAMPLE_SHAPES = {
    "grid": ((jax_env.GRID_CELLS,), np.int8),
    "hand": ((jax_env.HAND_SIZE,), np.int8),
    "discard": ((jax_env.ROUNDS,), np.int8),
    "round": ((), np.int8),
    "afterstate": ((), np.bool_),
    "policy": ((jax_env.ACTION_SIZE,), np.float32),
    "value": ((), np.float32),
}


class ReplayBuffer:
    """Ring buffer of the most recent self-play samples."""

    def __init__(self, capacity: int):
        self.data = {k: np.zeros((capacity, *shape), dtype) for k, (shape, dtype) in SAMPLE_SHAPES.items()}
        self.capacity = capacity
        self.pos = 0
        self.size = 0

    def add(self, samples: dict) -> None:
        n = len(samples["value"])
        idx = (self.pos + np.arange(n)) % self.capacity
        for k, v in samples.items():
            self.data[k][idx] = v
        self.pos = int((self.pos + n) % self.capacity)
        self.size = min(self.size + n, self.capacity)

    def sample(self, key: jax.Array, batch_size: int) -> dict:
        idx = np.asarray(jax.random.randint(key, (batch_size,), 0, self.size))
        return {k: v[idx] for k, v in self.data.items()}

    def state_dict(self) -> dict:
        return {"data": self.data, "pos": self.pos, "size": self.size}

    def load_state_dict(self, state: dict) -> None:
        self.data, self.pos, self.size = state["data"], state["pos"], state["size"]


def make_train_step(optimizer: optax.GradientTransformation, value_coef: float, l2_coef: float):
    @jax.jit
    def train_step(params, opt_state, batch, key):
        batch = AZ.augment(batch, key)
        (_, metrics), grads = jax.value_and_grad(AZ.loss_fn, has_aux=True)(params, batch, value_coef, l2_coef)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, metrics

    return train_step


def make_decks(key: jax.Array, n: int) -> jnp.ndarray:
    return jax.vmap(lambda k: jax.random.permutation(k, jax_env.DECK_SIZE))(jax.random.split(key, n)).astype(jnp.int32)


def _pickle_atomic(obj, path: str) -> None:
    tmp_path = path + ".tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(jax.device_get(obj), f)
    os.replace(tmp_path, path)


def save_checkpoint(save_dir: str, params, opt_state, iteration: int, key, run_name: str, replay: ReplayBuffer):
    _pickle_atomic(params, os.path.join(save_dir, PARAMS_FILE))
    checkpoint = {
        "params": params,
        "opt_state": opt_state,
        "iteration": iteration,
        "key": key,
        "run_name": run_name,
        "replay": replay.state_dict(),
    }
    _pickle_atomic(checkpoint, os.path.join(save_dir, CHECKPOINT_FILE))


def train(config: AlphaZeroConfig):
    if config.resume and config.init_params:
        raise ValueError("use either --resume or --init-params, not both")

    checkpoint = None
    if config.resume:
        with open(config.resume, "rb") as f:
            checkpoint = pickle.load(f)
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

    optimizer = optax.chain(optax.clip_by_global_norm(config.max_grad_norm), optax.adam(config.learning_rate))
    train_step = make_train_step(optimizer, config.value_coef, config.l2_coef)
    replay = ReplayBuffer(config.replay_size)

    key = jax.random.PRNGKey(config.seed)
    key, net_key = jax.random.split(key)
    start_iteration = 1
    if checkpoint is not None:
        params = jax.tree_util.tree_map(jnp.asarray, checkpoint["params"])
        opt_state = jax.tree_util.tree_map(jnp.asarray, checkpoint["opt_state"])
        key = jnp.asarray(checkpoint["key"])
        start_iteration = checkpoint["iteration"] + 1
        replay.load_state_dict(checkpoint["replay"])
        if start_iteration > config.iterations:
            raise ValueError(f"checkpoint is already at iteration {checkpoint['iteration']}; raise --iterations")
    else:
        if config.init_params:
            params = AZ.load_params(config.init_params)
            print(f"Initialized parameters from {config.init_params}")
        else:
            params = AZ.init_params(net_key, config.hidden_dim, config.num_layers)
        opt_state = optimizer.init(params)

    # Fixed evaluation decks, so evaluations are comparable across iterations and resumes.
    eval_decks = np.asarray(make_decks(jax.random.PRNGKey(config.seed + 1_000_003), config.eval_episodes))

    start_time = time.time()
    for iteration in range(start_iteration, config.iterations + 1):
        key, deck_key, play_key = jax.random.split(key, 3)
        samples, scores = AZ.self_play(
            params,
            make_decks(deck_key, config.games_per_iteration),
            play_key,
            config.num_simulations,
            config.max_num_considered_actions,
            config.gumbel_scale,
        )
        replay.add(samples)

        metrics = []
        for _ in range(config.train_steps):
            key, batch_key, aug_key = jax.random.split(key, 3)
            batch = replay.sample(batch_key, config.batch_size)
            params, opt_state, m = train_step(params, opt_state, batch, aug_key)
            metrics.append(m)

        games = iteration * config.games_per_iteration
        if iteration % config.log_every == 0:
            mean_metrics = {k: float(jnp.mean(jnp.stack([m[k] for m in metrics]))) for k in metrics[0]}
            elapsed = time.time() - start_time
            games_per_sec = (iteration - start_iteration + 1) * config.games_per_iteration / max(elapsed, 1e-6)
            print(
                f"[iter {iteration}/{config.iterations}] games={games} games/s={games_per_sec:.1f} "
                f"loss={mean_metrics['loss']:.4f} policy={mean_metrics['policy_loss']:.4f} "
                f"value={mean_metrics['value_loss']:.4f} afterstate_value={mean_metrics['afterstate_value_loss']:.4f} "
                f"selfplay_return={scores.mean():.1f} replay={replay.size}"
            )
            if writer is not None:
                for k, v in mean_metrics.items():
                    writer.add_scalar(f"train/{k}", v, games)
                writer.add_scalar("train/avg_return", float(scores.mean()), games)
                writer.add_scalar("train/games_per_sec", games_per_sec, games)
                writer.add_scalar("train/replay_size", replay.size, games)

        if iteration % config.eval_every == 0:
            eval_score = AZ.play(params, eval_decks, config.eval_simulations, config.max_num_considered_actions).mean()
            network_score = AZ.play(params, eval_decks, 0).mean()
            print(f"[eval {iteration}] mean_final_reward={eval_score:.2f} network_only={network_score:.2f}")
            if writer is not None:
                writer.add_scalar("eval/mean_final_reward", float(eval_score), games)
                writer.add_scalar("eval/network_mean_final_reward", float(network_score), games)

        if iteration % config.save_every == 0 or iteration == config.iterations:
            save_checkpoint(config.save_dir, params, opt_state, iteration, key, run_name, replay)

    print(f"Saved {PARAMS_FILE} and {CHECKPOINT_FILE} to {config.save_dir}")
    if writer is not None:
        writer.close()


def parse_args() -> AlphaZeroConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    help_text = {
        "iterations": "Total self-play iterations for the whole run (raise it to extend a resumed run).",
        "games_per_iteration": "Games played in parallel per iteration (16 searched moves each).",
        "num_simulations": "Search simulations per move during self-play.",
        "gumbel_scale": "Gumbel noise on root actions during self-play (evaluation uses 0).",
        "replay_size": "Replay buffer capacity in samples (20 per game: 16 moves + 4 afterstates).",
        "train_steps": "SGD steps on replay minibatches per iteration.",
        "eval_episodes": "Fixed evaluation decks (search with --eval-simulations, and network argmax).",
        "save_every": "Save params and checkpoint every N iterations (and at the end).",
        "resume": "Continue from alphazero_checkpoint.pkl (params, optimizer, iteration, RNG key, replay).",
        "init_params": "Start from alphazero_params.pkl with a fresh optimizer, replay and iteration count.",
    }
    for f in fields(AlphaZeroConfig):
        parser.add_argument(f"--{f.name.replace('_', '-')}", type=type(f.default), default=f.default,
                            help=help_text.get(f.name))
    return AlphaZeroConfig(**vars(parser.parse_args()))


if __name__ == "__main__":
    train(parse_args())
