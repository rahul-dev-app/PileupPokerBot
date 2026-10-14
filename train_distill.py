"""Distill the afterstate value MLP into a smaller MLP (for cheap value search).

Data: games played by the teacher's card-by-card policy with epsilon-random moves. From every
position, `--actions-per-state` random legal afterstates (all placements, including the
end-of-round afterstates with an empty hand), and at every round start `--round-decisions`
random end-of-round afterstates of whole round decisions (what value search ranks). Targets are
the teacher's values; the student sees a random board symmetry and suit relabeling of each
afterstate (so it learns the symmetrized teacher). Fresh games every iteration.

    python train_distill.py --hidden-dim 128 --num-iterations 3000 --save-dir models/distill_h128 --run-name distill_h128
"""
from __future__ import annotations

import argparse
import os
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax

from pileup_poker import afterstate as A
from pileup_poker import jax_env as J
from pileup_poker import jax_greedy as G
from pileup_poker import value_search as VS

try:
    from tensorboardX import SummaryWriter
except ImportError:  # pragma: no cover - optional dependency
    SummaryWriter = None


def _take(tree, idx):
    """Gather along axis 1 of (B, M, ...) leaves with idx (B, m)."""
    return jax.tree_util.tree_map(
        lambda x: jnp.take_along_axis(x, idx.reshape(idx.shape + (1,) * (x.ndim - 2)), axis=1), tree)


def _pick(legal, values, key, num: int, num_top: int):
    """(B, M) -> (B, num) indices: the `num_top` best legal entries, then random legal ones."""
    score = jnp.where(legal, jax.random.uniform(key, legal.shape), -1.0)
    if num_top:
        best = jax.lax.top_k(values, num_top)[1]
        score = score.at[jnp.arange(legal.shape[0])[:, None], best].add(jnp.where(
            jnp.take_along_axis(legal, best, axis=1), 2.0, 0.0))
    return jax.lax.top_k(score, num)[1]


@partial(jax.jit, static_argnames=("num_games", "actions_per_state", "round_decisions", "top_actions", "round_source"))
def make_data(teacher, key, num_games: int, epsilon: float, actions_per_state: int, round_decisions: int, value_scale,
              top_actions: int = 0, round_source: str = "random"):
    """Afterstates (N, ...), teacher values (N,) and whether each is a legal afterstate (N,)."""
    k_reset, k_play = jax.random.split(key)
    state, _, _ = J.reset(k_reset, num_games)
    v_fn = lambda a: A.value("mlp", teacher, a, value_scale)
    flat = lambda t, n: jax.tree_util.tree_map(lambda x: x.reshape((-1,) + x.shape[n:]), t)
    out_a, out_v, out_w = [], [], []
    for step, k in enumerate(jax.random.split(k_play, A.NUM_STEPS)):
        k_pick, k_eps, k_rand, k_round = jax.random.split(k, 4)
        r = step // J.PLAYS_PER_ROUND
        if step % J.PLAYS_PER_ROUND == 0 and r < J.ROUNDS - 1:
            hands = G.hand_cards(state)
            if round_source == "beam":  # half the best, half random end-of-round boards of a width-8 beam
                grids, discards, _, _, values, _ = jax.vmap(
                    lambda g, d, h: VS._beam_candidates("mlp", teacher, value_scale, g, d, h, r, 8))(
                    state.grid, state.discard, hands)
                idx = _pick(values > -jnp.inf, values, k_round, round_decisions, round_decisions // 2)
            else:
                n = len(G.ROUND_CANDIDATES[r][0])
                idx = jax.random.randint(k_round, (num_games, round_decisions), 0, n)
                grids, discards, _, _ = jax.vmap(jax.vmap(VS.apply_decision, in_axes=(None, None, None, None, 0)),
                                                  in_axes=(0, 0, 0, None, 0))(state.grid, state.discard, hands, r, idx)
                idx = jnp.broadcast_to(jnp.arange(round_decisions), idx.shape)
            grids = jnp.take_along_axis(grids, idx[..., None], axis=1)
            discards = jnp.take_along_axis(discards, idx[..., None], axis=1)
            unseen = ~(state.used | A._multi_hot(state.discard) | A._multi_hot(hands))
            m = num_games * round_decisions
            a = A.Afterstate(grid=grids.reshape(m, -1), discard=discards.reshape(m, -1),
                             hand=jnp.full((m, J.HAND_SIZE), -1, jnp.int32),
                             placed=jnp.full((m,), J.PLAYS_PER_ROUND * (r + 1), jnp.int32),
                             unseen=jnp.repeat(unseen, round_decisions, axis=0))
            out_a.append(a)
            out_v.append(v_fn(a))
            out_w.append(jnp.ones(m, bool))
        afterstates, mask = A.expand_afterstates(state)
        v = jnp.where(mask, v_fn(flat(afterstates, 2)).reshape(num_games, J.ACTION_SIZE), -jnp.inf)
        pick = _pick(mask, v, k_pick, actions_per_state, top_actions)
        if step < A.NUM_STEPS - 1:  # final boards have exact values
            out_a.append(flat(_take(afterstates, pick), 2))
            out_v.append(jnp.take_along_axis(v, pick, axis=1).reshape(-1))
            out_w.append(jnp.take_along_axis(mask, pick, axis=1).reshape(-1))
        greedy = jnp.argmax(v, axis=1)
        rand = jax.random.categorical(k_rand, jnp.where(mask, 0.0, -jnp.inf), axis=1)
        action = jnp.where(jax.random.uniform(k_eps, (num_games,)) < epsilon, rand, greedy)
        state, _, _ = J.step(state, action)
    afterstates = jax.tree_util.tree_map(lambda *x: jnp.concatenate(x), *out_a)
    legal = jnp.concatenate(out_w)  # fewer legal moves than actions_per_state: some picks are illegal
    return afterstates, jnp.where(legal, jnp.concatenate(out_v), 0.0), legal


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--teacher", default="models/afterstate_mlp/afterstate_params.pkl")
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--num-iterations", type=int, default=3000)
    p.add_argument("--num-games", type=int, default=256, help="New games per iteration.")
    p.add_argument("--epsilon", type=float, default=0.1)
    p.add_argument("--actions-per-state", type=int, default=16)
    p.add_argument("--round-decisions", type=int, default=16)
    p.add_argument("--top-actions", type=int, default=0, help="Of the afterstates per position, this many are the best.")
    p.add_argument("--round-source", choices=["random", "beam"], default="random",
                   help="Round decisions: uniform, or half best / half random of a width-8 beam search.")
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--eval-episodes", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-dir", required=True)
    p.add_argument("--run-name", default="")
    args = p.parse_args()

    payload = A.load_params(args.teacher)
    teacher, value_scale = payload["params"], payload["value_scale"]
    key = jax.random.PRNGKey(args.seed)
    key, init_key = jax.random.split(key)
    params = A.init_params("mlp", init_key, args.hidden_dim, args.num_layers)
    schedule = optax.cosine_decay_schedule(args.learning_rate, args.num_iterations, alpha=0.01)
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(schedule))
    opt_state = optimizer.init(params)
    os.makedirs(args.save_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join("runs/distill", args.run_name or os.path.basename(args.save_dir))) if SummaryWriter else None
    print(f"JAX backend: {jax.default_backend()}; student params: "
          f"{sum(x.size for x in jax.tree_util.tree_leaves(params))}", flush=True)

    @jax.jit
    def train_iteration(params, opt_state, key):
        k_data, k_perm, k_aug = jax.random.split(key, 3)
        afterstates, targets, legal = make_data(teacher, k_data, args.num_games, args.epsilon, args.actions_per_state,
                                                args.round_decisions, value_scale, args.top_actions, args.round_source)
        n = targets.shape[0] // args.batch_size * args.batch_size
        perm = jax.random.permutation(k_perm, targets.shape[0])[:n].reshape(-1, args.batch_size)

        def step(carry, xs):
            params, opt_state = carry
            idx, k = xs
            batch = jax.tree_util.tree_map(lambda x: x[idx], afterstates)
            if args.augment:
                batch = A.augment(batch, k)
            w = legal[idx]
            loss, grads = jax.value_and_grad(
                lambda p: jnp.sum(w * (A.value("mlp", p, batch, value_scale) - targets[idx]) ** 2) / jnp.sum(w))(params)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            return (optax.apply_updates(params, updates), opt_state), loss

        (params, opt_state), losses = jax.lax.scan(step, (params, opt_state),
                                                   (perm, jax.random.split(k_aug, perm.shape[0])))
        return params, opt_state, losses.mean()

    start = time.time()
    losses = []
    for it in range(1, args.num_iterations + 1):
        key, k = jax.random.split(key)
        params, opt_state, loss = train_iteration(params, opt_state, k)
        losses.append(float(loss))
        if it % 50 == 0:
            rmse = float(np.sqrt(np.mean(losses))) * value_scale
            losses.clear()
            print(f"[iter {it}] rmse vs teacher {rmse:.1f} points  {time.time() - start:.0f}s", flush=True)
            if writer:
                writer.add_scalar("train/rmse_points", rmse, it)
        if it % args.eval_every == 0 or it == args.num_iterations:
            state, _, _ = J.reset(jax.random.PRNGKey(12345), args.eval_episodes)
            *_, final = A.play_games("mlp", params, state, jax.random.PRNGKey(0), 0.0, 0.0, value_scale)
            score = float(jnp.mean(final))
            print(f"[eval {it}] afterstate agent mean score {score:.1f}", flush=True)
            if writer:
                writer.add_scalar("eval/mean_final_reward", score, it)
            A.save_params(os.path.join(args.save_dir, "afterstate_params.pkl"), "mlp", params, args.hidden_dim, value_scale)
    print(f"saved {args.save_dir}/afterstate_params.pkl")


if __name__ == "__main__":
    main()
