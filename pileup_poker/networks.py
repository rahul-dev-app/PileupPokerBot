"""Policy/value networks for train_jax_ppo.py, in pure JAX.

Every network maps the 796-dim observation from `jax_env.build_observation` to a policy
output and a value. `action_log_probs` turns the policy output into masked log-probabilities
over the 80 joint actions (slot * 16 + cell), for both action heads.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from pileup_poker import jax_env as J

NEG_INF = -1e9
NUM_TOKENS = J.GRID_CELLS + J.HAND_SIZE + J.ROUNDS + 1  # cells, hand slots, discard slots, round

_GRID_END = J.GRID_CELLS * J.DECK_SIZE
_HAND_END = _GRID_END + J.HAND_SIZE * J.DECK_SIZE
_DISCARD_END = _HAND_END + J.DECK_SIZE


@dataclass(frozen=True)
class ModelConfig:
    arch: str = "mlp"  # mlp | attention
    action_head: str = "joint"  # joint | factored
    hidden_dim: int = 256  # mlp width
    separate_critic: bool = False
    embed_dim: int = 64  # attention width
    num_layers: int = 3
    num_heads: int = 4


def _linear(key, fan_in: int, fan_out: int, scale: float = np.sqrt(2.0)):
    w = jax.random.normal(key, (fan_in, fan_out), dtype=jnp.float32) * (scale / np.sqrt(fan_in))
    return w, jnp.zeros((fan_out,), dtype=jnp.float32)


# ---------------------------------------------------------------- mlp


def _init_mlp(key, cfg: ModelConfig) -> dict:
    # Same key use as the original train_jax_ppo.init_mlp_params, so the default config starts identically.
    k1, k2, k3, k4 = jax.random.split(key, 4)
    p = {}
    p["w1"], p["b1"] = _linear(k1, J.OBS_SIZE, cfg.hidden_dim)
    p["w2"], p["b2"] = _linear(k2, cfg.hidden_dim, cfg.hidden_dim)
    p["w_pi"], p["b_pi"] = _linear(k3, cfg.hidden_dim, J.ACTION_SIZE, scale=0.01)
    p["w_v"], p["b_v"] = _linear(k4, cfg.hidden_dim, 1, scale=1.0)
    if cfg.action_head == "factored":
        p["w_slot"], p["b_slot"] = _linear(jax.random.fold_in(key, 1), cfg.hidden_dim, J.HAND_SIZE, scale=0.01)
    return p


def _mlp_trunk(p, obs):
    h = jnp.tanh(obs @ p["w1"] + p["b1"])
    return jnp.tanh(h @ p["w2"] + p["b2"])


def _mlp_policy(p, cfg: ModelConfig, h):
    logits = h @ p["w_pi"] + p["b_pi"]
    if cfg.action_head == "factored":
        return h @ p["w_slot"] + p["b_slot"], logits
    return logits


def _mlp_value(p, h):
    return (h @ p["w_v"] + p["b_v"]).squeeze(-1)


# ---------------------------------------------------------------- attention


def _layer_norm(x, g, b):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    return (x - mean) * jax.lax.rsqrt(var + 1e-5) * g + b


def _init_attention(key, cfg: ModelConfig) -> dict:
    d = cfg.embed_dim
    keys = iter(jax.random.split(key, 16 + 4 * cfg.num_layers))

    def emb(*shape):
        return jax.random.normal(next(keys), shape, dtype=jnp.float32) / np.sqrt(d)

    p = {
        "rank_emb": emb(J.DECK_SIZE // 4, d),
        "suit_emb": emb(4, d),
        "empty_emb": emb(d),
        "cell_pos": emb(J.GRID_CELLS, d),
        "slot_pos": emb(J.HAND_SIZE, d),
        "used_emb": emb(d),
        "discard_emb": emb(d),
        "round_emb": emb(J.ROUNDS, d),
        "round_type": emb(d),
        "ln_g": jnp.ones(d),
        "ln_b": jnp.zeros(d),
        "blocks": [],
    }
    for _ in range(cfg.num_layers):
        blk = {"ln1_g": jnp.ones(d), "ln1_b": jnp.zeros(d), "ln2_g": jnp.ones(d), "ln2_b": jnp.zeros(d)}
        blk["w_qkv"], blk["b_qkv"] = _linear(next(keys), d, 3 * d, scale=1.0)
        blk["w_o"], blk["b_o"] = _linear(next(keys), d, d, scale=1.0 / np.sqrt(2 * cfg.num_layers))
        blk["w_ff1"], blk["b_ff1"] = _linear(next(keys), d, 4 * d)
        blk["w_ff2"], blk["b_ff2"] = _linear(next(keys), 4 * d, d, scale=1.0 / np.sqrt(2 * cfg.num_layers))
        p["blocks"].append(blk)
    p["w_q"], _ = _linear(next(keys), d, d, scale=0.1)
    p["w_k"], _ = _linear(next(keys), d, d, scale=1.0)
    p["b_pi"] = jnp.zeros(J.ACTION_SIZE)
    if cfg.action_head == "factored":
        p["w_slot"], p["b_slot"] = _linear(next(keys), d, 1, scale=0.01)
    p["w_v1"], p["b_v1"] = _linear(next(keys), d, d)
    p["w_v"], p["b_v"] = _linear(next(keys), d, 1, scale=1.0)
    return p


def _tokens(p, obs):
    """(B, 796) observation -> (B, NUM_TOKENS, d) input tokens."""
    batch = obs.shape[0]
    card_table = jnp.repeat(p["rank_emb"], 4, axis=0) + jnp.tile(p["suit_emb"], (J.DECK_SIZE // 4, 1))  # card = rank*4+suit

    def cards(one_hot):  # (..., 36) one-hot or all zeros -> card or empty embedding
        return one_hot @ card_table + (1.0 - one_hot.sum(-1, keepdims=True)) * p["empty_emb"]

    grid = obs[:, :_GRID_END].reshape(batch, J.GRID_CELLS, J.DECK_SIZE)
    hand = obs[:, _GRID_END:_HAND_END].reshape(batch, J.HAND_SIZE, J.DECK_SIZE)
    discard = obs[:, _HAND_END:_DISCARD_END]
    round_one_hot = obs[:, _DISCARD_END:]

    cell_tok = cards(grid) + p["cell_pos"]
    used = jnp.einsum("bsc,bc->bs", hand, grid.sum(1))[..., None]  # hand card already on the board
    hand_tok = cards(hand) + p["slot_pos"] + used * p["used_emb"]
    present, idx = jax.lax.top_k(discard, J.ROUNDS)
    discard_tok = cards(jax.nn.one_hot(idx, J.DECK_SIZE) * present[..., None]) + p["discard_emb"]
    round_tok = (round_one_hot @ p["round_emb"] + p["round_type"])[:, None, :]
    return jnp.concatenate([cell_tok, hand_tok, discard_tok, round_tok], axis=1)


def _block(blk, x, num_heads: int):
    batch, tokens, d = x.shape
    head_dim = d // num_heads
    h = _layer_norm(x, blk["ln1_g"], blk["ln1_b"])
    q, k, v = jnp.split(h @ blk["w_qkv"] + blk["b_qkv"], 3, axis=-1)
    q, k, v = (t.reshape(batch, tokens, num_heads, head_dim) for t in (q, k, v))
    att = jax.nn.softmax(jnp.einsum("bthd,bshd->bhts", q, k) / np.sqrt(head_dim), axis=-1)
    out = jnp.einsum("bhts,bshd->bthd", att, v).reshape(batch, tokens, d)
    x = x + out @ blk["w_o"] + blk["b_o"]
    h = _layer_norm(x, blk["ln2_g"], blk["ln2_b"])
    return x + jax.nn.gelu(h @ blk["w_ff1"] + blk["b_ff1"]) @ blk["w_ff2"] + blk["b_ff2"]


def _attention_trunk(p, cfg: ModelConfig, obs):
    x = _tokens(p, obs)
    for blk in p["blocks"]:
        x = _block(blk, x, cfg.num_heads)
    return _layer_norm(x, p["ln_g"], p["ln_b"])


def _attention_policy(p, cfg: ModelConfig, x):
    cells = x[:, : J.GRID_CELLS]
    hand = x[:, J.GRID_CELLS : J.GRID_CELLS + J.HAND_SIZE]
    pair = jnp.einsum("bsd,bcd->bsc", hand @ p["w_q"], cells @ p["w_k"]) / np.sqrt(cfg.embed_dim)
    logits = pair.reshape(x.shape[0], J.ACTION_SIZE) + p["b_pi"]
    if cfg.action_head == "factored":
        return (hand @ p["w_slot"] + p["b_slot"]).squeeze(-1), logits
    return logits


def _attention_value(p, x):
    h = jnp.tanh(jnp.mean(x, axis=1) @ p["w_v1"] + p["b_v1"])
    return (h @ p["w_v"] + p["b_v"]).squeeze(-1)


# ---------------------------------------------------------------- public


_POLICY_KEYS = {"w_pi", "b_pi", "w_slot", "b_slot", "w_q", "w_k"}
_VALUE_KEYS = {"w_v", "b_v", "w_v1", "b_v1"}


def init_params(key: jax.Array, cfg: ModelConfig) -> dict:
    init = _init_mlp if cfg.arch == "mlp" else _init_attention
    params = init(key, cfg)
    if cfg.separate_critic:
        critic = init(jax.random.fold_in(key, 2), cfg)
        params = {k: v for k, v in params.items() if k not in _VALUE_KEYS}
        params["critic"] = {k: v for k, v in critic.items() if k not in _POLICY_KEYS}
    return params


def _trunk(p, cfg: ModelConfig, obs):
    return _mlp_trunk(p, obs) if cfg.arch == "mlp" else _attention_trunk(p, cfg, obs)


def _value(p, cfg: ModelConfig, features):
    return _mlp_value(p, features) if cfg.arch == "mlp" else _attention_value(p, features)


def forward(params, cfg: ModelConfig, obs):
    """Returns (policy output, values). Policy output: (B, 80) logits for the joint head,
    (slot logits (B, 5), cell logits (B, 80) laid out as slot * 16 + cell) for the factored head."""
    features = _trunk(params, cfg, obs)
    if cfg.arch == "mlp":
        policy = _mlp_policy(params, cfg, features)
    else:
        policy = _attention_policy(params, cfg, features)
    if cfg.separate_critic:
        critic = params["critic"]
        return policy, _value(critic, cfg, _trunk(critic, cfg, obs))
    return policy, _value(params, cfg, features)


def masked_log_probs(cfg: ModelConfig, policy, action_mask):
    """Log-probabilities (B, 80) of the joint actions; masked actions get about NEG_INF."""
    # Rows without any valid action (finished games) fall back to an unmasked distribution.
    has_valid = jnp.any(action_mask, axis=-1, keepdims=True)
    action_mask = jnp.where(has_valid, action_mask, True)
    if cfg.action_head == "joint":
        return jax.nn.log_softmax(jnp.where(action_mask, policy, NEG_INF), axis=-1)
    slot_logits, cell_logits = policy
    batch = action_mask.shape[0]
    mask = action_mask.reshape(batch, J.HAND_SIZE, J.GRID_CELLS)
    slot_lp = jax.nn.log_softmax(jnp.where(jnp.any(mask, -1), slot_logits, NEG_INF), axis=-1)
    cell_lp = jax.nn.log_softmax(jnp.where(mask, cell_logits.reshape(mask.shape), NEG_INF), axis=-1)
    # log p(slot) + log p(cell | slot); rows of invalid slots stay near NEG_INF.
    return (slot_lp[..., None] + cell_lp).reshape(batch, J.ACTION_SIZE)


def action_log_probs(params, cfg: ModelConfig, obs, action_mask):
    policy, values = forward(params, cfg, obs)
    return masked_log_probs(cfg, policy, action_mask), values


def entropy(log_probs):
    return -jnp.sum(jnp.exp(log_probs) * log_probs, axis=-1)
