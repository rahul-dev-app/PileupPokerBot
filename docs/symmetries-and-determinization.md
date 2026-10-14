# Symmetries and determinization

Two shared building blocks are used by several agents:

- **Symmetries** (`pileup_poker/symmetry.py`): transformations of a position that never change any score. They are used for data augmentation and to remove duplicate candidates in search.
- **Determinization** (`jax_env.determinize`): random copies of a game with the unseen cards shuffled. Search agents look ahead only on these copies, so no agent can see future cards.

## Board and suit symmetries

Board symmetries must keep rows as rows (or as columns), columns as columns (or as rows), and corners as corners:

- Rows 0 and 3 contain corners, rows 1 and 2 do not. So the allowed row maps are: identity, swap 0<->3, swap 1<->2, both swaps.
- The same 4 maps exist for columns.
- The board can also be transposed.
- Total: 4 x 4 x 2 = **32 cell permutations** (`CELL_PERMS`, index 0 = identity).

Suits only matter through "are these suits equal" (flushes). So any relabeling of the 4 suits is also a symmetry: **24 suit permutations** (`SUIT_PERMS`). Ranks have no symmetry, because straights depend on rank order.

`ACTION_PERMS[k]` maps actions (`slot * 16 + cell`) under cell symmetry `k`, and `inverse_perms` inverts any permutation table.

Where they are used:

| user | how |
|---|---|
| Afterstate TD and distillation (`train_afterstate.py`, `train_distill.py`: `--augment`, default on) | each training afterstate gets a random cell symmetry and suit relabeling |
| AlphaZero | each replay sample gets a random cell symmetry and suit relabeling (policy target permuted with `ACTION_PERMS`) |
| PPO (`--symmetry-aug`) | every rollout step shows each game under a fresh random symmetry, and the chosen action is mapped back; each behavior cloning sample also gets a random symmetry |
| MC search and value search | `search_agent.canonical_hash` removes candidates whose board + discard pile is a symmetric copy of a better candidate |

`canonical_hash(grids, discards, use_suit_symmetry)` computes, for each of two independent additive hashes, the minimum over all 32 (x 24) symmetries. Two positions get the same pair of words when they are symmetric copies; different positions collide with probability about 2^-64.

## Determinization: no peeking at future cards

In `jax_env` the whole deck order is stored in `state.deck`, because the environment needs it to deal the next hand. Agents must not use that order. The rules in this repo:

- **Policy and value networks** (PPO, AlphaZero network, afterstate models) only see the board, the discard pile, the current hand and the round. Afterstate features use the set of unseen cards, never their order.
- **Search agents** (MC search, AlphaZero tree search, value search) run on `jax_env.determinize(state, key)` copies.

`determinize` keeps every dealt card in place (board, discard pile, current hand) and shuffles the undealt part of the deck. The random sort key of a card depends on the card id, not on its position in the deck. So the result depends only on *which* cards are undealt, never on their real order: changing the order of the undealt cards in the real deck does not change any decision of any agent.

The [Chrome extension](chrome-extension.md#how-it-works) follows the same rule on the live site: it reads only the board, the hand and the discard pile, never the face-down draw pile (which is present in the page). Its search samples the next hands from the set of unseen cards.
