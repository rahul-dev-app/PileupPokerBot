"""Symmetries of Pile-Up Poker that leave every score unchanged.

Board: rows 0/3 hold corners and rows 1/2 do not, so allowed row maps are
{id, swap 0<->3, swap 1<->2, both}; the same for columns; plus transpose.
That gives 4 * 4 * 2 = 32 cell permutations. Each maps rows to rows (or columns),
columns to columns (or rows), and corners to corners.

Suits: scoring only checks whether suits are equal, so any of the 4! = 24 suit
relabelings is also a symmetry. Ranks are not (straights depend on rank order).
"""
from __future__ import annotations

from itertools import permutations

import numpy as np

from pileup_poker.jax_env import DECK_SIZE, GRID_CELLS, GRID_SIZE, HAND_SIZE

_LINE_MAPS = [(0, 1, 2, 3), (3, 1, 2, 0), (0, 2, 1, 3), (3, 2, 1, 0)]


def _cell_perms() -> np.ndarray:
    perms = []
    for transpose in (False, True):
        for rmap in _LINE_MAPS:
            for cmap in _LINE_MAPS:
                perm = np.empty(GRID_CELLS, dtype=np.int32)
                for r in range(GRID_SIZE):
                    for c in range(GRID_SIZE):
                        src_r, src_c = rmap[r], cmap[c]
                        if transpose:
                            src_r, src_c = src_c, src_r
                        perm[r * GRID_SIZE + c] = src_r * GRID_SIZE + src_c
                perms.append(perm)
    return np.stack(perms)


# CELL_PERMS[k, i] = source cell: transformed_grid[i] = grid[CELL_PERMS[k, i]]. CELL_PERMS[0] is identity.
CELL_PERMS = _cell_perms()
NUM_CELL_SYMMETRIES = CELL_PERMS.shape[0]

# SUIT_PERMS[k, card] = relabeled card id (card id = rank * 4 + suit). SUIT_PERMS[0] is identity.
SUIT_PERMS = np.stack(
    [np.array([(c // 4) * 4 + p[c % 4] for c in range(DECK_SIZE)], dtype=np.int32) for p in permutations(range(4))]
)
NUM_SUIT_SYMMETRIES = SUIT_PERMS.shape[0]


def inverse_perms(perms: np.ndarray) -> np.ndarray:
    return np.argsort(perms, axis=-1).astype(np.int32)


# ACTION_PERMS[k, a] = source action under cell symmetry k, with action = slot * 16 + cell.
# transformed_logits[..., a] = logits[..., ACTION_PERMS[k, a]].
ACTION_PERMS = np.stack(
    [np.array([s * GRID_CELLS + p[c] for s in range(HAND_SIZE) for c in range(GRID_CELLS)], dtype=np.int32) for p in CELL_PERMS]
)
