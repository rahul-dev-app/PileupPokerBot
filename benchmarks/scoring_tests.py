"""Property tests of the scoring rules (level 1 of the verification): the Python env's scorer and
jax_env.calculate_total_score are compared with hand-computed scores and with a small reference
scorer written from the rules in docs/environment.md, on hand-picked and random boards.

    JAX_PLATFORMS=cpu python benchmarks/scoring_tests.py --random-boards 200000

Writes benchmarks/results/scoring_tests.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax.numpy as jnp  # noqa: E402

from pileup_poker import jax_env as J  # noqa: E402
from pileup_poker.pileup_poker import DECK, PileupPokerEnv  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "scoring_tests.json")
VALUES = {"pair": 5, "two_pairs": 60, "flush": 80, "three": 125, "straight": 180, "four": 325, "straight_flush": 450}


def reference_hand(cards: list[int]) -> int:
    """Value of a complete 4-card hand, from the rules."""
    counts = sorted(Counter(c // 4 for c in cards).values(), reverse=True)
    ranks = sorted(c // 4 for c in cards)
    flush = len({c % 4 for c in cards}) == 1
    straight = len(set(ranks)) == 4 and ranks[3] - ranks[0] == 3  # ace high only, no wrap-around
    if counts[0] == 4:
        return VALUES["four"]
    if counts[0] == 3:
        return VALUES["three"]
    if counts[:2] == [2, 2]:
        return VALUES["two_pairs"]
    if counts[0] == 2:
        return VALUES["pair"]
    if straight and flush:
        return VALUES["straight_flush"]
    if flush:
        return VALUES["flush"]
    if straight:
        return VALUES["straight"]
    return 0


def reference_line(cards: list[int]) -> int:
    if len(cards) == 4:
        return reference_hand(cards)
    if len(cards) in (2, 3):  # partial lines: pair 5, three of a kind 125
        top = max(Counter(c // 4 for c in cards).values())
        return {2: VALUES["pair"], 3: VALUES["three"]}.get(top, 0)
    return 0


def reference_score(grid: list[int], discard: list[int]) -> int:
    cells = [grid[4 * r: 4 * r + 4] for r in range(4)]
    hands = [reference_line([c for c in row if c >= 0]) for row in cells]
    hands += [reference_line([cells[r][k] for r in range(4) if cells[r][k] >= 0]) for k in range(4)]
    corners = [grid[0], grid[3], grid[12], grid[15]]
    hands.append(2 * reference_hand(corners) if min(corners) >= 0 else 0)
    nine = sum(h != 0 for h in hands)
    hands.append(3 * reference_hand(discard) if nine == 9 else 0)
    scored = sum(h != 0 for h in hands)
    multiplier = {0: 1, 1: 1, 2: 2, 3: 2, 4: 3, 5: 3, 6: 4, 7: 4, 8: 5, 9: 5, 10: 6}[scored]
    return sum(hands) * multiplier


def python_score(env: PileupPokerEnv, grid: list[int], discard: list[int]) -> float:
    env.grid = [[DECK[grid[4 * r + k]] if grid[4 * r + k] >= 0 else "" for k in range(4)] for r in range(4)]
    env.discard_hand = [DECK[c] for c in discard if c >= 0]
    return float(env.get_total_score())


def c(name: str) -> int:
    return DECK.index(name)


def board(cells: dict[int, str]) -> list[int]:
    return [c(cells[i]) if i in cells else -1 for i in range(16)]


def row0(*cards: str) -> list[int]:
    return board({k: card for k, card in enumerate(cards)})


QUADS_BOARD = board({4 * r + k: f"{'6789'[r]}{'SHDC'[k]}" for r in range(4) for k in range(4)})
# Rows: four of a kind (325 each). Columns: straight flushes (450 each). Corners 6S 6C 9S 9C: two pairs, x2.
CASES = [
    # (description, grid, discard, hand-computed score)
    ("empty board", [-1] * 16, [], 0),
    ("row: no hand", row0("6S", "8H", "TD", "QC"), [], 0),
    ("row: pair", row0("6S", "6H", "8D", "TC"), [], 5),
    ("row: two pairs", row0("6S", "6H", "8D", "8C"), [], 60),
    ("row: three of a kind", row0("6S", "6H", "6D", "8C"), [], 125),
    ("row: four of a kind", row0("6S", "6H", "6D", "6C"), [], 325),
    ("row: flush", row0("6S", "8S", "TS", "QS"), [], 80),
    ("row: straight", row0("6S", "7H", "8D", "9C"), [], 180),
    ("row: straight flush", row0("6S", "7S", "8S", "9S"), [], 450),
    ("row: ace-high straight J Q K A", row0("JS", "QH", "KD", "AC"), [], 180),
    ("row: no wrap-around A 6 7 8", row0("AS", "6H", "7D", "8C"), [], 0),
    ("partial row: 2 cards pair", row0("6S", "6H"), [], 5),
    ("partial row: 3 cards with a pair", row0("6S", "6H", "8D"), [], 5),
    ("partial row: 3 cards three of a kind", row0("6S", "6H", "6D"), [], 125),
    ("partial row: 3-card straight does not score", row0("6S", "7H", "8D"), [], 0),
    ("corners: flush counts x2", board({0: "6S", 3: "8S", 12: "TS", 15: "QS"}), [], 160),
    ("two scoring hands: x2", board({1: "6S", 2: "6H", 5: "7D", 6: "8C", 9: "QS", 10: "QH"}), [], 20),
    ("all 9 + discard straight flush: x3 discard, x6", QUADS_BOARD, [c("TS"), c("JS"), c("QS"), c("KS")],
     (4 * 325 + 4 * 450 + 2 * 60 + 3 * 450) * 6),
    ("all 9 + discard pair: x6", QUADS_BOARD, [c("TS"), c("TH"), c("JS"), c("QD")],
     (4 * 325 + 4 * 450 + 2 * 60 + 3 * 5) * 6),
    ("all 9, discard scores nothing: x5", QUADS_BOARD, [c("TS"), c("JH"), c("QD"), c("AC")],
     (4 * 325 + 4 * 450 + 2 * 60) * 5),
]


def random_boards(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Full and partial random boards. Half the boards use a deck reduced to few ranks, so more hands score
    and the high multipliers and the discard rule are reached more often."""
    grids = np.full((n, 16), -1, np.int32)
    discards = np.full((n, 4), -1, np.int32)
    for i in range(n):
        pool = np.arange(36) if i % 2 == 0 else np.flatnonzero(np.isin(np.arange(36) // 4, rng.choice(9, 6, replace=False)))
        cards = rng.permutation(pool)[:20]
        filled = 16 if rng.random() < 0.5 else int(rng.integers(0, 17))
        cells = rng.permutation(16)[:filled]
        grids[i, cells] = cards[:filled]
        n_discard = 4 if filled == 16 else filled // 4
        discards[i, :n_discard] = cards[16:16 + n_discard]
    return grids, discards


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--random-boards", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    env = PileupPokerEnv()

    case_rows = []
    for name, grid, discard, expected in CASES:
        discard4 = (discard + [-1] * 4)[:4]
        jax_s = float(J.calculate_total_score(jnp.asarray([grid]), jnp.asarray([discard4]))[0])
        row = {"case": name, "expected": expected, "reference": reference_score(grid, discard4),
               "python_env": python_score(env, grid, discard4), "jax_env": jax_s}
        row["pass"] = row["reference"] == row["python_env"] == row["jax_env"] == expected
        case_rows.append(row)
        print(f"{'ok  ' if row['pass'] else 'FAIL'} {name}: {expected}")

    grids, discards = random_boards(args.random_boards, np.random.default_rng(args.seed))
    jax_scores = np.asarray(J.calculate_total_score(jnp.asarray(grids), jnp.asarray(discards)))
    py_scores = np.array([python_score(env, list(g), list(d)) for g, d in zip(grids, discards)])
    ref_scores = np.array([reference_score(list(g), list(d)) for g, d in zip(grids, discards)])
    multipliers = Counter()
    discard_counted = 0
    for g, d in zip(grids, discards):
        python_score(env, list(g), list(d))
        multipliers[env.multiplier] += 1
        discard_counted += env.hand_score_array[9] != 0
    full = np.all(grids >= 0, axis=1)
    result = {
        "hand_picked": {"cases": len(case_rows), "passed": sum(r["pass"] for r in case_rows), "rows": case_rows},
        "random": {
            "boards": args.random_boards, "full_boards": int(full.sum()), "seed": args.seed,
            "jax_vs_python_mismatches": int(np.sum(jax_scores != py_scores)),
            "jax_vs_reference_mismatches_full_boards": int(np.sum((jax_scores != ref_scores) & full)),
            "jax_vs_reference_mismatches_partial_boards": int(np.sum((jax_scores != ref_scores) & ~full)),
            "boards_per_multiplier": {str(k): v for k, v in sorted(multipliers.items())},
            "boards_with_scoring_discard": int(discard_counted),
            "max_score": float(jax_scores.max()),
        },
    }
    print(json.dumps(result["random"], indent=2))
    with open(RESULTS, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {RESULTS}")


if __name__ == "__main__":
    main()
