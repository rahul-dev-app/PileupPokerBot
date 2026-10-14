from __future__ import annotations

import argparse
import os

import jax
import jax.numpy as jnp
import numpy as np

from pileup_poker import jax_env


CARD_STRINGS = [f"{rank}{suit}" for rank in "6789TJQKA" for suit in "SHDC"]


def _card_to_str(card_id: int) -> str:
    if card_id < 0:
        return ".."
    return CARD_STRINGS[card_id]


def _get_hand_cards(state: jax_env.EnvState) -> np.ndarray:
    deck = np.asarray(state.deck[0])
    current_round = int(np.asarray(state.current_round[0]))
    start = current_round * jax_env.HAND_SIZE
    return deck[start : start + jax_env.HAND_SIZE]


def _print_board_with_indices(state: jax_env.EnvState) -> None:
    grid = np.asarray(state.grid[0]).reshape(jax_env.GRID_SIZE, jax_env.GRID_SIZE)
    print("\nBoard (index: card):")
    for r in range(jax_env.GRID_SIZE):
        cells = []
        for c in range(jax_env.GRID_SIZE):
            pos = r * jax_env.GRID_SIZE + c
            card = _card_to_str(int(grid[r, c]))
            cells.append(f"{pos:02d}:{card:>2}")
        print("  " + " | ".join(cells))


def _print_hand(state: jax_env.EnvState) -> None:
    hand_cards = _get_hand_cards(state)
    used = np.asarray(state.used[0])
    print("\nCurrent hand slots:")
    for i, card_id in enumerate(hand_cards):
        status = " (used)" if bool(used[int(card_id)]) else ""
        print(f"  slot {i}: {_card_to_str(int(card_id))}{status}")


def _print_valid_actions(state: jax_env.EnvState) -> None:
    mask = np.asarray(jax_env.get_action_mask(state)[0]).astype(bool)
    print("\nValid moves (slot -> positions):")
    for slot in range(jax_env.HAND_SIZE):
        positions = []
        base = slot * jax_env.GRID_CELLS
        for pos in range(jax_env.GRID_CELLS):
            if mask[base + pos]:
                positions.append(str(pos))
        if positions:
            print(f"  slot {slot}: {', '.join(positions)}")


def _prompt_action(state: jax_env.EnvState) -> int:
    mask = np.asarray(jax_env.get_action_mask(state)[0]).astype(bool)
    while True:
        raw = input(
            "\nEnter move as 'slot position' (e.g. '2 11'), or 'v' to print valid moves, or 'q' to quit: "
        ).strip()

        if raw.lower() in {"q", "quit", "exit"}:
            raise KeyboardInterrupt
        if raw.lower() in {"v", "valid"}:
            _print_valid_actions(state)
            continue

        parts = raw.split()
        if len(parts) != 2:
            print("Invalid input format. Use exactly two integers: slot position")
            continue

        try:
            slot = int(parts[0])
            position = int(parts[1])
        except ValueError:
            print("Both slot and position must be integers.")
            continue

        if not (0 <= slot < jax_env.HAND_SIZE):
            print(f"Slot must be in [0, {jax_env.HAND_SIZE - 1}]")
            continue
        if not (0 <= position < jax_env.GRID_CELLS):
            print(f"Position must be in [0, {jax_env.GRID_CELLS - 1}]")
            continue

        action = slot * jax_env.GRID_CELLS + position
        if not mask[action]:
            print("That move is invalid in the current state. Type 'v' to see valid moves.")
            continue

        return action


def run(seed: int | None) -> None:
    if seed is None:
        seed = int.from_bytes(os.urandom(4), "little")

    key = jax.random.PRNGKey(seed)
    state, _, _ = jax_env.reset(key, num_envs=1)
    print("Pileup Poker terminal mode")
    print("Fill the 4x4 board in 4 rounds by playing 4 cards out of each 5-card hand.")
    print("Card slot uses current hand index [0..4]. Board positions are [0..15] row-major.")

    try:
        while not bool(np.asarray(state.done[0])):
            current_round = int(np.asarray(state.current_round[0])) + 1
            current_move = int(np.asarray(state.num_moves[0])) + 1
            current_score = float(np.asarray(jax_env.calculate_total_score(state.grid, state.discard)[0]))
            print(f"\nRound {current_round}/{jax_env.ROUNDS}, move {current_move}/{jax_env.PLAYS_PER_ROUND}")
            print(f"Current score snapshot: {current_score:.2f}")
            _print_board_with_indices(state)
            _print_hand(state)
            _print_valid_actions(state)
            action = _prompt_action(state)
            state, reward, _ = jax_env.step(state, jnp.array([action], dtype=jnp.int32))
            print(f"Step reward: {float(np.asarray(reward[0])):.2f}")
    except KeyboardInterrupt:
        print("\nExiting game.")
    finally:
        done = bool(np.asarray(state.done[0]))
        if done:
            final_score = float(np.asarray(jax_env.calculate_total_score(state.grid, state.discard)[0]))
            print("\nFinal board:")
            _print_board_with_indices(state)
            print(f"Final score: {final_score:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Play Pileup Poker as a human in the terminal")
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed for reproducible deck order")
    args = parser.parse_args()
    run(seed=args.seed)


if __name__ == "__main__":
    main()
