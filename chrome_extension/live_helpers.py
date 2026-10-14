"""Shared helpers of the live extension tests (verify_live_extension.py, verify_live_compete.py)."""
import json
import os
import subprocess

# What is drawn on screen: a card belongs to a board cell if its center lies on that placeholder, to the
# hand if it lies in the tray below the board, and otherwise (above the tray) to the discard pile. The
# face-down draw pile is invisible and rendered off-screen below the tray, so it is never read; the game's
# "ethereal" preview copies of cards are skipped too.
READ_SCREEN = """() => {
  const center = (e) => { const r = e.getBoundingClientRect(); return [r.x + r.width / 2, r.y + r.height / 2]; };
  const inside = (p, e) => { const r = e.getBoundingClientRect(); return p[0] >= r.left && p[0] <= r.right && p[1] >= r.top && p[1] <= r.bottom; };
  const cells = [];
  for (let r = 3; r >= 0; r--) for (let c = 0; c < 4; c++) cells.push(document.getElementById(`card-PLACEHOLDER_tableau-${r}_${c}`));
  const trayTop = cells[12].getBoundingClientRect().bottom, trayBottom = trayTop + 2.5 * cells[12].getBoundingClientRect().height;
  const board = Array(16).fill(null), discard = [], hand = [];
  for (const e of document.querySelectorAll('[id^="card-"]:not([id*="PLACEHOLDER"]):not([id*="ethereal"])')) {
    if (getComputedStyle(e).opacity === "0") continue;
    const p = center(e), id = e.id.slice(5), i = cells.findIndex((c) => inside(p, c));
    if (i >= 0) board[i] = id;
    else if (p[1] < trayTop) discard.push(id);
    else if (p[1] < trayBottom) hand.push([p[0], id]);
  }
  return { board, discard, hand: hand.sort((a, b) => a[0] - b[0]).map((h) => h[1]) };
}"""

# Scores a final board with jax_env in the project environment (.venv); Puzzmo card ids are "H-10", ace = 1.
ENV_SCORE = """
import json, sys, jax.numpy as jnp
from pileup_poker import jax_env as J
from pileup_poker.pileup_poker import DECK
RANK = {"1": "A", "10": "T", "11": "J", "12": "Q", "13": "K"}
card = lambda c: DECK.index(RANK.get(c.split("-")[1], c.split("-")[1]) + c.split("-")[0])
b, d = json.loads(sys.argv[1])
print(float(J.calculate_total_score(jnp.array([[card(c) for c in b]]), jnp.array([[card(c) for c in d]]))[0]))
"""


def env_score(board: list, discard: list) -> float:
    """jax_env's total score of a final board (16 Puzzmo card ids) and discard pile (4 ids)."""
    env = {k: v for k, v in os.environ.items() if k != "LD_LIBRARY_PATH"} | {"JAX_PLATFORMS": "cpu"}
    cmd = [".venv/bin/python", "-c", ENV_SCORE, json.dumps([board, discard])]
    return float(subprocess.run(cmd, capture_output=True, text=True, check=True, env=env).stdout.split()[-1])
