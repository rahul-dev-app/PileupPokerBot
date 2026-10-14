"""Plot env throughput vs batch size from benchmarks/results/*.json (needs matplotlib).

    python benchmarks/plot_env_speed.py   # writes docs/images/env_throughput.png
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "benchmarks", "results")


def load(name):
    with open(os.path.join(RES, name)) as f:
        return json.load(f)


fig, ax = plt.subplots(figsize=(7, 4.5))
for backend, color in (("gpu", "tab:green"), ("cpu", "tab:blue")):
    rows = load(f"jax_env_{backend}.json")["results"]
    for mode, style, label in (("scan", "o-", "lax.scan over 16 steps"), ("loop", "s--", "one call per step")):
        pts = sorted((r["batch"], r["median_steps_per_s"]) for r in rows if r["mode"] == mode)
        ax.plot(*zip(*pts), style, color=color, label=f"JAX {backend.upper()}, {label}")
py = load("python_env.json")
ax.axhline(16 * py["single_process"]["median_games_per_s"], color="tab:red", label="Python env, 1 process")
if "multi_process" in py:
    ax.axhline(16 * py["multi_process"]["median_games_per_s"], color="tab:red", ls=":",
               label=f"Python env, {py['multi_process']['processes']} processes")
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xlabel("batch size (games in parallel)")
ax.set_ylabel("env steps per second")
ax.set_title("Pile-Up Poker: random legal actions, full games")
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=8)
fig.tight_layout()
out = os.path.join(ROOT, "docs", "images", "env_throughput.png")
fig.savefig(out, dpi=150)
print(f"wrote {out}")
