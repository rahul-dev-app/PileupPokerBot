"""End-to-end PPO training throughput: train_jax_ppo.py with the Gym backend vs the JAX backend.

Runs short trainings, timestamps every "[update k/n]" log line and reports env steps per second
over the updates after the warm-up (compile time excluded). Writes benchmarks/results/ppo_speed.json.

    python benchmarks/ppo_speed.py            # all configurations below
    python benchmarks/ppo_speed.py gym_gpu_64 jax_gpu_64

The CPU configurations set JAX_PLATFORMS=cpu for the child process.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "benchmarks", "results", "ppo_speed.json")
WARMUP_UPDATES = 3

COMMON = ["--arch", "mlp", "--num-steps", "64", "--log-every", "1", "--eval-every", "1000000",
          "--save-every", "1000000"]
SMALL = ["--num-envs", "64", "--no-reward-shaping"]
# name: (extra args, measured updates after warm-up, cpu only)
CONFIGS = {
    "gym_gpu_64": (SMALL + ["--env-backend", "gym"], 10, False),
    "gym_cpu_64": (SMALL + ["--env-backend", "gym"], 10, True),
    "gym_cpu_1024": (["--num-envs", "1024", "--no-reward-shaping", "--env-backend", "gym"], 3, True),
    "jax_cpu_64": (SMALL + ["--env-backend", "jax"], 20, True),
    "jax_gpu_64": (SMALL + ["--env-backend", "jax"], 20, False),
    "jax_cpu_1024": (["--num-envs", "1024", "--env-backend", "jax", "--gamma", "1.0"], 5, True),
    "jax_gpu_1024": (["--num-envs", "1024", "--env-backend", "jax", "--gamma", "1.0"], 20, False),
}
UPDATE_RE = re.compile(r"\[update (\d+)/\d+\] timesteps=(\d+)")


def run(name: str) -> dict:
    extra, measured, cpu = CONFIGS[name]
    num_envs = int(extra[extra.index("--num-envs") + 1])
    updates = WARMUP_UPDATES + measured
    cmd = [sys.executable, "-u", os.path.join(ROOT, "train_jax_ppo.py"), *COMMON, *extra,
           "--total-timesteps", str(updates * num_envs * 64), "--run-name", f"bench_{name}",
           "--save-dir", f"/tmp/ppo_speed/{name}", "--tensorboard-logdir", "/tmp/ppo_speed/tb"]
    env = dict(os.environ)
    if cpu:
        env["JAX_PLATFORMS"] = "cpu"
    print(" ".join(cmd[1:]), flush=True)
    t_start = time.perf_counter()
    stamps = []  # (update, timesteps, time)
    with subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as proc:
        for line in proc.stdout:
            m = UPDATE_RE.search(line)
            if m:
                stamps.append((int(m.group(1)), int(m.group(2)), time.perf_counter() - t_start))
                print(f"  {stamps[-1][2]:7.1f}s {line.strip()[:90]}", flush=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{name} failed with exit code {proc.returncode}")
    (u0, s0, t0), (u1, s1, t1) = stamps[WARMUP_UPDATES - 1], stamps[-1]
    result = {"args": ["train_jax_ppo.py", *cmd[3:]], "jax_platforms": "cpu" if cpu else "default (gpu)", "num_envs": num_envs,
              "warmup_updates": WARMUP_UPDATES, "measured_updates": u1 - u0,
              "warmup_s": t0, "steps_per_s": (s1 - s0) / (t1 - t0), "update_times": stamps}
    print(f"{name}: {result['steps_per_s']:.0f} steps/s (warm-up {t0:.1f} s)", flush=True)
    return result


def main() -> None:
    names = sys.argv[1:] or list(CONFIGS)
    results = {}
    if os.path.exists(RESULTS):
        with open(RESULTS) as f:
            results = json.load(f)
    for name in names:
        results[name] = run(name)
        with open(RESULTS, "w") as f:
            json.dump(results, f, indent=2)
    print(f"wrote {RESULTS}")


if __name__ == "__main__":
    main()
