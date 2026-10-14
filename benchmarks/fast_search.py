"""Score and cost of cheaper value-search settings on the compare_agents.py decks.

For each config: final scores on --num-decks decks (seed 0, the same decks as compare_agents.py),
value-model calls per game and round, and seconds per game after compilation (the first batch is
played once before timing). Paired differences are against per-deck scores of reference agents
in a compare_agents.py CSV (--reference-csv).

    python -m benchmarks.fast_search --configs card_K8S32 beam64_K8S32 --num-decks 1000

Per-deck scores and calls go to benchmarks/results/fast_search/<config>.npz, the summary to
benchmarks/results/fast_search.json (one entry per config, updated in place).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time

import numpy as np

from compare_agents import make_decks
from pileup_poker import afterstate as A
from pileup_poker import value_search as VS

OUT_DIR = "benchmarks/results/fast_search"
SUMMARY = "benchmarks/results/fast_search.json"
CARD = dict(inner="card")
CHEAP = dict(inner="card", exact_last=True)
HALVING_16 = ((16, 8), (8, 8), (4, 16), (2, 32))  # 16 candidates, up to 64 copies each

CONFIGS = {
    "round_K8S32": dict(num_candidates=8, num_samples=32),
    "argmax": dict(num_samples=0),
    "card_K8S32": dict(CARD, num_candidates=8, num_samples=32),
    "card_K16S64": dict(CARD, num_candidates=16, num_samples=64),
    "card_K32S64": dict(CARD, num_candidates=32, num_samples=64),
    "card_K8S32_sym": dict(CARD, num_candidates=8, num_samples=32, round1_symmetry=True),
    "card_K8S32_exact": dict(CHEAP, num_candidates=8, num_samples=32),
}
for _w in (8, 16, 32, 64, 128, 256):
    CONFIGS[f"beam{_w}_K8S32"] = dict(CHEAP, outer="beam", beam_width=_w, num_candidates=8, num_samples=32)
    CONFIGS[f"beam{_w}_argmax"] = dict(outer="beam", beam_width=_w, num_samples=0)
for _w in (2, 4, 8):
    CONFIGS[f"ibeam{_w}_K8S32"] = dict(CHEAP, inner="beam", inner_width=_w, num_candidates=8, num_samples=32)
HALVING_8 = ((8, 8), (4, 8), (2, 16))
CHEAP_BEAM = dict(CHEAP, outer="beam", beam_width=16)
CONFIGS["b16_hA16"] = dict(CHEAP_BEAM, halving=HALVING_16)
CONFIGS["b16_P1"] = dict(CHEAP_BEAM, halving_per_round=(HALVING_16, HALVING_8, HALVING_16))
CONFIGS["b16_P1_h128inner"] = dict(CONFIGS["b16_P1"], inner_model_path="models/td_h128/afterstate_params.pkl")
CONFIGS["b16_P1_h64inner"] = dict(CONFIGS["b16_P1"], inner_model_path="models/td_h64/afterstate_params.pkl")
CONFIGS["b8_P1_h64inner"] = dict(CONFIGS["b16_P1_h64inner"], beam_width=8)
CONFIGS["afterstate"] = dict(agent="afterstate")
# Legal afterstates per round of the card-by-card agent (5 cards x e empty cells, then 4 x (e - 1), ...);
# the last placement of the game is the exact final score.
AFTERSTATE_CALLS = np.array([sum((5 - k) * (e - k) for k in range(4)) for e in (16, 12, 8, 4)]) - np.array([0, 0, 0, 2])


def _flops_per_call(payload) -> int:
    """Multiply-adds of one value-model call (MLP layers only; features not counted)."""
    if payload["model"] != "mlp":
        return 0
    return sum(int(np.prod(w.shape)) for k, w in payload["params"].items() if k.startswith("w"))


def reference_scores(path: str) -> dict:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0] if k != "deck"}


def paired(scores, base):
    n = min(len(scores), len(base))  # references may cover fewer decks
    d = scores[:n] - base[:n]
    n = len(d)
    return dict(diff=float(d.mean()), diff_se=float(d.std(ddof=1) / np.sqrt(n)),
                wtl=[int((d > 0).sum()), int((d == 0).sum()), int((d < 0).sum())])


def run(name: str, options: dict, model_path: str, decks, game_batch: int, refs: dict, refs_used: list[str],
        seed: int = 0) -> dict:
    options = dict(options)
    if options.get("agent") == "afterstate":  # the plain card-by-card afterstate agent
        A.evaluate_on_decks(model_path, decks[:512])  # compile
        start = time.time()
        scores = A.evaluate_on_decks(model_path, decks)
        seconds = time.time() - start
        calls = np.stack([np.tile(AFTERSTATE_CALLS, (len(decks), 1)), np.zeros((len(decks), 4), int)], axis=-1)
    else:
        VS.evaluate_on_decks(model_path, decks[:game_batch], game_batch=game_batch, **options)  # compile
        start = time.time()
        scores, calls = VS.evaluate_on_decks(model_path, decks, game_batch=game_batch, return_calls=True, seed=seed,
                                             **options)
        seconds = time.time() - start
    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez(os.path.join(OUT_DIR, f"{name}.npz"), scores=scores, calls=calls)
    payload = A.load_params(model_path)
    macs = _flops_per_call(payload)
    n = len(scores)
    result = dict(
        options=options, model=model_path, num_decks=n, search_seed=seed,
        mean=float(scores.mean()), se=float(scores.std(ddof=1) / np.sqrt(n)),
        calls_per_game=float(calls[..., 0].sum(1).mean()), calls_per_round=calls[..., 0].mean(0).tolist(),
        inner_model_calls_per_game=float(calls[..., 1].sum(1).mean()),
        macs_per_call=macs, gflop_per_game=2 * macs * float(calls[..., 0].sum(1).mean()) / 1e9,
        seconds_per_game=seconds / n, game_batch=game_batch,
        paired={ref: paired(scores, refs[ref]) for ref in refs_used},
        mean_first_1000=float(scores[:1000].mean()),
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--configs", nargs="+", required=True, help=f"names from {list(CONFIGS)} or NAME=JSON options")
    parser.add_argument("--model", default="models/afterstate_mlp/afterstate_params.pkl")
    parser.add_argument("--tag", default="", help="Suffix for result names (e.g. the model).")
    parser.add_argument("--num-decks", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0, help="Deck seed.")
    parser.add_argument("--search-seed", type=int, default=0, help="Seed of the next-hand samples.")
    parser.add_argument("--game-batch", type=int, default=40)
    parser.add_argument("--reference-csv", default="logs/reference_1000.csv",
                        help="Per-deck scores written by compare_agents.py --out-csv on the same decks.")
    parser.add_argument("--references", nargs="+", help="Column names (agent specs) in --reference-csv; the first is "
                        "the baseline of the printed difference.", default=[
                            "search_value=models/afterstate_mlp/afterstate_params.pkl",
                            "afterstate=models/afterstate_mlp/afterstate_params.pkl"])
    args = parser.parse_args()

    decks = make_decks(args.num_decks, args.seed)
    refs = reference_scores(args.reference_csv)
    summary = json.load(open(SUMMARY)) if os.path.exists(SUMMARY) else {}
    for spec in args.configs:
        name, _, opts = spec.partition("=")
        options = json.loads(opts) if opts else CONFIGS[name]
        if "halving" in options:
            options["halving"] = tuple(map(tuple, options["halving"]))
        if "halving_per_round" in options:
            options["halving_per_round"] = tuple(tuple(map(tuple, h)) for h in options["halving_per_round"])
        name = name + args.tag
        result = run(name, options, args.model, decks, args.game_batch, refs, args.references, args.search_seed)
        pair = result["paired"][args.references[0]]
        print(f"{name}: {result['mean']:.0f} ± {result['se']:.0f}  vs {args.references[0]} {pair['diff']:+.0f} ± "
              f"{pair['diff_se']:.0f} W/T/L {pair['wtl']}  calls/game {result['calls_per_game']:.0f} "
              f"{[round(c) for c in result['calls_per_round']]} inner model {result['inner_model_calls_per_game']:.0f}  "
              f"{result['seconds_per_game']:.3f} s/game", flush=True)
        summary = json.load(open(SUMMARY)) if os.path.exists(SUMMARY) else {}
        summary[name] = result
        with open(SUMMARY, "w") as f:
            json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
