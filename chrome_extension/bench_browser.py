"""Speed of the in-browser value search inside the extension's service worker (headless Chrome for Testing).

    .tools-venv/bin/python -m chrome_extension.bench_browser [--games 20]

Loads chrome_extension/extension unpacked, waits for the networks, and in the service worker times:
games of the search on the benchmark decks (per round, single thread), compete-mode bot game + analysis
(assistant.js) for a player who places the first 4 cards of each hand, and the suggest-mode searches
at round starts. Decks: /tmp/pp_decks.json (python -m chrome_extension.bench_decks).
"""
import argparse
import asyncio
import json
import shutil
from pathlib import Path

from playwright.async_api import async_playwright

EXT = str(Path(__file__).parent / "extension")
PROFILE = "/tmp/pp_ext_bench_profile"

BENCH = """async ({decks}) => {
  const nets = await ready;
  const P = PPPolicy, S = PPSearch, A = PPAssistant;
  const rounds = [[], [], [], []], games = [], bot = [], scores = [];
  for (const [i, deck] of decks.entries()) {
    let grid = new Int8Array(16).fill(-1), discard = new Int8Array(4).fill(-1);
    const g0 = performance.now();
    for (let r = 0; r < 4; r++) {
      const t0 = performance.now();
      const res = S.searchRound(nets, {grid, discard, hand: deck.slice(5 * r, 5 * r + 5), round: r}, {random: S.rng(i, r)});
      rounds[r].push(performance.now() - t0);
      grid = res.grid; discard = res.discard;
    }
    games.push(performance.now() - g0);
    scores.push(S.totalScore(grid, 0, discard, 0));
    const hands = [0, 1, 2, 3].map((r) => deck.slice(5 * r, 5 * r + 5));
    const req = {hands: hands.map((h) => h.map(P.puzzmoId)), board: hands.flatMap((h) => h.slice(0, 4)).map(P.puzzmoId)};
    const t0 = performance.now();
    A.botGame(nets, req);
    bot.push(performance.now() - t0);
  }
  return {rounds, games, bot, scores, userAgent: navigator.userAgent};
}"""


def summary(xs):
    xs = sorted(xs)
    return {"mean_ms": round(sum(xs) / len(xs), 1), "median_ms": round(xs[len(xs) // 2], 1), "max_ms": round(xs[-1], 1)}


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=20)
    args = parser.parse_args()
    decks = json.load(open("/tmp/pp_decks.json"))[: args.games]
    shutil.rmtree(PROFILE, ignore_errors=True)
    async with async_playwright() as p:
        ctx = await p.chromium.launch_persistent_context(
            PROFILE, channel="chromium", headless=True,
            args=[f"--disable-extensions-except={EXT}", f"--load-extension={EXT}"])
        worker = ctx.service_workers[0] if ctx.service_workers else await ctx.wait_for_event("serviceworker")
        await worker.evaluate(BENCH, {"decks": decks[:2]})  # warm-up (JIT)
        res = await worker.evaluate(BENCH, {"decks": decks})
        await ctx.close()
    out = {
        "context": "extension service worker, headless Chrome for Testing, single thread",
        "user_agent": res["userAgent"],
        "games": len(decks),
        "game": summary(res["games"]),
        "rounds": [summary(r) for r in res["rounds"]],
        "bot_game_with_analysis": summary(res["bot"]),
        "mean_score": sum(res["scores"]) / len(decks),
    }
    print(json.dumps(out, indent=1))
    Path("benchmarks/results/browser_search_speed_chrome.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
