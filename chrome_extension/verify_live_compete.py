"""Live check of the extension's compete mode on Puzzmo (test only: it clicks for the player).

    .tools-venv/bin/python -m chrome_extension.verify_live_compete [--analytics]

Loads chrome_extension/extension unpacked into headless Chrome for Testing, selects "Compete" in the
popup, and plays a whole guest game with the first legal moves (the first 4 cards of each hand into the
first empty cells). Checks that no suggestion appears during play, that the replay overlay opens after
the game, that it shows both games (16 cards each), that the replay's final score for the player equals
the game's total score and jax_env's score, and that the bot's final score equals the JS scoring
(policy.js) of its board. The bot runs in the extension: every request to a local server is recorded
(there must be none). Screenshots of the replay go to docs/images/.
With --analytics the test starts a small analytics receiver on 127.0.0.1:8791, sets it as the
extension's analytics endpoint, and checks that the one event it receives has the game's scores.
"""
import asyncio
import json
import re
import shutil
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from playwright.async_api import async_playwright

from chrome_extension.live_helpers import READ_SCREEN, env_score

EXT = str(Path(__file__).parent / "extension")
PROFILE = "/tmp/pp_ext_compete_profile"
IMAGES = Path(__file__).parent.parent / "docs" / "images"
LOGS = Path(__file__).parent.parent / "logs" / "verify_live"
ANALYTICS_PORT = 8791
CATCH_RESULT = """() => window.addEventListener('message', (e) => {
  const m = e.data && e.data.ppExt;
  if (m && m.type === 'bot_game_result') window.__ppBotGame = m;
})"""
BOT_BOARD_SCORE = """() => {
  const d = window.__ppBotGame.data, board = Array(16).fill(-1), discard = [];
  d.bot.rounds.forEach((r) => { r.placements.forEach((p) => (board[p.cell] = PPPolicy.parseCard(p.card)));
                                discard.push(PPPolicy.parseCard(r.discard)); });
  return PPPolicy.totalScore(board, discard);
}"""


async def suggestion_shown(fr) -> bool:
    return await fr.evaluate("""() => ((document.getElementById('pp-assistant') || {}).textContent || '').includes('Best:')
                                || !!document.querySelector('[data-pp-hl]')""")


def start_analytics_receiver(events: list) -> HTTPServer:
    """Collects the JSON bodies of POST /analytics (with the CORS answers a fetch from the extension needs)."""
    class Handler(BaseHTTPRequestHandler):
        def _headers(self, code):
            self.send_response(code)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Allow-Private-Network", "true")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_OPTIONS(self):
            self._headers(204)

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            if self.path == "/analytics":
                events.append(json.loads(body))
            self._headers(204 if self.path == "/analytics" else 404)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", ANALYTICS_PORT), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


async def main() -> None:
    analytics = "--analytics" in sys.argv
    shutil.rmtree(PROFILE, ignore_errors=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    problems, local_requests, events = [], [], []
    receiver = start_analytics_receiver(events) if analytics else None
    async with async_playwright() as p:
        ctx = await p.chromium.launch_persistent_context(
            PROFILE, channel="chromium", headless=True, viewport={"width": 1280, "height": 1000},
            args=[f"--disable-extensions-except={EXT}", f"--load-extension={EXT}"])

        def on_request(request):  # pages and the extension's service worker
            if re.match(r"https?://(127\.0\.0\.1|localhost)", request.url):
                local_requests.append(request.url)
        ctx.on("request", on_request)
        worker = ctx.service_workers[0] if ctx.service_workers else await ctx.wait_for_event("serviceworker")
        ext_id = worker.url.split("/")[2]
        popup = await ctx.new_page()
        await popup.set_viewport_size({"width": 300, "height": 400})
        await popup.goto(f"chrome-extension://{ext_id}/popup.html")
        await popup.wait_for_timeout(500)
        await popup.check('input[value="compete"]')
        await popup.wait_for_timeout(300)
        await popup.screenshot(path=str(IMAGES / "popup.png"))
        await popup.close()
        if analytics:  # what the popup's Save button stores (it also asks for the origin's permission)
            await worker.evaluate(f"chrome.storage.sync.set({{analyticsUrl: 'http://127.0.0.1:{ANALYTICS_PORT}'}})")

        page = await ctx.new_page()
        await page.goto("https://www.puzzmo.com/game/pile-up-poker", wait_until="domcontentloaded", timeout=60000)
        await page.wait_for_timeout(8000)
        await page.get_by_text("Play Today").first.click()
        await page.wait_for_timeout(8000)
        fr = next(f for f in page.frames if "runtime.html" in f.url)
        await fr.evaluate(CATCH_RESULT)
        for r in range(4):
            for _ in range(50):
                hand = (await fr.evaluate(READ_SCREEN))["hand"]
                if len(hand) == 5:
                    break
                await fr.wait_for_timeout(300)
            await fr.wait_for_timeout(800)
            hand = (await fr.evaluate(READ_SCREEN))["hand"]
            for card in hand[:4]:
                board = (await fr.evaluate(READ_SCREEN))["board"]
                cell = board.index(None)
                await fr.locator("#card-" + card).click()
                await fr.wait_for_timeout(400)
                await fr.locator(f"#card-PLACEHOLDER_tableau-{3 - cell // 4}_{cell % 4}").click(force=True)
                await fr.wait_for_timeout(600)
                if await suggestion_shown(fr):
                    problems.append(f"round {r + 1}: a suggestion is shown in compete mode")
            print(f"round {r + 1}: hand {hand}, panel: {await fr.evaluate('(document.getElementById(`pp-assistant`) || {}).textContent')}", flush=True)
            await fr.wait_for_timeout(1500)
            button = "Finish" if r == 3 else "Next round"
            await fr.get_by_text(re.compile(f"^{button}$", re.I)).first.click(force=True)
            await fr.wait_for_timeout(1500)

        screen = await fr.evaluate(READ_SCREEN)
        await fr.wait_for_selector("#pp-overlay", timeout=10000)
        await fr.wait_for_selector("#pp-overlay .ppr-app", timeout=120000)
        app = fr.locator("#pp-overlay .ppr-app")
        await fr.wait_for_timeout(9000)  # autoplay reaches the final step
        await page.screenshot(path=str(LOGS / "compete_overlay_page.png"))
        await page.set_viewport_size({"width": 1280, "height": 2200})  # a tall game frame fits the whole replay
        await fr.wait_for_timeout(1500)
        for name, tab in [("round1", "Round 1"), ("round3", "Round 3"), ("final", "Final")]:
            await fr.locator(".ppr-tab", has_text=tab).click()
            await fr.wait_for_timeout(1500)
            await app.screenshot(path=str(IMAGES / f"compete_replay_{name}.png"))
        cards = await fr.evaluate("""() => ['you', 'bot'].map((s) => document.querySelectorAll(`.ppr-${s} .ppr-board .ppr-card`).length)""")
        replay_score = await fr.evaluate("Number(document.querySelector('.ppr-you .ppr-side-score').dataset.value)")
        header = await fr.evaluate("document.querySelector('.ppr-summary').textContent")
        result = await fr.evaluate("window.__ppBotGame")
        bot_board_score = await fr.evaluate(BOT_BOARD_SCORE)
        await fr.locator(".ppr-analysis").screenshot(path=str(IMAGES / "compete_replay_analysis.png"))
        await fr.locator(".ppr-close").click()
        await fr.wait_for_timeout(3000)
        page_text = await fr.evaluate("document.body.innerText")
        await page.screenshot(path=str(LOGS / "compete_final_game.png"))
        await ctx.close()
    total = re.search(r"Total score\s*\$?\s*([\d,]+)", page_text)
    shown = int(total.group(1).replace(",", "")) if total else None
    ours = env_score(screen["board"], screen["discard"])
    data = result["data"]
    if cards != [16, 16]:
        problems.append(f"replay boards have {cards} cards")
    if data["bot"]["final_score"] != bot_board_score:
        problems.append(f"bot score {data['bot']['final_score']} != JS score of its board {bot_board_score}")
    if data["player"]["final_score"] != ours:
        problems.append(f"bot_game player score {data['player']['final_score']} != {ours}")
    if analytics:
        receiver.shutdown()
        print(f"analytics events: {events}")
        expected = {"version": json.load(open(Path(EXT) / "manifest.json"))["version"], "player_score": data["player"]["final_score"],
                    "bot_score": data["bot"]["final_score"], "total_loss": data["analysis"]["total_loss"]}
        if events != [expected]:
            problems.append(f"analytics events {events} != [{expected}]")
        local_requests = [u for u in local_requests if not u.startswith(f"http://127.0.0.1:{ANALYTICS_PORT}/analytics")]
    if local_requests:
        problems.append(f"requests to local servers: {local_requests}")
    print(f"replay header {header!r}; bot game computed in the extension in {data['ms']} ms; replay player score "
          f"{replay_score}; game total score {shown}; jax_env score {ours:.0f}; bot score {data['bot']['final_score']} "
          f"(JS score of its board {bot_board_score}); problems: {problems}")
    print("RESULT", "PASS" if not problems and shown == replay_score == ours else "FAIL")


if __name__ == "__main__":
    asyncio.run(main())
