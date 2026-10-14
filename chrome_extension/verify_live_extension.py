"""Live check of the Chrome extension on Puzzmo (test only: it clicks for the player).

    .tools-venv/bin/python -m chrome_extension.verify_live_extension

Loads chrome_extension/extension unpacked into headless Chrome for Testing, opens Pile-Up Poker as a
guest and plays a whole game in suggest mode by always following the extension's panel. In round 2 it discards the
plan's discard card early to test that path. Checks that the panel updates after every move, that
every suggested move can be played, and compares the game's final total score with jax_env's score.
"""
import asyncio
import re
import shutil
from pathlib import Path

from playwright.async_api import async_playwright

from chrome_extension.live_helpers import READ_SCREEN, env_score

EXT = str(Path(__file__).parent / "extension")
PROFILE = "/tmp/pp_ext_profile"
OUT = "/tmp/pp_ext_"
SUITS = {"\u2660": "S", "\u2665": "H", "\u2666": "D", "\u2663": "C"}
RANKS = {"A": "1", "J": "11", "Q": "12", "K": "13"}


def puzzmo(label: str) -> str:
    return SUITS[label[-1]] + "-" + RANKS.get(label[:-1], label[:-1])


async def panel_text(fr) -> str:
    return await fr.evaluate("(document.getElementById('pp-assistant') || {}).textContent || ''")


async def wait_panel(fr, previous: str) -> str:
    for _ in range(100):
        text = await panel_text(fr)
        if "Best:" in text and text != previous:
            return text
        await fr.wait_for_timeout(200)
    raise RuntimeError(f"panel did not update (last: {text!r})")


async def main() -> None:
    shutil.rmtree(PROFILE, ignore_errors=True)
    problems = []
    async with async_playwright() as p:
        ctx = await p.chromium.launch_persistent_context(
            PROFILE, channel="chromium", headless=True, viewport={"width": 1280, "height": 900},
            args=[f"--disable-extensions-except={EXT}", f"--load-extension={EXT}"])
        # The extension starts in compete mode: select suggest mode in its popup.
        worker = ctx.service_workers[0] if ctx.service_workers else await ctx.wait_for_event("serviceworker")
        popup = await ctx.new_page()
        await popup.goto(f"chrome-extension://{worker.url.split('/')[2]}/popup.html")
        await popup.check('input[value="suggest"]')
        await popup.wait_for_timeout(500)
        await popup.close()
        page = await ctx.new_page()
        await page.goto("https://www.puzzmo.com/game/pile-up-poker", wait_until="domcontentloaded", timeout=60000)
        await page.wait_for_timeout(8000)
        await page.get_by_text("Play Today").first.click()
        await page.wait_for_timeout(8000)
        fr = next(f for f in page.frames if "runtime.html" in f.url)
        text = ""
        for r in range(4):
            for k in range(4):
                text = await wait_panel(fr, text)
                if k == 0:
                    print(text, flush=True)
                    await page.screenshot(path=f"{OUT}round{r + 1}.png")
                if r == 1 and k == 0:  # early discard: select the plan's discard card, then the discard pile
                    discard = puzzmo(re.search(r"Discard: (\S+)", text).group(1))
                    await fr.locator("#card-" + discard).click()
                    await fr.wait_for_timeout(400)
                    await fr.locator('[id^="card-PLACEHOLDER_discard"]').first.click(force=True)
                    await fr.wait_for_timeout(1500)  # the panel may stay the same: this was the plan's discard
                    if discard not in (await fr.evaluate(READ_SCREEN))["discard"]:
                        problems.append("early discard not on the discard pile")
                    text = await panel_text(fr)
                    print("after early discard:\n" + text, flush=True)
                    if discard in [puzzmo(m) for m in re.findall(r"(\S+)@r", text)]:
                        problems.append("plan uses the discarded card")
                best = re.search(r"Best: (\S+) -> row (\d) col (\d)", text)
                card, row, col = puzzmo(best.group(1)), int(best.group(2)) - 1, int(best.group(3)) - 1
                await fr.locator("#card-" + card).click()
                await fr.wait_for_timeout(400)
                await fr.locator(f"#card-PLACEHOLDER_tableau-{3 - row}_{col}").click(force=True)
                await fr.wait_for_timeout(600)
                board = (await fr.evaluate(READ_SCREEN))["board"]
                if board[4 * row + col] != card:
                    problems.append(f"round {r + 1} move {k + 1}: {card} not at row {row + 1} col {col + 1}")
            await fr.wait_for_timeout(1500)
            if r == 3:  # the final board, before the end screen changes the layout; the hand card is the last discard
                screen = await fr.evaluate(READ_SCREEN)
                screen["discard"] += screen["hand"]
            button = "Finish" if r == 3 else "Next round"
            await fr.get_by_text(re.compile(f"^{button}$", re.I)).first.click(force=True)
            await fr.wait_for_timeout(1500)
        shown, previous = None, -1
        for _ in range(30):  # the end screen counts the total score up: wait until it stops changing
            await fr.wait_for_timeout(1000)
            total = re.search(r"Total score\s*\$\s*([\d,]+)", await fr.evaluate("document.body.innerText"))
            shown = int(total.group(1).replace(",", "")) if total else None
            if shown is not None and shown == previous:
                break
            previous = shown
        await page.screenshot(path=f"{OUT}final.png")
        await ctx.close()
    ours = env_score(screen["board"], screen["discard"])
    print("final board and discard pile:", screen["board"], screen["discard"])
    print(f"game total score {shown}; jax_env score of the final board {ours:.0f}; problems: {problems}")
    print("RESULT", "PASS" if not problems and shown == ours else "FAIL")


if __name__ == "__main__":
    asyncio.run(main())
