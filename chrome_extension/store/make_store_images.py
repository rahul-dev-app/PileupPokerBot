"""Render the Chrome Web Store images into chrome_extension/store/images/ (needs Playwright).

    python -m http.server 8811 -d chrome_extension &   # serve the folder
    python chrome_extension/store/make_store_images.py
"""
import asyncio

from playwright.async_api import async_playwright

BASE = "http://127.0.0.1:8811"
OUT = "chrome_extension/store/images/"


async def main() -> None:
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": 1280, "height": 800})
        await page.emulate_media(reduced_motion="reduce")
        await page.goto(f"{BASE}/replay_demo.html?data=store/sample_game.json")
        await page.wait_for_timeout(1500)
        await page.add_style_tag(content=".demo-bar{display:none}")
        for label, name in [("Final", "1_replay_final"), ("Round 2", "2_replay_round2")]:
            await page.get_by_role("button", name=label, exact=True).click()
            await page.wait_for_timeout(1500)
            await page.evaluate("window.scrollTo(0, 0)")
            await page.screenshot(path=f"{OUT}{name}.png")
        await page.get_by_role("button", name="Final", exact=True).click()
        await page.wait_for_timeout(1000)
        await page.get_by_text("Your rounds, checked by the bot").scroll_into_view_if_needed()
        await page.evaluate("window.scrollBy(0, -20)")
        await page.screenshot(path=f"{OUT}3_replay_analysis.png")
        await page.goto(f"{BASE}/store/popup_scene.html")
        await page.wait_for_timeout(800)
        await page.screenshot(path=f"{OUT}4_modes.png")
        tile = await browser.new_page(viewport={"width": 440, "height": 280})
        await tile.goto(f"{BASE}/store/promo_tile.html")
        await tile.wait_for_timeout(500)
        await tile.screenshot(path=f"{OUT}promo_tile_440x280.png")
        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
