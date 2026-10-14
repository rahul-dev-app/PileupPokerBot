"""Render docs/images/greedy_vs_value_search.gif: the replay of the greedy baseline against the value search bot.

    python -m http.server 8811 -d chrome_extension &   # serve the folder
    python chrome_extension/store/make_readme_gif.py    # needs Playwright and Pillow
"""
import asyncio
import io
import time

from PIL import Image
from playwright.async_api import async_playwright

URL = "http://127.0.0.1:8811/store/readme_scene.html"
OUT = "docs/images/greedy_vs_value_search.gif"
HOLD = {"Round 1": 1.4, "Round 2": 1.4, "Round 3": 1.4, "Round 4": 1.4, "Final": 3.5}
ANIMATION = 0.9  # seconds of frames captured after each step change


async def main() -> None:
    frames, durations = [], []
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": 1024, "height": 900})
        await page.goto(URL)
        await page.wait_for_selector("body[data-ready='1']")
        await page.wait_for_timeout(500)
        app = page.locator(".ppr-app")
        for step, hold in HOLD.items():
            await page.get_by_role("button", name=step, exact=True).click()
            start = last = time.monotonic()
            while time.monotonic() - start < ANIMATION:
                frames.append(Image.open(io.BytesIO(await app.screenshot())).convert("RGB"))
                now = time.monotonic()
                durations.append(now - last)
                last = now
            await page.wait_for_timeout(200)
            frames.append(Image.open(io.BytesIO(await app.screenshot())).convert("RGB"))
            durations.append(hold)
        await browser.close()
    durations = durations[1:] + [durations[0]]  # each frame is shown until the next one was taken
    size = (frames[0].width * 2 // 3, frames[0].height * 2 // 3)
    small = [f.resize(size, Image.LANCZOS).quantize(colors=128, method=Image.Quantize.MEDIANCUT) for f in frames]
    small[0].save(OUT, save_all=True, append_images=small[1:], duration=[max(40, int(d * 1000)) for d in durations], loop=0, optimize=True)
    print(f"wrote {OUT}: {len(frames)} frames, {size[0]}x{size[1]}")


if __name__ == "__main__":
    asyncio.run(main())
