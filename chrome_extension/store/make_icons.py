"""Draw the PokerGrid 4x4 icons: extension/icons/icon{16,48,128}.png and store/images/store_icon_128.png.

The store icon follows the Chrome Web Store layout: 96x96 artwork centered in a 128x128 transparent canvas.
    python chrome_extension/store/make_icons.py   # needs Pillow
"""
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).parent
S = 8  # supersampling

# 4x4 board: card color ("r" red, "k" black, "." empty, "h" highlighted empty cell)
BOARD = ["krkr", "rr.k", "khrr", "rk.k"]


def artwork(size: int, detail: bool) -> Image.Image:
    n = size * S
    img = Image.new("RGBA", (n, n), (0, 0, 0, 0))
    bg = Image.new("RGBA", (n, n))
    top, bottom = (40, 45, 64), (70, 52, 110)
    for y in range(n):
        t = y / (n - 1)
        ImageDraw.Draw(bg).line([(0, y), (n, y)], fill=tuple(int(a + (b - a) * t) for a, b in zip(top, bottom)) + (255,))
    mask = Image.new("L", (n, n), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, n - 1, n - 1], radius=int(n * 0.22), fill=255)
    img.paste(bg, (0, 0), mask)
    d = ImageDraw.Draw(img)
    margin, gap = n * 0.14, n * (0.04 if detail else 0.06)
    cell = (n - 2 * margin - 3 * gap) / 4
    for r, row in enumerate(BOARD):
        for c, kind in enumerate(row):
            x0, y0 = margin + c * (cell + gap), margin + r * (cell + gap)
            box = [x0, y0, x0 + cell, y0 + cell]
            rad = cell * 0.22
            if kind in ".h":
                d.rounded_rectangle(box, radius=rad, fill=(255, 255, 255, 40))
                if kind == "h":
                    d.rounded_rectangle(box, radius=rad, outline=(255, 43, 214, 255), width=max(S, int(cell * 0.14)))
                continue
            d.rounded_rectangle(box, radius=rad, fill=(255, 255, 255, 255))
            if detail:  # a pip in the card's color
                color = (208, 48, 47, 255) if kind == "r" else (29, 34, 48, 255)
                cx, cy, pr = x0 + cell / 2, y0 + cell / 2, cell * 0.2
                d.ellipse([cx - pr, cy - pr, cx + pr, cy + pr], fill=color)
    return img.resize((size, size), Image.LANCZOS)


def main() -> None:
    icons = HERE.parent / "extension" / "icons"
    for size in (16, 48, 128):
        artwork(size, detail=size >= 48).save(icons / f"icon{size}.png")
    store = Image.new("RGBA", (128, 128), (0, 0, 0, 0))
    store.paste(artwork(96, detail=True), (16, 16))
    store.save(HERE / "images" / "store_icon_128.png")
    print("icons written")


if __name__ == "__main__":
    main()
