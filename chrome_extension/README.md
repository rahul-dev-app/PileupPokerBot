# PokerGrid 4x4 (Chrome extension)

PokerGrid 4x4 is a Chrome extension for [Pile-Up Poker on Puzzmo](https://www.puzzmo.com/game/pile-up-poker) with two modes, chosen in its popup. It never plays for you. Not affiliated with Puzzmo.

- **Compete** (default): no suggestions; after the game, the bot plays the same hands and a "You vs Bot" replay with a check of each round opens over the game.
- **Suggest**: the bot's plan for the round while you play.

The bot is the cheap value search. It runs in the extension's service worker, in JavaScript with a WebAssembly SIMD kernel: no server, and no network requests unless you set an analytics endpoint in the popup.

Install: open `chrome://extensions`, turn on Developer mode, click "Load unpacked" and select `chrome_extension/extension/`. `python chrome_extension/package.py` builds a zip in `chrome_extension/dist/`.

Documentation (modes, replay, analytics, files, tests): [docs/chrome-extension.md](../docs/chrome-extension.md). The algorithm: [docs/algorithm.md](../docs/algorithm.md). Privacy policy: [docs/privacy.md](../docs/privacy.md).
