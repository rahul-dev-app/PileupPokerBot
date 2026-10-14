# Privacy policy: PokerGrid 4x4

PokerGrid 4x4 is the Chrome extension in this repository ([documentation](chrome-extension.md)). It works with Pile-Up Poker on Puzzmo and is not affiliated with or endorsed by Puzzmo.

## What the extension reads

On `https://www.puzzmo.com/runtime.html*` (the game's frame) it reads the visible game state: the board, the cards in your hand, the discard pile and the round. It never reads the face-down draw pile. All computation (the bot's search, the suggestions and the replay) runs locally in your browser.

## What it stores

`chrome.storage.sync` holds only the settings: the mode (compete or suggest) and the analytics endpoint (empty by default). No game data is stored.

## What it sends

By default, nothing. The extension collects no data and makes no network requests.

The only exception is the optional analytics endpoint. If you enter a URL in the popup and click Save (Chrome then asks you for permission to send to that site), the extension sends one request to that URL at the end of each compete-mode game:

```
POST {endpoint}/analytics
{"version": "1.3.0", "player_score": 600, "bot_score": 7050, "total_loss": 6912.4}
```

These four fields are all that is sent: the extension version, your final score, the bot's final score and the total expected points lost in the analysis. No cards, no game state and no personal data are sent. Suggest mode never sends anything. Clear the field and click Save to stop sending.

## Contact

Questions about this policy: open an issue in the repository's [issue tracker](https://github.com/rahul-dev-app/PileupPokerBot/issues).
