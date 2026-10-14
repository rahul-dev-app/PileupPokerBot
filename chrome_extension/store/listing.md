# Chrome Web Store listing: PokerGrid 4x4

Text and answers for the Chrome Web Store developer dashboard. Images are in `images/` (rebuild with `make_store_images.py`). Upload `chrome_extension/dist/pokergrid-4x4-<version>.zip` from `python chrome_extension/package.py`.

## Store listing

**Name** (from the manifest): PokerGrid 4x4

**Summary** (from the manifest): Play the 4x4 poker card game against a strong bot, then watch a round-by-round replay of your game vs the bot's.

**Category:** Games (or Entertainment)

**Language:** English

**Description:**

```
PokerGrid 4x4 adds a strong bot to the 4x4 poker card game Pile-Up Poker on Puzzmo.

COMPETE MODE (default)
Play the game on your own, with no help. When you finish, the bot plays your exact four hands and you get a replay of both games side by side:
• both boards fill in round by round, with every scoring hand labeled
• the final score breakdown for you and for the bot
• a round-by-round check of your game: how many expected points each of your rounds cost, and what the bot would have played from your position

SUGGEST MODE
See the bot's best move while you play, with its plan for the rest of the round and the card to discard.

HOW THE BOT WORKS
The bot is a search over whole rounds guided by a neural network trained with reinforcement learning. It never looks at face-down cards: future hands are sampled only from cards you cannot see. It runs entirely in your browser (JavaScript and WebAssembly). No server, no account.

PRIVACY
No data is collected and nothing is sent anywhere. The only settings stored are the mode and an optional analytics endpoint, which is empty by default.

Open source: https://github.com/rahul-dev-app/PileupPokerBot

PokerGrid 4x4 is not affiliated with or endorsed by Puzzmo.
```

**Screenshots** (1280x800): `images/1_replay_final.png`, `images/2_replay_round2.png`, `images/3_replay_analysis.png`, `images/4_modes.png`

**Small promo tile** (440x280): `images/promo_tile_440x280.png`

**Store icon** (128x128, 96x96 artwork with transparent padding): `images/store_icon_128.png`

**Official URL / homepage:** https://github.com/rahul-dev-app/PileupPokerBot

**Support URL:** https://github.com/rahul-dev-app/PileupPokerBot/issues

## Privacy practices

**Single purpose:**

```
Shows a bot for the 4x4 poker card game on puzzmo.com: a replay and analysis of your game against the bot's game on the same hands after you finish (compete mode), or move suggestions while you play (suggest mode).
```

**Permission justifications:**

- `storage`: `Stores the user's settings: the mode (compete or suggest) and the optional analytics endpoint.`
- Host access through content scripts (`https://www.puzzmo.com/runtime.html*`): `The extension reads the visible game state (board, hand, discard pile) in the game's frame to compute the bot's moves and to show the suggestions and the replay on the page. It runs on no other page.`
- `optional_host_permissions` (`https://*/*`, `http://*/*`): `Only requested at runtime, and only for the one origin the user enters as an optional analytics endpoint. Not requested by default.`

**Remote code:** No, I am not using remote code. (All JavaScript and the WebAssembly module are in the package; the manifest's content security policy allows 'wasm-unsafe-eval' for the bundled WebAssembly module.)

**Data usage:** with the default settings the extension collects no user data. Tick no data types. Certify the three statements (no selling of data, no use unrelated to the single purpose, no creditworthiness use).

**Privacy policy URL:** https://github.com/rahul-dev-app/PileupPokerBot/blob/main/docs/privacy.md (works once the repository is public).
