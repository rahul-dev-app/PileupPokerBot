# The bot's algorithm

This page describes the algorithm that plays in the [Chrome extension](chrome-extension.md) (compete mode and suggest mode), from training to the move it picks. Details, experiments and the exact code paths are in [afterstate-td.md](afterstate-td.md) and [value-search.md](value-search.md).

Summary: a **learned value function** (an MLP trained by TD learning) is used inside a **depth-limited expectimax search** over whole rounds. The search is made about 1,450x cheaper by **beam search** for candidate moves, **successive halving** for the sampled futures, a **small network** for the moves inside the lookahead, and **exact play** of the last round. The result plays at the level of the full search (about 6,860-6,900 on the 1000-deck benchmark, versus 3,103 for the greedy agent and 6,294 for the best PPO policy) in about 0.16 s per game on one CPU core in the browser.

## 1. The decision to make

Each round the player sees 5 cards, places 4 on empty cells and discards the 5th. Nothing random happens inside a round, so the real decision is the whole round: which card to discard and where the other 4 go. We call this a **round decision**. The number of round decisions is 218,400 in round 1 (16 empty cells), 59,400 in round 2, 8,400 in round 3 and 120 in round 4.

Randomness enters only between rounds, when the next 5 cards are dealt from the unseen cards.

## 2. The value function

**Afterstate.** The position right after a card is placed, before any new card is dealt: the board, the discard pile, the cards still in hand, how many cards are on the board, and the set of unseen cards. After the 4th placement of a round the hand is empty and the 5th card is on the discard pile; this **end-of-round afterstate** is what a round decision leads to.

**V(afterstate)** estimates the expected final score from that afterstate:

```
V(a) = current_score(a) + 1000 * f(a)
```

`f` is an MLP (ReLU, 250 inputs -> 512 -> 512 -> 1). The 250 input features are, for each of the 10 scored lines (4 rows, 4 columns, corners, discard pile): rank counts, card count, "flush still possible", "straight still possible", unseen cards of the flush suit and the current line score; plus the hand and the unseen cards as 36-card multi-hot vectors, and small context features. For a full board `f = 0`, so V is the exact final score.

**Training** (`train_afterstate.py`, [afterstate-td.md](afterstate-td.md)): self-play with an epsilon-greedy afterstate agent (epsilon 0.05), 1,024 games per iteration, 10,000 iterations (10.2M games, 55 min on an RTX 3090). Targets are TD(lambda) returns with lambda 0.8: they bootstrap from the best afterstate value at the next move, and the last target is the final score. Each training afterstate gets a random board symmetry and suit relabeling. Used alone (place each card on the afterstate with the highest V), this agent scores 6,133.

**Small network for cheap moves.** A 128-wide MLP (250 -> 128 -> 128 -> 1, 8x fewer multiply-adds) is first distilled from the 512-wide one and then fine-tuned with the same TD training. Alone it scores 5,963: good enough to choose moves inside the lookahead, but too weak to judge positions (as the only network in the search it loses about 150 points). So the search uses it only for moves inside the lookahead, and the 512-wide network for all position values.

## 3. Value search (the full version)

At the start of a round, with hand H:

1. Score every round decision d by `V(end-of-round afterstate of d)`. Keep the best K decisions, with symmetric copies removed (rows/columns swaps that keep the corners, transposition, suit relabeling: 32 x 24 symmetries that change no score).
2. For each of S sampled next hands (5 cards drawn from the unseen cards) and each candidate: play the next round by its best decision under V, and take V of the resulting afterstate. All candidates use the **same** sampled hands (common random numbers), so differences between candidates are not sampling noise.
3. Play the candidate with the best mean.

This is a one-step expectimax: max over this round's decisions, average over the next hand, max over the next round's decisions, then V. In round 3 the next round is the last one, so V of the result is the exact final score. With K = 8 and S = 32 it scores **6,881**, but it needs about 17.6M MLP calls (about 14 TFLOP) per game, 98% of them for ranking all decisions of each sampled next round.

The search never sees the real order of the undealt cards: sampled hands come only from cards that are not visible to the player.

## 4. The cheap version (what the extension runs)

Four changes cut the cost to about 65k network calls (9.5 GFLOP) per game without a measurable loss:

**a. Candidates by beam search.** Instead of scoring all round decisions, build them card by card. Start from the current board. At each of the 4 placements, expand every beam entry by every legal (hand card, empty cell), drop afterstates already reached in another order, score them with the 512-wide V (it is trained on these partial afterstates) and keep the best 16. The 4th step gives end-of-round afterstates (the leftover card is the discard). Sort them by V, remove symmetric copies and keep the best 16 (8 in round 2). On the empty board the first card goes to one corner, one edge cell and one inner cell only (all other cells are symmetric copies). It plays as well as ranking all decisions, with about 1/70 of the network calls.

**b. Successive halving.** Sample the next hands once per search: 64 in rounds 1 and 3, 32 in round 2. Then spend them in stages. Each stage (n, s) gives the best n candidates (by their mean so far) s more hands, the same hands for all of them:

| round | stages (n candidates, s more hands each) | hands for the final pick |
|---|---|---|
| 1 and 3 | (16, 8), (8, 8), (4, 16), (2, 32) | 64 |
| 2 | (8, 8), (4, 8), (2, 16) | 32 |

Weak candidates are dropped after a few samples; the last two get all of them.

**c. Cheap leaves.** For each (candidate, sampled hand), the **128-wide** network plays the next round card by card (each card on the legal placement with the highest V), and the **512-wide** network scores the resulting end-of-round afterstate once. In round 3 the next round is the last one, so it is played by its exact best decision instead (120 decisions, exact scoring, no network).

**d. Exact last round.** In round 4 nothing is random: all 120 decisions are scored exactly and the best is played.

```
def plan_round(state, hand):
    if round == 4:
        return argmax(exact_final_score(d) for d in round_decisions(state, hand))
    candidates = beam_search(state, hand, width=16, value=V512)   # best 16 (8 in round 2) distinct
    hands = sample_next_hands(unseen_cards(state, hand), 64 or 32, seed)
    total = {c: 0 for c in candidates}; seen = {c: 0 for c in candidates}
    i = 0
    for n, s in schedule[round]:                                   # successive halving
        alive = best_by_mean(candidates, total, seen, n)
        for c in alive:
            for h in hands[i:i + s]:
                total[c] += leaf_value(c, h); seen[c] += 1
        i += s
    return best_by_mean(alive, total, seen, 1)

def leaf_value(c, next_hand):
    if next_round == 4:
        return max(exact_final_score(d) for d in round_decisions(c, next_hand))
    a = play_card_by_card(c, next_hand, value=V128)                 # 4 placements, discard the rest
    return V512(a)
```

Cost, strength and every tested variant: [value-search.md, Making value search cheap](value-search.md#making-value-search-cheap).

## 5. In the browser

`chrome_extension/extension/search.js` implements section 4 in JavaScript. The network layers run in a 2 KB WebAssembly SIMD kernel (`chrome_extension/wasm/src/lib.rs`). Next hands are drawn with a seeded PRNG, so results are reproducible. The search runs in the extension's service worker, off the page's main thread.

The port matches the JAX implementation (features, values, candidates, call counts and decisions) and takes about 0.16 s per game on one CPU core. Parity checks, strength and speed: [value-search.md, Browser port](value-search.md#browser-port).

## 6. Compete mode: the bot's game and the analysis

After the player finishes, the extension has the four hands as they were dealt and the player's board.

- **Bot game:** the bot plays the same four hands with the algorithm above. In round r it only knows hands 1 to r; later hands are sampled from its unseen cards like in any game.
- **Analysis of each player round:** from the player's actual position at the start of round r, run the search, and also evaluate the player's actual round decision as a forced candidate on the **same** sampled hands. The expected points lost in round r are `value(bot's choice) - value(player's choice)`. In round 4 both values are exact. In rounds 1-3 the loss can be slightly negative (sampling noise). Losses under 0.5 points are shown as "as good as the bot's choice", and the total loss adds up only the positive ones.
- The replay shows both games round by round, the scoring lines and these per-round losses.

**Suggest mode** uses the same search to plan the whole round when the hand is dealt and shows the plan move by move. If the player places a card elsewhere or discards early, the rest of the round is planned again with those choices fixed.
