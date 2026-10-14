from itertools import permutations, combinations

class GreedyPileupPokerAgent:
    def __init__(self, env):
        self.env = env

    def select_best_placement(self, observation):
        best_score = float('-inf')
        best_placement = None

        # Get available grid positions
        empty_positions = self.env.get_empty_positions_in_grid()
        
        # Get all combinations of 4 empty positions
        possible_positions = list(combinations(empty_positions, 4))

        # Get all combinations of cards in hand (4 cards out of 5)
        possible_hands = list(combinations(self.env.hand, 4))

        for hand in possible_hands:
            # Iterate over all possible position combinations
            for pos_combination in possible_positions:
                # For each position combination, try all card permutations
                for card_perm in permutations(hand):
                    for (card, (row, col)) in zip(card_perm, pos_combination):
                        self.env.place_card_on_grid(card, row, col)
                    
                    # Calculate the score for this placement
                    score = self.env.get_total_score()

                    # Check if this is the best score so far
                    if score > best_score:
                        best_score = score
                        best_placement = (card_perm, pos_combination)

                    # Restore the grid to the original state
                    for (card, (row, col)) in zip(card_perm, pos_combination):
                        self.env.erase_card_on_grid(row, col)

        return best_placement, best_score

    def play(self, custom_deck=None):
        observation, info = self.env.reset()
        if custom_deck is not None:
            observation, info = self.env.set_deck_with_observation(custom_deck)            
        done = False 
        self.env.render()       
        while not done:
            
            # Select the best placement
            best_placement, best_score = self.select_best_placement(observation)
            print(f"Best Placement: {best_placement}, Best Score: {best_score}")

            # # Place the cards in the best placement
            for (card, (row, col)) in zip(best_placement[0], best_placement[1]):
                card_index = self.env.hand.index(card)
                pos = (row * self.env.grid_size) + col
                action = card_index * self.env.grid_size * self.env.grid_size + pos
                observation, reward, done, info, _ = self.env.step(action) 

            # Render the environment after placing cards
            self.env.render()

        print(f"Final Score: {self.env.score}")
