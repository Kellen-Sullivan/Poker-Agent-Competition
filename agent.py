import numpy as np 
from texas_holdem_env import TexasHoldEm


class RandomAgent:
    def act(self, observation, env):
        """Returns a random index where the mask is 1"""
        action_mask = observation["action_mask"]
        valid_actions = np.flatnonzero(action_mask)
        return np.random.choice(valid_actions)
    

class AlwaysFoldAgent:
    def act(self, observation, env: TexasHoldEm):
        """
        In PettingZoo Texas Hold'em
        action mask: [Fold, Check, Call, Raise Half Pot, Raise Full Pot, All In]
        """
        action_mask = observation["action_mask"]
        valid_actions = np.flatnonzero(action_mask)  
        # Always try to Fold if it's legal.
        # If not, pick the first available legal action   
        if env.FOLD in valid_actions:
            return env.FOLD
        return valid_actions[0]
    
# THIS AGENT LITERALLY LOSES TO A RANDOM AGENT (FOLDS TOO MUCH)
class HeuristicAgent:
    def act(self, observation, env: TexasHoldEm):
        """Act based on a set of good heuristics. 
        Use human-readable observation"""
        action_mask = observation["action_mask"]
        valid_actions = np.flatnonzero(action_mask)

        state = observation["human_readable"]
        
        # Parse hand strength
        # Card format is (Suit, Rank)
        def get_rank_value(card_string):
            rank_map = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, 
                        '8': 8, '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
            return rank_map.get(card_string[1], 0)

        my_ranks = [get_rank_value(c) for c in state['hand']]
        board_ranks = [get_rank_value(c) for c in state['community_cards']]
        
        # === Calculate Heuristic score for current state ===
        score = 0
        
        # Pre-flop Rule
        if state['round'] == env.PREFLOP:
            if 14 in my_ranks: score += 5  # Ace is good
            if 13 in my_ranks: score += 3  # King is decent
            if my_ranks[0] == my_ranks[1]: score += 10 # Pocket pair!
            
        # Post-flop Rule
        # Check if any of my cards match the board
        hits = [r for r in my_ranks if r in board_ranks]
        if len(hits) > 0:
            score += 15 # We hit a pair or better!
            
        # === Decide action based on calculate Heuristic score ===
        action_mask = observation["action_mask"]
        
        # If score is high try to Raise
        if score > 10:
            if action_mask[env.RAISE_FULL_POT]: return env.RAISE_FULL_POT
            if action_mask[env.RAISE_HALF_POT]: return env.RAISE_HALF_POT
            
        # If score is mediocre just Call
        if score > 0:
            if action_mask[env.CHECK_CALL]: return env.CHECK_CALL
            
        # Never fold if we can check for free
        if state['amount_to_call'] == 0:
            return env.CHECK_CALL
            
        # Otherwise fold
        return env.FOLD
        
