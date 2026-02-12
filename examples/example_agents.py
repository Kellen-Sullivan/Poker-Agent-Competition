import numpy as np 
from examples.poker_eval import HandEvaluator, HandRank
from texas_holdem_env import TexasHoldEm, Action, Round


class RandomAgent:
    def act(self, observation, env):
        """Select a random legal action for the current turn."""

        # The action mask is a binary array indicating which actions are legal.
        # For example, if action_mask = [1, 0, 1, 1, 0], then actions 0, 2, and 3 are valid.
        action_mask = observation["action_mask"]
        valid_actions = np.flatnonzero(action_mask)  # For example turns: [1, 0, 1, 1, 0] -> [0, 2, 3]

        return np.random.choice(valid_actions) # return a random valid action
    

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
        if Action.FOLD in valid_actions:
            return Action.FOLD
        return valid_actions[0]
    
class HeuristicAgent:
    def act(self, observation, env: TexasHoldEm):
        """Act based on a set of good heuristics. 
        Use human-readable observation"""
        action_mask = observation["action_mask"]
        state = observation["human_readable"]
        valid_actions = np.flatnonzero(action_mask)  

        # === Preflop Strategy ===
        if state['round'] == Round.PREFLOP:
            # Simple heuristic: If we have a pair, or High cards (Sum > 20), play aggressive
            hand = state['hand']
            rank1 = HandEvaluator.RANK_MAP[hand[0][1]]
            rank2 = HandEvaluator.RANK_MAP[hand[1][1]]
            
            is_pair = rank1 == rank2
            is_high = (rank1 + rank2) > 20 # e.g. K(13) + 8(8) = 21
            
            if (is_pair or is_high) and Action.RAISE_HALF_POT in valid_actions:
                return Action.RAISE_HALF_POT
                
            # Otherwise try to Check, and fold/all-in as last result
            if Action.CHECK_CALL in valid_actions:
                return Action.CHECK_CALL
            return valid_actions[0] # forced to fold/all-in
        
        # === Postflop Strategy ===
        cards = state["hand"] + state["community_cards"]
        
        # Calculate Hand Strength
        hand_rank, tiebreaker = HandEvaluator.evaluate_hand(cards)

        # Go all in with Full House or better
        if hand_rank >= HandRank.FULL_HOUSE:
            if Action.ALL_IN in valid_actions: return Action.ALL_IN
            if Action.RAISE_FULL_POT in valid_actions: return Action.RAISE_FULL_POT

        # Raise a Strong hand
        if hand_rank >= HandRank.TWO_PAIR:
            if Action.RAISE_FULL_POT in valid_actions: return Action.RAISE_FULL_POT
            if Action.RAISE_HALF_POT in valid_actions: return Action.RAISE_HALF_POT
            
        # Small raise with at least a pair
        if hand_rank >= HandRank.PAIR:
            if Action.RAISE_HALF_POT in valid_actions: return Action.RAISE_HALF_POT
            
        # If nothing, try to check
        if Action.CHECK_CALL in valid_actions:
            return Action.CHECK_CALL

        # Never always fold, randomly play valid action to maybe bluff
        return np.random.choice(valid_actions)
        
