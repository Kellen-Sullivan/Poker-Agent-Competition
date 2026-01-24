import numpy as np 
from poker_eval import HandEvaluator, HandRank
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
    
class HeuristicAgent:
    def act(self, observation, env: TexasHoldEm):
        """Act based on a set of good heuristics. 
        Use human-readable observation"""
        action_mask = observation["action_mask"]
        state = observation["human_readable"]
        valid_actions = np.flatnonzero(action_mask)  

        # === Preflop Strategy ===
        if state['round'] == env.PREFLOP:
            # Simple heuristic: If we have a pair, or High cards (Sum > 20), play aggressive
            hand = state['hand']
            rank1 = HandEvaluator.RANK_MAP[hand[0][1]]
            rank2 = HandEvaluator.RANK_MAP[hand[1][1]]
            
            is_pair = rank1 == rank2
            is_high = (rank1 + rank2) > 20 # e.g. K(13) + 8(8) = 21
            
            if (is_pair or is_high) and env.RAISE_HALF_POT in valid_actions:
                return env.RAISE_HALF_POT
                
            # Otherwise try to Check, and fold/all-in as last result
            if env.CHECK_CALL in valid_actions:
                return env.CHECK_CALL
            return valid_actions[0] # forced to fold/all-in
        
        # === Postflop Strategy ===
        cards = state["hand"] + state["community_cards"]
        
        # Calculate Hand Strength
        hand_rank, tiebreaker = HandEvaluator.evaluate_hand(cards)

        # Go all in with Full House or better
        if hand_rank >= HandRank.FULL_HOUSE:
            if env.ALL_IN in valid_actions: return env.ALL_IN
            if env.RAISE_FULL_POT in valid_actions: return env.RAISE_FULL_POT

        # Raise a Strong hand
        if hand_rank >= HandRank.TWO_PAIR:
            if env.RAISE_FULL_POT in valid_actions: return env.RAISE_FULL_POT
            if env.RAISE_HALF_POT in valid_actions: return env.RAISE_HALF_POT
            
        # Small raise with at least a pair
        if hand_rank >= HandRank.PAIR:
            if env.RAISE_HALF_POT in valid_actions: return env.RAISE_HALF_POT
            
        # If nothing, try to check
        if env.CHECK_CALL in valid_actions:
            return env.CHECK_CALL

        # Never always fold, randomly play valid action to maybe bluff
        return np.random.choice(valid_actions)
        
