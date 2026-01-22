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