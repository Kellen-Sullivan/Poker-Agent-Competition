"""
Your poker agent.

Edit this file to implement your strategy, then run:
    python test_agent.py
"""

import numpy as np
from poker_env import TexasHoldEm, Action, Round, HandEvaluator, HandRank


class MyAgent:
    """Starter agent — replace this logic with your own strategy."""

    def act(self, observation, env):
        action_mask = observation["action_mask"]
        valid_actions = np.flatnonzero(action_mask)

        # Example: pick a random legal action
        return np.random.choice(valid_actions)
