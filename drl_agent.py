"""
Competition-ready DRL agent.

Loads a trained MaskablePPO model and exposes the standard act() interface
expected by the competition framework.
"""

import numpy as np
from sb3_contrib import MaskablePPO
from texas_holdem_env import Action
from feature_extractor import extract_features


class DRLAgent:
    """Deep Reinforcement Learning poker agent using a trained PPO policy.

    Usage
    -----
    >>> from drl_agent import DRLAgent
    >>> agent = DRLAgent("models/ppo_poker_final")
    >>> # Use in competition
    >>> from example_usage import run_competition
    >>> run_competition(10000, agent, HeuristicAgent())
    """

    def __init__(self, model_path="models/ppo_poker_final"):
        self.model = MaskablePPO.load(model_path)

    def act(self, observation, env):
        """Select an action given the current game observation.

        Parameters
        ----------
        observation : dict
            Contains 'action_mask' and 'human_readable' keys.
        env : TexasHoldEm
            The game environment (unused by this agent but required by the
            competition interface).

        Returns
        -------
        Action
            The chosen action enum value.
        """
        features = extract_features(observation)
        action_mask = observation["action_mask"]
        action, _ = self.model.predict(
            features, action_masks=action_mask, deterministic=True
        )
        return Action(int(action))
