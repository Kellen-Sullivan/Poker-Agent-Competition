"""
Competition-ready DRL agent.

Loads a trained MaskablePPO model and exposes the standard act() interface
expected by the competition framework.
"""

import numpy as np
from sb3_contrib import MaskablePPO
from poker_env import Action
from rl_example.feature_extractor import extract_features


class DRLAgent:
    """Deep Reinforcement Learning poker agent using a trained PPO policy.

    Usage:
        from rl_example.drl_agent import DRLAgent
        agent = DRLAgent("models/ppo_poker_final")
    """

    def __init__(self, model_path="models/ppo_poker_final"):
        try:
            self.model = MaskablePPO.load(model_path)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Could not find model at '{model_path}'. "
                f"Train a model first by running: python -m rl_example.train"
            )

    def act(self, observation):
        features = extract_features(observation)
        action_mask = observation["action_mask"]
        action, _ = self.model.predict(
            features, action_masks=action_mask, deterministic=True
        )
        return Action(int(action))
