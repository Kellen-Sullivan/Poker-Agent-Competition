"""
Gymnasium single-agent wrapper around the 2-player PettingZoo Texas Hold'em
environment.  Opponent turns are auto-played so that SB3 sees a standard
single-agent Gym env with action masking support for MaskablePPO.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from poker_env import TexasHoldEm, Action
from rl_example.feature_extractor import extract_features, FEATURE_DIM


class PokerGymEnv(gym.Env):
    """Single-agent Gymnasium wrapper for heads-up No-Limit Texas Hold'em."""

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, opponent_agent, render_mode=None):
        super().__init__()
        self._opponent = opponent_agent
        self._render_mode = render_mode

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(FEATURE_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(5)

        self._action_mask = np.ones(5, dtype=np.int8)
        self._hand_count = 0
        self._our_seat = "player_0"
        self._done = False

        self._env = TexasHoldEm(num_players=2, render_mode="ansi")

    def set_opponent(self, opponent_agent):
        """Swap the opponent policy (used during self-play curriculum)."""
        self._opponent = opponent_agent

    def action_masks(self):
        """Return the current legal-action mask.  Required by MaskablePPO."""
        return self._action_mask.copy()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._env.reset(seed=seed)
        self._hand_count += 1
        self._done = False

        if self._hand_count % 2 == 0:
            self._our_seat = "player_0"
        else:
            self._our_seat = "player_1"

        obs, reward, done = self._advance()
        if done:
            self._done = True
        return obs, {}

    def step(self, action):
        if self._done:
            return (
                np.zeros(FEATURE_DIM, dtype=np.float32),
                0.0,
                True,
                False,
                {},
            )

        self._env.step(int(action))
        obs, reward, done = self._advance()
        self._done = done
        return obs, reward, done, False, {}

    def render(self):
        if self._env is not None:
            return self._env.render()
        return None

    def close(self):
        if self._env is not None:
            self._env.close()

    def _advance(self):
        """Play through opponent / terminal turns until our next decision."""
        our_reward = 0.0
        pz_env = self._env.env

        while True:
            if not pz_env.agents:
                return (
                    np.zeros(FEATURE_DIM, dtype=np.float32),
                    our_reward,
                    True,
                )

            current_agent = pz_env.agent_selection
            observation, reward, term, trunc, info = self._env.last()

            if current_agent == self._our_seat:
                our_reward += reward

            if term or trunc:
                self._env.step(None)
                continue

            if current_agent == self._our_seat:
                self._action_mask = np.array(
                    observation["action_mask"], dtype=np.int8
                )
                features = extract_features(observation)
                return features, our_reward, False
            else:
                opp_action = self._opponent.act(observation)
                self._env.step(int(opp_action))
