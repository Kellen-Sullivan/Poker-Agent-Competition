"""
Gymnasium single-agent wrapper around the 2-player PettingZoo Texas Hold'em
environment.  Opponent turns are auto-played so that SB3 sees a standard
single-agent Gym env with action masking support for MaskablePPO.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from texas_holdem_env import TexasHoldEm, Action
from feature_extractor import extract_features, FEATURE_DIM


class PokerGymEnv(gym.Env):
    """Single-agent Gymnasium wrapper for heads-up No-Limit Texas Hold'em.

    Key design points:
    - Wraps the 2-player PettingZoo AEC env as a single-agent Gym env
    - Auto-plays opponent turns using the provided opponent agent
    - Returns engineered features (not raw observations) via extract_features()
    - Exposes action_masks() for sb3-contrib MaskablePPO
    - Alternates seats every hand for positional fairness
    - Reward = chips won/lost at the end of each hand (sparse)
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, opponent_agent, render_mode=None):
        super().__init__()
        self._opponent = opponent_agent
        self._render_mode = render_mode

        # Observation: fixed-size feature vector
        self.observation_space = spaces.Box(
            low=0.0, high=2.0, shape=(FEATURE_DIM,), dtype=np.float32
        )
        # 5 discrete actions: FOLD, CHECK_CALL, RAISE_HALF, RAISE_FULL, ALL_IN
        self.action_space = spaces.Discrete(5)

        self._action_mask = np.ones(5, dtype=np.int8)
        self._hand_count = 0
        self._our_seat = "player_0"
        self._done = False

        # Create the underlying PettingZoo env once; reuse across hands
        self._env = TexasHoldEm(num_players=2, render_mode="ansi")

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def set_opponent(self, opponent_agent):
        """Swap the opponent policy (used during self-play curriculum)."""
        self._opponent = opponent_agent

    def action_masks(self):
        """Return the current legal-action mask.  Required by MaskablePPO."""
        return self._action_mask.copy()

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._env.reset()
        self._hand_count += 1
        self._done = False

        # Alternate seats for positional fairness
        if self._hand_count % 2 == 0:
            self._our_seat = "player_0"
        else:
            self._our_seat = "player_1"

        # Advance past any opponent turns until it is our turn
        obs, reward, done = self._advance()
        if done:
            # Extremely rare: hand ended before we acted (e.g. forced all-in)
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

        # Execute our action in the PettingZoo env
        self._env.step(int(action))

        # Play through opponent turns (and any terminal bookkeeping)
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _advance(self):
        """Play through opponent / terminal turns until our next decision.

        Returns
        -------
        obs : np.ndarray   - feature vector (zeros if hand ended)
        reward : float      - accumulated reward for our seat this segment
        done : bool         - True when the hand is complete
        """
        our_reward = 0.0
        pz_env = self._env.env  # underlying PettingZoo AECEnv

        while True:
            # If no agents remain the episode is over
            if not pz_env.agents:
                return (
                    np.zeros(FEATURE_DIM, dtype=np.float32),
                    our_reward,
                    True,
                )

            current_agent = pz_env.agent_selection
            observation, reward, term, trunc, info = self._env.last()

            # Accumulate reward whenever we see our seat
            if current_agent == self._our_seat:
                our_reward += reward

            # Terminal state: step with None to let PettingZoo clean up
            if term or trunc:
                self._env.step(None)
                continue

            if current_agent == self._our_seat:
                # It is our turn - extract features and return control
                self._action_mask = np.array(
                    observation["action_mask"], dtype=np.int8
                )
                features = extract_features(observation)
                return features, our_reward, False
            else:
                # Opponent turn - auto-play
                opp_action = self._opponent.act(observation, self._env)
                self._env.step(int(opp_action))
