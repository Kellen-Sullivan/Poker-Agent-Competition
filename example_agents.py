"""
Built-in opponent agents for benchmarking.

Provides RandomAgent and HeuristicAgent, which can be used 
to test against using test_agent.py.
"""

import numpy as np
from poker_env import Action, Round, HandEvaluator, HandRank


class RandomAgent:
    """Selects a random legal action each turn."""

    def act(self, observation):
        action_mask = observation["action_mask"]
        valid_actions = np.flatnonzero(action_mask)
        return np.random.choice(valid_actions)


class HeuristicAgent:
    """Rule-based agent that plays a simple strategy."""

    def act(self, observation):
        action_mask = observation["action_mask"]
        state = observation["human_readable"]
        valid_actions = np.flatnonzero(action_mask)

        if state['round'] == Round.PREFLOP:
            hand = state['hand']
            rank1 = HandEvaluator.RANK_MAP[hand[0][1]]
            rank2 = HandEvaluator.RANK_MAP[hand[1][1]]

            is_pair = rank1 == rank2
            is_high = (rank1 + rank2) > 20

            if (is_pair or is_high) and Action.RAISE_HALF_POT in valid_actions:
                return Action.RAISE_HALF_POT

            if Action.CHECK_CALL in valid_actions:
                return Action.CHECK_CALL
            return valid_actions[0]

        cards = state["hand"] + state["community_cards"]
        hand_rank, _ = HandEvaluator.evaluate_hand(cards)

        if hand_rank >= HandRank.FULL_HOUSE:
            if Action.ALL_IN in valid_actions:
                return Action.ALL_IN
            if Action.RAISE_FULL_POT in valid_actions:
                return Action.RAISE_FULL_POT

        if hand_rank >= HandRank.TWO_PAIR:
            if Action.RAISE_FULL_POT in valid_actions:
                return Action.RAISE_FULL_POT
            if Action.RAISE_HALF_POT in valid_actions:
                return Action.RAISE_HALF_POT

        if hand_rank >= HandRank.PAIR:
            if Action.RAISE_HALF_POT in valid_actions:
                return Action.RAISE_HALF_POT

        if Action.CHECK_CALL in valid_actions:
            return Action.CHECK_CALL

        return np.random.choice(valid_actions)
