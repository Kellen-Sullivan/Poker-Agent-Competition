"""
Feature extractor for the DRL poker agent.

Converts the human-readable game observation into a fixed-size numeric
feature vector suitable for neural network input.
"""

import numpy as np
from poker_env import HandEvaluator, HandRank

FEATURE_DIM = 24

_STARTING_STACK = 200.0


def extract_features(observation):
    """Convert a game observation dict to a fixed-size feature vector.

    Feature layout (24-dim):
        [0]      hand_strength              (float 0-1)
        [1]      pot_odds                   (float 0-1)
        [2]      stack_to_pot_ratio         (float 0-1, clipped and normalised)
        [3]      my_stack_normalised        (float 0-1)
        [4]      opponent_stack_normalised  (float 0-1)
        [5]      opponent_stack_ratio       (float 0-1)
        [6]      commitment_ratio           (float 0-1)
        [7]      amount_to_call_normalised  (float 0-1)
        [8]      num_community_cards_norm   (float 0-1)
        [9]      is_dealer (position)       (binary 0/1)
        [10-13]  betting_round one-hot      (4-dim)
        [14-23]  hand_rank one-hot          (10-dim, zeros at preflop)
    """
    state = observation["human_readable"]
    features = np.zeros(FEATURE_DIM, dtype=np.float32)

    hand = state["hand"]
    community = state["community_cards"]
    pot = state["pot"]
    to_call = state["amount_to_call"]
    my_stack = state["my_stack"]
    opp_stack = state["opponent_stack"]

    idx = 0

    _hand_rank = None
    if len(community) >= 3:
        cards = hand + community
        _hand_rank, _ = HandEvaluator.evaluate_hand(cards)

    if _hand_rank is not None:
        features[idx] = _hand_rank / 9.0
    else:
        features[idx] = _preflop_strength(hand)
    idx += 1

    if to_call > 0 and (pot + to_call) > 0:
        features[idx] = to_call / (pot + to_call)
    idx += 1

    features[idx] = min(my_stack / max(pot, 1), 10.0) / 10.0
    idx += 1

    features[idx] = my_stack / _STARTING_STACK
    idx += 1

    features[idx] = opp_stack / _STARTING_STACK
    idx += 1

    total = my_stack + opp_stack
    features[idx] = opp_stack / total if total > 0 else 0.5
    idx += 1

    chips_bet = max(0.0, _STARTING_STACK - my_stack)
    features[idx] = chips_bet / _STARTING_STACK
    idx += 1

    if pot > 0:
        features[idx] = min(to_call / pot, 2.0) / 2.0
    idx += 1

    features[idx] = len(community) / 5.0
    idx += 1

    features[idx] = 1.0 if state["current_player"] == 0 else 0.0
    idx += 1

    round_val = state["round"]
    if 0 <= round_val <= 3:
        features[idx + round_val] = 1.0
    idx += 4

    if _hand_rank is not None:
        features[idx + int(_hand_rank)] = 1.0
    idx += 10

    return features


def _preflop_strength(hand):
    """Heuristic preflop hand strength from hole cards. Returns 0-1."""
    rank_map = HandEvaluator.RANK_MAP
    r1 = rank_map[hand[0][1]]
    r2 = rank_map[hand[1][1]]
    s1 = hand[0][0]
    s2 = hand[1][0]

    high = max(r1, r2)
    low = min(r1, r2)

    score = (high + low) / 28.0

    if r1 == r2:
        score += 0.20

    if s1 == s2:
        score += 0.05

    gap = high - low
    if gap <= 1:
        score += 0.05
    elif gap == 2:
        score += 0.02

    if high >= 12:
        score += 0.05

    return min(score, 1.0)
