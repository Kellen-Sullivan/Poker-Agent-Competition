"""
Feature extractor for the DRL poker agent.

Converts the human-readable game observation into a fixed-size numeric
feature vector suitable for neural network input.
"""

import numpy as np
from poker_eval import HandEvaluator, HandRank
from texas_holdem_env import Round

# Total number of features in the extracted vector
FEATURE_DIM = 24

# Starting chip stack configured in TexasHoldEm
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

    # 0: Hand strength (0-1)
    if len(community) >= 3:
        cards = hand + community
        hand_rank, _ = HandEvaluator.evaluate_hand(cards)
        features[idx] = hand_rank / 9.0
    else:
        features[idx] = _preflop_strength(hand)
    idx += 1

    # 1: Pot odds (0-1)
    if to_call > 0 and (pot + to_call) > 0:
        features[idx] = to_call / (pot + to_call)
    idx += 1

    # 2: Stack-to-pot ratio (clipped to 10, then /10)
    features[idx] = min(my_stack / max(pot, 1), 10.0) / 10.0
    idx += 1

    # 3: My stack normalised
    features[idx] = my_stack / _STARTING_STACK
    idx += 1

    # 4: Opponent stack normalised
    features[idx] = opp_stack / _STARTING_STACK
    idx += 1

    # 5: Opponent stack ratio
    total = my_stack + opp_stack
    features[idx] = opp_stack / total if total > 0 else 0.5
    idx += 1

    # 6: Commitment ratio (fraction of starting stack bet)
    chips_bet = max(0.0, _STARTING_STACK - my_stack)
    features[idx] = chips_bet / _STARTING_STACK
    idx += 1

    # 7: Amount to call normalised (by pot, clipped)
    if pot > 0:
        features[idx] = min(to_call / pot, 2.0) / 2.0
    idx += 1

    # 8: Number of community cards normalised
    features[idx] = len(community) / 5.0
    idx += 1

    # 9: Position / is dealer
    # In this env player_0 is always the dealer (button / SB)
    features[idx] = 1.0 if state["current_player"] == 0 else 0.0
    idx += 1

    # 10-13: Betting round one-hot (4-dim)
    round_val = state["round"]
    if 0 <= round_val <= 3:
        features[idx + round_val] = 1.0
    idx += 4

    # 14-23: Hand rank one-hot (10-dim, post-flop only)
    if len(community) >= 3:
        cards = hand + community
        hand_rank, _ = HandEvaluator.evaluate_hand(cards)
        features[idx + int(hand_rank)] = 1.0
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

    # Base: average of normalised card values (max 28 -> 1.0)
    score = (high + low) / 28.0

    # Pair bonus
    if r1 == r2:
        score += 0.20

    # Suited bonus
    if s1 == s2:
        score += 0.05

    # Connectedness bonus
    gap = high - low
    if gap <= 1:
        score += 0.05
    elif gap == 2:
        score += 0.02

    # High-card bonus (Queen+)
    if high >= 12:
        score += 0.05

    return min(score, 1.0)
