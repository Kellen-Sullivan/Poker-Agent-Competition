"""
Texas Hold'em poker environment with weak-hand multiplier variant.

Exports:
    TexasHoldEm  - PettingZoo AEC wrapper for heads-up No-Limit Hold'em
    Action       - IntEnum of the 5 possible actions
    Round        - IntEnum of betting rounds
    HandRank     - IntEnum of poker hand rankings (HIGH_CARD … ROYAL_FLUSH)
    HandEvaluator - Static hand evaluation utilities
    is_weak_hand - Returns True when hole cards qualify for the 2× bonus
"""

import enum
from collections.abc import Sequence
from enum import IntEnum
from itertools import combinations

import numpy as np
from pettingzoo.classic import texas_holdem_no_limit_v6


# ======================================================================
# Hand evaluation
# ======================================================================

class HandRank(enum.IntEnum):
    HIGH_CARD = 0
    PAIR = 1
    TWO_PAIR = 2
    THREE_OF_A_KIND = 3
    STRAIGHT = 4
    FLUSH = 5
    FULL_HOUSE = 6
    FOUR_OF_A_KIND = 7
    STRAIGHT_FLUSH = 8
    ROYAL_FLUSH = 9


class HandEvaluator:
    RANK_MAP = {
        '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
        '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14,
    }

    @staticmethod
    def evaluate_hand(cards: Sequence[str]) -> tuple[HandRank, list[int]]:
        """Evaluate best 5-card hand from any number of cards."""
        if len(cards) < 5:
            return HandRank.HIGH_CARD, []

        best_rank = HandRank.HIGH_CARD
        best_tiebreakers: list[int] = []

        for combo in combinations(cards, 5):
            rank, tiebreakers = HandEvaluator._evaluate_five_cards(list(combo))
            if (rank > best_rank) or (rank == best_rank and tiebreakers > best_tiebreakers):
                best_rank = rank
                best_tiebreakers = tiebreakers

        return best_rank, best_tiebreakers

    @staticmethod
    def _evaluate_five_cards(cards: list[str]) -> tuple[HandRank, list[int]]:
        sorted_cards = sorted(cards, key=lambda c: HandEvaluator.RANK_MAP[c[1]], reverse=True)

        ranks = [HandEvaluator.RANK_MAP[c[1]] for c in sorted_cards]
        suits = [c[0] for c in sorted_cards]

        rank_counts: dict[int, int] = {}
        for r in ranks:
            rank_counts[r] = rank_counts.get(r, 0) + 1

        unique_ranks = sorted(rank_counts.keys(), key=lambda r: (rank_counts[r], r), reverse=True)
        counts = sorted(rank_counts.values(), reverse=True)

        is_flush = len(set(suits)) == 1

        is_straight = False
        straight_high = ranks[0]

        if len(set(ranks)) == 5:
            if ranks[0] - ranks[4] == 4:
                is_straight = True
            elif ranks == [14, 5, 4, 3, 2]:
                is_straight = True
                straight_high = 5

        if is_straight and is_flush:
            if straight_high == 14 and ranks[1] == 13:
                return HandRank.ROYAL_FLUSH, []
            return HandRank.STRAIGHT_FLUSH, [straight_high]

        if counts == [4, 1]:
            return HandRank.FOUR_OF_A_KIND, [unique_ranks[0], unique_ranks[1]]

        if counts == [3, 2]:
            return HandRank.FULL_HOUSE, [unique_ranks[0], unique_ranks[1]]

        if is_flush:
            return HandRank.FLUSH, ranks

        if is_straight:
            return HandRank.STRAIGHT, [straight_high]

        if counts == [3, 1, 1]:
            return HandRank.THREE_OF_A_KIND, [unique_ranks[0]] + sorted(unique_ranks[1:], reverse=True)

        if counts == [2, 2, 1]:
            return HandRank.TWO_PAIR, sorted(unique_ranks[:2], reverse=True) + [unique_ranks[2]]

        if counts == [2, 1, 1, 1]:
            return HandRank.PAIR, [unique_ranks[0]] + sorted(unique_ranks[1:], reverse=True)

        return HandRank.HIGH_CARD, ranks


def is_weak_hand(hand_list):
    """Return True if hole cards qualify as a trash/garbage hand for the bonus multiplier."""
    ranks = sorted([c[1] for c in hand_list], reverse=True)
    suits = [c[0] for c in hand_list]

    val1 = HandEvaluator.RANK_MAP.get(ranks[0], 0)
    val2 = HandEvaluator.RANK_MAP.get(ranks[1], 0)

    is_suited = suits[0] == suits[1]
    is_connector = abs(val1 - val2) == 1

    if val1 != val2 and not is_suited and val1 < 10 and not is_connector:
        return True
    return False


# ======================================================================
# Action / Round enums
# ======================================================================

class Action(IntEnum):
    FOLD = 0
    CHECK_CALL = 1
    RAISE_HALF_POT = 2
    RAISE_FULL_POT = 3
    ALL_IN = 4

    def label(self) -> str:
        labels = {
            Action.FOLD: "Fold",
            Action.CHECK_CALL: "Check / Call",
            Action.RAISE_HALF_POT: "Raise Half Pot",
            Action.RAISE_FULL_POT: "Raise Full Pot",
            Action.ALL_IN: "All In",
        }
        return labels[self]


class Round(IntEnum):
    PREFLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3
    END_HIDDEN = 4
    SHOWDOWN = 5

    def label(self) -> str:
        labels = {
            Round.PREFLOP: "PREFLOP",
            Round.FLOP: "FLOP",
            Round.TURN: "TURN",
            Round.RIVER: "RIVER",
            Round.END_HIDDEN: "END_HIDDEN",
            Round.SHOWDOWN: "SHOWDOWN",
        }
        return labels[self]


# ======================================================================
# Environment
# ======================================================================

class TexasHoldEm:
    """Wrapper of texas_holdem_no_limit_v6 (PettingZoo / RLCard)."""

    def __init__(self, num_players=2, render_mode="ansi", seed=None):
        self.env = texas_holdem_no_limit_v6.env(num_players=num_players, render_mode=render_mode)

        rlcard_env = self.env.unwrapped.env
        config = {
            'game_num_players': num_players,
            'chips_for_each': 200,
            'dealer_id': 0,
        }
        rlcard_env.game.configure(config)

        self._seed = seed
        self.env.reset(seed=seed)
        self._terminal_multiplier = 1.0

    def reset(self, seed=None, **kwargs):
        self._terminal_multiplier = 1.0
        effective_seed = seed if seed is not None else self._seed
        if effective_seed is not None:
            kwargs["seed"] = effective_seed
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(action)

    def close(self):
        return self.env.close()

    @property
    def agent_iter(self):
        return self.env.agent_iter()

    @property
    def action_space(self):
        return self.env.action_space

    def last(self):
        observation, reward, termination, truncation, info = self.env.last()

        env = self.env.unwrapped.env
        game = env.game
        agent_name = self.env.agent_selection
        current_player_id = int(agent_name.split("_")[1])
        opponent_player_id = (current_player_id + 1) % 2
        state = game.get_state(current_player_id)

        human_readable_dict = {
            'current_player': state['current_player'],
            'round': state['stage'].value,
            'pot': int(state['pot']),
            'hand': state['hand'],
            'community_cards': state['public_cards'],
            'stacks': state['stakes'],
            'my_stack': state['stakes'][current_player_id],
            'opponent_stack': state['stakes'][opponent_player_id],
            'round_bets': state['all_chips'],
            'amount_to_call': max(
                0,
                state['all_chips'][opponent_player_id]
                - state['all_chips'][current_player_id],
            ),
            'is_weak_hand': is_weak_hand(state['hand']),
        }

        observation["human_readable"] = human_readable_dict

        base_reward = reward

        if termination:
            if self._terminal_multiplier == 1.0:
                if base_reward > 0:
                    winner_id = current_player_id
                elif base_reward < 0:
                    winner_id = opponent_player_id
                else:
                    winner_id = None

                if winner_id is not None:
                    winner_hole_cards = game.get_state(winner_id)['hand']
                    if is_weak_hand(winner_hole_cards):
                        self._terminal_multiplier = 2.0

            reward = base_reward * self._terminal_multiplier

        return observation, reward, termination, truncation, info

    def ansi_render(self):
        if not self.env.agents:
            return "(Hand complete -- no game state to render)"
        env = self.env.unwrapped.env
        game = env.game

        current_player = game.get_player_id()
        opp_player = (current_player + 1) % 2
        state = game.get_state(current_player)

        legal_acts_enum = [Action(int(x.value)) for x in state["legal_actions"]]
        legal_acts_str = ", ".join(a.label() for a in legal_acts_enum)

        community = " ".join([f"[{c}]" for c in state['public_cards']])
        if not community:
            community = "[  ] [  ] [  ] [  ] [  ]"

        my_hand = " ".join([f"[{c}]" for c in state['hand']])
        opp_hand = "[??] [??]"

        pot = int(state['pot'])
        my_stack = state['stakes'][current_player]
        opp_stack = state['stakes'][opp_player]
        my_bet = state['all_chips'][current_player]
        opp_bet = state['all_chips'][opp_player]
        to_call = max(0, opp_bet - my_bet)

        W = 80
        inner_width = W - 2
        h_line = "─" * inner_width
        top_border = f"┌{h_line}┐"
        mid_border = f"├{h_line}┤"
        bot_border = f"└{h_line}┘"

        def pad(text: str) -> str:
            return text[:inner_width].ljust(inner_width)

        def center(text: str) -> str:
            return f"│{text[:inner_width].center(inner_width)}│"

        empty_line = f"│{' ' * inner_width}│"

        stage_val = state['stage'].value
        round_label = Round(stage_val).label()

        render = [
            top_border,
            f"│{pad(f'Round: {round_label}    Total Pot: {pot}')}│",
            mid_border,
            center(f"P{opp_player}"),
            center(f"Stack: {opp_stack}"),
            center(f"{opp_hand}"),
            empty_line,
            center(f"(Bet: {opp_bet})"),
            empty_line,
            center("COMMUNITY CARDS"),
            center(community),
            empty_line,
            center(f"(Bet: {my_bet})"),
            empty_line,
            center(f"{my_hand}"),
            center(f"Stack: {my_stack}"),
            center(f"P{current_player}"),
            mid_border,
            f"│{pad(f'Min to Call: {to_call}')}│",
            f"│{pad(f'Legal Actions: {legal_acts_str}')}│",
            bot_border,
        ]

        return "\n".join(render)

    def render(self):
        if self.env.render_mode == "ansi":
            return self.ansi_render()
        return self.env.render()

    @staticmethod
    def action_to_string(action) -> str:
        if action is None:
            return "No action (terminal)"
        try:
            enum_val = Action(int(action))
            return enum_val.label()
        except (ValueError, TypeError):
            return str(action)
