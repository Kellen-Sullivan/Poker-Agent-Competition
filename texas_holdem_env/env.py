from pettingzoo.classic import texas_holdem_no_limit_v6
from examples.poker_eval import HandEvaluator, HandRank, is_weak_hand
import numpy as np
from enum import IntEnum


class Action(IntEnum):
    """Action enum for all agents and renderers.

    These values match the underlying PettingZoo action indices.
    """

    FOLD = 0
    CHECK_CALL = 1
    RAISE_HALF_POT = 2
    RAISE_FULL_POT = 3
    ALL_IN = 4

    def label(self) -> str:
        """Human‑readable label for use in logs and ANSI rendering."""
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


class TexasHoldEm():
    """
    Wrapper of texas_holdem_no_limit_v6 which is a wrapper of RLCardGame
    """

    def __init__(self, num_players=2, render_mode="ansi"):
        self.env = texas_holdem_no_limit_v6.env(num_players=num_players, render_mode=render_mode)

        # set starting stacks to 200 (100 is default)
        rlcard_env = self.env.unwrapped.env
        config = {
            'game_num_players': num_players,
            'chips_for_each': 200,
            'dealer_id': 0
        }
        rlcard_env.game.configure(config)

        self.env.reset()
        # Multiplier applied to terminal rewards for the current hand.
        # This is computed once per hand (on first terminal step) so that
        # both players' rewards are scaled identically, preserving zero‑sum.
        self._terminal_multiplier = 1.0

    def reset(self, **kwargs):
        # Reset underlying env and per‑hand reward shaping state
        self._terminal_multiplier = 1.0
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

        # Access player state
        env = self.env.unwrapped.env
        game = env.game
        # In PettingZoo's AEC API, the "current" agent is given by
        # env.agent_selection (e.g., "player_0", "player_1").  We use this
        # to pick the correct seat index and then query RLCard for state.
        agent_name = self.env.agent_selection  # "player_0" / "player_1"
        current_player_id = int(agent_name.split("_")[1])
        opponent_player_id = (current_player_id + 1) % 2
        state = game.get_state(current_player_id)

        human_readable_dict = {
            # Context
            'current_player': state['current_player'],
            'round': state['stage'].value,
            'pot': int(state['pot']),

            # Cards
            'hand': state['hand'],
            'community_cards': state['public_cards'],

            # Stacks
            'stacks': state['stakes'],
            'my_stack': state['stakes'][current_player_id],
            'opponent_stack': state['stakes'][opponent_player_id],

            # Round bets
            'round_bets': state['all_chips'],
            'amount_to_call': max(
                0,
                state['all_chips'][opponent_player_id]
                - state['all_chips'][current_player_id],
            ),

            # Weak Hand variant: True if hole cards qualify for 2x bonus on win
            'is_weak_hand': is_weak_hand(state['hand']),
        }
        
        observation["human_readable"] = human_readable_dict

        # =====================================================
        # CUSTOM VARIANT: Weak Hand Multiplier
        # Bonus 2x reward when winning with weak hole cards (e.g., 7‑2, J‑4 offsuit)
        #
        # To preserve the underlying game's zero‑sum property, we:
        #   1) Compute a single per‑hand multiplier based on the *winner's*
        #      hole cards (2x if weak, 1x otherwise) on the first terminal step.
        #   2) Apply that same multiplier to every player's terminal reward.
        #
        # Because the base environment is zero‑sum (r0 + r1 = 0), scaling both
        # sides by the same factor keeps the sum at zero.
        # =====================================================

        base_reward = reward

        if termination:
            # Only (re)compute the multiplier on the first terminal callback
            # for a hand. Subsequent terminal steps for the other player will
            # reuse the same value so that both rewards are scaled identically.
            if self._terminal_multiplier == 1.0:
                # Determine winner seat from the *base* reward sign.
                if base_reward > 0:
                    winner_id = current_player_id
                elif base_reward < 0:
                    winner_id = opponent_player_id
                else:
                    winner_id = None  # Split pot / no winner; no bonus applied.

                if winner_id is not None:
                    winner_hole_cards = game.get_state(winner_id)['hand']
                    if is_weak_hand(winner_hole_cards):
                        self._terminal_multiplier = 2.0

            # Apply the (possibly updated) multiplier to this agent's reward.
            reward = base_reward * self._terminal_multiplier

        return observation, reward, termination, truncation, info
    
    def ansi_render(self):
        env = self.env.unwrapped.env
        game = env.game
        
        current_player = game.get_player_id()
        opp_player = (current_player + 1) % 2
        state = game.get_state(current_player)

        # Human‑readable legal actions from the Action enum
        legal_acts_enum = [Action(int(x.value)) for x in state["legal_actions"]]
        legal_acts_str = ", ".join(a.label() for a in legal_acts_enum)
        
        # Cards
        community = " ".join([f"[{c}]" for c in state['public_cards']])
        if not community: 
            community = "[  ] [  ] [  ] [  ] [  ]"
            
        my_hand = " ".join([f"[{c}]" for c in state['hand']])
        opp_hand = "[??] [??]" 
        
        # Money
        pot = int(state['pot'])
        my_stack = state['stakes'][current_player]
        opp_stack = state['stakes'][opp_player]
        my_bet = state['all_chips'][current_player]
        opp_bet = state['all_chips'][opp_player]
        to_call = max(0, opp_bet - my_bet)
        
        # Solid Box Characters
        # Slightly wider box so all legal actions fit comfortably.
        W = 80
        inner_width = W - 2
        h_line = "─" * inner_width
        top_border = f"┌{h_line}┐"
        mid_border = f"├{h_line}┤"
        bot_border = f"└{h_line}┘"
        
        def pad(text: str) -> str:
            """Pad or truncate text so that the visible area between borders is consistent."""
            return text[:inner_width].ljust(inner_width)

        def center(text: str) -> str:
            return f"│{text[:inner_width].center(inner_width)}│"
        
        empty_line = f"│{' ' * inner_width}│"

        render = [
            top_border,
            f"│{pad(f'Round: {Round(state['stage'].value).label()}    Total Pot: {pot}')}│",
            mid_border,
            
            # --- Opponent (Top) ---
            center(f"P{opp_player}"),
            center(f"Stack: {opp_stack}"),
            center(f"{opp_hand}"),
            empty_line, 
            center(f"(Bet: {opp_bet})"),
            
            # --- Center Board ---
            empty_line,
            center("COMMUNITY CARDS"),
            center(community),
            empty_line,

            # --- Player (Bottom) ---
            center(f"(Bet: {my_bet})"),
            empty_line, 
            center(f"{my_hand}"),
            center(f"Stack: {my_stack}"),
            center(f"P{current_player}"),
            
            mid_border,
            # Put Min to Call and Legal Actions on their own aligned lines
            f"│{pad(f'Min to Call: {to_call}')}│",
            f"│{pad(f'Legal Actions: {legal_acts_str}')}│",
            bot_border
        ]
        
        return "\n".join(render)

    def render(self):
        if self.env.render_mode == "ansi":
            return self.ansi_render()
        return self.env.render()

    @staticmethod
    def action_to_string(action) -> str:
        """
        Convert an action (Enum, int, or None) into a human‑readable string.

        This is the single helper that callers (debug tools, logs, etc.) should
        use when they want to display an action.
        """
        if action is None:
            return "No action (terminal)"

        try:
            enum_val = Action(int(action))
            return enum_val.label()
        except (ValueError, TypeError):
            # Fall back to the raw representation if we can't interpret it.
            return str(action)
