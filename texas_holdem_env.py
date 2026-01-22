from pettingzoo.classic import texas_holdem_no_limit_v6
import numpy as np


"""
Wrap the texas_holdem_no_limit_v6 environment to include an ansi
option for rendering and add human readable observation option to the state representation

TODO: update observation and/or info to be more human readable and 
easy to understand for others

TODO: Figure out the best way to do readable constants

TODO: Make ansi version work
"""
    
class TexasHoldEm():
    """
    Wrapper of texas_holdem_no_limit_v6 which is a wrapper of RLCardGame
    """
    # Define readable constants following RLCard's implementation
    FOLD = 0
    CHECK_CALL = 1
    RAISE_HALF_POT = 2
    RAISE_FULL_POT = 3
    ALL_IN = 4

    PREFLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3
    END_HIDDEN = 4
    SHOWDOWN = 5

    # ACTION_NAMES = {FOLD: "Fold", CHECK_CALL: "Check or Call", RAISEHALFPOT: "Raise Half Pot", RAISEFULLPOT: "Raise Full Pot", ALLIN: "All In"}

    def __init__(self, num_players=2, render_mode="ansi"):
        self.env = texas_holdem_no_limit_v6.env(num_players=num_players, render_mode=render_mode)
        self.env.reset()

    def reset(self, **kwargs):
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
        current_player_id = game.get_player_id()
        opponent_player_id = (current_player_id + 1) % 2
        state = game.get_state(current_player_id)

        human_readable_dict = {
            # Context
            'current_player': state['current_player'],
            'round': state['stage'],
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
            'amount_to_call': max(0, state['all_chips'][opponent_player_id] - state['all_chips'][current_player_id]),

            # Actions
            'legal_actions': state['legal_actions'],             
        }
        
        observation["human_readable"] = human_readable_dict
        
        return observation, reward, termination, truncation, info
    
    def ansi_render(self):
        env = self.env.unwrapped.env
        game = env.game

        current_player_id = game.get_player_id()
        state = game.get_state(current_player_id)

        return state

    def render(self):
        if self.env.render_mode == "ansi":
            return self.ansi_render()
        return self.env.render()
