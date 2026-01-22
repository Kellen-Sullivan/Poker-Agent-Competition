from pettingzoo.classic import texas_holdem_no_limit_v6
import numpy as np


"""
Wrap the texas_holdem_v4 environment to include an ansi
option for rendering and make ___ changes to the state representation?

TODO: update observation and/or info to be more human readable and 
easy to understand for others
"""
    
class TexasHoldEm():
    """
    Wrapper of texas_holdem_v4 which is a wrapper of RLCardGame
    Structure: AECEnv -> texas_holdem_v4 -> RLCardGame
    """
    # Define readable constants following RLCard's implementation
    FOLD = 0
    CHECK = 1
    CALL = 2
    RAISEHALFPOT = 3
    RAISEPOT = 4
    ALLIN = 5

    ACTION_NAMES = {FOLD: "Fold", CHECK: "Check", CALL: "Call", RAISEHALFPOT: "Raise Half Pot", RAISEPOT: "Raise Pot", ALLIN: "All In"}

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
        state = game.get_state(game.get_player_id())

        info["hand_cards"] = state["hand"]
        
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
