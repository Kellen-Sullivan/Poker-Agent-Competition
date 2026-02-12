# Implement your agent here.
#
# Your agent must have an act(self, observation, env) method that returns
# a valid action index for the current turn.
#
# Example:
#   from texas_holdem_env import TexasHoldEm, Action
#   from examples.poker_eval import HandEvaluator, HandRank
#
#   class MyAgent:
#       def act(self, observation, env):
#           action_mask = observation["action_mask"]
#           valid_actions = np.flatnonzero(action_mask)
#           return valid_actions[0]  # Replace with your logic
