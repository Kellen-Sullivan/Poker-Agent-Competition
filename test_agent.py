"""
Test your agent against example agents.

Run from the project root:
    python test_agent.py

To test your own agent, implement it in workspace/agent.py and update the imports below.
"""
from texas_holdem_env import TexasHoldEm
from examples.example_usage import run_competition, run_hand
from examples.example_agents import AlwaysFoldAgent, RandomAgent, HeuristicAgent

# TODO: Import your agent from workspace and use it below
# from workspace.agent import MyAgent


def main():
    # Run a full competition (example agents)
    # Arguments: (Number of Hands, Player 1 Agent, Player 2 Agent)
    run_competition(10000, HeuristicAgent(), RandomAgent())

    # Run a single debug hand
    # Uncomment the line below to see a step-by-step log of a single game.
    # run_hand(RandomAgent(), AlwaysFoldAgent())


if __name__ == "__main__":
    main()
