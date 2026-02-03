from texas_holdem_env import TexasHoldEm
from example_usage import run_competition, run_hand
from example_agents import AlwaysFoldAgent, RandomAgent

# TODO: Import your agent class here
# from my_agent import MyAgent 


def main():
    # Run a full competition
    # Arguments: (Number of Hands, Player 1 Agent, Player 2 Agent)
    # This will print the win-rates and stats to the console.
    run_competition(10000, AlwaysFoldAgent(), RandomAgent())

    # Run a single debug hand
    # Uncomment the line below to see a step-by-step log of a single game.
    run_hand(RandomAgent(), AlwaysFoldAgent())

if __name__ == "__main__":
    main()