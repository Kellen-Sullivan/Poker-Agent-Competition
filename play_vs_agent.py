"""
Play a hand of Texas Hold'em against an AI agent in the terminal.

Usage:
    python play_vs_agent.py                           # play vs HeuristicAgent
    python play_vs_agent.py --agent drl               # play vs trained DRL agent
    python play_vs_agent.py --agent random            # play vs RandomAgent
    python play_vs_agent.py --hands 5                 # play multiple hands
"""

import argparse
import numpy as np
from texas_holdem_env import TexasHoldEm, Action
from example_agents import RandomAgent, HeuristicAgent


# ── Human "agent" that reads from stdin ──────────────────────────────

ACTION_KEYS = {
    "0": Action.FOLD,
    "1": Action.CHECK_CALL,
    "2": Action.RAISE_HALF_POT,
    "3": Action.RAISE_FULL_POT,
    "4": Action.ALL_IN,
    "f": Action.FOLD,
    "c": Action.CHECK_CALL,
    "h": Action.RAISE_HALF_POT,
    "r": Action.RAISE_FULL_POT,
    "a": Action.ALL_IN,
}


class HumanAgent:
    """Prompts the human player to choose an action via the terminal."""

    def act(self, observation, env):
        action_mask = observation["action_mask"]
        valid_actions = np.flatnonzero(action_mask)

        # Build the prompt showing only legal actions
        options = []
        for idx in valid_actions:
            a = Action(idx)
            shortcut = {
                Action.FOLD: "f",
                Action.CHECK_CALL: "c",
                Action.RAISE_HALF_POT: "h",
                Action.RAISE_FULL_POT: "r",
                Action.ALL_IN: "a",
            }[a]
            options.append(f"  [{shortcut}/{idx}] {a.label()}")

        prompt = "\n".join(options) + "\nYour action: "

        while True:
            choice = input(prompt).strip().lower()
            if choice in ACTION_KEYS and int(ACTION_KEYS[choice]) in valid_actions:
                return ACTION_KEYS[choice]
            # Also accept the raw int
            if choice.isdigit() and int(choice) in valid_actions:
                return Action(int(choice))
            print("Invalid choice. Try again.\n")


# ── Run one hand ─────────────────────────────────────────────────────

def play_hand(human_seat, opponent, hand_num=1):
    """Play a single hand. Returns chips won by the human."""
    env = TexasHoldEm(num_players=2, render_mode="ansi")
    env.reset()

    if human_seat == 0:
        seat_map = {"player_0": HumanAgent(), "player_1": opponent}
        labels = {"player_0": "You", "player_1": "Agent"}
    else:
        seat_map = {"player_0": opponent, "player_1": HumanAgent()}
        labels = {"player_0": "Agent", "player_1": "You"}

    human_key = f"player_{human_seat}"
    human_reward = 0.0
    step_idx = 0

    print(f"\n{'=' * 80}")
    print(f"  HAND {hand_num}   |   You are Player {human_seat}"
          f"  ({'Dealer/SB' if human_seat == 0 else 'Big Blind'})")
    print(f"{'=' * 80}")

    for agent in env.agent_iter:
        observation, reward, termination, truncation, info = env.last()

        if agent == human_key:
            human_reward += reward

        if termination or truncation:
            action = None
        else:
            who = labels[agent]
            # Only render from the current player's perspective
            print(f"\n--- {who}'s turn ---")
            print(env.render())

            policy = seat_map[agent]
            action = policy.act(observation, env)

            action_str = TexasHoldEm.action_to_string(action)
            if who == "Agent":
                print(f"Agent plays: {action_str}")

        step_idx += 1
        env.step(action)

    # Final result
    print(f"\n{'=' * 80}")
    if human_reward > 0:
        print(f"  You WIN  +{human_reward:.0f} chips!")
    elif human_reward < 0:
        print(f"  You LOSE  {human_reward:.0f} chips.")
    else:
        print(f"  Push (0 chips).")
    print(f"{'=' * 80}\n")

    env.close()
    return human_reward


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Play poker against an AI agent.")
    parser.add_argument(
        "--agent",
        choices=["random", "heuristic", "drl"],
        default="heuristic",
        help="Which agent to play against (default: heuristic)",
    )
    parser.add_argument(
        "--model",
        default="models/ppo_poker_final",
        help="Path to DRL model (only used with --agent drl)",
    )
    parser.add_argument(
        "--hands",
        type=int,
        default=1,
        help="Number of hands to play (default: 1)",
    )
    args = parser.parse_args()

    # Build opponent
    if args.agent == "random":
        opponent = RandomAgent()
        opp_name = "RandomAgent"
    elif args.agent == "drl":
        from drl_agent import DRLAgent
        opponent = DRLAgent(args.model)
        opp_name = "DRLAgent"
    else:
        opponent = HeuristicAgent()
        opp_name = "HeuristicAgent"

    print(f"\n  Playing against: {opp_name}")
    print(f"  Hands to play:  {args.hands}\n")

    total = 0.0
    for h in range(args.hands):
        seat = h % 2  # alternate dealer each hand
        result = play_hand(seat, opponent, hand_num=h + 1)
        total += result

    if args.hands > 1:
        print(f"Session result: {total:+.0f} chips over {args.hands} hands.")


if __name__ == "__main__":
    main()
