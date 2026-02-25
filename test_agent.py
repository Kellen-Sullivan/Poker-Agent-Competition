"""
Test your agent against built-in opponents.

Usage:
    python test_agent.py
"""

from poker_env import TexasHoldEm
from agents import RandomAgent, HeuristicAgent
from agent import MyAgent


def run_competition(num_hands, agent_a, agent_b, seed=None):
    """Simulate *num_hands* of heads-up poker and print results."""
    label_a = type(agent_a).__name__
    label_b = type(agent_b).__name__

    env = TexasHoldEm(num_players=2, render_mode="ansi", seed=seed)
    env.reset()

    total_rewards = {label_a: 0.0, label_b: 0.0}
    wins = {label_a: 0, label_b: 0}

    for i in range(num_hands):
        env.reset()

        if i % 2 == 0:
            seat_map = {"player_0": agent_a, "player_1": agent_b}
            agent_identity = {"player_0": label_a, "player_1": label_b}
        else:
            seat_map = {"player_0": agent_b, "player_1": agent_a}
            agent_identity = {"player_0": label_b, "player_1": label_a}

        hand_rewards = {label_a: 0.0, label_b: 0.0}

        for agent in env.agent_iter:
            observation, reward, termination, truncation, info = env.last()
            actual_agent_name = agent_identity[agent]
            hand_rewards[actual_agent_name] += reward

            if termination or truncation:
                action = None
            else:
                action = seat_map[agent].act(observation)

            env.step(action)

        for name, reward in hand_rewards.items():
            total_rewards[name] += reward
            if reward > 0:
                wins[name] += 1

        if (i + 1) % 1000 == 0:
            print(f"Hand {i + 1}/{num_hands} complete.")

    env.close()

    BIG_BLIND = 2
    max_len = max(len(n) for n in total_rewards)
    print("\n=== Final Results ===")
    print(f"Total Hands: {num_hands}")
    for name in total_rewards:
        avg_bb = total_rewards[name] / (num_hands * BIG_BLIND)
        win_rate = (wins[name] / num_hands) * 100
        padded = name.ljust(max_len)
        print(
            f"  {padded}  {total_rewards[name]:>+10.1f} chips | "
            f"{avg_bb * 100:>+8.2f} bb/100 | Win rate: {win_rate:5.1f}%"
        )


if __name__ == "__main__":
    run_competition(10_000, MyAgent(), HeuristicAgent())
