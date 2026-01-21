from agent import RandomAgent, AlwaysFoldAgent
from texas_holdem_env import TexasHoldEm


def run_competition(num_hands: int, agent_a, agent_b):
    env = TexasHoldEm(num_players=2, render_mode="ansi")
    env.reset()

    # define policies for each agent
    Agent_A = agent_a
    Agent_B = agent_b

    # number of hands to simulate to track cumulative reward
    num_hands = num_hands

    # number of chips each agent cumulatively won 
    total_rewards = {"Agent_A": 0.0, "Agent_B": 0.0}
    wins = {"Agent_A": 0.0, "Agent_B": 0.0}

    for i in range(num_hands):
        env.reset()

        # Swap who starts betting for each new hand
        if i % 2 == 0:
            seat_map = {"player_0": Agent_A, "player_1": Agent_B}
            agent_identity = {"player_0": "Agent_A", "player_1": "Agent_B"}
        else:
            seat_map = {"player_0": Agent_B, "player_1": Agent_A}
            agent_identity = {"player_0": "Agent_B", "player_1": "Agent_A"}

        # Track chips won in each hand
        hand_rewards = {"Agent_A": 0.0, "Agent_B": 0.0}

        # Simulate one hand until a plyer wins
        for agent in env.agent_iter:
            observation, reward, termination, truncation, info = env.last()

            actual_agent_name = agent_identity[agent]
            hand_rewards[actual_agent_name] += reward

            if termination or truncation:
                action = None
            else:
                policy = seat_map[agent]
                action = policy.act(observation, env)
            
            env.step(action)

        # Update win totals and reward totals after hand ends
        for agent, reward in hand_rewards.items():
            total_rewards[agent] += hand_rewards[agent]
            if reward > 0:
                wins[agent] += 1

        # Progress bar
        if (i + 1) % 100 == 0:
            print(f"Hand {i+1}/{num_hands} complete.")

        env.close()

    # Show the results
    print("\n=== Final Results ===")
    print(f"Total Hands: {num_hands}")
    for agent in total_rewards:
        avg_chips = total_rewards[agent] / num_hands
        win_rate = (wins[agent] / num_hands) * 100
        print(f"{agent}: {total_rewards[agent]:.1f} Total Chips | {avg_chips:.2f} Chips/Hand | Win Rate: {win_rate:.1f}%")


def main():
    run_competition(10000, RandomAgent(), RandomAgent())
    

if __name__ == "__main__":
    main()