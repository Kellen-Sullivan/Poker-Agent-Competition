from example_agents import RandomAgent, AlwaysFoldAgent, HeuristicAgent
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

        # Progress tracker
        if (i + 1) % 100 == 0:
            print(f"Hand {i+1}/{num_hands} complete.")

        env.close()

    # Show the results
    BIG_BLIND_SIZE = 2
    print("\n=== Final Results ===")
    print(f"Total Hands: {num_hands}")
    for agent in total_rewards:
        avg_bb_per_hand = total_rewards[agent] / (num_hands * BIG_BLIND_SIZE)
        win_rate = (wins[agent] / num_hands) * 100
        print(f"{agent}: {total_rewards[agent]:.1f} Total Chips | {(avg_bb_per_hand * 100):.2f} bb/100 Hands | Win Rate: {win_rate:.1f}%")


def run_hand(agent_a, agent_b):
    env = TexasHoldEm(num_players=2, render_mode="ansi")
    env.reset()

    # define policies for each agent
    policies = {"player_0": agent_a, "player_1": agent_b}

    # Helpful labels for the terminal output
    agent_labels = {
        "player_0": f"player_0 ({type(agent_a).__name__})",
        "player_1": f"player_1 ({type(agent_b).__name__})",
    }

    # Simulate one hand until a plyer wins
    step_idx = 0
    for agent in env.agent_iter:
        observation, reward, termination, truncation, info = env.last()

        print("\n" + "=" * 80)
        print(f"Step {step_idx} - To act: {agent_labels[agent]}")
        print(env.render())

        if termination or truncation:
            action = None
            print(f"Game is terminal. No action taken.")
        else:
            policy = policies[agent]
            action = policy.act(observation, env)

            action_str = TexasHoldEm.action_to_string(action)
            print(f"Action taken by {agent_labels[agent]}: {action_str}")

        step_idx += 1
        
        env.step(action)

    print("\n" + "=" * 80)
    print("Hand complete.\n")
    env.close()


def main():
    #run_hand(RandomAgent(), RandomAgent())
    run_competition(10000, HeuristicAgent(), RandomAgent())
    

if __name__ == "__main__":
    main()