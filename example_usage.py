import numpy as np 
from texas_holdem_env import TexasHoldEm


class RandomAgent:
    def act(self, observation, env):
        """Returns a random index where the mask is 1"""
        action_mask = observation["action_mask"]
        valid_actions = np.flatnonzero(action_mask)     
        return np.random.choice(valid_actions)
    

class AlwaysFoldAgent:
    def act(self, observation, env: TexasHoldEm):
        """
        In PettingZoo Texas Hold'em
        action mask: [Call, Raise, Fold, Check]
        """
        action_mask = observation["action_mask"]
        valid_actions = np.flatnonzero(action_mask)  
        # Always try to Fold if it's legal.
        # If not, pick the first available legal action   
        if env.FOLD in valid_actions:
            return env.FOLD
        return valid_actions[0]
    

def main():
    env = TexasHoldEm(num_players=2, render_mode="ansi")
    env.reset(seed=42)

    # define policies for each agent
    # Petting Zoo uses the "player_idx" to determine each agent
    policies = {
        "player_0": RandomAgent(),
        "player_1": RandomAgent()
    }

    # number of hands to simulate to track cumulative reward
    num_hands = 10000

    # number of chips each agent cumulatively won 
    total_chips = {"player_0": 0.0, "player_1": 0.0}
    wins = {"player_0": 0.0, "player_1": 0.0}

    for i in range(num_hands):
        env.reset(seed=42)

        # track rewards for this specific hand
        hand_rewards = {"player_0": 0.0, "player_1": 0.0}

        for agent in env.agent_iter:
            observation, reward, termination, truncation, info = env.last()

            # Add rewards to current hand rewards
            hand_rewards[agent] += reward

            if termination or truncation:
                action = None
            else:
                policy = policies[agent]

                action = policy.act(observation, env)
                #print(f"Agent {agent} chose to {env.ACTION_NAMES[action]}")
            
            env.step(action)

        # Update hand totals after hand ends
        for agent, reward in hand_rewards.items():
            total_chips[agent] += hand_rewards[agent]
            if reward > 0:
                wins[agent] += 1

        # Progress bar
        if (i + 1) % 100 == 0:
            print(f"Hand {i+1}/{num_hands} complete.")

        env.close()

    # Show the results
    print("\n=== Final Results ===")
    print(f"Total Hands: {num_hands}")
    for agent in total_chips:
        avg_chips = total_chips[agent] / num_hands
        win_rate = (wins[agent] / num_hands) * 100
        print(f"{agent}: {total_chips[agent]:.1f} Total Chips | {avg_chips:.2f} Chips/Hand | Win Rate: {win_rate:.1f}%")

if __name__ == "__main__":
    main()