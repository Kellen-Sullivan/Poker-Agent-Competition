"""
Training script for the DRL poker agent.

Three-phase curriculum:
  Phase 1 - vs RandomAgent     (100 K steps)  -- learn basic hand evaluation
  Phase 2 - vs HeuristicAgent  (200 K steps)  -- exploit predictable play
  Phase 3 - Self-play           (500 K steps) -- robust generalisation

Uses MaskablePPO from sb3-contrib with action masking.

Run:
    python -m rl_example.train
"""

import os
import tempfile

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback

from rl_example.gym_wrapper import PokerGymEnv
from rl_example.feature_extractor import extract_features, FEATURE_DIM
from agents import RandomAgent, HeuristicAgent
from poker_env import TexasHoldEm, Action


# ======================================================================
# Self-play helpers
# ======================================================================

class SelfPlayAgent:
    """Wraps a saved MaskablePPO checkpoint so it can act as an opponent."""

    def __init__(self):
        self.model = None

    def load_from_model(self, model):
        tmp_path = os.path.join(tempfile.gettempdir(), "sp_opponent_tmp")
        model.save(tmp_path)
        self.model = MaskablePPO.load(tmp_path)

    def act(self, observation, env):
        if self.model is None:
            mask = observation["action_mask"]
            valid = np.flatnonzero(mask)
            return Action(np.random.choice(valid))

        features = extract_features(observation)
        mask = observation["action_mask"]
        action, _ = self.model.predict(
            features, action_masks=mask, deterministic=False
        )
        return Action(int(action))


class SelfPlayCallback(BaseCallback):
    """Periodically copies current policy weights into the self-play opponent."""

    def __init__(self, poker_env, update_freq=10_000, verbose=0):
        super().__init__(verbose)
        self._poker_env = poker_env
        self._update_freq = update_freq

    def _on_step(self):
        if self.n_calls % self._update_freq == 0:
            sp = SelfPlayAgent()
            sp.load_from_model(self.model)
            self._poker_env.set_opponent(sp)
            if self.verbose:
                print(f"  [Self-play] Updated opponent at step {self.n_calls}")
        return True


# ======================================================================
# Evaluation helper
# ======================================================================

def evaluate(model, opponent, num_hands=1000):
    """Quick evaluation: returns bb/100 of the model vs an opponent."""
    env = TexasHoldEm(num_players=2, render_mode="ansi")
    total_reward = 0.0
    wins = 0

    for i in range(num_hands):
        env.reset()
        hand_reward = 0.0

        if i % 2 == 0:
            model_seat = "player_0"
        else:
            model_seat = "player_1"

        for agent in env.agent_iter:
            obs, reward, term, trunc, info = env.last()

            if agent == model_seat:
                hand_reward += reward

            if term or trunc:
                action = None
            elif agent == model_seat:
                features = extract_features(obs)
                mask = obs["action_mask"]
                a, _ = model.predict(
                    features, action_masks=mask, deterministic=True
                )
                action = int(a)
            else:
                action = int(opponent.act(obs, env))

            env.step(action)

        total_reward += hand_reward
        if hand_reward > 0:
            wins += 1

    env.close()
    bb100 = (total_reward / (num_hands * 2)) * 100
    win_pct = (wins / num_hands) * 100
    print(
        f"  Eval ({num_hands} hands): "
        f"{total_reward:+.1f} chips | "
        f"{bb100:+.2f} bb/100 | "
        f"Win rate: {win_pct:.1f}%"
    )
    return bb100


# ======================================================================
# Environment factory
# ======================================================================

def _mask_fn(env):
    return env.action_masks()


def make_env(opponent):
    """Create an ActionMasker-wrapped PokerGymEnv."""
    env = PokerGymEnv(opponent_agent=opponent)
    return ActionMasker(env, _mask_fn)


# ======================================================================
# Main training loop
# ======================================================================

def train():
    save_dir = "models"
    os.makedirs(save_dir, exist_ok=True)

    # Phase 1: vs Random
    print("=" * 60)
    print("Phase 1: Training vs RandomAgent (100K steps)")
    print("=" * 60)

    env1 = make_env(RandomAgent())
    model = MaskablePPO(
        "MlpPolicy",
        env1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        clip_range=0.2,
        policy_kwargs=dict(net_arch=[64, 64]),
        verbose=1,
    )
    model.learn(total_timesteps=100_000)
    model.save(os.path.join(save_dir, "ppo_phase1_vs_random"))
    print("\nPhase 1 complete.")
    evaluate(model, RandomAgent())
    env1.close()

    # Phase 2: vs Heuristic
    print("\n" + "=" * 60)
    print("Phase 2: Training vs HeuristicAgent (200K steps)")
    print("=" * 60)

    env2 = make_env(HeuristicAgent())
    model.set_env(env2)
    model.learn(total_timesteps=200_000, reset_num_timesteps=False)
    model.save(os.path.join(save_dir, "ppo_phase2_vs_heuristic"))
    print("\nPhase 2 complete.")
    evaluate(model, HeuristicAgent())
    env2.close()

    # Phase 3: Self-play
    print("\n" + "=" * 60)
    print("Phase 3: Self-play training (500K steps)")
    print("=" * 60)

    sp_opponent = SelfPlayAgent()
    sp_opponent.load_from_model(model)

    sp_gym = PokerGymEnv(opponent_agent=sp_opponent)
    env3 = ActionMasker(sp_gym, _mask_fn)
    model.set_env(env3)

    callback = SelfPlayCallback(
        poker_env=sp_gym, update_freq=10_000, verbose=1
    )
    model.learn(
        total_timesteps=500_000, callback=callback, reset_num_timesteps=False
    )
    model.save(os.path.join(save_dir, "ppo_phase3_selfplay"))
    env3.close()

    # Save final model and evaluate
    final_path = os.path.join(save_dir, "ppo_poker_final")
    model.save(final_path)
    print("\n" + "=" * 60)
    print(f"Training complete!  Final model: {final_path}.zip")
    print("=" * 60)

    print("\nFinal evaluation vs RandomAgent:")
    evaluate(model, RandomAgent())
    print("Final evaluation vs HeuristicAgent:")
    evaluate(model, HeuristicAgent())

    return model


if __name__ == "__main__":
    train()
