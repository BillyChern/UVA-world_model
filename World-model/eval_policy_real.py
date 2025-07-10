import argparse
from pathlib import Path
import numpy as np
import stable_baselines3 as sb3
from stable_baselines3.common.vec_env import DummyVecEnv

from World-model.real_pusht_env import RealPushTEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True, help="Path to trained_policy.zip")
    parser.add_argument("--episodes", type=int, default=20)
    args = parser.parse_args()

    env = DummyVecEnv([lambda: RealPushTEnv()])
    model = sb3.PPO.load(args.policy, env)

    success = []
    for ep in range(args.episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            total_reward += reward[0]
        success.append(total_reward)
        print(f"Episode {ep}: return={total_reward:.3f}")

    print("Mean return:", np.mean(success))


if __name__ == "__main__":
    main() 