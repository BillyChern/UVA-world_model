import argparse
import pathlib
import os

import stable_baselines3 as sb3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

from World-model.uva_world_env import UVAWorldModelEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to pretrained UVA .ckpt file (PushT)")
    parser.add_argument("--logdir", default="runs/ppo_uva", help="TensorBoard output directory")
    parser.add_argument("--timesteps", type=int, default=int(1e6), help="Total PPO training steps")
    parser.add_argument("--device", default="cuda:0", help="Torch device (e.g. cuda:0 or cpu)")
    args = parser.parse_args()

    # SB3 requires a vectorised env even for a single instance.
    def env_fn():
        return UVAWorldModelEnv(checkpoint_path=args.checkpoint, device=args.device)

    env = DummyVecEnv([env_fn])

    model = sb3.PPO(
        policy=sb3.common.policies.MultiInputActorCriticPolicy,
        env=env,
        n_steps=2048,
        batch_size=64,
        learning_rate=3e-4,
        tensorboard_log=args.logdir,
        verbose=1,
        device=args.device,
    )

    model.learn(total_timesteps=args.timesteps)

    save_path = pathlib.Path(args.logdir).joinpath("trained_policy.zip")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(save_path.as_posix())
    print(f"Saved trained policy to {save_path}")


if __name__ == "__main__":
    main() 