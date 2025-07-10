import argparse
from pathlib import Path
from collections import deque
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import tqdm
import stable_baselines3 as sb3
from stable_baselines3.common.vec_env import DummyVecEnv

from World-model.real_pusht_env import RealPushTEnv
from World-model.uva_world_env import UVAWorldModelEnv
from World-model.replay_buffer import EpisodeReplayBuffer
from unified_video_action.policy.unified_video_action_policy import UnifiedVideoActionPolicy


def collect_real_episodes(env, policy, n_eps: int, horizon: int):
    dataset = []
    for ep in range(n_eps):
        obs = env.reset()
        frames = [np.transpose(env.render(), (2, 0, 1)) / 255.0]
        actions = []
        done = False
        while not done:
            action, _ = policy.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            frames.append(np.transpose(env.render(), (2, 0, 1)) / 255.0)
            actions.append(action[0])
        dataset.append((np.stack(frames), np.stack(actions)))
    return dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cycles", type=int, default=10)
    parser.add_argument("--real_eps", type=int, default=5)
    parser.add_argument("--finetune_epochs", type=int, default=2)
    parser.add_argument("--ppo_steps", type=int, default=5000)
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load policy/world model via UVAWorldModelEnv
    # We will update the policy.model (world model) via finetuning loop.
    # First, load policy via checkpoint and extract for env.

    import hydra

    payload = torch.load(args.checkpoint, map_location="cpu")
    cfg = payload["cfg"]
    workspace_cls = hydra.utils.get_class(cfg.model._target_)
    tmp_dir = Path("/tmp/uva_joint").as_posix()
    workspace = workspace_cls(cfg, output_dir=tmp_dir)
    workspace.load_payload(payload)
    policy_obj: UnifiedVideoActionPolicy = (
        workspace.ema_model if hasattr(workspace, "ema_model") else workspace.model
    )
    policy_obj.to(device)
    policy_obj.train()

    # PPO agent (SB3) initialised with UVA env
    def uva_env_fn():
        return UVAWorldModelEnv(policy=policy_obj, device=args.device)

    uva_env = DummyVecEnv([uva_env_fn])
    ppo = sb3.PPO(sb3.common.policies.MultiInputActorCriticPolicy, uva_env, verbose=1, device=args.device)

    # Real environment for data collection
    real_env = RealPushTEnv()

    # Replay buffer and optimizer
    buffer = EpisodeReplayBuffer(capacity_episodes=200, horizon=16, device=device)
    optimizer = torch.optim.Adam(policy_obj.model.parameters(), lr=1e-5)

    for cycle in range(args.cycles):
        print(f"Cycle {cycle}: collecting real episodes…")
        collected = collect_real_episodes(real_env, ppo, args.real_eps, horizon=16)
        for frames, acts in collected:
            buffer.add_episode(frames, acts)

        # Finetune for fixed number of gradient steps per cycle
        steps_per_cycle = args.finetune_epochs * 100  # 100 batches each epoch default
        batch_size = 16
        losses = []
        for step in tqdm.tqdm(range(steps_per_cycle), desc="Finetuning world model"):
            if len(buffer) < 1:
                break
            batch = buffer.sample_batch(batch_size)
            loss, _ = policy_obj.compute_loss(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        if losses:
            print(f"  Finetune mean loss: {np.mean(losses):.4f}")

        # PPO training using updated world model
        print("  PPO training inside updated world model…")
        ppo.learn(total_timesteps=args.ppo_steps, reset_num_timesteps=False)

    # Save final policy
    ppo.save("joint_trained_policy.zip")
    torch.save(policy_obj.state_dict(), "finetuned_uva_final.ckpt")
    print("Training complete; saved policy and finetuned world model.")


if __name__ == "__main__":
    main() 