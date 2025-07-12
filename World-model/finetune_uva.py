import argparse
from collections import deque
from pathlib import Path
from typing import Deque, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import tqdm

from unified_video_action.unified_video_action.policy.unified_video_action_policy import (
    UnifiedVideoActionPolicy,
)
from World_model.real_pusht_env import RealPushTEnv

# -----------------------------------------------------------------------------
# Simple in-memory dataset to hold (obs_seq, action_seq)
# -----------------------------------------------------------------------------


class SequenceDataset(Dataset):
    def __init__(self, horizon: int):
        self.horizon = horizon
        self.frames: Deque[np.ndarray] = deque(maxlen=10000)  # each is (C,H,W)
        self.actions: Deque[np.ndarray] = deque(maxlen=10000)  # each is (2,)

    def add_episode(self, frames: np.ndarray, actions: np.ndarray):
        """frames: (T+1,3,H,W)  actions: (T,2)"""
        T = actions.shape[0]
        for t in range(T - self.horizon + 1):
            self.frames.append(frames[t : t + self.horizon + 1])  # includes cond+target
            self.actions.append(actions[t : t + self.horizon])

    # torch dataset interface
    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        imgs = self.frames[idx]  # (H+1,3,H,W)
        acts = self.actions[idx]
        # UVA expects B,T,C,H,W where T=16 (8 cond+8 future) and action shape matches
        imgs_t = torch.tensor(imgs, dtype=torch.float32)  # [T+1,3,H,W]
        # Use first 16 frames (some padding if shorter)
        obs = {
            "image": imgs_t[:16],
            "agent_pos": torch.zeros(16, 2),  # placeholder (not used in loss)
        }
        sample = {"obs": obs, "action": torch.tensor(acts[:16], dtype=torch.float32)}
        # Add batch dim later in collate_fn or DataLoader default stack
        return sample


# -----------------------------------------------------------------------------
# Finetune utility
# -----------------------------------------------------------------------------


def collate_fn(batch):
    # Stack along batch dimension and add time dim
    B = len(batch)
    images = torch.stack([b["obs"]["image"] for b in batch])  # B,T,C,H,W
    agent_pos = torch.stack([b["obs"]["agent_pos"] for b in batch])
    actions = torch.stack([b["action"] for b in batch])
    return {"obs": {"image": images, "agent_pos": agent_pos}, "action": actions}



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Pretrained UVA .ckpt path")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--episodes", type=int, default=50, help="Real env episodes to collect")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--save", default="finetuned_uva.ckpt")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load UVA policy (weights and normalizer)
    payload = torch.load(args.checkpoint, map_location="cpu")
    cfg = payload["cfg"]
    workspace_cls = __import__(cfg.model._target_.rsplit(".", 1)[0], fromlist=[cfg.model._target_.split(".")[-1]]).
    workspace = workspace_cls.TrainUnifiedVideoActionWorkspace(cfg, output_dir="/tmp/fine")
    workspace.load_payload(payload)
    policy: UnifiedVideoActionPolicy = workspace.model.to(device)
    policy.train()  # enable grads

    optimizer = torch.optim.Adam(policy.model.parameters(), lr=args.lr)

    # Collect real env data
    env = RealPushTEnv()
    dataset = SequenceDataset(horizon=16)
    for ep in range(args.episodes):
        obs = env.reset()
        frames = [np.transpose(env.render(), (2, 0, 1)) / 255.0]
        actions = []
        done = False
        while not done:
            action = env.action_space.sample()  # random policy for data collection
            obs, reward, done, _ = env.step(action)
            frames.append(np.transpose(env.render(), (2, 0, 1)) / 255.0)
            actions.append(action)
        dataset.add_episode(np.stack(frames), np.stack(actions))
        print(f"Collected episode {ep}, dataset size={len(dataset)}")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    for epoch in range(args.epochs):
        losses = []
        for batch in tqdm.tqdm(loader, desc=f"Fine-tune epoch {epoch}"):
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else {sk: sv.to(device) for sk, sv in v.items()}) for k, v in batch.items()}
            loss, _ = policy.compute_loss(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f"Epoch {epoch} mean loss: {np.mean(losses):.4f}")

    # Save finetuned weights
    torch.save(policy.state_dict(), args.save)
    print("Saved finetuned model to", args.save)


if __name__ == "__main__":
    main() 