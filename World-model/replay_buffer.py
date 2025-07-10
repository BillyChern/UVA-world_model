import numpy as np
import torch
from collections import deque
from typing import Tuple, Deque, List

class EpisodeReplayBuffer:
    """Fixed-capacity episode replay buffer supporting random sequence sampling."""

    def __init__(self, capacity_episodes: int, horizon: int, device: str = "cpu"):
        self.capacity = capacity_episodes
        self.horizon = horizon
        self.device = device
        self.episodes: Deque[Tuple[np.ndarray, np.ndarray]] = deque()

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.episodes)

    # ------------------------------------------------------------------
    def add_episode(self, frames: np.ndarray, actions: np.ndarray):
        """Store one trajectory.

        Args:
            frames: (T+1, 3, H, W) array in range [0,1]. One extra frame so that
                    `horizon` frames can be used as conditioning if needed.
            actions: (T, action_dim) array.
        """
        assert frames.ndim == 4 and actions.ndim == 2
        assert frames.shape[0] == actions.shape[0] + 1
        self.episodes.append((frames.astype(np.float32), actions.astype(np.float32)))
        while len(self.episodes) > self.capacity:
            self.episodes.popleft()

    # ------------------------------------------------------------------
    def sample_batch(self, batch_size: int):
        imgs: List[torch.Tensor] = []
        acts: List[torch.Tensor] = []
        for _ in range(batch_size):
            ep_frames, ep_actions = self.episodes[np.random.randint(len(self.episodes))]
            T = ep_actions.shape[0]
            start = 0 if T <= self.horizon else np.random.randint(0, T - self.horizon + 1)
            frame_seq = ep_frames[start : start + self.horizon]
            act_seq = ep_actions[start : start + self.horizon]
            imgs.append(torch.tensor(frame_seq, dtype=torch.float32, device=self.device))
            acts.append(torch.tensor(act_seq, dtype=torch.float32, device=self.device))
        # Shape to (B, T, C, H, W) and (B, T, ...)
        images = torch.stack(imgs)  # (B,T,3,H,W)
        actions = torch.stack(acts)
        obs = {
            "image": images,
            "agent_pos": torch.zeros(images.size(0), self.horizon, 2, device=self.device),
        }
        return {"obs": obs, "action": actions} 