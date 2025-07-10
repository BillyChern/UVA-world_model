import os
import pathlib
import math
from typing import Tuple, Dict, List

import numpy as np
import gym
from gym import spaces

import torch

# Import UVA utilities – we rely on the existing repository already in the workspace
from unified_video_action.policy.unified_video_action_policy import (
    UnifiedVideoActionPolicy,
)
from unified_video_action.utils.data_utils import normalize_action
from unified_video_action.utils.data_utils import resize_image_eval


class UVAWorldModelEnv(gym.Env):
    """A lightweight Gym-style wrapper around a pretrained UVA video+action model.

    The environment keeps a fixed-length history of rendered frames (``n_obs_steps``)
    and predicts the next frame given the agent’s action using UVA’s *dynamic* model
    head.  Reward is approximated by the distance between the predicted block centroid
    and the (fixed) goal position, inspired by the original PushT reward function.
    """

    metadata = {"render.modes": ["rgb_array"], "video.frames_per_second": 10}

    def __init__(
        self,
        checkpoint_path: str = None,
        *,
        policy: UnifiedVideoActionPolicy = None,
        device: str = "cuda:0",
        n_obs_steps: int = 8,
        n_action_steps: int = 8,
        render_size: int = 96,
        max_episode_steps: int = 200,
    ) -> None:
        super().__init__()

        self.device = torch.device(device)
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.render_size = render_size
        self.max_episode_steps = max_episode_steps

        # ------------------------------------------------------------------
        # Obtain policy (either passed in or loaded from checkpoint)
        # ------------------------------------------------------------------
        if policy is not None:
            self.policy = policy
        else:
            assert (
                checkpoint_path is not None
            ), "Either policy or checkpoint_path must be provided."
            import hydra

            payload = torch.load(checkpoint_path, map_location="cpu")
            cfg = payload["cfg"]
            workspace_cls = hydra.utils.get_class(cfg.model._target_)

            tmp_dir = pathlib.Path("/tmp/uva_workspace").as_posix()
            os.makedirs(tmp_dir, exist_ok=True)
            workspace = workspace_cls(cfg, output_dir=tmp_dir)
            workspace.load_payload(payload)

            self.policy: UnifiedVideoActionPolicy = (
                workspace.ema_model if hasattr(workspace, "ema_model") else workspace.model
            )
        self.policy.eval()
        self.policy.to(self.device)

        # Freeze parameters for safety.
        for p in self.policy.parameters():
            p.requires_grad_(False)

        self.vae = self.policy.vae_model  # AutoencoderKL instance
        self.vae.to(self.device)
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad_(False)

        # Observation consists of stacked RGB frames and current agent position.
        # We concatenate frames along the channel dimension to leverage SB3’s default
        # CNN extractor (C = n_obs_steps * 3).
        img_shape = (self.n_obs_steps * 3, self.render_size, self.render_size)
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(0.0, 1.0, shape=img_shape, dtype=np.float32),
                "agent_pos": spaces.Box(0.0, 512.0, shape=(2,), dtype=np.float32),
            }
        )

        # Action = target (x,y) in pixel space – we normalise to [-1, 1] for PPO and
        # rescale internally.
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

        # Pre-compute goal centre (fixed for PushT when fix_goal=True).
        self.goal_xy = np.array([self.render_size / 2, self.render_size / 2])
        self.max_dist = math.sqrt(2 * (self.goal_xy[0] ** 2))

        # Internal state holders
        self._frame_history: List[np.ndarray] = []  # list of (C,H,W) uint8 images
        self._agent_pos: np.ndarray = np.zeros(2, dtype=np.float32)
        self._step_count: int = 0

    # ---------------------------------------------------------------------
    # Gym API
    # ---------------------------------------------------------------------
    def reset(self):
        self._step_count = 0

        # Random initial positions roughly following PushT’s ranges
        self._agent_pos = np.random.uniform(low=50, high=450, size=(2,)).astype(
            np.float32
        )

        # Bootstrap image history with blank frames (white background)
        blank = np.ones((3, self.render_size, self.render_size), dtype=np.float32)
        self._frame_history = [blank.copy() for _ in range(self.n_obs_steps)]

        return self._get_obs()

    def step(self, action: np.ndarray):
        # Rescale action from [-1,1] to [0,512]
        action_real = ((action + 1.0) / 2.0) * 512.0
        action_real = np.clip(action_real, 0, 512)

        # Build NACTIONS tensor expected by UVA (shape: [1, n_action_steps, 2])
        action_seq = np.repeat(action_real.reshape(1, 1, 2), self.n_action_steps, axis=1)
        action_torch = torch.from_numpy(action_seq).float().to(self.device)
        nactions = normalize_action(
            self.policy.normalizer,
            self.policy.normalizer_type,
            action_torch,
        )

        # Build conditioning frames tensor expected by UVA (shape: [1, n_obs_steps, 3, H, W])
        cond_frames = np.stack(self._frame_history, axis=0)  # (T,C,H,W)
        cond_frames_t = torch.from_numpy(cond_frames).float().unsqueeze(0).to(self.device)
        cond_frames_t = cond_frames_t * 2.0 - 1.0  # Map [0,1] -> [-1,1]

        # Predict next frame latent using UVA’s *dynamic model* head
        with torch.no_grad():
            pred_latents, _ = self.policy.model.sample_tokens(
                bsz=1,
                cond=cond_frames_t,
                num_iter=self.policy.autoregressive_model_params.num_iter,
                cfg=self.policy.autoregressive_model_params.cfg,
                cfg_schedule=self.policy.autoregressive_model_params.cfg_schedule,
                temperature=self.policy.autoregressive_model_params.temperature,
                progress=False,
                nactions=nactions,
                task_mode="dynamic_model",
                vae_model=self.vae,
            )

            # pred_latents has shape (B*T, C, h, w). We take the first frame only.
            pred_latents = pred_latents[0].unsqueeze(0)
            pred_img = self.vae.decode(pred_latents)  # (-1,1)
            pred_img = (pred_img.clamp(-1.0, 1.0) + 1.0) / 2.0  # (0,1)
            pred_img = pred_img.squeeze(0).cpu().numpy()  # (3, H, W)

        # Update internal buffers
        self._frame_history.pop(0)
        self._frame_history.append(pred_img.astype(np.float32))
        self._agent_pos = action_real.astype(np.float32)
        self._step_count += 1

        # Compute reward – approximate block centre via grey-pixel centroid
        reward, done = self._compute_reward_done(pred_img)
        if self._step_count >= self.max_episode_steps:
            done = True

        info = {}
        return self._get_obs(), reward, done, info

    def render(self, mode="rgb_array"):
        assert mode == "rgb_array"
        # Return most recent frame in HWC uint8 format
        img = (self._frame_history[-1] * 255).clip(0, 255).astype(np.uint8)
        return np.transpose(img, (1, 2, 0))  # HWC

    def _get_obs(self):
        # Stack frames along channel dim (C = n_obs_steps * 3)
        stacked_frames = np.concatenate(self._frame_history, axis=0)
        obs = {
            "image": stacked_frames.astype(np.float32),
            "agent_pos": self._agent_pos.copy(),
        }
        return obs

    # ------------------------------------------------------------------
    # Reward helper
    # ------------------------------------------------------------------
    def _compute_reward_done(self, img_chw: np.ndarray) -> Tuple[float, bool]:
        """Crude approximation: detect block (grey) centroid and measure distance
        to fixed goal centre. Colour thresholds are heuristic and may not work for
        every frame but suffice for a proof-of-concept."""
        img = (img_chw * 255).astype(np.uint8)
        img_hwc = np.transpose(img, (1, 2, 0))  # HWC

        # Simple grey mask – R,G,B within 15 of each other and mid-range intensity.
        r, g, b = img_hwc[:, :, 0], img_hwc[:, :, 1], img_hwc[:, :, 2]
        grey_mask = (
            (np.abs(r.astype(int) - g.astype(int)) < 15)
            & (np.abs(g.astype(int) - b.astype(int)) < 15)
            & (r > 80)
            & (r < 200)
        )

        ys, xs = np.where(grey_mask)
        if len(xs) == 0:
            # No block detected – give small negative reward
            return 0.0, False

        cx, cy = xs.mean(), ys.mean()
        dist = math.sqrt(((cx - self.goal_xy[0]) ** 2) + ((cy - self.goal_xy[1]) ** 2))
        reward = max(0.0, 1.0 - (dist / self.max_dist))
        done = dist < 5.0  # pixel threshold
        return reward, done 