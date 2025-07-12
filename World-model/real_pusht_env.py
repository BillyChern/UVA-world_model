from typing import List, Tuple
import numpy as np
import gym
from gym import spaces
import cv2
import functools

# Monkeypatch cv2.drawMarker to gracefully handle numpy-int inputs (tuple conversion)
_orig_draw_marker = cv2.drawMarker

def _safe_draw_marker(img, position, *args, **kwargs):
    if not isinstance(position, tuple):
        try:
            position = tuple(int(x) for x in position)
        except Exception:
            position = (0, 0)
    return _orig_draw_marker(img, position, *args, **kwargs)

cv2.drawMarker = _safe_draw_marker

from unified_video_action.env.pusht.pusht_image_env import PushTImageEnv
from unified_video_action.gym_util.multistep_wrapper import MultiStepWrapper


class RealPushTEnv(gym.Env):
    """Adapter around PushTImageEnv that matches UVAWorldModelEnv obs/action layout.

    We treat this as the "real world" oracle for evaluation.  Unlike UVAWorldModelEnv,
    this environment simulates true physics (pymunk) and computes reward via its own
    internal logic.  Observations are formatted to align with the PPO agent trained
    inside the world model (stacked frame history).
    """

    metadata = {"render.modes": ["rgb_array"], "video.frames_per_second": 10}

    def __init__(self, n_obs_steps: int = 8, render_size: int = 96, max_episode_steps: int = 200):
        super().__init__()

        self.n_obs_steps = n_obs_steps
        self.render_size = render_size
        self.max_episode_steps = max_episode_steps

        base_env = PushTImageEnv(render_size=render_size, fix_goal=True)
        # Wrap to enforce fixed horizon and frame stacking similar to UVA env
        self.env = MultiStepWrapper(
            base_env,
            n_obs_steps=n_obs_steps,
            n_action_steps=1,
            max_episode_steps=max_episode_steps,
        )

        # Observation: stacked images (C = n_obs_steps*3) + agent_pos (2)
        img_shape = (n_obs_steps * 3, render_size, render_size)
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(0.0, 1.0, shape=img_shape, dtype=np.float32),
                "agent_pos": spaces.Box(0.0, 512.0, shape=(2,), dtype=np.float32),
            }
        )
        # Actions are agent target xy in pixel space [-1,1] after scaling.
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

        # Internal frame buffer
        self._frame_history: List[np.ndarray] = []
        self._step_count: int = 0

    def reset(self):
        obs = self.env.reset()
        self._frame_history = [obs["image"].astype(np.float32) / 255.0] * self.n_obs_steps
        self._step_count = 0
        return self._format_obs(obs)

    def step(self, action: np.ndarray):
        # Rescale [-1,1] → [0,512] since PushT expects pixel coords
        action_real = ((action + 1.0) / 2.0) * 512.0
        obs, reward, done, info = self.env.step(action_real)

        self._frame_history.pop(0)
        self._frame_history.append(obs["image"].astype(np.float32) / 255.0)
        self._step_count += 1
        if self._step_count >= self.max_episode_steps:
            done = True
        return self._format_obs(obs), reward, done, info

    def render(self, mode="rgb_array"):
        assert mode == "rgb_array"
        frame = self._frame_history[-1]
        img = (frame * 255).astype(np.uint8)
        return np.transpose(img, (1, 2, 0))

    # ------------------------------------------------------------------
    def _format_obs(self, raw_obs):
        stacked = np.concatenate(self._frame_history, axis=0)
        return {
            "image": stacked.astype(np.float32),
            "agent_pos": raw_obs["agent_pos"].astype(np.float32),
        } 