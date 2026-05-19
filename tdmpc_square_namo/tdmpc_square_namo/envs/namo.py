"""Adapter that exposes the NAMO-RL `DiffDriveCarEnv` to the TD-MPC framework.

The upstream env returns a dict observation, a stubbed reward of 0.0, and uses
gymnasium's 5-tuple step API. TD-MPC's TensorWrapper expects:
    reset() -> (obs_np, info)
    step(a) -> (obs_np, reward, terminated, truncated, info)
    env.observation_space: Box
    env.action_space: Box
    env.max_episode_steps: int

This adapter flattens the dict observation into a single Box, wires in a dense
shaping reward (progress + terminal bonuses + action-rate penalty), and forwards
the rest of the gym API.
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import gymnasium as gym
import numpy as np

# Make the in-repo NAMO-RL package importable without `pip install -e`.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_NAMO_REPO = _REPO_ROOT / "external" / "namo-rl"
if str(_NAMO_REPO) not in sys.path:
    sys.path.insert(0, str(_NAMO_REPO))

from namo_rl import DiffDriveCarEnv, EnvConfig  # noqa: E402
from namo_rl.config import (  # noqa: E402
    ActionLimits,
    GoalTolerance,
    ObsConfig,
    PhysicsTune,
)


def _flat_obs_dim(n_max: int) -> int:
    # robot(7) + goal(3) + movables(n_max,3) + mask(n_max) + last_action(2)
    return 7 + 3 + n_max * 3 + n_max + 2


def _flatten_obs(obs: dict) -> np.ndarray:
    return np.concatenate(
        [
            obs["robot"].astype(np.float32).ravel(),
            obs["goal"].astype(np.float32).ravel(),
            obs["movables"].astype(np.float32).ravel(),
            obs["mask"].astype(np.float32).ravel(),
            obs["last_action"].astype(np.float32).ravel(),
        ],
        axis=0,
    ).astype(np.float32)


class NamoCarTDMPCEnv(gym.Env):
    """Flat-Box + dense-reward shim around `DiffDriveCarEnv`.

    Reward = progress (Δdist_to_goal) − action_rate_penalty + terminal bonuses.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        env_config: EnvConfig,
        reward_style: str = "additive",
        reward_progress_coef: float = 10.0,
        reward_action_rate_coef: float = 0.01,
        reward_step_penalty: float = 0.0,
        reward_goal_bonus: float = 100.0,
        reward_wall_penalty: float = 10.0,
        reward_tolerance_margin: float = 1.0,
        reward_tolerance_terminal_bonus: float = 0.0,
        reward_tolerance_wall_penalty: float = 0.0,
        render_mode: str | None = None,
        render_size: tuple[int, int] = (240, 320),
        camera: str = "top_down",
        recursive_scene_glob: bool = False,
    ):
        super().__init__()
        # The upstream env globs scene_dir non-recursively. The hop_* benchmark
        # suites nest XMLs under run_NNNN/, so we resolve them ourselves and
        # patch the inner env's _xml_paths after construction.
        scene_root = Path(env_config.scene_dir)
        if not scene_root.is_absolute():
            scene_root = Path.cwd() / scene_root
        if recursive_scene_glob:
            xml_paths = sorted(scene_root.rglob("*.xml"))
            if not xml_paths:
                raise FileNotFoundError(
                    f"no .xml scenes found recursively under {scene_root}"
                )
            # Hand the inner constructor a directory that's guaranteed non-empty
            # for its own glob check, then overwrite the path list below.
            env_config = replace(env_config, scene_dir=str(xml_paths[0].parent))

        self._inner = DiffDriveCarEnv(
            env_config,
            render_mode=render_mode,
            render_size=render_size,
            camera=camera,
        )
        if recursive_scene_glob:
            self._inner._xml_paths = xml_paths  # noqa: SLF001 -- intentional override
        self.render_mode = render_mode
        self._cfg = env_config
        self._n_max = int(env_config.obs.max_movables)
        self._v_max = float(env_config.action.v_max)
        self._w_max = float(env_config.action.w_max)

        self.action_space = self._inner.action_space
        obs_dim = _flat_obs_dim(self._n_max)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.max_episode_steps = int(env_config.max_episode_steps)

        self._reward_style = str(reward_style)
        if self._reward_style not in ("additive", "tolerance"):
            raise ValueError(
                f"reward_style must be 'additive' or 'tolerance', got {reward_style!r}"
            )
        self._reward_progress_coef = float(reward_progress_coef)
        self._reward_action_rate_coef = float(reward_action_rate_coef)
        self._reward_step_penalty = float(reward_step_penalty)
        self._reward_goal_bonus = float(reward_goal_bonus)
        self._reward_wall_penalty = float(reward_wall_penalty)
        # Tolerance-style (DMControl/HumanoidBench-aligned) parameters.
        self._reward_tolerance_margin = float(reward_tolerance_margin)
        self._reward_tolerance_terminal_bonus = float(reward_tolerance_terminal_bonus)
        self._reward_tolerance_wall_penalty = float(reward_tolerance_wall_penalty)
        self._goal_pos_tol = float(env_config.goal_tolerance.position)
        # Lazy-import dm_control's tolerance only when needed; it's already a
        # transitive dep via the dmcontrol env adapter.
        self._tolerance_fn = None

        self._prev_dist: float | None = None
        self._prev_action: np.ndarray = np.zeros(2, dtype=np.float32)

    def reset(self, seed: int | None = None, options: dict | None = None):
        obs_dict, info = self._inner.reset(seed=seed, options=options)
        # Initialise progress baseline from goal vector (robot-frame Δ to goal).
        goal = obs_dict["goal"]
        self._prev_dist = float(np.hypot(goal[0], goal[1]))
        self._prev_action = np.zeros(2, dtype=np.float32)
        return _flatten_obs(obs_dict), info

    def step(self, action: np.ndarray):
        a = np.asarray(action, dtype=np.float32).reshape(2)
        obs_dict, _, terminated, truncated, info = self._inner.step(a)

        dist = float(info.get("dist_to_goal", 0.0))
        prev_dist = self._prev_dist if self._prev_dist is not None else dist
        progress = prev_dist - dist
        self._prev_dist = dist

        # Normalised action (used by both reward styles).
        a_norm = np.array(
            [a[0] / max(self._v_max, 1e-8), a[1] / max(self._w_max, 1e-8)],
            dtype=np.float32,
        )
        da = a_norm - np.array(
            [self._prev_action[0] / max(self._v_max, 1e-8),
             self._prev_action[1] / max(self._w_max, 1e-8)],
            dtype=np.float32,
        )
        action_rate_sq = float(da[0] * da[0] + da[1] * da[1])
        self._prev_action = a.copy()

        term = info.get("termination", None)

        if self._reward_style == "additive":
            reward = (
                self._reward_progress_coef * progress
                - self._reward_action_rate_coef * action_rate_sq
                - self._reward_step_penalty
            )
            if term == "goal":
                reward += self._reward_goal_bonus
                info["success"] = 1.0
            elif term == "wall":
                reward -= self._reward_wall_penalty
                info["success"] = 0.0
            elif term == "truncated":
                info.setdefault("success", 0.0)
        else:  # "tolerance" — DMControl/HumanoidBench-style bounded multiplicative
            if self._tolerance_fn is None:
                from dm_control.utils import rewards as _dm_rewards
                self._tolerance_fn = _dm_rewards.tolerance
            tol = self._tolerance_fn
            # near_target ∈ [0, 1]; ramps from 1 inside the goal-tolerance ball
            # down toward `value_at_margin` (default 0.1) at `margin` metres.
            near_target = float(
                tol(dist,
                    bounds=(0.0, self._goal_pos_tol),
                    margin=self._reward_tolerance_margin)
            )
            # Per-component control magnitude penalty, in [0, 1] each, averaged.
            ctrl = tol(a_norm, margin=1.0,
                       value_at_margin=0.0, sigmoid="quadratic")
            small_ctrl = (4.0 + float(np.mean(ctrl))) / 5.0
            reward = near_target * small_ctrl
            if term == "goal":
                reward += self._reward_tolerance_terminal_bonus
                info["success"] = 1.0
            elif term == "wall":
                reward -= self._reward_tolerance_wall_penalty
                info["success"] = 0.0
            elif term == "truncated":
                info.setdefault("success", 0.0)

        return _flatten_obs(obs_dict), float(reward), bool(terminated), bool(truncated), info

    def render(self):
        return self._inner.render()

    def close(self):
        return self._inner.close()


def _build_env_config(cfg) -> EnvConfig:
    """Translate the (Hydra) training config into a NAMO `EnvConfig`."""
    namo_cfg = getattr(cfg, "namo", None)

    def g(key, default):
        if namo_cfg is None:
            return default
        return namo_cfg.get(key, default) if hasattr(namo_cfg, "get") else getattr(
            namo_cfg, key, default
        )

    scene_dir = g("scene_dir", None)
    if scene_dir is None:
        scene_dir = str(_NAMO_REPO / "scenes" / "car")
    scene_dir = str(Path(scene_dir).expanduser())
    if not Path(scene_dir).is_absolute():
        scene_dir = str(_NAMO_REPO / scene_dir)

    return EnvConfig(
        scene_dir=scene_dir,
        frame_skip=int(g("frame_skip", 25)),
        max_episode_steps=int(g("max_episode_steps", 500)),
        goal_tolerance=GoalTolerance(
            position=float(g("goal_position_tol", 0.05)),
            heading=float(g("goal_heading_tol", 3.1416)),
        ),
        action=ActionLimits(
            v_max=float(g("v_max", 0.1)),
            w_max=float(g("w_max", 2.4)),
            v_accel_max=float(g("v_accel_max", 0.2)),
            w_accel_max=float(g("w_accel_max", 4.0)),
            wheel_radius=float(g("wheel_radius", 0.015)),
            wheelbase=float(g("wheelbase", 0.075)),
        ),
        obs=ObsConfig(max_movables=int(g("max_movables", 16))),
        physics=PhysicsTune(
            wheel_armature=float(g("wheel_armature", 5e-4)),
            wheel_damping=float(g("wheel_damping", 0.0)),
        ),
        seed=int(getattr(cfg, "seed", 0)),
    )


def make_env(cfg):
    """Entrypoint used by `tdmpc_square_namo/train.py`.

    Accepts any task whose name starts with `namo_`. Returns a `gym.Env` that the
    upstream `TensorWrapper` (applied by the caller) can consume.
    """
    task = str(getattr(cfg, "task", ""))
    if not task.startswith("namo"):
        raise ValueError(f"Unknown task for tdmpc_square_namo: {task!r}")

    env_config = _build_env_config(cfg)
    namo_cfg = getattr(cfg, "namo", None)

    def g(key, default):
        if namo_cfg is None:
            return default
        return namo_cfg.get(key, default) if hasattr(namo_cfg, "get") else getattr(
            namo_cfg, key, default
        )

    enable_render = bool(getattr(cfg, "save_video", False))
    render_h = int(g("render_height", 240))
    render_w = int(g("render_width", 320))
    camera = str(g("render_camera", "top_down"))

    env = NamoCarTDMPCEnv(
        env_config,
        reward_style=str(g("reward_style", "additive")),
        reward_progress_coef=float(g("reward_progress_coef", 10.0)),
        reward_action_rate_coef=float(g("reward_action_rate_coef", 0.01)),
        reward_step_penalty=float(g("reward_step_penalty", 0.0)),
        reward_goal_bonus=float(g("reward_goal_bonus", 100.0)),
        reward_wall_penalty=float(g("reward_wall_penalty", 10.0)),
        reward_tolerance_margin=float(g("reward_tolerance_margin", 1.0)),
        reward_tolerance_terminal_bonus=float(g("reward_tolerance_terminal_bonus", 0.0)),
        reward_tolerance_wall_penalty=float(g("reward_tolerance_wall_penalty", 0.0)),
        render_mode="rgb_array" if enable_render else None,
        render_size=(render_h, render_w),
        camera=camera,
        recursive_scene_glob=bool(g("recursive_scene_glob", False)),
    )
    return env
