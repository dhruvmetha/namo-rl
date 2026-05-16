"""Generate top-down demo videos for DiffDriveCarEnv (chassis-velocity action space).

Policy outputs are (v, omega) in (m/s, rad/s). The env internally rate-limits
and maps them to wheel velocity setpoints via the diff-drive kinematic split.

Run with:
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python tests/render_videos.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from namo_rl import DiffDriveCarEnv, EnvConfig  # noqa: E402

OUT_DIR = REPO_ROOT / "tests" / "videos"
FPS = None             # None → derive from config; integer to force a playback speed
RENDER_SIZE = (512, 512)


def policy_random(obs, rng, low, high):
    return rng.uniform(low, high, size=2).astype(np.float32)


def policy_forward(obs, rng, low, high):
    return np.array([high[0], 0.0], dtype=np.float32)


def policy_backward(obs, rng, low, high):
    return np.array([low[0], 0.0], dtype=np.float32)


def policy_spin(obs, rng, low, high):
    return np.array([0.0, high[1]], dtype=np.float32)


def policy_arc(obs, rng, low, high):
    """Half forward, third yaw — smooth circular motion."""
    return np.array([0.5 * high[0], 0.33 * high[1]], dtype=np.float32)


def policy_greedy_to_goal(obs, rng, low, high):
    """Level-1 controller: forward modulated by cos(heading_err), turn proportional.
    Never demands both max-forward and max-turn simultaneously, so wheel
    commands stay sub-saturation.
    """
    dx_b, dy_b, _ = obs["goal"]
    heading_err = math.atan2(float(dy_b), float(dx_b))
    forward = high[0] * max(0.0, math.cos(heading_err))
    turn = high[1] * float(np.tanh(heading_err))
    return np.clip(np.array([forward, turn], dtype=np.float32), low, high)


POLICIES = {
    "random": policy_random,
    "forward": policy_forward,
    "backward": policy_backward,
    "spin": policy_spin,
    "arc": policy_arc,
    "greedy_to_goal": policy_greedy_to_goal,
}


def run_episode(env: DiffDriveCarEnv, policy, rng, max_steps: int, xml_path: Path):
    obs, info = env.reset(seed=int(rng.integers(0, 1 << 31)), options={"xml_path": str(xml_path)})
    low = env.action_space.low
    high = env.action_space.high
    frames = [env.render()]
    cause = "running"
    for _ in range(max_steps):
        action = policy(obs, rng, low, high)
        obs, _r, term, trunc, info = env.step(action)
        frames.append(env.render())
        if term or trunc:
            cause = info.get("termination", "unknown")
            break
    return frames, cause


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = EnvConfig.from_yaml(REPO_ROOT / "configs" / "car.yaml")
    env = DiffDriveCarEnv(
        cfg, render_mode="rgb_array", render_size=RENDER_SIZE, camera="top_down"
    )
    fps = FPS if FPS is not None else int(round(1.0 / (cfg.frame_skip * 0.002)))

    scenes = sorted((REPO_ROOT / "scenes" / "car").glob("*.xml"))
    print(f"scenes: {[s.name for s in scenes]}")
    print(f"action_space (v, omega): {env.action_space}")

    rng = np.random.default_rng(7)
    cap = 200
    for policy_name, policy in POLICIES.items():
        for xml in scenes:
            frames, cause = run_episode(env, policy, rng, max_steps=cap, xml_path=xml)
            out = OUT_DIR / f"{policy_name}__{xml.stem}.mp4"
            iio.imwrite(str(out), np.stack(frames, axis=0), fps=fps, codec="libx264")
            print(f"wrote {out.relative_to(REPO_ROOT)}  [{len(frames)} frames, cause={cause}]")

    env.close()
    print(f"\nall videos under {OUT_DIR.relative_to(REPO_ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
