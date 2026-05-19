"""Smoke test: build the NAMO env, run a few random steps, confirm dims + reward shape."""
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
_REPO_ROOT = os.path.abspath(os.path.join(_PKG_ROOT, ".."))
for _p in (_PKG_ROOT, _REPO_ROOT, os.path.join(_REPO_ROOT, "tdmpc_square")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
from types import SimpleNamespace

from tdmpc_square_namo.envs.namo import make_env


def _cfg():
    return SimpleNamespace(task="namo_car", seed=0, namo=None)


def main():
    env = make_env(_cfg())
    print("obs_space:", env.observation_space)
    print("act_space:", env.action_space)
    print("max_episode_steps:", env.max_episode_steps)

    obs, info = env.reset(seed=0)
    print("reset obs:", obs.shape, obs.dtype, "info:", {k: info[k] for k in info if k != "goal_xy"})

    total_r = 0.0
    n_steps = 50
    for t in range(n_steps):
        a = env.action_space.sample()
        obs, r, term, trunc, info = env.step(a)
        total_r += r
        if term or trunc:
            print(f"  terminated at t={t} term={term} trunc={trunc} info_term={info.get('termination')}")
            obs, info = env.reset()
    print(f"ran {n_steps} steps, total_reward={total_r:.4f}, last_obs_shape={obs.shape}")
    print("SMOKE OK")


if __name__ == "__main__":
    main()
