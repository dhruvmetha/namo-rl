"""Smoke test: instantiate DiffDriveCarEnv, run 5 random-policy episodes,
print termination counts. Pass criterion: no exceptions, all episodes terminate
within max_episode_steps + 1 calls.

Run from repo root:
    python tests/test_smoke.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from namo_rl import DiffDriveCarEnv, EnvConfig  # noqa: E402


def main() -> int:
    cfg_path = REPO_ROOT / "configs" / "car.yaml"
    cfg = EnvConfig.from_yaml(cfg_path)

    env = DiffDriveCarEnv(cfg)

    assert env.action_space.shape == (2,), env.action_space
    assert set(env.observation_space.spaces.keys()) == {
        "robot", "goal", "movables", "mask", "last_action"
    }

    rng = np.random.default_rng(42)
    causes: Counter[str] = Counter()
    step_budget = cfg.max_episode_steps + 1

    for ep in range(5):
        obs, info = env.reset(seed=int(rng.integers(0, 1 << 31)))
        n_steps = 0
        terminated = truncated = False
        last_info: dict = {}
        while not (terminated or truncated):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, last_info = env.step(action)
            n_steps += 1
            if n_steps > step_budget:
                raise RuntimeError(
                    f"episode {ep} ran past step budget {step_budget} (no termination signal)"
                )
        cause = last_info.get("termination", "unknown")
        causes[cause] += 1
        print(
            f"ep {ep}: xml={Path(info['xml_path']).name} steps={n_steps} "
            f"cause={cause} dist={last_info.get('dist_to_goal', float('nan')):.3f}"
        )

    env.close()
    print(f"\nsummary: {dict(causes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
