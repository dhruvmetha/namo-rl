from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class GoalTolerance:
    position: float = 0.05
    heading: float = 3.1416


@dataclass
class ActionLimits:
    wheel_max: float = 25.0


@dataclass
class ObsConfig:
    max_movables: int = 16


@dataclass
class EnvConfig:
    scene_dir: str = "scenes/car"
    frame_skip: int = 5
    max_episode_steps: int = 500
    goal_tolerance: GoalTolerance = field(default_factory=GoalTolerance)
    action: ActionLimits = field(default_factory=ActionLimits)
    obs: ObsConfig = field(default_factory=ObsConfig)
    seed: int | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "EnvConfig":
        data = yaml.safe_load(Path(path).read_text()) or {}
        return cls(
            scene_dir=data.get("scene_dir", cls.scene_dir),
            frame_skip=int(data.get("frame_skip", cls.frame_skip)),
            max_episode_steps=int(data.get("max_episode_steps", cls.max_episode_steps)),
            goal_tolerance=GoalTolerance(**data.get("goal_tolerance", {})),
            action=ActionLimits(**data.get("action", {})),
            obs=ObsConfig(**data.get("obs", {})),
            seed=data.get("seed"),
        )
