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
    """Chassis-velocity command envelope and rate limits.

    The env's action space is `(v, ω)`. `v_accel_max` and `w_accel_max`
    bound how fast the *commanded* velocities can change per env.step,
    which is what keeps the wheels out of impulsive transients.
    """

    v_max: float = 0.025          # m/s
    w_max: float = 0.6            # rad/s
    v_accel_max: float = 0.05     # m/s^2
    w_accel_max: float = 1.0      # rad/s^2
    wheel_radius: float = 0.015   # m
    wheelbase: float = 0.075      # m (lateral distance between wheels)


@dataclass
class PhysicsTune:
    """Python-side model overrides applied after MjModel load.

    The bare car XML has zero rotor inertia on the wheel hinges, which makes
    the velocity actuator's response time (~8 μs) much faster than the sim
    timestep (~2 ms). That causes a chassis over-acceleration transient
    whenever wheel ctrl changes. Setting a realistic `armature` (reflected
    motor rotor inertia) puts the actuator time constant on the timestep
    scale, matching published MuJoCo Menagerie wheeled robots (Stretch 3
    uses armature ≈ 500× bare J; Robot Soccer Kit uses ~1300×). We default
    to ≈ 80× bare J, which is enough to eliminate the overshoot.
    """
    wheel_armature: float = 5e-4   # kg·m²  (= reflected rotor inertia)
    wheel_damping: float = 0.0     # joint damping (Nm·s/rad)


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
    physics: PhysicsTune = field(default_factory=PhysicsTune)
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
            physics=PhysicsTune(
                **{k: float(v) for k, v in data.get("physics", {}).items()}
            ),
            seed=data.get("seed"),
        )
