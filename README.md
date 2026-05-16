# namo-rl

A pure-Python `gymnasium.Env` for kinodynamic RL on the NAMO diff-drive car.

The policy outputs a chassis-velocity command `(v, ω)`. The env clips and
rate-limits it, splits it into wheel-velocity setpoints with the standard
diff-drive kinematics, and writes those into MuJoCo's `<velocity>` actuators.
No wavefront, no push controller, no nav state machine. Walls terminate the
episode; movable obstacles can be bumped or shoved freely.

## Install

```bash
pip install -e .
```

## Quick start

```python
from namo_rl import DiffDriveCarEnv, EnvConfig

cfg = EnvConfig.from_yaml("configs/car.yaml")
env = DiffDriveCarEnv(cfg)
obs, info = env.reset(seed=0)
for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
```

## Run smoke test

```bash
python tests/test_smoke.py
```

## Render demo videos

Offscreen rendering needs an OpenGL context. On a node with a GPU + EGL:

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python tests/render_videos.py
```

Writes `tests/videos/<policy>__<scene>.mp4` for four policies (random, constant
forward, spin-in-place, crude greedy-to-goal P controller) on every scene in
`scenes/car/`.

## Layout

```
namo_rl/
  env.py        # DiffDriveCarEnv(gymnasium.Env)
  contacts.py   # robot-vs-wall contact detection
  config.py     # YAML-backed dataclass config
configs/
  car.yaml      # frame_skip, action bounds, tolerances, scene_dir, ...
scenes/car/     # MuJoCo scene XMLs (fully self-contained, copied from NAMO)
tests/
  test_smoke.py # 5-episode random-policy rollout
```

## Control loop

| Quantity              | Value (default `configs/car.yaml`)              |
| --------------------- | ----------------------------------------------- |
| MuJoCo timestep       | 0.002 s                                         |
| `frame_skip`          | 25 substeps per `env.step()`                    |
| Control period        | 0.050 s (**20 Hz**)                             |
| `max_episode_steps`   | 500 (≈ 25 s of sim time)                        |

One `env.step(action)` clips and rate-limits the command, runs `frame_skip`
MuJoCo substeps, then returns the next obs.

## Action space

`Box(low=[-v_max, -w_max], high=[+v_max, +w_max], shape=(2,), dtype=float32)` —
chassis-frame `(v, ω)` setpoint in m/s and rad/s.

Bounds and rate limits come from `EnvConfig.action`:

| Field           | YAML default | Units    | Meaning                                              |
| --------------- | ------------ | -------- | ---------------------------------------------------- |
| `v_max`         | 0.10         | m/s      | linear-velocity command saturation                   |
| `w_max`         | 2.4          | rad/s    | angular-velocity command saturation                  |
| `v_accel_max`   | 0.20         | m/s²     | max change in commanded `v` per `env.step`           |
| `w_accel_max`   | 4.0          | rad/s²   | max change in commanded `ω` per `env.step`           |
| `wheel_radius`  | 0.015        | m        | wheel radius `r` for the kinematic split             |
| `wheelbase`     | 0.075        | m        | lateral wheel separation `L`                         |

After clipping and rate-limiting `(v, ω)`, the env writes wheel-velocity
setpoints to the actuators:

```
ω_left  = (v − ω · L/2) / r
ω_right = (v + ω · L/2) / r
```

The rate-limit state `(v_prev, ω_prev)` is reset to zero on every `reset()`.

## Observation space

`Dict` with five entries; all leaves are `float32` except `mask`.

| Key           | Shape         | Bounds       | Contents                                                                                       |
| ------------- | ------------- | ------------ | ---------------------------------------------------------------------------------------------- |
| `robot`       | `(7,)`        | unbounded    | `(x, y, sinθ, cosθ, vx_body, vy_body, ω)` — world pose, body-frame linear vel, yaw rate         |
| `goal`        | `(3,)`        | unbounded    | `(Δx_body, Δy_body, Δθ)` — goal pose in robot frame, `Δθ` wrapped to `[-π, π]`                  |
| `movables`    | `(N_max, 3)`  | unbounded    | per-movable world-frame `(x, y, θ)`, zero-padded to `N_max`                                    |
| `mask`        | `(N_max,)`    | `{0, 1}`     | int8; `1` where a movable exists at that row of `movables`                                     |
| `last_action` | `(2,)`        | action-space | the post-clip / post-rate-limit `(v_prev, ω_prev)` that was actually commanded last step       |

`N_max = EnvConfig.obs.max_movables` (default 16). Pose is read from each
movable body's free joint; heading comes from the quaternion.

## Termination

- `terminated=True` if any robot body (chassis or wheel) contacts a wall —
  `info['termination'] = 'wall'`.
- `terminated=True` if the robot is within `goal_tolerance.position` (and
  `goal_tolerance.heading`) of the scene's `<site name="goal">` —
  `info['termination'] = 'goal'`.
- `truncated=True` if `max_episode_steps` is exceeded —
  `info['termination'] = 'truncated'`.

YAML defaults: `position = 0.05 m`, `heading = π` (i.e. heading is ignored,
XY-only goal).

`info` also carries `dist_to_goal` (meters) every step, and on `reset()`:
`xml_path`, `goal_xy`, `n_movables`.

## Reward

**Currently stubbed at `0.0`.** The literal `return obs, 0.0, terminated,
truncated, info` lives at the bottom of `DiffDriveCarEnv.step` in
[`namo_rl/env.py`](namo_rl/env.py) (look for the `return` at the end of
`step`). Two ways to wire in a real reward:

1. **In-place** — replace the `0.0` in `step()` with a function of `obs`,
   `info['dist_to_goal']`, `info.get('termination')`, and the action. Keep the
   shaping logic in a helper method (e.g. `_reward(...)`) so it stays
   swappable.
2. **As a wrapper** — leave the env alone and add a `gymnasium.Wrapper` that
   recomputes `reward` from `(obs, action, info)`. Better for ablations and
   keeps `env.py` task-agnostic.

Useful signals already available without extra bookkeeping:
`info['dist_to_goal']`, the `goal` observation (Δ to goal in body frame),
`info['termination']` ∈ `{'wall', 'goal', 'truncated'}`, and `last_action` for
action-rate penalties.

## Scenes

`EnvConfig.scene_dir` (default `scenes/car`) is globbed **non-recursively**
for `*.xml` — `reset()` picks one uniformly at random unless
`reset(options={"xml_path": "..."})` is passed. Nested benchmark suites under
`scenes/car/hop_*/...` are *not* picked up automatically; point `scene_dir`
at the specific subdirectory you want to train on.

Each scene XML must define:
- a body named `car` (the chassis subtree, with `left_wheel_joint` /
  `right_wheel_joint` hinges and `left_wheel_drive` / `right_wheel_drive`
  velocity actuators),
- a `<site name="goal">` at the goal XY,
- zero or more bodies whose names contain `wall` (terminal on contact) or
  `movable` (free to be pushed; exposed in the obs).
