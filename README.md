# namo-rl

A pure-Python `gymnasium.Env` for kinodynamic RL on the NAMO diff-drive car.

The policy writes wheel-velocity setpoints `(ωL, ωR)` directly into MuJoCo's `<velocity>` actuators. No wavefront, no push controller, no nav state machine. Walls terminate the episode; movable obstacles can be bumped or shoved freely.

## Install

```bash
pip install -e .
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
  car.yaml      # frame_skip, tolerances, scene_dir, ...
scenes/car/     # MuJoCo scene XMLs (fully self-contained, copied from NAMO)
tests/
  test_smoke.py # 5-episode random-policy rollout
```

## Action / observation

- **action**: `Box([-25, 25]², float32)` — left and right wheel angular velocity (rad/s).
- **observation** (Dict):
  - `robot`: `(x, y, sinθ, cosθ, vx_body, vy_body, ω)` (7,)
  - `goal`: `(Δx, Δy, Δθ)` in robot frame (3,)
  - `movables`: `(N_max, 3)` world-frame `(x, y, θ)` per movable obstacle, zero-padded
  - `mask`: `(N_max,)` int8, 1 where a movable exists

## Termination

- `terminated=True` if any robot body (chassis or wheel) contacts a wall — `info['termination']='wall'`.
- `terminated=True` if the robot is within `goal_tolerance.position` (and `heading`) of the scene's `<site name="goal">` — `info['termination']='goal'`.
- `truncated=True` if `max_episode_steps` is exceeded.

## Reward

Stubbed at `0.0`. Wrap the env or modify `env.py` once the policy is learning the dynamics.
