# tdmpc_square_namo

Vanilla TD-MPC²-Square training on the [NAMO-RL](../external/namo-rl) diff-drive
car environment. Reuses the existing `tdmpc_square` package (agent, buffer,
trainer, logger, planner) and only adds an env adapter + Hydra config + a thin
training entrypoint.

## Layout

```
tdmpc_square_namo/
  setup.py
  tdmpc_square_namo/
    __init__.py
    config.yaml          # Hydra config (extends tdmpc_square's defaults)
    train.py             # entrypoint — wires NAMO env into tdmpc_square's OnlineTrainer
    envs/
      __init__.py
      namo.py            # DiffDriveCarEnv adapter (flat Box obs + dense reward)
  tests/
    smoke_env.py         # 50-step random-policy rollout, dim sanity
```

## Observation, action, reward

- **Observation** (76-D `Box`): flattened concatenation of NAMO's dict obs —
  `robot(7)` + `goal(3)` + `movables(16×3)` + `mask(16)` + `last_action(2)`.
- **Action** (2-D `Box`): chassis-frame `(v, ω)` setpoint with the same bounds
  the upstream env exposes (`v_max=0.1 m/s`, `w_max=2.4 rad/s` by default).
- **Reward**: NAMO returns 0.0; we wrap with a dense shaping reward inside
  `NamoCarTDMPCEnv`. Two schemes are selectable via `namo.reward_style`:

  - `"additive"` (default — MyoSuite-style, unbounded mixed-sign):
    - `+progress_coef · Δdist_to_goal` (closer = positive)
    - `−action_rate_coef · ‖Δa_norm‖²`
    - `+goal_bonus` on `termination="goal"` (sparse terminal)
    - `−wall_penalty` on `termination="wall"`
    - `−step_penalty` per step (off by default)

  - `"tolerance"` (DMControl/HumanoidBench-aligned, bounded [0, 1]/step):
    - `r = near_target × small_ctrl` where both are
      `dm_control.utils.rewards.tolerance(...)` outputs.
    - `near_target`: 1.0 inside the goal-tolerance ball, ramps down over
      `reward_tolerance_margin` metres.
    - `small_ctrl`: per-component action magnitude tolerance, rescaled to
      `[0.8, 1.0]` like DMControl point_mass.
    - Optional `reward_tolerance_terminal_bonus` / `reward_tolerance_wall_penalty`
      for explicit terminal events (off by default — matches the pure
      DMControl/HB form).

  Coefficients live in `config.yaml` under `namo.reward_*`. To A/B test:
  `python -m tdmpc_square_namo.train ... namo.reward_style=tolerance`.

## Smoke test

```bash
conda activate tdmpc-square
python tdmpc_square_namo/tests/smoke_env.py
```

Expected: `obs_space: Box(...,(76,)..)`, `act_space: Box(...,(2,)..)`, prints
`SMOKE OK` after 50 random steps.

## Train (vanilla TD-MPC²-Square)

```bash
conda activate tdmpc-square
# from repo root
python -m tdmpc_square_namo.train \
    task=namo_car seed=1 \
    steps=1000000 eval_freq=25000 eval_episodes=5 \
    model_size=5 device=cuda:0 \
    save_video=false eval_oracle_mpc=false \
    exp_name=vanilla_tdmpc_square_namo_seed1
```

Logs land under `tdmpc_square_namo/logs/namo_car/<seed>/<exp_name>/`.

## Switching scene suites

The default scene set is `external/namo-rl/scenes/car/` (the `nav_env*.xml`
scenes). To train on the harder hop benchmarks, override `namo.scene_dir`:

```bash
python -m tdmpc_square_namo.train \
    task=namo_car \
    namo.scene_dir=external/namo-rl/scenes/car/hop_1/benchmark_3 \
    ...
```

Relative paths are resolved against `external/namo-rl/`.

## Eval videos

`save_video=true` (the default) writes one mp4 per eval pass to
`<work_dir>/eval_video/results_video_<step>.mp4`. The recorder lives in
`tdmpc_square_namo/video.py` (`LocalVideoRecorder`) — a wandb-free drop-in for
the upstream `VideoRecorder` that uses `imageio-ffmpeg`. The camera is the
top-down fit-to-walls view; resolution defaults to 320×240, fps 15. Override
with `namo.render_height=…`, `namo.render_width=…`, `namo.render_camera=…`, or
`video_fps=…`.

## Notes

- `tdmpc_square` is imported directly from its source folder via `sys.path`
  fixups in `train.py`; no install needed for either package.
- All TD-MPC-Square extensions (frontier curriculum, CMA-MPPI, multistep TD,
  CVaR, etc.) are **disabled** in `config.yaml` to give a clean "vanilla"
  baseline. Flip individual flags to enable.
- `oracle_env` is set to `None` since this task has no analytic oracle planner;
  `eval_oracle_mpc` is off in the default config.
