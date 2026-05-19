#!/bin/bash
# F1+F3 fix — hop_3, GPU 0, max_episode_steps=750
set -u
cd "$(dirname "$0")/.."
export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=0
PYBIN=$(which python)
SCENES="$(cd ../external/namo-rl/scenes/car && pwd)"
LOGFILE="screen_logs/f1f3_hop3_seed1.log"
echo "[$(date)] starting f1f3 hop_3 run on $(hostname), GPU=$CUDA_VISIBLE_DEVICES, py=$PYBIN" | tee "$LOGFILE"
$PYBIN -u -m tdmpc_square_namo.train \
  task=namo_car seed=1 steps=1000000 eval_freq=25000 eval_episodes=5 \
  save_video=true save_agent=true eval_oracle_mpc=false model_size=5 \
  device='cuda:0' \
  exp_name='vanilla_namo_hop3_seed1_tol_gtol05_tb100_wp10_eps750' \
  namo.reward_style=tolerance namo.goal_position_tol=0.05 \
  namo.reward_tolerance_terminal_bonus=100.0 \
  namo.reward_tolerance_wall_penalty=10.0 \
  namo.max_episode_steps=750 \
  namo.scene_dir="$SCENES/hop_3/benchmark_3" namo.recursive_scene_glob=true \
  2>&1 | tee -a "$LOGFILE"
echo "[$(date)] f1f3 hop_3 run exited" | tee -a "$LOGFILE"
