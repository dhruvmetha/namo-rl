"""Entrypoint for training vanilla TD-MPC²-Square on the NAMO-RL car env.

Reuses the upstream `tdmpc_square` package for the agent, buffer, trainer, and
logger; only the env construction is overridden so we can point at NAMO-RL's
`DiffDriveCarEnv` (see `tdmpc_square_namo/envs/namo.py`).

Usage:
    python -m tdmpc_square_namo.train task=namo_car seed=1
"""
import os
import sys
from pathlib import Path

if sys.platform != "darwin":
    os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("LAZY_LEGACY_OP", "0")

import warnings
warnings.filterwarnings("ignore")

import gymnasium as gym  # noqa: F401  (matches upstream import order)
import hydra
import torch
from termcolor import colored

# Make the sibling `tdmpc_square` package importable when this file is run from
# the repo root (no install needed for either package).
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "tdmpc_square")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tdmpc_square.common.parser import parse_cfg
from tdmpc_square.common.seed import set_seed
from tdmpc_square.common.buffer import Buffer
from tdmpc_square.common.logger import Logger
from tdmpc_square.tdmpc_square import TDMPC2
from tdmpc_square.trainer.online_trainer import OnlineTrainer
from tdmpc_square.envs.wrappers.tensor import TensorWrapper

from tdmpc_square_namo.envs.namo import make_env as make_namo_env
from tdmpc_square_namo.video import LocalVideoRecorder

torch.backends.cudnn.benchmark = True


def _build_env(cfg):
    env = make_namo_env(cfg)
    env = TensorWrapper(env)
    # Wire obs/action dims into cfg the same way upstream's make_env does.
    try:  # Dict obs (unused here, but kept for parity)
        cfg.obs_shape = {k: v.shape for k, v in env.observation_space.spaces.items()}
    except Exception:
        cfg.obs_shape = {cfg.get("obs", "state"): env.observation_space.shape}
    cfg.action_dim = env.action_space.shape[0]
    cfg.episode_length = int(env.max_episode_steps)
    cfg.seed_steps = max(1000, 5 * cfg.episode_length)
    return env


@hydra.main(config_name="config", config_path=".", version_base="1.1")
def train(cfg):
    assert cfg.steps > 0, "Must train for at least 1 step."
    cfg = parse_cfg(cfg)
    set_seed(cfg.seed)
    print(colored("Work dir:", "yellow", attrs=["bold"]), cfg.work_dir)
    print("cfg:", cfg)

    # Capture the user's intent before Logger touches cfg.save_video — the
    # upstream Logger zeroes save_video when wandb is disabled (logger.py:128).
    want_video = bool(cfg.save_video)

    env = _build_env(cfg)
    # `OnlineTrainer` stores `oracle_env` but only reads it when
    # `cfg.eval_oracle_mpc=true`, which we leave off for this task — keep it None.
    oracle_env = None

    logger = Logger(cfg)
    if want_video:
        # Swap in a local mp4 recorder (writes to disk; also pushes to wandb when
        # wandb is enabled). The trainer accesses `logger.video` (property →
        # `logger._video`) for init/record/save.
        logger._video = LocalVideoRecorder(
            cfg,
            fps=int(getattr(cfg, "video_fps", 15)),
            wandb=getattr(logger, "_wandb", None),
        )
        cfg.save_video = True
        sink = "local mp4" + (" + wandb" if getattr(logger, "_wandb", None) else "")
        print(colored(f"Video recorder active ({sink}):", "cyan", attrs=["bold"]),
              str(Path(cfg.work_dir) / "eval_video"))

    trainer = OnlineTrainer(
        cfg=cfg,
        env=env,
        oracle_env=oracle_env,
        agent=TDMPC2(cfg),
        buffer=Buffer(cfg),
        logger=logger,
    )
    trainer.train()
    print("\nTraining completed successfully")


if __name__ == "__main__":
    train()
