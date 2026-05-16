"""Print SE(2) state + derivatives + irregularity indicators for each policy.

For each policy run on nav_env.xml, log per env-step:
  q  : (x, y, theta)
  q' : (vx_world, vy_world, omega_z)
  irregularity:  z (vertical excursion), pitch, roll (should all be ~0 on flat floor)
  wheel slip   : wheel_speed * r vs measured chassis forward speed
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from namo_rl import DiffDriveCarEnv, EnvConfig  # noqa: E402


def quat_to_rpy(qw, qx, qy, qz):
    roll = math.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (qw * qy - qz * qx))))
    yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    return roll, pitch, yaw


def trace(env: DiffDriveCarEnv, action_fn, n_steps: int, label: str, xml: str):
    obs, info = env.reset(seed=0, options={"xml_path": xml})
    m = env._scene.model
    d = env._data
    qadr = env._scene.car_qpos_adr
    vadr = env._scene.car_qvel_adr
    lw_vadr = int(m.jnt_dofadr[m.joint("left_wheel_joint").id])
    rw_vadr = int(m.jnt_dofadr[m.joint("right_wheel_joint").id])
    r = 0.015

    print(f"\n=== {label} ({xml.split('/')[-1]}) ===")
    print(
        f"{'t':>3} | {'x':>7} {'y':>7} {'th':>7} | "
        f"{'vx_w':>7} {'vy_w':>7} {'wz':>7} | "
        f"{'z':>7} {'roll':>6} {'pitch':>6} | "
        f"{'v_kin':>7} {'vx_b':>7} {'slip':>6}"
    )
    for t in range(n_steps):
        a = action_fn(obs, env.action_space)
        obs, _r, term, trunc, info = env.step(a)
        qp, qv = d.qpos, d.qvel
        x = float(qp[qadr]); y = float(qp[qadr + 1]); z = float(qp[qadr + 2])
        qw, qx, qy, qz = (
            float(qp[qadr + 3]), float(qp[qadr + 4]),
            float(qp[qadr + 5]), float(qp[qadr + 6]),
        )
        roll, pitch, yaw = quat_to_rpy(qw, qx, qy, qz)
        vx_w = float(qv[vadr]); vy_w = float(qv[vadr + 1])
        wz = float(qv[vadr + 5])  # body-z; ≈ world-yaw when chassis ~level
        wL, wR = float(qv[lw_vadr]), float(qv[rw_vadr])
        v_kin = 0.5 * r * (wL + wR)
        # forward chassis speed in body frame
        c, s = math.cos(yaw), math.sin(yaw)
        vx_b = vx_w * c + vy_w * s
        slip = vx_b - v_kin
        print(
            f"{t:>3} | {x:>+7.4f} {y:>+7.4f} {yaw:>+7.3f} | "
            f"{vx_w:>+7.4f} {vy_w:>+7.4f} {wz:>+7.3f} | "
            f"{z:>+7.4f} {roll:>+6.3f} {pitch:>+6.3f} | "
            f"{v_kin:>+7.4f} {vx_b:>+7.4f} {slip:>+6.3f}"
        )
        if term or trunc:
            break


def main():
    cfg = EnvConfig.from_yaml(REPO_ROOT / "configs" / "car.yaml")
    env = DiffDriveCarEnv(cfg)
    v_max = cfg.action.v_max
    w_max = cfg.action.w_max
    xml = str(REPO_ROOT / "scenes" / "car" / "nav_env.xml")
    rng = np.random.default_rng(7)

    policies = {
        "forward": lambda obs, sp: np.array([v_max, 0.0], dtype=np.float32),
        "backward": lambda obs, sp: np.array([-v_max, 0.0], dtype=np.float32),
        "spin": lambda obs, sp: np.array([0.0, w_max], dtype=np.float32),
        "arc": lambda obs, sp: np.array([0.5 * v_max, 0.5 * w_max], dtype=np.float32),
        "random": lambda obs, sp: sp.sample(),
    }
    n_steps = 30
    for name, fn in policies.items():
        trace(env, fn, n_steps, name, xml)


if __name__ == "__main__":
    main()
