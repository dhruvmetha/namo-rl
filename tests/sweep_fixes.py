"""Sweep over Python-side model parameter tweaks to find the cleanest fix.

For each config:
  1. Load model, modify (armature, kv, forcerange) in-place
  2. Run a forward command from rest for 30 env-steps
  3. Measure: peak chassis vx, steady-state vx, settle time, peak slip
  4. Print summary table

Run: python tests/sweep_fixes.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from dataclasses import dataclass

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


@dataclass
class FixConfig:
    armature: float = 0.0
    kv: float = 0.75
    forcerange: float = 0.3
    label: str = ""


def settle(model, data, n=100):
    data.ctrl[:] = 0.0
    for _ in range(n):
        mujoco.mj_step(model, data)
    data.qvel[:] = 0.0


def run_config(xml_path: str, cfg: FixConfig, target_v: float = 0.025,
               v_accel: float = 0.05, frame_skip: int = 25, n_env_steps: int = 30,
               max_dwheel_per_sub: float = 0.02) -> dict:
    """Apply config, run a forward-from-rest sequence, return summary statistics."""
    model = mujoco.MjModel.from_xml_path(xml_path)

    L_act = model.actuator("left_wheel_drive").id
    R_act = model.actuator("right_wheel_drive").id
    lw_dof = int(model.jnt_dofadr[model.joint("left_wheel_joint").id])
    rw_dof = int(model.jnt_dofadr[model.joint("right_wheel_joint").id])

    # Apply parameter tweaks
    model.dof_armature[lw_dof] = cfg.armature
    model.dof_armature[rw_dof] = cfg.armature
    model.actuator_gainprm[L_act, 0] = cfg.kv
    model.actuator_gainprm[R_act, 0] = cfg.kv
    # Bias for velocity actuator is also -kv (the actuator type adds gainprm*ctrl + biasprm*qvel)
    model.actuator_biasprm[L_act, 2] = -cfg.kv
    model.actuator_biasprm[R_act, 2] = -cfg.kv
    model.actuator_forcerange[L_act] = [-cfg.forcerange, cfg.forcerange]
    model.actuator_forcerange[R_act] = [-cfg.forcerange, cfg.forcerange]

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    settle(model, data, 100)

    car_body = model.body("car").id
    jadr = int(model.body_jntadr[car_body])
    car_vadr = int(model.jnt_dofadr[jadr])
    car_qadr = int(model.jnt_qposadr[jadr])
    r = 0.015
    L_half = 0.0375

    dt_env = frame_skip * float(model.opt.timestep)
    dv_max = v_accel * dt_env

    v_cmd_log, vx_log, wheel_log, z_log, pitch_log = [], [], [], [], []
    v_cmd = 0.0
    for ti in range(n_env_steps):
        # Rate-limit chassis v
        target = min(target_v, v_cmd + dv_max)
        v_cmd = target
        # Kinematic split (w_cmd = 0 for forward only)
        target_wheel = v_cmd / r

        # Per-substep current-tracking ctrl
        for _ in range(frame_skip):
            cur_L = float(data.qvel[lw_dof])
            cur_R = float(data.qvel[rw_dof])
            data.ctrl[L_act] = cur_L + float(np.clip(target_wheel - cur_L, -max_dwheel_per_sub, max_dwheel_per_sub))
            data.ctrl[R_act] = cur_R + float(np.clip(target_wheel - cur_R, -max_dwheel_per_sub, max_dwheel_per_sub))
            mujoco.mj_step(model, data)

        vx = float(data.qvel[car_vadr])
        wL = float(data.qvel[lw_dof])
        v_kin = wL * r
        z = float(data.qpos[car_qadr + 2])
        qw, qx, qy, qz = (float(data.qpos[car_qadr + i]) for i in (3, 4, 5, 6))
        pitch = math.asin(max(-1.0, min(1.0, 2 * (qw * qy - qz * qx))))

        v_cmd_log.append(v_cmd)
        vx_log.append(vx)
        wheel_log.append(v_kin)
        z_log.append(z)
        pitch_log.append(pitch)

    vx_arr = np.array(vx_log)
    wheel_arr = np.array(wheel_log)
    pitch_arr = np.array(pitch_log)
    z_arr = np.array(z_log)

    # Steady-state: last 5 samples
    ss_vx = float(vx_arr[-5:].mean())
    ss_wheel = float(wheel_arr[-5:].mean())
    peak_vx = float(np.abs(vx_arr).max())
    peak_slip = float(np.abs(vx_arr - wheel_arr).max())
    peak_pitch_deg = float(np.degrees(np.abs(pitch_arr).max()))
    peak_z_dev_mm = float(np.abs(z_arr - z_arr[0]).max() * 1000)
    overshoot_ratio = peak_vx / ss_vx if abs(ss_vx) > 1e-6 else float("nan")

    return {
        "cfg": cfg,
        "ss_vx": ss_vx,
        "ss_wheel_kin": ss_wheel,
        "peak_vx": peak_vx,
        "overshoot_ratio": overshoot_ratio,
        "peak_slip": peak_slip,
        "peak_pitch_deg": peak_pitch_deg,
        "peak_z_dev_mm": peak_z_dev_mm,
    }


def main():
    xml = str(REPO_ROOT / "scenes" / "car" / "nav_env.xml")
    target_v = 0.025

    configs = [
        FixConfig(label="baseline (no change)"),
        FixConfig(armature=1e-4, label="arm=1e-4"),
        FixConfig(armature=5e-4, label="arm=5e-4"),
        FixConfig(armature=1e-3, label="arm=1e-3"),
        FixConfig(armature=5e-3, label="arm=5e-3"),
        FixConfig(kv=0.05, label="kv=0.05"),
        FixConfig(kv=0.01, label="kv=0.01"),
        FixConfig(forcerange=0.05, label="force=0.05"),
        FixConfig(forcerange=0.01, label="force=0.01"),
        FixConfig(armature=1e-3, kv=0.05, label="arm=1e-3 + kv=0.05"),
        FixConfig(armature=1e-3, forcerange=0.05, label="arm=1e-3 + force=0.05"),
        FixConfig(armature=5e-4, kv=0.1, forcerange=0.05, label="arm=5e-4 + kv=0.1 + force=0.05"),
    ]

    print(f"target_v = {target_v} m/s, expected steady-state chassis vx = {target_v}")
    print(f"{'config':<35} {'ss_vx':>8} {'peak_vx':>8} {'over_x':>7} {'peak_slip':>9} {'pitch°':>7} {'z_mm':>6}")
    print("-" * 90)
    for cfg in configs:
        r = run_config(xml, cfg, target_v=target_v)
        print(f"{cfg.label:<35} {r['ss_vx']:>+8.4f} {r['peak_vx']:>+8.4f} "
              f"{r['overshoot_ratio']:>7.2f} {r['peak_slip']:>9.4f} "
              f"{r['peak_pitch_deg']:>7.2f} {r['peak_z_dev_mm']:>6.3f}")


if __name__ == "__main__":
    main()
