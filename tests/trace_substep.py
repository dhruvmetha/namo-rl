"""Per-substep instrumentation of the velocity overshoot.

For a forward command starting from rest, log at every mj_step:
  - wheel ctrl, wheel actual ω
  - wheel angular acceleration
  - chassis x, vx_world, vz_world
  - chassis pitch, z
  - friction force at the wheel-floor contact (computed from solver-applied force)
  - number of active contacts

This isolates the exact substep when the chassis acquires its over-shoot velocity.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def settle(model, data, n=100):
    data.ctrl[:] = 0.0
    for _ in range(n):
        mujoco.mj_step(model, data)
    data.qvel[:] = 0.0


def contact_normal_force(data, body_ids: set[int], model: mujoco.MjModel) -> float:
    """Sum of normal-force magnitudes at contacts involving any body in body_ids."""
    total = 0.0
    if data.ncon == 0:
        return 0.0
    force = np.zeros(6, dtype=np.float64)
    for i in range(data.ncon):
        c = data.contact[i]
        b1 = int(model.geom_bodyid[c.geom1])
        b2 = int(model.geom_bodyid[c.geom2])
        if b1 in body_ids or b2 in body_ids:
            mujoco.mj_contactForce(model, data, i, force)
            total += abs(float(force[0]))  # normal component
    return total


def trace_substeps(xml_path: str, ctrl_value: float, n_substeps: int = 80,
                   max_dwheel: float = 0.02):
    """Apply a constant `target wheel velocity` to both wheels (with current-tracking
    ctrl logic) and print state at every substep."""
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    L_act = model.actuator("left_wheel_drive").id
    R_act = model.actuator("right_wheel_drive").id
    car_qadr = int(model.jnt_qposadr[model.joint("car_freejoint").id]) if "car_freejoint" in [model.joint(i).name for i in range(model.njnt)] else 0
    # Robust: find car body, its first joint = freejoint
    car_body = model.body("car").id
    jadr = int(model.body_jntadr[car_body])
    car_qadr = int(model.jnt_qposadr[jadr])
    car_vadr = int(model.jnt_dofadr[jadr])
    lw_dof = int(model.jnt_dofadr[model.joint("left_wheel_joint").id])
    rw_dof = int(model.jnt_dofadr[model.joint("right_wheel_joint").id])

    car_bodies = set()
    # find descendants of "car"
    stack = [car_body]
    while stack:
        b = stack.pop()
        car_bodies.add(b)
        for i in range(model.nbody):
            if int(model.body_parentid[i]) == b:
                stack.append(i)

    mujoco.mj_forward(model, data)
    settle(model, data, 100)

    target = float(ctrl_value)
    print(f"target wheel velocity: {target} rad/s, max_dwheel/substep: {max_dwheel}")
    print(f"actuator kv = {float(model.actuator_gainprm[L_act, 0]):.3f}  "
          f"forcerange = ±{float(model.actuator_forcerange[L_act, 1]):.3f}  "
          f"wheel J = {float(model.dof_M0[lw_dof]):.2e}  "
          f"armature = {float(model.dof_armature[lw_dof]):.2e}")
    print()
    print(f"{'sub':>4} {'ctrlL':>7} {'wL_act':>8} {'a_wh':>9} {'tau':>8} "
          f"{'vx_w':>8} {'ax_w':>9} {'z':>8} {'pitch':>7} {'F_norm':>7} {'ncon':>4}")

    prev_wL = 0.0
    prev_vx = 0.0
    prev_z = float(data.qpos[car_qadr + 2])
    for k in range(n_substeps):
        # current-tracking ctrl
        cur_L = float(data.qvel[lw_dof])
        cur_R = float(data.qvel[rw_dof])
        data.ctrl[L_act] = cur_L + float(np.clip(target - cur_L, -max_dwheel, max_dwheel))
        data.ctrl[R_act] = cur_R + float(np.clip(target - cur_R, -max_dwheel, max_dwheel))
        ctrlL = float(data.ctrl[L_act])

        mujoco.mj_step(model, data)

        wL = float(data.qvel[lw_dof])
        alpha = (wL - prev_wL) / float(model.opt.timestep)
        # Implied torque using the actuator equation, clipped
        kv = float(model.actuator_gainprm[L_act, 0])
        force_lim = float(model.actuator_forcerange[L_act, 1])
        tau_unclamped = kv * (ctrlL - prev_wL)
        tau = max(-force_lim, min(force_lim, tau_unclamped))
        vx = float(data.qvel[car_vadr])
        ax = (vx - prev_vx) / float(model.opt.timestep)
        z = float(data.qpos[car_qadr + 2])
        qw = float(data.qpos[car_qadr + 3])
        qx = float(data.qpos[car_qadr + 4])
        qy = float(data.qpos[car_qadr + 5])
        qz = float(data.qpos[car_qadr + 6])
        pitch = math.asin(max(-1.0, min(1.0, 2 * (qw * qy - qz * qx))))
        F_norm = contact_normal_force(data, car_bodies, model)

        print(f"{k:>4d} {ctrlL:>+7.4f} {wL:>+8.4f} {alpha:>+9.1f} {tau:>+8.4f} "
              f"{vx:>+8.4f} {ax:>+9.2f} {z:>+8.5f} {pitch:>+7.4f} {F_norm:>7.3f} {int(data.ncon):>4d}")

        prev_wL = wL
        prev_vx = vx
        prev_z = z


if __name__ == "__main__":
    xml = str(REPO_ROOT / "scenes" / "car" / "nav_env.xml")
    trace_substeps(xml, ctrl_value=1.667, n_substeps=60)  # ω corresponding to v=0.025 m/s
