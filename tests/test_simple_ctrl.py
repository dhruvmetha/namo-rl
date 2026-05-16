"""Test: with armature applied, is the per-substep current-tracking ctrl still needed?
Compare:
  A. Per-substep current-tracking (current env.py)
  B. Set ctrl once per env-step (simpler)

Both with armature=5e-4 applied. Same forward-from-rest sequence.
"""

import math, sys
from pathlib import Path
import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def run(use_substep_tracking: bool):
    xml = str(REPO_ROOT / "scenes" / "car" / "nav_env.xml")
    model = mujoco.MjModel.from_xml_path(xml)
    L_act = model.actuator("left_wheel_drive").id
    R_act = model.actuator("right_wheel_drive").id
    lw_dof = int(model.jnt_dofadr[model.joint("left_wheel_joint").id])
    rw_dof = int(model.jnt_dofadr[model.joint("right_wheel_joint").id])
    model.dof_armature[lw_dof] = 5e-4
    model.dof_armature[rw_dof] = 5e-4

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    data.ctrl[:] = 0
    for _ in range(100):
        mujoco.mj_step(model, data)
    data.qvel[:] = 0.0

    car_body = model.body("car").id
    jadr = int(model.body_jntadr[car_body])
    car_vadr = int(model.jnt_dofadr[jadr])
    r = 0.015
    frame_skip = 25
    v_accel = 0.05
    target_v = 0.025

    dt_env = frame_skip * float(model.opt.timestep)
    dv_max = v_accel * dt_env

    v_cmd = 0.0
    log = []
    for ti in range(30):
        v_cmd = min(target_v, v_cmd + dv_max)
        target_wheel = v_cmd / r

        if use_substep_tracking:
            max_dwheel = 0.02
            for _ in range(frame_skip):
                cur_L = float(data.qvel[lw_dof])
                cur_R = float(data.qvel[rw_dof])
                data.ctrl[L_act] = cur_L + float(np.clip(target_wheel - cur_L, -max_dwheel, max_dwheel))
                data.ctrl[R_act] = cur_R + float(np.clip(target_wheel - cur_R, -max_dwheel, max_dwheel))
                mujoco.mj_step(model, data)
        else:
            data.ctrl[L_act] = target_wheel
            data.ctrl[R_act] = target_wheel
            for _ in range(frame_skip):
                mujoco.mj_step(model, data)

        vx = float(data.qvel[car_vadr])
        wL = float(data.qvel[lw_dof])
        log.append((vx, wL * r))
    return log


print("forward ramp 0→0.025 m/s, armature=5e-4")
print(f"{'t':>3} | {'A vx_b':>8} {'A v_kin':>8} | {'B vx_b':>8} {'B v_kin':>8}  (A=current-track, B=simple)")
A = run(use_substep_tracking=True)
B = run(use_substep_tracking=False)
for i, ((avx, avk), (bvx, bvk)) in enumerate(zip(A, B)):
    print(f"{i:>3} | {avx:>+8.4f} {avk:>+8.4f} | {bvx:>+8.4f} {bvk:>+8.4f}")
