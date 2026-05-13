from __future__ import annotations

import mujoco


def robot_wall_contact(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    robot_body_ids: set[int],
    wall_body_ids: set[int],
) -> bool:
    """Return True if any active contact pairs a robot body with a wall body.

    MuJoCo stores active contacts in `data.contact[:data.ncon]`. Each contact
    carries `geom1` / `geom2`; we map those geoms back to their parent bodies
    via `model.geom_bodyid` and check for any robot-wall pair.
    """
    for i in range(data.ncon):
        c = data.contact[i]
        b1 = int(model.geom_bodyid[c.geom1])
        b2 = int(model.geom_bodyid[c.geom2])
        if (b1 in robot_body_ids and b2 in wall_body_ids) or (
            b2 in robot_body_ids and b1 in wall_body_ids
        ):
            return True
    return False
