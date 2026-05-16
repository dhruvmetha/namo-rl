from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium.spaces import Box, Dict as DictSpace

from namo_rl.config import EnvConfig
from namo_rl.contacts import robot_wall_contact


@dataclass
class _SceneIndex:
    """Per-XML cached ids and qpos addresses. Recomputed on each reset()."""

    model: mujoco.MjModel
    left_act: int
    right_act: int
    left_wheel_dof: int                    # qvel index for left wheel hinge
    right_wheel_dof: int                   # qvel index for right wheel hinge
    car_qpos_adr: int
    car_qvel_adr: int
    robot_body_ids: frozenset[int]
    wall_body_ids: frozenset[int]
    movable_body_ids: tuple[int, ...]      # canonical order
    movable_qpos_adrs: tuple[int, ...]     # parallel to movable_body_ids
    goal_xy: np.ndarray                    # shape (2,)
    goal_theta: float                      # 0.0 unless overridden


def _find_descendants(model: mujoco.MjModel, root_id: int) -> set[int]:
    """All body ids in the subtree rooted at `root_id` (inclusive). Relies on
    MuJoCo's topological body ordering: parents always come before children."""
    desc = {root_id}
    for i in range(root_id + 1, model.nbody):
        if int(model.body_parentid[i]) in desc:
            desc.add(i)
    return desc


def _body_freejoint_qpos_adr(model: mujoco.MjModel, body_id: int) -> int:
    """Return the qpos address of the freejoint on `body_id`. Errors if missing."""
    j_adr = int(model.body_jntadr[body_id])
    if j_adr < 0:
        raise ValueError(f"body id {body_id} ({model.body(body_id).name}) has no joint")
    if int(model.jnt_type[j_adr]) != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError(
            f"body {model.body(body_id).name} first joint is not free "
            f"(type={int(model.jnt_type[j_adr])})"
        )
    return int(model.jnt_qposadr[j_adr])


def _body_freejoint_qvel_adr(model: mujoco.MjModel, body_id: int) -> int:
    j_adr = int(model.body_jntadr[body_id])
    return int(model.jnt_dofadr[j_adr])


def _quat_to_yaw(qw: float, qx: float, qy: float, qz: float) -> float:
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def _wrap_to_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _build_scene_index(model: mujoco.MjModel, cfg: EnvConfig) -> _SceneIndex:
    car_id = model.body("car").id

    robot_ids = frozenset(_find_descendants(model, car_id))

    wall_ids: set[int] = set()
    movable_ids: list[int] = []
    for i in range(model.nbody):
        name = model.body(i).name.lower()
        if i != 0 and "wall" in name:
            wall_ids.add(i)
        if "movable" in name:
            movable_ids.append(i)
    movable_ids.sort()

    car_qpos = _body_freejoint_qpos_adr(model, car_id)
    car_qvel = _body_freejoint_qvel_adr(model, car_id)
    movable_qpos = tuple(_body_freejoint_qpos_adr(model, b) for b in movable_ids)

    site_id = model.site("goal").id
    goal_xy = np.asarray(model.site_pos[site_id, :2], dtype=np.float64).copy()

    left_wheel_dof = int(model.jnt_dofadr[model.joint("left_wheel_joint").id])
    right_wheel_dof = int(model.jnt_dofadr[model.joint("right_wheel_joint").id])

    return _SceneIndex(
        model=model,
        left_act=int(model.actuator("left_wheel_drive").id),
        right_act=int(model.actuator("right_wheel_drive").id),
        left_wheel_dof=left_wheel_dof,
        right_wheel_dof=right_wheel_dof,
        car_qpos_adr=car_qpos,
        car_qvel_adr=car_qvel,
        robot_body_ids=robot_ids,
        wall_body_ids=frozenset(wall_ids),
        movable_body_ids=tuple(movable_ids),
        movable_qpos_adrs=movable_qpos,
        goal_xy=goal_xy,
        goal_theta=0.0,
    )


class DiffDriveCarEnv(gym.Env):
    """Pure-Python Gymnasium env: diff-drive car with direct wheel-velocity control.

    Action: (omega_left, omega_right) in rad/s, written to MuJoCo's `<velocity>`
    actuators. Episode terminates on robot-wall contact (fail) or when the robot
    reaches the goal site. The reward is stubbed at 0.0; wrap or override to add
    shaping.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        config: EnvConfig,
        render_mode: str | None = None,
        *,
        render_size: tuple[int, int] = (480, 640),  # (H, W)
        camera: str | int | mujoco.MjvCamera = -1,  # -1 = free, "top_down" = auto-fit overhead
    ):
        super().__init__()
        self._cfg = config
        self.render_mode = render_mode
        self._render_h, self._render_w = render_size
        self._camera_spec = camera
        self._camera_obj: mujoco.MjvCamera | None = None
        self._renderer: mujoco.Renderer | None = None
        self._renderer_model: mujoco.MjModel | None = None

        scene_dir = Path(config.scene_dir)
        if not scene_dir.is_absolute():
            scene_dir = Path.cwd() / scene_dir
        self._xml_paths: list[Path] = sorted(scene_dir.glob("*.xml"))
        if not self._xml_paths:
            raise FileNotFoundError(f"no .xml scenes found under {scene_dir}")

        self._model_cache: dict[Path, mujoco.MjModel] = {}
        self._rng = np.random.default_rng(config.seed)

        v_max = float(config.action.v_max)
        w_max = float(config.action.w_max)
        self.action_space = Box(
            low=np.array([-v_max, -w_max], dtype=np.float32),
            high=np.array([+v_max, +w_max], dtype=np.float32),
            dtype=np.float32,
        )
        n_max = int(config.obs.max_movables)
        self.observation_space = DictSpace(
            {
                "robot": Box(-np.inf, np.inf, (7,), dtype=np.float32),
                "goal": Box(-np.inf, np.inf, (3,), dtype=np.float32),
                "movables": Box(-np.inf, np.inf, (n_max, 3), dtype=np.float32),
                "mask": Box(0, 1, (n_max,), dtype=np.int8),
                "last_action": Box(
                    low=np.array([-v_max, -w_max], dtype=np.float32),
                    high=np.array([+v_max, +w_max], dtype=np.float32),
                    dtype=np.float32,
                ),
            }
        )

        # populated by reset()
        self._scene: _SceneIndex | None = None
        self._data: mujoco.MjData | None = None
        self._step_count: int = 0
        self._current_xml: Path | None = None
        # rate-limit state: previous commanded (v, ω). Reset to zero at reset().
        self._v_prev: float = 0.0
        self._w_prev: float = 0.0

    # ---- gym API --------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict, dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        if options and "xml_path" in options:
            xml_path = Path(options["xml_path"])
        else:
            idx = int(self._rng.integers(0, len(self._xml_paths)))
            xml_path = self._xml_paths[idx]

        model = self._model_cache.get(xml_path)
        if model is None:
            model = mujoco.MjModel.from_xml_path(str(xml_path))
            # Apply physics tuning: add reflected rotor inertia to the wheel
            # joints. This brings the actuator response time onto the timestep
            # scale, matching how published MuJoCo wheeled-robot models are
            # set up (see Menagerie Stretch 3, Robot Soccer Kit). Without it,
            # the wheel responds 250× faster than one mj_step and every ctrl
            # change becomes an impulse to the chassis.
            phys = self._cfg.physics
            lw_dof = int(model.jnt_dofadr[model.joint("left_wheel_joint").id])
            rw_dof = int(model.jnt_dofadr[model.joint("right_wheel_joint").id])
            if phys.wheel_armature > 0.0:
                model.dof_armature[lw_dof] = phys.wheel_armature
                model.dof_armature[rw_dof] = phys.wheel_armature
            if phys.wheel_damping > 0.0:
                model.dof_damping[lw_dof] = phys.wheel_damping
                model.dof_damping[rw_dof] = phys.wheel_damping
            self._model_cache[xml_path] = model

        scene = _build_scene_index(model, self._cfg)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        # Settle the chassis onto the floor before exposing state to the agent.
        # Spawn z is set above the floor in the XMLs; without this, the first
        # env.step combines a drop transient with the wheel command and the
        # chassis can briefly tumble.
        data.ctrl[scene.left_act] = 0.0
        data.ctrl[scene.right_act] = 0.0
        for _ in range(100):  # 100 substeps ≈ 200 ms of zero-command settling
            mujoco.mj_step(model, data)
        data.qvel[:] = 0.0  # zero velocities after settling so step 1 starts at rest

        self._scene = scene
        self._data = data
        self._step_count = 0
        self._current_xml = xml_path
        self._v_prev = 0.0
        self._w_prev = 0.0

        obs = self._build_obs()
        info = {
            "xml_path": str(xml_path),
            "goal_xy": scene.goal_xy.tolist(),
            "n_movables": len(scene.movable_body_ids),
        }
        return obs, info

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        assert self._scene is not None and self._data is not None, "call reset() first"
        scene, data = self._scene, self._data

        a = np.asarray(action, dtype=np.float64).reshape(2)
        cfg_a = self._cfg.action
        # Bound the commanded (v, omega) to the action-space envelope.
        v_cmd = float(np.clip(a[0], -cfg_a.v_max, +cfg_a.v_max))
        w_cmd = float(np.clip(a[1], -cfg_a.w_max, +cfg_a.w_max))
        # Rate-limit: max change per env.step is accel_max * dt_env.
        dt_env = self._cfg.frame_skip * float(scene.model.opt.timestep)
        dv_max = cfg_a.v_accel_max * dt_env
        dw_max = cfg_a.w_accel_max * dt_env
        v_cmd = float(np.clip(v_cmd, self._v_prev - dv_max, self._v_prev + dv_max))
        w_cmd = float(np.clip(w_cmd, self._w_prev - dw_max, self._w_prev + dw_max))
        self._v_prev = v_cmd
        self._w_prev = w_cmd
        # Kinematic split → wheel velocity setpoints.
        L_half = 0.5 * cfg_a.wheelbase
        r = cfg_a.wheel_radius
        data.ctrl[scene.left_act]  = (v_cmd - w_cmd * L_half) / r
        data.ctrl[scene.right_act] = (v_cmd + w_cmd * L_half) / r

        for _ in range(self._cfg.frame_skip):
            mujoco.mj_step(scene.model, data)

        self._step_count += 1

        wall_hit = robot_wall_contact(
            scene.model, data, scene.robot_body_ids, scene.wall_body_ids
        )

        x, y, theta = self._car_pose()
        dx = scene.goal_xy[0] - x
        dy = scene.goal_xy[1] - y
        dist = math.hypot(dx, dy)
        dtheta = _wrap_to_pi(scene.goal_theta - theta)
        goal_hit = (
            dist <= self._cfg.goal_tolerance.position
            and abs(dtheta) <= self._cfg.goal_tolerance.heading
        )

        terminated = False
        info: dict = {}
        if wall_hit:
            terminated = True
            info["termination"] = "wall"
        elif goal_hit:
            terminated = True
            info["termination"] = "goal"

        truncated = (not terminated) and self._step_count >= self._cfg.max_episode_steps
        if truncated:
            info["termination"] = "truncated"

        obs = self._build_obs()
        info["dist_to_goal"] = dist
        return obs, 0.0, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self.render_mode != "rgb_array":
            return None
        assert self._scene is not None and self._data is not None, "call reset() first"
        model, data = self._scene.model, self._data
        if self._renderer is None or self._renderer_model is not model:
            if self._renderer is not None:
                self._renderer.close()
            # MuJoCo's default offscreen framebuffer is 480x640; bump to fit.
            if int(model.vis.global_.offheight) < self._render_h:
                model.vis.global_.offheight = self._render_h
            if int(model.vis.global_.offwidth) < self._render_w:
                model.vis.global_.offwidth = self._render_w
            self._renderer = mujoco.Renderer(
                model, height=self._render_h, width=self._render_w
            )
            self._renderer_model = model
            self._camera_obj = self._resolve_camera(model)
        cam = self._camera_obj if self._camera_obj is not None else self._camera_spec
        self._renderer.update_scene(data, camera=cam)
        return self._renderer.render()

    def _resolve_camera(self, model: mujoco.MjModel) -> mujoco.MjvCamera | None:
        """Build a top-down MjvCamera fit to the current scene if requested.
        Returns None to fall through to update_scene's int/string handling."""
        if self._camera_spec != "top_down":
            return None
        # Fit overhead: lookat = world XY midpoint of all wall geoms; distance
        # chosen so the half-extent + margin fits the camera's FOV.
        wall_ids = self._scene.wall_body_ids if self._scene else set()
        xs, ys = [], []
        for g in range(model.ngeom):
            if int(model.geom_bodyid[g]) in wall_ids:
                p = model.geom_pos[g]
                s = model.geom_size[g]
                xs.extend([p[0] - s[0], p[0] + s[0]])
                ys.extend([p[1] - s[1], p[1] + s[1]])
        if not xs:
            # fallback: world bounds via model.stat.extent
            cx = cy = 0.0
            half = float(model.stat.extent)
        else:
            cx = 0.5 * (min(xs) + max(xs))
            cy = 0.5 * (min(ys) + max(ys))
            half = max(max(xs) - min(xs), max(ys) - min(ys)) * 0.5
        fovy_deg = float(model.vis.global_.fovy)
        # vertical fit: distance = half / tan(fovy/2), with a 10% margin
        dist = 1.1 * half / math.tan(math.radians(fovy_deg) * 0.5)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = [cx, cy, 0.0]
        cam.distance = dist
        cam.azimuth = 90.0      # +x right, +y up in image
        cam.elevation = -90.0   # straight down
        return cam

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
            self._renderer_model = None
        self._scene = None
        self._data = None

    # ---- helpers --------------------------------------------------------------

    def _car_pose(self) -> tuple[float, float, float]:
        scene, data = self._scene, self._data
        adr = scene.car_qpos_adr
        x = float(data.qpos[adr])
        y = float(data.qpos[adr + 1])
        qw = float(data.qpos[adr + 3])
        qx = float(data.qpos[adr + 4])
        qy = float(data.qpos[adr + 5])
        qz = float(data.qpos[adr + 6])
        theta = _quat_to_yaw(qw, qx, qy, qz)
        return x, y, theta

    def _car_body_velocity(self, theta: float) -> tuple[float, float, float]:
        """Return (vx_body, vy_body, omega_z_body).

        MuJoCo freejoint qvel is `(vx_world, vy_world, vz_world, wx_body, wy_body, wz_body)`.
        We rotate the world-frame linear part into the body frame.
        """
        scene, data = self._scene, self._data
        v = scene.car_qvel_adr
        vx_w = float(data.qvel[v])
        vy_w = float(data.qvel[v + 1])
        wz_b = float(data.qvel[v + 5])
        c, s = math.cos(theta), math.sin(theta)
        vx_b = vx_w * c + vy_w * s
        vy_b = -vx_w * s + vy_w * c
        return vx_b, vy_b, wz_b

    def _build_obs(self) -> dict:
        scene, data = self._scene, self._data
        x, y, theta = self._car_pose()
        vx_b, vy_b, wz = self._car_body_velocity(theta)

        robot = np.array(
            [x, y, math.sin(theta), math.cos(theta), vx_b, vy_b, wz],
            dtype=np.float32,
        )

        dx_w = scene.goal_xy[0] - x
        dy_w = scene.goal_xy[1] - y
        c, s = math.cos(theta), math.sin(theta)
        dx_b = dx_w * c + dy_w * s
        dy_b = -dx_w * s + dy_w * c
        dtheta = _wrap_to_pi(scene.goal_theta - theta)
        goal = np.array([dx_b, dy_b, dtheta], dtype=np.float32)

        n_max = int(self._cfg.obs.max_movables)
        movables = np.zeros((n_max, 3), dtype=np.float32)
        mask = np.zeros((n_max,), dtype=np.int8)
        for k, adr in enumerate(scene.movable_qpos_adrs):
            if k >= n_max:
                break
            mx = float(data.qpos[adr])
            my = float(data.qpos[adr + 1])
            mqw = float(data.qpos[adr + 3])
            mqx = float(data.qpos[adr + 4])
            mqy = float(data.qpos[adr + 5])
            mqz = float(data.qpos[adr + 6])
            mtheta = _quat_to_yaw(mqw, mqx, mqy, mqz)
            movables[k, 0] = mx
            movables[k, 1] = my
            movables[k, 2] = mtheta
            mask[k] = 1

        last_action = np.array([self._v_prev, self._w_prev], dtype=np.float32)
        return {
            "robot": robot,
            "goal": goal,
            "movables": movables,
            "mask": mask,
            "last_action": last_action,
        }
