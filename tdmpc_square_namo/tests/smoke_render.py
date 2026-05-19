"""Render-path smoke: confirm the env returns rgb frames and LocalVideoRecorder writes mp4."""
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MUJOCO_GL", "egl")

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
_REPO_ROOT = os.path.abspath(os.path.join(_PKG_ROOT, ".."))
for _p in (_PKG_ROOT, _REPO_ROOT, os.path.join(_REPO_ROOT, "tdmpc_square")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tdmpc_square_namo.envs.namo import make_env
from tdmpc_square_namo.video import LocalVideoRecorder


def main():
    work_dir = Path(_HERE) / "_smoke_video_out"
    work_dir.mkdir(exist_ok=True)
    cfg = SimpleNamespace(task="namo_car", seed=0, namo=None, save_video=True, work_dir=work_dir)

    env = make_env(cfg)
    obs, _ = env.reset(seed=0)

    frame = env.render()
    print("frame:", None if frame is None else (frame.shape, frame.dtype))
    assert frame is not None and frame.ndim == 3 and frame.shape[2] == 3, "render() must return HxWx3"

    rec = LocalVideoRecorder(cfg, fps=15)
    rec.init(env, enabled=True)
    for _ in range(30):
        a = env.action_space.sample()
        env.step(a)
        rec.record(env)
    out = rec.save(step=42, key="results/video")
    print("wrote:", out, "size:", Path(out).stat().st_size)
    print("RENDER SMOKE OK")


if __name__ == "__main__":
    main()
