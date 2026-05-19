"""Wandb-free mp4 recorder for evaluation episodes.

Drop-in replacement for `tdmpc_square.common.logger.VideoRecorder` that writes
locally via imageio-ffmpeg instead of uploading to wandb. Same public API:
`init(env, enabled=True)`, `record(env)`, `save(step, key=...)`.
"""
from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np


class LocalVideoRecorder:
    """Records a single env episode and writes it to `<work_dir>/eval_video/<key>_<step>.mp4`.

    Matches the duck-typed interface used by `tdmpc_square` online_trainer.eval():
    `init(env, enabled)` resets the frame buffer (enabled flag suppresses the
    trailing episodes of multi-episode eval — the trainer only enables it for
    `i == 0`). `record(env)` appends a frame. `save(step, key)` writes the mp4.
    """

    def __init__(self, cfg, fps: int = 15, wandb=None):
        self.cfg = cfg
        self._save_dir = Path(cfg.work_dir) / "eval_video"
        self._save_dir.mkdir(parents=True, exist_ok=True)
        self.fps = int(fps)
        self.frames: list[np.ndarray] = []
        self.enabled = False
        # Optional wandb handle — when provided, save() also logs the video to wandb
        # under the given key, matching the upstream VideoRecorder behavior.
        self._wandb = wandb

    def init(self, env, enabled: bool = True):
        self.frames = []
        self.enabled = bool(enabled)
        if self.enabled:
            self.record(env)

    def record(self, env):
        if not self.enabled:
            return
        try:
            frame = env.render()
        except Exception:
            frame = None
        if frame is None:
            return
        frame = np.asarray(frame)
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        self.frames.append(frame)

    def save(self, step: int, key: str = "eval_video"):
        if not self.enabled or not self.frames:
            return None
        # Use forward slashes in `key` (matches the upstream wandb call site
        # which passes e.g. "results/video"); flatten to a filesystem-safe name.
        safe = key.replace("/", "_")
        out_path = self._save_dir / f"{safe}_{int(step):08d}.mp4"
        writer = imageio.get_writer(
            str(out_path),
            fps=self.fps,
            codec="libx264",
            macro_block_size=1,
            quality=8,
        )
        try:
            for fr in self.frames:
                writer.append_data(fr)
        finally:
            writer.close()
        # Push to wandb if a handle was provided.
        if self._wandb is not None:
            try:
                frames = np.stack(self.frames).transpose(0, 3, 1, 2)
                self._wandb.log(
                    {key: self._wandb.Video(frames, fps=self.fps, format="mp4")},
                    step=int(step),
                )
            except Exception as e:  # pragma: no cover -- defensive
                print(f"[LocalVideoRecorder] wandb upload failed: {e}")
        return str(out_path)
