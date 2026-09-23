"""Helpers every shot renderer shares: CLI, frame sinks, exact-duration encoding.

A renderer only ever has to do two things:

    args = shot_args("my shot")                     # --out/--duration/--fps/--width/--height
    with FrameWriter(args) as w:                    # or: write_frames(args, frames_iterable)
        for t in w.times():                         # t in [0, duration)
            w.write(rgb_uint8_HxWx3)

and the encoding, pixel format, exact frame count and edge fades are handled here.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

VIDEO_DIR = Path(__file__).resolve().parent
REPO = VIDEO_DIR.parent
MEDIA = VIDEO_DIR / "media"
BUILD = VIDEO_DIR / "build"

sys.path.insert(0, str(VIDEO_DIR))
import style  # noqa: E402

FFMPEG = os.environ.get("FFMPEG", "/home/thumm/miniconda3/envs/chmp-video/bin/ffmpeg")
if not Path(FFMPEG).exists():           # fall back to whatever is on PATH
    FFMPEG = "ffmpeg"


def shot_args(description: str = "", extra=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--duration", required=True, type=float)
    p.add_argument("--fps", type=int, default=style.FPS)
    p.add_argument("--width", type=int, default=style.WIDTH)
    p.add_argument("--height", type=int, default=style.HEIGHT)
    p.add_argument("--fade", type=float, default=0.0, help="fade in/out seconds (0 = none)")
    if extra is not None:
        extra(p)
    a = p.parse_args()
    a.n_frames = int(round(a.duration * a.fps))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    return a


def hex2rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


class FrameWriter:
    """Pipe raw RGB frames into ffmpeg; guarantees exactly `n_frames` frames."""

    def __init__(self, args, crf: int = 16):
        self.a = args
        self.crf = crf
        self.n = 0
        self.proc = None

    def __enter__(self):
        cmd = [
            FFMPEG, "-y", "-v", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{self.a.width}x{self.a.height}", "-r", str(self.a.fps),
            "-i", "pipe:0",
            "-an", "-c:v", "libx264", "-preset", "slow", "-crf", str(self.crf),
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(self.a.out),
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        return self

    def times(self):
        """Frame timestamps in seconds, [0, duration)."""
        for i in range(self.a.n_frames):
            yield i / self.a.fps

    def write(self, rgb: np.ndarray):
        a = np.ascontiguousarray(rgb.astype(np.uint8, copy=False))
        assert a.shape == (self.a.height, self.a.width, 3), a.shape
        self._last = a.tobytes()
        self.proc.stdin.write(self._last)
        self.n += 1

    def __exit__(self, *exc):
        if exc[0] is None:
            # pad/trim to the exact contract length
            if self.n == 0:
                blank = np.zeros((self.a.height, self.a.width, 3), np.uint8)
                blank[:] = hex2rgb(style.BG)
                for _ in range(self.a.n_frames):
                    self.write(blank)
            while self.n < self.a.n_frames:
                self.proc.stdin.write(self._last)
                self.n += 1
        self.proc.stdin.close()
        self.proc.wait()
        if exc[0] is None and self.proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed for {self.a.out}")
        return False

    # remember the last frame cheaply for padding
    _last = b""


def write_frames(args, frames, crf: int = 16):
    """Encode an iterable of HxWx3 uint8 arrays, padding/trimming to args.n_frames."""
    last = None
    with FrameWriter(args, crf=crf) as w:
        for i, f in enumerate(frames):
            if i >= args.n_frames:
                break
            w.write(f)
            last = f
        while w.n < args.n_frames and last is not None:
            w.write(last)


def conform(src: Path, out: Path, duration: float, fps: int = style.FPS,
            width: int = style.WIDTH, height: int = style.HEIGHT,
            vf_extra: str = "", crf: int = 16, start: float = 0.0,
            speed: float = 1.0, loop: bool = False):
    """Re-encode an existing clip to the shot contract (exact duration, canvas, silent)."""
    vf = []
    if speed != 1.0:
        vf.append(f"setpts={1.0 / speed}*PTS")
    if vf_extra:
        vf.append(vf_extra)
    vf += [
        f"scale={width}:{height}:force_original_aspect_ratio=decrease:flags=lanczos",
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color={style.BG}",
        f"fps={fps}", "format=yuv420p",
    ]
    cmd = [FFMPEG, "-y", "-v", "error"]
    if loop:
        cmd += ["-stream_loop", "-1"]
    cmd += ["-ss", f"{start}", "-i", str(src), "-t", f"{duration}",
            "-vf", ",".join(vf), "-an", "-c:v", "libx264", "-preset", "slow",
            "-crf", str(crf), "-pix_fmt", "yuv420p", str(out)]
    subprocess.run(cmd, check=True)


def ease(t: float, kind: str = "smooth") -> float:
    """Clamped easing on t in [0, 1]."""
    t = float(np.clip(t, 0.0, 1.0))
    if kind == "linear":
        return t
    if kind == "smooth":                 # smoothstep
        return t * t * (3 - 2 * t)
    if kind == "smoother":               # smootherstep
        return t * t * t * (t * (6 * t - 15) + 10)
    if kind == "out":                    # ease-out cubic
        return 1 - (1 - t) ** 3
    if kind == "in":
        return t ** 3
    raise ValueError(kind)


def seg(t: float, t0: float, t1: float, kind: str = "smooth") -> float:
    """Eased progress of a sub-interval [t0, t1] of the shot timeline."""
    if t1 <= t0:
        return 1.0 if t >= t1 else 0.0
    return ease((t - t0) / (t1 - t0), kind)
