"""Contact sheet of the assembled film -- one tile per sampled second, for review.

    python video/contactsheet.py                      # whole film, 1 tile / 4 s
    python video/contactsheet.py --shot s09 --every 1 # one shot, 1 tile / s
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

VIDEO_DIR = Path(__file__).resolve().parent
FFMPEG = "/home/thumm/miniconda3/envs/chmp-video/bin/ffmpeg"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shot", default="")
    ap.add_argument("--every", type=float, default=4.0)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--out", type=Path, default=VIDEO_DIR / "build/contactsheet.png")
    a = ap.parse_args()

    src = (VIDEO_DIR / f"build/shots/{a.shot}.mp4") if a.shot else (VIDEO_DIR / "build/silent.mp4")
    if not src.exists():
        raise SystemExit(f"no {src}; run build.py shots/assemble first")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    dur = float(subprocess.run(
        [FFMPEG.replace("ffmpeg", "ffprobe"), "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(src)], capture_output=True, text=True).stdout.strip())
    rows = max(1, math.ceil(dur / a.every / a.cols))      # ffmpeg's tile needs an explicit NxM
    subprocess.run([FFMPEG, "-y", "-v", "error", "-i", str(src),
                    "-vf", f"fps=1/{a.every},scale=480:-1,"
                           f"tile={a.cols}x{rows}:padding=6:color=#05070B",
                    "-frames:v", "1", str(a.out)], check=True)
    print(a.out)
    tl = VIDEO_DIR / "build/timeline.json"
    if tl.exists() and not a.shot:
        for i in json.loads(tl.read_text()):
            print(f"  {i['start']:7.2f}  {i['sid']}  {i['module']}")


if __name__ == "__main__":
    main()
