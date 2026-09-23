#!/usr/bin/env python
"""s16 -- the real-world deployment.

"On hardware: a Franka arm, a RealSense camera, the same pipeline at twenty-five
hertz.  Blue is the human's conformal occupancy over the robot's stopping
horizon.  When it meets the robot's reachable set, the shield brakes.  In every
tested instance the robot stood still long before the operator could reach it."

Everything on screen is the live deployment recording
`video/media/realworld_source.mp4` (RViz screen grab, 2560x1440, 30 fps).  The
RViz chrome, options tree, window frame and cursor are cropped away; the 3-D
view becomes the film frame and the 2-D pose panel becomes an inset card.  The
state badge is read frame by frame out of the capsule colour in the capture
(see `_realworld_common.display_state`), it is not scripted.

Three cuts, in capture time (see s16_notes.md):
    14.60 - 17.95 s   operator at working distance, trajectory verified
    43.00 - 51.00 s   braked with the operator close, he clears, the shield
                      re-verifies, he comes back and it brakes again at 50.67 s
    53.50 - 60.05 s   the decisive approach; brake from 57.83 s to the end
The brake onset itself (57.60 - 58.97 s) runs at 1.0x, untouched.  The two
approach clips are eased to 0.75x / 0.85x and the final 1.08 s -- the operator's
hand at the stopped robot -- to 0.42x.

Motion discipline (SPEC): the kicker, the equipment strip and the state badge
only fade up on opacity at fixed positions and then stay.  Nothing slides and
nothing animates out; the only thing that ever removes an overlay is the
dip-to-canvas at a footage cut, which is a graphic transition the picture
itself rides.

    <video-py> video/shots/s16_realworld.py --out build/shots/s16.mp4 --duration 21
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import style  # noqa: E402
import _realworld_common as rw  # noqa: E402

# 3-D view crop: 1600 x 900 of the render area (x1.2 to the film frame).
VIEW = (900, 400, 2500, 1300)

XF = 0.42                   # dip-to-canvas transition between the three cuts
SEGMENTS = [
    # in-points chosen so that the shield state is unambiguous at every cut
    ("play", 14.60, 17.95, 0.75),           # verified, operator at working distance
    ("play", 43.00, 51.00, 1.00, XF),       # braked close -> clears -> verified -> brakes
    ("play", 53.50, 57.60, 0.85, XF),       # the decisive approach, still verified
    ("play", 57.60, 58.97, 1.00),           # the brake onset, real time, untouched
    ("play", 58.97, 60.05, 0.42),           # the operator's hand at the stopped robot
]

# Inset card: the RViz 2-D pose panel, upscaled.
CARD = (1304, 140, 520, 292)                # x, y, w, h  (bottom 432 << 928)

STRIP = "Franka Emika  ·  Intel RealSense D435i  ·  25 Hz  ·  SARA shield"


DIM = rw.dim_map(vig=0.44, top=230, top_s=0.48, bottom=800, bottom_s=0.60)


def footage(src: rw.Source, ts: float) -> np.ndarray:
    return rw.resize(rw.grade(rw.crop(src.at_blend(ts), VIEW)),
                     style.WIDTH, style.HEIGHT)


def main():
    a = common.shot_args(__doc__)
    src = rw.Source()
    tl = rw.Timeline(SEGMENTS, a.duration, min_hold=0.8)
    t_end = tl.core_end

    with common.FrameWriter(a) as w:
        for T in w.times():
            # Transitions dip through the film canvas rather than
            # cross-dissolving: superimposing two cuts would put red and green
            # capsules -- contradictory shield states -- in the same frame.
            parts = tl.blend(T)
            if len(parts) == 1:
                ts_now, k = parts[0][0], 1.0
            else:
                u = parts[1][1]
                ts_now = parts[0][0] if u < 0.5 else parts[1][0]
                k = max(0.0, abs(2.0 * u - 1.0))
            img = footage(src, ts_now)
            rw.apply_dim(img, DIM)
            if k < 0.999:
                img[:] = np.clip(img.astype(np.float32) * k +
                                 np.array(common.hex2rgb(style.BG), np.float32)
                                 * (1 - k), 0, 255).astype(np.uint8)

            # Every overlay below enters on **opacity only**, at a fixed
            # position, over <= 0.25 s, and never animates out (SPEC "Motion
            # discipline").  The only thing that ever takes an overlay off the
            # screen is the dip-to-canvas at a footage cut -- `k`, the same
            # graphic transition the picture itself rides.
            fade = common.seg(T, 0.0, 0.25, "out")

            # --- inset: the 2-D pose the pipeline actually consumes
            cx, cy, cw, ch = CARD
            card_a = fade * common.seg(T, 0.55, 0.80, "out") * k
            if card_a > 0.01:
                pose = rw.grade(rw.resize(rw.crop(src.at(ts_now), rw.CROP_POSE),
                                          cw, ch),
                                sat=0.80, gamma=1.18, gain=0.92)
                rw.paste_card(img, pose, cx, cy, 14, alpha=card_a,
                              border=style.GRID, border_alpha=0.9)

            o = rw.Overlay()
            o.text((96, 58), "Results · real-world deployment", size=style.CAPTION_SIZE,
                   weight="medium", color=style.FG_MUTED, alpha=int(255 * fade))
            # (the inset card carries no caption: the equipment strip below already names
            #  the RealSense D435i, and the 2-D overlay reads as what it is)

            # --- persistent equipment strip
            sa = fade * common.seg(T, 0.30, 0.55, "out")
            if sa > 0.01:
                wd = o.width(STRIP, 24, "medium")
                o.rrect([96, 846, 96 + wd + 52, 902], 14, fill=style.BG_PANEL,
                        alpha=int(150 * sa))
                o.text((96 + 26, 860), STRIP, size=24, weight="medium",
                       color=style.FG_MUTED, alpha=int(235 * sa))

            # --- shield state, read out of the capture's capsule colour
            # The badge rides `k`: it is dark exactly at the midpoint of a
            # dip-to-canvas cut, which is where the shield state jumps -- so it
            # can never carry the previous clip's state into the next one.  (It
            # used to have its own fade-out gate; that was a text exit.)
            ba = fade * common.seg(T, 0.80, 1.05, "out") * k
            if ba > 0.01:
                braking = rw.braking_at(ts_now)
                col = style.C_DANGER if braking else style.C_TRUTH
                label = "FAIL-SAFE STOP" if braking else "VERIFIED"
                # a slow breath on the held final frame keeps it alive
                pulse = 1.0
                if T > t_end:
                    pulse = 0.70 + 0.30 * float(
                        np.cos(2 * np.pi * (T - t_end) / 2.2) * 0.5 + 0.5)
                al = int(255 * ba)
                wd = o.width(label, 26, "semibold")
                o.rrect([1824 - wd - 62, 46, 1840, 104], 16,
                        fill=style.BG_PANEL, alpha=int(150 * ba))
                o.dot((1824 - wd - 36, 75), 8, col, int(al * pulse))
                o.text((1824, 60), label, size=26, weight="semibold",
                       color=col, alpha=al, anchor="ra")
            o.apply(img)
            w.write(img)
    src.close()


if __name__ == "__main__":
    main()
