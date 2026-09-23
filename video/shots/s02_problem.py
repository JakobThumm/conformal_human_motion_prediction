#!/usr/bin/env python
"""s02 -- the film's opening title card, over the live deployment.

Narration: "Our goal: from camera input alone, guarantee the robot is always at
a complete stop before it could touch the human."

This is now the **first shot of the film**.  The deployment recording plays
*fullscreen* and the film's title sits on top of it:

    Vision-Based Safe Human-Robot Collaboration
    guaranteeing human safety from camera input alone

The second line *is* the goal statement -- the standalone goal block of the
earlier cut is gone, so the claim is made once, in the title, and the frame
belongs to the footage.  The passage chosen is the one in which the operator
walks up and **actually puts his hand on the robot**; the arm is standing still
while he does it (the shield has already braked), so the footage is the claim.
A ring marks the contact point.

Motion discipline (SPEC "Motion discipline"): the two title lines, the camera
label and the word `contact` only **fade up on opacity** at fixed positions --
nothing slides, nothing scales, nothing ever animates out.  The only motion in
the shot is the footage itself and the contact ring / pulse, which are graphics.

Source: `video/media/realworld_source.mp4`, the RViz screen capture of the live
deployment; only the raw camera panel is used -- no pose overlay, no skeleton,
no RViz chrome.

    <video-py> video/shots/s02_problem.py --out build/shots/s02.mp4 --duration 8.5

No prepare step: the only cached artefact under `video/media/` that the
real-world shots share is the shield-state npz, which this shot only reads to
document (in `s02_notes.md`) that the arm is braking during the contact.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import style  # noqa: E402
import _realworld_common as rw  # noqa: E402

# ---------------------------------------------------------------- the footage
# Fullscreen: a 16:9 sub-rectangle of `rw.CROP_RGB` (the RViz "RGB Image" panel,
# 0,553 -> 668,927) scaled to fill the whole 1920x1080 frame.  Only the dead
# black curtain on the far left (panel x < 99) and the empty floor at the bottom
# are trimmed; everything else of the camera image is kept, so the upscale stays
# as gentle as a 668 px-wide source region allows: 569 x 320 -> 1920 x 1080,
# x3.375 Lanczos.  It is soft -- that is inherent to the source and accepted
# (the author asked for fullscreen); the grade is pulled down to hide it and a
# light unsharp pass puts the edges back.
CROP = (99, 553, 668, 873)                  # x0, y0, x1, y1 in source pixels
CROP_W, CROP_H = CROP[2] - CROP[0], CROP[3] - CROP[1]     # 569 x 320 = 16:9

# Source timeline.  The operator reaches for the arm three times; the third
# reach ends with his hand resting on the robot's link and staying there for
# ~0.4 s (source frames 956-967).  The last two segments slow down into that
# rest -- real frames throughout, then ~1.2 s on the contact.
SEGMENTS = [
    ("play", 25.80, 31.60, 1.00),           # three approaches, real time
    ("play", 31.60, 32.05, 0.45),           # the final reach, slowed
    ("play", 32.05, 32.20, 0.30),           # settling onto the link
]

# ---------------------------------------------------------------- the contact
# Measured on source frame 963 (t = 32.10 s) and cross-checked on 957 / 960 /
# 966: the operator's fingers lie on the white robot link at RGB-panel pixel
# (448, 130).  Stable to +-4 px over 31.87-32.23 s (frames 956-967); after
# 32.3 s the hand slides back down and off the link.  Over the same window the
# arm is motionless (mean abs. pixel difference of the robot-only region
# 480,80 -> 640,200 stays <= 1.2/255 against frame 960) and
# `_realworld_common.display_state()` reads *braking* for every frame in
# 940-975 -- the robot is stopped before he touches it, which is the claim.
CONTACT_SRC = (448, 130)                    # RGB-panel pixels
CONTACT_T0, CONTACT_T1 = 31.92, 32.06       # source seconds: ring eases in here

# ---------------------------------------------------------------- the words
TITLE1 = "Vision-Based Safe Human–Robot Collaboration"
TITLE2 = "guaranteeing human safety from camera input alone"
CAPTION = "Intel RealSense D435i"

T1_Y, T1_PX = 58, 58                        # top margin 40 px respected
T2_Y, T2_PX = 138, 34                       # glyph bottom ~ 180, head top ~ 224

# Opacity-only entries (SPEC "Motion discipline"): <= 0.25 s fades, no exits.
T1_IN = (0.10, 0.35)
T2_IN = (0.32, 0.57)
CAP_IN = (0.90, 1.15)

# ---------------------------------------------------------------- scrim
SCRIM_SOLID = 186          # full-strength band under the title ...
SCRIM_FADE = 330           # ... then a smooth ramp back to the picture
SCRIM_STRENGTH = 0.86

_BGF = np.array(common.hex2rgb(style.BG), np.float32)
_scrim_cache: dict = {}


def title_scrim(img: np.ndarray) -> None:
    """Darken the top band to `style.BG` so the title always reads.  In place.

    `rw.top_scrim` ramps from y = 0, which leaves the second line sitting at
    ~45 % darkening -- illegible over the lab's whiteboard.  This one holds full
    strength down to `SCRIM_SOLID` and only then ramps out.
    """
    key = (img.shape[0], img.shape[1])
    if key not in _scrim_cache:
        a = np.zeros(img.shape[0], np.float32)
        a[:SCRIM_SOLID] = 1.0
        n = SCRIM_FADE - SCRIM_SOLID
        r = np.linspace(0.0, 1.0, n, dtype=np.float32)
        a[SCRIM_SOLID:SCRIM_FADE] = 1.0 - r * r * (3 - 2 * r)
        _scrim_cache[key] = (a * SCRIM_STRENGTH)[:, None, None]
    m = _scrim_cache[key]
    h = SCRIM_FADE
    img[:h] = np.clip(img[:h].astype(np.float32) * (1 - m[:h]) + _BGF * m[:h],
                      0, 255).astype(np.uint8)


def sharpen(img: np.ndarray, amount: float = 0.42, sigma: float = 1.8) -> np.ndarray:
    """Gentle unsharp mask -- puts some edge back after the x3.4 upscale."""
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    return np.clip(img.astype(np.float32) * (1 + amount)
                   - blur.astype(np.float32) * amount, 0, 255).astype(np.uint8)


def contact_xy() -> tuple[float, float]:
    """Contact pixel -> canvas coordinates of the fullscreen framing."""
    cx = (CONTACT_SRC[0] - (CROP[0] - rw.CROP_RGB[0])) * style.WIDTH / CROP_W
    cy = (CONTACT_SRC[1] - (CROP[1] - rw.CROP_RGB[1])) * style.HEIGHT / CROP_H
    return cx, cy


def draw_contact(o: rw.Overlay, cx: float, cy: float, a: float, pulse: float) -> None:
    """Ring + one expanding pulse + a label at the contact point.

    The ring and the pulse are *graphics* and keep their motion; the word
    `contact` is type and therefore only fades up, at a fixed position.
    """
    if a <= 0.0:
        return
    col = style.C_DANGER
    if 0.0 < pulse < 1.0:                       # one ring grows out and dissolves
        r = 30 + 54 * common.ease(pulse, "out")
        o.ring((cx, cy), r, col, alpha=int(150 * a * (1 - pulse) ** 1.6), w=3)
    r0 = 36 * (0.55 + 0.45 * common.ease(a, "out"))
    o.ring((cx, cy), r0 + 6, col, alpha=int(55 * a), w=9)      # soft halo
    o.ring((cx, cy), r0, col, alpha=int(245 * a), w=3)
    o.dot((cx, cy), 4, col, alpha=int(245 * a))
    # leader + label, hung below the mark at a *fixed* position
    ty = cy + 62
    o.line([(cx, cy + r0 + 5), (cx, ty)], col, alpha=int(200 * a), w=2)
    lab = "contact"
    tw = o.width(lab, style.CAPTION_SIZE + 2, "medium")
    box = [cx - tw / 2 - 16, ty + 2, cx + tw / 2 + 16, ty + 46]
    o.rrect(box, 22, fill=style.BG, alpha=int(225 * a))
    o.rrect(box, 22, outline=col, alpha=int(120 * a), w=2)
    o.text((cx, ty + 24), lab, size=style.CAPTION_SIZE + 2, weight="medium",
           color=col, alpha=int(255 * a), anchor="mm")


def main():
    a = common.shot_args(__doc__)
    src = rw.Source()
    tl = rw.Timeline(SEGMENTS, a.duration, min_hold=0.6)
    cx, cy = contact_xy()
    mid = style.WIDTH // 2

    with common.FrameWriter(a) as w:
        for T in w.times():
            ts = tl.src(T)
            img = rw.grade(rw.resize(rw.crop(src.at_blend(ts), CROP),
                                     style.WIDTH, style.HEIGHT),
                           sat=0.82, gamma=1.20, gain=0.90)
            img = sharpen(img)
            rw.vignette(img, 0.38)
            title_scrim(img)
            rw.bottom_scrim(img, 806, 0.66)

            fade = common.seg(T, 0.0, 0.50, "out")
            if fade < 0.999:                     # the picture fades up from canvas
                img[:] = np.clip(img.astype(np.float32) * fade
                                 + _BGF * (1.0 - fade), 0, 255).astype(np.uint8)

            o = rw.Overlay()
            o.text((mid, T1_Y), TITLE1, size=T1_PX, weight="semibold",
                   color=style.FG, anchor="ma",
                   alpha=int(255 * common.seg(T, *T1_IN, "out")))
            o.text((mid, T2_Y), TITLE2, size=T2_PX,
                   color=style.FG_MUTED, anchor="ma",
                   alpha=int(245 * common.seg(T, *T2_IN, "out")))
            o.text((96, 866), CAPTION, size=style.CAPTION_SIZE,
                   color=style.FG_MUTED,
                   alpha=int(190 * common.seg(T, *CAP_IN, "out")))

            ca = common.ease((ts - CONTACT_T0) / (CONTACT_T1 - CONTACT_T0), "out")
            draw_contact(o, cx, cy, ca * fade,
                         (ts - CONTACT_T0) / 0.34 if ts >= CONTACT_T0 else -1.0)
            o.apply(img)
            w.write(img)
    src.close()


if __name__ == "__main__":
    main()
