#!/usr/bin/env python
"""s08 -- "The spheres become capsules: the human's reachable occupancy.  SARA
shield intersects them with the robot's.  No intersection, go.  Otherwise,
fail-safe stop."

Two beats:

  0.  the manuscript's pipeline figure (`_overview.py`) with the **safety
      verification** stage lit -- the verification test of eq. (7) and the
      reachable occupancies built from the conformal prediction sets.  The rest
      of the pipeline falls away and the lit block dissolves into --
  1.  live shield output, not a re-render: the 3-D view of
      `video/media/realworld_source.mp4`, cropped free of every piece of RViz
      chrome and graded into the film's palette, with our own callouts, a state
      badge and a small intersection schematic on top.

Source events (measured, see s08_notes.md).  SARA shield draws the robot's
reachable-occupancy capsules **green** while the candidate trajectory is
verified and **red** while it is failing verification and the robot is braking;
the badge and the schematic are driven frame by frame from that read-out
(`_realworld_common.shield_state`), never from a script.

  * 46.20 -- 50.10 s: 3.9 s of *verified* operator motion, the two occupancies
    closing in with a clear gap;
  * 0.4 s cross-dissolve (the two moments frame almost identically);
  * 33.30 -- 37.10 s: the cleanest disjoint -> intersecting transition in the
    recording -- frame 1024 (34.1333 s) is the last green frame, 1025 the first
    red one, then ~3 s of sustained braking.

    <video-py> video/shots/s08_shield.py --out build/shots/s08.mp4 --duration 14.7
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
import _overview as OV  # noqa: E402
import _realworld_common as rw  # noqa: E402

# --------------------------------------------------------------------------- footage
# 3-D view crop: 1560 x 877 of the render area (which is x 692..2542, y 98..1414),
# upscaled x1.231 with Lanczos to the film frame.  Wider than the earlier cut so the
# geometry sits in the upper-left two thirds and the callouts, the badge and the
# schematic all have empty canvas to live on (measured union of the coloured
# geometry over every frame this shot uses: x 418..1287, y 45..752).
VIEW = (860, 412, 2420, 1289)

FR = 1.0 / rw.SRC_FPS

# play / hold / speed-ramp plan, in *footage* time (see module docstring + notes)
SEGMENTS = [
    # -- clip A: a long run of verified motion -------------------------------
    ("play", 46.200, 50.100, 0.88),
    # -- clip B: the approach, the contact of the two sets, the braking ------
    # frame 1024 (34.1333 s) is the last frame whose capsules are still green;
    # 1025 is the first red one.  The freeze lands exactly on that boundary.
    ("play", 33.300, 1024 * FR, 0.45, 0.40),   # 0.40 s cross-dissolve out of clip A
    ("hold", 1024 * FR, 0.70),                 # freeze on the last verified frame
    ("play", 1024 * FR, 34.500, 1.00),         # the sets meet -- real time, untouched
    ("play", 34.500, 35.500, 0.42),            # braking, slowed
    ("play", 35.500, 37.100, 1.00),            # braked, real time
]

# --------------------------------------------------------------------------- overview beat
# `OV.REGION_SHIELD` = (2, 548, 2098, 1107) -- the figure's **entire bottom row**
# (intended trajectory -> monitored-trajectory planner -> robot reachable
# occupancy -> Verification -> the reachable occupancies built from the conformal
# sets).  That row *is* SARA shield, so the shot lights all of it; the earlier
# local override (1000, 555, 2098, 1105), which lit only the right half, is gone.
REGION = OV.REGION_SHIELD

T_REV = (0.08, 0.55)         # the figure fades up
T_FOC = (0.50, 1.15)         # the safety-verification stage lights up
T_ISO = (2.05, 2.55)         # the rest of the pipeline dissolves away
T_MOVE = (2.15, 3.00)        # ... the lit block grows towards the live geometry
T_XF = (2.40, 3.00)          # ... and cross-dissolves into it
T_FOOT0 = 2.45               # footage clock starts here
# The lit block is now 1310 x 358 px on canvas (centre 960, 722) instead of the
# 606 x 347 of the half-row, i.e. it already spans two thirds of the frame.  The
# old x1.55 grow pushed it 2030 px wide -- past both frame edges, so the ends of
# the row (and of eq. (7)) fell off screen before the dissolve finished.  x1.32
# is the largest grow that keeps the whole row inside 1920 px while still
# reading as "move into the live scene".
OV_ZOOM = 1.32
OV_TARGET = (930.0, 430.0)   # where the lit block heads for: the live action

TITLE = "Verify the monitored trajectory"

# --------------------------------------------------------------------------- overlay layout
KICK = (96, 58)
HUMAN_TXT = (1306, 176)      # right column: free of geometry at every frame
HUMAN_SUB = ("Conformal prediction sets at the point\n"
             "in time when the robot is stopped.")
HUMAN_LEAD = [(1290, 196), (1090, 196), (938, 246)]
HUMAN_DOT = (930, 252)

ROBOT_TXT = (96, 782)        # bottom-left band, clear of the geometry (y < 752)
ROBOT_SUB = ("Robot occupancy along its monitored trajectory.\n"
             "The monitored trajectory ends with a stopped robot.")
ROBOT_LEAD = [(272, 772), (272, 548), (655, 393)]
ROBOT_DOT = (665, 385)

# Soft scrims behind the two callouts: the RViz floor grid runs straight through
# both text blocks otherwise (it reads as a strike-through on the headlines).
HUMAN_SCRIM = (1278, 152, 1800, 322)
ROBOT_SCRIM = (70, 758, 748, 914)

BOX = (1364, 600, 1844, 880)          # verification schematic

# Schematic spheres: equal radii r, centre distance d.  The two discs interpenetrate
# by exactly (2r - d); with d = 1.6 r that is 0.4 r, i.e. **20 % of the diameter**
# (40 % of the radius), which is the author's "intersect by 20 %" and is the
# smallest overlap that still unmistakably reads as *intersecting* rather than
# *touching* at 50 % scale (d = 1.8 r gives only 0.2 r and reads as a kiss).
# Disjoint state uses d = 2.55 r -- a clear gap of 0.55 r.
SPH_R = 52
SPH_D_DISJOINT = 2.55 * SPH_R
SPH_D_INTERSECT = 1.60 * SPH_R


def draw_badge(o: rw.Overlay, braking: bool, alpha: float):
    if alpha <= 0.01:
        return
    col = style.C_DANGER if braking else style.C_TRUTH
    label = "FAILSAFE STOP" if braking else "VERIFIED"
    a = int(255 * alpha)
    x1, y = 1824, 58
    wd = o.width(label, 26, "semibold")
    o.rrect([x1 - wd - 62, y - 14, x1 + 16, y + 44], 16,
            fill=style.BG_PANEL, alpha=int(150 * alpha))
    o.dot((x1 - wd - 36, y + 15), 8, col, a)
    o.text((x1, y), label, size=26, weight="semibold", color=col, alpha=a,
           anchor="ra")


def _lens(o: rw.Overlay, cx_l: float, cx_r: float, cy: float, r: float,
          color: str, alpha: int):
    """Fill the lens where two equal-radius discs overlap (scanline exact)."""
    d = cx_r - cx_l
    if d >= 2 * r:
        return
    half = np.sqrt(max(r * r - (d / 2.0) ** 2, 0.0))
    for dy in range(-int(half), int(half) + 1):
        s = np.sqrt(max(r * r - dy * dy, 0.0))
        x0, x1 = cx_r - s, cx_l + s
        if x1 > x0:
            o.line([(x0, cy + dy), (x1, cy + dy)], color, alpha, 1)


def draw_schematic(o: rw.Overlay, braking: bool, alpha: float, mix: float):
    """Corner diagram of the verification test, eq. (7) of the manuscript."""
    if alpha <= 0.01:
        return
    a = int(255 * alpha)
    x0, y0, x1, y1 = BOX
    o.rrect([x0, y0, x1, y1], 16, fill=style.BG_PANEL, alpha=int(190 * alpha),
            outline=style.GRID, w=2)
    o.text((x0 + 28, y0 + 20), "Verification", size=24, weight="medium",
           color=style.FG_MUTED, alpha=int(215 * alpha))

    cy = y0 + 150
    cx = (x0 + x1) / 2.0
    d = SPH_D_DISJOINT + (SPH_D_INTERSECT - SPH_D_DISJOINT) * mix
    rx_, hx_ = cx - d / 2.0, cx + d / 2.0      # robot LEFT, human RIGHT
    rcol = style.C_DANGER if braking else style.C_TRUTH

    o.dot((rx_, cy), SPH_R - 2, rcol, int(46 * alpha))
    o.dot((hx_, cy), SPH_R - 2, style.C_OURS, int(46 * alpha))
    if mix > 0.02:
        _lens(o, rx_, hx_, cy, SPH_R - 2, style.C_DANGER, int(170 * alpha * mix))
    o.ring((rx_, cy), SPH_R, rcol, a, 3)
    o.ring((hx_, cy), SPH_R, style.C_OURS, a, 3)
    o.text((rx_ - 12, cy), "R", size=26, weight="semibold", color=rcol,
           alpha=a, anchor="mm")
    o.text((hx_ + 12, cy), "H", size=26, weight="semibold", color=style.C_OURS,
           alpha=a, anchor="mm")

    txt = ("Sets intersect   →   failsafe stop" if braking
           else "Sets disjoint   →   verified safe")
    o.text((x0 + 28, y1 - 54), txt, size=25, weight="semibold",
           color=style.C_DANGER if braking else style.C_TRUTH, alpha=a)


# --------------------------------------------------------------------------- overview stills
def overview_stills(width: int, height: int):
    """flat figure, lit figure, lit block alone, and the block's frame rect."""
    flat = np.array(OV.compose(REGION, reveal=1.0, focus=0.0, accent=style.C_OURS,
                               width=width, height=height), np.float32)
    lit = np.array(OV.compose(REGION, reveal=1.0, focus=1.0, accent=style.C_OURS,
                              width=width, height=height), np.float32)
    sx = OV.FIG_W / 2100.0
    box = (OV.FIG_X + REGION[0] * sx - 6, OV.FIG_Y + REGION[1] * sx - 6,
           OV.FIG_X + REGION[2] * sx + 6, OV.FIG_Y + REGION[3] * sx + 6)
    f = 46.0                                   # soft edge of the isolation mask
    bx = (box[0] - f, box[1] - f, box[2] + f, box[3] + f)
    x = np.arange(width, dtype=np.float32)
    y = np.arange(height, dtype=np.float32)
    mx = np.clip((x - bx[0]) / f, 0, 1) * np.clip((bx[2] - x) / f, 0, 1)
    my = np.clip((y - bx[1]) / f, 0, 1) * np.clip((bx[3] - y) / f, 0, 1)
    mx = mx * mx * (3.0 - 2.0 * mx)
    my = my * my * (3.0 - 2.0 * my)
    keep = (my[:, None] * mx[None, :])[:, :, None]
    bg = np.array(common.hex2rgb(style.BG), np.float32)
    iso = lit * keep + bg * (1.0 - keep)
    return flat, lit, iso, box


_SCRIM_CACHE: dict = {}


def scrim_mask(rect, w: int, h: int) -> np.ndarray:
    """Soft rounded-rect alpha map used to push the footage back behind type."""
    key = (rect, w, h)
    if key not in _SCRIM_CACHE:
        m = np.zeros((h, w), np.float32)
        x0, y0, x1, y1 = rect
        cv2.rectangle(m, (x0, y0), (x1, y1), 1.0, -1)
        m = cv2.GaussianBlur(m, (0, 0), 30.0)
        _SCRIM_CACHE[key] = m[:, :, None]
    return _SCRIM_CACHE[key]


def main():
    a = common.shot_args(__doc__)
    src = rw.Source()
    tl = rw.Timeline(SEGMENTS, a.duration - T_FOOT0, min_hold=0.6)

    t_freeze = T_FOOT0 + tl.spans[2][0]        # the freeze on the last green frame
    t_meet = T_FOOT0 + tl.spans[3][0]          # the two sets touch

    flat, lit, iso, box = overview_stills(a.width, a.height)
    box_c = np.array([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0])
    bgcol = common.hex2rgb(style.BG)
    bgf = np.array(bgcol, np.float32)

    def warp(img: np.ndarray, s: float, dst: np.ndarray) -> np.ndarray:
        M = np.float32([[s, 0.0, dst[0] - s * box_c[0]],
                        [0.0, s, dst[1] - s * box_c[1]]])
        return cv2.warpAffine(img, M, (a.width, a.height), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=bgcol)

    def footage(T: float) -> np.ndarray:
        """One graded, cropped, resized live frame (cross-dissolved where needed)."""
        parts = tl.blend(max(T - T_FOOT0, 0.0))
        if len(parts) == 1:
            bgr = rw.crop(src.at_blend(parts[0][0]), VIEW)
        else:
            acc = None
            for ts, wt in parts:
                f = rw.crop(src.at_blend(ts), VIEW).astype(np.float32) * wt
                acc = f if acc is None else acc + f
            bgr = np.clip(acc, 0, 255).astype(np.uint8)
        img = rw.resize(rw.grade(bgr), a.width, a.height)
        rw.vignette(img, 0.40)
        rw.top_scrim(img, 150, 0.35)
        rw.bottom_scrim(img, 690, 0.72)
        return img

    with common.FrameWriter(a) as w:
        for T in w.times():
            xf = common.seg(T, *T_XF, "smooth")
            ts = tl.src(max(T - T_FOOT0, 0.0))

            if xf >= 0.996:
                img = footage(T)
            else:
                rev = common.seg(T, *T_REV, "out")
                foc = common.seg(T, *T_FOC, "smooth")
                isok = common.seg(T, *T_ISO, "smooth")
                tau = common.seg(T, *T_MOVE, "smooth")
                ov = bgf + (flat - bgf) * rev
                if foc > 0.001:
                    ov = ov + (lit - ov) * foc
                if isok > 0.001:
                    ov = ov + (iso - ov) * isok
                ov = np.clip(ov, 0, 255).astype(np.uint8)
                if tau > 0.001:
                    ov = warp(ov, 1.0 + (OV_ZOOM - 1.0) * tau,
                              box_c + (np.array(OV_TARGET) - box_c) * tau)
                img = ov
                if xf > 0.004:
                    live = footage(T)
                    img = np.clip(ov.astype(np.float32) * (1 - xf)
                                  + live.astype(np.float32) * xf, 0, 255).astype(np.uint8)

            braking = rw.braking_at(ts)
            # Text entries are opacity-only and <= 0.25 s (SPEC "Motion
            # discipline"); nothing here ever animates out.
            ca = common.seg(T, 3.10, 3.35, "out")
            cb = common.seg(T, 4.35, 4.60, "out")
            for rect, amt in ((HUMAN_SCRIM, 0.62 * ca), (ROBOT_SCRIM, 0.62 * cb)):
                if amt > 0.01:
                    m = scrim_mask(rect, a.width, a.height) * amt
                    img[:] = np.clip(img.astype(np.float32) * (1 - m) + bgf * m,
                                     0, 255).astype(np.uint8)

            o = rw.Overlay()
            o.text(KICK, "Safety verification", size=style.CAPTION_SIZE,
                   weight="medium", color=style.FG_MUTED,
                   alpha=int(255 * common.seg(T, 0.05, 0.30, "out")))
            # The title fades up once and then *stays* for the whole shot -- it
            # names the live footage just as well as it names the figure.  (It
            # used to fade out into the transition; exit animations are out.)
            o.text((96, 104), TITLE, size=style.H1_SIZE, weight="medium",
                   color=style.FG, alpha=int(255 * common.seg(T, 0.20, 0.45, "out")))

            # --- callout 1: the human's reachable occupancy (blue)
            if ca > 0:
                al = int(255 * ca)
                o.line(HUMAN_LEAD, style.C_OURS, int(200 * ca), 2)
                o.dot(HUMAN_DOT, 5, style.FG, al)
                o.text(HUMAN_TXT, "Human reachable occupancy", size=30,
                       weight="semibold", color=style.C_OURS, alpha=al)
                o.text((HUMAN_TXT[0], HUMAN_TXT[1] + 42), HUMAN_SUB, size=25,
                       color=style.FG_MUTED, alpha=int(225 * ca), spacing=8)

            # --- callout 2: the robot's reachable occupancy (green / red)
            if cb > 0:
                al = int(255 * cb)
                o.line(ROBOT_LEAD, style.FG_MUTED, int(200 * cb), 2)
                o.dot(ROBOT_DOT, 5, style.FG, al)
                o.text(ROBOT_TXT, "Robot reachable occupancy", size=30,
                       weight="semibold", color=style.FG, alpha=al)
                o.text((ROBOT_TXT[0], ROBOT_TXT[1] + 42), ROBOT_SUB, size=25,
                       color=style.FG_MUTED, alpha=int(225 * cb), spacing=8)

            draw_badge(o, braking, common.seg(T, 3.00, 3.25, "out"))
            # `mix` moves the two schematic discs together -- a *graphic*, so it
            # keeps its motion; the panel and its labels only fade up.
            mix = common.seg(T, t_meet, t_meet + 0.55, "smooth")
            draw_schematic(o, braking, common.seg(T, 5.70, 5.95, "out"), mix)
            o.apply(img)
            w.write(img)
    src.close()


if __name__ == "__main__":
    main()
