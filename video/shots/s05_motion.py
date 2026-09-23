#!/usr/bin/env python
"""s05 -- "A transformer reads two seconds of this uncertain history, and predicts the next four
hundred milliseconds, with covariances."

Two beats.

1.  The manuscript's pipeline figure (`_overview.py`) with the **uncertainty-aware motion
    prediction** stage lit, so the viewer knows where in the system we are.  It then pushes in on
    the lit block and cross-dissolves into --
2.  one real H36M test window (subject S5, action "Walking", window index 13892 of the test
    split), built around one unmistakable walking figure:

      grey   the K_I = 50 observed poses -- a dotted wrist / ankle / root trail plus 8 fading
             ghost skeletons, ending in ONE solid, thick, clearly readable current pose
      teal   the K_P = 10 predicted poses -- a dotted joint trail, the +400 ms mean pose drawn
             solid and *on top of* its own shells, and every predicted joint's 99.99 %
             covariance ball

All geometry is real, from `video/media/s05_window.npz`.  Build that cache once with

    cd /home/thumm/code/conformal_human_motion_prediction
    JAX_PLATFORMS=cpu .venv/bin/python video/shots/_human3d_prepare.py

Render:
    /home/thumm/miniconda3/envs/chmp-video/bin/python video/shots/s05_motion.py \
        --out video/build/shots/s05.mp4 --duration 11.5
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _overview as OV                            # noqa: E402
import common                                     # noqa: E402
import style                                      # noqa: E402
from _human3d import (Camera, Canvas, Scene, Text, draw_ground, fit_view, font,  # noqa: E402
                      hex2f, load_window, mix, motion_azimuth)

SS = 2                      # super-sampling factor
ELEV = 12.0                 # camera elevation, degrees
# Off the pure side view: a 3/4 read of the stride, and it foreshortens the 1.18 m of travel so
# the history does not smear the figure across the frame.
YAW_OFF = 62.0
ORBIT = 16.0                # total azimuth sweep over the 3-D beat, degrees
N_GHOSTS = 8                # ghost skeletons drawn out of the 50 observed poses
TRAIL_STRIDE = 2            # dot every 2nd observed frame

# Dotted-trail joints: both wrists, both ankles (JOINT_NAMES_13 indices).  The root is derived as
# the hip midpoint -- there is no root joint in the 13-joint layout.
TRAIL_JOINTS = (5, 6, 11, 12)
HIPS = (7, 8)

# ---- text rail ------------------------------------------------------------
# Hanging indent: the legend bullet sits on the 60 px margin (centre 70, r 9 -> left edge 61) and
# the text hangs at 100.  The bullet in front of each K-line *is* the legend, which is why there
# is no separate legend block any more.
X_BULLET, X_TEXT = 70, 100

# ---- beat 1: the overview figure -----------------------------------------
# Beat 1 is budgeted in ABSOLUTE seconds (beat 2 takes the rest), so a re-timed narration
# lengthens the 3-D beat -- where the covariance balls grow -- instead of stretching a diagram
# the viewer has already read.  Squeezed proportionally if the shot ever drops below
# T_X1 / 0.42 = 8.2 s.
T_OV_IN = (0.00, 0.22)      # the figure appears (opacity only)
T_FOCUS = (0.24, 0.74)      # rest of the pipeline dims, the motion stage lights (graphic)
T_ISO = (2.22, 2.88)        # ... and then falls away completely: only the lit stage is left
T_X0, T_X1 = 2.92, 3.48     # cross-dissolve, lit stage -> 3-D scene

# Every piece of type in this shot enters on opacity alone, over FADE seconds, and never
# animates out (SPEC "Motion discipline"); beat 1's type is hard-cut at T_X1.
FADE = 0.22
T_TITLE = 0.06              # the overview's own headline appears
T_CAPTION = 0.30            # ... and the lit stage is named under it

# The lit block in output-frame pixels.  The transition is an *opening* of exactly this box: a
# soft-edged rectangle that grows from it to past the frame edge, with the 3-D scene revealed
# inside it and the accent outline riding its boundary -- so the highlighted stage of the figure
# literally becomes the scene.
_SX = OV.FIG_W / 2100.0
LIT_BOX = (OV.FIG_X + OV.REGION_MOTION[0] * _SX - 6, OV.FIG_Y + OV.REGION_MOTION[1] * _SX - 6,
           OV.FIG_X + OV.REGION_MOTION[2] * _SX + 6, OV.FIG_Y + OV.REGION_MOTION[3] * _SX + 6)


# --------------------------------------------------------------------------- math type
# Math is the one place the film allows a second face (SPEC "Type"): K_I / K_P are set in
# Computer Modern via matplotlib's mathtext, cap-height-matched to the Inter line they sit in, so
# the subscript is real typesetting rather than a baseline-shifted guess.
_MATH: dict = {}


def _math_raster(expr: str, dpi: float):
    """(coverage [h,w] in 0..1, baseline row) for `expr` set in Computer Modern at `dpi`.

    `matplotlib.mathtext.MathTextParser("agg")` hands back the raw glyph coverage plus the
    typeset box's `depth` (pixels below the baseline), which is what lets the maths sit on the
    *measured* Inter baseline instead of an eyeballed offset.
    """
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["mathtext.fontset"] = "cm"
    from matplotlib.font_manager import FontProperties
    from matplotlib.mathtext import MathTextParser
    r = MathTextParser("agg").parse(expr, dpi=dpi, prop=FontProperties(size=40))
    return np.asarray(r.image, np.float32) / 255.0, r.height - r.depth


def math_glyph(expr: str, size: int, weight: str = "semibold"):
    """(coverage [h,w], baseline-from-top) for `expr`, cap-height-matched to Inter `size`."""
    key = (expr, size, weight)
    if key not in _MATH:
        bb = font(size, weight).getbbox("K")            # (x0, cap_top, x1, baseline)
        cap_h = bb[3] - bb[1]
        # Cap height of a Computer Modern "K" at a reference dpi -> the dpi that renders it ~4x
        # oversampled, then the exact scale measured at *that* dpi (not extrapolated).
        a0, base0 = _math_raster(r"$K$", 200.0)
        dpi = 200.0 * (4.0 * cap_h) / (base0 - np.nonzero(a0.max(1) > 0.03)[0][0])
        a1, base1 = _math_raster(r"$K$", dpi)
        scale = cap_h / (base1 - np.nonzero(a1.max(1) > 0.03)[0][0])

        a, base = _math_raster(expr, dpi)
        cols = np.nonzero(a.max(0) > 0.03)[0]
        a = a[:, cols[0]:cols[-1] + 1]                  # crop the side bearings
        im = Image.fromarray((a * 255.0 + 0.5).astype(np.uint8), "L")
        w = max(1, int(round(im.width * scale)))
        h = max(1, int(round(im.height * scale)))
        cov = np.asarray(im.resize((w, h), Image.LANCZOS), np.float32) / 255.0
        _MATH[key] = (cov, base * h / a.shape[0])
    return _MATH[key]


def math_line(tx: Text, xy, expr: str, rest: str, size: int, weight: str, colour: str,
              alpha: float) -> None:
    """Draw `expr` (maths) followed by `rest` (Inter) as one line anchored at `xy` ("la")."""
    if alpha <= 0.004:
        return
    cov, base = math_glyph(expr, size, weight)
    baseline = xy[1] + font(size, weight).getbbox("K")[3]      # Inter's own baseline for this line
    r, g, b = (int(colour.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
    rgba = np.zeros(cov.shape + (4,), np.uint8)
    rgba[..., 0], rgba[..., 1], rgba[..., 2] = r, g, b
    rgba[..., 3] = np.clip(cov * float(np.clip(alpha, 0, 1)) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    tx.img.alpha_composite(Image.fromarray(rgba, "RGBA"),
                           dest=(int(round(xy[0])), int(round(baseline - base))))
    gap = font(size, weight).getlength(" ")
    tx.label((xy[0] + cov.shape[1] + gap, xy[1]), rest, size, weight, colour, alpha)


def lit_mask(w: int, h: int, feather: float = 44.0) -> np.ndarray:
    """1 over the lit block, 0 far outside, smoothstep across `feather` px.

    The box is padded by exactly `feather` so the plateau starts on the block's own edge: the
    accent outline and the panel's caption stay fully lit, only the surroundings fall away.
    """
    box = (LIT_BOX[0] - feather, LIT_BOX[1] - feather, LIT_BOX[2] + feather, LIT_BOX[3] + feather)
    x = np.arange(w, dtype=np.float32)
    y = np.arange(h, dtype=np.float32)
    sx = np.clip((x - box[0]) / feather, 0, 1) * np.clip((box[2] - x) / feather, 0, 1)
    sy = np.clip((y - box[1]) / feather, 0, 1) * np.clip((box[3] - y) / feather, 0, 1)
    sx = sx * sx * (3.0 - 2.0 * sx)
    sy = sy * sy * (3.0 - 2.0 * sy)
    return (sy[:, None] * sx[None, :])[:, :, None]


def bullet_y(xy, size: int, weight: str) -> int:
    """Vertical centre of the cap band of a line drawn at `xy` -- where its bullet belongs."""
    bb = font(size, weight).getbbox("K")
    return int(round(xy[1] + 0.5 * (bb[1] + bb[3])))


# --------------------------------------------------------------------------- main
def main():
    args = common.shot_args(__doc__)
    D, W, H = args.duration, args.width, args.height

    # `_overview.compose` re-decodes and re-scales the cached figure on every call; memoise both
    # (this process only -- the module itself is shared and must not be edited).
    _dark = OV.dark_figure()
    OV.dark_figure = lambda: _dark
    _fit_memo: dict = {}
    _orig_fit = OV._fit

    def _fit(fig, width):
        if width not in _fit_memo:
            _fit_memo[width] = _orig_fit(fig, width)
        return _fit_memo[width]
    OV._fit = _fit

    d = load_window("s05")
    conn = d["connections"]
    inp = d["input_poses"].astype(np.float64)        # [50,13,3] m, camera-observed (noisy)
    pred = d["pred"].astype(np.float64)              # [10,13,3] m
    # s05 draws the model's own covariance ball at the paper's 1-eps = 99.99 % level,
    # r = sqrt(chi2_3(0.9999) * lambda_max(C)) -- i.e. BEFORE conformal calibration (that is s06's
    # job; s09 draws the calibrated set).  Cached by _human3d_prepare.py from the repo's
    # utils.eval_utils.convert_covariance_matrices_to_set on the real predicted covariances.
    r_cov = d["r_model"].astype(np.float64)          # [10,13] m
    horizon_ms = d["horizon_ms"].astype(np.float64)  # [10] -> 40 .. 400 ms
    K_I, K_P, J = inp.shape[0], pred.shape[0], inp.shape[1]

    trail_pts = np.concatenate(
        [inp[:, list(TRAIL_JOINTS)], inp[:, HIPS, :].mean(axis=1, keepdims=True)], axis=1)

    # ---- framing ----------------------------------------------------------
    # Fit on the HERO only -- the current pose and the predictions inflated by their covariance
    # balls.  Including the whole 1.18 m trail in the fit is what used to shrink the figure to an
    # unreadable size; the trail is checked against a looser box afterwards instead.
    Wss, Hss = W * SS, H * SS
    hero = np.concatenate([inp[-1], pred.reshape(-1, 3)])
    hero_r = np.concatenate([np.zeros(J), r_cov.ravel()])
    body = np.concatenate([inp.reshape(-1, 3), pred.reshape(-1, 3)])
    target = np.array([hero[:, 0].mean(), hero[:, 1].mean(),
                       0.50 * (body[:, 2].min() + body[:, 2].max())])
    az0 = motion_azimuth(target, inp[0], pred[-1], ELEV, Wss, Hss) + YAW_OFF
    azims = [az0 + ORBIT * f for f in (0.0, 0.35, 0.7, 1.0)]

    HERO_BOX = (0.400 * Wss, 0.060 * Hss, 0.975 * Wss, 0.850 * Hss)
    LOOSE_BOX = (0.320 * Wss, 0.055 * Hss, 0.992 * Wss, 0.856 * Hss)
    dist, shift = fit_view(target, azims, ELEV, hero, hero_r, Wss, Hss, HERO_BOX)

    def trail_fits(dd, sh):
        for az in azims:
            cam = Camera(target, dd, az, ELEV, Wss, Hss, shift=sh)
            xy, z = cam.project(trail_pts.reshape(-1, 3))
            if z.min() <= 0.2 or xy.min(0)[0] < LOOSE_BOX[0] or xy.min(0)[1] < LOOSE_BOX[1] \
                    or xy.max(0)[0] > LOOSE_BOX[2] or xy.max(0)[1] > LOOSE_BOX[3]:
                return False
        return True

    if not trail_fits(dist, shift):
        # the history would leave the canvas -> re-fit on everything against the looser box
        all_pts = np.concatenate([hero, trail_pts.reshape(-1, 3)])
        all_r = np.concatenate([hero_r, np.zeros(trail_pts.shape[0] * trail_pts.shape[1])])
        dist, shift = fit_view(target, azims, ELEV, all_pts, all_r, Wss, Hss, LOOSE_BOX)

    # ---- palette: history cool grey, prediction teal -----------------------
    C_HIST = hex2f(style.FG_MUTED)
    C_NOW = hex2f(style.FG)
    C_P = hex2f(style.C_PRED)
    C_P_HOT = mix(C_P, np.ones(3, np.float32), 0.30)

    # ---- beat-2 timeline (fractions of the 3-D beat, which is [x0, D] of the shot) ----
    # The extra second the shot was given goes here: the horizon (T_PRED) opens out over ~3.9 s
    # instead of ~2.6 s, and the finished frame is held for ~2.4 s.
    T_GRID = (0.00, 0.05)
    T_TRAIL = (0.00, 0.17)      # the history streams in fast -- it is already filling the frame
    T_NOW = (0.03, 0.17)        # ... and the hero pose solidifies *under* the cross-dissolve, so
                                #     the paper's little skeleton becomes the big one
    T_PRED = (0.27, 0.72)
    T_HOLD = 0.72

    ghosts = list(range(K_I - 1 - 6 * (N_GHOSTS - 1), K_I - 1, 6))   # 8 of the 50 poses
    canvas = Canvas(W, H, ss=SS)

    # ---- beat 1 -----------------------------------------------------------
    keep = lit_mask(W, H)
    sq = min(1.0, 0.42 * D / T_X1)            # squeeze beat 1 if the shot is ever very short
    S = lambda s: s * sq                      # absolute seconds -> this shot's clock  # noqa
    x0, x1 = S(T_X0), S(T_X1)
    # wall-clock cues for beat 2's type (the geometry stays on the fractions above)
    t_kp = x0 + (T_PRED[0] + 0.02) * (D - x0)     # the K_P line, as the first shells appear
    t_msg = x0 + T_HOLD * (D - x0)                # the closing line, on the finished frame

    def overview(t: float) -> np.ndarray:
        # The headline and the stage name are NOT handed to `compose`: they are drawn in
        # `frame()` so their opacity is on their own short clock and they are hard-cut at the
        # beat change, instead of riding the figure's fade-up and its cross-dissolve out.
        img = OV.compose(OV.REGION_MOTION, reveal=common.seg(t, S(T_OV_IN[0]), S(T_OV_IN[1]),
                                                             "out"),
                         focus=common.seg(t, S(T_FOCUS[0]), S(T_FOCUS[1]), "smooth"),
                         kicker="", title="", caption="",
                         accent=style.C_PRED).astype(np.float32)
        # the rest of the pipeline falls away, so the shot cross-dissolves out of the lit stage
        # alone rather than out of a busy diagram
        iso = common.seg(t, S(T_ISO[0]), S(T_ISO[1]), "smooth")
        if iso > 0.001:
            bg = np.array(common.hex2rgb(style.BG), np.float32)
            k = (1.0 - iso) + iso * keep
            img = img * k + bg * (1.0 - k)
        return img

    # ---- beat 2 -----------------------------------------------------------
    def scene(v: float, t: float) -> np.ndarray:
        canvas.clear()
        cam = Camera(target, dist, az0 + ORBIT * common.ease(v, "smooth"), ELEV, Wss, Hss,
                     shift=shift)
        sc = Scene(cam, canvas)

        a_grid = common.seg(v, *T_GRID, "out")
        draw_ground(sc, target[:2], half=2.5, step=0.5, alpha=0.55 * a_grid, fade=2.2)

        tp = common.seg(v, *T_TRAIL, "smooth") * (K_I - 1)

        # --- dotted history trail: wrists, ankles, hip midpoint --------------
        for i in range(0, K_I - 1, TRAIL_STRIDE):
            rev = np.clip(tp - i, 0.0, 1.0)
            if rev <= 0.02:
                break
            age = i / (K_I - 2.0)
            a = rev * (0.22 + 0.55 * age ** 1.6)
            for p in trail_pts[i]:
                sc.dot(p, (2.0 + 2.4 * age) * SS, C_HIST, a)

        # --- a few fading ghost skeletons -----------------------------------
        for i in ghosts:
            rev = np.clip(tp - i, 0.0, 1.0)
            if rev <= 0.01:
                continue
            age = i / (K_I - 1.0)
            sc.skeleton(inp[i], conn, C_HIST, 1.9 * SS, rev * (0.035 + 0.25 * age ** 3.0))

        # --- THE current observed pose: solid, thick, unmistakable -----------
        a_now = common.seg(v, *T_NOW, "out")
        if a_now > 0.01:
            sc.skeleton(inp[-1], conn, C_NOW, 5.4 * SS, a_now, joint_px=7.4 * SS,
                        depth_bias=-0.55)

        # --- the K_P predictions ---------------------------------------------
        hp = common.seg(v, *T_PRED, "smooth") * K_P
        for k in range(K_P):
            rev = common.ease(np.clip(hp - k, 0.0, 1.0), "out")
            if rev <= 0.01:
                continue
            g = (k + 1) / K_P
            # "comet" weight: the frontier of the horizon glows, older shells settle back, so 130
            # spheres never become 130 competing rings.
            hot = float(np.exp(-max(hp - k - 1.0, 0.0) / 1.7))
            for j in range(J):
                sc.sphere(pred[k][j], float(r_cov[k, j]) * rev, C_P,
                          fill=(0.005 + 0.055 * hot) * rev,
                          rim=(0.026 + 0.40 * hot) * rev,
                          rim_width=0.10 + 0.05 * (1.0 - hot))
                sc.dot(pred[k][j], (1.8 + 1.8 * g) * SS, C_P_HOT,
                       rev * (0.35 + 0.5 * g), depth_bias=-0.55)
            if k < K_P - 1:                      # the in-between means stay a whisper
                sc.skeleton(pred[k], conn, C_P, 1.2 * SS, rev * (0.05 + 0.13 * g))

        # the +400 ms mean pose, solid and biased in front of its own shells
        rev_last = common.ease(np.clip(hp - (K_P - 1), 0.0, 1.0), "out")
        if rev_last > 0.01:
            sc.skeleton(pred[-1], conn, C_P, 4.2 * SS, 0.98 * rev_last, joint_px=5.6 * SS,
                        depth_bias=-0.6)

        sc.paint()
        img = canvas.to_rgb8()

        # ---- left rail ----------------------------------------------------
        tx = Text(W, H)

        # The bullet in front of each line carries the colour legend: white = the observed pose,
        # teal = the predicted mean + covariance.  (There is no separate legend block.)
        a_in = common.seg(t, x1 + 0.20, x1 + 0.20 + FADE, "out")
        p_in = (X_TEXT, 112)
        tx.swatch((X_BULLET, bullet_y(p_in, 32, "semibold")), style.FG, a_in * 0.85)
        math_line(tx, p_in, r"$K_I$", "= 50 observed poses", 32, "semibold", style.FG, a_in)
        tx.label((X_TEXT, 152), "2 s of uncertain history @ 25 Hz", 26, "regular",
                 style.FG_MUTED, a_in)

        a_out = common.seg(t, t_kp, t_kp + FADE, "out")
        p_out = (X_TEXT, 226)
        tx.swatch((X_BULLET, bullet_y(p_out, 32, "semibold")), style.C_PRED, a_out)
        math_line(tx, p_out, r"$K_P$", "= 10 predicted poses", 32, "semibold", style.C_PRED,
                  a_out)
        tx.label((X_TEXT, 266), "400 ms, each joint with its covariance", 26, "regular",
                 style.FG_MUTED, a_out)

        a_msg = common.seg(t, t_msg, t_msg + FADE, "out")
        tx.label((X_TEXT, 340), "the ball grows with the horizon", 32, "semibold", style.C_PRED,
                 a_msg)

        # ---- horizon read-out ---------------------------------------------
        a_h = common.seg(t, t_kp, t_kp + FADE, "out")
        hp_c = float(np.clip(hp, 0.0, K_P))
        ms = 0.0 if hp_c <= 0 else float(horizon_ms[int(np.ceil(hp_c)) - 1])
        tx.label((X_TEXT, 726), "HORIZON", 26, "medium", style.FG_MUTED, a_h)
        tx.label((X_TEXT, 758), f"+{ms:.0f} ms", 62, "semibold", style.C_PRED, a_h)
        bar_w, bar_y = 300, 858
        tx.rule((X_TEXT, bar_y), bar_w, 3, style.GRID, a_h * 0.9)
        tx.rule((X_TEXT, bar_y), max(2, int(bar_w * hp_c / K_P)), 3, style.C_PRED, a_h)
        return tx.over(img)

    # ---- per-frame composite ---------------------------------------------
    def frame(t: float) -> np.ndarray:
        if t <= x0:
            img = overview(t)
        else:
            v = (t - x0) / max(D - x0, 1e-3)
            img = scene(v, t).astype(np.float32)
            if t < x1:
                q = common.seg(t, x0, x1, "smooth")
                img = overview(t) * (1.0 - q) + img * q
        img = np.clip(img, 0, 255).astype(np.uint8)

        tk = Text(W, H)
        tk.label((64, 46), "Motion prediction", 26, "regular", style.FG_MUTED,
                 common.seg(t, 0.00, FADE, "out"))
        if t < x1:
            # beat 1's type: same origin and size `_overview.compose` would have used, but on
            # its own opacity clock -- and hard-cut (not faded) when beat 2 takes over.
            tk.label((64, 92), "From pose history to the next 400 ms", 42, "medium",
                     style.FG, common.seg(t, S(T_TITLE), S(T_TITLE) + FADE, "out"))
            tk.label((64, 146), "Uncertainty-aware motion prediction", 30, "medium",
                     style.C_PRED, common.seg(t, S(T_CAPTION), S(T_CAPTION) + FADE, "out"))
        return tk.over(img)

    with common.FrameWriter(args) as w:
        for t in w.times():
            w.write(frame(t))


if __name__ == "__main__":
    main()
