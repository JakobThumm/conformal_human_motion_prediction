"""s03 -- the pipeline's perception stage, then two camera views with per-joint covariance.

Narration: "It starts with two views.  An adapted YOLO twenty-six returns, for every joint,
not a pixel, but a covariance."

The shot opens on the manuscript's pipeline figure (`_overview.py`) with the
**2-D pose estimation + uncertainty-aware triangulation** stage lit, holds ~1.2 s, then the lit
box grows into -- and cross-dissolves to -- the live two-view footage, which gets the rest of
the shot.  Type never moves: every label appears on opacity alone and nothing animates out.

Everything after the transition is real output of the repo's deployed 2D front-end (custom
ultralytics fork, YOLO26 `Pose26` head with per-keypoint sigma) on real H36M frames --
S11 / "Greeting" / cameras 55011271 + 60457274, frames 1524..1607.  See `_pose_data.py`.

Cache (run once, repo venv, GPU):

    XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python video/shots/s03_pose2d.py --prepare

Render (offline, no model, no dataset):

    .venv/bin/python video/shots/s03_pose2d.py --out video/build/shots/s03.mp4 --duration 7.3
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

VIDEO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VIDEO_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _overview as OV  # noqa: E402
import common  # noqa: E402
import style  # noqa: E402

CACHE = VIDEO_DIR / "media" / "s03_pose2d.npz"

# 13-joint skeleton, h36m_settings.CONNECTIONS_13
BONES = [(0, 1), (0, 2), (1, 3), (3, 5), (2, 4), (4, 6), (1, 2), (1, 7), (2, 8),
         (7, 8), (7, 9), (9, 11), (8, 10), (10, 12)]
JOINT_NAMES = ['nose', 'left shoulder', 'right shoulder', 'left elbow', 'right elbow',
               'left wrist', 'right wrist', 'left hip', 'right hip', 'left knee',
               'right knee', 'left ankle', 'right ankle']

# ------------------------------------------------------------------ layout (1920x1080)
# PW/PH must stay at 4:5 -- `prepare()` picks the cached crop rects with that aspect.
PW, PH = 512, 640                     # camera panel
PX = (100, 640)                       # panel left edges (28 px gutters, 100 px margins)
PY = 200
CX, CY, CS = 1180, 200, 640           # magnifier card
CARD_NATIVE = 200.0                   # native pixels shown inside the card
CAP_Y = PY + PH + 32                  # caption baseline row (872)
MAG_CAM = 1                           # magnify camera 2 (the panel next to the card)
MARGIN = 100                          # type left edge, flush with panel 1 (as in s04)

# ------------------------------------------------------------------ timeline (seconds)
# Beat 1 (the pipeline figure) is budgeted in ABSOLUTE seconds; beat 2 (the live footage) takes
# whatever is left.  At 7.3 s a fraction-based beat 1 either flickers or eats the footage, which
# is the substance of the shot.  Everything is squeezed proportionally if the shot is ever cut
# below OV_END / 0.42 = 6.8 s, so the contract still holds at any --duration.
OV_REVEAL = 0.22                      # the figure appears (opacity only)
OV_LIGHT = (0.26, 0.68)               # the perception stage lights up (graphic)
OV_HOLD = 1.88                        # ... and is held, legible, until here
OV_ISO = (1.88, 2.32)                 # the rest of the pipeline dissolves, the lit box stays
OV_MOVE = (2.06, 2.86)                # the lit box grows towards the camera panels
OV_XF = 2.34                          # ... while the footage dissolves in (ends with OV_MOVE)
OV_END = OV_MOVE[1]
LIVE_T0 = 2.18                        # live-phase clock starts just under the dissolve
CARD_V0, CARD_V1 = 0.36, 0.52         # magnifier fly-in, as a fraction of the live phase
FADE = 0.22                           # every text entry: opacity only, no movement, no exit
OV_ZOOM = 2.05                        # how far the box grows before it is gone
LIVE_Z0 = 0.92                        # the footage settles in from slightly under size

FONT_DIR = Path("/usr/share/fonts/opentype/inter")


# =============================================================================== prepare
def prepare() -> None:
    import _pose_data as pd

    d = pd.run()
    rects = pd.crop_rects(d["kp"], d["cov"], aspect=PW / PH)
    jpegs, sizes = [], []
    for c in range(2):
        j, s = pd.encode_jpegs(d["frames"][c], rects[c], scale=1.0, quality=88)
        jpegs.append(j)
        sizes.append(s)

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CACHE,
        jpeg0=np.array(jpegs[0], dtype=object), jpeg1=np.array(jpegs[1], dtype=object),
        size=np.asarray(sizes), rect=rects,
        kp=d["kp"], cov=d["cov"], conf=d["conf"],
        meta=np.array([pd.SUBJECT, pd.ACTION, pd.CAMERAS[0], pd.CAMERAS[1],
                       str(pd.START_FRAME), str(pd.N_FRAMES), str(pd.SRC_FPS)]),
        allow_pickle=True,
    )
    print(f"wrote {CACHE}  ({CACHE.stat().st_size / 1e6:.1f} MB)")
    sig = np.sqrt(np.stack([d["cov"][:, :, :, 0, 0], d["cov"][:, :, :, 1, 1]], -1))
    for j in (5, 6):
        print(f"  {JOINT_NAMES[j]:>13s}: median sigma cam2 = "
              f"{np.median(sig[1, :, j, 0]):.1f} x {np.median(sig[1, :, j, 1]):.1f} px")


# ================================================================================ render
def rgb(h: str) -> tuple[int, int, int]:
    return common.hex2rgb(h)


_fonts: dict = {}


def font(size: int, weight: str = "Regular"):
    key = (size, weight)
    if key not in _fonts:
        _fonts[key] = ImageFont.truetype(str(FONT_DIR / f"Inter-{weight}.otf"), size)
    return _fonts[key]


def text(canvas: np.ndarray, xy, s: str, size: int, colour: str, weight: str = "Regular",
         alpha: float = 1.0, anchor: str = "ls", tracking: float = 0.0) -> None:
    """Draw text with PIL straight onto the uint8 canvas."""
    if alpha <= 0.004 or not s:
        return
    im = Image.fromarray(canvas)
    dr = ImageDraw.Draw(im, "RGBA")
    f = font(size, weight)
    if tracking:
        x, y = xy
        for ch in s:
            dr.text((x, y), ch, font=f, anchor=anchor, fill=rgb(colour) + (int(255 * alpha),))
            x += dr.textlength(ch, font=f) + tracking
    else:
        dr.text(xy, s, font=f, anchor=anchor, fill=rgb(colour) + (int(255 * alpha),))
    canvas[:] = np.asarray(im)


SH = 4          # cv2 sub-pixel shift bits
SC = 1 << SH


def _pt(p):
    return (int(round(p[0] * SC)), int(round(p[1] * SC)))


def blend(base: np.ndarray, layer: np.ndarray, a: float) -> None:
    if a >= 0.999:
        base[:] = layer
    elif a > 0.004:
        cv2.addWeighted(layer, a, base, 1.0 - a, 0.0, dst=base)


def ellipse_pts(mu, cov, k: float, n: int = 96) -> np.ndarray:
    """k-sigma ellipse of a 2x2 covariance as a polygon."""
    w, v = np.linalg.eigh(cov)
    w = np.maximum(w, 1e-9)
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    circle = np.stack([np.cos(th), np.sin(th)], 1) * (k * np.sqrt(w))[None, :]
    return circle @ v.T + np.asarray(mu)[None, :]


def treat(img: np.ndarray, sat: float = 0.30, gain: float = 0.62) -> np.ndarray:
    """Desaturate + darken + pull towards BG so the overlay pops."""
    g = img.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
    out = img.astype(np.float32) * sat + g[..., None] * (1.0 - sat)
    out *= gain
    bg = np.asarray(rgb(style.BG), np.float32)
    out = out * 0.90 + bg * 0.10
    return np.clip(out, 0, 255).astype(np.uint8)


def paint(panel: np.ndarray, mask: np.ndarray, colour: str) -> None:
    """Alpha-composite one flat colour through a float mask (no double-darkening on overlap)."""
    if mask.max() <= 0.004:
        return
    m = mask[..., None]
    c = np.asarray(rgb(colour), np.float32)
    panel[:] = np.clip(panel.astype(np.float32) * (1 - m) + c * m, 0, 255).astype(np.uint8)


def _mask(shape) -> np.ndarray:
    return np.zeros(shape[:2], np.float32)


def _stamp(mask: np.ndarray, draw, alpha: float) -> None:
    """Rasterise one AA shape and merge it into `mask` with max() (union, not accumulation)."""
    if alpha <= 0.004:
        return
    tmp = np.zeros(mask.shape, np.uint8)
    draw(tmp)
    np.maximum(mask, tmp.astype(np.float32) * (alpha / 255.0), out=mask)


def draw_overlay(panel: np.ndarray, kp: np.ndarray, cov: np.ndarray, scale: float,
                 origin: np.ndarray, p_skel: float, p_ell: float,
                 dot: float = 4.9, lw: int = 3, ell_lw: int = 2,
                 focus: int | None = None, dim: float = 1.0) -> None:
    """2-sigma ellipses first, skeleton on top.  All progress values in [0, 1]."""
    P = (kp - origin[None, :]) * scale

    # --- covariance ellipses, staggered bloom with a slight overshoot
    if p_ell > 0:
        fill, line = _mask(panel.shape), _mask(panel.shape)
        for j in range(13):
            q = np.clip((p_ell - 0.030 * j) / 0.55, 0, 1)
            if q <= 0:
                continue
            e = q * q * (3 - 2 * q)
            k = e * (1.0 + 0.14 * np.sin(np.pi * e))          # bloom overshoot
            w = dim if (focus is None or j == focus) else dim * 0.42
            poly = ellipse_pts(P[j], cov[j] * (scale ** 2), 2.0 * k)
            pts = np.round(poly * SC).astype(np.int32)[None]
            _stamp(fill, lambda t, p=pts: cv2.fillPoly(t, p, 255, cv2.LINE_AA, SH),
                   0.15 * e * w)
            _stamp(line, lambda t, p=pts: cv2.polylines(t, p, True, 255, ell_lw,
                                                        cv2.LINE_AA, SH), 0.85 * e * w)
        paint(panel, fill, style.C_OURS)
        paint(panel, line, style.C_OURS)

        if focus is not None:                       # principal axes of the focused joint
            ax = _mask(panel.shape)
            wv, vv = np.linalg.eigh(cov[focus] * (scale ** 2))
            for i in range(2):
                d = vv[:, i] * 2.0 * np.sqrt(max(wv[i], 1e-9))
                _stamp(ax, lambda t, c=P[focus], dd=d: cv2.line(
                    t, _pt(c - dd), _pt(c + dd), 255, 1, cv2.LINE_AA, SH), 0.5 * dim)
            paint(panel, ax, style.C_OURS)

    # --- bones, staggered grow-in, drawn over the ellipses
    if p_skel > 0:
        bone = _mask(panel.shape)
        for i, (aa, bb) in enumerate(BONES):
            q = np.clip((p_skel - 0.030 * i) / 0.55, 0, 1)
            q = q * q * (3 - 2 * q)
            if q <= 0:
                continue
            pa, pb = P[aa], P[aa] + (P[bb] - P[aa]) * q
            _stamp(bone, lambda t, u=pa, v=pb: cv2.line(t, _pt(u), _pt(v), 255, lw,
                                                        cv2.LINE_AA, SH), dim)
        paint(panel, bone, style.C_PRED)

        core, ring = _mask(panel.shape), _mask(panel.shape)
        for j in range(13):
            q = np.clip((p_skel - 0.020 * j - 0.10) / 0.35, 0, 1)
            if q <= 0:
                continue
            r = int(round(dot * q * SC))
            _stamp(core, lambda t, c=P[j], rr=r: cv2.circle(t, _pt(c), rr, 255, -1,
                                                            cv2.LINE_AA, SH), dim)
            _stamp(ring, lambda t, c=P[j], rr=r: cv2.circle(t, _pt(c), rr, 255, max(2, lw - 1),
                                                            cv2.LINE_AA, SH), dim)
        paint(panel, core, style.BG)
        paint(panel, ring, style.C_PRED)


def frame_border(canvas: np.ndarray, x, y, w, h, colour: str, a: float = 1.0,
                 lw: int = 1) -> None:
    lay = canvas.copy()
    cv2.rectangle(lay, (int(x), int(y)), (int(x + w - 1), int(y + h - 1)), rgb(colour), lw,
                  cv2.LINE_AA)
    blend(canvas, lay, a)


def main() -> None:
    if "--prepare" in sys.argv:
        prepare()
        return

    a = common.shot_args(__doc__)
    z = np.load(CACHE, allow_pickle=True)
    kp, cov = z["kp"], z["cov"]
    rect, meta = z["rect"], [str(s) for s in z["meta"]]
    n_src = kp.shape[1]

    native = [cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1]
              for b in z["jpeg0"]], \
             [cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1]
              for b in z["jpeg1"]]

    scale = [PW / rect[c][2] for c in range(2)]
    origin = [rect[c][:2] for c in range(2)]

    # magnified joint: the left wrist -- the same joint s04 back-projects and lifts to 3D
    sig = np.sqrt(np.stack([cov[:, :, :, 0, 0], cov[:, :, :, 1, 1]], -1))
    MJ = 5
    # smoothed magnifier centre (native pixels) so the inset does not jitter
    ctr = kp[MAG_CAM, :, MJ].copy()
    k = np.ones(21) / 21.0
    ctr = np.stack([np.convolve(np.pad(ctr[:, i], 10, mode="edge"), k, "valid")
                    for i in range(2)], 1)
    czoom = CS / CARD_NATIVE

    bg = np.zeros((a.height, a.width, 3), np.uint8)
    bg[:] = rgb(style.BG)

    # ---- geometry of the figure -> footage move: the lit region box, and where it lands
    sxf = OV.FIG_W / 2100.0
    reg = [OV.FIG_X + OV.REGION_POSE[0] * sxf, OV.FIG_Y + OV.REGION_POSE[1] * sxf,
           OV.FIG_X + OV.REGION_POSE[2] * sxf, OV.FIG_Y + OV.REGION_POSE[3] * sxf]
    reg_c = np.array([(reg[0] + reg[2]) / 2.0, (reg[1] + reg[3]) / 2.0])
    pan_c = np.array([(PX[0] + PX[1] + PW) / 2.0, PY + PH / 2.0])

    # soft mask of the lit box: during the move everything around it is dissolved away, so the
    # box -- and nothing else -- is what grows into the two camera panels
    _m = np.zeros((a.height, a.width), np.float32)
    cv2.rectangle(_m, (int(reg[0]) - 18, int(reg[1]) - 18), (int(reg[2]) + 18, int(reg[3]) + 18),
                  1.0, -1)
    REG_MASK = cv2.GaussianBlur(_m, (0, 0), 16.0)[..., None]
    BGF = np.asarray(rgb(style.BG), np.float32)

    def isolate(img: np.ndarray, k: float) -> np.ndarray:
        """Fade everything outside the lit box towards the background (k: 0 = keep, 1 = gone)."""
        keep = REG_MASK + (1.0 - REG_MASK) * (1.0 - k)
        return np.clip(BGF + (img.astype(np.float32) - BGF) * keep, 0, 255).astype(np.uint8)

    def warp(img: np.ndarray, s: float, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        """Uniform scale by `s` mapping point `src` onto `dst`."""
        M = np.float32([[s, 0.0, dst[0] - s * src[0]], [0.0, s, dst[1] - s * src[1]]])
        return cv2.warpAffine(img, M, (a.width, a.height), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=rgb(style.BG))

    def live_frame(i0, i1, fr, p_skel, p_ell, p_card) -> np.ndarray:
        """The two camera panels + magnifier, without any type."""
        canvas = bg.copy()
        for c in range(2):
            img = cv2.addWeighted(native[c][i0], 1 - fr, native[c][i1], fr, 0)
            panel = cv2.resize(treat(img), (PW, PH), interpolation=cv2.INTER_CUBIC)
            kpi = kp[c, i0] * (1 - fr) + kp[c, i1] * fr
            cvi = cov[c, i0] * (1 - fr) + cov[c, i1] * fr
            draw_overlay(panel, kpi, cvi, scale[c], origin[c], p_skel, p_ell)
            x, y = PX[c], PY
            canvas[y:y + PH, x:x + PW] = panel
            frame_border(canvas, x, y, PW, PH, style.GRID, 0.9)

        if p_card > 0.004:
            cc = ctr[i0] * (1 - fr) + ctr[i1] * fr
            nx0 = cc[0] - CARD_NATIVE / 2
            ny0 = cc[1] - CARD_NATIVE / 2
            src = cv2.addWeighted(native[MAG_CAM][i0], 1 - fr,
                                  native[MAG_CAM][i1], fr, 0)
            # crop in native-crop coordinates
            sx = int(round(nx0 - rect[MAG_CAM][0]))
            sy = int(round(ny0 - rect[MAG_CAM][1]))
            sw = int(round(CARD_NATIVE))
            sx = int(np.clip(sx, 0, src.shape[1] - sw))
            sy = int(np.clip(sy, 0, src.shape[0] - sw))
            card = cv2.resize(treat(src[sy:sy + sw, sx:sx + sw], 0.30, 0.55),
                              (CS, CS), interpolation=cv2.INTER_LANCZOS4)
            o = np.array([rect[MAG_CAM][0] + sx, rect[MAG_CAM][1] + sy])
            kpi = kp[MAG_CAM, i0] * (1 - fr) + kp[MAG_CAM, i1] * fr
            cvi = cov[MAG_CAM, i0] * (1 - fr) + cov[MAG_CAM, i1] * fr
            draw_overlay(card, kpi, cvi, czoom, o, 1.0, 1.0,
                         dot=7.5, lw=5, ell_lw=3, focus=MJ)

            # fly-in: scale about the source rectangle on the panel
            s = 0.62 + 0.38 * p_card
            cw = max(8, int(round(CS * s)))
            cx = int(round(CX + CS / 2 - cw / 2))
            cy = int(round(CY + CS / 2 - cw / 2))
            small = cv2.resize(card, (cw, cw), interpolation=cv2.INTER_AREA)
            sub = canvas[cy:cy + cw, cx:cx + cw]
            blend(sub, small, p_card)
            frame_border(canvas, cx, cy, cw, cw, style.FG_MUTED, 0.55 * p_card)

            # source rectangle on the panel + two connectors
            rx = PX[MAG_CAM] + (o[0] - origin[MAG_CAM][0]) * scale[MAG_CAM]
            ry = PY + (o[1] - origin[MAG_CAM][1]) * scale[MAG_CAM]
            rw = CARD_NATIVE * scale[MAG_CAM]
            lay = canvas.copy()
            cv2.rectangle(lay, _pt((rx, ry)), _pt((rx + rw, ry + rw)),
                          rgb(style.FG_MUTED), 1, cv2.LINE_AA, SH)
            cv2.line(lay, _pt((rx + rw, ry)), _pt((cx, cy)),
                     rgb(style.GRID), 1, cv2.LINE_AA, SH)
            cv2.line(lay, _pt((rx + rw, ry + rw)), _pt((cx, cy + cw)),
                     rgb(style.GRID), 1, cv2.LINE_AA, SH)
            blend(canvas, lay, 0.75 * p_card)
        return canvas

    # beat 1 runs on the clock, not on a fraction of the shot (see the timeline block above)
    sq = min(1.0, 0.42 * a.duration / OV_END)
    S = lambda s: s * sq                                     # noqa: E731
    live0 = S(LIVE_T0)
    live_len = max(a.duration - live0, 1e-3)
    t_card = live0 + CARD_V0 * live_len                      # when the magnifier caption appears

    with common.FrameWriter(a) as w:
        for t in w.times():
            # ---- timeline (beat 1 in seconds, beat 2 as a fraction of what is left)
            p_rev = common.seg(t, 0.00, S(OV_REVEAL), "out")     # overview figure appears
            p_foc = common.seg(t, S(OV_LIGHT[0]), S(OV_LIGHT[1]), "smooth")   # stage lights up
            p_iso = common.seg(t, S(OV_ISO[0]), S(OV_ISO[1]), "smooth")   # surroundings dissolve
            tau = common.seg(t, S(OV_MOVE[0]), S(OV_MOVE[1]), "smooth")   # box grows into panels
            xf = common.seg(t, S(OV_XF), S(OV_MOVE[1]), "smooth")   # ... dissolving to footage
            v = float(np.clip((t - live0) / live_len, 0.0, 1.0))

            p_skel = common.seg(v, 0.03, 0.22, "smooth")     # skeleton draws in
            p_ell = common.seg(v, 0.20, 0.47, "smooth")      # ellipses bloom
            p_card = common.seg(v, CARD_V0, CARD_V1, "out")  # magnifier flies in

            # source clip plays across the live phase (slow motion), settling at the end
            fsrc = min(v / 0.88, 1.0) * (n_src - 1)  # clip settles ~0.6 s before the cut
            i0 = int(np.floor(fsrc))
            i1 = min(i0 + 1, n_src - 1)
            fr = fsrc - i0

            # ---- overview beat -> footage
            if xf >= 0.996:
                canvas = live_frame(i0, i1, fr, p_skel, p_ell, p_card)
            else:
                ov = np.array(OV.compose(OV.REGION_POSE, reveal=p_rev, focus=p_foc,
                                         accent=style.C_PRED,
                                         width=a.width, height=a.height))
                if p_iso > 0.001:
                    ov = isolate(ov, p_iso)
                if tau > 0.001:
                    ov = warp(ov, 1.0 + (OV_ZOOM - 1.0) * tau, reg_c,
                              reg_c + (pan_c - reg_c) * tau)
                canvas = ov
                if xf > 0.004:
                    live = live_frame(i0, i1, fr, p_skel, p_ell, p_card)
                    zl = LIVE_Z0 + (1.0 - LIVE_Z0) * tau
                    if zl < 0.999:
                        live = warp(live, zl, pan_c, pan_c)
                    blend(canvas, live, xf)

            # ---- type: fixed position, opacity-only entry (<= FADE s), never animated out
            text(canvas, (MARGIN, 62), "Perception", style.CAPTION_SIZE, style.FG_MUTED,
                 "Medium", anchor="ls", tracking=1.6,
                 alpha=common.seg(t, 0.00, FADE, "out"))
            text(canvas, (MARGIN, 152), "Two views, one covariance per joint",
                 style.H1_SIZE, style.FG, "Medium",
                 alpha=common.seg(t, 0.06, 0.06 + FADE, "out"))

            cap_a = common.seg(t, S(OV_END) - 0.06, S(OV_END) - 0.06 + FADE, "out")
            for c in range(2):
                text(canvas, (PX[c], CAP_Y), f"camera {c + 1}", style.CAPTION_SIZE,
                     style.FG_MUTED, alpha=cap_a)
            text(canvas, (1820, CAP_Y), f"H36M · {meta[0]} · {meta[1]}", style.CAPTION_SIZE,
                 style.FG_MUTED, alpha=0.55 * cap_a, anchor="rs")

            # the label is static; only the number it carries follows the clip (that is data)
            card_a = common.seg(t, t_card, t_card + FADE, "out")
            if card_a > 0.004:
                sx_, sy_ = sig[MAG_CAM, i0, MJ] * (1 - fr) + sig[MAG_CAM, i1, MJ] * fr
                text(canvas, (CX, CAP_Y), f"{JOINT_NAMES[MJ]} · 2σ",
                     style.CAPTION_SIZE, style.FG_MUTED, alpha=card_a)
                text(canvas, (CX, CAP_Y + 34),
                     f"σ = {sx_:.0f} × {sy_:.0f} px", style.CAPTION_SIZE,
                     style.C_OURS, alpha=card_a)

            w.write(canvas)


if __name__ == "__main__":
    main()
