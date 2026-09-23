"""s04 -- uncertainty-aware triangulation: two views -> one 3D skeleton of covariance ellipsoids.

Narration: "Uncertainty-aware triangulation lifts pose and covariance into three dimensions.
Every joint arrives as an ellipsoid."

Every number and every shape is real: the 2D means/covariances come from the repo's YOLO26
`Pose26` head, the 3D points and 3x3 covariances from
`utils.batched_transform_torch.triangulate_points_with_covariance_batched` (linear triangulation
+ first-order covariance propagation), the camera centres from
`models/pose_estimation/camera-parameters.json`.  Clip: H36M S11 / "Greeting" /
cameras 55011271 + 60457274, frames 1524..1607.  See `_pose_data.py`.

Cache (run once, repo venv, GPU):

    XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python video/shots/s04_triangulate.py --prepare

Render (offline, no model, no dataset):

    .venv/bin/python video/shots/s04_triangulate.py --out video/build/shots/s04.mp4 --duration 9.0
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

VIDEO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VIDEO_DIR))
import common  # noqa: E402
import style  # noqa: E402

CACHE = VIDEO_DIR / "media" / "s04_triangulate.npz"

BONES = [(0, 1), (0, 2), (1, 3), (3, 5), (2, 4), (4, 6), (1, 2), (1, 7), (2, 8),
         (7, 8), (7, 9), (9, 11), (8, 10), (10, 12)]
JOINT_NAMES = ['nose', 'left shoulder', 'right shoulder', 'left elbow', 'right elbow',
               'left wrist', 'right wrist', 'left hip', 'right hip', 'left knee',
               'right knee', 'left ankle', 'right ankle']
FOCUS = 5                       # left wrist -- the joint whose back-projected cones we draw

# ------------------------------------------------------------------ layout (1920x1080)
VW, VH = 260, 325               # small 2D view panels, stacked on the left
VX, VY = 100, (232, 587)
SCENE = (340, 110, 1440, 810)   # x, y, w, h of the 3D viewport
PANEL_SCALE = 0.75              # cached jpeg scale for the small panels

# virtual-camera orbit (world azimuth in degrees; the two real H36M cameras sit at -70 deg
# and -113 deg, so the sweep stays roughly broadside to them and the two ray fans separate)
AZ0, AZ1 = float(os.environ.get("S04_AZ0", 212.0)), float(os.environ.get("S04_AZ1", 264.0))
EL0, EL1 = 10.0, 17.0
DIST = 4.9

# ------------------------------------------------------------------ timing
# Everything graphical is a fraction of --duration (tuned at 9.0 s); every piece of *type*
# enters on opacity alone, over FADE seconds of wall clock, and never leaves (SPEC "Motion
# discipline").
FADE = 0.22                     # text entry, seconds
NUM_U = 0.60                    # when the ellipsoid read-out appears, as a fraction of the shot
RAY_A = 0.52                    # back-projection rays: one constant alpha, no fade-out

FONT_DIR = Path("/usr/share/fonts/opentype/inter")


# =============================================================================== prepare
def prepare() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import _pose_data as pd

    d = pd.run()
    rects = pd.crop_rects(d["kp"], d["cov"], aspect=VW / VH)
    jpegs = []
    for c in range(2):
        j, _ = pd.encode_jpegs(d["frames"][c], rects[c], scale=PANEL_SCALE, quality=84)
        jpegs.append(j)

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CACHE,
        jpeg0=np.array(jpegs[0], dtype=object), jpeg1=np.array(jpegs[1], dtype=object),
        rect=rects, scale=np.asarray([PANEL_SCALE]),
        kp=d["kp"], cov=d["cov"], p3=d["p3"], C3=d["C3"],
        cam_centers=d["cam_centers"], P=d["P"],
        meta=np.array([pd.SUBJECT, pd.ACTION, pd.CAMERAS[0], pd.CAMERAS[1],
                       str(pd.START_FRAME), str(pd.N_FRAMES)]),
        allow_pickle=True,
    )
    print(f"wrote {CACHE}  ({CACHE.stat().st_size / 1e6:.1f} MB)")
    C = d["C3"][:, FOCUS] / 1e6                       # mm^2 -> m^2
    ax = 2.0 * np.sqrt(np.maximum(np.linalg.eigvalsh(C), 0))
    print(f"  {JOINT_NAMES[FOCUS]} median 2-sigma axes: "
          f"{np.median(ax, 0).round(3)} m")


# ================================================================================ render
def rgb(h: str) -> tuple[int, int, int]:
    return common.hex2rgb(h)


_fonts: dict = {}


def font(size: int, weight: str = "Regular"):
    if (size, weight) not in _fonts:
        _fonts[(size, weight)] = ImageFont.truetype(
            str(FONT_DIR / f"Inter-{weight}.otf"), size)
    return _fonts[(size, weight)]


def text(canvas, xy, s, size, colour, weight="Regular", alpha=1.0, anchor="ls",
         tracking=0.0) -> None:
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


SH = 4
SC = 1 << SH


def _pt(p):
    return (int(round(float(p[0]) * SC)), int(round(float(p[1]) * SC)))


def paint(img: np.ndarray, mask: np.ndarray, colour: str) -> None:
    if mask.max() <= 0.004:
        return
    m = mask[..., None]
    c = np.asarray(rgb(colour), np.float32)
    img[:] = np.clip(img.astype(np.float32) * (1 - m) + c * m, 0, 255).astype(np.uint8)


class Layer:
    """Collect AA shapes with per-shape alpha, rasterise them once, composite once.

    Shapes are drawn in ascending alpha so overlaps take the larger value (a union, not an
    accumulation) -- one buffer per layer instead of one per shape.
    """

    def __init__(self, shape):
        self.shape = shape[:2]
        self.items: list = []

    def add(self, alpha: float, fn) -> None:
        if alpha > 0.004:
            self.items.append((min(alpha, 1.0), fn))

    def flush(self, img: np.ndarray, colour: str) -> None:
        if not self.items:
            return
        tmp = np.zeros(self.shape, np.uint8)
        for al, fn in sorted(self.items, key=lambda it: it[0]):
            fn(tmp, int(round(max(1.0, al * 255.0))))
        paint(img, tmp.astype(np.float32) * (1.0 / 255.0), colour)
        self.items = []


def treat(img, sat: float = 0.30, gain: float = 0.60):
    g = img.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
    out = img.astype(np.float32) * sat + g[..., None] * (1.0 - sat)
    out = out * gain * 0.90 + np.asarray(rgb(style.BG), np.float32) * 0.10
    return np.clip(out, 0, 255).astype(np.uint8)


def ellipse_pts(mu, cov, k: float, n: int = 72) -> np.ndarray:
    w, v = np.linalg.eigh(cov)
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return (np.stack([np.cos(th), np.sin(th)], 1)
            * (k * np.sqrt(np.maximum(w, 1e-12)))[None, :]) @ v.T + np.asarray(mu)[None, :]


# --------------------------------------------------------------------- 3D pinhole camera
class Cam:
    """Right-handed look-at pinhole; image y points down."""

    def __init__(self, eye, target, f, cx, cy, up=(0.0, 0.0, 1.0)):
        eye = np.asarray(eye, float)
        fwd = np.asarray(target, float) - eye
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, np.asarray(up, float))
        right /= np.linalg.norm(right)
        camup = np.cross(right, fwd)
        self.R = np.stack([right, -camup, fwd])         # world -> camera
        self.eye, self.f, self.cx, self.cy = eye, f, cx, cy

    def view(self, X):
        return (np.atleast_2d(X) - self.eye) @ self.R.T

    def project(self, X):
        Xc = self.view(X)
        z = np.maximum(Xc[:, 2], 1e-4)
        return np.stack([self.cx + self.f * Xc[:, 0] / z,
                         self.cy + self.f * Xc[:, 1] / z], 1), Xc[:, 2]


def sphere_dirs(nu: int = 14, nv: int = 24) -> np.ndarray:
    u = np.linspace(0, np.pi, nu)
    v = np.linspace(0, 2 * np.pi, nv, endpoint=False)
    su, cu = np.sin(u)[:, None], np.cos(u)[:, None]
    return np.stack([(su * np.cos(v)).ravel(), (su * np.sin(v)).ravel(),
                     (cu * np.ones_like(v)).ravel()], 1)


SPHERE = sphere_dirs()
RING_T = np.linspace(0, 2 * np.pi, 80)


def ellipsoid_geom(mu, C, k: float):
    """Return (surface points, three principal great-circle rings) of the k-sigma ellipsoid."""
    w, v = np.linalg.eigh(C)
    a = k * np.sqrt(np.maximum(w, 1e-14))
    surf = mu + (SPHERE * a) @ v.T
    rings = []
    ct, st = np.cos(RING_T), np.sin(RING_T)
    for i, j in ((0, 2), (1, 2)):          # the two sections through the major axis
        rings.append(mu + (np.outer(ct, v[:, i] * a[i]) + np.outer(st, v[:, j] * a[j])))
    return surf, rings


def backproject_cone(P, mu2, cov2, X, k: float, n: int = 48):
    """Exact back-projection of the k-sigma image ellipse to the depth plane of X.

    Returns (ring points on that plane, camera centre).
    """
    M, p4 = P[:, :3], P[:, 3]
    Minv = np.linalg.inv(M)
    C = -Minv @ p4                                   # camera centre
    nrm = Minv @ np.array([mu2[0], mu2[1], 1.0])     # principal ray direction
    nrm = nrm / np.linalg.norm(nrm)
    pts = ellipse_pts(mu2, cov2, k, n)
    d = (np.hstack([pts, np.ones((n, 1))]) @ Minv.T)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    t = ((X - C) @ nrm) / (d @ nrm)
    return C + d * t[:, None], C


def main() -> None:
    if "--prepare" in sys.argv:
        prepare()
        return

    a = common.shot_args(__doc__)
    z = np.load(CACHE, allow_pickle=True)
    kp, cov, rect = z["kp"], z["cov"], z["rect"]
    p3_mm = z["p3"]
    p3 = p3_mm / 1000.0                                    # mm -> m
    C3 = z["C3"] / 1e6                                     # mm^2 -> m^2
    P = z["P"]
    meta = [str(s) for s in z["meta"]]
    n_src = p3.shape[0]

    native = ([cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1]
               for b in z["jpeg0"]],
              [cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1]
               for b in z["jpeg1"]])

    # ---- fixed orbit centre / floor so the scene does not swim
    centre = p3.reshape(-1, 3).mean(0)
    feet = p3[:, [11, 12]].reshape(-1, 3).mean(0)
    sx, sy, sw, sh = SCENE
    cx, cy = sx + sw / 2, sy + sh / 2
    f3d = 1520.0
    cam_c = z["cam_centers"] / 1000.0

    # floor grid (0.5 m spacing, 5 m across) at z = 0
    g, step = 2.5, 0.5
    floor = []
    for t in np.arange(-g, g + 1e-6, step):
        floor.append(np.array([[feet[0] + t, feet[1] - g, 0.0],
                               [feet[0] + t, feet[1] + g, 0.0]]))
        floor.append(np.array([[feet[0] - g, feet[1] + t, 0.0],
                               [feet[0] + g, feet[1] + t, 0.0]]))

    bg = np.zeros((a.height, a.width, 3), np.uint8)
    bg[:] = rgb(style.BG)
    T_NUM = NUM_U * a.duration                  # the read-out appears here (seconds)
    scene_shape = (sh, sw, 3)
    off = np.array([sx, sy], float)

    with common.FrameWriter(a) as w:
        for t in w.times():
            u = t / a.duration
            canvas = bg.copy()
            a_type = common.seg(t, 0.00, FADE, "out")    # all type: opacity in, never out

            p_in = common.seg(u, 0.00, 0.06, "out")       # 2D views + floor fade up
            p_ray = common.seg(u, 0.02, 0.20, "smooth")   # rays reach towards the cameras
            p_skel = common.seg(u, 0.08, 0.28, "out")     # 3D skeleton
            p_cone = common.seg(u, 0.17, 0.38, "smooth")  # 2-sigma cones of the focus joint
            p_ell = common.seg(u, 0.32, 0.64, "smooth")   # ellipsoids bloom
            # Nothing recedes: the rays and the two cones stay up to the cut (SPEC "Motion
            # discipline" -- no exit animations), so the geometry the shot is about is on
            # screen for the whole second half.
            p_num = common.seg(t, T_NUM, T_NUM + FADE, "out")

            settle = min(u / 0.86, 1.0)                   # pose + orbit hold the last ~1.25 s
            fsrc = settle * (n_src - 1)
            i0 = int(np.floor(fsrc))
            i1 = min(i0 + 1, n_src - 1)
            fr = fsrc - i0
            lerp = lambda A: A[i0] * (1 - fr) + A[i1] * fr     # noqa: E731

            X = lerp(p3)
            Xmm = lerp(p3_mm)
            C = lerp(C3)

            # ------------------------------------------------------------- 3D viewport
            az = np.deg2rad(AZ0 + (AZ1 - AZ0) * common.ease(settle, "smoother"))
            el = np.deg2rad(EL0 + (EL1 - EL0) * common.ease(settle, "smooth"))
            eye = centre + DIST * np.array([np.cos(az) * np.cos(el),
                                           np.sin(az) * np.cos(el), np.sin(el)])
            cam = Cam(eye, centre, f3d, cx, cy)
            scene = np.zeros(scene_shape, np.uint8)
            scene[:] = rgb(style.BG)

            def proj(pts):
                q, zc = cam.project(np.atleast_2d(pts))
                return q - off, zc

            L = Layer(scene_shape)

            # ---- floor grid
            for seg3 in floor:
                q, zc = proj(seg3)
                if zc.min() < 0.2:
                    continue
                L.add(0.55 * p_in,
                      lambda t_, v, A=q[0], B=q[1]: cv2.line(t_, _pt(A), _pt(B), v, 1,
                                                             cv2.LINE_AA, SH))
            L.flush(scene, style.GRID)

            # ---- back-projection rays towards the two real camera centres
            if p_ray > 0.004:
                fade = RAY_A                              # constant: the rays never fade out
                for c in range(2):
                    for j in range(13):
                        d = cam_c[c] - X[j]
                        d /= np.linalg.norm(d)
                        q, zc = proj(X[j] + np.outer(np.linspace(0, 0.62 * p_ray, 9), d))
                        if zc.min() < 0.2:
                            continue
                        for k_ in range(8):
                            L.add((1.0 - k_ / 8.0) ** 2.6 * 0.52 * fade,
                                  lambda t_, v, A=q[k_], B=q[k_ + 1]: cv2.line(
                                      t_, _pt(A), _pt(B), v, 1, cv2.LINE_AA, SH))
                L.flush(scene, style.FG_MUTED)

            # ---- the two 2-sigma cones of the focus joint (exact back-projection)
            if p_cone > 0.004:
                fade = p_cone
                for c in range(2):
                    k2, c2 = lerp(kp[c]), lerp(cov[c])
                    ring_mm, Cc_mm = backproject_cone(P[c], k2[FOCUS], c2[FOCUS],
                                                      Xmm[FOCUS], 2.0)
                    ring, Cc = ring_mm / 1000.0, Cc_mm / 1000.0
                    q, zc = proj(ring)
                    if zc.min() > 0.2:
                        pts = np.round(q * SC).astype(np.int32)[None]
                        L.add(0.80 * fade,
                              lambda t_, v, p=pts: cv2.polylines(t_, p, True, v, 1,
                                                                 cv2.LINE_AA, SH))
                    d = Cc - X[FOCUS]
                    d /= np.linalg.norm(d)
                    for k_ in range(0, len(ring), 4):          # truncated cone generators
                        q2, zc2 = proj(ring[k_] + np.outer(np.linspace(0, 0.30, 5), d))
                        if zc2.min() < 0.2:
                            continue
                        for m_ in range(4):
                            L.add((1.0 - m_ / 4.0) ** 1.8 * 0.42 * fade,
                                  lambda t_, v, A=q2[m_], B=q2[m_ + 1]: cv2.line(
                                      t_, _pt(A), _pt(B), v, 1, cv2.LINE_AA, SH))
                L.flush(scene, style.C_OURS)

            # ---- bones
            if p_skel > 0.004:
                q, zc = proj(X)
                for i, (aa, bb) in enumerate(BONES):
                    pr = np.clip((p_skel - 0.035 * i) / 0.5, 0, 1)
                    pr = pr * pr * (3 - 2 * pr)
                    if pr <= 0 or min(zc[aa], zc[bb]) < 0.2:
                        continue
                    L.add(1.0, lambda t_, v, A=q[aa], B=q[aa] + (q[bb] - q[aa]) * pr:
                          cv2.line(t_, _pt(A), _pt(B), v, 3, cv2.LINE_AA, SH))
                L.flush(scene, style.C_PRED)

            # ---- covariance ellipsoids: translucent hull + three principal great circles
            if p_ell > 0.004:
                rings_all = []
                for j in range(13):
                    pr = np.clip((p_ell - 0.030 * j) / 0.55, 0, 1)
                    if pr <= 0:
                        continue
                    e = pr * pr * (3 - 2 * pr)
                    k = 2.0 * e * (1.0 + 0.12 * np.sin(np.pi * e))
                    surf, rings = ellipsoid_geom(X[j], C[j], k)
                    q, zc = proj(surf)
                    if zc.min() < 0.2:
                        continue
                    hi = 1.55 if j == FOCUS else 1.0
                    hull = cv2.convexHull(np.round(q * SC).astype(np.int32))
                    L.add(0.11 * e * hi, lambda t_, v, h=hull: cv2.fillConvexPoly(
                        t_, h, v, cv2.LINE_AA, SH))
                    for r in rings:
                        qr, zr = proj(r)
                        if zr.min() < 0.2:
                            continue
                        rings_all.append((0.52 * e * hi, 2 if j == FOCUS else 1,
                                          np.round(qr * SC).astype(np.int32)[None]))
                L.flush(scene, style.C_OURS)
                for al, lw, pts in rings_all:
                    L.add(al, lambda t_, v, p=pts, k=lw: cv2.polylines(
                        t_, p, True, v, k, cv2.LINE_AA, SH))
                L.flush(scene, style.C_OURS)

            # ---- joint markers
            if p_skel > 0.004:
                q, zc = proj(X)
                rad = []
                for j in range(13):
                    pr = np.clip((p_skel - 0.022 * j - 0.06) / 0.4, 0, 1)
                    rad.append(int(round(4.4 * pr * SC)))
                for j in range(13):
                    if zc[j] < 0.2 or rad[j] <= 0:
                        continue
                    L.add(1.0, lambda t_, v, A=q[j], rr=rad[j]: cv2.circle(
                        t_, _pt(A), rr, v, -1, cv2.LINE_AA, SH))
                L.flush(scene, style.BG)
                for j in range(13):
                    if zc[j] < 0.2 or rad[j] <= 0:
                        continue
                    L.add(1.0, lambda t_, v, A=q[j], rr=rad[j]: cv2.circle(
                        t_, _pt(A), rr, v, 2, cv2.LINE_AA, SH))
                L.flush(scene, style.C_PRED)

            canvas[sy:sy + sh, sx:sx + sw] = scene

            # ------------------------------------------------------------- the two views
            for c in range(2):
                img = cv2.addWeighted(native[c][i0], 1 - fr, native[c][i1], fr, 0)
                panel = cv2.resize(treat(img), (VW, VH), interpolation=cv2.INTER_AREA)
                s2 = VW / rect[c][2]
                k2 = (lerp(kp[c]) - rect[c][:2]) * s2
                c2 = lerp(cov[c]) * s2 ** 2
                PL = Layer(panel.shape)
                for j in range(13):
                    pts = np.round(ellipse_pts(k2[j], c2[j], 2.0, 48) * SC).astype(np.int32)
                    PL.add(0.15, lambda t_, v, p=pts[None]: cv2.fillPoly(t_, p, v,
                                                                         cv2.LINE_AA, SH))
                PL.flush(panel, style.C_OURS)
                for j in range(13):
                    pts = np.round(ellipse_pts(k2[j], c2[j], 2.0, 48) * SC).astype(np.int32)
                    PL.add(0.75, lambda t_, v, p=pts[None]: cv2.polylines(t_, p, True, v, 1,
                                                                          cv2.LINE_AA, SH))
                PL.flush(panel, style.C_OURS)
                for aa, bb in BONES:
                    PL.add(1.0, lambda t_, v, A=k2[aa], B=k2[bb]: cv2.line(
                        t_, _pt(A), _pt(B), v, 2, cv2.LINE_AA, SH))
                PL.flush(panel, style.C_PRED)

                y = VY[c]
                canvas[y:y + VH, VX:VX + VW] = (
                    panel.astype(np.float32) * p_in
                    + np.asarray(rgb(style.BG), np.float32) * (1 - p_in)).astype(np.uint8)
                lay = canvas.copy()
                cv2.rectangle(lay, (VX, y), (VX + VW - 1, y + VH - 1), rgb(style.GRID), 1,
                              cv2.LINE_AA)
                cv2.addWeighted(lay, 0.9 * p_in, canvas, 1 - 0.9 * p_in, 0, dst=canvas)
                text(canvas, (VX, y - 16), f"camera {c + 1}", style.CAPTION_SIZE,
                     style.FG_MUTED, alpha=a_type)

            # ------------------------------------------------------------- type
            text(canvas, (100, 62), "Perception", style.CAPTION_SIZE, style.FG_MUTED,
                 "Medium", anchor="ls", tracking=1.6, alpha=a_type)
            text(canvas, (100, 152), "Uncertainty-aware 3D triangulation", style.H1_SIZE,
                 style.FG, "Medium", alpha=common.seg(t, 0.06, 0.06 + FADE, "out"))

            # real 2-sigma semi-axes of the focus joint, in metres (from the cached C3)
            axes = 2.0 * np.sqrt(np.maximum(np.linalg.eigvalsh(C[FOCUS]), 0))[::-1]
            text(canvas, (460, 856), f"{JOINT_NAMES[FOCUS]} · 2σ ellipsoid",
                 style.CAPTION_SIZE, style.FG_MUTED, alpha=p_num)
            text(canvas, (460, 892),
                 f"{axes[0]:.2f} × {axes[1]:.2f} × {axes[2]:.2f} m",
                 style.CAPTION_SIZE, style.C_OURS, alpha=p_num)
            text(canvas, (1820, 892), f"H36M · {meta[0]} · {meta[1]}",
                 style.CAPTION_SIZE, style.FG_MUTED, alpha=0.5 * a_type, anchor="rs")

            w.write(canvas)


if __name__ == "__main__":
    main()
