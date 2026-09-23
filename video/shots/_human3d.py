"""A tiny painter's-algorithm 3-D rasteriser, shared by the two human-motion shots (s05, s09).

Private to those two shots.  Runs under the conda `chmp-video` interpreter (numpy + Pillow only,
no matplotlib 3-D axes, no OpenGL).  Everything is drawn back-to-front into a super-sampled float
canvas and box-downsampled, which gives clean silhouettes and lets spheres be *soft translucent
shells* rather than wireframes.

Primitives
    ground grid   depth-faded line segments on z = 0
    bones         capsules (thick AA segments with a cylindrical brightness ramp)
    joints        small solid discs
    spheres       radial-gradient discs: a faint interior (chord-length profile) plus a bright rim

Conventions
    world: H36M metres, +Z up, subject standing on z ~ 0
    camera: orbit around a target point, perspective pinhole
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import style

# --------------------------------------------------------------------------- type
_FONT_DIR = Path("/usr/share/fonts/opentype/inter")
_FONT_FILES = {
    "regular": _FONT_DIR / "Inter-Regular.otf",
    "medium": _FONT_DIR / "Inter-Medium.otf",
    "semibold": _FONT_DIR / "Inter-SemiBold.otf",
    "bold": _FONT_DIR / "Inter-Bold.otf",
}
_font_cache: dict = {}


def font(size: int, weight: str = "regular"):
    key = (size, weight)
    if key not in _font_cache:
        path = _FONT_FILES.get(weight, _FONT_FILES["regular"])
        if path.exists():
            _font_cache[key] = ImageFont.truetype(str(path), size)
        else:                                      # pragma: no cover - dev fallback
            _font_cache[key] = ImageFont.load_default(size)
    return _font_cache[key]


def hex2f(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], np.float32) / 255.0


def mix(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return a * (1.0 - t) + b * t


# --------------------------------------------------------------------------- camera
class Camera:
    """Perspective pinhole camera orbiting a target point."""

    def __init__(self, target, dist, azim_deg, elev_deg, w, h, fov_v_deg=34.0, shift=(0.0, 0.0)):
        self.target = np.asarray(target, np.float64)
        az, el = np.radians(azim_deg), np.radians(elev_deg)
        self.pos = self.target + dist * np.array(
            [np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        fwd = self.target - self.pos
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
        right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        self.R = np.stack([right, up, fwd])            # rows: view basis
        self.w, self.h = w, h
        self.f = 0.5 * h / np.tan(0.5 * np.radians(fov_v_deg))
        self.cx, self.cy = 0.5 * w + shift[0], 0.5 * h + shift[1]

    def view(self, p: np.ndarray) -> np.ndarray:
        """World -> view coords, [..., 3] (x right, y up, z forward)."""
        return (np.asarray(p, np.float64) - self.pos) @ self.R.T

    def project(self, p: np.ndarray):
        """World -> (screen xy [...,2], depth z [...])."""
        v = self.view(p)
        z = np.maximum(v[..., 2], 1e-3)
        sx = self.cx + self.f * v[..., 0] / z
        sy = self.cy - self.f * v[..., 1] / z
        return np.stack([sx, sy], -1), v[..., 2]

    def radius_px(self, z: float, r_world: float) -> float:
        z = float(max(z, r_world + 1e-3))
        return float(self.f * r_world / np.sqrt(max(z * z - r_world * r_world, 1e-6)))


# --------------------------------------------------------------------------- framing
def _bbox(target, dist, azims, elev, pts, radii, w, h, shift, fov):
    lo = np.array([np.inf, np.inf])
    hi = np.array([-np.inf, -np.inf])
    ok = True
    for az in azims:
        cam = Camera(target, dist, az, elev, w, h, fov, shift)
        xy, z = cam.project(pts)
        if np.min(z) <= 0.2:
            ok = False
            continue
        r = (cam.f * radii / np.maximum(z, 1e-3))[:, None]
        lo = np.minimum(lo, (xy - r).min(0))
        hi = np.maximum(hi, (xy + r).max(0))
    return lo, hi, ok


def fit_view(target, azims, elev, pts, radii, w, h, box, fov=34.0, iters=5,
             lo_d=1.0, hi_d=18.0):
    """Orbit distance + principal-point shift that centre `pts` (inflated by `radii`) in `box`.

    `box` is (x0, y0, x1, y1) in canvas pixels.  Returns (dist, shift).  The shift lets a tall,
    narrow subject sit off-centre (text rail on the left) without wasting vertical room.
    """
    box = np.asarray(box, float)
    bc = np.array([0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])])
    shift = np.zeros(2)
    dist = hi_d
    for _ in range(iters):
        a, b = lo_d, hi_d

        def fits(dd):
            lo, hi, ok = _bbox(target, dd, azims, elev, pts, radii, w, h, shift, fov)
            return ok and np.all(lo >= box[:2]) and np.all(hi <= box[2:])

        if fits(b):
            for _ in range(46):
                m = 0.5 * (a + b)
                if fits(m):
                    b = m
                else:
                    a = m
        dist = b
        lo, hi, _ = _bbox(target, dist, azims, elev, pts, radii, w, h, shift, fov)
        shift = shift + (bc - 0.5 * (lo + hi))     # re-centre, then re-fit against the new shift
    return float(dist), tuple(shift)


def motion_azimuth(target, pts_start, pts_end, elev, w, h):
    """Azimuth whose view axis is perpendicular to the travel direction, motion left -> right."""
    dxy = np.asarray(pts_end, float).mean(0)[:2] - np.asarray(pts_start, float).mean(0)[:2]
    if np.linalg.norm(dxy) < 1e-3:
        dxy = np.array([1.0, 0.0])
    base = np.degrees(np.arctan2(dxy[1], dxy[0]))
    best, best_dx = base + 90.0, -1e18
    for az in (base + 90.0, base - 90.0):
        cam = Camera(target, 4.0, az, elev, w, h)
        xy0, _ = cam.project(np.asarray(pts_start))
        xy1, _ = cam.project(np.asarray(pts_end))
        dx = float(xy1[:, 0].mean() - xy0[:, 0].mean())
        if dx > best_dx:
            best, best_dx = az, dx
    return best


# --------------------------------------------------------------------------- canvas
class Canvas:
    """Super-sampled float RGB canvas with alpha compositing and painter ordering."""

    def __init__(self, w, h, ss=2, bg=style.BG):
        self.ss = ss
        self.W, self.H = w * ss, h * ss
        self.out_w, self.out_h = w, h
        self.bg = hex2f(bg)
        self.buf = np.empty((self.H, self.W, 3), np.float32)
        self.clear()

    def clear(self):
        self.buf[:] = self.bg

    # -- low level -----------------------------------------------------------
    def _blend(self, x0, y0, alpha, colour):
        """alpha: [h,w] float; colour: [3] or [h,w,3]."""
        h, w = alpha.shape
        x1, y1 = x0 + w, y0 + h
        sx0, sy0 = max(0, x0), max(0, y0)
        sx1, sy1 = min(self.W, x1), min(self.H, y1)
        if sx1 <= sx0 or sy1 <= sy0:
            return
        a = alpha[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0][..., None]
        c = colour
        if isinstance(c, np.ndarray) and c.ndim == 3:
            c = c[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0]
        dst = self.buf[sy0:sy1, sx0:sx1]
        dst *= (1.0 - a)
        dst += a * c

    def to_rgb8(self) -> np.ndarray:
        s = self.ss
        b = self.buf.reshape(self.out_h, s, self.out_w, s, 3).mean(axis=(1, 3))
        return (np.clip(b, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


# --------------------------------------------------------------------------- primitives
class Scene:
    """Collects primitives with a view-space depth, then paints them far-to-near."""

    def __init__(self, cam: Camera, canvas: Canvas):
        self.cam, self.cv = cam, canvas
        self.items: list = []

    def add(self, depth: float, fn):
        self.items.append((float(depth), fn))

    def paint(self):
        for _, fn in sorted(self.items, key=lambda it: -it[0]):
            fn()
        self.items.clear()

    # -- spheres -------------------------------------------------------------
    def sphere(self, centre_world, r_world, colour, fill=0.10, rim=0.55,
               rim_width=0.10, depth_bias=0.0, min_alpha=0.004):
        if max(fill, rim) < min_alpha or r_world <= 1e-4:
            return                                   # invisible: skip before it costs anything
        (sx, sy), z = self.cam.project(np.asarray(centre_world, np.float64))
        if z <= 0.05:
            return
        R = self.cam.radius_px(z, r_world)
        if R < 0.6 or R > 6000:
            return
        col = np.asarray(colour, np.float32)
        rim_col = mix(col, np.ones(3, np.float32), 0.45)

        def draw(sx=sx, sy=sy, R=R, col=col, rim_col=rim_col):
            pad = 2
            x0, y0 = int(np.floor(sx - R - pad)), int(np.floor(sy - R - pad))
            n = int(np.ceil(2 * (R + pad))) + 1
            if x0 + n < 0 or y0 + n < 0 or x0 > self.cv.W or y0 > self.cv.H:
                return
            gx = np.arange(n, dtype=np.float32) + x0 - sx
            gy = np.arange(n, dtype=np.float32) + y0 - sy
            d = np.sqrt(gx[None, :] ** 2 + gy[:, None] ** 2)
            rho = d / R
            body = np.sqrt(np.clip(1.0 - rho * rho, 0.0, 1.0))
            a = fill * body ** 1.5
            a += rim * np.exp(-((1.0 - np.clip(rho, 0, 1)) / rim_width) ** 2) * (rho <= 1.0)
            # anti-aliased silhouette
            a *= np.clip((R - d) + 0.5, 0.0, 1.0)
            if a.max() <= 1e-4:
                return
            a = np.clip(a, 0.0, 0.92)
            t = (a / max(a.max(), 1e-6))[..., None]
            c = col[None, None, :] * (1.0 - t * 0.55) + rim_col[None, None, :] * (t * 0.55)
            self.cv._blend(x0, y0, a.astype(np.float32), c.astype(np.float32))

        self.add(z + depth_bias, draw)

    # -- capsules / bones ----------------------------------------------------
    def bone(self, p0, p1, colour, width_px, alpha=1.0, depth_bias=0.0):
        pts = np.stack([np.asarray(p0, np.float64), np.asarray(p1, np.float64)])
        (xy), z = self.cam.project(pts)
        if z.min() <= 0.05 or alpha <= 0.002:
            return
        a0, b0 = xy[0], xy[1]
        col = np.asarray(colour, np.float32)
        hot = mix(col, np.ones(3, np.float32), 0.35)

        def draw(a0=a0, b0=b0, col=col, hot=hot, w=float(width_px)):
            pad = w + 2
            x0 = int(np.floor(min(a0[0], b0[0]) - pad))
            y0 = int(np.floor(min(a0[1], b0[1]) - pad))
            x1 = int(np.ceil(max(a0[0], b0[0]) + pad))
            y1 = int(np.ceil(max(a0[1], b0[1]) + pad))
            if x1 < 0 or y1 < 0 or x0 > self.cv.W or y0 > self.cv.H:
                return
            nx, ny = x1 - x0, y1 - y0
            if nx <= 0 or ny <= 0 or nx * ny > 40_000_000:
                return
            gx = np.arange(nx, dtype=np.float32) + x0
            gy = np.arange(ny, dtype=np.float32) + y0
            px = gx[None, :] - a0[0]
            py = gy[:, None] - a0[1]
            ex, ey = b0[0] - a0[0], b0[1] - a0[1]
            ll = max(ex * ex + ey * ey, 1e-6)
            t = np.clip((px * ex + py * ey) / ll, 0.0, 1.0)
            dx = px - t * ex
            dy = py - t * ey
            d = np.sqrt(dx * dx + dy * dy)
            a = np.clip((w - d) + 0.5, 0.0, 1.0) * alpha
            if a.max() <= 1e-4:
                return
            shade = np.clip(1.0 - (d / max(w, 1e-3)) ** 2, 0.0, 1.0)[..., None]
            c = col[None, None, :] * (1.0 - shade * 0.5) + hot[None, None, :] * (shade * 0.5)
            self.cv._blend(x0, y0, a.astype(np.float32), c.astype(np.float32))

        self.add(float(z.mean()) + depth_bias, draw)

    def dot(self, centre_world, r_px, colour, alpha=1.0, depth_bias=-1e-3):
        (sx, sy), z = self.cam.project(np.asarray(centre_world, np.float64))
        if z <= 0.05 or alpha <= 0.002:
            return
        col = np.asarray(colour, np.float32)

        def draw(sx=sx, sy=sy, R=float(r_px), col=col):
            pad = 2
            x0, y0 = int(np.floor(sx - R - pad)), int(np.floor(sy - R - pad))
            n = int(np.ceil(2 * (R + pad))) + 1
            gx = np.arange(n, dtype=np.float32) + x0 - sx
            gy = np.arange(n, dtype=np.float32) + y0 - sy
            d = np.sqrt(gx[None, :] ** 2 + gy[:, None] ** 2)
            a = np.clip((R - d) + 0.5, 0.0, 1.0) * alpha
            if a.max() <= 1e-4:
                return
            self.cv._blend(x0, y0, a.astype(np.float32), col)

        self.add(z + depth_bias, draw)

    # -- skeleton convenience ------------------------------------------------
    def skeleton(self, pose, connections, colour, width_px, alpha=1.0, joint_px=0.0,
                 depth_bias=0.0):
        """`depth_bias` < 0 pulls the whole skeleton towards the camera in the painter order,
        which is how a pose stays readable *through* its own translucent spheres."""
        for i, j in connections:
            self.bone(pose[i], pose[j], colour, width_px, alpha, depth_bias)
        if joint_px > 0:
            for p in pose:
                self.dot(p, joint_px, mix(np.asarray(colour, np.float32),
                                          np.ones(3, np.float32), 0.35), alpha,
                         depth_bias=depth_bias - 1e-3)


# --------------------------------------------------------------------------- ground
def draw_ground(scene: Scene, centre_xy, half=3.0, step=0.5, z=0.0,
                colour=None, alpha=0.55, n_sub=24, fade=2.6):
    """A depth-faded grid on the floor plane, drawn behind everything else."""
    col = hex2f(style.GRID) if colour is None else colour
    col_hi = mix(col, hex2f(style.FG_MUTED), 0.35)
    cx, cy = float(centre_xy[0]), float(centre_xy[1])
    lines = []
    k = int(round(half / step))
    for i in range(-k, k + 1):
        u = i * step
        lines.append((np.array([cx + u, cy - half, z]), np.array([cx + u, cy + half, z]), i == 0))
        lines.append((np.array([cx - half, cy + u, z]), np.array([cx + half, cy + u, z]), i == 0))
    for p0, p1, axis in lines:
        ts = np.linspace(0.0, 1.0, n_sub + 1)
        for s0, s1 in zip(ts[:-1], ts[1:]):
            q0 = p0 + (p1 - p0) * s0
            q1 = p0 + (p1 - p0) * s1
            mid = 0.5 * (q0 + q1)
            r = np.hypot(mid[0] - cx, mid[1] - cy)
            w = np.clip(1.0 - (r / fade) ** 1.6, 0.0, 1.0)
            if w <= 0.02:
                continue
            c = col_hi if axis else col
            scene.bone(q0, q1, c, 1.6, alpha * w, depth_bias=+0.35)


# --------------------------------------------------------------------------- text layer
class Text:
    """Overlay text/rules drawn at output resolution on top of the rasterised 3-D view."""

    def __init__(self, w, h):
        self.img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        self.d = ImageDraw.Draw(self.img)

    def label(self, xy, s, size=26, weight="regular", colour=style.FG, alpha=1.0, anchor="la"):
        if alpha <= 0.004 or not s:
            return
        r, g, b = (int(colour.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
        self.d.text(xy, s, font=font(size, weight), fill=(r, g, b, int(255 * np.clip(alpha, 0, 1))),
                    anchor=anchor)

    def rule(self, xy, w, h=2, colour=style.GRID, alpha=1.0):
        if alpha <= 0.004:
            return
        r, g, b = (int(colour.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
        self.d.rectangle([xy[0], xy[1], xy[0] + w, xy[1] + h],
                         fill=(r, g, b, int(255 * np.clip(alpha, 0, 1))))

    def swatch(self, xy, colour, alpha=1.0, r=9):
        if alpha <= 0.004:
            return
        cr, cg, cb = (int(colour.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
        a = int(255 * np.clip(alpha, 0, 1))
        self.d.ellipse([xy[0] - r, xy[1] - r, xy[0] + r, xy[1] + r], outline=(cr, cg, cb, a),
                       width=2)
        self.d.ellipse([xy[0] - r + 3, xy[1] - r + 3, xy[0] + r - 3, xy[1] + r - 3],
                       fill=(cr, cg, cb, int(a * 0.35)))

    def over(self, rgb8: np.ndarray) -> np.ndarray:
        ov = np.asarray(self.img, np.float32) / 255.0
        a = ov[..., 3:4]
        return (np.clip(rgb8.astype(np.float32) / 255.0 * (1 - a) + ov[..., :3] * a, 0, 1)
                * 255.0 + 0.5).astype(np.uint8)


# --------------------------------------------------------------------------- data
def load_window(sid: str):
    from common import MEDIA
    p = MEDIA / f"{sid}_window.npz"
    if not p.exists():
        raise SystemExit(
            f"missing cache {p} -- run the prepare step first:\n"
            f"    JAX_PLATFORMS=cpu .venv/bin/python video/shots/_human3d_prepare.py")
    return np.load(p)
