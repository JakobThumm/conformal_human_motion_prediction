"""Tiny offline 3-D renderer shared by shots s11 / s12 / s13.

No GPU, no DISPLAY, no scene graph: a numpy ray-caster for the only two primitives the safety
shield knows about -- **spheres** (human joint occupancies, bounding spheres) and **capsules**
(robot links) -- plus a perspective ground grid and soft contact shadows.

Why hand-rolled: the whole film is capsules and spheres, analytic ray-primitive intersection is
exact (no tessellation seams on a 0.1 m capsule seen from 0.4 m), it is deterministic, and it has
no headless-GL failure mode.  Anti-aliasing is 2x supersampling of the primitive layer only, so
the ground grid stays crisp.

Layers are composited back to front:
  * ground   -- ray/plane, analytic anti-aliased grid, shadow texture, distance fog
  * "solid"  -- z-buffered opaque union (robot capsules, true human occupancy)
  * "glass"  -- z-buffered union rendered as one translucent shell (conformal sets, witness ball)

All geometry is in metres, z up, and every shot uses the same camera language (see CAM_* below).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

VIDEO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(VIDEO_DIR))
import style  # noqa: E402

MEDIA = VIDEO_DIR / "media"
CACHE = MEDIA / "s1x_shield_sim.npz"

FONT_DIR = Path("/usr/share/fonts/opentype/inter")
_FONTS: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}


def font(size: int, weight: str = "Regular"):
    key = (weight, int(size))
    if key not in _FONTS:
        # The film uses one sans family (Inter) everywhere.  A ticking counter still needs
        # non-jittering digits, which Inter's proportional figures do not give and which we
        # cannot get from OpenType `tnum` (this Pillow has no libraqm) -- so the counter is
        # drawn glyph-by-glyph on a fixed advance by `Overlay.tabular` instead of by switching
        # to a monospace face.  "Mono" is kept as an alias for the semibold weight.
        p = FONT_DIR / f"Inter-{'SemiBold' if weight == 'Mono' else weight}.otf"
        if not p.exists():
            p = FONT_DIR / "Inter-Regular.otf"
        _FONTS[key] = ImageFont.truetype(str(p), int(size))
    return _FONTS[key]


def rgb(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], np.float32) / 255.0


BG = rgb(style.BG)


# --------------------------------------------------------------------------- camera


class Camera:
    """Right-handed look-at camera.  Camera space: +x right, +y up, +z forward (depth)."""

    def __init__(self, eye, target, fov_y=32.0, width=style.WIDTH, height=style.HEIGHT,
                 roll=0.0, shift_x=0.0, shift_y=0.0):
        self.eye = np.asarray(eye, np.float64)
        self.target = np.asarray(target, np.float64)
        f = self.target - self.eye
        f /= np.linalg.norm(f)
        up = np.array([0.0, 0.0, 1.0])
        r = np.cross(f, up)
        r /= np.linalg.norm(r)
        u = np.cross(r, f)
        if roll:
            c, s = np.cos(roll), np.sin(roll)
            r, u = c * r + s * u, -s * r + c * u
        self.R = np.stack([r, u, f])                       # world -> camera (rows)
        self.w, self.h = int(width), int(height)
        self.focal = (self.h * 0.5) / np.tan(np.radians(fov_y) * 0.5)
        # principal-point offset, in 1x pixels: pure composition (shifts the frame without
        # tilting the camera, so the ground plane keeps its perspective).
        self.cx = self.w * 0.5 + float(shift_x)
        self.cy = self.h * 0.5 + float(shift_y)

    def to_cam(self, p):
        return (np.asarray(p, np.float64) - self.eye) @ self.R.T

    def project(self, p):
        """World points -> (sx, sy, depth)."""
        c = self.to_cam(p)
        z = np.maximum(c[..., 2], 1e-6)
        sx = self.cx + self.focal * c[..., 0] / z
        sy = self.cy - self.focal * c[..., 1] / z
        return sx, sy, c[..., 2]


def _scaled_camera(cam, s):
    """A copy of `cam` at `s` times the raster resolution (same view, fewer pixels)."""
    import copy
    c = copy.copy(cam)
    c.w, c.h = int(round(cam.w * s)), int(round(cam.h * s))
    c.focal, c.cx, c.cy = cam.focal * s, cam.cx * s, cam.cy * s
    return c


def orbit_eye(target, dist, az_deg, el_deg):
    """Camera position on a sphere around `target` (az measured from +x, towards +y)."""
    a, e = np.radians(az_deg), np.radians(el_deg)
    return np.asarray(target, np.float64) + dist * np.array(
        [np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])


# --------------------------------------------------------------------------- primitive layer


class Layer:
    """A bag of spheres and capsules rendered as one z-buffered union."""

    def __init__(self):
        self.sph_c: list = []
        self.sph_r: list = []
        self.sph_col: list = []
        self.cap_a: list = []
        self.cap_b: list = []
        self.cap_r: list = []
        self.cap_col: list = []

    def sphere(self, c, r, col):
        self.sph_c.append(np.asarray(c, np.float64))
        self.sph_r.append(float(r))
        self.sph_col.append(np.asarray(col, np.float32))

    def spheres(self, c, r, col):
        c = np.asarray(c, np.float64).reshape(-1, 3)
        r = np.broadcast_to(np.asarray(r, np.float64).ravel(), (c.shape[0],))
        col = np.asarray(col, np.float32)
        if col.ndim == 1:
            col = np.broadcast_to(col, (c.shape[0], 3))
        for i in range(c.shape[0]):
            if r[i] > 0:
                self.sphere(c[i], r[i], col[i])

    def capsule(self, a, b, r, col):
        self.cap_a.append(np.asarray(a, np.float64))
        self.cap_b.append(np.asarray(b, np.float64))
        self.cap_r.append(float(r))
        self.cap_col.append(np.asarray(col, np.float32))

    def capsules(self, a, b, r, col):
        a = np.asarray(a, np.float64).reshape(-1, 3)
        b = np.asarray(b, np.float64).reshape(-1, 3)
        r = np.asarray(r, np.float64).ravel()
        col = np.asarray(col, np.float32)
        if col.ndim == 1:
            col = np.broadcast_to(col, (a.shape[0], 3))
        for i in range(a.shape[0]):
            if r[i] > 0:
                self.capsule(a[i], b[i], r[i], col[i])

    def __len__(self):
        return len(self.sph_r) + len(self.cap_r)

    # -- footprints for the shadow texture -------------------------------------------------
    def footprints(self):
        pts, rad = [], []
        for c, r in zip(self.sph_c, self.sph_r):
            pts.append(c)
            rad.append(r)
        for a, b, r in zip(self.cap_a, self.cap_b, self.cap_r):
            pts.append(0.5 * (a + b))
            rad.append(r + 0.5 * float(np.linalg.norm(b - a)))
        if not pts:
            return np.zeros((0, 3)), np.zeros(0)
        return np.asarray(pts), np.asarray(rad)


LIGHT = np.array([-0.45, -0.75, 0.85])
LIGHT /= np.linalg.norm(LIGHT)
LIGHT2 = np.array([0.8, 0.35, 0.25])
LIGHT2 /= np.linalg.norm(LIGHT2)


def _shade(n, d_norm, base, ndl, ndl2, amb=0.30):
    """Lambert key + cool fill + rim, all in camera space.  Arrays are [...,3] / [...]."""
    rim = np.clip(1.0 - np.abs((n * d_norm).sum(-1)), 0.0, 1.0) ** 3
    lit = amb + 0.82 * np.clip(ndl, 0.0, 1.0) + 0.16 * np.clip(ndl2, 0.0, 1.0)
    out = base * lit[..., None]
    spec = np.clip(ndl, 0.0, 1.0) ** 26
    return out + 0.20 * rim[..., None] + 0.22 * spec[..., None]


def _cap_intersect(dx, dy, a3, b3, r, near):
    """Ray(0, (dx,dy,1)) vs capsule(a3,b3,r) in camera space -> (t, mask)."""
    dz = np.float32(1.0)
    ba = (b3 - a3).astype(np.float32)
    oa = (-a3).astype(np.float32)
    baba = np.float32(ba @ ba)
    bard = dx * ba[0] + dy * ba[1] + dz * ba[2]
    baoa = np.float32(ba @ oa)
    rdoa = dx * oa[0] + dy * oa[1] + dz * oa[2]
    dd2 = dx * dx + dy * dy + np.float32(1.0)
    aq = baba * dd2 - bard * bard
    bq = baba * rdoa - baoa * bard
    cq = baba * np.float32(oa @ oa) - baoa * baoa - np.float32(r * r) * baba
    hq = bq * bq - aq * cq
    hitq = hq > 0.0
    t = (-bq - np.sqrt(np.where(hitq, hq, 0.0))) / np.where(np.abs(aq) < 1e-12, 1e-12, aq)
    y = baoa + t * bard
    body = hitq & (y > 0.0) & (y < baba) & (t > near)
    pc = np.where((y <= 0.0)[..., None], oa[None, :], (-b3.astype(np.float32))[None, :])
    b2 = dx * pc[..., 0] + dy * pc[..., 1] + dz * pc[..., 2]
    c2 = (pc * pc).sum(-1) - np.float32(r * r)
    h2 = b2 * b2 - dd2 * c2
    capm = hitq & (~body) & (h2 > 0.0)
    t2 = (-b2 - np.sqrt(np.where(capm, h2, 0.0))) / dd2
    capm &= t2 > near
    return np.where(body, t, t2), (body | capm)


def render_layer(cam: Camera, layer: Layer, ss: int = 2, near: float = 0.05,
                 glass: bool = False, glass_col=None, glass_alpha: float = 0.5,
                 fog: float | None = None, zbuf_solid: np.ndarray | None = None,
                 budget: int = 2_000_000, need_depth: bool = True):
    """Z-buffered union of `layer`.

    Returns (color [h,w,3], alpha [h,w], depth [h,w]) at 1x; `depth` is the supersample-min
    camera-space depth (inf where empty) and can be passed back as `zbuf_solid` so a later
    (translucent) layer is occluded by this one.  `glass=True` shades the union's *front
    surface* as a single translucent shell, which keeps a union of overlapping spheres from
    looking like a pile of bubbles rather than one occupancy volume.

    Implementation: each primitive emits (pixel, depth, primitive id) for the samples it
    covers -- no shading, no normals.  One descending-depth scatter resolves the frame, and
    only the ~1 surface sample per pixel that survives is shaded, so a thousand tiny spheres
    cost about what their screen area costs.
    """
    H, W = cam.h * ss, cam.w * ss
    f = np.float32(cam.focal * ss)
    CX, CY = np.float32(cam.cx * ss), np.float32(cam.cy * ss)
    near = np.float32(near)
    nsph = len(layer.sph_r)
    ci, ct, cp = [], [], []

    def emit(idx, t, pid):
        if idx.size:
            ci.append(idx.astype(np.int64))
            ct.append(t.astype(np.float32))
            cp.append(pid.astype(np.int32))

    # ---- spheres: bucketed by screen size, fully vectorised ------------------------------
    if nsph:
        C = cam.to_cam(np.asarray(layer.sph_c)).astype(np.float32)
        Rr = np.asarray(layer.sph_r, np.float32)
        cz = C[:, 2]
        ok = cz > near + Rr
        denom = np.sqrt(np.maximum(cz * cz - Rr * Rr, 1e-9))
        rs = f * Rr / denom * 1.05
        cx = CX + f * C[:, 0] / np.maximum(cz, 1e-6)
        cy = CY - f * C[:, 1] / np.maximum(cz, 1e-6)
        ok &= (cx + rs > 0) & (cx - rs < W) & (cy + rs > 0) & (cy - rs < H)
        sz = np.where(ok, 2 * np.ceil(rs) + 2, 0)
        bucket = np.where(ok, 4 * np.ceil(np.maximum(sz, 4) / 4).astype(np.int64), 0)
        q = (C * C).sum(1) - Rr * Rr
        for S in np.unique(bucket[ok]):
            sel = np.flatnonzero(bucket == S)
            S = int(S)
            off = (np.arange(S) - S // 2).astype(np.int64)
            step = max(1, budget // (S * S))
            for s0 in range(0, sel.size, step):
                k = sel[s0: s0 + step]
                px = np.rint(cx[k]).astype(np.int64)[:, None, None] + off[None, None, :]
                py = np.rint(cy[k]).astype(np.int64)[:, None, None] + off[None, :, None]
                px, py = np.broadcast_arrays(px, py)
                m = (px >= 0) & (px < W) & (py >= 0) & (py < H)
                dx = (px + np.float32(0.5) - CX).astype(np.float32) / f
                dy = -(py + np.float32(0.5) - CY).astype(np.float32) / f
                Ck = C[k][:, None, None, :]
                a = dx * dx + dy * dy + np.float32(1.0)
                b = -2.0 * (dx * Ck[..., 0] + dy * Ck[..., 1] + Ck[..., 2])
                disc = b * b - 4.0 * a * q[k][:, None, None]
                m &= disc > 0.0
                t = (-b - np.sqrt(np.where(m, disc, 0.0))) / (2.0 * a)
                m &= t > near
                if not m.any():
                    continue
                pid = np.broadcast_to(k[:, None, None], m.shape)
                emit((py * W + px)[m], t[m], pid[m])

    # ---- capsules: few, so a per-primitive bbox is fine ----------------------------------
    if layer.cap_r:
        A = cam.to_cam(np.asarray(layer.cap_a)).astype(np.float32)
        B = cam.to_cam(np.asarray(layer.cap_b)).astype(np.float32)
        Rr = np.asarray(layer.cap_r, np.float32)
        for i in range(len(Rr)):
            r = float(Rr[i])
            a3, b3 = A[i], B[i]
            if max(a3[2], b3[2]) <= near + r:
                continue
            sx, sy = [], []
            for p in (a3, b3):
                z = max(float(p[2]), float(near) + r * 0.5)
                dd = np.sqrt(max(z * z - r * r, 1e-9))
                rr = f * r / dd * 1.05
                sx += [CX + f * p[0] / z - rr, CX + f * p[0] / z + rr]
                sy += [CY - f * p[1] / z - rr, CY - f * p[1] / z + rr]
            x0 = max(int(np.floor(min(sx))) - 1, 0)
            x1 = min(int(np.ceil(max(sx))) + 2, W)
            y0 = max(int(np.floor(min(sy))) - 1, 0)
            y1 = min(int(np.ceil(max(sy))) + 2, H)
            if x1 <= x0 or y1 <= y0:
                continue
            ix = np.arange(x0, x1, dtype=np.int64)
            iy = np.arange(y0, y1, dtype=np.int64)
            dx, dy = np.broadcast_arrays(
                ((ix + 0.5 - CX).astype(np.float32) / f)[None, :],
                (-(iy + 0.5 - CY).astype(np.float32) / f)[:, None])
            tt, m = _cap_intersect(dx, dy, a3, b3, r, near)
            if not m.any():
                continue
            gidx = (iy[:, None] * W) + ix[None, :]
            emit(gidx[m], tt[m], np.full(int(m.sum()), nsph + i, np.int32))

    empty = (np.zeros((cam.h, cam.w, 3), np.float32), np.zeros((cam.h, cam.w), np.float32),
             np.full((cam.h, cam.w), np.inf, np.float32))
    if not ci:
        return empty
    idx = np.concatenate(ci)
    tt = np.concatenate(ct)
    pid = np.concatenate(cp)
    if zbuf_solid is not None:
        occl = np.repeat(np.repeat(zbuf_solid, ss, 0), ss, 1).ravel()
        keep = tt < occl[idx]
        idx, tt, pid = idx[keep], tt[keep], pid[keep]
        if idx.size == 0:
            return empty

    # depth resolve: write candidates far-to-near, so the nearest one survives
    order = np.ascontiguousarray(np.argsort(tt)[::-1])
    winner = np.full(H * W, -1, np.int64)
    winner[idx[order]] = order
    pix = np.flatnonzero(winner >= 0)
    if pix.size == 0:
        return empty
    sel = winner[pix]
    zs = tt[sel]
    ps = pid[sel]
    iy, ix = pix // W, pix % W
    gx = (ix + 0.5 - CX).astype(np.float32) / f
    gy = -(iy + 0.5 - CY).astype(np.float32) / f

    # ---- normals + base colour, recomputed for the surviving samples only ----------------
    n = np.empty((pix.size, 3), np.float32)
    colb = np.empty((pix.size, 3), np.float32)
    ms = ps < nsph
    if ms.any():
        k = ps[ms]
        Ck = C[k]
        t = zs[ms]
        rk = np.asarray(layer.sph_r, np.float32)[k]
        n[ms] = np.stack([(t * gx[ms] - Ck[:, 0]) / rk, (t * gy[ms] - Ck[:, 1]) / rk,
                          (t - Ck[:, 2]) / rk], -1)
        colb[ms] = np.asarray(layer.sph_col, np.float32)[k]
    mc = ~ms
    if mc.any():
        k = ps[mc] - nsph
        a3 = A[k]
        ba = B[k] - a3
        t = zs[mc]
        baba = (ba * ba).sum(1)
        rk = np.asarray(layer.cap_r, np.float32)[k]
        pxw = np.stack([t * gx[mc], t * gy[mc], t], -1)
        yb = np.clip(((pxw - a3) * ba).sum(1) / np.maximum(baba, 1e-12), 0.0, 1.0)
        n[mc] = (pxw - a3 - ba * yb[:, None]) / rk[:, None]
        colb[mc] = np.asarray(layer.cap_col, np.float32)[k]

    dmag = np.sqrt(gx * gx + gy * gy + 1.0)
    dn = np.stack([gx / dmag, gy / dmag, 1.0 / dmag], -1).astype(np.float32)
    lc = (LIGHT @ cam.R.T).astype(np.float32)
    lc2 = (LIGHT2 @ cam.R.T).astype(np.float32)
    ndl, ndl2 = n @ lc, n @ lc2
    if glass:
        base = colb if glass_col is None else np.asarray(glass_col, np.float32)
        fres = np.clip(1.0 - np.abs((n * dn).sum(-1)), 0.0, 1.0)
        alpha_s = np.clip(glass_alpha * (0.22 + 0.95 * fres ** 1.7), 0.0, 0.97).astype(np.float32)
        col_s = (base * (0.78 + 0.5 * np.clip(ndl, 0.0, 1.0))[:, None]
                 + 0.5 * (fres ** 3)[:, None])
    else:
        col_s = _shade(n, dn, colb, ndl, ndl2)
        alpha_s = np.ones(pix.size, np.float32)
    if fog:
        fk = 1.0 - np.exp(-zs / fog)
        col_s = col_s * (1.0 - fk[:, None]) + BG * fk[:, None]
    col_s = np.clip(col_s, 0.0, 1.6)

    npix = cam.h * cam.w
    dst = (iy // ss) * cam.w + (ix // ss)
    inv = 1.0 / (ss * ss)
    al = np.bincount(dst, alpha_s, npix).astype(np.float32) * inv
    acc = np.stack([np.bincount(dst, col_s[:, k] * alpha_s, npix) for k in range(3)], -1)
    out = (acc * inv) / np.maximum(al, 1e-6)[:, None]
    zmin = np.full(npix, np.inf, np.float32)
    if need_depth:
        o2 = np.ascontiguousarray(np.argsort(zs)[::-1])
        zmin[dst[o2]] = zs[o2]
    return (out.reshape(cam.h, cam.w, 3).astype(np.float32), al.reshape(cam.h, cam.w),
            zmin.reshape(cam.h, cam.w))


# --------------------------------------------------------------------------- ground


def shadow_texture(pts, rad, extent, res=384, strength=1.0):
    """Top-down soft-shadow accumulation over [-extent, extent]^2."""
    import cv2
    tex = np.zeros((res, res), np.float32)
    if len(pts) == 0:
        return tex
    pts = np.asarray(pts)
    rad = np.asarray(rad)
    u = (pts[:, 0] + extent) / (2 * extent) * (res - 1)
    v = (pts[:, 1] + extent) / (2 * extent) * (res - 1)
    ok = (u >= 0) & (u < res) & (v >= 0) & (v < res)
    w = np.clip(rad, 0.02, 3.0) ** 2 * (res / (2 * extent)) ** 2
    np.add.at(tex, (v[ok].astype(int), u[ok].astype(int)), w[ok] * strength)
    k = max(3, int(res / (2 * extent) * 0.55) | 1)
    tex = cv2.GaussianBlur(tex, (k, k), 0)
    return np.clip(tex, 0.0, 1.0)


def render_ground(cam: Camera, z=0.0, grid=1.0, radius=11.0, fog=26.0,
                  shadow=None, shadow_extent=12.0, base=None, grid_col=None,
                  grid_gain=1.0, rings=(), glow=None, return_depth=False):
    """Perspective ground plane with an analytically anti-aliased grid.  Returns [h,w,3]."""
    H, W = cam.h, cam.w
    f = cam.focal
    xs = (np.arange(W, dtype=np.float32) + 0.5 - np.float32(cam.cx)) / f
    ys = -(np.arange(H, dtype=np.float32) + 0.5 - np.float32(cam.cy)) / f
    d = np.stack([np.broadcast_to(xs[None, :], (H, W)),
                  np.broadcast_to(ys[:, None], (H, W)),
                  np.ones((H, W), np.float32)], -1)
    dw = d @ cam.R                                   # camera -> world
    t = (z - cam.eye[2]) / np.where(np.abs(dw[..., 2]) < 1e-9, 1e-9, dw[..., 2])
    vis = (t > 0) & (dw[..., 2] < 0)
    t = np.where(vis, t, 0.0)
    X = cam.eye[0] + t * dw[..., 0]
    Y = cam.eye[1] + t * dw[..., 1]
    depth = t                                        # camera-space z (d_z == 1)

    world_px = np.maximum(depth / f, 1e-6)
    img = np.broadcast_to(BG, (H, W, 3)).copy()
    gcol = rgb(style.GRID) * 2.35 if grid_col is None else np.asarray(grid_col, np.float32)
    bcol = rgb("#12161F") if base is None else np.asarray(base, np.float32)

    rr = np.sqrt(X * X + Y * Y)
    inside = vis & (rr < radius * 1.35)
    if not inside.any():
        return (img, np.full((H, W), np.inf, np.float32)) if return_depth else img
    fade = np.clip((radius * 1.28 - rr) / (radius * 0.45), 0.0, 1.0)
    fogk = np.exp(-depth / fog)
    g = np.zeros((H, W), np.float32)
    for period, gain in ((grid, 0.55), (grid * 5.0, 1.0)):
        u = X / period
        v = Y / period
        wpx = world_px / period * 1.0
        du = np.abs(np.mod(u + 0.5, 1.0) - 0.5) / np.maximum(wpx, 1e-6)
        dv = np.abs(np.mod(v + 0.5, 1.0) - 0.5) / np.maximum(wpx, 1e-6)
        g = np.maximum(g, gain * np.clip(1.0 - np.minimum(du, dv) / 1.1, 0.0, 1.0))
    g = g * fade * fogk * grid_gain
    floor = bcol[None, None, :] * (0.35 + 0.65 * fogk)[..., None]
    sh = 0.0
    if shadow is not None:
        res = shadow.shape[0]
        su = np.clip(((X + shadow_extent) / (2 * shadow_extent) * (res - 1)), 0, res - 1)
        sv = np.clip(((Y + shadow_extent) / (2 * shadow_extent) * (res - 1)), 0, res - 1)
        sh = shadow[sv.astype(np.int32), su.astype(np.int32)] * fade
    floor = floor * (1.0 - 0.75 * np.asarray(sh)[..., None])
    gnd = floor * (1.0 - g[..., None]) + gcol[None, None, :] * g[..., None]
    if glow is not None:                        # (radius m, colour, strength) -- a pool of light
        gr, gcol2, gstr = glow
        k = np.clip(1.0 - (rr / gr) ** 2, 0.0, 1.0) ** 1.6 * gstr * fade
        gnd = gnd + np.asarray(gcol2, np.float32)[None, None, :] * k[..., None]
    for rr_, rcol, rw, ra in rings:            # (radius m, colour, line width m, alpha)
        if ra <= 0.003:
            continue
        e = np.abs(rr - rr_) / np.maximum(np.maximum(world_px, 1e-6) * 1.2, rw * 0.5)
        k = np.clip(1.0 - e, 0.0, 1.0) * ra * fogk * fade
        gnd = gnd * (1.0 - k[..., None]) + np.asarray(rcol, np.float32)[None, None, :] * k[..., None]
    gnd = gnd * fade[..., None] + BG * (1.0 - fade)[..., None]
    img[inside] = gnd[inside]
    img = np.clip(img, 0.0, 1.0)
    if return_depth:
        return img, np.where(inside, depth, np.float32(np.inf)).astype(np.float32)
    return img


def render_glass(cam: Camera, layer: Layer, occluder=None, down: int = 3, ss: int = 1,
                 col=None, alpha: float = 0.32):
    """Translucent union, rasterised at `scale` resolution and upsampled.

    A 0.4 m sphere seen from 3 m covers a quarter of the frame, so a full-resolution glass pass
    is the one thing in this renderer that is genuinely expensive -- and the least in need of
    resolution, because its edges are a soft fresnel ramp anyway.  `col=None` shades each
    primitive in its own colour, so a grey volume and a blue volume can share one depth-resolved
    pass (which is what makes their front surfaces read as two separate solids).
    """
    import cv2
    if len(layer) == 0:
        return np.zeros((cam.h, cam.w, 3), np.float32), np.zeros((cam.h, cam.w), np.float32)
    k = max(1, int(down))
    assert cam.w % k == 0 and cam.h % k == 0, "down must divide the canvas"
    small = _scaled_camera(cam, 1.0 / k)
    occ = None
    if occluder is not None:
        occ = occluder.reshape(small.h, k, small.w, k).min((1, 3))
    c, al, _ = render_layer(small, layer, ss=ss, glass=True, glass_col=col,
                            glass_alpha=alpha, zbuf_solid=occ, need_depth=False)
    if (small.w, small.h) != (cam.w, cam.h):
        c = cv2.resize(c, (cam.w, cam.h), interpolation=cv2.INTER_CUBIC)
        al = cv2.resize(al, (cam.w, cam.h), interpolation=cv2.INTER_CUBIC)
        b = 2 * k + 1
        c = cv2.GaussianBlur(c, (b, b), 0)
        al = np.clip(cv2.GaussianBlur(al, (b, b), 0), 0.0, 1.0)
    return c, al


# --------------------------------------------------------------------------- composite


def over(bg, col, alpha):
    a = alpha[..., None]
    return bg * (1.0 - a) + col * a


def to_u8(img):
    return np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- maths type
# The film uses ONE sans family (Inter).  Maths is the single exception SPEC "Type" allows, and
# only for symbols -- here PFH_D, which needs a real subscript.  Set in Computer Modern through
# matplotlib's mathtext and cap-height-matched to the Inter line it sits in, so the glyph shares
# that line's baseline instead of being nudged by eye.  (Same technique as s05_motion.py.)
_MATH: dict = {}


def _math_raster(expr: str, dpi: float):
    """(coverage [h,w] in 0..1, baseline row from the top) for `expr` set in CM at `dpi`."""
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["mathtext.fontset"] = "cm"
    from matplotlib.font_manager import FontProperties
    from matplotlib.mathtext import MathTextParser
    r = MathTextParser("agg").parse(expr, dpi=dpi, prop=FontProperties(size=40))
    return np.asarray(r.image, np.float32) / 255.0, r.height - r.depth


def math_glyph(expr: str, size: int, weight: str = "Regular"):
    """(coverage [h,w], baseline-from-top) for `expr`, cap-height-matched to Inter `size`."""
    key = (expr, int(size), weight)
    if key not in _MATH:
        bb = font(size, weight).getbbox("K")             # (x0, cap_top, x1, baseline)
        cap_h = bb[3] - bb[1]
        # CM "K" at a reference dpi -> the dpi that renders it ~4x oversampled, then the exact
        # scale measured at *that* dpi rather than extrapolated.
        a0, base0 = _math_raster(r"$K$", 200.0)
        dpi = 200.0 * (4.0 * cap_h) / (base0 - np.nonzero(a0.max(1) > 0.03)[0][0])
        a1, base1 = _math_raster(r"$K$", dpi)
        scale = cap_h / (base1 - np.nonzero(a1.max(1) > 0.03)[0][0])

        a, base = _math_raster(expr, dpi)
        cols = np.nonzero(a.max(0) > 0.03)[0]
        a = a[:, cols[0]:cols[-1] + 1]                   # crop the side bearings
        im = Image.fromarray((a * 255.0 + 0.5).astype(np.uint8), "L")
        w = max(1, int(round(im.width * scale)))
        h = max(1, int(round(im.height * scale)))
        cov = np.asarray(im.resize((w, h), Image.LANCZOS), np.float32) / 255.0
        _MATH[key] = (cov, base * h / a.shape[0])
    return _MATH[key]


# --------------------------------------------------------------------------- 2-D overlay


class Overlay:
    """PIL text/vector overlay on top of a uint8 RGB frame."""

    def __init__(self, frame_u8):
        self.im = Image.fromarray(frame_u8)
        self.d = ImageDraw.Draw(self.im, "RGBA")

    @staticmethod
    def _c(col, a=1.0):
        if isinstance(col, str):
            col = tuple(int(col.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
        else:
            col = tuple(int(round(255 * v)) for v in np.asarray(col).ravel()[:3])
        return col + (int(round(255 * np.clip(a, 0, 1))),)

    def text(self, xy, s, size=32, col=style.FG, weight="Regular", a=1.0, anchor="la",
             spacing=6):
        if a <= 0.003:
            return
        self.d.multiline_text(xy, s, font=font(size, weight), fill=self._c(col, a),
                              anchor=anchor if "\n" not in s else None, spacing=spacing)

    def advance(self, s, size=32, weight="Regular"):
        """Pen advance of `s` (unlike `measure`, includes the side bearings and trailing spaces),
        so runs of Inter and maths can be laid out on one line without gaps drifting."""
        return float(self.d.textlength(s, font=font(size, weight)))

    def math(self, xy, expr, size=32, col=style.FG, weight="Regular", a=1.0):
        """Draw the maths `expr` (Computer Modern) on the Inter baseline of a line at `xy`.

        Returns the glyph's advance so an Inter run can continue on the same line.
        """
        cov, base = math_glyph(expr, size, weight)
        w, h = cov.shape[1], cov.shape[0]
        if a > 0.003:
            baseline = xy[1] + font(size, weight).getbbox("K")[3]   # Inter's own baseline
            r, g, b, _ = self._c(col, 1.0)
            mask = Image.fromarray(
                np.clip(cov * float(np.clip(a, 0, 1)) * 255.0 + 0.5, 0, 255).astype(np.uint8), "L")
            self.im.paste(Image.new("RGB", (w, h), (r, g, b)),
                          (int(round(xy[0])), int(round(baseline - base))), mask)
        return float(w)

    def tabular(self, xy, s, size=32, col=style.FG, weight="SemiBold", a=1.0, anchor="la"):
        """Draw a number with fixed digit advance, so a counter does not jitter.

        Inter's default figures are proportional (a "1" is 2/3 the width of a "4"), which makes
        an animated counter shimmer.  Every digit is placed on the widest-digit advance; the
        separators keep their natural width.
        """
        if a <= 0.003:
            return
        f = font(size, weight)
        adv = max(self.d.textlength(c, font=f) for c in "0123456789")
        widths = [adv if c.isdigit() else self.d.textlength(c, font=f) for c in s]
        x = xy[0] - (sum(widths) if anchor[0] == "r" else
                     sum(widths) / 2 if anchor[0] == "m" else 0.0)
        for c, w in zip(s, widths):
            self.d.text((x + (w - self.d.textlength(c, font=f)) / 2, xy[1]), c, font=f,
                        fill=self._c(col, a), anchor="l" + anchor[1])
            x += w

    def line(self, p0, p1, col=style.FG_MUTED, a=1.0, w=2):
        if a <= 0.003:
            return
        self.d.line([tuple(p0), tuple(p1)], fill=self._c(col, a), width=w)

    def circle(self, c, r, col=style.FG, a=1.0, w=2, fill=None, fill_a=1.0):
        if a <= 0.003 and fill is None:
            return
        box = [c[0] - r, c[1] - r, c[0] + r, c[1] + r]
        self.d.ellipse(box, outline=self._c(col, a), width=w,
                       fill=None if fill is None else self._c(fill, fill_a))

    def rect(self, box, col=None, a=1.0, fill=None, fill_a=1.0, w=2, radius=0):
        self.d.rounded_rectangle(box, radius=radius, width=w,
                                 outline=None if col is None else self._c(col, a),
                                 fill=None if fill is None else self._c(fill, fill_a))

    def measure(self, s, size=32, weight="Regular"):
        b = self.d.textbbox((0, 0), s, font=font(size, weight))
        return b[2] - b[0], b[3] - b[1]

    def out(self):
        return np.asarray(self.im)


# --------------------------------------------------------------------------- shared scene bits

# H36M 13-joint skeleton (JOINT_NAMES_13 order, see motion_prediction/h36m_settings.py):
# Nose, LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist, LHip, RHip, LKnee, RKnee,
# LAnkle, RAnkle
BONES = [(0, 1), (0, 2), (1, 2), (1, 3), (3, 5), (2, 4), (4, 6),
         (1, 7), (2, 8), (7, 8), (7, 9), (9, 11), (8, 10), (10, 12)]


def add_skeleton(layer, pts, col, r=0.022, joint_r=None):
    """Thin bone capsules over the 13 H36M joints, so an occupancy blob still reads as a body."""
    pts = np.asarray(pts, np.float64).reshape(-1, 3)
    for i, k in BONES:
        layer.capsule(pts[i], pts[k], r, col)
    if joint_r:
        for p in pts:
            layer.sphere(p, joint_r, col)


def load_cache():
    if not CACHE.exists():
        raise SystemExit(
            f"missing {CACHE}\nrun the prepare step first:\n"
            f"  XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python "
            f"video/shots/s1x_shield_data.py")
    with np.load(CACHE) as z:                 # materialise: the render loop must not re-read
        return {k: z[k] for k in z.files}


def yaw_mat(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def place_human_in_robot_frame(pts, yaw, t):
    """The sim places the ROBOT at (yaw, t) in the human's frame: p_world = Rz(yaw) p + t.

    Shots s11/s12 show the equivalent picture with the robot at the origin, i.e. the human
    carried into the robot base frame by the inverse transform.
    """
    return (np.asarray(pts).reshape(-1, 3) - np.asarray(t)) @ yaw_mat(yaw)


def robot_capsules(cache, phase):
    """Interval-0 (= actual) robot link capsules at a phase of the 4 ms pick-and-place grid."""
    p = int(np.clip(phase, 0, cache["robot_p1"].shape[0] - 1))
    return (cache["robot_p1"][p].astype(np.float64),
            cache["robot_p2"][p].astype(np.float64),
            cache["robot_r"][p].astype(np.float64))


def add_robot(layer, cache, phase, col=None, rot=None, off=None, rscale=1.0):
    p1, p2, r = robot_capsules(cache, phase)
    if rot is not None:
        p1, p2 = p1 @ np.asarray(rot).T, p2 @ np.asarray(rot).T
    if off is not None:
        p1, p2 = p1 + off, p2 + off
    c = rgb(style.C_ROBOT) if col is None else col
    shades = np.linspace(0.62, 1.0, p1.shape[0])[:, None] * np.asarray(c)[None, :]
    layer.capsules(p1, p2, r * rscale, shades.astype(np.float32))
    return p1, p2, r


def pedestal(layer, top_z, col=None, r=0.16):
    """The 0.85 m pedestal the Panda's capsules sit on (their z starts at ~1.02 m)."""
    c = rgb("#5A6273") if col is None else col
    layer.capsule([0, 0, 0.02], [0, 0, top_z], r, c)
