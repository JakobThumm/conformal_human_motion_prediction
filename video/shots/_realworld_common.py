"""Shared plumbing for the three real-world-footage shots (s02, s08, s16).

All three recompose the same source capture, `video/media/realworld_source.mp4`
(2560x1440, 30 fps, 60.07 s, RViz screen grab of the live SARA-shield deployment).
This module owns

  * the measured crop rectangles of the three useful regions of that capture
    (raw RGB panel, 2-D pose overlay panel, 3-D view) -- the RViz chrome, the
    options tree, the title bar and the cursor are all outside them,
  * a colour grade that pushes the capture into the film's dark palette,
  * a frame reader, a speed-ramp timeline, rounded cards and text helpers,
  * the *measured* shield state (robot reachable-set capsules red = braking,
    green = verified) per source frame, cached to
    `video/media/realworld_shield_state.npz`.

Nothing here invents data: the shield state is read straight out of the pixels
of the capture (see `compute_shield_state`).

Regenerate the cache with::

    /home/thumm/miniconda3/envs/chmp-video/bin/python video/shots/_realworld_common.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common  # noqa: E402
import style  # noqa: E402

SRC = common.MEDIA / "realworld_source.mp4"
STATE_CACHE = common.MEDIA / "realworld_shield_state.npz"

SRC_W, SRC_H, SRC_FPS = 2560, 1440, 30.0
SRC_N = 1802                      # frames in the capture (60.067 s)

# --------------------------------------------------------------------------- crops
# Measured on the capture (see s16_notes.md).  (x0, y0, x1, y1), exclusive ends.
CROP_RGB = (0, 553, 668, 927)      # RViz "RGB Image" panel content  -> 668 x 374
CROP_POSE = (0, 999, 668, 1373)    # RViz "Pose 2D Overlay" content  -> 668 x 374
#   (the overlay panel is drawn 1.7 % taller than the RGB panel; these bounds are
#    the sub-rectangle that registers with CROP_RGB to <1 px, measured with ECC)
CROP_3D = (692, 98, 2542, 1414)    # RViz 3-D render area            -> 1850 x 1316

# --------------------------------------------------------------------------- type
_FONT_DIR = Path("/usr/share/fonts/opentype/inter")
_WEIGHTS = {
    "regular": "Inter-Regular.otf",
    "medium": "Inter-Medium.otf",
    "semibold": "Inter-SemiBold.otf",
    "bold": "Inter-Bold.otf",
    "light": "Inter-Light.otf",
}
_font_cache: dict = {}


def font(size: int, weight: str = "regular") -> ImageFont.FreeTypeFont:
    key = (size, weight)
    if key not in _font_cache:
        p = _FONT_DIR / _WEIGHTS[weight]
        if not p.exists():                                   # pragma: no cover
            p = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
        _font_cache[key] = ImageFont.truetype(str(p), size)
    return _font_cache[key]


def rgb(hexstr: str, a: int | None = None):
    c = common.hex2rgb(hexstr)
    return c if a is None else (*c, a)


# --------------------------------------------------------------------------- source
class Source:
    """Random-access reader for the capture; sequential reads stay cheap."""

    def __init__(self, path: Path = SRC):
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open {path}")
        self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or SRC_N
        self._pos = 0
        self._cache: dict[int, np.ndarray] = {}
        self._order: list[int] = []

    def frame(self, idx: int) -> np.ndarray:
        idx = int(np.clip(idx, 0, self.n - 1))
        if idx in self._cache:
            return self._cache[idx]
        if idx < self._pos or idx > self._pos + 24:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            self._pos = idx
        while True:
            ok, fr = self.cap.read()
            if not ok:
                raise RuntimeError(f"read failed at {idx}")
            cur, self._pos = self._pos, self._pos + 1
            if cur == idx:
                break
        self._cache[idx] = fr
        self._order.append(idx)
        if len(self._order) > 48:
            self._cache.pop(self._order.pop(0), None)
        return fr

    def at(self, t: float) -> np.ndarray:
        return self.frame(int(round(t * SRC_FPS)))

    def at_blend(self, t: float) -> np.ndarray:
        """Motion-blended fetch -- keeps speed-ramped segments free of judder.

        At 1.0x playback from a frame-aligned in-point the fraction is 0 and
        this returns the untouched source frame.  Across a shield state change
        blending is switched off (it would mix the red and the green capsules
        into a meaningless olive), and the nearest frame is used instead.
        """
        f = t * SRC_FPS
        i0 = int(np.floor(f))
        u = float(f - i0)
        if u < 0.02:
            return self.frame(i0)
        if u > 0.98:
            return self.frame(i0 + 1)
        st = shield_state()
        j0 = int(np.clip(i0, 0, len(st) - 1))
        j1 = int(np.clip(i0 + 1, 0, len(st) - 1))
        if st[j0] != st[j1]:
            return self.frame(i0 if u < 0.5 else i0 + 1)
        a = self.frame(i0).astype(np.float32)
        b = self.frame(i0 + 1).astype(np.float32)
        return (a + (b - a) * u).astype(np.uint8)

    def close(self):
        self.cap.release()


def crop(fr: np.ndarray, rect) -> np.ndarray:
    x0, y0, x1, y1 = rect
    return fr[y0:y1, x0:x1]


def resize(img: np.ndarray, w: int, h: int) -> np.ndarray:
    interp = cv2.INTER_LANCZOS4 if (w > img.shape[1]) else cv2.INTER_AREA
    return cv2.resize(img, (w, h), interpolation=interp)


# --------------------------------------------------------------------------- grade
_BG = np.array(common.hex2rgb(style.BG), np.float32)


def grade(bgr: np.ndarray, sat: float = 0.86, gamma: float = 1.24,
          gain: float = 0.93, floor: np.ndarray = _BG) -> np.ndarray:
    """BGR uint8 capture -> RGB uint8 graded into the film's dark palette.

    Slightly desaturated, midtones pulled down, and the black point lifted to
    `style.BG` so the RViz grey background merges with the film canvas while
    the blue occupancy and the red/green robot capsules keep their punch.
    """
    x = bgr[:, :, ::-1].astype(np.float32) / 255.0
    luma = (x * np.array([0.2126, 0.7152, 0.0722], np.float32)).sum(2, keepdims=True)
    x = luma + (x - luma) * sat
    np.clip(x, 0.0, 1.0, out=x)
    x **= gamma
    x *= gain
    out = floor + x * (255.0 - floor)
    return np.clip(out, 0, 255).astype(np.uint8)


def canvas() -> np.ndarray:
    c = np.empty((style.HEIGHT, style.WIDTH, 3), np.uint8)
    c[:] = common.hex2rgb(style.BG)
    return c


# --------------------------------------------------------------------------- scrims
def _ramp(n: int, invert: bool = False) -> np.ndarray:
    r = np.linspace(0.0, 1.0, n, dtype=np.float32)
    r = r * r * (3 - 2 * r)
    return r[::-1] if invert else r


def bottom_scrim(img: np.ndarray, top: int = 760, strength: float = 0.82) -> None:
    """Darken the subtitle band so burned-in text always reads.  In place."""
    h = img.shape[0] - top
    a = (_ramp(h) * strength)[:, None, None]
    band = img[top:].astype(np.float32)
    img[top:] = np.clip(band * (1 - a) + _BG * a, 0, 255).astype(np.uint8)


def top_scrim(img: np.ndarray, bottom: int = 260, strength: float = 0.55) -> None:
    a = (_ramp(bottom, invert=True) * strength)[:, None, None]
    band = img[:bottom].astype(np.float32)
    img[:bottom] = np.clip(band * (1 - a) + _BG * a, 0, 255).astype(np.uint8)


_dim_cache: dict = {}


def dim_map(vig: float = 0.42, top: int = 0, top_s: float = 0.0,
            bottom: int = 0, bottom_s: float = 0.0,
            h: int = style.HEIGHT, w: int = style.WIDTH) -> np.ndarray:
    """One combined "fade towards style.BG" alpha map (vignette + scrims).

    Precomputed once per parameter set; `apply_dim` then costs a single
    full-frame lerp instead of three.
    """
    key = (round(vig, 3), top, round(top_s, 3), bottom, round(bottom_s, 3), h, w)
    if key not in _dim_cache:
        a = np.zeros((h, w), np.float32)
        if vig > 0:
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            r = np.sqrt(((xx / w - 0.5) / 0.5) ** 2 + ((yy / h - 0.5) / 0.5) ** 2)
            a = np.clip((r - 0.72) / 0.78, 0, 1) ** 1.7 * vig
        if top > 0 and top_s > 0:
            a[:top] = np.maximum(a[:top], (_ramp(top, invert=True) * top_s)[:, None])
        if bottom > 0 and bottom_s > 0:
            n = h - bottom
            a[bottom:] = np.maximum(a[bottom:], (_ramp(n) * bottom_s)[:, None])
        _dim_cache[key] = a[:, :, None]
    return _dim_cache[key]


def apply_dim(img: np.ndarray, a: np.ndarray) -> np.ndarray:
    img[:] = np.clip(img.astype(np.float32) * (1 - a) + _BG * a,
                     0, 255).astype(np.uint8)
    return img


_vig_cache: dict = {}


def vignette(img: np.ndarray, strength: float = 0.42) -> None:
    h, w = img.shape[:2]
    key = (h, w, round(strength, 3))
    if key not in _vig_cache:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        r = np.sqrt(((xx / w - 0.5) / 0.5) ** 2 + ((yy / h - 0.5) / 0.5) ** 2)
        a = np.clip((r - 0.72) / 0.78, 0, 1) ** 1.7 * strength
        _vig_cache[key] = a[:, :, None]
    a = _vig_cache[key]
    img[:] = np.clip(img.astype(np.float32) * (1 - a) + _BG * a, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- overlay
class Overlay:
    """RGBA scratch layer -> alpha-composited onto an RGB uint8 canvas."""

    def __init__(self, w: int = style.WIDTH, h: int = style.HEIGHT):
        self.img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        self.d = ImageDraw.Draw(self.img)

    def text(self, xy, s, size=26, weight="regular", color=style.FG, alpha=255,
             anchor="la", spacing=6):
        if alpha <= 0:
            return
        self.d.multiline_text(xy, s, font=font(size, weight),
                              fill=rgb(color, int(alpha)), anchor=anchor,
                              spacing=spacing)

    def width(self, s, size=26, weight="regular") -> int:
        return int(self.d.textlength(s, font=font(size, weight)))

    def line(self, pts, color=style.FG_MUTED, alpha=255, w=2):
        if alpha <= 0:
            return
        self.d.line(pts, fill=rgb(color, int(alpha)), width=w, joint="curve")

    def dot(self, c, r, color, alpha=255):
        if alpha <= 0:
            return
        self.d.ellipse([c[0] - r, c[1] - r, c[0] + r, c[1] + r], fill=rgb(color, int(alpha)))

    def ring(self, c, r, color, alpha=255, w=2):
        if alpha <= 0:
            return
        self.d.ellipse([c[0] - r, c[1] - r, c[0] + r, c[1] + r],
                       outline=rgb(color, int(alpha)), width=w)

    def rrect(self, box, r, fill=None, outline=None, alpha=255, w=1):
        self.d.rounded_rectangle(box, radius=r,
                                 fill=None if fill is None else rgb(fill, int(alpha)),
                                 outline=None if outline is None else rgb(outline, int(alpha)),
                                 width=w)

    def apply(self, base: np.ndarray) -> np.ndarray:
        ov = np.asarray(self.img, np.float32)
        a = ov[:, :, 3:4] / 255.0
        base[:] = np.clip(base.astype(np.float32) * (1 - a) + ov[:, :, :3] * a,
                          0, 255).astype(np.uint8)
        return base


# --------------------------------------------------------------------------- cards
_mask_cache: dict = {}


def _round_mask(w: int, h: int, r: int) -> np.ndarray:
    key = (w, h, r)
    if key not in _mask_cache:
        ss = 4
        m = Image.new("L", (w * ss, h * ss), 0)
        ImageDraw.Draw(m).rounded_rectangle([0, 0, w * ss - 1, h * ss - 1],
                                            radius=r * ss, fill=255)
        m = m.resize((w, h), Image.LANCZOS)
        _mask_cache[key] = (np.asarray(m, np.float32) / 255.0)[:, :, None]
    return _mask_cache[key]


def paste_card(base: np.ndarray, img_rgb: np.ndarray, x: int, y: int,
               radius: int = 16, alpha: float = 1.0,
               border: str | None = style.GRID, border_alpha: float = 1.0) -> None:
    """Alpha-composite `img_rgb` at (x, y) with rounded corners + hairline border."""
    h, w = img_rgb.shape[:2]
    m = _round_mask(w, h, radius) * float(np.clip(alpha, 0, 1))
    dst = base[y:y + h, x:x + w].astype(np.float32)
    base[y:y + h, x:x + w] = np.clip(dst * (1 - m) + img_rgb.astype(np.float32) * m,
                                     0, 255).astype(np.uint8)
    if border is not None and border_alpha > 0:
        o = Overlay(base.shape[1], base.shape[0])
        o.rrect([x, y, x + w - 1, y + h - 1], radius, outline=border,
                alpha=int(255 * np.clip(alpha, 0, 1) * border_alpha), w=2)
        o.apply(base)


# --------------------------------------------------------------------------- timeline
class Timeline:
    """Maps output time -> source time through play / hold / speed-ramp segments.

    Segments: ("play", src_a, src_b, speed[, xfade]) or
    ("hold", src_t, out_seconds).  `xfade` cross-dissolves into that segment.
    A trailing hold of at least `min_hold` absorbs the remaining runtime; if the
    requested duration is shorter than the segment list, everything is scaled.
    """

    def __init__(self, segments, duration: float, min_hold: float = 0.6):
        core = 0.0
        for s in segments:
            core += (s[2] - s[1]) / s[3] if s[0] == "play" else s[2]
        scale = 1.0
        if duration - min_hold < core:
            scale = max(1e-6, (duration - min_hold) / core)
        self.spans = []                         # (out0, out1, kind, a, b, xfade)
        t = 0.0
        for s in segments:
            if s[0] == "play":
                d = (s[2] - s[1]) / s[3] * scale
                xf = (s[4] if len(s) > 4 else 0.0) * scale
                self.spans.append((t, t + d, "play", s[1], s[2], xf))
            else:
                d = s[2] * scale
                self.spans.append((t, t + d, "hold", s[1], s[1], 0.0))
            t += d
        self.core_end = t
        self.spans.append((t, max(t, duration) + 10.0, "hold",
                           self.spans[-1][4], self.spans[-1][4], 0.0))

    def _eval(self, span, T):
        o0, o1, kind, a, b = span[:5]
        if kind == "hold":
            return a
        u = 0.0 if o1 <= o0 else (T - o0) / (o1 - o0)
        return a + (b - a) * u

    def src(self, T: float) -> float:
        for sp in self.spans:
            if sp[0] <= T < sp[1]:
                return self._eval(sp, T)
        return self.spans[-1][3]

    def blend(self, T: float):
        """Return [(src_time, weight), ...] -- 2 entries inside a cross-fade."""
        i = len(self.spans) - 1
        for k, sp in enumerate(self.spans):
            if sp[0] <= T < sp[1]:
                i = k
                break
        cur = self._eval(self.spans[i], T)
        xf = self.spans[i][5]
        if xf <= 0 or i == 0:
            return [(cur, 1.0)]
        o0 = self.spans[i][0]
        if T - o0 >= xf:
            return [(cur, 1.0)]
        prev = self.spans[i - 1]
        if prev[2] != "play":
            return [(cur, 1.0)]
        # extrapolate the previous segment past its out-point
        rate = (prev[4] - prev[3]) / max(1e-6, prev[1] - prev[0])
        pv = prev[4] + rate * (T - prev[1])
        w = common.ease((T - o0) / xf, "smooth")
        return [(pv, 1.0 - w), (cur, w)]


# --------------------------------------------------------------------------- shield state
def compute_shield_state(path: Path = SRC) -> np.ndarray:
    """Per-source-frame shield state read out of the 3-D view pixels.

    SARA shield renders the robot's reachable occupancy capsules **red** while
    the candidate trajectory fails verification (fail-safe braking) and
    **green** while it is verified.  Both never appear at once, so counting
    strongly-red vs strongly-green pixels inside the 3-D view is an exact
    read-out.  Returns uint8, 1 = braking -- raw, one value per source frame,
    no smoothing, so it always agrees with the pixels on screen.
    """
    cap = cv2.VideoCapture(str(path))
    x0, y0, x1, y1 = CROP_3D
    raw = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        v = fr[y0:y1:4, x0:x1:4].astype(np.int16)
        b, g, r = v[..., 0], v[..., 1], v[..., 2]
        red = (((r - np.maximum(g, b)) > 25) & (r > 70)).sum()
        grn = (((g - np.maximum(r, b)) > 25) & (g > 70)).sum()
        raw.append(1 if red > grn else 0)
    cap.release()
    return np.array(raw, np.uint8)


_state = None


def shield_state() -> np.ndarray:
    global _state
    if _state is None:
        if STATE_CACHE.exists():
            _state = np.load(STATE_CACHE)["braking"]
        else:                                                # pragma: no cover
            _state = compute_shield_state()
            np.savez_compressed(STATE_CACHE, braking=_state)
    return _state


def _dilate(x: np.ndarray, k: int) -> np.ndarray:
    out = x.copy()
    for i in range(1, k + 1):
        out[i:] |= x[:-i]
    return out


_disp = None


def display_state() -> np.ndarray:
    """Shield state as shown on the badge.  Conservative in both directions.

    The verification loop runs far faster than the 30 fps screen grab, so on
    the boundary the pixel read-out toggles on single frames.  We (a) latch
    *braking* for 7 frames (0.23 s) after every red frame and (b) fill any
    remaining "verified" gap shorter than 9 frames (0.3 s).  Both operations
    can only ever turn a displayed "verified" into "braking", never the other
    way round, so the badge never claims a verified trajectory while the
    capsules on screen are red.
    """
    global _disp
    if _disp is None:
        raw = shield_state().astype(bool)
        lat = _dilate(raw, 7)
        closed = _dilate(lat, 9)                       # dilate ...
        closed = ~_dilate((~closed)[::-1], 9)[::-1]    # ... then erode = closing
        _disp = (lat | closed).astype(np.uint8)
    return _disp


def braking_at(t: float, latch: int = 7) -> bool:
    """Shield state for display at source time `t`.

    The verification loop runs far faster than the 30 fps screen grab, so on
    the boundary the read-out toggles on single frames.  The badge therefore
    latches the *braking* state for `latch` frames (0.23 s): it can never
    claim "verified" while the capsules on screen are red, only the
    conservative way round.
    """
    s = display_state()
    return bool(s[int(np.clip(round(t * SRC_FPS), 0, len(s) - 1))])


if __name__ == "__main__":                                   # regenerate the cache
    st = compute_shield_state()
    np.savez_compressed(STATE_CACHE, braking=st)
    print(f"{STATE_CACHE}: {len(st)} frames, {st.mean() * 100:.1f} % braking")
