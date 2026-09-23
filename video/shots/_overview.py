"""The manuscript's pipeline figure, converted to the film's dark palette, with region highlights.

Shared by s03/s04 (perception) and s05 (motion prediction): each of those shots opens on the
overview with *its* stage lit and the rest of the pipeline dimmed, so the viewer always knows
where in the system they are.

Source: ``Vision-Based-Safe-Human-Robot-Collaboration/figures/overview_full_2.png`` (2100x1109,
Fig. 1 of the manuscript).  It is a white-background line drawing, so it is converted by
**inverting luminance in HLS** rather than inverting RGB: black ink becomes white, the white page
becomes the film's background, and the figure's green/blue accents keep their hue.

    import _overview as OV
    frame = OV.compose(OV.REGION_POSE, reveal=1.0, focus=1.0, title="From two views to 3-D")
"""
from __future__ import annotations

import colorsys
import sys
from pathlib import Path

import numpy as np

VIDEO_DIR = Path(__file__).resolve().parents[1]
if str(VIDEO_DIR) not in sys.path:
    sys.path.insert(0, str(VIDEO_DIR))
import common  # noqa: E402
import style  # noqa: E402

FIG = VIDEO_DIR.parent / "Vision-Based-Safe-Human-Robot-Collaboration/figures/overview_full_2.png"
CACHE = VIDEO_DIR / "media" / "overview_dark.png"

# Regions in *source figure* pixels (x0, y0, x1, y1), measured on the 2100x1109 original.
REGION_POSE = (2, 2, 575, 545)        # 2-D pose estimation + uncertainty-aware triangulation
REGION_MOTION = (700, 95, 1445, 545)  # pose history + uncertainty-aware motion prediction
REGION_CONFORMAL = (1450, 95, 2098, 545)
# The whole bottom row: intended trajectory -> monitored-trajectory planner -> robot reachable
# occupancy -> Verification -> the reachable occupancies built from the conformal sets.  That
# row *is* SARA shield, so lighting only its right half told half the story.
REGION_SHIELD = (2, 548, 2098, 1107)
# Both monitor trapezoids ("Pose monitor" ~ (540,130,695,480), "Motion monitor" ~
# (1448,105,1610,532)) plus the prediction blocks between them -- one box cannot isolate
# the two trapezoids alone.
REGION_MONITORS = (536, 100, 1614, 538)

# Where the figure sits in the 1920x1080 frame (above the subtitle band, below the title).
# 1300 px wide -> 687 px tall, so the figure ends at y = 883, clear of the
# subtitle band that starts at y = 928.
FIG_W, FIG_X, FIG_Y = 1300, 310, 210


def _invert_luminance(rgb: np.ndarray) -> np.ndarray:
    """L -> 1 - L in HLS, hue and saturation untouched (vectorised colorsys)."""
    flat = rgb.reshape(-1, 3) / 255.0
    h, l, s = np.vectorize(colorsys.rgb_to_hls)(flat[:, 0], flat[:, 1], flat[:, 2])
    r, g, b = np.vectorize(colorsys.hls_to_rgb)(h, 1.0 - l, s)
    return (np.stack([r, g, b], 1).reshape(rgb.shape) * 255.0).astype(np.uint8)


def dark_figure() -> np.ndarray:
    """The figure on the film's background, cached (the HLS pass takes a few seconds)."""
    from PIL import Image
    if CACHE.exists():
        return np.asarray(Image.open(CACHE).convert("RGB"))
    src = Image.open(FIG)
    bg = Image.new("RGBA", src.size, (255, 255, 255, 255))   # flatten the alpha onto white first
    rgb = np.asarray(Image.alpha_composite(bg, src.convert("RGBA")).convert("RGB"))
    out = _invert_luminance(rgb)
    # The inverted page is pure black; lift it to the film's background so the card reads as part
    # of the same canvas rather than a hole in it.
    page = np.array(common.hex2rgb(style.BG), np.float32)
    lum = out.astype(np.float32).mean(2, keepdims=True) / 255.0
    out = (out.astype(np.float32) * lum + page * (1.0 - lum)).astype(np.uint8)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out).save(CACHE)
    return out


def _fit(fig: np.ndarray, width: int) -> np.ndarray:
    from PIL import Image
    h = int(round(width * fig.shape[0] / fig.shape[1]))
    return np.asarray(Image.fromarray(fig).resize((width, h), Image.LANCZOS))


def compose(region=None, reveal: float = 1.0, focus: float = 0.0, title: str = "",
            caption: str = "", kicker: str = "", accent: str = style.C_PRED,
            width: int = style.WIDTH, height: int = style.HEIGHT) -> np.ndarray:
    """One frame: the dark overview figure with `region` lit and everything else dimmed.

    reveal : 0..1  the figure fading in
    focus  : 0..1  how strongly the rest of the pipeline is pushed back (0 = flat, 1 = full)
    """
    from PIL import Image, ImageDraw

    frame = np.zeros((height, width, 3), np.float32)
    frame[:] = common.hex2rgb(style.BG)
    fig = _fit(dark_figure(), width=FIG_W).astype(np.float32)
    fh, fw = fig.shape[:2]
    sx = fw / (2100.0)

    if region is not None and focus > 0.0:
        mask = np.full((fh, fw, 1), 1.0 - 0.80 * focus, np.float32)
        x0, y0, x1, y1 = (int(round(v * sx)) for v in region)
        mask[y0:y1, x0:x1] = 1.0
        fig = fig * mask

    a = common.ease(reveal, "out")
    sl = (slice(FIG_Y, FIG_Y + fh), slice(FIG_X, FIG_X + fw))
    frame[sl] = frame[sl] * (1.0 - a) + fig * a

    img = Image.fromarray(np.clip(frame, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img, "RGBA")

    if region is not None and focus > 0.0:
        x0, y0, x1, y1 = (int(round(v * sx)) for v in region)
        box = (FIG_X + x0 - 6, FIG_Y + y0 - 6, FIG_X + x1 + 6, FIG_Y + y1 + 6)
        col = common.hex2rgb(accent)
        for i, w in enumerate((10, 6, 3)):                 # cheap soft glow
            al = int(26 * focus) if i < 2 else int(230 * focus)
            d.rounded_rectangle([box[0] - i * 3, box[1] - i * 3, box[2] + i * 3, box[3] + i * 3],
                                radius=14 + i * 3, outline=col + (al,), width=w)

    _text(d, (64, 46), kicker, style.CAPTION_SIZE, style.FG_MUTED, "Regular", a=reveal)
    _text(d, (64, 92), title, 42, style.FG, "Medium", a=reveal)
    if caption:                      # the lit stage, named right under the title
        _text(d, (64, 146), caption, 30, accent, "Medium", a=focus)
    return np.asarray(img)


_FONTS: dict = {}


def font(px: int, weight: str = "Regular"):
    """Inter at `px`, cached.  The whole film uses this family -- no second sans anywhere."""
    from PIL import ImageFont
    key = (px, weight)
    if key not in _FONTS:
        path = f"/usr/share/fonts/opentype/inter/Inter-{weight}.otf"
        if not Path(path).exists():
            path = "/usr/share/fonts/opentype/inter/Inter-Regular.otf"
        _FONTS[key] = ImageFont.truetype(path, px)
    return _FONTS[key]


def _text(d, xy, s, px, colour, weight="Regular", a=1.0):
    if not s or a <= 0.01:
        return
    d.text(xy, s, font=font(px, weight), fill=common.hex2rgb(colour) + (int(255 * min(a, 1.0)),))


if __name__ == "__main__":                                  # quick visual check
    from PIL import Image
    for name, reg in (("pose", REGION_POSE), ("motion", REGION_MOTION)):
        Image.fromarray(compose(reg, reveal=1.0, focus=1.0,
                                kicker="Method", title=f"overview -- {name}",
                                caption="highlighted stage")).save(f"/tmp/ov_{name}_frame.png")
        print(f"/tmp/ov_{name}_frame.png")
