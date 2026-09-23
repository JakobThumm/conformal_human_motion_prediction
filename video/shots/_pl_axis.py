"""Shared bits for the manim shots s01 / s06 / s14 / s17 (owned by this shot group).

Nothing here is imported by ``build.py``; it only exists so that the ISO 13849-1 performance-level
axis -- the motif introduced in s01 and paid off in s14 -- is *the same object* in both shots.

The PL bands are the ones the repo itself uses to classify a PFH_D value
(``src/conformal_human_motion_prediction/generate_plots/conformal_results_common.py::PL_BANDS``),
and PL d matches the manuscript (S1: "PL d ... spans 1e-7 <= PFH_D < 1e-6").
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

VIDEO_DIR = Path(__file__).resolve().parents[1]
if str(VIDEO_DIR) not in sys.path:
    sys.path.insert(0, str(VIDEO_DIR))
import style  # noqa: E402

# --------------------------------------------------------------------------- type scale
# manim's `font_size` is not pixels: measured on Inter at 1920x1080 (frame_height = 8 units),
# an em of `font_size` F is F * 1.83 px tall.  style.py specifies px, so convert.
PX_PER_FS = 1.83


def fs(px: float) -> float:
    """style.py pixel size -> manim font_size."""
    return px / PX_PER_FS


# Below ~40, manim quantises Pango's glyph advances onto its own grid and word spaces visibly
# collapse ("conformal prediction sets for" -> "conformalprediction setsfor").  Rendering at
# RENDER_FS and scaling down is exact and costs nothing.
RENDER_FS = 48.0


def _big(cls, s: str, px: float, color: str, weight: str, **kw):
    target = fs(px)
    k = max(1.0, RENDER_FS / target)
    m = cls(s, font=style.FONT, font_size=target * k, color=color, weight=weight, **kw)
    return m.scale(1.0 / k)


def txt(s: str, px: float = style.BODY_SIZE, color: str = style.FG, weight: str = "NORMAL",
        **kw):
    from manim import Text
    return _big(Text, s, px, color, weight, **kw)


def mtxt(s: str, px: float = style.BODY_SIZE, color: str = style.FG, weight: str = "NORMAL",
         **kw):
    """Pango-markup text (for <sub>/<sup>), same metrics as :func:`txt`."""
    from manim import MarkupText
    return _big(MarkupText, s, px, color, weight, **kw)


def kicker(s: str):
    """The film's top-left shot label."""
    from manim import LEFT, UP
    k = txt(s, px=style.CAPTION_SIZE, color=style.FG_MUTED, weight="MEDIUM")
    k.to_edge(UP, buff=0.42).to_edge(LEFT, buff=0.62)
    return k


# --------------------------------------------------------------------------- PL axis motif
# ISO 13849-1:2023 performance levels, keyed by the PFH_D band [lo, hi) in failures per hour.
# Listed left -> right as drawn: PL a (weakest, highest PFH_D) on the left, PL e on the right.
PL_BANDS = [
    ("a", 1e-5, 1e-4),
    ("b", 3e-6, 1e-5),
    ("c", 1e-6, 3e-6),
    ("d", 1e-7, 1e-6),
    ("e", 1e-8, 1e-7),
]

# Subtle "the further right, the safer" ramp: how far each band's fill/label is blended towards
# style.GREEN.  PL e is the green end; PL a stays neutral.  The *target* band is called out with
# a green outline instead of more fill, so it stays identifiable on top of the ramp.
BAND_RAMP = {"a": 0.00, "b": 0.04, "c": 0.09, "d": 0.17, "e": 0.30}
LABEL_RAMP = {"a": 0.00, "b": 0.18, "c": 0.38, "d": 0.66, "e": 1.00}


def mix(a: str, b: str, t: float) -> str:
    """Blend two #rrggbb colours (t = 0 -> a, t = 1 -> b)."""
    ai = [int(a.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)]
    bi = [int(b.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)]
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(ai, bi))


class PLAxis:
    """A horizontal log axis of the ISO 13849-1 PL bands, built in place at ``center``.

    The axis runs **PL a (left) -> PL e (right)**, so PFH_D *decreases* to the right:
    10^-4 at the left edge, 10^-8 at the right edge.  Use :meth:`direction_labels` to spell that
    out when there is vertical room for it.

    Attributes
    ----------
    group : VGroup        everything (bar + band labels + decade ticks)
    bands : dict          name -> the band's Rectangle
    labels : dict         name -> the band's "PL x" Text
    """

    LO, HI = 1e-8, 1e-4

    def __init__(self, center=(0.0, 0.0, 0.0), width: float = 10.4, bar_h: float = 0.44,
                 target: str = "d", label_px: float = 26.0, tick_px: float = 24.0):
        from manim import VGroup, Rectangle, Line, MathTex, DOWN

        self.center = np.array(center, dtype=float)
        self.width = width
        self.bar_h = bar_h
        self.target = target
        self._l0, self._l1 = np.log10(self.LO), np.log10(self.HI)

        self.bands, self.labels = {}, {}
        bar = VGroup()
        for name, lo, hi in PL_BANDS:
            x0, x1 = self.x_of(hi), self.x_of(lo)          # hi is the LEFT edge now
            is_t = name == target
            r = Rectangle(
                width=abs(x1 - x0), height=bar_h,
                stroke_width=2.4 if is_t else 1.4,
                stroke_color=style.GREEN if is_t else style.GRID,
                fill_color=mix(style.BG_PANEL, style.GREEN, BAND_RAMP[name]),
                fill_opacity=1.0,
            )
            r.move_to(np.array([0.5 * (x0 + x1), self.center[1], 0.0]))
            lab = txt(f"PL {name}", px=label_px, weight="MEDIUM",
                      color=mix(style.FG_MUTED, style.GREEN, LABEL_RAMP[name]))
            lab.move_to(r.get_center())
            self.bands[name], self.labels[name] = r, lab
            bar.add(r, lab)

        ticks = VGroup()
        for e in range(int(self._l0), int(self._l1) + 1):
            x = self.x_of(10.0 ** e)
            t = Line([x, self.center[1] - bar_h / 2, 0], [x, self.center[1] - bar_h / 2 - 0.13, 0],
                     stroke_width=1.6, color=style.FG_MUTED)
            lab = MathTex(rf"10^{{{e}}}", color=style.FG_MUTED).scale(0.46)
            lab.next_to(t, DOWN, buff=0.13)
            ticks.add(t, lab)

        self.bar, self.ticks = bar, ticks
        self.group = VGroup(bar, ticks)

    # ---------------------------------------------------------------- geometry
    def x_of(self, pfh: float) -> float:
        """x coordinate of a PFH_D value on the axis (clamped to the drawn range).

        Reversed: large PFH_D is on the left, small on the right.
        """
        f = (np.log10(np.clip(pfh, self.LO, self.HI)) - self._l0) / (self._l1 - self._l0)
        return self.center[0] + (0.5 - f) * self.width

    def top_of(self) -> float:
        return self.center[1] + self.bar_h / 2

    def bottom_of(self) -> float:
        return self.center[1] - self.bar_h / 2

    # ---------------------------------------------------------------- decorations
    def threshold(self, pfh: float = 1e-6, color: str = style.YELLOW, height: float = 1.05):
        """Dashed vertical rule at ``pfh`` (the 'required' line)."""
        from manim import DashedLine
        x = self.x_of(pfh)
        return DashedLine([x, self.bottom_of() - 0.05, 0], [x, self.bottom_of() + height, 0],
                          dash_length=0.10, dashed_ratio=0.55, stroke_width=2.6, color=color)

    def marker(self, pfh: float, color: str = style.BLUE, size: float = 0.20):
        """A small triangle pointing down onto the bar from above."""
        from manim import Triangle
        tri = Triangle(color=color, fill_color=color, fill_opacity=1.0, stroke_width=0)
        tri.rotate(np.pi).scale(size)
        tri.move_to([self.x_of(pfh), self.top_of() + 0.20, 0])
        return tri

    def region(self, lo: float, hi: float, color: str = style.GREEN, opacity: float = 0.16,
               pad: float = 0.0):
        """A translucent wash over the bar between two PFH_D values (e.g. the compliant half)."""
        from manim import Rectangle
        x0, x1 = self.x_of(hi), self.x_of(lo)
        r = Rectangle(width=abs(x1 - x0), height=self.bar_h + 2 * pad,
                      stroke_width=0, fill_color=color, fill_opacity=opacity)
        r.move_to([0.5 * (x0 + x1), self.center[1], 0])
        return r

    def direction_labels(self, px: float = 22.0, buff: float = 0.30,
                         left: str = "\u2190  more dangerous failures",
                         right: str = "fewer dangerous failures  \u2192"):
        """Because the axis is reversed, say so: PFH_D grows to the *left*.

        Returns a VGroup of two muted captions sitting under the decade ticks; it is deliberately
        **not** part of :attr:`group` because s14 has no vertical room for it.
        """
        from manim import VGroup
        y = self.ticks.get_bottom()[1] - buff
        l = txt(left, px=px, color=style.FG_MUTED)
        r = txt(right, px=px, color=style.FG_MUTED)
        l.move_to([self.center[0] - self.width / 2 + l.width / 2, y - l.height / 2, 0])
        r.move_to([self.center[0] + self.width / 2 - r.width / 2, y - r.height / 2, 0])
        return VGroup(l, r)


# --------------------------------------------------------------------------- exact-length scenes
class TimedScene:
    """Mixin that tracks elapsed scene time so a shot can end at exactly ``self.D`` seconds.

    Usage inside ``construct``::

        self.pl(FadeIn(m), run_time=0.8)      # instead of self.play
        self.hold(0.4)                        # instead of self.wait
        self.finish()                         # pads to self.D
    """

    D = 1.0
    _t = 0.0

    def pl(self, *anims, **kw):
        rt = kw.get("run_time", 1.0)
        self.play(*anims, **kw)
        self._t += rt

    def hold(self, dt: float):
        if dt > 1e-6:
            self.wait(dt)
            self._t += dt

    def at(self, t: float):
        """Wait until absolute scene time ``t``."""
        self.hold(t - self._t)

    def finish(self, min_hold: float = 0.6, tail: float = 0.25):
        """Hold the last frame to the end, plus a short ``tail`` so that ``common.conform``'s
        ``-t duration`` trim always has material and lands on the exact frame count."""
        rest = self.D - self._t
        if rest < min_hold - 1e-6:
            print(f"WARNING: only {rest:.2f}s left at the end (wanted >= {min_hold}s)")
        self.hold(max(rest, 1.0 / 30.0) + tail)
