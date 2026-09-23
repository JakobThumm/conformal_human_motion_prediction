#!/usr/bin/env python
"""s15 -- "Pipeline results": the OOD-handling ablation and the runtime breakdown.

Every symbol on the slide is set as real maths (Computer Modern via matplotlib's mathtext,
`math_glyph` / `Canvas.rich` below, the technique from `s05_motion.py`): N_req in the ablation
legend, and f_2D / SLU_2D / f_mot / SLU_mot in the runtime legend, exactly as the manuscript
writes them.

Narration: "The fallback cuts invalid predictions by twenty-four percent, and the pipeline
runs in sixty-six milliseconds."

  (a) OOD handling, manuscript Table `figures/result_tables/full_pipeline_results.tex`
      (identical to results/final/full_pipeline/full_pipeline_results.tex, written by
      generate_plots/generate_full_pipeline_results.py):
          N_req   H invalid [%]   Motion valid [%]   MPJPE [mm]
          3 (ours)    12.63            74.74            55.15
          50          16.57            68.70            54.14
      -> invalid pose buffers -23.8 % (results/final/full_pipeline/full_pipeline_sentence.tex),
         MPJPE +1.9 %.
  (b) Runtime, results/final/runtime/runtime_stages.csv (= manuscript
      `figures/result_tables/runtime.tex`): per-stage medians over 450 pipeline steps on one
      NVIDIA RTX 5090; total median 65.86 ms.  The two sketched-Lanczos monitors account for
      36.39 + 9.08 = 45.47 ms = 69 % (results/final/runtime/runtime_sentence.tex).

Prepare (no GPU needed -- parses the two committed result files into video/media/s15_results.npz;
either interpreter works):

    cd /home/thumm/code/conformal_human_motion_prediction
    /home/thumm/miniconda3/envs/chmp-video/bin/python video/shots/s15_ood_results.py --prepare

Render (conda env chmp-video):

    /home/thumm/miniconda3/envs/chmp-video/bin/python video/shots/s15_ood_results.py \
        --out video/build/shots/s15.mp4 --duration 7.0
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import numpy as np

VIDEO_DIR = Path(__file__).resolve().parents[1]
REPO = VIDEO_DIR.parent
sys.path.insert(0, str(VIDEO_DIR))

CACHE = VIDEO_DIR / "media" / "s15_results.npz"

TEX = REPO / "results/final/full_pipeline/full_pipeline_results.tex"
RUNTIME_CSV = REPO / "results/final/runtime/runtime_stages.csv"

# stage key -> (label runs, colour role).  A run that starts with "$" is set as real maths
# (Computer Modern via mathtext, see `math_glyph`), everything else is Inter -- so the symbols
# read exactly as the manuscript writes them: $f_{\mathrm{2D}}$, $\mathrm{SLU}_{\mathrm{2D}}$,
# $f_{\mathrm{mot}}$, $\mathrm{SLU}_{\mathrm{mot}}$ (S3 methodology, runtime.tex).
# Both OOD monitors share the OOD purple; both network forward passes share the prediction teal.
# Order = pipeline order.
STAGES = [
    ("pose_2d",     ["2-D pose  ", r"$f_{\mathrm{2D}}$"],              "C_PRED"),
    ("ood_pose",    ["pose OOD  ", r"$\mathrm{SLU}_{\mathrm{2D}}$"],   "C_OOD"),
    ("triangulate", ["triangulation"],                                 "C_ROBOT"),
    ("motion",      ["motion  ", r"$f_{\mathrm{mot}}$"],               "C_PRED"),
    ("ood_motion",  ["motion OOD  ", r"$\mathrm{SLU}_{\mathrm{mot}}$"], "C_OOD"),
    ("set",         ["prediction sets"],                               "C_OURS"),
]


def plain(runs) -> str:
    """ASCII form of a label run list -- only used for the cache / stdout."""
    return "".join(r.strip("$").replace(r"\mathrm", "").replace("{", "").replace("}", "")
                   for r in runs)


def prepare() -> None:
    rows = {}
    for line in TEX.read_text().splitlines():
        m = re.match(r"\s*(\d+)(?:\s*\(ours\))?\s*&(.+?)\\\\", line)
        if not m:
            continue
        n = int(m.group(1))
        vals = [float(re.sub(r"[^0-9.]", "", c)) for c in m.group(2).split("&")]
        rows[n] = vals            # [H invalid %, motion valid %, MPJPE mm]
    assert 3 in rows and 50 in rows, rows

    stage_ms, stage_lbl = [], []
    with open(RUNTIME_CSV) as f:
        rt = {r["stage"]: r for r in csv.DictReader(f)}
    for key, runs, _ in STAGES:
        stage_ms.append(float(rt[key]["median_ms"]))
        stage_lbl.append(plain(runs))
    total = float(rt["total"]["median_ms"])

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CACHE,
        abl_before=np.array(rows[50], float),      # N_req = 50
        abl_after=np.array(rows[3], float),        # N_req = 3  (ours)
        stage_ms=np.array(stage_ms, float),
        stage_lbl=np.array(stage_lbl),
        total_ms=np.float64(total),
        n_steps=np.int64(int(rt["total"]["n"])),
        device=np.array(rt["total"]["device"]),
    )
    print(f"wrote {CACHE}")
    print(f"  N_req=50 -> {rows[50]}   N_req=3 -> {rows[3]}")
    print(f"  stages {dict(zip(stage_lbl, np.round(stage_ms, 2)))}  total {total:.2f} ms "
          f"(sum {sum(stage_ms):.2f})")


# ==========================================================================================
# drawing kit (PIL, 2x supersampled)
# ==========================================================================================
import style  # noqa: E402
import common  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

SS = 2
_FDIR = Path("/usr/share/fonts/opentype/inter")
_FONTS: dict = {}


def _font(size: int, weight: str = "Regular"):
    key = (size, weight)
    if key not in _FONTS:
        _FONTS[key] = ImageFont.truetype(str(_FDIR / f"Inter-{weight}.otf"), int(size * SS))
    return _FONTS[key]


def _rgba(hexcol: str, alpha: float = 1.0):
    r, g, b = common.hex2rgb(hexcol)
    return (r, g, b, int(round(255 * max(0.0, min(1.0, alpha)))))


# --------------------------------------------------------------------------- math type
# Math is the one place the film allows a second face (SPEC "Type").  Technique copied from
# `s05_motion.py` (same film, already shipping) and adapted to this shot's supersampled PIL
# canvas: matplotlib's `MathTextParser("agg")` typesets Computer Modern without a LaTeX
# subprocess and hands back the glyph coverage plus the box depth, so the maths can be
# cap-height-matched to the Inter line it sits in and placed on that line's *measured*
# baseline rather than an eyeballed offset.
_MATH: dict = {}


def _math_raster(expr: str, dpi: float):
    """(coverage [h,w] in 0..1, baseline row) for `expr` set in Computer Modern at `dpi`."""
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["mathtext.fontset"] = "cm"
    from matplotlib.font_manager import FontProperties
    from matplotlib.mathtext import MathTextParser
    r = MathTextParser("agg").parse(expr, dpi=dpi, prop=FontProperties(size=40))
    return np.asarray(r.image, np.float32) / 255.0, r.height - r.depth


def math_glyph(expr: str, size: int, weight: str = "Regular"):
    """(coverage [h,w] in *supersampled* px, baseline-from-top), matched to Inter `size`."""
    key = (expr, size, weight)
    if key not in _MATH:
        bb = _font(size, weight).getbbox("K")           # (x0, cap_top, x1, baseline), SS px
        cap_h = bb[3] - bb[1]
        # dpi that renders a Computer Modern "K" ~4x oversampled, then the exact scale
        # measured at *that* dpi (not extrapolated).
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


class Canvas:
    def __init__(self, w: int, h: int):
        self.w, self.h = w, h
        self.im = Image.new("RGB", (w * SS, h * SS), common.hex2rgb(style.BG))
        self.d = ImageDraw.Draw(self.im, "RGBA")

    def text(self, xy, s, size, color, weight="Regular", a=1.0, anchor="la"):
        if a <= 0.004:
            return
        self.d.text((xy[0] * SS, xy[1] * SS), s, font=_font(size, weight),
                    fill=_rgba(color, a), anchor=anchor)

    def tw(self, s, size, weight="Regular") -> float:
        return self.d.textlength(s, font=_font(size, weight)) / SS

    def mw(self, expr, size, weight="Regular") -> float:
        """Advance width of a maths run, in canvas px."""
        return math_glyph(expr, size, weight)[0].shape[1] / SS

    def math(self, xy, expr, size, color, weight="Regular", a=1.0) -> None:
        """Draw `expr` (a "$...$" string) on the Inter baseline of a "la"-anchored line."""
        if a <= 0.004:
            return
        cov, base = math_glyph(expr, size, weight)
        baseline = xy[1] * SS + _font(size, weight).getbbox("K")[3]
        mask = Image.fromarray(
            np.clip(cov * float(min(max(a, 0.0), 1.0)) * 255.0 + 0.5, 0, 255).astype(np.uint8),
            "L")
        solid = Image.new("RGB", mask.size, common.hex2rgb(color))
        self.im.paste(solid, (int(round(xy[0] * SS)), int(round(baseline - base))), mask)

    def rich(self, xy, runs, size, color, weight="Regular", a=1.0, anchor="la") -> float:
        """Draw a mixed Inter / Computer-Modern line; runs starting with "$" are maths.

        Returns the total advance width, so callers can lay out what follows.
        """
        ws = [self.mw(r, size, weight) if r.startswith("$") else self.tw(r, size, weight)
              for r in runs]
        total = sum(ws)
        x = xy[0] - total if anchor[0] == "r" else (
            xy[0] - total / 2 if anchor[0] == "m" else xy[0])
        for r, w in zip(runs, ws):
            if r.startswith("$"):
                self.math((x, xy[1]), r, size, color, weight, a)
            else:
                self.text((x, xy[1]), r, size, color, weight, a=a)
            x += w
        return total

    def line(self, pts, color, w=2.0, a=1.0):
        if a <= 0.004:
            return
        self.d.line([(p[0] * SS, p[1] * SS) for p in pts], fill=_rgba(color, a),
                    width=max(1, int(round(w * SS))), joint="curve")

    def rect(self, box, fill=None, outline=None, w=2.0, r=0.0, a=1.0):
        x0, y0, x1, y1 = [v * SS for v in box]
        if x1 <= x0 or y1 <= y0:
            return
        kw = {}
        if fill is not None:
            kw["fill"] = _rgba(fill, a)
        if outline is not None:
            kw["outline"] = _rgba(outline, a)
            kw["width"] = max(1, int(round(w * SS)))
        if r > 0:
            self.d.rounded_rectangle([x0, y0, x1, y1], radius=r * SS, **kw)
        else:
            self.d.rectangle([x0, y0, x1, y1], **kw)

    def dot(self, c, r, fill, a=1.0):
        self.d.ellipse([(c[0] - r) * SS, (c[1] - r) * SS, (c[0] + r) * SS, (c[1] + r) * SS],
                       fill=_rgba(fill, a))

    def frame(self) -> np.ndarray:
        return np.asarray(self.im.resize((self.w, self.h), Image.LANCZOS), dtype=np.uint8)


# ==========================================================================================
# layout
# ==========================================================================================
KICK_Y = 46
HEAD_Y, SUB_Y = 118, 178

AX_L, AX_R = 112, 880            # left panel
TRACK_L, TRACK_R = 150, 800
ROWS_Y = (314, 494, 674)

BX_L, BX_R = 1000, 1830          # right panel
BAR_Y, BAR_H = 430, 62
LEG_Y = 566
LEG_DY = 58

# (label, unit, lo, hi, better_is_low)
ROW_SPEC = [
    ("Pose buffer ℋ invalid", "%", 10.0, 20.0, True),
    ("Motion prediction valid", "%", 60.0, 80.0, False),
    ("MPJPE", "mm", 50.0, 60.0, True),
]
ROW_SPEC[0] = ("Invalid pose buffers", "%", 10.0, 20.0, True)


def render(args) -> None:
    if not CACHE.exists():
        raise SystemExit(f"missing cache {CACHE} -- run with --prepare first")
    D = np.load(CACHE, allow_pickle=False)
    before, after = D["abl_before"], D["abl_after"]
    stage_ms = D["stage_ms"]
    stage_lbl = [str(x) for x in D["stage_lbl"]]
    total = float(D["total_ms"])
    n_steps = int(D["n_steps"])

    dur = args.duration
    tA = 0.03 * dur
    tR = (0.11 * dur, 0.24 * dur, 0.37 * dur)     # the three dumbbell rows
    tB = 0.46 * dur
    tBar = 0.52 * dur
    tLeg = 0.64 * dur

    with common.FrameWriter(args) as fw:
        for t in fw.times():
            c = Canvas(args.width, args.height)
            c.text((112, KICK_Y), "Pipeline results",
                   style.CAPTION_SIZE, style.FG_MUTED, "Medium",
                   a=common.seg(t, 0.0, 0.05 * dur))
            _panel_ablation(c, t, dur, tA, tR, before, after)
            _panel_runtime(c, t, dur, tB, tBar, tLeg, stage_ms, stage_lbl, total, n_steps)
            fw.write(c.frame())


def _panel_ablation(c, t, dur, tA, tR, before, after):
    a = common.seg(t, tA, tA + 0.06 * dur)
    if a <= 0.004:
        return
    c.text((AX_L, HEAD_Y), "Reusing the last prediction", style.H1_SIZE, style.FG,
           "SemiBold", a=a)
    c.text((AX_L, SUB_Y), "H36M test · full pipeline · 49 249 frames",
           style.CAPTION_SIZE, style.FG_MUTED, a=a)

    # legend for the two states -- N_req set as real maths, as in the manuscript
    c.dot((AX_L + 8, 236), 7.0, style.FG_MUTED, a=a)
    w1 = c.rich((AX_L + 24, 223), [r"$N_{\mathrm{req}} = 50$", "   no reuse"],
                25, style.FG_MUTED, a=a)
    x2 = AX_L + 24 + w1 + 44
    c.dot((x2 + 8, 236), 7.0, style.C_OURS, a=a)
    c.rich((x2 + 24, 223), [r"$N_{\mathrm{req}} = 3$", "   ours"],
           25, style.C_OURS, "Medium", a=a)

    for i, (y0, (name, unit, lo, hi, low_good)) in enumerate(zip(ROWS_Y, ROW_SPEC)):
        ai = common.ease(common.seg(t, tR[i], tR[i] + 0.10 * dur), "out")
        if ai <= 0.004:
            continue
        b, f = float(before[i]), float(after[i])
        xb = TRACK_L + (b - lo) / (hi - lo) * (TRACK_R - TRACK_L)
        xf = TRACK_L + (f - lo) / (hi - lo) * (TRACK_R - TRACK_L)
        yt = y0 + 96

        c.text((AX_L, y0), name, 28, style.FG, "Medium", a=ai)
        # delta chip
        # relative change, the same statistic the repo reports in
        # results/final/full_pipeline/full_pipeline_sentence.tex (-23.8 % / +1.9 %)
        d_rel = (f - b) / b * 100.0
        chip = f"{d_rel:+.1f} %".replace("-", "\u2212")
        good = (d_rel < 0) if low_good else (d_rel > 0)
        col = style.C_TRUTH if good else style.YELLOW
        wch = c.tw(chip, 26, "SemiBold") + 28
        c.rect((AX_R - wch, y0 - 4, AX_R, y0 + 38), fill=col, r=9, a=0.15 * ai)
        c.text((AX_R - wch / 2, y0 + 17), chip, 26, col, "SemiBold", a=ai, anchor="mm")

        # track
        c.line([(TRACK_L, yt), (TRACK_R, yt)], style.GRID, 2.0, a=0.8 * ai)
        for x, lab in ((TRACK_L, f"{lo:g}"), (TRACK_R, f"{hi:g}")):
            c.line([(x, yt - 5), (x, yt + 5)], style.GRID, 2.0, a=0.8 * ai)
        c.text((TRACK_L, yt + 12), f"{lo:g}", 20, style.FG_MUTED, a=0.8 * ai, anchor="ma")
        c.text((TRACK_R, yt + 12), f"{hi:g} {unit}", 20, style.FG_MUTED, a=0.8 * ai,
               anchor="ma")

        xnow = xb + (xf - xb) * ai
        c.line([(xb, yt), (xnow, yt)], style.C_OURS, 4.0, a=0.55 * ai)
        c.dot((xb, yt), 9.0, style.FG_MUTED, a=ai)
        c.dot((xnow, yt), 10.5, style.C_OURS, a=ai)
        c.text((xb, yt + 12), f"{b:.2f}", 24, style.FG_MUTED, a=ai, anchor="ma")
        c.text((xnow, yt - 46), f"{f:.2f}", 32, style.C_OURS, "SemiBold", a=ai, anchor="ma")


def _panel_runtime(c, t, dur, tB, tBar, tLeg, stage_ms, stage_lbl, total, n_steps):
    a = common.seg(t, tB, tB + 0.06 * dur)
    if a <= 0.004:
        return
    c.line([(952, 96), (952, 880)], style.GRID, 2, a=0.8 * a)
    c.text((BX_L, HEAD_Y), "One pipeline step", style.H1_SIZE, style.FG, "SemiBold", a=a)
    c.text((BX_L, SUB_Y), f"median of {n_steps} steps · one NVIDIA RTX 5090",
           style.CAPTION_SIZE, style.FG_MUTED, a=a)
    c.text((BX_R, HEAD_Y - 18), f"{total:.2f} ms", 62, style.FG, "Bold", a=a, anchor="ra")
    c.text((BX_R, SUB_Y - 2), f"{1000.0 / total:.0f} Hz closed loop", 26, style.FG_MUTED,
           a=a, anchor="ra")

    ab = common.ease(common.seg(t, tBar, tBar + 0.16 * dur), "out")
    W = BX_R - BX_L
    gap = 3.0
    x = BX_L
    ood_marks = []
    for i, (ms, (_, _runs, role)) in enumerate(zip(stage_ms, STAGES)):
        w = ms / total * (W - gap * (len(stage_ms) - 1)) * ab
        col = getattr(style, role)
        c.rect((x, BAR_Y, x + w, BAR_Y + BAR_H), fill=col, r=5,
               a=0.90 if role != "C_ROBOT" else 0.55)
        if role == "C_OOD":
            ood_marks.append((x, x + w))
        x += w + gap * ab

    # the two monitors, called out under the bar
    am = common.seg(t, tBar + 0.12 * dur, tBar + 0.20 * dur)
    if am > 0.004 and ood_marks:
        ood_ms = sum(ms for ms, (_, _, role) in zip(stage_ms, STAGES) if role == "C_OOD")
        for (x0, x1) in ood_marks:
            c.rect((x0, BAR_Y + BAR_H + 9, x1, BAR_Y + BAR_H + 13), fill=style.C_OOD,
                   r=2, a=0.9 * am)
        c.text((BX_L, BAR_Y - 46),
               f"OOD monitoring  {ood_ms:.2f} ms  ·  {100 * ood_ms / total:.0f} % of the step",
               25, style.C_OOD, "Medium", a=am)

    al = common.seg(t, tLeg, tLeg + 0.10 * dur)
    if al <= 0.004:
        return
    for i, (ms, (_, runs, role)) in enumerate(zip(stage_ms, STAGES)):
        col = getattr(style, role)
        cx = BX_L + (i % 2) * 420
        cy = LEG_Y + (i // 2) * LEG_DY
        c.rect((cx, cy + 8, cx + 14, cy + 22), fill=col, r=3, a=al)
        c.rich((cx + 26, cy), runs, 25, style.FG, a=al)
        c.text((cx + 380, cy), f"{ms:.2f}", 25, style.FG_MUTED, a=al, anchor="ra")
    yb = LEG_Y + 3 * LEG_DY
    c.line([(BX_L, yb - 8), (BX_L + 400, yb - 8)], style.GRID, 2, a=al)
    c.text((BX_L + 26, yb + 2), "total", 25, style.FG, "SemiBold", a=al)
    c.text((BX_L + 380, yb + 2), f"{total:.2f}", 25, style.FG, "SemiBold", a=al, anchor="ra")
    c.text((BX_L + 392, yb + 2), "ms", 25, style.FG_MUTED, a=al)


def main() -> None:
    if "--prepare" in sys.argv:
        prepare()
        return
    args = common.shot_args("s15 -- OOD handling + runtime")
    render(args)


if __name__ == "__main__":
    main()
