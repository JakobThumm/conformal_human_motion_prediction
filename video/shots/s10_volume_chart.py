#!/usr/bin/env python
"""s10 -- the headline comparison: prediction-set volume and miss-rate (results).

Narration: "Across the test set our sets are seven point six times smaller, and the truth
escapes them one point six times in ten thousand."

Everything on screen is computed from the real H36M test run, not from a table:

  * per-sphere set volumes -- results/final/conformal_prediction_sets/
    motion_prediction_results_test.cloudpickle  (predictions / targets / covariance_matrices /
    last_input_poses, 59 472 windows x 10 horizon steps x 13 joints).
    - ours   r = conformal_set_radius(model_cov, input_cov, calibrator) with
             models/motion_prediction/conformal_calibration/conformal_calibrator.npz
             (conditional conformal, 1 - eps = 99.99 %), V = 4/3 pi r^3.
    - ISO 13855 baseline: eval_utils.compute_sara_predictions with v_h,max = 2.0 m/s
      (V_HUMAN_ISO) around the last input pose, same 99.99 % input-uncertainty sphere.
    Padding-masked exactly like utils.eval_utils.simple_coverage_stats_sara.
    The resulting 5/50/95 percentiles reproduce the committed CSVs to 6 decimals:
      ours 0.015309 / 0.090136 / 0.653729 m^3  (coverage_stats_conformal_prediction_sets.csv)
      ISO  0.016880 / 0.685770 / 3.243962 m^3  (coverage_stats_sara.csv)
    and manuscript Table `figures/result_tables/all_conformal_prediction_results.tex`.
  * miss-rate = fraction of valid (window, step, joint) spheres whose ground-truth joint lies
    outside the sphere -> ours 1.6e-4, ISO 5.0e-6 (same table, "Miss-rate" column).
  * PFH_D footnote values 9.50e-7 / 1.62e-7 per hour: same manuscript table.

Prepare (repo venv; writes video/media/s10_volume.npz, ~20 kB):

    cd /home/thumm/code/conformal_human_motion_prediction
    XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cpu \
      .venv/bin/python video/shots/s10_volume_chart.py --prepare

Render (conda env chmp-video):

    /home/thumm/miniconda3/envs/chmp-video/bin/python video/shots/s10_volume_chart.py \
        --out video/build/shots/s10.mp4 --duration 9.0
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

VIDEO_DIR = Path(__file__).resolve().parents[1]
REPO = VIDEO_DIR.parent
sys.path.insert(0, str(VIDEO_DIR))

CACHE = VIDEO_DIR / "media" / "s10_volume.npz"

LOG_LO, LOG_HI, NBIN = -2.7, 1.4, 320
# PFH_D per hour, manuscript figures/result_tables/all_conformal_prediction_results.tex
PFHD_OURS, PFHD_ISO = 9.50e-7, 1.62e-7
# Miss-rates AS PRINTED in that table (= 100 % - the 4-decimal coverage of
# results/final/conformal_prediction_sets/coverage_stats_{conformal_prediction_sets,sara}.csv).
# The live recomputation in --prepare gives 1.626e-4 and 4.527e-6; the ISO figure differs from
# the table only through that 4-decimal rounding of the coverage percentage.  The table value is
# shown so the film and the paper carry the same digits.
MISS_OURS, MISS_ISO = 1.6e-4, 5.0e-6


def prepare() -> None:
    import cloudpickle
    sys.path.insert(0, str(REPO / "src"))
    from conformal_human_motion_prediction.motion_prediction.inference_helper import (
        load_conformal_calibrator, conformal_set_radius)
    from conformal_human_motion_prediction.utils.eval_utils import (
        convert_covariance_matrices_to_set, compute_sara_predictions)
    from conformal_human_motion_prediction.motion_prediction.h36m_settings import (
        N_JOINTS, SET_LIKELIHOOD, V_HUMAN_ISO, PREDICTION_HORIZON_LENGTH)
    FPS_CAM = 25.0     # examples/motion_prediction.py :: FPS

    src = REPO / "results/final/conformal_prediction_sets/motion_prediction_results_test.cloudpickle"
    with open(src, "rb") as f:
        d = cloudpickle.load(f)
    pred = np.asarray(d["predictions"])
    tgt = np.asarray(d["targets"])
    cov = np.asarray(d["covariance_matrices"])
    li = np.asarray(d["last_input_poses"])
    in_cov = li[..., N_JOINTS * 3:N_JOINTS * 3 + N_JOINTS * 9].reshape(-1, N_JOINTS, 3, 3)
    lip = li[..., :N_JOINTS * 3].reshape(-1, N_JOINTS, 3)

    calib = load_conformal_calibrator(
        str(REPO / "models/motion_prediction/conformal_calibration/conformal_calibrator.npz"))
    r_ours = conformal_set_radius(cov, in_cov, calib)                       # [N,T,J] mm

    times = [(t + 1) / FPS_CAM for t in range(PREDICTION_HORIZON_LENGTH)]
    in_unc_m = convert_covariance_matrices_to_set(in_cov, likelihood=SET_LIKELIHOOD) / 1000.0
    sara_pred, r_iso = compute_sara_predictions(lip, times, V_HUMAN_ISO, in_unc_m)

    mask = np.logical_or(np.all(pred == 0.0, axis=(2, 3)), np.all(tgt == 0.0, axis=(2, 3)))
    full = np.repeat(mask[:, :, None], pred.shape[2], axis=-1)              # True = padding
    keep = ~full

    out = {}
    for name, r, p in (("ours", r_ours, pred), ("iso", r_iso, sara_pred)):
        v = 4.0 / 3.0 * np.pi * (r[keep] / 1000.0) ** 3
        h, _ = np.histogram(np.log10(np.clip(v, 1e-12, None)), bins=NBIN,
                            range=(LOG_LO, LOG_HI))
        dist = np.linalg.norm(p - tgt, axis=-1)[keep]
        miss = float((dist > r[keep]).mean())
        out[f"{name}_hist"] = h.astype(np.int64)
        out[f"{name}_p5"], out[f"{name}_p50"], out[f"{name}_p95"] = np.percentile(v, [5, 50, 95])
        out[f"{name}_miss"] = miss
        out[f"{name}_n_miss"] = float((dist > r[keep]).sum())
        print(f"{name:5s} n={v.size}  p5/p50/p95 = "
              f"{np.percentile(v, 5):.6f} / {np.percentile(v, 50):.6f} / "
              f"{np.percentile(v, 95):.6f} m^3   miss={miss:.3e} "
              f"({int((dist > r[keep]).sum())} spheres)")
    out["n_spheres"] = float(keep.sum())
    out["n_windows"] = float(pred.shape[0])
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, **out)
    print(f"wrote {CACHE}")


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

    def line(self, pts, color, w=2.0, a=1.0):
        if a <= 0.004:
            return
        self.d.line([(p[0] * SS, p[1] * SS) for p in pts], fill=_rgba(color, a),
                    width=max(1, int(round(w * SS))), joint="curve")

    def dash(self, p0, p1, color, w=2.0, a=1.0, dash=10.0, gap=8.0):
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        L = float(np.hypot(*(p1 - p0)))
        if L < 1e-6:
            return
        u = (p1 - p0) / L
        t = 0.0
        while t < L:
            t2 = min(t + dash, L)
            self.line([tuple(p0 + u * t), tuple(p0 + u * t2)], color, w, a)
            t = t2 + gap

    def rect(self, box, fill=None, outline=None, w=2.0, r=0.0, a=1.0):
        x0, y0, x1, y1 = [v * SS for v in box]
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

    def poly(self, pts, fill, a=1.0):
        if a <= 0.004 or len(pts) < 3:
            return
        self.d.polygon([(p[0] * SS, p[1] * SS) for p in pts], fill=_rgba(fill, a))

    def dot(self, c, r, fill, a=1.0):
        self.d.ellipse([(c[0] - r) * SS, (c[1] - r) * SS, (c[0] + r) * SS, (c[1] + r) * SS],
                       fill=_rgba(fill, a))

    def frame(self) -> np.ndarray:
        return np.asarray(self.im.resize((self.w, self.h), Image.LANCZOS), dtype=np.uint8)


def smooth(h: np.ndarray, sigma: float = 4.0) -> np.ndarray:
    k = np.arange(-4 * int(sigma), 4 * int(sigma) + 1)
    g = np.exp(-0.5 * (k / sigma) ** 2)
    g /= g.sum()
    y = np.convolve(h.astype(float), g, mode="same")
    return y / max(y.max(), 1e-12)


# ==========================================================================================
# layout
# ==========================================================================================
KICK_Y = 46
HEAD_Y, SUB_Y = 118, 178

VX0, VX1 = 150, 1215              # volume axis
V_ISO_LBL = 236
V_ISO_BASE = 440
V_MED_Y = 482                     # the two median read-outs live in the gap between ridges
V_ARROW_Y = 542
V_OUR_LBL = 566
V_OUR_BASE = 758
RIDGE_H = 140
V_AXIS_Y = 796
V_CAP_Y = 858

MX0, MX1 = 1382, 1770             # miss-rate axis
M_DIV_X = 1300
M_HEAD_Y = 236
M_ISO_Y = 400
M_OUR_Y = 512
M_AXIS_Y = 586
M_CAP_Y = 672

MISS_LO, MISS_HI = -6.5, -3.3     # log10 miss-rate axis


def vx(v: float) -> float:
    return VX0 + (np.log10(max(v, 1e-12)) - LOG_LO) / (LOG_HI - LOG_LO) * (VX1 - VX0)


def mx(v: float) -> float:
    return MX0 + (np.log10(max(v, 1e-12)) - MISS_LO) / (MISS_HI - MISS_LO) * (MX1 - MX0)


def sci(v: float) -> str:
    e = int(np.floor(np.log10(v)))
    m = v / 10.0 ** e
    sup = str(abs(e)).translate(str.maketrans("0123456789", "⁰¹²³"
                                                            "⁴⁵⁶⁷"
                                                            "⁸⁹"))
    return f"{m:.1f} × 10⁻{sup}"


# ==========================================================================================
def render(args) -> None:
    if not CACHE.exists():
        raise SystemExit(f"missing cache {CACHE} -- run with --prepare in the repo venv first")
    D = np.load(CACHE)
    y_iso, y_our = smooth(D["iso_hist"], 5.0), smooth(D["ours_hist"], 5.0)
    xs = LOG_LO + (np.arange(NBIN) + 0.5) * (LOG_HI - LOG_LO) / NBIN
    p50_iso, p50_our = float(D["iso_p50"]), float(D["ours_p50"])
    p5_iso, p95_iso = float(D["iso_p5"]), float(D["iso_p95"])
    p5_our, p95_our = float(D["ours_p5"]), float(D["ours_p95"])
    miss_iso, miss_our = MISS_ISO, MISS_OURS
    assert abs(np.log10(float(D["ours_miss"]) / MISS_OURS)) < 0.05
    assert abs(np.log10(float(D["iso_miss"]) / MISS_ISO)) < 0.06
    ratio = p50_iso / p50_our
    n_sph = int(D["n_spheres"])

    dur = args.duration
    t_head = 0.02 * dur
    t_i0, t_i1 = 0.07 * dur, 0.33 * dur
    t_o0, t_o1 = 0.15 * dur, 0.41 * dur
    t_med = 0.36 * dur
    t_hero = 0.44 * dur
    t_miss = 0.60 * dur

    with common.FrameWriter(args) as fw:
        for t in fw.times():
            c = Canvas(args.width, args.height)
            c.text((112, KICK_Y), "Results · how tight are the sets?",
                   style.CAPTION_SIZE, style.FG_MUTED, "Medium",
                   a=common.seg(t, 0.0, 0.05 * dur))
            fh = common.seg(t, t_head, t_head + 0.06 * dur)
            c.text((112, HEAD_Y), "Reachable-occupancy volume per joint",
                   style.H1_SIZE, style.FG, "SemiBold", a=fh)
            c.text((112, SUB_Y),
                   "H36M test · " + f"{n_sph:,}".replace(",", " ")
                   + " joint spheres · 1 − ε = 99.99 %",
                   style.CAPTION_SIZE, style.FG_MUTED, a=fh)

            _axis(c, t, dur, t_i0)
            _ridge(c, xs, y_iso, V_ISO_BASE, style.C_ISO,
                   common.ease(common.seg(t, t_i0, t_i1), "out"))
            _ridge(c, xs, y_our, V_OUR_BASE, style.C_OURS,
                   common.ease(common.seg(t, t_o0, t_o1), "out"))

            ai = common.seg(t, t_i0, t_i0 + 0.05 * dur)
            _series_label(c, V_ISO_LBL, style.C_ISO, "ISO 13855",
                          "constant velocity, vₕ = 2.0 m/s", ai)
            ao = common.seg(t, t_o0, t_o0 + 0.05 * dur)
            _series_label(c, V_OUR_LBL, style.C_OURS, "Ours",
                          "conformal prediction sets", ao)

            am = common.ease(common.seg(t, t_med, t_med + 0.09 * dur), "out")
            _median(c, p50_iso, V_ISO_BASE, y_iso, xs, style.C_ISO, am, up=True)
            _median(c, p50_our, V_OUR_BASE, y_our, xs, style.C_OURS, am, up=False)
            _spread(c, p5_iso, p95_iso, V_ISO_BASE, style.C_ISO, am)
            _spread(c, p5_our, p95_our, V_OUR_BASE, style.C_OURS, am)

            _hero(c, t, dur, t_hero, p50_iso, p50_our, ratio)

            acap = common.seg(t, t_hero + 0.06 * dur, t_hero + 0.12 * dur)
            c.text((150, V_CAP_Y),
                   "distribution of every joint sphere   ·   whisker: 5th – 95th percentile",
                   23, style.FG_MUTED, a=acap)

            _miss_panel(c, t, dur, t_miss, miss_iso, miss_our)
            fw.write(c.frame())


def _axis(c, t, dur, t0):
    a = common.seg(t, t0, t0 + 0.05 * dur)
    c.line([(VX0, V_AXIS_Y), (VX1, V_AXIS_Y)], style.GRID, 2, a=a)
    for e, lab in ((-2, "0.01"), (-1, "0.1"), (0, "1"), (1, "10")):
        x = vx(10.0 ** e)
        c.dash((x, V_ISO_BASE - RIDGE_H - 16), (x, V_ISO_BASE), style.GRID, 1.6,
               a=0.45 * a, dash=5, gap=9)
        c.dash((x, V_OUR_BASE - RIDGE_H - 16), (x, V_OUR_BASE), style.GRID, 1.6,
               a=0.45 * a, dash=5, gap=9)
        c.line([(x, V_AXIS_Y), (x, V_AXIS_Y + 7)], style.GRID, 2, a=a)
        c.text((x, V_AXIS_Y + 14), lab, 22, style.FG_MUTED, a=a, anchor="ma")
    c.text((VX1 + 14, V_AXIS_Y + 14), "m³", 22, style.FG_MUTED, a=a, anchor="la")


def _ridge(c, xs, ys, base, col, prog):
    if prog <= 0.004:
        return
    pts = [(VX0, base)]
    for i in range(len(xs)):
        x = VX0 + (xs[i] - LOG_LO) / (LOG_HI - LOG_LO) * (VX1 - VX0)
        pts.append((x, base - ys[i] * RIDGE_H * prog))
    pts.append((VX1, base))
    c.poly(pts, col, a=0.22 * prog)
    c.line(pts[1:-1], col, 2.8, a=0.95 * prog)
    c.line([(VX0, base), (VX1, base)], col, 1.4, a=0.35 * prog)


def _series_label(c, y, col, name, sub, a):
    if a <= 0.004:
        return
    c.rect((150, y + 10, 164, y + 24), fill=col, r=3, a=a)
    c.text((180, y), name, 32, col, "SemiBold", a=a)
    c.text((180 + c.tw(name, 32, "SemiBold") + 16, y + 6), "· " + sub, 24,
           style.FG_MUTED, a=a)


def _median(c, med, base, ys, xs, col, a, up=True):
    """Median tick on the ridge + its value, read out in the gap band between the ridges."""
    if a <= 0.004:
        return
    x = vx(med)
    i = int(np.clip((np.log10(med) - LOG_LO) / (LOG_HI - LOG_LO) * len(xs), 0, len(xs) - 1))
    top = base - ys[i] * RIDGE_H
    c.line([(x, base), (x, top)], style.BG, 5.0, a=a)
    c.line([(x, base), (x, top)], col, 2.6, a=a)
    c.dot((x, top), 6.0, col, a=a)
    if up:
        c.dash((x, V_MED_Y + 40), (x, base), col, 1.8, a=0.55 * a, dash=6, gap=7)
    else:
        # two segments so the guide never crosses the series label that sits between them
        c.dash((x, V_MED_Y + 44), (x, V_OUR_LBL - 10), col, 1.8, a=0.55 * a, dash=6, gap=7)
        c.dash((x, V_OUR_LBL + 44), (x, top), col, 1.8, a=0.55 * a, dash=6, gap=7)
    c.text((x, V_MED_Y), f"{med:.3f} m³", 32, col, "SemiBold", a=a, anchor="ma")
    c.text((x, V_MED_Y - 26), "median", 21, style.FG_MUTED, a=0.9 * a, anchor="ma")


def _spread(c, p5, p95, base, col, a):
    if a <= 0.004:
        return
    y = base + 17
    c.line([(vx(p5), y), (vx(p95), y)], col, 3.0, a=0.5 * a)
    for p in (p5, p95):
        c.line([(vx(p), y - 6), (vx(p), y + 6)], col, 2.4, a=0.7 * a)
    c.text((vx(p5) - 12, y - 11), f"{p5:.3f}", 21, style.FG_MUTED, a=0.85 * a, anchor="ra")
    c.text((vx(p95) + 12, y - 11), f"{p95:.3f}", 21, style.FG_MUTED, a=0.85 * a, anchor="la")


def _hero(c, t, dur, t0, p50_iso, p50_our, ratio):
    """The one number: a stat tile top-right of the chart, plus the arrow joining the medians."""
    a = common.ease(common.seg(t, t0, t0 + 0.10 * dur), "out")
    if a <= 0.004:
        return
    c.text((VX1, HEAD_Y - 18), f"{ratio:.1f}×", 62, style.C_OURS, "Bold", a=a, anchor="ra")
    c.text((VX1, SUB_Y - 2), "smaller median volume", 26, style.FG, a=a, anchor="ra")
    xi, xo = vx(p50_iso), vx(p50_our)
    xa = xo + (xi - xo) * (1.0 - a)
    c.line([(xi, V_ARROW_Y), (xa, V_ARROW_Y)], style.FG_MUTED, 2.2, a=0.8 * a)
    c.poly([(xa, V_ARROW_Y), (xa + 14, V_ARROW_Y - 7), (xa + 14, V_ARROW_Y + 7)],
           style.FG_MUTED, a=0.8 * a)


def _miss_panel(c, t, dur, t0, miss_iso, miss_our):
    a = common.seg(t, t0, t0 + 0.06 * dur)
    if a <= 0.004:
        return
    x0 = MX0 - 42
    c.line([(M_DIV_X, 214), (M_DIV_X, 760)], style.GRID, 2, a=0.8 * a)
    c.text((x0, M_HEAD_Y), "Miss-rate", 34, style.FG, "SemiBold", a=a)
    c.text((x0, M_HEAD_Y + 46), "ground truth outside the set", 23, style.FG_MUTED, a=a)

    ab = common.ease(common.seg(t, t0 + 0.04 * dur, t0 + 0.16 * dur), "out")
    for y, val, col, name in ((M_ISO_Y, miss_iso, style.C_ISO, "ISO 13855"),
                              (M_OUR_Y, miss_our, style.C_OURS, "Ours")):
        c.text((x0, y - 46), name, 24, col, "Medium", a=a)
        c.text((x0 + c.tw(name, 24, "Medium") + 18, y - 52), sci(val), 32, col,
               "SemiBold", a=ab)
        c.line([(x0, y), (MX1, y)], style.GRID, 2.0, a=0.7 * a)
        c.dot((x0 + (mx(val) - x0) * ab, y), 9.5, col, a=a)

    c.line([(x0, M_AXIS_Y), (MX1, M_AXIS_Y)], style.GRID, 2, a=a)
    for e, lab in ((-6, "10⁻⁶"), (-5, "10⁻⁵"), (-4, "10⁻⁴")):
        x = mx(10.0 ** e)
        c.line([(x, M_AXIS_Y), (x, M_AXIS_Y + 7)], style.GRID, 2, a=a)
        c.text((x, M_AXIS_Y + 14), lab, 22, style.FG_MUTED, a=a, anchor="ma")

    ac = common.seg(t, t0 + 0.14 * dur, t0 + 0.22 * dur)
    c.text((x0, M_CAP_Y), "Ours misses more often.", 25, style.FG, "Medium", a=ac)
    c.text((x0, M_CAP_Y + 38), "Both clear the safety bar:", 24, style.FG_MUTED, a=ac)
    c.text((x0, M_CAP_Y + 72), "PFH_D  9.50 × 10⁻⁷ / h  (ours)", 23, style.FG_MUTED, a=ac)
    c.text((x0, M_CAP_Y + 104), "PFH_D  1.62 × 10⁻⁷ / h  (ISO 13855)", 23,
           style.FG_MUTED, a=ac)


def main() -> None:
    if "--prepare" in sys.argv:
        prepare()
        return
    args = common.shot_args("s10 -- prediction-set volume + miss-rate")
    render(args)


if __name__ == "__main__":
    main()
