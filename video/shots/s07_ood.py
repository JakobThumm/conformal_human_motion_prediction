#!/usr/bin/env python
"""s07 -- the out-of-distribution monitor (method).

Narration: "Conformal guarantees assume tomorrow looks like yesterday.  A sketched-Lanczos
monitor watches the networks' gradients, and out of distribution, we reuse the previous
prediction instead of guessing."

Three beats, all driven by REAL repository data (see `--prepare` below):

  (0) the manuscript's pipeline figure (`_overview.py`) with the two OOD monitors -- the
      "Pose monitor" and "Motion monitor" trapezoids -- lit and the rest of the pipeline
      dimmed, so the viewer knows which block the shot is about.  It then dissolves into
  (a) OOD scoring with sketched-Lanczos -- the SLU score of the 2-D pose network on
      in-distribution H36M test frames vs out-of-distribution tiger-pose frames, with the
      threshold tau_2D, and
  (b) OOD handling (Algorithm 1, manuscript content/S3_methodology.tex) animated over a REAL
      12-frame H36M test episode in which SLU_2D flags three consecutive frames.

Data sources (every number on screen is traceable):
  * ID scores      results/final/full_pipeline/n_correct_poses_required_3/
                   full_pipeline_results.cloudpickle -> 'poses_3d_ood_scores'  (49 249 frames)
  * OOD scores     results/pose_prediction_ood/pose_ood_scores.cloudpickle -> 'OOD (tiger-pose)'
  * tau_2D = 0.07  the calibration condition of the manuscript is
                   P(SLU <= tau) >= 1 - eps_OOD with eps_OOD ~ 10 %; tau = 0.07 keeps
                   98.77 % of the 49 249 H36M test frames below it (so the condition holds
                   with margin) and flags 93.2 % of the tiger-pose frames.  The value
                   shipped in pose_estimation/h36m_settings.py :: OOD_THRESHOLD is 0.2.
  * AUROC          recomputed in --prepare from the two real score sets (rank statistic);
                   threshold-independent.
  * episode        the same full-pipeline run, frames 40 413..40 424 ('poses_3d_ood_scores',
                   'poses_3d_is_ood', 'motions_is_ood'), N_req = 3 from
                   motion_prediction/h36m_settings.py (N_CORRECT_POSES_REQUIRED).
                   The run's own `poses_3d_is_ood` over this window is EXACTLY
                   (score > 0.07), so the animated flag pattern is the deployed pipeline's.

Prepare (repo venv, writes video/media/s07_ood.npz, ~200 kB):

    cd /home/thumm/code/conformal_human_motion_prediction
    XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cpu \
      .venv/bin/python video/shots/s07_ood.py --prepare

Render (conda env chmp-video):

    /home/thumm/miniconda3/envs/chmp-video/bin/python video/shots/s07_ood.py \
        --out video/build/shots/s07.mp4 --duration 9.5
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

VIDEO_DIR = Path(__file__).resolve().parents[1]
REPO = VIDEO_DIR.parent
sys.path.insert(0, str(VIDEO_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

CACHE = VIDEO_DIR / "media" / "s07_ood.npz"

# --- constants ----------------------------------------------------------------------------
# tau_2D: the manuscript calibrates tau so that P(SLU <= tau) >= 1 - eps_OOD, eps_OOD ~ 10 %.
# 0.07 keeps 98.77 % of the H36M test frames below it and flags 93.2 % of the tiger-pose
# frames; the deployed constant in h36m_settings.py :: OOD_THRESHOLD is the looser 0.2.
TAU_2D = 0.07
N_REQ = 3             # motion_prediction/h36m_settings.py :: N_CORRECT_POSES_REQUIRED
K_P = 10              # motion_prediction/h36m_settings.py :: PREDICTION_HORIZON_LENGTH
EP_START, EP_LEN = 40413, 12   # the real OOD episode (3 consecutive flagged frames)


# ==========================================================================================
# prepare
# ==========================================================================================
def prepare() -> None:
    import cloudpickle

    fp = (REPO / "results/final/full_pipeline/n_correct_poses_required_3"
                 "/full_pipeline_results.cloudpickle")
    with open(fp, "rb") as f:
        run = cloudpickle.load(f)
    id_scores = np.asarray(run["poses_3d_ood_scores"], dtype=np.float64).ravel()
    is_ood = np.asarray(run["poses_3d_is_ood"], dtype=bool)
    buf_good = np.asarray(run["pose_buffers_good"], dtype=bool)
    mot_is_ood = np.asarray(run["motions_is_ood"], dtype=bool).ravel()

    tp = REPO / "results/pose_prediction_ood/pose_ood_scores.cloudpickle"
    with open(tp, "rb") as f:
        tiger = cloudpickle.load(f)
    ood_scores = np.asarray(tiger["OOD (tiger-pose)"], dtype=np.float64).ravel()

    # AUROC = P(score_ood > score_id), rank statistic (ties counted at 1/2).
    both = np.concatenate([id_scores, ood_scores])
    order = np.argsort(both, kind="mergesort")
    ranks = np.empty_like(both)
    ranks[order] = np.arange(1, both.size + 1, dtype=np.float64)
    # average ranks for ties
    uniq, inv, cnt = np.unique(both, return_inverse=True, return_counts=True)
    sums = np.zeros(uniq.size)
    np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    n_i, n_o = id_scores.size, ood_scores.size
    auroc = (ranks[n_i:].sum() - n_o * (n_o + 1) / 2.0) / (n_i * n_o)

    a, b = EP_START, EP_START + EP_LEN
    ep_scores = id_scores[a:b]
    # The episode's flags are the threshold applied to the real scores.  Over this window the
    # run's own poses_3d_is_ood (deployed tau = 0.2, plus the covariance rule) agrees exactly,
    # which is why this episode was chosen -- see the assertion below.
    ep_ood = ep_scores > TAU_2D
    assert np.array_equal(ep_ood, is_ood[a:b]), "episode flags disagree with the run"
    # motion arrays only carry the frames whose pose buffer was full/good -> map the index.
    m0 = int(buf_good[:a].sum())
    ep_mot_ood = mot_is_ood[m0:m0 + EP_LEN] if m0 + EP_LEN <= mot_is_ood.size \
        else np.zeros(EP_LEN, bool)

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CACHE,
        id_scores=id_scores.astype(np.float32),
        ood_scores=ood_scores.astype(np.float32),
        ep_scores=ep_scores.astype(np.float32),
        ep_ood=ep_ood,
        ep_mot_ood=ep_mot_ood,
        ep_start=np.int64(EP_START),
        tau=np.float64(TAU_2D),
        auroc=np.float64(auroc),
        frac_id_below=np.float64((id_scores <= TAU_2D).mean()),
        frac_ood_above=np.float64((ood_scores > TAU_2D).mean()),
        id_median=np.float64(np.median(id_scores)),
        ood_median=np.float64(np.median(ood_scores)),
    )
    print(f"wrote {CACHE}")
    print(f"  ID  n={id_scores.size}  median={np.median(id_scores):.5f}  "
          f"p90={np.percentile(id_scores, 90):.5f}")
    print(f"  OOD n={ood_scores.size}  median={np.median(ood_scores):.5f}")
    print(f"  AUROC={auroc:.4f}   P(SLU<=tau)={100 * (id_scores <= TAU_2D).mean():.2f} %   "
          f"TPR={100 * (ood_scores > TAU_2D).mean():.2f} %")
    print(f"  episode scores={np.round(ep_scores, 3)}")
    print(f"  episode is_ood={ep_ood.astype(int)}  motion_is_ood={ep_mot_ood.astype(int)}")


# ==========================================================================================
# drawing kit (PIL, 2x supersampled)
# ==========================================================================================
import style  # noqa: E402
import common  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

SS = 2
_FDIR = Path("/usr/share/fonts/opentype/inter")
_FONTS: dict = {}


def _font_px(px: int, weight: str = "Regular"):
    """Inter at exactly `px` device pixels (cached).  One sans family in the whole film."""
    key = (int(px), weight)
    if key not in _FONTS:
        path = _FDIR / f"Inter-{weight}.otf"
        if not path.exists():
            path = _FDIR / "Inter-Regular.otf"
        _FONTS[key] = ImageFont.truetype(str(path), int(px))
    return _FONTS[key]


def _font(size: int, weight: str = "Regular"):
    return _font_px(int(size * SS), weight)


def _rgba(hexcol: str, alpha: float = 1.0):
    r, g, b = common.hex2rgb(hexcol)
    return (r, g, b, int(round(255 * max(0.0, min(1.0, alpha)))))


# --------------------------------------------------------------------------- math type
# Math is the one place the film allows a second face (SPEC "Type"): tau_2D, SLU_2D, f_mot,
# N_req, H and M are set in Computer Modern via matplotlib's mathtext, cap-height-matched to
# the Inter line they sit in, so the subscripts are real typesetting rather than baseline-
# shifted guesses.  Same technique as s05 (see s05_motion.py :: math_glyph).
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


def math_glyph(expr: str, px: int, weight: str = "Regular"):
    """(coverage [h,w], baseline-from-top) for `expr`, cap-height-matched to Inter at `px`."""
    key = (expr, int(px), weight)
    if key not in _MATH:
        bb = _font_px(px, weight).getbbox("K")            # (x0, cap_top, x1, baseline)
        cap_h = bb[3] - bb[1]
        # Cap height of a Computer Modern "K" at a reference dpi -> the dpi that renders it
        # ~4x oversampled, then the exact scale measured at *that* dpi (not extrapolated).
        a0, base0 = _math_raster(r"$K$", 200.0)
        dpi = 200.0 * (4.0 * cap_h) / (base0 - np.nonzero(a0.max(1) > 0.03)[0][0])
        a1, base1 = _math_raster(r"$K$", dpi)
        scale = cap_h / (base1 - np.nonzero(a1.max(1) > 0.03)[0][0])

        a, base = _math_raster(expr, dpi)
        cols = np.nonzero(a.max(0) > 0.03)[0]
        a = a[:, cols[0]:cols[-1] + 1]                    # crop the side bearings
        im = Image.fromarray((a * 255.0 + 0.5).astype(np.uint8), "L")
        w = max(1, int(round(im.width * scale)))
        h = max(1, int(round(im.height * scale)))
        cov = np.asarray(im.resize((w, h), Image.LANCZOS), np.float32) / 255.0
        _MATH[key] = (cov, base * h / a.shape[0])
    return _MATH[key]


class Canvas:
    """Thin PIL wrapper: logical 1920x1080 coordinates, 2x supersampled internally."""

    def __init__(self, w: int, h: int, bg=None):
        self.w, self.h = w, h
        if bg is None:
            self.im = Image.new("RGB", (w * SS, h * SS), common.hex2rgb(style.BG))
        else:
            self.im = Image.fromarray(np.ascontiguousarray(bg)).resize(
                (w * SS, h * SS), Image.BILINEAR)
        self.d = ImageDraw.Draw(self.im, "RGBA")

    def text(self, xy, s, size, color, weight="Regular", a=1.0, anchor="la"):
        if a <= 0.004:
            return
        self.d.text((xy[0] * SS, xy[1] * SS), s, font=_font(size, weight),
                    fill=_rgba(color, a), anchor=anchor)

    def tw(self, s, size, weight="Regular") -> float:
        return self.d.textlength(s, font=_font(size, weight)) / SS

    # ---- maths --------------------------------------------------------------------------
    def mw(self, expr, size, weight="Regular") -> float:
        """Advance width of a maths run, in logical px."""
        return math_glyph(expr, int(size * SS), weight)[0].shape[1] / SS

    def math(self, xy, expr, size, color, weight="Regular", a=1.0) -> float:
        """Draw `expr` with its baseline on Inter's baseline for a line anchored at `xy`."""
        cov, base = math_glyph(expr, int(size * SS), weight)
        if a <= 0.004:
            return cov.shape[1] / SS
        baseline = xy[1] * SS + _font(size, weight).getbbox("K")[3]
        col = Image.new("RGB", (cov.shape[1], cov.shape[0]), common.hex2rgb(color))
        mask = Image.fromarray(
            np.clip(cov * float(np.clip(a, 0.0, 1.0)) * 255.0 + 0.5, 0, 255).astype(np.uint8),
            "L")
        self.im.paste(col, (int(round(xy[0] * SS)), int(round(baseline - base))), mask)
        return cov.shape[1] / SS

    def rich(self, xy, parts, size, color, weight="Regular", a=1.0, anchor="l") -> float:
        """A single line mixing Inter prose and Computer-Modern maths.

        `parts` is a list of ("t"|"m", string[, colour]); `anchor` is "l" or "m" (centred).
        Returns the line width in logical px.
        """
        width = 0.0
        for p in parts:
            width += self.mw(p[1], size, weight) if p[0] == "m" else self.tw(p[1], size, weight)
        x = xy[0] - (width / 2.0 if anchor == "m" else 0.0)
        if a > 0.004:
            for p in parts:
                col = p[2] if len(p) > 2 else color
                if p[0] == "m":
                    x += self.math((x, xy[1]), p[1], size, col, weight, a)
                else:
                    self.text((x, xy[1]), p[1], size, col, weight, a)
                    x += self.tw(p[1], size, weight)
        return width

    # ---- primitives ---------------------------------------------------------------------
    def line(self, pts, color, w=2.0, a=1.0):
        if a <= 0.004:
            return
        self.d.line([(p[0] * SS, p[1] * SS) for p in pts], fill=_rgba(color, a),
                    width=max(1, int(round(w * SS))), joint="curve")

    def dash(self, p0, p1, color, w=2.0, a=1.0, dash=11.0, gap=8.0):
        p0 = np.asarray(p0, float)
        p1 = np.asarray(p1, float)
        L = float(np.hypot(*(p1 - p0)))
        if L < 1e-6:
            return
        u = (p1 - p0) / L
        t = 0.0
        while t < L:
            t2 = min(t + dash, L)
            self.line([tuple(p0 + u * t), tuple(p0 + u * t2)], color, w, a)
            t = t2 + gap

    def rect(self, box, fill=None, outline=None, w=2.0, r=0.0, a=1.0, ao=None):
        x0, y0, x1, y1 = [v * SS for v in box]
        kw = {}
        if fill is not None:
            kw["fill"] = _rgba(fill, a)
        if outline is not None:
            kw["outline"] = _rgba(outline, a if ao is None else ao)
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


def kde(values, lo, hi, n=260, sigma=5.0):
    """Histogram + gaussian smoothing on a log10 grid, normalised to its own peak."""
    h, edges = np.histogram(np.log10(np.clip(values, 1e-12, None)), bins=n, range=(lo, hi))
    k = np.arange(-4 * int(sigma), 4 * int(sigma) + 1)
    g = np.exp(-0.5 * (k / sigma) ** 2)
    g /= g.sum()
    y = np.convolve(h.astype(float), g, mode="same")
    y = y / max(y.max(), 1e-12)
    x = 0.5 * (edges[:-1] + edges[1:])
    return x, y


# ==========================================================================================
# maths strings (manuscript notation, content/S3_methodology.tex)
# ==========================================================================================
M_TAU = r"$\tau_{\mathrm{2D}}$"
M_SLU2D = r"$\mathrm{SLU}_{\mathrm{2D}}$"
M_H = r"$\mathcal{H}$"
M_M = r"$\mathcal{M}$"
M_M0 = r"$\mathcal{M}[0]$"
M_NREQ = r"$N_{\mathrm{req}}$"
M_ACCEPT = r"$\mathcal{M} \leftarrow f_{\mathrm{mot}}(\mathcal{H})$"
M_SHIFT = r"$\mathcal{M} \leftarrow \mathrm{shift}(\mathcal{M})$"
M_REUSE = r"$\mathcal{H} \leftarrow \mathcal{M}[0]$"
M_SUMV = r"$\sum v_i$"


# ==========================================================================================
# layout
# ==========================================================================================
KICK_Y = 46
KICK_X = 112
HEAD_SIZE = 42                   # both panel heads; matches the overview title

# SPEC "Motion discipline": words appear, pictures may move.  In this shot every head,
# label, caption, equation and axis tick enters on opacity alone over TXT_FADE seconds and
# nothing is animated out: the beat change is a HARD CUT (no "fall away", no cross-dissolve),
# which is also where the 1.4 s this shot gained went -- into reading time on both sides.
# What still moves: the highlight over the pipeline figure, the two ridges drawing in, the
# tau marker, the Algorithm-1 timeline stepping frame by frame and the buffer cells.
TXT_FADE = 0.20

# beat 0: the overview figure.  The two OOD monitors -- the "Pose monitor" and "Motion
# monitor" trapezoids -- measured on the 2100x1109 source figure; the named regions of
# _overview.py do not isolate them, so the shot passes its own box.
REGION_MONITORS = (536, 100, 1614, 538)
T_OV_IN = (0.000, 0.020)      # the figure fades up (a picture)
T_FOCUS = (0.030, 0.105)      # the rest of the pipeline dims, the two monitors light
T_SUB = 0.115                 # "Pose monitor . motion monitor" -- opacity only
T_CUT = 0.255                 # hard cut, overview -> the two panels

# panel A ---------------------------------------------------------------------------------
AX0, AX1 = 112, 872              # x span of the score axis
LOG_LO, LOG_HI = -3.55, 0.15     # log10 SLU score range
A_HEAD_Y = 126
A_LBL1_Y = 232
A_BASE1 = 432                    # ID ridge baseline
A_LBL2_Y = 470
A_BASE2 = 670                    # OOD ridge baseline
RIDGE_H = 132
A_AXIS_Y = 692
A_CAP_Y = 780

# panel B ---------------------------------------------------------------------------------
BX0 = 1000
B_PITCH, B_CELL = 60.0, 43.0
B_HEAD_Y = 126
B_S_LBL = 232
B_S_BASE = 398                   # score-bar baseline
B_S_H = 130                      # score-bar full height
B_H_LBL = 448
B_H_TOP, B_H_BOT = 482, 532
B_M_LBL = 600
B_M_TOP, B_M_BOT = 634, 684
B_STAT_Y = 748
B_CAP_Y = 820
B_SUMV_X = 1452                  # the "sum v_i . . . / N_req" read-out, on the validity row

M_CELL, M_GAP = 44.0, 10.0

BAR_LO, BAR_HI = -2.6, -0.2      # log10 range of the per-frame score bars


def sx(v: float) -> float:
    """SLU score -> panel-A x (log axis)."""
    return AX0 + (np.log10(max(v, 1e-12)) - LOG_LO) / (LOG_HI - LOG_LO) * (AX1 - AX0)


def bar_h(v: float) -> float:
    """SLU score -> panel-B bar height (log, clamped)."""
    f = (np.log10(max(v, 1e-12)) - BAR_LO) / (BAR_HI - BAR_LO)
    return float(np.clip(f, 0.0, 1.0)) * B_S_H


def cell_x(i: int) -> float:
    return BX0 + i * B_PITCH


# ==========================================================================================
# beat 0 -- the overview
# ==========================================================================================
def overview_stills(width: int, height: int):
    """(flat, lit, lit_sub) stills of the overview beat.

    `flat` and `lit` differ only in the pipeline figure's highlight -- a *graphic* move, the
    one the SPEC explicitly allows.  The headline is identical in all three, so cross-fading
    between them never moves or re-renders a word.  `lit_sub` adds the accent line, which is
    therefore the only thing the third blend can bring up: a pure opacity entry for text.
    """
    import _overview as OV

    def shell(focus: float, sub: bool) -> np.ndarray:
        img = Image.fromarray(np.array(OV.compose(
            REGION_MONITORS, reveal=1.0, focus=focus, kicker="", title="", caption="",
            accent=style.C_OOD, width=width, height=height)))
        d = ImageDraw.Draw(img, "RGBA")
        d.text((KICK_X, 92), "Two monitors, one on each network",
               font=_font_px(42, "Medium"), fill=_rgba(style.FG, 1.0))
        if sub:
            d.text((KICK_X, 146), "Pose monitor  ·  motion monitor",
                   font=_font_px(30, "Medium"), fill=_rgba(style.C_OOD, 1.0))
        return np.asarray(img).astype(np.float32)

    return shell(0.0, False), shell(1.0, False), shell(1.0, True)


# ==========================================================================================
# render
# ==========================================================================================
def render(args) -> None:
    if not CACHE.exists():
        raise SystemExit(f"missing cache {CACHE} -- run with --prepare in the repo venv first")
    D = np.load(CACHE, allow_pickle=False)
    id_s, ood_s = D["id_scores"].astype(float), D["ood_scores"].astype(float)
    ep_s, ep_ood = D["ep_scores"].astype(float), D["ep_ood"].astype(bool)
    tau = float(D["tau"])
    auroc = float(D["auroc"])
    n = ep_s.size

    x_id, y_id = kde(id_s, LOG_LO, LOG_HI, sigma=6.0)
    x_od, y_od = kde(ood_s, LOG_LO, LOG_HI, sigma=9.0)

    # --- Algorithm 1 state trace over the real episode ------------------------------------
    v = (~ep_ood).astype(int)                      # v_k: pose came from the image
    streak, accept, nan_after = [], [], []
    run = N_REQ                                    # steady state before the episode
    m_nan = 0
    for k in range(n):
        run = run + 1 if v[k] else 0
        st = min(run, N_REQ)
        ok = st >= N_REQ
        m_nan = 0 if ok else min(m_nan + 1, K_P)
        streak.append(st)
        accept.append(ok)
        nan_after.append(m_nan)
    streak, accept, nan_after = map(np.asarray, (streak, accept, nan_after))

    dur = args.duration
    # Every keyframe is a fraction of the shot length -- the narration may move.  The shot is
    # now 12.0 s (was 10.6): the overview holds ~0.7 s longer *and* is no longer eaten by a
    # dissolve, and the Algorithm-1 timeline gets ~0.9 s more to be read (5.9 s for 12 frames
    # instead of 5.0), which is where the author asked the extra time to go.
    tA0, tA1 = 0.262 * dur, 0.385 * dur    # ID ridge  (a picture: it draws in)
    tB0, tB1 = 0.320 * dur, 0.445 * dur    # OOD ridge
    tT0, tT1 = 0.400 * dur, 0.465 * dur    # tau marker
    tC0 = 0.480 * dur                      # panel-A caption
    tH0 = 0.390 * dur                      # panel-B headers
    tS0, tS1 = 0.455 * dur, 0.945 * dur    # the algorithm steps -> 0.66 s hold on the last

    w = np.ones(n)
    w[ep_ood] = 1.65
    resume = int(np.argmax(accept[int(np.argmin(v)):]) + int(np.argmin(v))) \
        if (~accept).any() else n
    for k in range(n):
        if not ep_ood[k] and k >= int(np.argmin(v)) and k <= resume:
            w[k] = 1.45
        elif k > resume:
            w[k] = 0.80
    edges = np.concatenate([[0.0], np.cumsum(w) / w.sum()])
    step_t = tS0 + edges * (tS1 - tS0)

    flat, lit, lit_sub = overview_stills(args.width, args.height)
    fade_u = TXT_FADE / dur                # text entry, as a fraction of the shot

    def overview(u: float) -> np.ndarray:
        a_in = common.seg(u, *T_OV_IN, "out")            # the figure (a picture) fades up
        a_focus = common.seg(u, *T_FOCUS, "smooth")      # the highlight travels over it
        a_sub = common.seg(u, T_SUB, T_SUB + fade_u, "out")   # the accent line: opacity only
        bg = np.full_like(flat, 0.0)
        bg[:] = np.array(common.hex2rgb(style.BG), np.float32)
        img = bg * (1.0 - a_in) + flat * a_in
        img = img * (1.0 - a_focus) + lit * a_focus
        return img * (1.0 - a_sub) + lit_sub * a_sub

    def panels(t: float) -> np.ndarray:
        c = Canvas(args.width, args.height)
        _panel_a(c, t, dur, x_id, y_id, x_od, y_od, tau, auroc,
                 id_s.size, ood_s.size, tA0, tA1, tB0, tB1, tT0, tT1, tC0)
        _panel_b(c, t, dur, ep_s, ep_ood, v, streak, accept, nan_after, step_t, tH0, resume)
        return c.frame().astype(np.float32)

    def with_kicker(img: np.ndarray, a: float) -> np.ndarray:
        """The shot's kicker, drawn at 1x on top of whatever the beat produced."""
        if a <= 0.004:
            return img
        pim = Image.fromarray(img)
        ImageDraw.Draw(pim, "RGBA").text(
            (KICK_X, KICK_Y), "Out-of-distribution monitor",
            font=_font_px(style.CAPTION_SIZE, "Medium"), fill=_rgba(style.FG_MUTED, a))
        return np.asarray(pim)

    with common.FrameWriter(args) as fw:
        for t in fw.times():
            u = t / dur
            # Hard cut between the two beats -- no cross-dissolve, nothing animates out.
            img = overview(u) if u < T_CUT else panels(t)
            fw.write(with_kicker(np.clip(img, 0, 255).astype(np.uint8),
                                 common.seg(t, 0.0, TXT_FADE, "out")))


def _panel_a(c, t, dur, x_id, y_id, x_od, y_od, tau, auroc, n_id, n_ood,
             tA0, tA1, tB0, tB1, tT0, tT1, tC0):
    fade = common.seg(t, T_CUT * dur, T_CUT * dur + TXT_FADE, "out")
    X0, X1 = AX0, AX1
    c.text((112, A_HEAD_Y), "OOD scoring with Sketched-Lanczos", HEAD_SIZE, style.FG,
           "SemiBold", a=fade)

    # axis
    aA = common.seg(t, tA0, tA0 + TXT_FADE, "out")
    c.line([(X0, A_AXIS_Y), (X1, A_AXIS_Y)], style.GRID, 2, a=aA)
    for e in (-3, -2, -1, 0):
        xx = sx(10.0 ** e)
        c.line([(xx, A_AXIS_Y), (xx, A_AXIS_Y + 7)], style.GRID, 2, a=aA)
        lab = {-3: "0.001", -2: "0.01", -1: "0.1", 0: "1"}[e]
        c.text((xx, A_AXIS_Y + 14), lab, 22, style.FG_MUTED, a=aA, anchor="ma")
    c.text(((X0 + X1) / 2, A_AXIS_Y + 46), "SLU score  (log scale)", 22, style.FG_MUTED,
           a=aA, anchor="ma")

    def ridge(xs, ys, base, col, prog):
        if prog <= 0.004:
            return
        pts = [(X0, base)]
        for i in range(len(xs)):
            xx = X0 + (xs[i] - LOG_LO) / (LOG_HI - LOG_LO) * (AX1 - AX0)
            pts.append((xx, base - ys[i] * RIDGE_H * prog))
        pts.append((X1, base))
        c.poly(pts, col, a=0.20 * prog)
        c.line(pts[1:-1], col, 2.6, a=0.95 * prog)
        c.line([(X0, base), (X1, base)], col, 1.4, a=0.35 * prog)

    pA = common.ease(common.seg(t, tA0, tA1), "out")
    pB = common.ease(common.seg(t, tB0, tB1), "out")
    ridge(x_id, y_id, A_BASE1, style.FG_MUTED, pA)
    ridge(x_od, y_od, A_BASE2, style.C_OOD, pB)

    lab_a = common.seg(t, tA0, tA0 + TXT_FADE, "out")
    c.rect((112, A_LBL1_Y + 9, 125, A_LBL1_Y + 22), fill=style.FG_MUTED, r=3, a=lab_a)
    c.text((139, A_LBL1_Y), "in distribution", 27, style.FG, "Medium", a=lab_a)
    c.text((139 + c.tw("in distribution", 27, "Medium") + 12, A_LBL1_Y + 3),
           "· H36M test, " + f"{n_id:,}".replace(",", " ") + " frames",
           23, style.FG_MUTED, a=lab_a)
    lab_b = common.seg(t, tB0, tB0 + TXT_FADE, "out")
    c.rect((112, A_LBL2_Y + 9, 125, A_LBL2_Y + 22), fill=style.C_OOD, r=3, a=lab_b)
    c.text((139, A_LBL2_Y), "out of distribution", 27, style.C_OOD, "Medium", a=lab_b)
    c.text((139 + c.tw("out of distribution", 27, "Medium") + 12, A_LBL2_Y + 3),
           f"· tiger-pose, {n_ood} frames", 23, style.FG_MUTED, a=lab_b)

    # tau
    pT = common.ease(common.seg(t, tT0, tT1), "out")
    if pT > 0.004:
        xt = sx(tau)
        top = A_LBL1_Y + 40
        c.rect((xt, top, X1, A_AXIS_Y), fill=style.YELLOW, a=0.05 * pT)
        c.dash((xt, top), (xt, A_AXIS_Y), style.YELLOW, 2.4, a=0.9 * pT)
        c.rich((xt + 12, top - 4), [("m", M_TAU), ("t", f" = {tau:.2f}")], 28, style.YELLOW,
               "SemiBold", a=pT)
        c.text((xt + 12, top + 32), "flagged →", 22, style.YELLOW, a=0.75 * pT)

    pC = common.seg(t, tC0, tC0 + TXT_FADE, "out")
    c.text((112, A_CAP_Y), f"AUROC {auroc:.3f}", 30, style.FG, "SemiBold", a=pC)


def _panel_b(c, t, dur, ep_s, ep_ood, v, streak, accept, nan_after, step_t, tH0, resume):
    fade = common.seg(t, tH0, tH0 + TXT_FADE, "out")
    if fade <= 0.004:
        return
    n = ep_s.size
    c.text((BX0, B_HEAD_Y), "OOD handling (Alg. 1)", HEAD_SIZE, style.FG, "SemiBold",
           a=fade)

    # how far through the step sequence are we
    k_now = int(np.searchsorted(step_t, t, side="right") - 1)
    k_now = int(np.clip(k_now, -1, n - 1))
    sub = ((t - step_t[k_now]) / max(step_t[k_now + 1] - step_t[k_now], 1e-6)
           if k_now >= 0 else 0.0)
    sub = float(np.clip(sub, 0.0, 1.0))

    # ---------------- score row
    c.rich((BX0, B_S_LBL), [("m", M_SLU2D), ("t", " score per frame")], 24, style.FG_MUTED,
           a=fade)
    xend = cell_x(n - 1) + B_CELL
    c.line([(BX0 - 10, B_S_BASE), (xend + 8, B_S_BASE)], style.GRID, 2, a=fade)
    yt = B_S_BASE - bar_h(TAU_2D)
    c.dash((BX0 - 10, yt), (xend + 8, yt), style.YELLOW, 2.0, a=0.8 * fade)
    c.math((xend + 16, yt - 15), M_TAU, 22, style.YELLOW, "Medium", a=0.9 * fade)

    for k in range(min(k_now + 1, n)):
        g = common.ease(min(1.0, (sub if k == k_now else 1.0) * 2.2), "out")
        x0 = cell_x(k)
        h = bar_h(ep_s[k]) * g
        col = style.C_OOD if ep_ood[k] else style.FG_MUTED
        c.rect((x0, B_S_BASE - h, x0 + B_CELL, B_S_BASE), fill=col, r=4,
               a=0.95 if ep_ood[k] else 0.55)

    # ---------------- pose buffer H (validity)
    c.rich((BX0, B_H_LBL), [("t", "Validity of the pose buffer "), ("m", M_H)], 24,
           style.FG_MUTED, a=fade)
    for k in range(min(k_now + 1, n)):
        g = common.ease(min(1.0, (sub if k == k_now else 1.0) * 2.5), "out")
        x0 = cell_x(k)
        box = (x0, B_H_TOP, x0 + B_CELL, B_H_BOT)
        if v[k]:
            c.rect(box, fill=style.C_PRED, r=6, a=0.85 * g)
            c.text((x0 + B_CELL / 2, (B_H_TOP + B_H_BOT) / 2), "1", 24, style.BG,
                   "SemiBold", a=g, anchor="mm")
        else:
            c.rect(box, fill=style.C_OOD, r=6, a=0.18 * g)
            c.rect(box, outline=style.C_OOD, w=2.2, r=6, a=0.95 * g)
            c.text((x0 + B_CELL / 2, (B_H_TOP + B_H_BOT) / 2), "0", 24, style.C_OOD,
                   "SemiBold", a=g, anchor="mm")

    # the sum v_i = N_req condition of Algorithm 1, on the validity row
    if k_now >= 0:
        xd = B_SUMV_X
        xd += c.math((xd, B_H_LBL), M_SUMV, 24, style.FG_MUTED, a=fade) + 16
        for j in range(N_REQ):
            cx = xd + 9 + j * 25
            on = j < int(streak[k_now])
            c.dot((cx, B_H_LBL + 13), 7.5, style.C_TRUTH if on else style.GRID, a=fade)
        c.rich((xd + N_REQ * 25 + 8, B_H_LBL), [("t", "/ "), ("m", M_NREQ)], 24,
               style.FG_MUTED, a=fade)

    f0 = int(np.argmin(v))
    if k_now >= f0:
        last = int(np.max(np.where(v == 0)[0]))
        aa = common.ease(min(1.0, (k_now - f0 + sub) * 3.0), "out")
        held = k_now <= last
        # Centred on the WHOLE burst, not on the part revealed so far: the caption must not
        # re-centre (i.e. travel) as the timeline steps through the three flagged frames.
        xa, xb = cell_x(f0), cell_x(last) + B_CELL
        c.rich(((xa + xb) / 2, B_H_BOT + 14),
               [("m", M_REUSE), ("t", "   reuse the prediction")], 23, style.C_OOD,
               "Medium", a=(0.95 if held else 0.45) * aa, anchor="m")

    # ---------------- motion buffer M
    c.rich((BX0, B_M_LBL), [("t", "Motion buffer "), ("m", M_M)], 24, style.FG_MUTED, a=fade)
    nan_now = int(nan_after[k_now]) if k_now >= 0 else 0
    nan_prev = int(nan_after[k_now - 1]) if k_now >= 1 else 0
    shifting = (k_now >= 0) and (not accept[k_now]) and nan_now > nan_prev
    # The one thing that translates in this shot: the motion buffer sliding one slot.  That
    # slide *is* `shift(M)`, the algorithm's own operation on a picture of a ring buffer -- the
    # "-" placeholder inside an empty cell rides along because it is a mark on the cell, not a
    # caption.  No label, number or sentence moves anywhere in s07.
    slide = (1.0 - common.ease(min(1.0, sub * 1.8), "out")) if shifting else 0.0
    flash = 0.0
    if k_now >= 0 and accept[k_now] and (k_now == 0 or not accept[k_now - 1]):
        flash = 1.0 - common.ease(min(1.0, sub * 1.6), "out")
    mw = K_P * (M_CELL + M_GAP) - M_GAP
    for j in range(K_P):
        x0 = BX0 + j * (M_CELL + M_GAP) + slide * (M_CELL + M_GAP)
        if x0 > BX0 + mw:
            continue
        box = (x0, B_M_TOP, x0 + M_CELL, B_M_BOT)
        if j < (K_P - nan_now):
            c.rect(box, fill=style.C_OURS, r=6, a=(0.30 + 0.42 * (1 - j / K_P)) * fade)
            c.rect(box, outline=style.C_OURS, w=2.0, r=6, a=0.8 * fade)
        else:
            c.rect(box, outline=style.GRID, w=2.0, r=6, a=0.95 * fade)
            c.text((x0 + M_CELL / 2, (B_M_TOP + B_M_BOT) / 2), "–", 24,
                   style.FG_MUTED, a=0.7 * fade, anchor="mm")
    # M[0] -- the slot the pose buffer borrows while the frames are OOD
    held_now = (k_now >= 0) and (v[k_now] == 0)
    c.rect((BX0 - 4, B_M_TOP - 4, BX0 + M_CELL + 4, B_M_BOT + 4), outline=style.C_OOD,
           w=2.2, r=8, a=0.95 if held_now else 0.0)
    c.rich((BX0 + M_CELL / 2, B_M_BOT + 12), [("m", M_M0)], 21,
           style.C_OOD if held_now else style.FG_MUTED, "Medium",
           a=(1.0 if held_now else 0.55) * fade, anchor="m")
    if flash > 0.02:
        c.rect((BX0 - 6, B_M_TOP - 6, BX0 + mw + 6, B_M_BOT + 6),
               outline=style.C_OURS, w=2.4, r=10, a=flash)

    # ---------------- status
    if k_now >= 0:
        ok = bool(accept[k_now])
        col = style.C_OURS if ok else style.C_OOD
        parts = ([("m", M_ACCEPT), ("t", "   accept the new prediction")] if ok
                 else [("m", M_SHIFT), ("t", "   continue on the prior prediction")])
        wpill = c.rich((0, 0), parts, 26, col, "Medium", a=0.0) + 40
        c.rect((BX0, B_STAT_Y, BX0 + wpill, B_STAT_Y + 46), fill=col, r=10, a=0.16)
        c.rect((BX0, B_STAT_Y, BX0 + wpill, B_STAT_Y + 46), outline=col, w=1.8, r=10, a=0.55)
        c.rich((BX0 + 20, B_STAT_Y + 9), parts, 26, col, "Medium", a=1.0)

    if k_now >= resume:
        aa = common.ease(min(1.0, (k_now - resume + sub) * 3.0), "out")
        c.rich((BX0, B_CAP_Y), [("t", "normal operation resumes after "), ("m", M_NREQ),
                                ("t", " = 3 consecutive image-based poses")],
               24, style.C_TRUTH, "Medium", a=0.95 * aa)
    elif k_now >= f0:
        # only while the buffer is actually being reused -- before the burst there is nothing
        # to say, and saying it would contradict the "accept" pill above.
        c.text((BX0, B_CAP_Y), "the conformal sets keep publishing from the shifted buffer",
               24, style.FG_MUTED,
               a=0.9 * common.ease(min(1.0, (k_now - f0 + sub) * 3.0), "out"))


def main() -> None:
    if "--prepare" in sys.argv:
        prepare()
        return
    args = common.shot_args("s07 -- out-of-distribution monitor")
    render(args)


if __name__ == "__main__":
    main()
