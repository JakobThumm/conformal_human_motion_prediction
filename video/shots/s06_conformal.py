"""s06 -- the conformal argument: overview, score, calibration histogram, quantile, inflated set.

Narration: "But a learned covariance is a guess, not a guarantee.  So we calibrate: on held-out
data we measure how far the truth actually lands, in units of predicted standard deviation.  Its
99.99th percentile scales every sphere into a conformal prediction set."

Four beats:

  0.  the manuscript's pipeline figure (`_overview.py`) with the **conformal prediction sets**
      stage lit; the rest of the pipeline falls away and the lit block dissolves into --
  1.  one real predicted joint: its 1-sigma ellipse, the true position off it, and eq. (2)
      turned into real millimetres,
  2.  the same score on all 39 440 calibration windows, with the split-conformal quantile
      alpha_k^j marked in the tail,
  3.  the conformal prediction set: the sphere of radius alpha_k^j * sqrt(lambda_max(C)).

Everything numeric is computed from the repository's own calibration results -- nothing is invented.

  * scores ........ eq. (2) of the manuscript (``content/S3_methodology.tex``,
        ``eq:non_conformity_score``):  A_k^j = ||d_k^j||_2 / sqrt(lambda_max(C_k^j)),
        computed on every joint (J = 13) and horizon step (K_P = 10) of every window of
        ``results/final/conformal_prediction_sets/motion_prediction_results_validation.cloudpickle``
        (n = 39 440 held-out windows, none of them OOD; the H36M calibration split).
  * alpha_k^j ..... the **per-joint, per-timestep** threshold of Sec. "Conformal Prediction Sets"
        / "Conformal Calibration": the ceil((1-eps)(|Z^cal|+1))-th smallest score of that one
        (joint, horizon step) cell, with 1 - eps = 99.99 % (``content/S4_experiments.tex``).
        On n = 39 440 the index is 39 438.  The shot tells the whole story with **one** cell --
        CELL_J / CELL_K below (left knee, horizon step 3 = +120 ms, alpha_k^j = 5.46) -- so that
        the example residual, the histogram and the drawn sphere are all the same number.
        The full 10x13 table (min 4.51, median 8.13, max 14.62) is printed by --prepare.
        NOTE: this is *not* the 14.68 of the earlier cut -- that was alpha_max, the paper's
        single-threshold ablation (quantile of the per-window maximum score), which is
        necessarily larger.
  * example joint . window i = EX_I of the same file, same (joint, horizon step) cell:
        ||d|| = 40.6 mm, sqrt(lambda_max(C)) = 18.8 mm, A = 2.15.  The 2-D drawing is the exact
        (u1, u2) plane spanned by the covariance's major eigenvector and the residual, so the
        ellipse semi-axes and the residual length on screen are the real ones.

Prepare (needs jax to unpickle the results; run in the repo venv, ~1 min, writes a 5 kB cache)::

    cd /home/thumm/code/conformal_human_motion_prediction
    XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cpu \
        .venv/bin/python video/shots/s06_conformal.py --prepare

Timing (25.0 s; the author asked for "+3 s on part 1, +2 s on part 2, part 3 unchanged"):
beat 0 keeps 0.00-2.45 s, part 1 runs 2.45-10.10 s (was 2.45-7.10), part 2 runs 10.10-16.95 s
(was 7.10-11.95) and part 3 keeps its 8.05 s, 16.95-25.00 s (was 11.95-20.00).

Per ``SPEC.md`` "Motion discipline": every word, number, caption, legend and equation enters on
**opacity only**, in <= 0.25 s, and **nothing ever exits** -- the three beats are separated by
hard cuts.  The motion that is left is all graphical: the overview figure's crossfades and
push-in, the covariance ellipse and the residual being drawn, the histogram growing, the
quantile marker sliding onto alpha, and the conformal ball inflating.

Render::

    /home/thumm/miniconda3/envs/chmp-video/bin/python video/shots/s06_conformal.py \
        --out video/build/shots/s06.mp4 --duration 25.0
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

VIDEO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VIDEO_DIR))
sys.path.insert(0, str(VIDEO_DIR / "shots"))

import common  # noqa: E402
import style  # noqa: E402
import _pl_axis as pla  # noqa: E402

SID = "s06"
CACHE = common.MEDIA / "s06_scores.npz"
RESULTS = common.REPO / "results/final/conformal_prediction_sets/motion_prediction_results_validation.cloudpickle"
CALIBRATOR = common.REPO / "models/motion_prediction/conformal_calibration/conformal_calibrator.npz"

LEVEL = 0.9999                      # 1 - eps, S4_experiments.tex
# The one (joint, horizon step) cell the whole shot talks about.  Chosen among the cells with the
# smallest alpha_k^j (so the calibrated sphere is only ~5.5x the 1-sigma ellipse and both fit on
# screen at one scale) and at a horizon that is worth predicting: k = 3 -> +120 ms.
CELL_K, CELL_J = 2, 9               # 0-based horizon step, joint index -> +120 ms, LKnee
EX_I = 1310                         # example window: A = 2.15, sigma = 18.8 mm, b/a = 0.64
# src/conformal_human_motion_prediction/pose_estimation/h36m_settings.py::JOINT_NAMES_13
JOINT_NAMES_13 = ['Nose', 'LShoulder', 'RShoulder', 'LElbow', 'RElbow', 'LWrist', 'RWrist',
                  'LHip', 'RHip', 'LKnee', 'RKnee', 'LAnkle', 'RAnkle']
PRETTY_JOINT = {'LKnee': 'left knee', 'RKnee': 'right knee', 'LShoulder': 'left shoulder',
                'RShoulder': 'right shoulder', 'LElbow': 'left elbow', 'RElbow': 'right elbow',
                'LWrist': 'left wrist', 'RWrist': 'right wrist', 'LAnkle': 'left ankle',
                'RAnkle': 'right ankle', 'LHip': 'left hip', 'RHip': 'right hip', 'Nose': 'nose'}
STEP_MS = 40.0                      # 25 fps -> horizon step k is 40*k ms ahead
N_BINS, X_MAX = 60, 6.0             # histogram of the cell's scores (max score 5.84)


# --------------------------------------------------------------------------- prepare
def prepare() -> None:
    import cloudpickle

    with open(RESULTS, "rb") as f:
        d = cloudpickle.load(f)
    P = np.asarray(d["predictions"], dtype=np.float64)          # (N, K, J, 3)   mm
    T = np.asarray(d["targets"], dtype=np.float64)
    C = np.asarray(d["covariance_matrices"], dtype=np.float64)  # (N, K, J, 3, 3) mm^2
    assert int(np.asarray(d["is_oods"]).sum()) == 0, "calibration file must be OOD-free"

    res = T - P
    dnorm = np.linalg.norm(res, axis=-1)                        # (N, K, J)
    evals, evecs = np.linalg.eigh(C)                            # ascending
    sigma = np.sqrt(evals[..., -1])                             # sqrt(lambda_max)
    A = dnorm / sigma                                           # eq. (2)

    n, n_steps, n_joints = A.shape
    idx = math.ceil(LEVEL * (n + 1))                            # split-conformal index
    # per-joint, per-timestep thresholds: the idx-th smallest score of each (k, j) cell
    alpha_kj = np.sort(A, axis=0)[idx - 1]                      # (K, J)

    k, j = CELL_K, CELL_J
    alpha = float(alpha_kj[k, j])
    cell = A[:, k, j]
    n_above = int((cell > alpha).sum())
    counts, edges = np.histogram(cell, bins=N_BINS, range=(0.0, X_MAX))
    assert cell.max() <= X_MAX, f"histogram range too small ({cell.max():.3f})"

    # ---- the example window of that cell, exactly as drawn ----------------------------------
    i = EX_I
    u1 = evecs[i, k, j, :, -1]                                  # major eigenvector
    dvec = res[i, k, j]
    if float(dvec @ u1) < 0.0:
        u1 = -u1                                                # sign of an eigenvector is free
    dperp = dvec - np.dot(dvec, u1) * u1
    u2 = dperp / np.linalg.norm(dperp)
    a = math.sqrt(float(u1 @ C[i, k, j] @ u1))                  # == sqrt(lambda_max)
    b = math.sqrt(float(u2 @ C[i, k, j] @ u2))                  # minor semi-axis in-plane
    dx, dy = float(dvec @ u1), float(dvec @ u2)

    # the paper's single-threshold ablation, for the notes (not shown in the shot)
    A_max = A.reshape(n, -1).max(axis=1)
    alpha_max = float(np.sort(A_max)[idx - 1])
    alpha_npz = float(np.load(CALIBRATOR, allow_pickle=True)["alpha_max"])
    from scipy.stats import chi2

    np.savez_compressed(
        CACHE,
        counts=counts, edges=edges, n=n, idx=idx, alpha=alpha, n_above=n_above,
        level=LEVEL, x_max=X_MAX,
        a=a, b=b, dx=dx, dy=dy, dnorm=float(dnorm[i, k, j]), A_ex=float(A[i, k, j]),
        window=i, joint=JOINT_NAMES_13[j], joint_pretty=PRETTY_JOINT[JOINT_NAMES_13[j]],
        step=k + 1, horizon_ms=STEP_MS * (k + 1), n_steps=n_steps, n_joints=n_joints,
        cell_med=float(np.median(cell)), cell_max=float(cell.max()),
        alpha_kj_min=float(alpha_kj.min()), alpha_kj_med=float(np.median(alpha_kj)),
        alpha_kj_max=float(alpha_kj.max()), alpha_kj=alpha_kj,
        alpha_max=alpha_max, alpha_npz=alpha_npz,
        gauss_alpha=float(math.sqrt(chi2.ppf(LEVEL, 3))),
    )
    print(f"wrote {CACHE}")
    print(f"  n = {n}, index ceil({LEVEL}*(n+1)) = {idx}")
    print(f"  alpha_k^j table: min {alpha_kj.min():.3f}  median {np.median(alpha_kj):.3f}  "
          f"max {alpha_kj.max():.3f}")
    print(f"  medians by step: " + ", ".join(f"{v:.2f}" for v in np.median(alpha_kj, axis=1)))
    print(f"  CELL {JOINT_NAMES_13[j]} step {k + 1} (+{STEP_MS * (k + 1):.0f} ms): "
          f"alpha_k^j = {alpha:.4f}, {n_above} of {n} windows above, "
          f"cell median {np.median(cell):.2f}, max {cell.max():.3f}")
    print(f"  example window {i}: |d| = {dnorm[i, k, j]:.1f} mm, sqrt(lmax) = {a:.1f} mm, "
          f"A = {A[i, k, j]:.3f}, minor {b:.1f} mm, alpha*sigma = {alpha * a:.1f} mm")
    print(f"  (ablation alpha_max = {alpha_max:.4f}; shipped npz {alpha_npz:.4f}; "
          f"analytic sqrt(chi2_3) = {math.sqrt(chi2.ppf(LEVEL, 3)):.3f})")


# --------------------------------------------------------------------------- overview beat
def overview_frames(width: int, height: int):
    """The three stills of beat 0: flat figure, lit region, lit region alone."""
    import _overview as OV

    flat = np.array(OV.compose(OV.REGION_CONFORMAL, reveal=1.0, focus=0.0, accent=style.C_OURS,
                               width=width, height=height))
    lit = np.array(OV.compose(OV.REGION_CONFORMAL, reveal=1.0, focus=1.0, accent=style.C_OURS,
                              width=width, height=height))

    # everything but the lit block falls away -> the shot dissolves out of the block alone
    sx = OV.FIG_W / 2100.0
    box = (OV.FIG_X + OV.REGION_CONFORMAL[0] * sx - 6, OV.FIG_Y + OV.REGION_CONFORMAL[1] * sx - 6,
           OV.FIG_X + OV.REGION_CONFORMAL[2] * sx + 6, OV.FIG_Y + OV.REGION_CONFORMAL[3] * sx + 6)
    f = 44.0
    bx = (box[0] - f, box[1] - f, box[2] + f, box[3] + f)
    x = np.arange(width, dtype=np.float32)
    y = np.arange(height, dtype=np.float32)
    sxm = np.clip((x - bx[0]) / f, 0, 1) * np.clip((bx[2] - x) / f, 0, 1)
    sym = np.clip((y - bx[1]) / f, 0, 1) * np.clip((bx[3] - y) / f, 0, 1)
    sxm = sxm * sxm * (3.0 - 2.0 * sxm)
    sym = sym * sym * (3.0 - 2.0 * sym)
    keep = (sym[:, None] * sxm[None, :])[:, :, None]
    bg = np.array(common.hex2rgb(style.BG), np.float32)
    iso = (lit.astype(np.float32) * keep + bg * (1.0 - keep)).astype(np.uint8)
    return flat, lit, iso, box


# --------------------------------------------------------------------------- render
def build_scene(duration: float, C):
    from manim import (Scene, VGroup, Group, Circle, Ellipse, Dot, DashedLine, Line, Rectangle,
                       MathTex, ImageMobject, FadeIn, FadeOut, Create, GrowFromEdge, LaggedStart,
                       Transform, DOWN, LEFT, RIGHT, ORIGIN, rate_functions)

    alpha = float(C["alpha"])
    counts, edges = C["counts"], C["edges"]
    x_max = float(C["x_max"])
    n, idx, n_above = int(C["n"]), int(C["idx"]), int(C["n_above"])
    a_mm, b_mm = float(C["a"]), float(C["b"])
    dx_mm, dy_mm = float(C["dx"]), float(C["dy"])
    dnorm, A_ex = float(C["dnorm"]), float(C["A_ex"])
    joint_pretty = str(C["joint_pretty"])
    horizon_ms = float(C["horizon_ms"])
    ball_mm = alpha * a_mm

    def grp(x: float) -> str:
        """1234 -> '1 234' (thin space)."""
        return f"{int(x):,}".replace(",", " ")

    def tgrp(x: float) -> str:
        return f"{int(x):,}".replace(",", r"\,")

    # ---- mixed Inter / LaTeX lines -----------------------------------------------------------
    # Prose is Inter (via pla.txt), maths is LaTeX; they are set on a common baseline.  Inter
    # tokens are placed by their ink bottom (none of the strings below has a descender, so that
    # *is* the baseline); maths is centred on the cap band, which is where a formula's axis sits.
    _cap: dict = {}

    def cap_h(px: float) -> float:
        if px not in _cap:
            _cap[px] = pla.txt("H", px=px).height
        return _cap[px]

    def mth(s: str, px: float = style.CAPTION_SIZE, color: str = style.FG):
        """MathTex scaled so its cap height matches Inter at ``px``."""
        m = MathTex(s, color=color)
        return m.scale(cap_h(px) / MathTex("H").height)

    def line_of(tokens, left: float, baseline: float, px: float = style.CAPTION_SIZE,
                color: str = style.FG, buff: float = 0.13):
        """``tokens`` = [("t", inter), ("m", latex), ...] set left-to-right on one baseline.

        A third element overrides the gap in front of that token (0.0 = set it tight).
        """
        g = VGroup()
        x, ch = left, cap_h(px)
        for tok in tokens:
            kind, s = tok[0], tok[1]
            if g.submobjects and len(tok) > 2:
                x += tok[2] - buff
            if kind == "t":
                m = pla.txt(s, px=px, color=color)
                m.move_to([x + m.width / 2, baseline + m.height / 2, 0])
            else:
                m = mth(s, px=px, color=color)
                m.move_to([x + m.width / 2, baseline + 0.47 * ch, 0])
            g.add(m)
            x += m.width + buff
        return g

    class S06(pla.TimedScene, Scene):
        D = duration

        # ---------------------------------------------------------------- pieces
        def geometry(self, origin, mm: float):
            """Predicted position, its 1-sigma ellipse and the true position, drawn to scale.

            ``mm`` is manim units per millimetre.  The picture is the real (u1, u2) plane --
            u1 = major eigenvector of C, u2 = the in-plane direction of the residual -- rotated
            in plane so u1 is horizontal (the plane's orientation in world space is arbitrary, so
            an in-plane rotation costs nothing).  The residual then leaves at its true angle to
            the major axis.
            """
            O = np.array([origin[0], origin[1], 0.0])
            e1 = np.array([1.0, 0.0, 0.0])          # u1 on screen
            e2 = np.array([0.0, 1.0, 0.0])          # u2 on screen
            tru = O + mm * (dx_mm * e1 + dy_mm * e2)
            ell = Ellipse(width=2 * a_mm * mm, height=2 * b_mm * mm,
                          stroke_color=style.C_PRED, stroke_width=2.6,
                          fill_color=style.C_PRED, fill_opacity=0.10).move_to(O)
            pdot = Dot(O, radius=0.075, color=style.C_PRED)
            tdot = Dot(tru, radius=0.085, color=style.C_TRUTH)
            sig = Line(O, O + a_mm * mm * e1, stroke_width=3.0, color=style.YELLOW)
            resid = DashedLine(O, tru, dash_length=0.11, dashed_ratio=0.6,
                               stroke_width=2.2, color=style.FG)
            return dict(O=O, tru=tru, ell=ell, pdot=pdot, tdot=tdot, sig=sig, resid=resid)

        def histogram(self, x0, x1, ybase, height):
            cmax = float(counts.max())
            span = math.log10(1.0 + cmax)

            def hx(v):
                return x0 + (v / x_max) * (x1 - x0)

            def hy(c):
                return ybase + height * math.log10(1.0 + c) / span

            grid = VGroup()
            for c in (1, 10, 100, 1000):
                y = hy(c)
                grid.add(Line([x0, y, 0], [x1, y, 0], stroke_width=1.0, color=style.GRID))
                lab = pla.txt(grp(c), px=22, color=style.FG_MUTED)
                lab.next_to([x0, y, 0], LEFT, buff=0.16)
                grid.add(lab)
            bars = VGroup()
            w = (hx(edges[1]) - hx(edges[0])) * 0.82
            for c, e0, e1 in zip(counts, edges[:-1], edges[1:]):
                if c <= 0:
                    continue
                h = hy(c) - ybase
                r = Rectangle(width=w, height=h, stroke_width=0,
                              fill_color=style.C_OURS, fill_opacity=0.9)
                r.move_to([0.5 * (hx(e0) + hx(e1)), ybase + h / 2, 0])
                bars.add(r)
            axis = Line([x0, ybase, 0], [x1 + 0.1, ybase, 0], stroke_width=1.8,
                        color=style.FG_MUTED)
            ticks = VGroup()
            for v in range(0, int(x_max) + 1):
                t = Line([hx(v), ybase, 0], [hx(v), ybase - 0.12, 0], stroke_width=1.6,
                         color=style.FG_MUTED)
                lab = pla.txt(str(v), px=24, color=style.FG_MUTED)
                lab.next_to(t, DOWN, buff=0.10)
                ticks.add(t, lab)
            # axis titles
            xlab = line_of([("t", "non-conformity score"), ("m", r"A^j_k")],
                           0.0, ybase - 0.72, px=26, color=style.FG_MUTED)
            xlab.move_to([0.5 * (x0 + x1), ybase - 0.62, 0])
            ylab = pla.txt("calibration windows (log count)", px=24, color=style.FG_MUTED)
            ylab.rotate(np.pi / 2).move_to([x0 - 1.02, ybase + height / 2, 0])
            return bars, VGroup(grid, axis, ticks, xlab, ylab), hx, hy

        # ---------------------------------------------------------------- timeline
        def construct(self):
            kick = pla.kicker("Conformal prediction sets")

            # ---- 0: where we are in the pipeline -------------------------------------- 0.0 s
            flat, lit, iso, _box = overview_frames(1920, 1080)
            # z_index: the figure is a backdrop -- the kicker and the title ride on top of it
            ov_flat, ov_lit, ov_iso = (ImageMobject(im).move_to(ORIGIN).set_z_index(-5)
                                       for im in (flat, lit, iso))
            title = pla.txt("From a learned covariance to a guaranteed set",
                            px=42, color=style.FG, weight="MEDIUM")
            title.move_to([kick.get_left()[0] + title.width / 2, 4.0 - 113 / 135.0, 0])

            # SPEC.md "Motion discipline": words only ever fade up (<= 0.25 s) and nothing
            # ever fades out.  The overview figure's crossfades and the push-in are *graphics*
            # and keep their motion; the shot's three beats are separated by hard cuts.
            self.add(ov_flat)
            self.pl(FadeIn(ov_flat), run_time=0.35)
            self.pl(FadeIn(kick), FadeIn(title), run_time=0.20)
            self.pl(FadeIn(ov_lit), FadeOut(ov_flat), run_time=0.45)    # graphic crossfade
            self.hold(1.05)
            self.pl(FadeIn(ov_iso), FadeOut(ov_lit), run_time=0.40)     # graphic crossfade

            # ---- A: one predicted position -------------------------------------------- 2.45 s
            MM1 = 0.048                                    # manim units per mm
            g = self.geometry((-4.05, -0.05), MM1)
            ell, pdot, tdot, sig, resid = g["ell"], g["pdot"], g["tdot"], g["sig"], g["resid"]

            plab = line_of([("t", "predicted position"), ("m", r"\hat{\boldsymbol{p}}^j_k")],
                           0.0, 0.0, px=26, color=style.C_PRED)
            plab.move_to(g["O"] + np.array([0.0, -1.12, 0.0]))
            tlab = line_of([("t", "true position"), ("m", r"\boldsymbol{p}^j_k")],
                           0.0, 0.0, px=26, color=style.C_TRUTH)
            tlab.move_to(g["tru"] + np.array([tlab.width / 2 + 0.20, 0.28, 0.0]))

            dlab = mth(r"\lVert \boldsymbol{d}^j_k \rVert_2 = %.1f\ \text{mm}" % dnorm,
                       px=27, color=style.FG)
            mid = 0.5 * (g["O"] + g["tru"])
            dlab.move_to(mid + np.array([-0.34 - dlab.width / 2, 0.38, 0.0]))
            slab = mth(r"\sqrt{\lambda_{\max}(\boldsymbol{C}^j_k)} = %.1f\ \text{mm}" % a_mm,
                       px=27, color=style.YELLOW)
            slab.move_to(g["O"] + np.array([0.30 + slab.width / 2, -0.62, 0.0]))

            geo = Group(ell, sig, resid, pdot, tdot, plab, tlab, dlab, slab)
            # push in on the lit block (it is scaled about its own centre, so the block stays
            # put while the rest of the page flies out of frame) and dissolve into the geometry.
            # The title is *removed*, not faded out -- no exit animations anywhere in this shot.
            bx = np.array([(_box[0] + _box[2]) / 2.0, (_box[1] + _box[3]) / 2.0])
            bc = np.array([(bx[0] - 960.0) / 135.0, (540.0 - bx[1]) / 135.0, 0.0])
            self.remove(title)
            self.pl(ov_iso.animate.scale(1.55, about_point=bc).set_opacity(0.0),
                    FadeIn(pdot, scale=0.4), run_time=0.62)               # graphics
            self.pl(FadeIn(plab), run_time=0.22)
            self.at(3.50)
            self.pl(Create(ell), FadeIn(sig), run_time=0.70)              # graphics
            self.pl(FadeIn(slab), run_time=0.22)
            self.at(4.80)
            self.pl(FadeIn(tdot, scale=0.4), run_time=0.35)               # graphic
            self.pl(FadeIn(tlab), run_time=0.22)
            self.at(5.80)
            self.pl(Create(resid), run_time=0.50)                         # graphic
            self.pl(FadeIn(dlab), run_time=0.22)

            # ---- B: the non-conformity score ------------------------------------------ 7.10 s
            self.at(7.10)
            fcap = pla.txt("Non-conformity score:", px=28, color=style.FG)
            fcap.move_to([0.95 + fcap.width / 2, 1.80, 0])
            form = MathTex(
                r"A^j_k &= \frac{\lVert \boldsymbol{d}^j_k \rVert_2}"
                r"{\sqrt{\lambda_{\max}(\boldsymbol{C}^j_k)}} \\[10pt]",
                r"&= \frac{%.1f\ \text{mm}}{%.1f\ \text{mm}} \;=\; " % (dnorm, a_mm),
                r"%.2f" % A_ex,
                color=style.FG).scale(0.70)
            form[2].set_color(style.YELLOW)
            form.move_to([1.15 + form.width / 2, -0.10, 0])
            eqno = MathTex(r"(2)", color=style.FG_MUTED).scale(0.60)
            eqno.move_to([form.get_right()[0] + 0.62, form[0].get_center()[1], 0])

            self.pl(FadeIn(fcap), FadeIn(form[0]), FadeIn(eqno), run_time=0.25)
            self.at(8.25)
            self.pl(FadeIn(form[1]), FadeIn(form[2]), run_time=0.25)

            # ---- C: the whole calibration set ---------------------------------------- 10.10 s
            self.at(10.10)
            rule = line_of(
                [("t", "Calibrate the score threshold"), ("m", r"\alpha^j_k"), ("t", "as the"),
                 ("m", r"\big\lceil (1-\epsilon)\,(|\mathcal{Z}^{\text{cal}}|+1) \big\rceil"),
                 ("t", "-th smallest score in", 0.02)],
                -6.34, 2.72, px=27, color=style.FG)
            setline = mth(r"\left\{ A^j_k(\boldsymbol{z}_i) \;\mid\; \boldsymbol{z}_i \in "
                          r"\mathcal{Z}^{\text{cal}} \right\}", px=27, color=style.FG)
            setline.move_to([-6.34 + setline.width / 2, 2.06, 0])

            HB, HH = -1.95, 2.95                          # baseline, height
            bars, haxes, hx, hy = self.histogram(-4.90, 5.55, HB, HH)

            # Hard cut out of part 1: removed, never faded away.  NOTE: manim's Cairo
            # ``Scene.remove`` calls ``restructure_mobjects(..., extract_families=False)``, so
            # it only drops the *exact* objects that were added -- passing a parent group is a
            # silent no-op.  Every list below is therefore unpacked to what ``play`` added.
            self.remove(*geo, fcap, *form, eqno, ov_iso)
            self.pl(FadeIn(rule), FadeIn(setline), FadeIn(haxes), run_time=0.25)
            self.at(10.70)
            self.pl(LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars], lag_ratio=0.013),
                    run_time=1.60)                                        # graphic

            # ---- D: the quantile ----------------------------------------------------- 12.90 s
            self.at(12.90)
            xa = hx(alpha)
            qline = Line([xa, HB, 0], [xa, HB + 0.62 * HH, 0], stroke_width=3.4,
                         color=style.YELLOW)
            qmove = qline.copy().shift(RIGHT * (hx(x_max) - xa + 0.45))
            shade = Rectangle(width=xa - hx(0.0), height=HH, stroke_width=0,
                              fill_color=style.C_OURS, fill_opacity=0.09)
            shade.move_to([0.5 * (hx(0.0) + xa), HB + HH / 2, 0])
            qlab = mth(r"\alpha^j_k = %.2f" % alpha, px=40, color=style.YELLOW)
            qlab.move_to([xa - 0.26 - qlab.width / 2, HB + 0.83 * HH, 0])
            tag = pla.txt(f"{joint_pretty}  ·  +{horizon_ms:.0f} ms", px=26, color=style.FG_MUTED)
            tag.move_to([xa - 0.26 - tag.width / 2, HB + 0.83 * HH - 0.48, 0])
            qcap = pla.txt(f"{grp(idx)}-th smallest of {grp(n)}  ·  {n_above} above",
                           px=23, color=style.FG_MUTED)
            qcap.move_to([xa - 0.26 - qcap.width / 2, HB + 0.83 * HH - 0.90, 0])

            # the quantile marker sliding onto its value is a *graphic*, and stays
            self.pl(Transform(qmove, qline), FadeIn(shade), run_time=0.75,
                    rate_func=rate_functions.ease_out_cubic)
            self.at(13.80)
            self.pl(FadeIn(qlab), FadeIn(tag), run_time=0.25)
            self.at(14.40)
            self.pl(FadeIn(qcap), run_time=0.25)

            # ---- E: the conformal prediction set ------------------------------------- 16.95 s
            self.at(16.95)
            R2 = 1.80                                     # radius of the alpha-sphere on screen
            MM2 = R2 / ball_mm
            g2 = self.geometry((-4.50, 0.60), MM2)
            sphere = Circle(radius=a_mm * MM2, color=style.C_OURS, stroke_width=3.0,
                            fill_color=style.C_OURS, fill_opacity=0.0).move_to(g2["O"])
            big = Circle(radius=R2, color=style.C_OURS, stroke_width=3.0,
                         fill_color=style.C_OURS, fill_opacity=0.12).move_to(g2["O"])

            plab2 = line_of([("t", "prediction"), ("m", r"\hat{\boldsymbol{p}}^j_k")],
                            0.0, 0.0, px=26, color=style.C_PRED)
            plab2.move_to(g2["O"] + np.array([-0.70, -0.57, 0.0]))
            tlab2 = line_of([("t", "ground truth"), ("m", r"\boldsymbol{p}^j_k")],
                            0.0, 0.0, px=26, color=style.C_TRUTH)
            tlab2.move_to(g2["tru"] + np.array([-0.18, 0.42, 0.0]))
            slab_gp = line_of([("t", "conformal prediction set"), ("m", r"\mathcal{S}^j_k")],
                              0.0, 0.0, px=27, color=style.C_OURS)
            slab_gp.move_to([g2["O"][0], g2["O"][1] + R2 + 0.40, 0])

            # the calibrated radius, drawn down-right out of the same centre as the 1-sigma one
            ang = math.radians(-38.0)
            rdir = np.array([math.cos(ang), math.sin(ang), 0.0])
            rline = Line(g2["O"], g2["O"] + R2 * rdir, stroke_width=3.2, color=style.C_OURS)
            # both radii are named underneath the ball, colour-keyed to the lines above
            slab2 = mth(r"\sqrt{\lambda_{\max}(\boldsymbol{C}^j_k)} = %.1f\ \text{mm}" % a_mm,
                        px=26, color=style.YELLOW)
            slab2.move_to([g2["O"][0], g2["O"][1] - R2 - 0.44, 0])
            rlab = mth(r"\alpha^j_k \sqrt{\lambda_{\max}(\boldsymbol{C}^j_k)} = %.1f\ \text{mm}"
                       % ball_mm, px=26, color=style.C_OURS)
            rlab.move_to([g2["O"][0], g2["O"][1] - R2 - 1.04, 0])

            setform = mth(r"\mathcal{S}^j_k=\mathcal{B}\!\left(\hat{\boldsymbol{p}}^j_k,\;"
                          r"\alpha^j_k\sqrt{\lambda_{\max}(\boldsymbol{C}^j_k)}\right)",
                          px=37, color=style.FG)
            setform.move_to([-0.95 + setform.width / 2, 0.35, 0])
            guarantee = mth(r"P\!\left(\boldsymbol{p}^j_k\in\mathcal{S}^j_k\right)\;\ge\;"
                            r"1-\epsilon\;=\;99.99\,\%", px=37, color=style.C_TRUTH)
            guarantee.move_to([-0.95 + guarantee.width / 2, -1.15, 0])

            # alpha and its cell tag are carried into part 3's right rail.  They used to
            # *travel* there (`qlab.animate.move_to(...)`); SPEC.md forbids moving words, so
            # part 2's pair is simply removed at the cut and an identical pair is drawn, in
            # place, in the new position.
            qlab2 = mth(r"\alpha^j_k = %.2f" % alpha, px=40, color=style.YELLOW)
            qlab2.move_to([-0.95 + qlab2.width / 2, 2.30, 0])
            tag2 = pla.txt(f"{joint_pretty}  ·  +{horizon_ms:.0f} ms", px=26,
                           color=style.FG_MUTED)
            tag2.move_to([-0.95 + tag2.width / 2, 1.78, 0])

            # Hard cut out of part 2.
            self.remove(*bars, haxes, shade, qmove, rule, setline, qcap, qlab, tag)
            self.pl(FadeIn(g2["ell"]), FadeIn(g2["pdot"]), FadeIn(g2["tdot"]),
                    FadeIn(g2["resid"]), FadeIn(g2["sig"]), FadeIn(sphere),
                    run_time=0.45)                                        # graphics
            self.pl(FadeIn(slab2), FadeIn(plab2), FadeIn(tlab2), FadeIn(qlab2), FadeIn(tag2),
                    run_time=0.25)
            self.at(17.75)
            self.pl(Transform(sphere, big), run_time=1.10,                # graphic: it grows
                    rate_func=rate_functions.ease_out_cubic)
            self.pl(Create(rline), run_time=0.55)                         # graphic
            self.pl(FadeIn(slab_gp), FadeIn(rlab), run_time=0.25)
            self.at(20.30)
            self.pl(FadeIn(setform), run_time=0.25)
            self.at(21.40)
            self.pl(FadeIn(guarantee), run_time=0.25)
            self.finish()

    return S06


def main() -> None:
    args = common.shot_args("s06 conformal prediction sets")
    if not CACHE.exists():
        raise SystemExit(f"missing {CACHE} -- run the --prepare step (see module docstring)")
    C = np.load(CACHE, allow_pickle=False)

    from manim import config
    config.media_dir = f"/tmp/manim_{SID}"
    config.disable_caching = True
    config.output_file = SID
    config.verbosity = "ERROR"
    config.progress_bar = "none"
    style.manim_config(config, quality_fps=args.fps)
    config.pixel_width, config.pixel_height = args.width, args.height
    config.background_color = style.BG

    scene = build_scene(args.duration, C)()
    scene.render()
    common.conform(Path(scene.renderer.file_writer.movie_file_path), args.out, args.duration,
                   fps=args.fps, width=args.width, height=args.height)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    if "--prepare" in sys.argv:
        prepare()
    else:
        main()
