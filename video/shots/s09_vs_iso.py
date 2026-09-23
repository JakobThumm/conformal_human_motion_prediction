#!/usr/bin/env python
"""s09 -- "A real test sequence. Orange is the constant-velocity model of ISO thirteen eight
fifty-five. Blue is ours: the same confidence, a fraction of the space."

One real H36M test window (subject S5, action "Greeting 2", window index 13327 of the test split).
Both set families are drawn on the same skeleton over the same 400 ms horizon:

  orange  C_ISO    SARA shield's ISO 13855 constant-velocity reachable set: a sphere centred on the
                   LAST OBSERVED pose whose radius is  r_in^j + v_h,max * t,  v_h,max = 2.0 m/s
                   (h36m_settings.V_HUMAN_ISO) and r_in^j the 99.99 % input-uncertainty radius of
                   that joint.  Built by the repo's utils.eval_utils.compute_sara_predictions --
                   exactly the call examples/motion_prediction.py makes for the paper's table.
  blue    C_OURS   our conformal prediction set around the predicted mean pose, from the deployed
                   calibrator models/motion_prediction/conformal_calibration/conformal_calibrator.npz
                   via motion_prediction.inference_helper.conformal_set_radius.
  green   C_TRUTH  the mocap ground-truth future pose -- inside both families, for this window.

On-screen numbers (median per-sphere set volume over the 59 472-window H36M test split) come from
  results/final/conformal_prediction_sets/coverage_stats_sara.csv                  -> 0.686 m^3
  results/final/conformal_prediction_sets/coverage_stats_conformal_prediction_sets.csv -> 0.090 m^3
cached into the npz by the prepare step.

Prepare the cache once (repo venv):
    cd /home/thumm/code/conformal_human_motion_prediction
    JAX_PLATFORMS=cpu .venv/bin/python video/shots/_human3d_prepare.py

Render:
    /home/thumm/miniconda3/envs/chmp-video/bin/python video/shots/s09_vs_iso.py \
        --out video/build/shots/s09.mp4 --duration 9.0
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import common                                     # noqa: E402
import style                                      # noqa: E402
from _human3d import (Camera, Canvas, Scene, Text, draw_ground, fit_view, hex2f,  # noqa: E402
                      load_window, mix)

SS = 2
ELEV = 12.0
ORBIT = 24.0

# SPEC "Motion discipline": words appear, pictures may move.  Every label/number/caption in
# this shot enters on opacity alone over TXT_FADE seconds and is never animated out -- the
# only things that move are the camera orbit and the two set families along the horizon clock.
TXT_FADE = 0.20


def main():
    args = common.shot_args(__doc__)
    D, W, H = args.duration, args.width, args.height

    d = load_window("s09")
    conn = d["connections"]
    last = d["last_pose"].astype(np.float64)         # [13,3] m -- last OBSERVED (camera) pose
    pred = d["pred"].astype(np.float64)              # [10,13,3] m
    truth = d["target"].astype(np.float64)           # [10,13,3] m (mocap)
    r_ours = d["r_ours"].astype(np.float64)          # [10,13] m
    r_iso = d["r_iso"].astype(np.float64)            # [10,13] m
    horizon_ms = d["horizon_ms"].astype(np.float64)
    v_iso = float(d["v_iso_median_m3"])
    v_ours = float(d["v_ours_median_m3"])
    v_ratio = float(d["v_ratio"])
    v_h_max = float(d["v_h_max"])
    K_P, J = pred.shape[0], pred.shape[1]

    # ---- framing: the ISO family sets the scale, so fit on it ---------------
    Wss, Hss = W * SS, H * SS
    body = np.concatenate([last[None], pred, truth]).reshape(-1, 3)
    target = np.array([body[:, 0].mean(), body[:, 1].mean(),
                       0.50 * (body[:, 2].min() + body[:, 2].max())])
    # the camera faces the subject's front-left 3/4 so the greeting arm reads
    facing = last[1] - last[2]                       # LShoulder - RShoulder
    az0 = np.degrees(np.arctan2(facing[1], facing[0])) + 90.0 + 34.0
    azims = [az0 + ORBIT * f for f in (0.0, 0.35, 0.7, 1.0)]
    fit_pts = np.concatenate([np.repeat(last[None], K_P, 0).reshape(-1, 3), pred.reshape(-1, 3)])
    fit_rad = np.concatenate([r_iso.ravel(), r_ours.ravel()])
    box = (0.295 * Wss, 0.095 * Hss, 0.945 * Wss, 0.815 * Hss)
    dist, shift = fit_view(target, azims, ELEV, fit_pts, fit_rad, Wss, Hss, box)

    C_I = hex2f(style.C_ISO)
    C_O = hex2f(style.C_OURS)
    C_T = hex2f(style.C_TRUTH)
    C_NOW = hex2f(style.FG)

    # ---- timeline (fractions of the shot) ----------------------------------
    T_GRID = (0.00, 0.08)
    T_POSE = (0.05, 0.17)
    T_ISO = (0.16, 0.44)          # ISO family inflates along the horizon clock
    T_OURS = (0.50, 0.70)         # our sets appear along the same clock ("Blue is ours", 5.6 s)
    T_FADE = (0.70, 0.82)         # orange steps back
    T_NUM = 0.71                  # "... seven point six times less space" lands at 6.7-7.9 s

    canvas = Canvas(W, H, ss=SS)

    def app(u: float, u0: float) -> float:
        """Text entry: opacity only, TXT_FADE seconds, starting at fraction `u0`."""
        return common.seg(u, u0, u0 + TXT_FADE / D, "out")

    def frame(t: float):
        u = t / D
        canvas.clear()
        cam = Camera(target, dist, az0 + ORBIT * common.ease(u, "smooth"), ELEV, Wss, Hss,
                     shift=shift)
        sc = Scene(cam, canvas)

        a_grid = common.seg(u, *T_GRID, "out")
        draw_ground(sc, target[:2], half=3.0, step=0.5, alpha=0.55 * a_grid, fade=3.0)

        a_pose = common.seg(u, *T_POSE, "out")
        a_iso = common.seg(u, *T_ISO, "smooth")
        a_ours = common.seg(u, *T_OURS, "smooth")
        fade = common.seg(u, *T_FADE, "smooth")
        iso_mul = 1.0 - 0.86 * fade

        # --- last observed pose (the centre of the ISO set) -----------------
        if a_pose > 0.01:
            sc.skeleton(last, conn, C_NOW, 3.2 * SS, 0.80 * a_pose, joint_px=4.6 * SS)

        # --- ground truth future: the +400 ms mocap pose, plus a dotted 10-step joint trail.
        #     (10 stacked skeletons would read as a green blob on an in-place greeting motion.)
        for k in range(K_P):
            rev = common.seg(u, 0.07 + 0.012 * k, 0.20 + 0.012 * k, "out")
            if rev <= 0.01:
                continue
            g = (k + 1) / K_P
            for j in range(J):
                sc.dot(truth[k][j], (1.9 + 1.9 * g) * SS,
                       mix(C_T, np.ones(3, np.float32), 0.25), rev * (0.30 + 0.55 * g))
        rev_t = common.seg(u, 0.18, 0.30, "out")
        if rev_t > 0.01:
            sc.skeleton(truth[-1], conn, C_T, 2.7 * SS, 0.88 * rev_t)

        # --- ISO 13855 constant-velocity sets -------------------------------
        hp_i = a_iso * K_P
        if iso_mul > 0.02:
            for k in range(K_P):
                rev = common.ease(np.clip(hp_i - k, 0.0, 1.0), "out")
                if rev <= 0.01:
                    continue
                age = hp_i - k
                hot = float(np.exp(-max(age - 1.0, 0.0) / 0.9))
                taper = float(np.clip((3.0 - age) / 1.0, 0.0, 1.0))   # also the render-cost cap
                for j in range(J):
                    sc.sphere(last[j], float(r_iso[k, j]) * rev, C_I,
                              fill=(0.002 + 0.046 * hot) * rev * iso_mul * taper,
                              rim=(0.004 + 0.34 * hot) * rev * iso_mul * taper,
                              rim_width=0.07 + 0.05 * (1.0 - hot))

        # --- our conformal sets ---------------------------------------------
        hp_o = a_ours * K_P
        for k in range(K_P):
            rev = common.ease(np.clip(hp_o - k, 0.0, 1.0), "out")
            if rev <= 0.01:
                continue
            g = (k + 1) / K_P
            age = hp_o - k
            hot = float(np.exp(-max(age - 1.0, 0.0) / 1.1))
            taper = float(np.clip((3.5 - age) / 1.2, 0.0, 1.0))
            sc.skeleton(pred[k], conn, C_O, (0.9 + 1.2 * g) * SS,
                        rev * (0.05 + 0.32 * g ** 2.4))
            for j in range(J):
                sc.sphere(pred[k][j], float(r_ours[k, j]) * rev, C_O,
                          fill=(0.006 + 0.072 * hot) * rev * taper,
                          rim=(0.008 + 0.52 * hot) * rev * taper,
                          rim_width=0.10 + 0.05 * (1.0 - hot))

        sc.paint()
        img = canvas.to_rgb8()

        # ---- overlay -------------------------------------------------------
        tx = Text(W, H)
        tx.label((60, 44), "Results · how tight are the sets?", 26, "medium", style.FG_MUTED,
                 app(u, 0.00))
        tx.label((1860, 44), "H36M · subject S5 · Greeting · test window 13327", 26, "regular",
                 style.GRID, app(u, 0.08), anchor="ra")

        a_i = app(u, 0.17)
        tx.label((60, 112), "SaRA using ISO 13855 baseline", 32, "semibold", style.C_ISO,
                 a_i)
        tx.label((60, 152), f"constant velocity {v_h_max:.1f} m/s", 26, "regular",
                 style.FG_MUTED, a_i)
        tx.label((60, 184), "from the last observed pose", 26, "regular", style.FG_MUTED, a_i)

        a_o = app(u, 0.52)
        tx.label((60, 258), "Ours", 32, "semibold", style.C_OURS, a_o)
        tx.label((60, 298), "conformal prediction sets", 26, "regular", style.FG_MUTED, a_o)
        tx.label((60, 330), "99.99 % confidence", 26, "regular", style.FG_MUTED, a_o)

        a_t = app(u, 0.20)
        tx.swatch((70, 404), style.C_TRUTH, a_t)
        tx.label((98, 390), "ground truth · inside both", 26, "regular", style.C_TRUTH, a_t)

        # horizon read-out, top right
        a_h = app(u, 0.17)
        hp_c = float(np.clip(max(hp_i, hp_o), 0.0, K_P))
        ms = 0.0 if hp_c <= 0 else float(horizon_ms[int(np.ceil(hp_c)) - 1])
        tx.label((1860, 104), "HORIZON", 26, "medium", style.FG_MUTED, a_h, anchor="ra")
        tx.label((1860, 134), f"+{ms:.0f} ms", 46, "semibold", style.FG, a_h, anchor="ra")

        # Volume panel, lower left.  s10 (the volume-distribution chart) is cut from the
        # film, so this is the *only* place the 0.686 -> 0.090 m3 / 7.6x comparison is stated:
        # it is sized to be the read-out the narration's last clause points at, and the whole
        # block sits above y = 860 (the subtitles own y > 928).
        a_v0 = app(u, 0.26)
        a_v1 = app(u, 0.58)
        a_v2 = app(u, T_NUM)
        tx.label((60, 606), "MEDIAN SET VOLUME · 59 472 TEST WINDOWS", 26, "medium",
                 style.FG_MUTED, max(a_v0, a_v1))
        tx.swatch((70, 660), style.C_ISO, a_v0)
        tx.label((100, 634), f"{v_iso:.3f} m³", 46, "semibold", style.C_ISO, a_v0)
        tx.swatch((70, 728), style.C_OURS, a_v1)
        tx.label((100, 702), f"{v_ours:.3f} m³", 46, "semibold", style.C_OURS, a_v1)
        tx.rule((60, 774), 300, 2, style.GRID, a_v2 * 0.8)
        tx.label((60, 790), f"{v_ratio:.1f}× smaller", 56, "semibold", style.FG, a_v2)
        return tx.over(img)

    with common.FrameWriter(args) as w:
        for t in w.times():
            w.write(frame(t))


if __name__ == "__main__":
    main()
