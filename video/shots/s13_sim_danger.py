"""s13 -- the honest failure case: one of the four dangerous trials, replayed exactly.

"Four come back dangerous. The shield verified its trajectory, but the true occupancy escaped the
prediction and touched the robot. This is one of the four, replayed exactly."

The trial is row 0 of `dangerous_trials` in
`results/final/robot_shield_risk_volume/trials_ours_ood.npz`:

    window m = 238   trajectory phase j = 544   yaw = 159.6 deg   t = (0.361, -1.281, -0.165) m

Re-scored by `video/shots/s1x_shield_data.py` with the repo's own `TrialEvaluator` on the CPU in
float64:  **verified = True, contact = True -> dangerous**, and it passes both the level-4 and the
level-5 gates.  The same pass measures, with `simulate_robot_shield.segment_point_distances`:

  * verification clearance  **+2.60 mm** -- the closest any of the 9 monitored intervals' planned
    capsules comes to any of the 13 conformal prediction spheres.  Empty intersection, so
    c_safe = 1 and the shield lets the trajectory run.
  * contact overlap **+2.96 mm** at interval 8 (t + 0.16 s), robot link 7 against the head
    sphere, horizon step 4.  The window's recorded escape is **7.3 cm** at the head, step 8
    (`escape_m`/`escape_joint`/`escape_step`).

Everything drawn is that geometry, in the trial's own frame, shifted so the robot base sits at the
origin and the floor is z = 0 (the placement's z-offset is kept: p_display = p_world - (tx, ty, 0)).
The arm is the real interval-0 occupancy of the 4 ms grid, advanced from phase j.

Prepare (once, repo venv):
    XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python video/shots/s1x_shield_data.py

Render:
    /home/thumm/miniconda3/envs/chmp-video/bin/python video/shots/s13_sim_danger.py \
        --out video/build/shots/s13.mp4 --duration 11.5
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import style  # noqa: E402
import s1x_render3d as R  # noqa: E402

KICKER = "Results · dangerous-failure simulation"
# JOINT_NAMES_13, motion_prediction/h36m_settings.py
JOINTS = ["head", "L shoulder", "R shoulder", "L elbow", "R elbow", "L wrist", "R wrist",
          "L hip", "R hip", "L knee", "R knee", "L ankle", "R ankle"]
TRIAL = 0               # which of the four dangerous_trials rows to replay
TAU_END = 0.32          # s of replayed wall-clock (= horizon step 8, the escape maximum)
DT_ROBOT = 0.004        # the planning grid


def lerp(a, b, u):
    return a + (b - a) * u


def main():
    a = common.shot_args(__doc__, extra=lambda p: p.add_argument("--stills", default=""))
    c = R.load_cache()
    D = a.duration
    d = TRIAL
    m, j = int(c["dang_m"][d]), int(c["dang_j"][d])
    yaw, tx, ty, tz = c["dang_pose"][d]
    rotz = R.yaw_mat(yaw)
    shift = np.array([tx, ty, 0.0])
    base_off = np.array([0.0, 0.0, tz])

    def hum(p):                       # human/world point -> display
        return np.asarray(p, np.float64) - shift

    def rob(p):                       # robot-base point -> display
        return np.asarray(p, np.float64) @ rotz.T + base_off

    hz = c["horizon_times"]
    pred_c, pred_r = c["pred_c"][m], c["pred_r"][m]
    true_c, true_r = c["true_c"][m], c["true_r"][m]
    n_rows = int(c["n_rows"][d])
    pl_p1, pl_p2, pl_r = c["pl_p1"][d, :n_rows], c["pl_p2"][d, :n_rows], c["pl_r"][d, :n_rows]
    pl_tp, pl_step = c["pl_tp"][d, :n_rows], c["pl_step"][d, :n_rows]
    clearance = float(c["clearance"][d])
    esc_max = float(c["escape_m"][m])             # the run's recorded escape for this window
    esc_joint = int(c["escape_joint"][m])          # 0 = Nose, the head sphere
    tau_contact = float(pl_tp[int(c["contact_row"][d])])

    col_blue = R.rgb(style.C_OURS)
    col_green = R.rgb(style.C_TRUTH)
    col_robot = R.rgb(style.C_ROBOT)
    col_red = R.rgb(style.C_DANGER)

    # the escape of the head, per horizon step (the quantity `escape_margins` maximises)
    esc = (np.linalg.norm(true_c[:, esc_joint] - pred_c[:, esc_joint], axis=-1)
           + true_r[:, esc_joint] - pred_r[:, esc_joint])
    # integrity: the number annotated on screen must be the run's own escape_m (step 0 is the
    # measured pose, which `escape_margins` excludes).
    assert abs(float(esc[1:].max()) - esc_max) < 1e-6, (float(esc[1:].max()), esc_max)

    def contact_point(tau):
        """Deepest (robot link, human body) overlap at wall-clock tau; None if separated."""
        ph = j + int(round(tau / DT_ROBOT))
        p1, p2, rr = (rob(c["robot_p1"][ph]), rob(c["robot_p2"][ph]), c["robot_r"][ph])
        s = int(np.argmin(np.abs(hz - tau)))
        hp, hr = hum(true_c[s]), true_r[s]
        ab = p2 - p1
        t = np.clip(((hp[None] - p1[:, None]) * ab[:, None]).sum(-1)
                    / np.maximum((ab * ab).sum(-1), 1e-18)[:, None], 0, 1)
        proj = p1[:, None] + t[..., None] * ab[:, None]
        dist = np.linalg.norm(hp[None] - proj, axis=-1)
        pen = rr[:, None] + hr[None, :] - dist
        i, k = np.unravel_index(int(np.argmax(pen)), pen.shape)
        if pen[i, k] <= 0:
            return None
        mid = 0.5 * (proj[i, k] + hp[k])
        return mid, int(i), int(k), float(pen[i, k])

    stills = {int(s) for s in a.stills.split(",") if s.strip()}

    # beats (fractions of D so a duration change keeps the rhythm)
    B_SWEEP0, B_SWEEP1 = 0.10 * D, 0.38 * D          # the verification sweep
    B_PLAY0, B_PLAY1 = 0.40 * D, 0.76 * D            # the replay

    def frame(t):
        u = common.seg(t, 0.15, 0.72 * D, "smoother")
        dist = lerp(6.7, 5.35, u)
        el = lerp(20.0, 13.0, u)
        az = lerp(-40.0, -6.0, common.seg(t, 0.0, 0.86 * D, "smooth"))
        tgt = np.array([-0.15, lerp(0.56, 0.64, u), lerp(0.84, 0.88, u)])
        cam = R.Camera(R.orbit_eye(tgt, dist, az, el), tgt, fov_y=24.0,
                       shift_x=lerp(50.0, 120.0, u), shift_y=lerp(-4.0, 10.0, u))

        sweep = common.seg(t, B_SWEEP0, B_SWEEP1, "smooth")
        play = common.seg(t, B_PLAY0, B_PLAY1, "smooth")
        tau = TAU_END * play
        s_now = int(np.argmin(np.abs(hz - tau)))
        replaying = t >= B_PLAY0 - 0.05
        i_sweep = int(np.clip(sweep * n_rows, 0, n_rows))       # intervals verified so far
        s_pred = int(pl_step[min(i_sweep, n_rows - 1)]) if not replaying else s_now

        hit = contact_point(tau) if replaying else None
        flash = 0.0
        if hit is not None:
            flash = np.exp(-max(0.0, (tau - tau_contact)) / 0.05)

        # ---- solid layer: the moving arm, the pedestal, (later) the true human ----------
        L = R.Layer()
        ph = j + int(round(tau / DT_ROBOT))
        p1, p2, rr = rob(c["robot_p1"][ph]), rob(c["robot_p2"][ph]), c["robot_r"][ph]
        shades = np.linspace(0.62, 1.0, 7)[:, None] * col_robot[None, :]
        if hit is not None:
            shades[hit[1]] = col_red * (1.05 + 0.5 * flash)
        L.capsules(p1, p2, rr, shades.astype(np.float32))
        R.pedestal(L, 1.02 + tz, r=0.12)
        ta = common.seg(t, B_PLAY0 - 0.35, B_PLAY0 + 0.25)
        if ta > 0.004:
            cols = np.broadcast_to(col_green * lerp(0.35, 1.0, ta), (13, 3)).copy()
            if hit is not None:                       # the contacting body sphere goes red
                cols[hit[2]] = col_red * (1.05 + 0.5 * flash)
            L.spheres(hum(true_c[s_now]), true_r[s_now], cols)
            R.add_skeleton(L, hum(true_c[s_now]), col_green * lerp(0.4, 1.35, ta), r=0.024)

        # ---- glass layer: the verified robot occupancy + the conformal set --------------
        G = R.Layer()
        n_show = n_rows if replaying else i_sweep
        if n_show:
            G.capsules(rob(pl_p1[:n_show].reshape(-1, 3)), rob(pl_p2[:n_show].reshape(-1, 3)),
                       pl_r[:n_show].ravel(), col_robot * 0.9)
        sa = common.seg(t, 0.06 * D, 0.14 * D)
        if sa > 0.004:
            G.spheres(hum(pred_c[s_pred]), pred_r[s_pred], col_blue)
            R.add_skeleton(L, hum(pred_c[s_pred]), R.rgb(style.C_PRED) * lerp(0.2, 1.25, sa),
                           r=0.021, joint_r=0.035)

        pts_s, rad_s = L.footprints()
        img, gz = R.render_ground(cam, shadow=R.shadow_texture(pts_s, rad_s, 4.0, strength=1.4),
                                  shadow_extent=4.0, grid_gain=1.3, radius=5.0, fog=14.0,
                                  return_depth=True, rings=(),
                                  glow=(2.2, R.rgb(style.C_ROBOT), 0.10))
        col, al, zs = R.render_layer(cam, L, ss=2, fog=22.0)
        img = R.over(img, col, al)
        # once contact is on screen the blue haze steps back, so the red tangency is legible
        fade_set = 1.0 - 0.45 * (0.0 if hit is None else
                                 common.seg(tau, tau_contact, tau_contact + 0.05))
        gc2, ga2 = R.render_glass(cam, G, occluder=np.minimum(zs, gz), down=2,
                                  alpha=0.50 * fade_set * min(1.0, sa + (n_show > 0)))
        img = R.over(img, gc2, ga2)

        o = R.Overlay(R.to_u8(img))
        o.text((64, 46), KICKER, style.CAPTION_SIZE, style.FG_MUTED)
        o.text((64, 92), "One of the four, replayed exactly", 40, style.FG, "Medium",
               a=common.seg(t, 0.1, 0.9))

        # -- k_D headline ---------------------------------------------------------------
        ka = common.seg(t, 0.3, 1.1)
        if ka > 0.004:
            o.text((64, 168), "4", 62, style.C_DANGER, "SemiBold", a=ka)
            o.text((116, 200), "dangerous trials in 4 × 10⁹", style.BODY_SIZE,
                   style.FG_MUTED, a=ka)

        # -- the shield's verdict card ---------------------------------------------------
        va = common.seg(t, 0.30 * D, 0.40 * D)
        if va > 0.004:
            o.text((64, 286), "c_safe = 1   ·   trajectory verified", style.BODY_SIZE,
                   style.C_TRUTH, "Medium", a=va)
            o.text((64, 326), f"clears the swept occupancy by {clearance * 1000:.1f} mm",
                   style.CAPTION_SIZE, style.FG_MUTED, a=va)
        da = 0.0 if hit is None else common.seg(tau, tau_contact, tau_contact + 0.02)
        if da > 0.004:
            o.text((64, 386), f"contact   ·   link {hit[1] + 1} × {JOINTS[hit[2]]}",
                   style.BODY_SIZE, style.C_DANGER, "Medium", a=da)
            o.text((64, 426), "verified, and in contact  →  dangerous",
                   style.CAPTION_SIZE, style.FG_MUTED, a=da)

        # -- legend ----------------------------------------------------------------------
        la = (common.seg(t, 0.08 * D, 0.18 * D)
              * (1.0 - common.seg(t, 0.44 * D, 0.52 * D)))
        if la > 0.004:
            for i, (cc, lab) in enumerate([
                    (style.C_ROBOT, "robot swept occupancy"),
                    (style.C_OURS, "conformal prediction set"),
                    (style.C_PRED, "predicted mean pose"),
                    (style.C_TRUTH, "true human occupancy")]):
                y = 172 + i * 44
                o.rect((1528, y, 1548, y + 20), fill=cc, fill_a=0.85 * la, w=0)
                o.text((1564, y - 4), lab, style.CAPTION_SIZE, style.FG_MUTED, a=la)

        # -- the escape callout on the head ----------------------------------------------
        if replaying and esc[s_now] > 0.001:
            ea = common.seg(t, B_PLAY0 + 0.4, B_PLAY0 + 1.0)
            hx, hy, _ = cam.project(hum(true_c[s_now, esc_joint]))
            o.line((hx + 18, hy - 14), (hx + 96, hy - 92), style.C_TRUTH, 0.55 * ea, 2)
            o.text((hx + 106, hy - 132), f"{esc[s_now] * 100:.1f} cm outside the set", 32,
                   style.C_TRUTH, "Medium", a=ea)
            o.text((hx + 106, hy - 96), f"true head, horizon step {s_now}",
                   style.CAPTION_SIZE, style.FG_MUTED, a=ea)

        # -- the contact marker ------------------------------------------------------------
        if hit is not None:
            cx, cy, _ = cam.project(hit[0])
            o.circle((cx, cy), 34 + 34 * flash, style.C_DANGER, 0.85 - 0.5 * flash, 3)

        # -- the trial's identity ------------------------------------------------------
        ia = common.seg(t, 0.72 * D, 0.82 * D)
        if ia > 0.004:
            o.text((64, 812), f"replayed exactly:   window m = {m}   ·   "
                              f"trajectory phase j = {j}",
                   style.CAPTION_SIZE, style.FG_MUTED, a=ia)
            o.text((64, 848), f"yaw = {np.degrees(yaw):.1f}°   ·   base t = "
                              f"({tx:.2f}, {ty:.2f}, {tz:.2f}) m",
                   style.CAPTION_SIZE, style.FG_MUTED, a=ia)

        return o.out()

    if stills:
        from PIL import Image
        for i in sorted(stills):
            Image.fromarray(frame(i / a.fps)).save(f"/tmp/s13_{i:04d}.png")
        return

    with common.FrameWriter(a) as w:
        for t in w.times():
            w.write(frame(t))


if __name__ == "__main__":
    main()
