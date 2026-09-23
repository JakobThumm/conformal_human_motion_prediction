"""s11 -- the PFH_D simulation, set up.

"Smaller is only worth having if it stays safe. So we take every prediction failure and drop it
around the robot: random place in a ten-meter circle, random orientation, random phase of a
pick-and-place trajectory."

Everything on screen is real:
  * the Panda's pick-and-place is played out of the 2358 interval-0 rows of
    `datasets/robot_reachable_sets/ablation_conformal_prediction_sets_panda_4ms.csv`
    (the 4 ms planning grid the shield monitors);
  * the 434 little humans are the *actual* true occupancy spheres (13 joint spheres, DIN 33402-2
    radii) of the 434 prediction-failure windows of
    `results/final/robot_shield_risk_volume/trials_ours_ood.npz`, i.e. the k_F = 434 of the paper
    run (row 6 of shield_risk_volume_results.csv);
  * the placement law is the simulator's own: (x, y) area-uniform in a disk of
    --pose_radius = 10 m, yaw uniform on [-pi, pi] (simulate_robot_shield.sample_robot_poses).
    The counter reads the simulation's real trial budget, N_D = 2.65e11 equivalent uniform
    placements (`equivalent_uniform_placements` of the paper run's CSV row, cached in the npz);
    the *animation* shows the 434 failing windows landing, which is all that can be drawn.
    The sim places the ROBOT around the human; this shot shows the identical relative geometry
    with the robot fixed at the origin, so `place_human_in_robot_frame` applies the inverse
    transform.  The cylinder's z-slab (+/-0.2 m) is drawn at its midpoint so every window stands
    on one floor.

Prepare (once, repo venv, ~3 min):
    XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python video/shots/s1x_shield_data.py

Render:
    /home/thumm/miniconda3/envs/chmp-video/bin/python video/shots/s11_sim_setup.py \
        --out video/build/shots/s11.mp4 --duration 12.5
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

N_FAIL = 434            # k_F, trials_ours_ood.npz: window_idx.size
N_PHASE = 2358          # monitored trajectories (4 ms grid), shield_risk_volume_results.csv
POSE_RADIUS = 10.0      # --pose_radius of final_results/robot_shield_safety_results_risk_volume.sh

# SPEC "Motion discipline": words appear, pictures may move.  Every label, number and caption
# in this shot enters on opacity alone over TXT_FADE seconds, sits at a FIXED screen position
# (leaders may track the 3-D point; the type may not) and is never animated out -- the opening
# robot caption is *instantly cleared* at the beat change, when the cloud starts to rain in.
# What moves: the camera pull-back, the placements landing, the counter's digits.
TXT_FADE = 0.20

# Chapter label, shared with s12.  One sans family (Inter) for the prose; PFH_D is a symbol, so
# it is set in Computer Modern on the same baseline -- the film's only licensed second face.
KICKER_PRE, KICKER_SYM, KICKER_POST = "Results · ", r"$\mathrm{PFH_D}$", " simulation"

_SUP = str.maketrans("0123456789-", "\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077"
                                    "\u2078\u2079\u207b")


def draw_kicker(o, xy=(64, 46), a=1.0):
    """Draw "Results · PFH_D simulation" at `xy` (top-left), returning nothing."""
    x, y = xy
    o.text((x, y), KICKER_PRE, style.CAPTION_SIZE, style.FG_MUTED, a=a)
    x += o.advance(KICKER_PRE, style.CAPTION_SIZE)
    x += o.math((x, y), KICKER_SYM, style.CAPTION_SIZE, style.FG_MUTED, a=a)
    o.text((x, y), KICKER_POST, style.CAPTION_SIZE, style.FG_MUTED, a=a)


def sci(x: float, digits: int = 2) -> str:
    """`2.6455e11` -> "2.65 × 10¹¹" (Inter has the superscript figures; no second face needed)."""
    e = int(np.floor(np.log10(abs(x))))
    return f"{x / 10.0 ** e:.{digits}f} × 10{str(e).translate(_SUP)}"


def lerp(a, b, u):
    return a + (b - a) * u


def build_placements(cache):
    """The 434 windows' true occupancy, placed by the simulator's own placement law."""
    rng = np.random.default_rng(11)
    n = N_FAIL
    rr = POSE_RADIUS * np.sqrt(rng.random(n))
    th = rng.uniform(0.0, 2.0 * np.pi, n)
    yaw = rng.uniform(-np.pi, np.pi, n)
    tx, ty = rr * np.cos(th), rr * np.sin(th)
    true_c, true_r = cache["true_c"], cache["true_r"]
    pts = np.empty((n, true_c.shape[2], 3), np.float32)
    for i in range(n):
        pts[i] = R.place_human_in_robot_frame(true_c[i, 0], yaw[i], (tx[i], ty[i], 0.0))
    # Landing order: a wave that sweeps from the far side of the disk towards the camera, so
    # the placements nearest the lens arrive last, when the camera has pulled back and they no
    # longer fill the frame.  Jittered so the wavefront is not a straight line.
    view = np.radians(-110.0)
    from_cam = 37.0 - (tx * np.cos(view) + ty * np.sin(view))
    order = np.argsort(-(from_cam + rng.normal(0.0, 1.4, n)))
    rank = np.empty(n, np.int64)
    rank[order] = np.arange(n)
    return dict(pts=pts, rad=true_r[:, 0], dist=rr, rank=rank, yaw=yaw,
                t=np.stack([tx, ty, np.zeros(n)], 1),
                shade=rng.uniform(0.86, 1.14, n).astype(np.float32))


def main():
    a = common.shot_args(__doc__, extra=lambda p: p.add_argument("--stills", default=""))
    cache = R.load_cache()
    P = build_placements(cache)
    D = a.duration

    # N_D -- the simulation's real trial budget, from the paper run's CSV row cached in the npz
    # (results/final/robot_shield_risk_volume/shield_risk_volume_results.csv,
    #  `equivalent_uniform_placements` = 2.6455e11).  Never hardcoded.
    csv = dict(zip([str(x) for x in cache["csv_keys"]], [str(x) for x in cache["csv_vals"]]))
    n_d_txt = sci(float(csv["equivalent_uniform_placements"]))
    # Fixed 3-digit field for the counter, so "/ 2.65 x 10^11 ..." never shifts as X grows.
    _dig = R.font(62, "SemiBold")
    field_w = 3.0 * max(_dig.getlength(c) for c in "0123456789")

    # narration beats (fractions of the shot, so a duration change keeps the rhythm)
    t_drop0, t_drop1 = 0.24 * D, 0.78 * D
    land = np.linspace(t_drop0, t_drop1, N_FAIL)[P["rank"]]

    green = R.rgb(style.C_TRUTH)
    robot_col = R.rgb(style.C_ROBOT)
    stills = {int(s) for s in a.stills.split(",") if s.strip()}

    def app(t, t0):
        """Text entry: opacity only, TXT_FADE seconds, starting at `t0` seconds."""
        return common.seg(t, t0, t0 + TXT_FADE)

    def frame(t):
        u = common.seg(t, 1.9, 0.60 * D, "smoother")            # the pull-back
        dist = lerp(4.7, 37.0, u)
        el = lerp(13.5, 25.5, u)
        az = -118.0 + 16.0 * common.seg(t, 0.0, D, "smooth")
        tgt = np.array([0.0, 0.0, lerp(1.34, 0.92, u)])
        cam = R.Camera(R.orbit_eye(tgt, dist, az, el), tgt, fov_y=24.0,
                       shift_y=lerp(96.0, 14.0, u))

        L = R.Layer()
        phase = int(np.clip(t / D, 0, 1) * (N_PHASE - 1))
        R.add_robot(L, cache, phase, col=robot_col)
        R.pedestal(L, 1.02, r=0.12)

        n_landed = 0
        for i in range(N_FAIL):
            dt = t - land[i]
            if dt < -0.001:
                continue
            n_landed += 1
            drop = 1.0 - common.ease(dt / 0.5, "out")
            dz = 2.3 * drop
            k = (0.25 + 0.26 * np.exp(-P["dist"][i] / 7.0)) * P["shade"][i]
            k *= 1.0 + 1.9 * drop                                 # flash brighter on landing
            pts = P["pts"][i] + (0.0, 0.0, dz)
            L.spheres(pts, P["rad"][i], green * k)

        pts_s, rad_s = L.footprints()
        ring_g = common.seg(t, 0.52 * D, 0.62 * D)      # the ring itself: a graphic, it may ease
        ring_a = app(t, 0.52 * D)                       # its label: opacity only, 0.2 s
        rings = [(POSE_RADIUS, R.rgb(style.FG_MUTED), 0.05, 0.28 + 0.62 * ring_g),
                 (1.35, robot_col, 0.035, 0.55 * (1.0 - 0.5 * u))]
        shadow = R.shadow_texture(pts_s, rad_s, 12.0)
        img = R.render_ground(cam, shadow=shadow, shadow_extent=12.0, grid_gain=1.45,
                              radius=POSE_RADIUS * 1.06, fog=lerp(20.0, 52.0, u), rings=rings,
                              glow=(3.0, R.rgb(style.C_ROBOT), 0.11))
        col, al, _ = R.render_layer(cam, L, ss=2, fog=lerp(26.0, 66.0, u), need_depth=False)
        img = R.over(img, col, al)

        o = R.Overlay(R.to_u8(img))
        draw_kicker(o, (64, 46), a=app(t, 0.10))
        o.text((64, 92), "Every prediction failure, replayed around the robot", 40,
               style.FG, "Medium", a=app(t, 0.40))

        # -- opening caption on the robot ---------------------------------------------
        # Fixed type at CAP_XY (the robot projects to x = 960 throughout and drifts < 60 px
        # vertically, but type does not travel); only the leader follows the 3-D point.  It is
        # cleared INSTANTLY at the beat change, one frame before the first placement lands --
        # no fade-out anywhere in this film.
        ra = app(t, 0.70) if t < 0.235 * D else 0.0
        if ra > 0.004:
            sx, sy, _ = cam.project(np.array([0.0, 0.0, 1.62]))
            o.line((sx + 26, sy - 14), (1052, 430), style.FG_MUTED, 0.55 * ra, 2)
            o.text((1064, 386), "Panda pick-and-place", 30, style.FG, "Medium", a=ra)
            o.text((1064, 420), "real swept occupancy, replayed",
                   style.CAPTION_SIZE, style.FG_MUTED, a=ra)

        # -- placement counter ---------------------------------------------------------
        # The visible cloud is the 434 failing windows; the denominator is the simulation's
        # actual N_D = 2.65e11 equivalent uniform placements.  X rises with the cloud, to 434.
        ca = app(t, 0.235 * D)     # swaps in on the exact frame the robot caption clears
        if ca > 0.004:
            x = 64.0
            o.text((x, 196), "Evaluating", style.BODY_SIZE, style.FG_MUTED, a=ca)
            x += o.advance("Evaluating ", style.BODY_SIZE)
            o.tabular((x + field_w, 162), f"{n_landed:d}", 62, style.C_TRUTH, "SemiBold",
                      a=ca, anchor="ra")
            o.text((x + field_w + 20, 196), f"/ {n_d_txt}  random placements",
                   style.BODY_SIZE, style.FG_MUTED, a=ca)

        # -- the three facts, revealed on the words ------------------------------------
        facts = [
            (f"{N_PHASE}", "trajectory phases · 4 ms planning grid", 0.33 * D),
            (f"{POSE_RADIUS:.0f} m", "placement radius, area-uniform disk", 0.57 * D),
            ("uniform", "base orientation, uniform on [-\u03c0, \u03c0]", 0.75 * D),
        ]
        for i, (big, small, t0) in enumerate(facts):
            fa = app(t, t0)
            if fa <= 0.004:
                continue
            y = 150 + i * 98
            o.text((1392, y), big, 44, style.FG, "SemiBold", a=fa)
            o.text((1392, y + 54), small, style.CAPTION_SIZE, style.FG_MUTED, a=fa)

        # -- keep the robot findable once the crowd closes in (2-D annotation) -----------
        # The ring and its leader are graphics and may follow the robot; the word "robot" is
        # pinned (the robot's own projection drifts ~16 px over the hold, type must not).
        ma = app(t, 0.46 * D)
        if ma > 0.004:
            ang = np.linspace(0, 2 * np.pi, 120)
            mk = np.stack([1.75 * np.cos(ang), 1.75 * np.sin(ang), np.zeros_like(ang)], -1)
            sx, sy, dep = cam.project(mk)
            for q in range(0, 120, 2):
                if dep[q] > 0.1 and dep[q + 1] > 0.1:
                    o.line((sx[q], sy[q]), (sx[q + 1], sy[q + 1]), style.C_ROBOT, 0.42 * ma, 2)
            rx, ry, _ = cam.project(np.array([0.0, 0.0, 1.75]))
            o.line((rx, ry - 8), (rx, 452), style.C_ROBOT, 0.35 * ma, 2)
            o.text((960, 410), "robot", style.CAPTION_SIZE, style.C_ROBOT, "Medium",
                   a=0.85 * ma, anchor="ma")

        # -- the 10 m ring label --------------------------------------------------------
        if ring_a > 0.004:
            ang = np.linspace(0, 2 * np.pi, 180)
            ring = np.stack([POSE_RADIUS * np.cos(ang), POSE_RADIUS * np.sin(ang),
                             np.zeros_like(ang)], -1)
            sx, sy, dep = cam.project(ring)
            k = int(np.argmax(np.where(dep > 0.1, sx, -1e9)))
            o.line((sx[k] + 4, sy[k]), (1714, 666), style.FG_MUTED, 0.5 * ring_a, 2)
            o.text((1726, 644), "10 m", 30, style.FG, "Medium", a=ring_a)
        return o.out()

    if stills:
        from PIL import Image
        for i in sorted(stills):
            Image.fromarray(frame(i / a.fps)).save(f"/tmp/s11_{i:04d}.png")
        return

    with common.FrameWriter(a) as w:
        for t in w.times():
            w.write(frame(t))


if __name__ == "__main__":
    main()
