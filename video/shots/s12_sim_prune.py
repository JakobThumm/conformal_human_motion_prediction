"""s12 -- the two geometric filters: why 2.65e11 placements are checkable.

"Almost none of those placements can reach the robot; two geometric filters prove it. In total
we check two hundred sixty-five billion placements on the GPU."

Continues s11 exactly: same 434 placements (`s11_sim_setup.build_placements`, seed 11), same
chapter label and camera language, now pushing in on the robot.

Like the manuscript (Sec. "Probability of Dangerous Failures Per Hour", where the implementation
detail is commented out), the film states the *simplified* story: we filter placements that
provably cannot produce a contact, and check the rest.  The closed-form witness-region sampling
that makes N_D = 2.65e11 affordable is deliberately not shown -- no "witness region W", no
P(W|F), no trial-count breakdown, so the shot is the animation plus three live counts.

What is actually computed here, live, with the shield's own arrays (cached by
`s1x_shield_data.py` from `simulate_robot_shield.build_shield_state`):

  * every shown placement is given a uniform trajectory phase j, exactly as the sim draws it;
  * **first filter** (the shield's level 4): || h_c[m, k_j] - A_j || <= R_j + h_r[m, k_j], i.e.
    the human's entire-motion bounding sphere must touch the robot's swept-motion bounding
    sphere.  In the robot's frame that is one ball of radius rho = R_j + h_r centred on A_j --
    the Minkowski sum drawn on screen.  7 of the 434 shown placements pass;
  * **second filter** (the shield's level 5): the per-robot-link / per-human-body sphere
    refinement (`TrialEvaluator.gate5`).  2 of those 7 survive.

The one run number on screen, N_D = 2.65e11 ("placements checked in total"), is read from the
cached CSV row of the paper run (`equivalent_uniform_placements` of
`results/final/robot_shield_risk_volume/shield_risk_volume_results.csv`, the 4e9-trial
conditional-conformal + OOD row).  It is the denominator s11's counter already showed.

Prepare (once, repo venv):
    XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python video/shots/s1x_shield_data.py

Render:
    /home/thumm/miniconda3/envs/chmp-video/bin/python video/shots/s12_sim_prune.py \
        --out video/build/shots/s12.mp4 --duration 9.2
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
import s11_sim_setup as S11  # noqa: E402 -- the identical placement draw, for continuity

N_FAIL = S11.N_FAIL
POSE_RADIUS = S11.POSE_RADIUS

# SPEC "Motion discipline": words appear, pictures may move.  Every label, number and caption
# here enters on opacity alone over TXT_FADE seconds, at a fixed screen position, and nothing
# is ever animated out -- the ball's caption now holds to the cut instead of fading away.
# What moves: the push-in, the cloud evaporating, the first-filter ball growing.
TXT_FADE = S11.TXT_FADE


def lerp(a, b, u):
    return a + (b - a) * u


def cull(cache, P):
    """First/second-filter verdicts for the 434 shown placements, with the shield's own test."""
    rng = np.random.default_rng(12)
    n = N_FAIL
    j = rng.integers(0, cache["traj_sphere_r"].size, n)          # uniform phase, as in the sim
    k = cache["k_of_traj"][j]
    A, Rj = cache["traj_sphere_c"], cache["traj_sphere_r"]
    hmi_c, hmi_r = cache["hmi_true_c"], cache["hmi_true_r"]
    oa_c, oa_r = cache["oa_true_c"], cache["oa_true_r"]
    Lc, Lr = cache["link_sphere_c"], cache["link_sphere_r"]

    rho = Rj[j] + hmi_r[np.arange(n), k]
    hc = np.stack([R.place_human_in_robot_frame(hmi_c[i, k[i]], P["yaw"][i], P["t"][i])[0]
                   for i in range(n)])
    l4 = np.linalg.norm(hc - A[j], axis=1) <= rho
    l5 = np.zeros(n, bool)
    for i in np.flatnonzero(l4):
        hb = R.place_human_in_robot_frame(oa_c[i, k[i]], P["yaw"][i], P["t"][i])
        d = np.linalg.norm(Lc[j[i]][:, None, :] - hb[None, :, :], axis=-1)
        l5[i] = bool((d <= Lr[j[i]][:, None] + oa_r[i, k[i]][None, :]).any())
    return dict(j=j, k=k, rho=rho, hc=hc, l4=l4, l5=l5)


def main():
    a = common.shot_args(__doc__, extra=lambda p: p.add_argument("--stills", default=""))
    cache = R.load_cache()
    P = S11.build_placements(cache)
    C = cull(cache, P)
    D = a.duration

    csv = dict(zip([str(x) for x in cache["csv_keys"]], [str(x) for x in cache["csv_vals"]]))
    n_d_txt = S11.sci(float(csv["equivalent_uniform_placements"]))   # 2.6455e11 -> "2.65 x 10^11"
    n_l4, n_l5 = int(C["l4"].sum()), int(C["l5"].sum())

    hero = int(np.flatnonzero(C["l5"])[0])          # the drawn filter ball belongs to this one
    hero_A = cache["traj_sphere_c"][C["j"][hero]].astype(np.float64)
    hero_R = float(cache["traj_sphere_r"][C["j"][hero]])
    hero_rho = float(C["rho"][hero])

    green = R.rgb(style.C_TRUTH)
    robot_col = R.rgb(style.C_ROBOT)
    stills = {int(s) for s in a.stills.split(",") if s.strip()}

    def app(t, t0):
        """Text entry: opacity only, TXT_FADE seconds, starting at `t0` seconds."""
        return common.seg(t, t0, t0 + TXT_FADE)

    def frame(t):
        u = common.seg(t, 0.16 * D, 0.58 * D, "smoother")        # the push-in
        dist = lerp(37.0, 9.2, u)
        el = lerp(25.5, 19.0, u)
        az = -102.0 + 10.0 * common.seg(t, 0.0, D, "smooth")
        tgt = np.array([0.0, 0.0, lerp(0.92, 1.15, u)])
        cam = R.Camera(R.orbit_eye(tgt, dist, az, el), tgt, fov_y=24.0,
                       shift_x=lerp(0.0, -60.0, u), shift_y=lerp(14.0, 40.0, u))

        # The placements outside W do not merely dim: they evaporate (radius -> 0), nearest
        # first, so the camera can fly into the cell without pushing through a wall of them.
        cut5 = common.seg(t, 0.40 * D, 0.52 * D, "smooth")

        L = R.Layer()
        phase = int(700 + 200 * t / D)
        R.add_robot(L, cache, phase, col=robot_col)
        R.pedestal(L, 1.02, r=0.12)
        for i in range(N_FAIL):
            base = (0.25 + 0.26 * np.exp(-P["dist"][i] / 7.0)) * P["shade"][i]
            if C["l5"][i]:
                col, rs = green * lerp(base, 1.35, common.seg(t, 0.05 * D, 0.22 * D)), 1.0
            elif C["l4"][i]:
                g = common.seg(t, 0.05 * D, 0.22 * D)
                col = green * lerp(base, lerp(1.15, 0.20, cut5), g)
                rs = 1.0 - 0.45 * cut5
            else:
                t0 = 0.04 * D + 0.14 * D * (P["dist"][i] / POSE_RADIUS)   # near ones go first
                g = common.seg(t, t0, t0 + 0.10 * D, "smooth")
                if g > 0.96:
                    continue
                col, rs = green * base * (1.0 - 0.7 * g), (1.0 - g) ** 0.6
            L.spheres(P["pts"][i], P["rad"][i] * rs, col)

        # -- the first filter's ball: robot swept bound (+) human motion bound ---------
        gb = common.seg(t, 0.26 * D, 0.34 * D, "smooth")          # robot bounding sphere
        gw = common.seg(t, 0.32 * D, 0.44 * D, "smoother")        # grow to the filter ball
        G = R.Layer()
        if gb > 0.004:
            G.sphere(hero_A, lerp(hero_R, hero_rho, gw), R.rgb(style.YELLOW))

        pts_s, rad_s = L.footprints()
        shadow = R.shadow_texture(pts_s, rad_s, 12.0)
        img, gz = R.render_ground(
            cam, shadow=shadow, shadow_extent=12.0, grid_gain=1.45, radius=10.6,
            fog=lerp(52.0, 22.0, u), return_depth=True,
            rings=[(10.0, R.rgb(style.FG_MUTED), 0.05, 0.9 * (1.0 - u) ** 2),
                   (1.35, robot_col, 0.035, 0.34)],
            glow=(3.0, R.rgb(style.C_ROBOT), 0.11))
        col, al, zs = R.render_layer(cam, L, ss=2, fog=lerp(66.0, 30.0, u))
        img = R.over(img, col, al)
        if gb > 0.004:
            occ = np.minimum(zs, gz)
            gc, ga = R.render_glass(cam, G, occluder=occ, down=2,
                                    col=R.rgb(style.YELLOW), alpha=0.22 * gb)
            img = R.over(img, gc, ga)

        o = R.Overlay(R.to_u8(img))
        S11.draw_kicker(o, (64, 46))
        o.text((64, 92), "Almost no placement can reach the robot", 40, style.FG, "Medium",
               a=app(t, 0.10))

        # -- the filter ladder, left column --------------------------------------------
        # 434 -> 7 -> 2, all three computed live by `cull` from the shield's own tests.  The
        # second line of each rung says what the filter actually compares; keeping it on its
        # own line holds the column narrow enough not to crowd the 3-D cell.
        steps = [
            (f"{N_FAIL}", "placements shown", "", style.C_TRUTH, 0.10 * D),
            (f"{n_l4}", "pass first filter", "entire motion sphere", style.FG, 0.24 * D),
            (f"{n_l5}", "pass second filter", "link and body spheres", style.FG, 0.42 * D),
        ]
        for i, (big, small, sub, ccol, t0) in enumerate(steps):
            sa = app(t, t0)
            if sa <= 0.004:
                continue
            y = 168 + i * 104
            o.text((64, y), big, 42, ccol, "SemiBold", a=sa)
            w, _ = o.measure(big, 42, "SemiBold")
            xs = 64 + max(w, 92) + 18
            o.text((xs, y + 4), small, style.CAPTION_SIZE, style.FG_MUTED, a=sa)
            if sub:
                o.text((xs, y + 36), sub, style.CAPTION_SIZE, style.FG_MUTED, a=0.60 * sa)
            if i:
                o.line((78, y - 26), (78, y - 8), style.GRID, 0.8 * sa, 2)

        # -- the ball's caption ---------------------------------------------------------
        # Fixed in the empty lower-left, with a leader onto the ball: the ball moves a lot
        # during the push-in and a caption pinned to it would collide with the ladder.
        ba = app(t, 0.34 * D)          # entry only -- it holds to the cut
        if ba > 0.004:
            sx, sy, sd = cam.project(hero_A)
            rs = cam.focal * hero_rho / max(float(sd), 1e-3)
            sub = "robot motion sphere  ⊕  human motion sphere"
            o.text((96, 686), "First filter", 32, style.YELLOW, "Medium", a=ba)
            o.text((96, 726), sub, style.CAPTION_SIZE, style.FG_MUTED, a=ba)
            w, _ = o.measure(sub, style.CAPTION_SIZE)
            o.line((96 + w + 22, 718), (sx - rs * 0.74, sy + rs * 0.60),
                   style.YELLOW, 0.45 * ba, 2)

        # -- the one surviving statistic, on s11's right-hand column axis ----------------
        # The narration's second clause ("in total we check two hundred sixty-five billion
        # placements on the GPU") runs 3.9-6.0 s of 9.2 s; the block lands just before it.
        pa = app(t, 0.42 * D)
        if pa > 0.004:
            o.text((1392, 168), n_d_txt, 52, style.FG, "SemiBold", a=pa)
            o.text((1392, 240), "placements checked in total", style.CAPTION_SIZE,
                   style.FG_MUTED, a=pa)
            o.text((1392, 274), "on one RTX 5090", style.CAPTION_SIZE, style.FG_MUTED, a=pa)
        return o.out()

    if stills:
        from PIL import Image
        for i in sorted(stills):
            Image.fromarray(frame(i / a.fps)).save(f"/tmp/s12_{i:04d}.png")
        return

    with common.FrameWriter(a) as w:
        for t in w.times():
            w.write(frame(t))


if __name__ == "__main__":
    main()
