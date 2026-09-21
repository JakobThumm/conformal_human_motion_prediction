"""Bound the robot shield's PFH_D by *importance-sampling the placement space by volume*.

Same factorisation as :mod:`examples.simulate_shield_failure_risk`, same lemma, but the placement
integral is no longer a rejection sampler. Instead of drawing placements uniformly over the whole
workspace and letting the culling hierarchy throw ~99 % of them away, we draw them *directly from
the sub-region that can produce a dangerous failure at all* and pay for that with an exactly
computable volume ratio.

The events (all per verification cycle):

  * ``F`` -- prediction failure: the true human occupancy escapes the predicted set at some
    horizon step / joint,  ``exists j,k: p_k^j not in S_k^j``.
  * ``D`` -- dangerous failure: a contact occurs although the shield verified the trajectory
    (``c_safe = 1``). Because the predicted sets over-approximate the true occupancy, an empty
    intersection in the verification excludes contact unless the truth left the set, so

        D  =>  F,        P(D) = P(F) P(D | F).                                                 (1)

  * ``W`` -- the *witness region*: the placements at which the robot's swept occupancy can reach
    the human at all. It is the shield's own level-4 bounding-sphere test, hence a **necessary
    condition for contact**, hence

        D  =>  W,        P(D | F) = P(W | F) P(D | F, W).                                      (2)

A trial here is one verification cycle: a triple ``(m, j, x)`` of a failing prediction window
``m``, a phase ``j`` of the long-horizon robot trajectory, and a relative placement
``x = (yaw, t)`` of the robot base. Under the reference (uniform) placement distribution all three
are independent and uniform, so ``P(W | F)`` is a pure volume ratio,

    P(W | F) = V(W) / V(F) = mean over (m, j) of  vol(B_mj n Cyl) / vol(Cyl),                  (3)

where ``Cyl`` is the placement cylinder (an xy-disk of ``--pose_radius`` times the z-slab
``+/- --pose_z_offset``) and ``B_mj`` is the level-4 ball: the base translations at which the
bounding sphere of trajectory ``j``'s *true* capsules touches the bounding sphere of window ``m``'s
true occupancy over ``j``'s duration. Two geometric facts make (3) exact and closed form:

  * the ball's z-offset ``h_z - A_jz`` does not depend on the base yaw (yaw rotates about z), so
    ``vol(B_mj n Cyl)`` is yaw-independent and equals a spherical-band integral; and
  * yaw only translates the ball inside the disk, so as long as the ball never pokes out of the
    disk (asserted at startup) the disk constraint is inactive.

``V(W)`` is therefore computed, not estimated -- no Monte-Carlo error enters (3) -- and
``(m, j, yaw, t)`` can be drawn *exactly* from the uniform-on-``W`` conditional: pick ``(m, j)``
with probability proportional to ``vol(B_mj n Cyl)``, ``yaw`` uniform, then ``t`` uniform in the
band. Every trial is a placement that matters.

Refining W for free. The level-5 test (per robot-link sphere vs. per human-body sphere) is a
*strictly tighter* necessary condition for contact than the level-4 ball. A trial it rejects is
**provably** not dangerous, so it is resolved as a 0 outcome for 91 distance tests instead of the
full capsule kernel -- it still counts in ``N_W``, and no extra statistics are needed. On H36M this
removes ~74 % of the drawn trials, so the effective region is ~4.5e-3 of the cylinder while
``P(W|F)`` in (3) stays the exact level-4 number.

Confidence composition. The two factors are measured on independent samples (recorded windows /
placements we generate), so both bounds hold simultaneously at

    1 - eps_D = (1 - eps_F)(1 - eps_{D|F}),   allocated symmetrically as
    eps_F = eps_{D|F} = 1 - sqrt(1 - eps_D).                                                   (4)

With ``B^-1(q; a, b)`` the Beta quantile (so ``B^-1(1-eps; k+1, n-k)`` is the one-sided
Clopper-Pearson upper limit after ``k`` events in ``n`` trials) and ``N_h = 3600 s / t_cycle``
verification cycles per operating hour,

    PFH_D <= N_h  B^-1(1-eps_F; k_eff+1, n_eff-k_eff)  [V(W)/V(F)]
                  B^-1(1-eps_{D|F}; k_D+1, N_W-k_D)                                            (5)

for ``k_D`` dangerous trials among the ``N_W`` placements drawn *from W* -- which are i.i.d. by
construction, so their binomial bound needs no independence assumption -- and, on the data side,
for the ``k_F`` failures among the ``N_F = |Z_test|`` recorded prediction windows.

Note the sample size in (5) is ``N_W``, not the ``N_D`` a uniform sampler would need: the Beta
quantile bounds the *conditional* ``P(D | F, W)``, and ``V(W)/V(F)`` supplies the rest. The
equivalent uniform count is ``N_D = N_W / P(W|F)`` -- reported by the run -- and the two routes
give the identical bound, because ``D => W`` means a dangerous uniform draw always lands in W, so
the same ``k_D`` is observed either way.

Those windows are NOT independent: consecutive ones share ``K_I - 1`` input poses and re-observe
the same physical escape. The dependence is handled on the *interval*, never on the point estimate
``P_hat(F) = k_F / N_F`` (which is the estimator eq. 1 is derived for): the count is deflated to an
effective sample size

    n_eff = N_F / (2 L_corr),        k_eff = P_hat(F) n_eff,                                   (6)

with ``L_corr = 1 + 2 sum_l rho_l`` the integrated autocorrelation time of the failure indicator
(Sokal automatic windowing; ``--l_corr`` overrides it) and the 2 the H36M loader's two 50->25 fps
phase offsets, which put every motion in the series twice and are therefore not removed by
sub-sampling at any stride. The P(F) factor of eq. (5) is then
``B^-1(1-eps_F; k_eff+1, n_eff-k_eff)``. This is the same P(F) treatment -- the same functions --
as :mod:`examples.simulate_shield_failure_risk`, so the two estimators' data factors are directly
comparable; only the placement factor differs.

Why this is fast. The old script's unit of work is one placement evaluated against *every* failing
window and *every* monitored trajectory (a dense ``n_traj x M`` GPU kernel), and ~88 % of
placements are discarded by the level-1 gate before that. Here the unit of work is one
``(m, j, placement)`` triple -- ~77 robot capsules against ~13 human body spheres -- and every
triple lies in ``W``. The same bound comes from ~1e-2 of the draws at ~1e-4 of the work per draw.

Both self-tests of the chain are available and cheap on this kernel:

  ``--verify_gate N``   draw N trials from the *full* cylinder (not from W), score all of them
                        exactly, and assert that no **contact** falls outside W -- the empirical
                        test of (2). Contact, not the far rarer dangerous event, is what W claims
                        to bracket (``D => contact => W``), so this exercises the implication with
                        ~1e4 events per 1e6 trials instead of ~0. Also yields an independent,
                        unbiased P(D|F) estimate to cross-check the volume-weighted one.
  ``--verify_lemma N``  draw N trials on *non-failure* windows (from their own W, the most
                        sensitive place to look) and assert zero verified-AND-contact -- the
                        empirical test of (1).
  ``--parity N``        assert the per-trial kernel reproduces ``run_pose_pure`` exactly.

Units: metres, seconds, rates 1/h. Run::

    XLA_PYTHON_CLIENT_PREALLOCATE=false python -m \
        conformal_human_motion_prediction.examples.simulate_shield_failure_risk_volume \
        --backend gpu --num_trials 1000000000 --verify_gate 2000000 --verify_lemma 10000000

Measured throughput on an RTX 5090: ~4.5e6 trials/s, i.e. ~2.6e8 equivalent uniform placements per
second. The cross-product script reaches ~7e3 placements/s, so a bound that took it hours lands
here in seconds; the run is bound by the host-side sampler, not the GPU.
"""
import argparse
import os
import time
from types import SimpleNamespace

import numpy as np
from tqdm import tqdm

# Importing this module also performs its argv-peeking jax-platform setup (--backend gpu keeps the
# GPU and enables x64; anything else pins jax to the CPU), so it must happen before jax is touched.
from conformal_human_motion_prediction.examples.simulate_robot_shield import (
    N_LINKS,
    bound_spheres,
    build_shield_state,
    last_step_below,
    nearest_step,
    root_dir,
    run_pose_pure,
    sample_robot_poses,
    write_results_csv,
)
# Every statistic the two estimators share is imported, not re-derived, so the P(F) factor and the
# confidence composition cannot drift between them. Only the placement factor differs.
from conformal_human_motion_prediction.examples.simulate_shield_failure_risk import (
    autocorrelation,
    build_failure_state,
    clopper_pearson,
    clopper_pearson_upper,
    integrated_autocorr_time,
    pl_label,
    run_lengths,
    split_confidence,
)

# Padded capsule slots get radius -PAD_R, so ``distance <= radius_sum`` can never fire for them
# (distances are >= 0). Representable in float32, the default kernel dtype.
PAD_R = 1e30
# The (m, j) draw uses rejection against a uniform proposal while its acceptance rate stays above
# this, and falls back to the inverse-CDF otherwise. Rejection wins because the weights span only
# a small range (one gather per proposal vs ~20 cache-missing binary-search steps per sample); the
# fallback keeps a pathological weight distribution from stalling the loop. Both are exact.
MIN_PROPOSAL_ACCEPT = 0.05
# The H36M loader emits both 50->25 fps phase offsets (the `for offset in [0, 1]` loop is OUTER),
# so every motion appears twice in the window series: 20 ms apart in time, one contiguous block
# apart in row order -- and therefore NOT removed by sub-sampling at any stride. It deflates the
# effective sample size behind P(F) on top of L_corr (eq. 6).
PHASE_OFFSETS = 2
# Relative tolerance on the aggregate counts of the --parity check. Only exact-tangency pairs may
# differ between the kernel and run_pose_pure (same arithmetic, different order); float32 flips
# ~3e-5 of them, float64 none. The verified-but-contact SET is compared with no tolerance at all.
PARITY_COUNT_TOL = 1e-4


# --------------------------------------------------------------------------- step 2: the region W


def ball_slab_volume(rho, z0, z_off):
    """Volume of a ball of radius ``rho``, centred ``z0`` above the mid-plane of the slab
    ``|z| <= z_off``, intersected with that slab.

    The slab cross-section at height ``z`` is a disk of area ``pi (rho^2 - (z - z0)^2)``, so

        vol = pi * int_lo^hi (rho^2 - (z - z0)^2) dz,   lo = max(-z_off, z0 - rho),
                                                        hi = min(+z_off, z0 + rho),

    which is the closed form below. Broadcasts over array arguments; a ball missing the slab
    entirely gives ``hi = lo`` and hence exactly 0.
    """
    lo = np.maximum(-z_off, z0 - rho)
    hi = np.maximum(np.minimum(z_off, z0 + rho), lo)
    return np.pi * (rho ** 2 * (hi - lo) - ((hi - z0) ** 3 - (lo - z0) ** 3) / 3.0)


def build_trial_geometry(ctx, args):
    """Per-``(window, trajectory)`` level-4 balls, their exact volumes and the sampling weights.

    The trial space is ``(m, j, yaw, t)``: failing window ``m``, monitored trajectory ``j``, base
    yaw and base translation. For a fixed ``(m, j)`` the level-4 true-side test of
    :func:`simulate_robot_shield.traj_candidates` reads

        || h_c[m, k_j] - (Rz(yaw) A_j + t) || <= R_j + h_r[m, k_j]  =:  rho_mj ,

    i.e. ``t`` lies in a ball of radius ``rho_mj`` centred at ``h_c[m,k_j] - Rz(yaw) A_j``, where
    ``(A_j, R_j)`` is the bounding sphere of trajectory ``j``'s *true* (future interval-0) capsules
    in the base frame and ``(h_c, h_r)[m, k_j]`` is window ``m``'s true occupancy bounded over
    horizon steps ``0..k_j``. Both are exactly the arrays the shield's own hierarchy uses, so the
    test is sound by inheritance: anything it culls provably cannot touch.

    Yaw enters only through ``Rz(yaw) A_j``, which leaves the z-coordinate alone and merely
    translates the ball inside the xy-disk. The ball's slab-truncated volume is therefore
    yaw-independent (:func:`ball_slab_volume` with ``z0 = h_z - A_jz``), and the disk constraint is
    inactive provided the ball never reaches the disk boundary -- asserted here, because a poking
    ball would break both the closed-form volume and the sampler.

    Trajectories with no valid future interval-0 row cannot produce a contact at all; they get
    ``rho = 0`` and weight 0, which keeps them in the ``(m, j)`` trial space (so the
    ``1/(M n_traj)`` average of eq. 3 stays over the full space) while never being drawn.

    The ``[M, n_traj]`` weight table is the only quadratic array; the per-trial ``rho``/``z0`` are
    recomputed from the ``[M, kmax+1]`` hierarchy and the ``[n_traj]`` trajectory arrays rather
    than stored, and the inverse-CDF (a second such array) is built only if the rejection sampler
    is not applicable. Cap ``M`` with ``--max_failures_eval`` if the table gets unwieldy.
    """
    st = ctx.st
    if not st.overapprox:
        raise SystemExit("--no-overapprox is not supported: the witness region IS the "
                         "over-approximation hierarchy's level-4/5 test.")
    nT, M = ctx.n_traj, ctx.M
    k_of_traj = np.array([m[3] for m in st.traj_meta], dtype=np.int64)
    A = np.zeros((nT, 3))
    R = np.zeros(nT)
    Lc = np.zeros((nT, N_LINKS, 3))
    Lr = np.full((nT, N_LINKS), -PAD_R)
    has_true = np.zeros(nT, dtype=bool)
    for j, meta in enumerate(st.traj_meta):
        rct_c, rct_r = meta[7], meta[8]
        if rct_c is None:                      # no future interval-0 row -> no contact possible
            continue
        c, r = bound_spheres(rct_c[None], rct_r[None])
        A[j], R[j] = c[0], float(r[0])
        Lc[j], Lr[j] = rct_c, rct_r
        has_true[j] = True

    geom = SimpleNamespace(
        M=M, nT=nT, k_of_traj=k_of_traj, A=A, R=np.where(has_true, R, -PAD_R), Lc=Lc, Lr=Lr,
        has_true=has_true, hc_k=st.hmi_true_c, hr_k=st.hmi_true_r, z_off=float(args.pose_z_offset),
        v_cyl=np.pi * args.pose_radius ** 2 * 2.0 * args.pose_z_offset, vol=None,
    )
    # [M, nT] weight table. rho/z0 come from the same helper the sampler uses, so the weights and
    # the drawn points can never disagree about which ball is being sampled.
    rho, z0 = level4_ball(geom, np.arange(M)[:, None], np.arange(nT)[None, :])
    vol = np.where(rho > 0.0, ball_slab_volume(np.maximum(rho, 0.0), z0, geom.z_off), 0.0)
    if not vol.any():
        raise SystemExit("Every level-4 ball is empty -- no placement in the cylinder can bring "
                         "the robot near a failing window. Check --pose_z_offset / the robot CSV.")

    # The closed-form volume and the sampler both assume the ball lies strictly inside the disk for
    # every yaw. ||h_xy|| + ||A_xy|| + rho is a yaw-free upper bound on its reach from the centre.
    reach = float((np.linalg.norm(geom.hc_k[:, k_of_traj, :2], axis=-1)
                   + np.linalg.norm(A[None, :, :2], axis=-1) + np.maximum(rho, 0.0)).max())
    if reach > args.pose_radius:
        raise SystemExit(
            f"--pose_radius {args.pose_radius:g} m is too small: a level-4 ball reaches "
            f"{reach:.3f} m from the disk centre, so it pokes out of the placement cylinder and "
            f"eq. (3) would over-count V(W). Use --pose_radius >= {np.ceil(reach * 10) / 10:g}.")

    pos = rho > 0.0
    geom.reach = reach
    geom.n_nonempty = int((vol > 0).sum())
    geom.rho_stats = (float(rho[pos].min()), float(np.median(rho[pos])), float(rho[pos].max()))
    geom.v_w = float(vol.mean())
    geom.p_w = geom.v_w / geom.v_cyl
    geom.vol = vol
    geom.vol_max = float(vol.max())
    geom.accept = geom.v_w / geom.vol_max
    geom.cdf = None
    if geom.accept < MIN_PROPOSAL_ACCEPT:
        cdf = np.cumsum(vol.ravel())
        geom.cdf = cdf / cdf[-1]
    return geom


def level4_ball(geom, mi, ji):
    """``(rho, z0)`` of the level-4 ball of trial ``(mi, ji)``; broadcasts over the two indices.

    ``rho <= 0`` marks a trajectory that has no true capsules at all (no contact possible): the
    sentinel ``-PAD_R`` in ``geom.R`` makes such a pair fall out of every downstream comparison
    rather than needing a separate mask.
    """
    k = geom.k_of_traj[ji]
    rho = geom.R[ji] + geom.hr_k[mi, k]
    z0 = geom.hc_k[mi, k, 2] - geom.A[ji, 2]
    return rho, z0


def yaw_rotate_xy(vec, yaw):
    """``(Rz(yaw) v)`` for a batch of 3-vectors ``vec`` [B,3] (and a batch of yaws) -> [B,3].

    Matches ``v @ pose_rt(pose)[0]`` from :func:`simulate_robot_shield.pose_rt`, which is how the
    shield applies a base pose, so the sampler and the kernel use one convention.
    """
    c, s = np.cos(yaw), np.sin(yaw)
    return np.stack([c * vec[:, 0] - s * vec[:, 1], s * vec[:, 0] + c * vec[:, 1], vec[:, 2]],
                    axis=1)


def draw_index(geom, rng, n):
    """Draw ``n`` i.i.d. ``(m, j)`` pairs with probability proportional to ``geom.vol``.

    Rejection against a uniform-over-the-grid proposal whenever the acceptance rate
    ``mean(vol)/max(vol)`` is workable (0.59 on H36M): one gather plus three uniforms per proposal,
    versus the ~20 cache-missing binary-search steps per sample that an inverse-CDF costs on a
    table of ``M x n_traj`` entries -- measured ~3.5x faster, and it is the run's bottleneck.
    Truncating the last over-drawn block keeps the kept pairs i.i.d., so the draw stays exact; the
    ``geom.cdf`` branch is the exact fallback for a pathological weight spread.
    """
    if geom.cdf is not None:
        flat = np.minimum(np.searchsorted(geom.cdf, rng.random(n), side="right"),
                          geom.cdf.size - 1)
        return flat // geom.nT, flat % geom.nT
    out_m = np.empty(n, np.int64)
    out_j = np.empty(n, np.int64)
    filled = 0
    while filled < n:
        # Over-draw by ~10 % so the loop almost always finishes in one pass.
        k = int((n - filled) / geom.accept * 1.1) + 64
        mi = rng.integers(0, geom.M, k)
        ji = rng.integers(0, geom.nT, k)
        keep = rng.random(k) * geom.vol_max <= geom.vol[mi, ji]
        mi, ji = mi[keep], ji[keep]
        take = min(mi.size, n - filled)
        out_m[filled: filled + take] = mi[:take]
        out_j[filled: filled + take] = ji[:take]
        filled += take
    return out_m, out_j


def draw_trials(geom, args, rng, n):
    """Draw ``n`` trials exactly from the uniform-on-``W`` conditional. Returns ``(m, j, poses)``.

    Because the reference measure is uniform over ``(m, j, yaw, t)``, conditioning on ``W`` gives
    ``p(m, j | W) = vol_mj / sum vol``, ``yaw`` still uniform and ``t`` uniform on the
    slab-truncated ball. Sampled in that order:

      1. ``(m, j)`` by :func:`draw_index`, proportional to the volume weights;
      2. ``yaw ~ U[-pi, pi]``;
      3. ``z`` from the band density ``propto rho^2 - (z - z0)^2`` on ``[lo, hi]`` by rejection
         against its own maximum over that interval, which accepts >90 % of proposals because the
         slab is thin next to ``rho``;
      4. ``(x, y)`` area-uniform in the disk of radius ``sqrt(rho^2 - (z - z0)^2)`` centred at
         ``h_xy - (Rz(yaw) A_j)_xy``.

    ``poses`` is the ``[n, 4]`` ``(yaw, tx, ty, tz)`` layout :func:`simulate_robot_shield.pose_rt`
    expects, so a drawn trial can be fed to the reference CPU path unchanged.
    """
    mi, ji = draw_index(geom, rng, n)
    rho, z0 = level4_ball(geom, mi, ji)
    z_off = geom.z_off
    lo = np.maximum(-z_off, z0 - rho)
    hi = np.minimum(z_off, z0 + rho)

    # Tight envelope: the band density peaks at the point of [lo, hi] closest to the ball centre.
    env = rho ** 2 - (np.clip(z0, lo, hi) - z0) ** 2
    tz = np.empty(n)
    todo = np.arange(n)
    while todo.size:
        prop = rng.uniform(lo[todo], hi[todo])
        keep = rng.random(todo.size) * env[todo] <= rho[todo] ** 2 - (prop - z0[todo]) ** 2
        tz[todo[keep]] = prop[keep]
        todo = todo[~keep]

    rad = np.sqrt(np.maximum(rho ** 2 - (tz - z0) ** 2, 0.0)) * np.sqrt(rng.random(n))
    ang = rng.uniform(0.0, 2.0 * np.pi, n)
    yaw = rng.uniform(-np.pi, np.pi, n)
    centre = geom.hc_k[mi, geom.k_of_traj[ji]] - yaw_rotate_xy(geom.A[ji], yaw)
    return mi, ji, np.stack([yaw, centre[:, 0] + rad * np.cos(ang),
                             centre[:, 1] + rad * np.sin(ang), tz], axis=1)


def in_level4_ball(geom, mi, ji, poses):
    """Bool mask [B]: does trial ``(m, j, pose)`` satisfy the level-4 test that defines ``W``?

    Used by ``--verify_gate`` to confirm empirically that ``D => W``; the drawn trials satisfy it
    by construction (asserted there too, since a sampler bug would silently shrink ``N_W``).
    """
    rho, _ = level4_ball(geom, mi, ji)
    rc = yaw_rotate_xy(geom.A[ji], poses[:, 0]) + poses[:, 1:4]
    return np.linalg.norm(geom.hc_k[mi, geom.k_of_traj[ji]] - rc, axis=1) <= rho


# --------------------------------------------------------------------------- the per-trial kernel


def build_capsule_tables(st, n_traj):
    """Per-trajectory capsule tables for the per-trial kernel, padded to a rectangular width.

    One entry per ``(interval row, link)`` of a trajectory -- at most ``11 * 7 = 77`` of them, so
    the tables are tiny and a trial's whole verification/contact check is a single dense
    ``[capsules, bodies]`` sweep. No grouping by horizon step is needed (unlike
    :mod:`examples.shield_gpu`, which shares one capsule table across all trajectories): each
    capsule carries its own step index and gathers its human spheres from it, which avoids the
    padding a step-bucketed table would waste at this width.

    Predicted side (verification): the trajectory's own link occupancy at
    ``s = last_step_below(hz, tp)``, grown by ``addr = (tp - hz[s]) * v_human`` to bridge to the
    exact interval time -- identical to ``run_pose_pure``'s predicted branch.
    True side (contact): the *future* interval-0 link occupancy at ``s = nearest_step(hz, tp)``,
    with a ``fast`` flag when the link speed exceeds ``V_ROBOT_ISO``.

    Padded slots get radius ``-PAD_R``, so they can never register a hit.
    """
    hz, vh, vr = st.horizon_times, st.v_human, st.v_robot
    pred, true = [], []
    for meta in st.traj_meta:
        rows, frow_rows = meta[0], meta[1]
        pb, tb = [], []
        for i, row in enumerate(rows):
            row = int(row)
            tp = st.tp_start[row]
            s_pred = last_step_below(hz, tp)
            addr = (tp - hz[s_pred]) * vh
            for a in range(N_LINKS):
                pb.append((row * N_LINKS + a, float(st.rr[row, a] + addr), s_pred))
            frow = int(frow_rows[i])
            if frow < 0:
                continue
            s_true = nearest_step(hz, tp)
            for a in range(N_LINKS):
                tb.append((frow * N_LINKS + a, float(st.rr[frow, a]), s_true,
                           st.spd[frow, a] > vr))
        pred.append(pb)
        true.append(tb)
    Cp = max(max((len(b) for b in pred), default=1), 1)
    Ct = max(max((len(b) for b in true), default=1), 1)

    tab = dict(pgid=np.zeros((n_traj, Cp), np.int64), peff=np.full((n_traj, Cp), -PAD_R),
               pstep=np.zeros((n_traj, Cp), np.int64),
               tgid=np.zeros((n_traj, Ct), np.int64), trad=np.full((n_traj, Ct), -PAD_R),
               tstep=np.zeros((n_traj, Ct), np.int64), tfast=np.zeros((n_traj, Ct), bool),
               Cp=Cp, Ct=Ct)
    for j, b in enumerate(pred):
        for i, (g, e, s) in enumerate(b):
            tab["pgid"][j, i], tab["peff"][j, i], tab["pstep"][j, i] = g, e, s
    for j, b in enumerate(true):
        for i, (g, r, s, f) in enumerate(b):
            tab["tgid"][j, i], tab["trad"][j, i] = g, r
            tab["tstep"][j, i], tab["tfast"][j, i] = s, f
    return tab


def _seg_hit(xp, a, b, hc, hr, cap_r):
    """Any-body hit per capsule: ``dist(segment(a,b), hc) <= cap_r + hr``. Returns [B,C] bool.

    ``a``, ``b`` are [B,C,3] transformed endpoints, ``hc``/``hr`` the [B,C,J,3]/[B,C,J] human
    spheres each capsule is tested against and ``cap_r`` the [B,C] capsule radius. Same arithmetic
    (and the same degenerate-capsule fallback) as
    :func:`simulate_robot_shield.segment_point_distances`, so the verdicts agree to float rounding.
    """
    ab = b - a                                                         # [B,C,3]
    denom = (ab * ab).sum(-1)                                          # [B,C]
    ap = hc - a[:, :, None, :]                                         # [B,C,J,3]
    t = xp.clip((ap * ab[:, :, None, :]).sum(-1) / xp.maximum(denom, 1e-18)[:, :, None], 0.0, 1.0)
    proj = a[:, :, None, :] + t[..., None] * ab[:, :, None, :]
    dist = xp.where(denom[:, :, None] <= 1e-18, xp.linalg.norm(ap, axis=-1),
                    xp.linalg.norm(hc - proj, axis=-1))                # [B,C,J]
    return (dist <= cap_r[:, :, None] + hr).any(axis=2)                # [B,C]


class TrialEvaluator:
    """Level-5 gate + exact shield verdict for a batch of independent ``(m, j, pose)`` trials.

    Unlike :class:`shield_gpu.GpuShieldEvaluator` -- which evaluates one placement against every
    (trajectory, window) pair -- every trial here carries its own window, trajectory *and* pose.
    That is what makes the trials i.i.d. (so eq. 5's binomial bound is exact) and what makes a
    batch of them one rectangular kernel.

    :meth:`gate5` is the per-link/per-body refinement of the level-4 sampling ball: a trial it
    rejects provably has no contact, so the caller scores it 0 without running :meth:`evaluate`.
    :meth:`evaluate` is the exact test -- the same segment-point arithmetic as ``run_pose_pure``,
    restricted to one (window, trajectory) pair -- returning ``(not_verified, contact, unsafe)``.

    The jitted kernels take fixed shapes, so every batch is padded to the configured width and the
    results are sliced back; a run therefore compiles each kernel exactly once.
    """

    def __init__(self, st, geom, args):
        self.backend = args.backend
        self.st = st
        self.geom = geom
        self.tab = build_capsule_tables(st, geom.nT)
        self.gate_batch = int(args.gate_batch)
        self.kernel_batch = int(args.kernel_batch)
        self.host = dict(self.tab, P1=st.p1.reshape(-1, 3), P2=st.p2.reshape(-1, 3),
                         pred_c=st.pred_c, pred_r=st.pred_r, true_c=st.true_c, true_r=st.true_r,
                         Lc=geom.Lc, Lr=geom.Lr, oa_c=st.oa_true_c, oa_r=st.oa_true_r,
                         k_of_traj=geom.k_of_traj)
        if self.backend != "gpu":
            self.dev = self.host
            return
        import jax
        import jax.numpy as jnp
        self.jnp = jnp
        self.f = jnp.float64 if args.gpu_dtype == "float64" else jnp.float32
        flt = ("P1", "P2", "pred_c", "pred_r", "true_c", "true_r", "peff", "trad", "Lc", "Lr",
               "oa_c", "oa_r")
        self.dev = {k: jnp.asarray(v, self.f if k in flt else None)
                    for k, v in self.host.items() if not isinstance(v, int)}
        self._gate5 = jax.jit(lambda *a: self._gate5_xp(jnp, self.dev, *a))
        self._eval = jax.jit(lambda *a: self._verdict(jnp, self.dev, *a))

    # ---- batching ---------------------------------------------------------------------------

    @staticmethod
    def _rot4(poses):
        """Pack ``(rot_t, t)`` into one [B,4,3] block: rows 0-2 are ``rot_t``, row 3 is ``t``.

        ``rot_t = Rz(yaw).T`` exactly as :func:`simulate_robot_shield.pose_rt` builds it, so a
        point transforms as ``p @ rot_t + t``.
        """
        poses = np.asarray(poses, dtype=np.float64)
        c, s = np.cos(poses[:, 0]), np.sin(poses[:, 0])
        out = np.zeros((poses.shape[0], 4, 3))
        out[:, 0, 0] = c
        out[:, 0, 1] = s
        out[:, 1, 0] = -s
        out[:, 1, 1] = c
        out[:, 2, 2] = 1.0
        out[:, 3, :] = poses[:, 1:4]
        return out

    def _batched(self, fn, width, mi, ji, poses, n_out):
        """Run ``fn(mi, ji, rot4)`` over fixed-width slices, padding the tail. Returns ``n_out``
        bool arrays of length ``len(mi)``."""
        n = mi.size
        out = [np.zeros(n, bool) for _ in range(n_out)]
        for b0 in range(0, n, width):
            k = min(width, n - b0)
            sl = slice(b0, b0 + k)
            arrs = (mi[sl], ji[sl], self._rot4(poses[sl]))
            if k < width:                                   # pad to the compiled shape
                arrs = tuple(np.concatenate([a, np.repeat(a[:1], width - k, axis=0)])
                             for a in arrs)
            if self.backend == "gpu":
                res = fn(self.jnp.asarray(arrs[0]), self.jnp.asarray(arrs[1]),
                         self.jnp.asarray(arrs[2], self.f))
            else:
                res = fn(*arrs)
            res = res if isinstance(res, tuple) else (res,)
            for o, r in zip(out, res):
                o[sl] = np.asarray(r)[:k]
        return out

    # ---- level-5 gate -----------------------------------------------------------------------

    @staticmethod
    def _gate5_xp(xp, arr, mi, ji, rot):
        """Level 5: any (robot link sphere, human body sphere) pair of this trial within reach."""
        lc = xp.einsum("blk,bkm->blm", arr["Lc"][ji], rot[:, :3, :3]) + rot[:, 3, :][:, None, :]
        k = arr["k_of_traj"][ji]
        hb_c, hb_r = arr["oa_c"][mi, k], arr["oa_r"][mi, k]            # [B,J,3], [B,J]
        dist = xp.linalg.norm(lc[:, :, None, :] - hb_c[:, None, :, :], axis=-1)    # [B,L,J]
        return (dist <= arr["Lr"][ji][:, :, None] + hb_r[:, None, :]).any(axis=(1, 2))

    def gate5(self, mi, ji, poses):
        """Bool mask [B]: can trajectory ``j``'s true capsules touch window ``m``'s occupancy?"""
        fn = self._gate5 if self.backend == "gpu" else (
            lambda *a: self._gate5_xp(np, self.host, *a))
        return self._batched(fn, self.gate_batch, mi, ji, poses, 1)[0]

    # ---- exact per-trial verdict -------------------------------------------------------------

    @staticmethod
    def _verdict(xp, arr, mi, ji, rot):
        """``(not_verified, contact, unsafe)`` [B] for one padded batch of trials."""
        rt, tv = rot[:, :3, :3], rot[:, 3, :][:, None, :]

        def endpoints(gid):
            return (xp.einsum("bck,bkl->bcl", arr["P1"][gid], rt) + tv,
                    xp.einsum("bck,bkl->bcl", arr["P2"][gid], rt) + tv)

        pg, ps = arr["pgid"][ji], arr["pstep"][ji]                     # [B,Cp]
        a, b = endpoints(pg)
        hit = _seg_hit(xp, a, b, arr["pred_c"][mi[:, None], ps], arr["pred_r"][mi[:, None], ps],
                       arr["peff"][ji])
        not_verified = hit.any(axis=1)

        tg, ts = arr["tgid"][ji], arr["tstep"][ji]                     # [B,Ct]
        a, b = endpoints(tg)
        hit = _seg_hit(xp, a, b, arr["true_c"][mi[:, None], ts], arr["true_r"][mi[:, None], ts],
                       arr["trad"][ji])
        return not_verified, hit.any(axis=1), (hit & arr["tfast"][ji]).any(axis=1)

    def evaluate(self, mi, ji, poses):
        """Exact ``(not_verified, contact, unsafe)`` bool arrays for a batch of trials."""
        fn = self._eval if self.backend == "gpu" else (
            lambda *a: self._verdict(np, self.host, *a))
        return self._batched(fn, self.kernel_batch, mi, ji, poses, 3)


# --------------------------------------------------------------------------- step 3: the trials


def run_trials(ev, geom, args, rng, n_trials, desc, uniform=False, collect=0):
    """Draw and score ``n_trials`` trials, streaming in blocks so memory stays bounded.

    Trials rejected by the level-5 gate are scored 0 without touching the exact kernel -- sound
    because the gate is a necessary condition for contact -- and still counted in ``n``, which is
    the ``N_W`` of eq. (5).

    ``uniform=True`` draws ``(m, j)`` and the placement from the *whole* trial space instead of
    from ``W`` and scores every trial exactly (the ``--verify_gate`` self-test). It then counts how
    many **contacts** -- not just the far rarer dangerous events -- fall outside the level-4 ball
    and outside the level-5 gate. Both must be 0: the two gates are necessary conditions for
    contact, which is what makes ``D => W`` hold, and contacts are ~1e4x more common than dangerous
    failures, so this is where the implication can actually be exercised.

    ``collect`` keeps up to that many dangerous trials as ``(m, j, yaw, tx, ty, tz)`` rows.
    """
    c = dict(n=0, n_gate5=0, n_contact=0, n_verified=0, n_dangerous=0, n_dangerous_fast=0,
             n_contact_outside_w4=0, n_contact_outside_w5=0, n_dang_outside_w=0,
             trials_per_m=np.zeros(geom.M, np.int64), dangerous_per_m=np.zeros(geom.M, np.int64))
    kept, n_kept = [], 0
    bar = tqdm(total=n_trials, desc=desc, unit="trial", dynamic_ncols=True, mininterval=2.0,
               unit_scale=True)
    done = 0
    while done < n_trials:
        n = int(min(args.trial_block, n_trials - done))
        if uniform:
            mi = rng.integers(0, geom.M, n)
            ji = rng.integers(0, geom.nT, n)
            poses = sample_robot_poses(n, args.pose_radius, args.pose_z_offset, rng)
        else:
            mi, ji, poses = draw_trials(geom, args, rng, n)
        np.add.at(c["trials_per_m"], mi, 1)
        c["n"] += n
        gate = None if (uniform or not args.gate5) else ev.gate5(mi, ji, poses)
        sel = np.arange(n) if gate is None else np.flatnonzero(gate)
        c["n_gate5"] += int(sel.size)
        if sel.size:
            nv, ct, us = ev.evaluate(mi[sel], ji[sel], poses[sel])
            dang = ~nv & ct
            c["n_verified"] += int((~nv).sum())
            c["n_contact"] += int(ct.sum())
            c["n_dangerous"] += int(dang.sum())
            c["n_dangerous_fast"] += int((~nv & us).sum())
            if uniform and ct.any():
                # The gates are necessary conditions for CONTACT, so every contact must satisfy
                # them; a violation here breaks D => W and with it the V(W)/V(F) factor.
                touch = sel[ct]
                out4 = ~in_level4_ball(geom, mi[touch], ji[touch], poses[touch])
                out5 = ~ev.gate5(mi[touch], ji[touch], poses[touch])
                c["n_contact_outside_w4"] += int(out4.sum())
                c["n_contact_outside_w5"] += int(out5.sum())
                c["n_dang_outside_w"] += int(((out4 | out5) & dang[ct]).sum())
            if dang.any():
                hit = sel[dang]
                np.add.at(c["dangerous_per_m"], mi[hit], 1)
                if n_kept < collect:
                    take = hit[: collect - n_kept]
                    kept.append(np.column_stack([mi[take], ji[take], poses[take]]))
                    n_kept += take.size
        done += n
        bar.update(n)
        bar.set_postfix_str(f"gate5={c['n_gate5']:,} contact={c['n_contact']:,} "
                            f"dangerous={c['n_dangerous']:,}", refresh=False)
    bar.close()
    c["kept"] = np.concatenate(kept) if kept else np.zeros((0, 6))
    return c


# --------------------------------------------------------------------------- self-tests


def check_parity(ev, geom, args, rng, n_poses):
    """Assert the per-trial kernel reproduces ``run_pose_pure`` for whole placements.

    For each of ``n_poses`` placements (drawn from ``W``, so they are guaranteed to be doing
    something) the reference path runs once over all ``M x n_traj`` (window, trajectory) pairs and
    the kernel runs on exactly those pairs.

    Two things are compared, with deliberately different strictness:

      * the *set* of verified-but-contact pairs -- EXACTLY, no tolerance. That is the event
        ``k_D`` counts, so any disagreement there invalidates the bound.
      * the aggregate verified / contact / unsafe counts -- to within
        ``PARITY_COUNT_TOL`` relative (floor 2 pairs). At ``--gpu_dtype float32`` a pair sitting
        within ~1e-6 m of exact tangency can legitimately land on either side of the ``<=``: the
        two paths do the same arithmetic in a different order. Measured on the H36M test split,
        float32 flips 2 of 75,456 pairs (2.6e-5) while float64 reproduces all of them, so the
        tolerance is sized well below anything a real logic error could hide in. Run
        ``--gpu_dtype float64`` for a zero-tolerance check.
    """
    st = ev.st
    st.failure_triples = True
    M, nT = geom.M, geom.nT
    grid_m = np.repeat(np.arange(M), nT)
    grid_j = np.tile(np.arange(nT), M)
    _, _, poses = draw_trials(geom, args, rng, n_poses)
    ok = True
    for p in range(n_poses):
        pose = poses[p]
        t0 = time.monotonic()
        ref = run_pose_pure(st, pose)
        t_ref = time.monotonic() - t0
        t0 = time.monotonic()
        nv, ct, us = ev.evaluate(grid_m, grid_j, np.repeat(pose[None], M * nT, axis=0))
        t_ker = time.monotonic() - t0
        got = dict(n_verified=int((~nv).sum()), n_contact=int(ct.sum()), n_unsafe=int(us.sum()),
                   n_verified_contact=int((~nv & ct).sum()),
                   n_verified_unsafe=int((~nv & us).sum()))
        dev = {k: (ref[k], got[k]) for k in got if ref[k] != got[k]}
        over = {k: v for k, v in dev.items()
                if abs(v[0] - v[1]) > max(2, PARITY_COUNT_TOL * max(v[0], 1))}
        ref_pairs = {(int(tj), int(m)) for tj, m, _ in ref["failures"]}
        got_pairs = {(int(grid_j[i]), int(grid_m[i])) for i in np.flatnonzero(~nv & ct)}
        set_ok = ref_pairs == got_pairs
        if not dev:
            verdict = "counts MATCH exactly"
        elif not over:
            worst = max(abs(a - b) / max(a, 1) for a, b in dev.values())
            verdict = (f"counts match to {worst:.1e} relative ({dev}) -- within the "
                       f"{args.gpu_dtype} tangency tolerance")
        else:
            verdict = f"COUNT MISMATCH beyond tolerance {over}"
        print(f"  pose {p}: reference {t_ref:.2f} s / kernel {t_ker:.2f} s over {M * nT:,} pairs; "
              + verdict
              + f"; verified-but-contact pairs ref={len(ref_pairs)} kernel={len(got_pairs)}"
              + ("" if set_ok else " <- SET MISMATCH"))
        ok &= (not over) and set_ok
    st.failure_triples = False
    print("PARITY OK: the per-trial kernel agrees with run_pose_pure." if ok else
          "!! PARITY FAILED -- the per-trial kernel and run_pose_pure disagree. Do not report.")
    return ok


def check_lemma(args, results, flag, rng, n):
    """Empirical test of ``D => F`` (eq. 1): the same trials on *non-failure* windows.

    The witness region of the non-failure windows is where a lemma violation would have to show up,
    so the trials are drawn from *their* ``W`` rather than uniformly -- far more sensitive than a
    uniform replay at the same count. Expectation: zero verified-AND-contact.
    """
    non_fail = np.flatnonzero(~flag)
    take = min(int(args.lemma_windows), non_fail.size)
    sel = np.sort(np.random.default_rng(args.seed + 99).choice(non_fail, take, replace=False))
    ctx_l = build_shield_state(args, results=results, subset=sel)
    geom_l = build_trial_geometry(ctx_l, args)
    ev_l = TrialEvaluator(ctx_l.st, geom_l, args)
    return run_trials(ev_l, geom_l, args, rng, n, "verify_lemma"), sel.size, geom_l


# --------------------------------------------------------------------------- main


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    # ---- inputs (names/defaults mirror simulate_shield_failure_risk) ----
    parser.add_argument("--results_file", type=str,
                        default="results/motion_prediction/motion_prediction_results_test.cloudpickle")
    parser.add_argument("--conformal_calibrator", type=str,
                        default="models/motion_prediction/conformal_calibration/conformal_calibrator.npz",
                        help="Conditional-conformal calibrator .npz forming the predicted human "
                             "set radius. '' / 'none' (or a missing file) falls back to the affine "
                             "calibration; --human_set sara ignores it entirely.")
    parser.add_argument("--robot_csv", type=str,
                        default="datasets/robot_reachable_sets/ablation_conformal_prediction_sets_panda_4ms.csv")
    parser.add_argument("--config", type=str, default="h36m", choices=["h36m", "rgbd_yolo"])
    parser.add_argument("--fps", type=float, default=25.0,
                        help="Window spacing of the recorded data, hence the human horizon dt.")
    parser.add_argument("--human_set", type=str, default="conformal", choices=["conformal", "sara"])
    parser.add_argument("--mask_ood", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ood_threshold", type=float, default=None,
                        help="Override OOD_THRESHOLD (head-specific; see simulate_robot_shield).")
    parser.add_argument("--mask_too_fast", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--calibrate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--robot_origin", type=str, default="0,0,0")
    parser.add_argument("--overapprox", action=argparse.BooleanOptionalAction, default=True,
                        help=argparse.SUPPRESS)      # the witness region needs the hierarchy
    parser.add_argument("--motion_group_size", type=int, default=50)
    parser.add_argument("--max_human_samples", type=int, default=None)
    parser.add_argument("--max_robot_timesteps", type=int, default=None)
    parser.add_argument("--robot_stride", type=int, default=1,
                        help="Use every Nth monitored trajectory. 1 (the default here, unlike the "
                             "cross-product script) costs nothing: a trial uses ONE trajectory "
                             "phase, so the full 4 ms grid is the honest phase distribution.")
    parser.add_argument("--t_cycle", type=float, default=None,
                        help="Safety-function cycle time (s); N_h = 3600/t_cycle verification "
                             "cycles per operating hour. Default: the robot planning grid.")
    # ---- the placement distribution (a modelling choice; V(F) is built from it) ----
    parser.add_argument("--pose_radius", type=float, default=10.0,
                        help="Radius (m) of the disk of base (x, y) positions. V(F) scales with "
                             "its square, so this choice moves the bound: it must describe the "
                             "expected HRC cell, not be tuned.")
    parser.add_argument("--pose_z_offset", type=float, default=0.2)
    # ---- compute ----
    parser.add_argument("--backend", type=str, default="gpu", choices=["cpu", "gpu"])
    parser.add_argument("--gpu_dtype", type=str, default="float32", choices=["float32", "float64"])
    parser.add_argument("--num_trials", type=int, default=10_000_000,
                        help="N_W: i.i.d. (window, trajectory phase, placement) triples drawn "
                             "from the witness region W. NOT the uniform-placement count N_D of "
                             "eq. (5)'s reference measure -- that is N_W / P(W|F), which the run "
                             "reports.")
    parser.add_argument("--trial_block", type=int, default=4_000_000,
                        help="Trials drawn per streaming block (bounds host memory; ~130 MB of "
                             "float64 placements at the default).")
    parser.add_argument("--gate_batch", type=int, default=131_072,
                        help="Trials per level-5 gate kernel launch.")
    parser.add_argument("--kernel_batch", type=int, default=16_384,
                        help="Trials per exact capsule kernel launch. Its [B, capsules, bodies, 3] "
                             "temporaries scale with this; measured flat from 4k to 64k, so the "
                             "run is sampler-bound rather than kernel-bound.")
    parser.add_argument("--gate5", action=argparse.BooleanOptionalAction, default=True,
                        help="Resolve trials failing the level-5 necessary condition as 0 without "
                             "the exact kernel. --no-gate5 is an equivalence/timing ablation: it "
                             "must produce the same k_D.")
    parser.add_argument("--seed", type=int, default=0)
    # ---- statistics ----
    parser.add_argument("--epsilon_d", type=float, default=1e-5,
                        help="Total error budget eps_D: the reported PFH_D holds at joint "
                             "confidence 1 - epsilon_d. Split symmetrically over the two factors, "
                             "eps_F = eps_(D|F) = 1 - sqrt(1 - epsilon_d). Default 1e-5 = "
                             "99.999%% joint confidence.")
    parser.add_argument("--l_corr", type=float, default=None,
                        help="Override the measured integrated autocorrelation time of the failure "
                             "indicator, which deflates the P(F) count to n_eff = N_F / "
                             "(2 L_corr). For sensitivity checks; the run prints the measured "
                             "value either way.")
    parser.add_argument("--failure_stride", type=int, default=1,
                        help="Diagnostic only: print P_hat(F) over every Nth window beside the "
                             "all-window estimate that actually enters the bound. Window "
                             "dependence is handled by L_corr, not by striding.")
    parser.add_argument("--max_failures_eval", type=int, default=None,
                        help="Cap the failure set the placement integral uses (smoke runs, or to "
                             "bound the [M, n_traj] weight table). P(F) still uses every window.")
    # ---- self-tests ----
    parser.add_argument("--parity", type=int, default=0,
                        help="Check N whole placements of the per-trial kernel against "
                             "run_pose_pure (slow: the reference sweeps M x n_traj pairs).")
    parser.add_argument("--verify_gate", type=int, default=0,
                        help="Draw N trials from the FULL trial space to test D => W empirically.")
    parser.add_argument("--verify_lemma", type=int, default=0,
                        help="Draw N trials on non-failure windows to test D => F empirically.")
    parser.add_argument("--lemma_windows", type=int, default=2000,
                        help="Non-failure windows the --verify_lemma pass replays.")
    # ---- artefacts ----
    parser.add_argument("--save_trials", type=str, default=None,
                        help="Path to a .npz: the per-(window, trajectory) volume weights, the "
                             "per-window trial/dangerous counts and up to --collect_dangerous "
                             "dangerous trials.")
    parser.add_argument("--collect_dangerous", type=int, default=10_000)
    parser.add_argument("--results_csv", type=str, default=None,
                        help="Append this run's summary as one row (header-migrating).")
    # build_shield_state also consults the --save_failures geometry dump, which this script does
    # not expose (a trial's geometry is recoverable from the (m, j, pose) row alone).
    parser.set_defaults(save_failures=None, max_failures=0)
    return parser


def main():
    args = build_parser().parse_args()

    # ------------------------------------------------------------------ step 1: the failure set F
    try:
        ctx, results, fail_stats = build_failure_state(args)
    except SystemExit as exc:
        # build_shield_state's M == 0 guard fires before the k_F == 0 check below can, and its
        # message blames the OOD / too-fast filters. With a low-miss-rate calibrator (the
        # alpha_max ablation has k_F = 0 on the H36M validation split and k_F = 1 on test) the
        # real cause is an empty FAILURE set, which is a different problem with a different fix.
        if "No eligible human samples" in str(exc):
            raise SystemExit(
                "Empty human sub-population. Either --mask_ood/--mask_too_fast removed every "
                "window, or -- more likely -- this configuration has NO prediction failures on "
                "this split (k_F = 0), so the factorisation has no failure to replay: P(D) is "
                "then bounded only through P(F), which this estimator does not model. Check the "
                "'restricted to a sub-population of N human sample(s)' line above; if N = 0, use "
                "a split with more failures (test rather than validation) or a calibrator with a "
                "lower target coverage.") from exc
        raise
    flag = fail_stats["flag"]
    escape = fail_stats["escape"]
    N_F = int(flag.size)
    k_F = int(flag.sum())
    if k_F == 0:
        raise SystemExit("No prediction failures in this results file -- nothing to price. "
                         "P(F) is then bounded only by its own binomial upper limit, which the "
                         "factorisation cannot turn into a P(D) without a failure to replay.")
    if args.verify_lemma <= 0:
        results = None          # only --verify_lemma rebuilds a state; else free the ~GB dict
    eps_f, eps_df = split_confidence(args.epsilon_d)
    # Point estimate over EVERY window -- the estimator eq. (1) is derived for. --failure_stride is
    # only a diagnostic sub-sample printed beside it.
    p_f_hat = k_F / N_F
    stride = max(1, args.failure_stride)
    sub = flag[::stride]
    p_f_hat_sub = float(sub.mean()) if sub.size else float("nan")
    # Dependence enters the INTERVAL, not the estimate (eq. 6): deflate the window count by the
    # integrated autocorrelation time and by the loader's two phase offsets.
    L_corr = float(args.l_corr) if args.l_corr is not None else integrated_autocorr_time(flag)
    n_eff = N_F / (PHASE_OFFSETS * L_corr)
    k_eff = p_f_hat * n_eff
    p_f_lo, p_f_hi = clopper_pearson(k_eff, n_eff)            # diagnostic 95% two-sided
    p_f_up = clopper_pearson_upper(k_eff, n_eff, eps_f)       # the bound that enters PFH_D
    episodes = run_lengths(flag)
    L_meas = float(episodes.mean()) if episodes.size else 1.0
    acf = autocorrelation(flag)
    t_cycle = args.t_cycle if args.t_cycle is not None else ctx.t_cycle
    N_h = 3600.0 / t_cycle

    print("\n=================== Step 1: the prediction-failure set F ===================")
    print(f"Predicted human occupancy model : {args.human_set}"
          + (" (ISO 13855 constant-velocity set)" if args.human_set == "sara"
             else (" (conditional-conformal set)" if ctx.calibrator is not None
                   else " (affine/raw covariance set)")))
    print(f"Prediction windows |Z_test| = N_F  : {N_F:,}")
    print(f"Failures k_F (truth escapes the predicted set at some step, joint): {k_F:,}")
    esc_f = escape[flag]
    print(f"  escape size (m): median {np.median(esc_f):.4f}  mean {esc_f.mean():.4f}  "
          f"p95 {np.quantile(esc_f, 0.95):.4f}  max {esc_f.max():.4f}")
    print(f"P_hat(F) = k_F / N_F               : {p_f_hat:.6e}   (over EVERY window)")
    if stride > 1:
        print(f"  diagnostic sub-sample, every {stride}th window -> {p_f_hat_sub:.6e} "
              f"({p_f_hat_sub / p_f_hat if p_f_hat else float('nan'):.2f}x; a large gap means "
              f"--failure_stride landed on an unlucky phase)")
    print("Window dependence -> the INTERVAL, not the estimate (eq. 6):")
    print(f"  integrated autocorrelation time L_corr = {L_corr:.2f} windows "
          f"(= {1000.0 * L_corr / args.fps:.0f} ms)"
          + ("  <- --l_corr override" if args.l_corr is not None else "  (Sokal windowing)"))
    print("  indicator autocorrelation: "
          + ", ".join(f"lag {lag}: {v:+.3f}" for lag, v in acf.items()))
    print(f"  failure episodes: {episodes.size:,} runs, mean length {L_meas:.2f} windows -- the "
          f"evidence behind L_corr, NOT used as a rate")
    print(f"  n_eff = N_F / ({PHASE_OFFSETS} L_corr) = {n_eff:,.0f}, k_eff = {k_eff:,.1f}   "
          f"(the {PHASE_OFFSETS} = the loader's two 25 fps phase offsets)")
    print(f"P(F) <= B^-1(1-eps_F; k_eff+1, n_eff-k_eff) = {p_f_up:.6e}   "
          f"(1-eps_F = {1.0 - eps_f:.8g})")
    print(f"  diagnostic 95% two-sided on n_eff: [{p_f_lo:.6e}, {p_f_hi:.6e}]")
    print(f"t_cycle = {t_cycle:g} s -> N_h = 3600/t_cycle = {N_h:,.0f} verification cycles per hour")
    print("  (one trial per SHIELD planning cycle, NOT per prediction: a prediction is reused for")
    print("   K_P cycles and every one of them inherits its failure status)")
    print("=" * 76)

    # ------------------------------------------------------------------ step 2: the region W
    geom = build_trial_geometry(ctx, args)
    rho_lo, rho_med, rho_hi = geom.rho_stats
    print("\n================== Step 2: the witness region W (volume) ==================")
    print(f"Trial space: {geom.M:,} failing window(s) x {geom.nT:,} trajectory phase(s) x "
          f"(yaw, t) in the cylinder")
    print(f"  placement cylinder: xy-disk r = {args.pose_radius:g} m, z in "
          f"+/-{args.pose_z_offset:g} m -> V(F) = {geom.v_cyl:.4f} m^3")
    print(f"  level-4 ball radius rho_mj: min {rho_lo:.3f} m, median {rho_med:.3f} m, "
          f"max {rho_hi:.3f} m")
    print(f"  max reach from the disk centre {geom.reach:.3f} m << {args.pose_radius:g} m, so the "
          f"balls never touch the disk wall and eq. (3) is exact")
    print(f"  (m, j) pairs with a non-empty ball: {geom.n_nonempty:,}/{geom.M * geom.nT:,}")
    print(f"mean_mj vol(B_mj n Cyl)  = {geom.v_w:.6f} m^3")
    print(f"P(W | F) = V(W) / V(F)   = {geom.p_w:.6e}   <- closed form, no Monte-Carlo error")
    print(f"  one trial drawn from W is worth {1.0 / geom.p_w:,.0f} uniform placements")
    print("  (m, j) draw: "
          + (f"inverse-CDF (proposal acceptance {geom.accept:.3f} < {MIN_PROPOSAL_ACCEPT:g})"
             if geom.cdf is not None else
             f"rejection on a uniform proposal, acceptance {geom.accept:.3f}"))
    print("=" * 76)

    # ------------------------------------------------------------------ step 3: the trials
    ev = TrialEvaluator(ctx.st, geom, args)
    print(f"\nCapsule tables: {ev.tab['Cp']} predicted / {ev.tab['Ct']} true capsules per "
          f"trajectory, {ctx.J} human body spheres per horizon step")
    rng = np.random.default_rng(args.seed + 1234)
    if args.parity > 0:
        print(f"\n--- Parity: {args.parity} placement(s), per-trial kernel vs run_pose_pure ---")
        if not check_parity(ev, geom, args, rng, args.parity):
            raise SystemExit("Parity check failed.")

    print(f"\nDrawing {args.num_trials:,} trial(s) from W on the {args.backend} backend ...")
    t0 = time.monotonic()
    c = run_trials(ev, geom, args, rng, args.num_trials, "trials",
                   collect=args.collect_dangerous if args.save_trials else 0)
    dt = time.monotonic() - t0
    N_W, k_D = c["n"], c["n_dangerous"]
    p_dgw_hat = k_D / N_W
    p_dgw_up = clopper_pearson_upper(k_D, N_W, eps_df)
    p_d_given_f_hat = geom.p_w * p_dgw_hat
    p_d_given_f_up = geom.p_w * p_dgw_up

    print("\n============ Step 3: P(D | F, W) on i.i.d. trials drawn from W ============")
    print(f"Trials N_W (drawn from W)          : {N_W:,}   ({dt:.1f} s, "
          f"{N_W / max(dt, 1e-9):,.0f} trial/s)")
    print(f"Resolved by the level-5 gate       : {N_W - c['n_gate5']:,} "
          f"({100.0 * (1.0 - c['n_gate5'] / N_W):.2f}%) -- provably contact-free, scored 0 "
          f"without the exact kernel")
    print(f"Exact kernel evaluations           : {c['n_gate5']:,}")
    print(f"  verified                         : {c['n_verified']:,}")
    print(f"  in contact                       : {c['n_contact']:,}")
    print(f"Dangerous k_D (verified AND contact) : {k_D:,}   in "
          f"{int((c['dangerous_per_m'] > 0).sum()):,} of {geom.M:,} failing windows")
    print(f"  of which unsafe (link speed > {ctx.v_robot} m/s): {c['n_dangerous_fast']:,} "
          f"(diagnostic only; PFH_D uses contact at any speed)")
    print(f"P_hat(D | F, W) = k_D / N_W        : {p_dgw_hat:.6e}"
          + ("   <- 0: below resolution, the bound below is the usable number" if k_D == 0 else ""))
    print(f"P(D | F, W) <= B^-1(1-eps_{{D|F}}; k_D+1, N_W-k_D) = {p_dgw_up:.6e}   "
          f"(1-eps_{{D|F}} = {1.0 - eps_df:.8g})")
    print(f"P(D | F) = P(W|F) P(D|F,W) <= {geom.p_w:.4e} x {p_dgw_up:.4e} = "
          f"{p_d_given_f_up:.6e}")
    print(f"  equivalent uniform placements N_D = N_W / P(W|F) = {N_W / geom.p_w:,.0f}   "
          f"(what a uniform sampler would have to draw for the same bound)")
    print("=" * 76)

    # ------------------------------------------------------------------ step 4: compose
    pfh_hat = N_h * p_f_hat * p_d_given_f_hat
    pfh_up = N_h * p_f_up * p_d_given_f_up
    print("\n========= Step 4: PFH_D = N_h P(F) P(W|F) P(D|F,W)   (eq. 5) =========")
    print(f"{'factor':>22} | {'point estimate':>16} | {'upper bound':>16}")
    print("-" * 62)
    print(f"{'N_h [1/h]':>22} | {N_h:>16,.0f} | {N_h:>16,.0f}")
    print(f"{'P(F)':>22} | {p_f_hat:>16.6e} | {p_f_up:>16.6e}")
    print(f"{'P(W|F) = V(W)/V(F)':>22} | {geom.p_w:>16.6e} | {geom.p_w:>16.6e}")
    print(f"{'P(D|F,W)':>22} | {p_dgw_hat:>16.6e} | {p_dgw_up:>16.6e}")
    print("-" * 62)
    print(f"{'PFH_D [1/h]':>22} | {pfh_hat:>16.6e} | {pfh_up:>16.6e}")
    print(f"\n>>> PFH_D <= {pfh_up:.6e} 1/h   at joint confidence "
          f"{100.0 * (1.0 - args.epsilon_d):.6g}%   -> PL {pl_label(pfh_up)}")
    print("CAVEAT: P(W|F) is exact for the placement distribution CHOSEN here (area-uniform in a")
    print(f"  {args.pose_radius:g} m disk, random yaw, random trajectory phase). It scales as")
    print("  1/pose_radius^2, so the bound is a statement about THAT cell. The distribution is a")
    print("  modelling input that must mirror the expected HRC situation, not a free parameter.")
    print("=" * 76)

    # ------------------------------------------------------------------ self-tests
    c_gate, gate_est = None, float("nan")
    if args.verify_gate > 0:
        print(f"\n--- Verify D => W: {args.verify_gate:,} trial(s) over the FULL trial space ---")
        print("W is defined by the level-4 ball, and the level-5 gate refines it; both are")
        print("necessary conditions for CONTACT, which is what makes D => contact => W. So every")
        print("contact drawn uniformly must satisfy them -- a far more sensitive test than waiting")
        print("for the rare dangerous event. Both counts below must be zero, or eq. (2) is wrong.")
        c_gate = run_trials(ev, geom, args, rng, args.verify_gate, "verify_gate (uniform)",
                            uniform=True)
        print(f"  {c_gate['n_contact']:,} contact / {c_gate['n_dangerous']:,} dangerous in "
              f"{c_gate['n']:,} uniform trials")
        print(f"  contacts OUTSIDE the level-4 ball (= W) : {c_gate['n_contact_outside_w4']:,}")
        print(f"  contacts OUTSIDE the level-5 gate       : {c_gate['n_contact_outside_w5']:,}")
        if c_gate["n_contact_outside_w4"] or c_gate["n_contact_outside_w5"]:
            print("!" * 100)
            print(f"!! D => W VIOLATED: {c_gate['n_dang_outside_w']:,} of them were dangerous. The "
                  f"witness region does not bracket contact, so the V(W)/V(F) factor omits part "
                  f"of the event. Do not report PFH_D.")
            print("!" * 100)
        elif c_gate["n_contact"] == 0:
            print("  -> not exercised: no contact at all at this resolution; raise --verify_gate.")
        else:
            print(f"  D => W HOLDS: all {c_gate['n_contact']:,} uniformly drawn contacts lie "
                  f"inside W.")
        if c_gate["n_dangerous"]:
            gate_est = c_gate["n_dangerous"] / c_gate["n"]
            print(f"  independent P_hat(D|F) over the full space = {gate_est:.6e}  vs the "
                  f"volume-weighted {p_d_given_f_hat:.6e}")

    lemma = None
    if args.verify_lemma > 0:
        print(f"\n--- Verify D => F: {args.verify_lemma:,} trial(s) on NON-failure windows ---")
        print("Expectation: ZERO verified-AND-contact. That implication is what makes the")
        print("factorisation exact; if it fires, the escape test misses something the shield can")
        print("be fooled by and PFH_D above is NOT an upper bound.")
        lemma, n_lw, _ = check_lemma(args, results, flag, rng, args.verify_lemma)
        if lemma["n_dangerous"] == 0:
            print(f"LEMMA HOLDS: 0 verified-AND-contact in {lemma['n']:,} trials over {n_lw:,} "
                  f"non-failure windows ({lemma['n_contact']:,} contacts, every one of them "
                  f"correctly NOT verified).")
        else:
            print("!" * 100)
            print(f"!! LEMMA VIOLATED: {lemma['n_dangerous']:,} verified-AND-contact trial(s) on "
                  f"NON-failure windows -> the escape test does not capture everything the shield "
                  f"can be fooled by. Do not report PFH_D.")
            print("!" * 100)

    # ------------------------------------------------------------------ artefacts
    if args.save_trials:
        out = (args.save_trials if os.path.isabs(args.save_trials)
               else os.path.join(root_dir, args.save_trials))
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        ev_idx = fail_stats["evaluated"]
        np.savez(out, window_idx=ev_idx, results_row=ctx.keep_idx, escape_m=escape[ev_idx],
                 escape_step=fail_stats["step"][ev_idx], escape_joint=fail_stats["joint"][ev_idx],
                 vol_mj=(geom.vol.astype(np.float32) if geom.vol.size <= 5_000_000
                         else np.zeros(0, np.float32)),
                 v_cyl=np.float64(geom.v_cyl), p_w=np.float64(geom.p_w),
                 k_of_traj=geom.k_of_traj, traj_sphere_c=geom.A, traj_sphere_r=geom.R,
                 trials_per_m=c["trials_per_m"], dangerous_per_m=c["dangerous_per_m"],
                 dangerous_trials=c["kept"], n_trials=np.int64(N_W), n_dangerous=np.int64(k_D))
        print(f"\nSaved trial detail to {out} (np.load(path))")

    if args.results_csv:
        used_cal = None if args.human_set == "sara" else ctx.calibrator
        if args.human_set == "sara":
            set_kind = "sara"
        elif used_cal is None:
            set_kind = "affine" if args.calibrate else "raw"
        else:
            # "max_conformal" / "uncalibrated" are what conformal_results_common.
            # shield_method_key keys the ablation rows on -- keep the three writers in sync
            # (simulate_robot_shield, simulate_shield_failure_risk, here). Without this the
            # alpha_max and no-calibration rows are indistinguishable from the conditional one
            # and collapse onto the same table row.
            set_kind = {"max": "max_conformal", "uncalibrated": "uncalibrated"}.get(
                used_cal.get("mode"), "conditional_conformal")
        row = dict(
            results_file=os.path.basename(args.results_file), human_set=args.human_set,
            set_kind=set_kind,
            # Recorded verbatim so a row stays identifiable even if set_kind's mapping changes.
            conformal_calibrator=os.path.basename(args.conformal_calibrator or ""),
            set_likelihood=(float(used_cal["level"]) if used_cal is not None
                            else float(ctx.set_likelihood)),
            mask_ood=args.mask_ood, mask_too_fast=args.mask_too_fast, backend=args.backend,
            gpu_dtype=args.gpu_dtype, seed=args.seed, fps=args.fps,
            robot_stride=args.robot_stride, n_trajectories=geom.nT,
            pose_radius=args.pose_radius, pose_z_offset=args.pose_z_offset,
            t_cycle=t_cycle, N_h=N_h,
            epsilon_d=args.epsilon_d, epsilon_f=eps_f, epsilon_d_given_f=eps_df,
            n_windows=N_F, k_F=k_F, n_failures_evaluated=geom.M,
            p_f_hat=p_f_hat, p_f_hat_strided=p_f_hat_sub, failure_stride=stride,
            l_corr=L_corr, l_corr_override=(args.l_corr is not None),
            n_eff=n_eff, k_eff=k_eff, p_f_ci_lo=p_f_lo, p_f_ci_hi=p_f_hi, p_f_upper=p_f_up,
            n_episodes=int(episodes.size), episode_len_mean=L_meas,
            episode_len_max=int(episodes.max()) if episodes.size else 0,
            escape_median_m=float(np.median(esc_f)), escape_max_m=float(esc_f.max()),
            v_cyl=geom.v_cyl, v_w=geom.v_w, p_w=geom.p_w,
            n_trials=N_W, n_kernel=c["n_gate5"], n_contact=c["n_contact"],
            k_D=k_D, k_D_unsafe=c["n_dangerous_fast"],
            p_d_given_f_w_hat=p_dgw_hat, p_d_given_f_w_upper=p_dgw_up,
            p_d_given_f_hat=p_d_given_f_hat, p_d_given_f_upper=p_d_given_f_up,
            pfh_d_hat=pfh_hat, pfh_d_upper=pfh_up, pl=pl_label(pfh_up),
            equivalent_uniform_placements=N_W / geom.p_w,
            trial_rate_per_s=N_W / max(dt, 1e-9),
            verify_gate_n=(c_gate["n"] if c_gate else 0),
            verify_gate_dangerous=(c_gate["n_dangerous"] if c_gate else -1),
            verify_gate_contact=(c_gate["n_contact"] if c_gate else -1),
            verify_gate_contact_outside_w4=(c_gate["n_contact_outside_w4"] if c_gate else -1),
            verify_gate_contact_outside_w5=(c_gate["n_contact_outside_w5"] if c_gate else -1),
            verify_gate_p_d_given_f=gate_est,
            lemma_n_trials=(lemma["n"] if lemma else 0),
            lemma_n_dangerous=(lemma["n_dangerous"] if lemma else -1),
        )
        out_csv = (args.results_csv if os.path.isabs(args.results_csv)
                   else os.path.join(root_dir, args.results_csv))
        write_results_csv(out_csv, row, fieldnames=list(row.keys()))
        print(f"\nAppended run summary to {out_csv}")


if __name__ == "__main__":
    main()
