"""Simulate a SARA-style safety shield on recorded human scenes against a robot trajectory.

We are given, per robot planning timestep (``time``), a *monitored trajectory*: a sequence of
time intervals, each describing the robot's predicted link occupancy (7 capsules) and link
speeds over a short look-ahead (``tp_start``..``tp_end`` seconds after the planning time). The
first interval (``interval_idx == 0``) of every timestep matches the robot's *true* occupancy at
that instant.

We drop this robot (base at the origin) into recorded human motion-prediction samples and ask:
**how often would the shield verify a monitored trajectory as safe, while the ground truth has a
(possibly unsafe) human-robot contact?**

For each (monitored trajectory k, human sample m) pair, and each interval of k:

  * predicted robot occupancy  -- the interval's 7 capsules (directly from the CSV).
  * true robot occupancy       -- interval 0 of the robot timestep planned at ``time + tp``
                                  (the first interval of a future timestep is the true state).
  * predicted human occupancy  -- the conformal/covariance reachable spheres of sample m at the
                                  horizon step at/below ``tp``, with radius grown by
                                  ``(tp - horizon_time) * V_HUMAN_ISO`` to bridge to the exact
                                  interval time (radius growth folded onto the robot capsule).
  * true human occupancy        -- the ground-truth (``targets``) spheres of sample m at the
                                  horizon step nearest ``tp``.

Shield verdict (per pair): ``verified`` iff *no* predicted robot capsule intersects any predicted
human sphere, across all intervals. Ground truth: ``contact`` iff *any* true robot capsule
intersects any true human sphere; ``unsafe_contact`` iff such a contact occurs while the
contacting robot link's true speed exceeds ``V_ROBOT_ISO``.

We report how often ``verified`` pairs nevertheless had a contact / unsafe contact -- the rate at
which the shield is fooled.

Run::

    XLA_PYTHON_CLIENT_PREALLOCATE=false python -m \
        conformal_human_motion_prediction.examples.simulate_robot_shield
"""
import argparse
import multiprocessing as mp
import os
import time as _time
from types import SimpleNamespace

import cloudpickle
import numpy as np
from scipy.spatial import cKDTree

from conformal_human_motion_prediction.motion_prediction.inference_helper import (
    calibrate_covariance_matrices,
    compute_human_occupancies,
)
from conformal_human_motion_prediction.utils.eval_utils import (
    convert_covariance_matrices_to_set,
    get_too_fast_human_movement,
)

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

N_LINKS = 7  # robot capsules per interval (cap_r_0 .. cap_r_6)

# ISO 13849-1 Performance Levels, keyed by the PFH_D band [lo, hi) (failures per hour).
PL_BANDS = [
    ("e", 1e-8, 1e-7),
    ("d", 1e-7, 1e-6),
    ("c", 1e-6, 3e-6),
    ("b", 3e-6, 1e-5),
    ("a", 1e-5, 1e-4),
]


def pl_from_pfh(pfh):
    """Map a PFH_D value (1/h) to the achievable ISO 13849-1 Performance Level."""
    if pfh < PL_BANDS[0][1]:
        return "e (better than required)"
    for name, lo, hi in PL_BANDS:
        if lo <= pfh < hi:
            return name
    return "none (worse than PL a)"


def pfh_d_upper_bound(N, k, t_cycle, confidence):
    """One-sided upper confidence bound on PFH_D from k dangerous failures in N test cycles.

    Each cycle is a Bernoulli trial; the per-cycle dangerous-failure probability gets the exact
    Clopper-Pearson upper limit p_up = Beta.ppf(confidence, k+1, N-k) (for k=0 this is the closed
    form 1-(1-C)^(1/N)). Converted to an hourly rate via the cycle time:
        PFH_D = PFC_D * (3600 s/h) / t_cycle.
    Returns (pfc_d_upper, pfh_d_upper).
    """
    from scipy.stats import beta
    pfc = 1.0 if k >= N else float(beta.ppf(confidence, k + 1, N - k))
    return pfc, pfc * 3600.0 / t_cycle


# --------------------------------------------------------------------------- robot CSV


def load_robot_trajectories(csv_path, origin):
    """Parse the robot reachable-set CSV (positions in meters).

    Returns a dict with, per CSV row:
      time, tp_start, tp_end, interval_idx, p1 [R,7,3], p2 [R,7,3], r [R,7], speed [R,7]
    plus ``traj_rows`` (time_ms -> sorted row indices of that monitored trajectory) and
    ``interval0_row`` (time_ms -> row index of that timestep's interval 0).
    """
    import pandas as pd

    df = pd.read_csv(csv_path, skipinitialspace=True)
    R = len(df)
    p1 = np.empty((R, N_LINKS, 3), dtype=np.float64)
    p2 = np.empty((R, N_LINKS, 3), dtype=np.float64)
    rad = np.empty((R, N_LINKS), dtype=np.float64)
    speed = np.empty((R, N_LINKS), dtype=np.float64)
    for a in range(N_LINKS):
        p1[:, a, 0] = df[f"cap_r_{a}_x1"]
        p1[:, a, 1] = df[f"cap_r_{a}_y1"]
        p1[:, a, 2] = df[f"cap_r_{a}_z1"]
        p2[:, a, 0] = df[f"cap_r_{a}_x2"]
        p2[:, a, 1] = df[f"cap_r_{a}_y2"]
        p2[:, a, 2] = df[f"cap_r_{a}_z2"]
        rad[:, a] = df[f"cap_r_{a}_r"]
        speed[:, a] = df[f"cap_r_{a}_speed"]
    origin = np.asarray(origin, dtype=np.float64)
    p1 += origin
    p2 += origin

    time = df["time"].to_numpy(dtype=np.float64)
    tp_start = df["tp_start"].to_numpy(dtype=np.float64)
    tp_end = df["tp_end"].to_numpy(dtype=np.float64)
    interval_idx = df["interval_idx"].to_numpy(dtype=np.int64)

    # 1 ms planning grid -> integer-ms keys for exact lookup.
    time_ms = np.round(time * 1000).astype(np.int64)
    traj_rows = {}
    interval0_row = {}
    for i in range(R):
        traj_rows.setdefault(int(time_ms[i]), []).append(i)
        if interval_idx[i] == 0:
            interval0_row[int(time_ms[i])] = i

    return dict(
        time=time, time_ms=time_ms, tp_start=tp_start, tp_end=tp_end,
        interval_idx=interval_idx, p1=p1, p2=p2, r=rad, speed=speed,
        traj_rows=traj_rows, interval0_row=interval0_row,
    )


# --------------------------------------------------------------------------- human scenes


def build_human_arrays(results, fps, mask_ood, mask_too_fast, ood_threshold,
                       set_likelihood, sara_meas_unc, human_radius,
                       calibrate, cov_ct, cov_it, cov_factors, max_samples, rng):
    """Build per-step human occupancy spheres (centers in m, radii in m).

    Returns horizon_times [S], pred_centers [M,S,J,3], pred_r [M,S,J],
    true_centers [M,S,J,3], true_r [M,S,J]. Step 0 is the current (observed) pose;
    steps 1..PH are the prediction horizon at (t)*dt seconds.
    """
    predictions = np.asarray(results["predictions"], dtype=np.float64)   # [N,PH,J,3] mm
    targets = np.asarray(results["targets"], dtype=np.float64)
    cov = np.asarray(results["covariance_matrices"], dtype=np.float64)    # [N,PH,J,3,3]
    ood_scores = np.asarray(results["ood_scores"], dtype=np.float64)
    last_input = np.asarray(results["last_input_poses"], dtype=np.float64)
    N, PH, J, _ = predictions.shape
    last_input = last_input[..., : J * 3].reshape(N, J, 3)

    dt = 1.0 / fps
    keep = np.ones(N, dtype=bool)
    if mask_ood:
        keep &= ood_scores <= ood_threshold
    if mask_too_fast:
        too_fast = get_too_fast_human_movement(targets, 2.0, dt)  # uses V_HUMAN_ISO=2.0
        keep &= ~np.any(too_fast, axis=(1, 2))
    idx = np.flatnonzero(keep)
    print(f"  human samples: {N} total -> {idx.size} after filtering "
          f"(ood={mask_ood}, too_fast={mask_too_fast})")
    if max_samples is not None and idx.size > max_samples:
        idx = rng.choice(idx, size=max_samples, replace=False)
        idx.sort()
        print(f"  sub-sampled to {idx.size} human samples")

    predictions, targets, cov, last_input = (
        predictions[idx], targets[idx], cov[idx], last_input[idx],
    )

    if calibrate:
        cov = calibrate_covariance_matrices(
            covariance_matrices=cov,
            constant_time_factor=cov_ct,
            increase_time_factor=cov_it,
            joint_calibration_factors=cov_factors,
        )
    # Conformal sphere radius from covariance, mm -> m.
    radius_pred = convert_covariance_matrices_to_set(cov, likelihood=set_likelihood) / 1000.0

    # Assemble step 0 (current pose) + horizon steps, convert mm -> m.
    M = idx.size
    S = PH + 1
    pred_centers = np.empty((M, S, J, 3))
    true_centers = np.empty((M, S, J, 3))
    pred_centers[:, 0] = last_input / 1000.0
    true_centers[:, 0] = last_input / 1000.0
    pred_centers[:, 1:] = predictions / 1000.0
    true_centers[:, 1:] = targets / 1000.0

    pred_unc = np.empty((M, S, J))
    pred_unc[:, 0] = sara_meas_unc           # current pose measurement uncertainty (m)
    pred_unc[:, 1:] = radius_pred
    true_unc = np.zeros((M, S, J))            # ground truth: no positional uncertainty

    human_radius = np.asarray(human_radius, dtype=np.float64)  # meters, [J]
    # compute_human_occupancies = uncertainty + body radius, with all-zero centers -> 0.
    pred_r = compute_human_occupancies(pred_centers, pred_unc, human_radius)
    true_r = compute_human_occupancies(true_centers, true_unc, human_radius)

    horizon_times = np.array([0.0] + [(t + 1) * dt for t in range(PH)])  # [S]
    return horizon_times, pred_centers, pred_r, true_centers, true_r


# --------------------------------------------------------------------------- geometry


def segment_point_distances(a, b, pts):
    """Distance from each point in ``pts`` [n,3] to the segment a-b. Returns [n]."""
    ab = b - a
    denom = float(ab @ ab)
    ap = pts - a
    if denom <= 1e-18:  # degenerate capsule (p1 == p2) -> point
        return np.linalg.norm(ap, axis=1)
    t = np.clip(ap @ ab / denom, 0.0, 1.0)
    proj = a + t[:, None] * ab
    return np.linalg.norm(pts - proj, axis=1)


def capsule_candidates(tree, a, b, query_r):
    """Indices of tree points within ``query_r`` of segment a-b (broad phase).

    Single bounding-sphere query: the capsule's influence region is contained in the sphere
    centered at the segment midpoint with radius ``seg_len/2 + query_r``. The exact
    segment-point test downstream removes the few extra candidates this admits.
    """
    mid = 0.5 * (a + b)
    r = 0.5 * float(np.linalg.norm(b - a)) + query_r
    return np.asarray(tree.query_ball_point(mid, r), dtype=np.int64)


def nearest_step(horizon_times, tp):
    return int(np.argmin(np.abs(horizon_times - tp)))


def last_step_below(horizon_times, tp):
    le = np.flatnonzero(horizon_times <= tp + 1e-9)
    return int(le[-1]) if le.size else 0


def bound_spheres(centers, radii):
    """o(S): bounding sphere of a set of spheres. centers [...,n,3], radii [...,n].

    Center = mean of sub-centers; radius = max_i(||center - c_i|| + r_i). A sound (not minimal)
    over-approximation that commutes with rigid transforms. Returns (center [...,3], radius [...]).
    """
    center = centers.mean(axis=-2)
    d = np.linalg.norm(centers - center[..., None, :], axis=-1)  # [...,n]
    return center, (d + radii).max(axis=-1)


def cumulative_human_overapprox(centers, radii, kmax):
    """Cumulative whole-time-range bounding spheres for human occupancies.

    centers [M,S,J,3], radii [M,S,J] -> (oa_c [M,K,J,3], oa_r [M,K,J]) for K=kmax+1, where
    entry k bounds each joint's occupancy over horizon steps 0..k: center = mean position over
    those steps, radius = max(dist-to-center + occupancy radius). Sound over-approximation of
    the union of the per-step spheres.
    """
    M, S, J, _ = centers.shape
    oa_c = np.empty((M, kmax + 1, J, 3), np.float32)
    oa_r = np.empty((M, kmax + 1, J), np.float32)
    for k in range(kmax + 1):
        c = centers[:, : k + 1]                              # [M,k+1,J,3]
        mean = c.mean(axis=1)                                # [M,J,3]
        d = np.linalg.norm(c - mean[:, None], axis=3)        # [M,k+1,J]
        oa_c[:, k] = mean.astype(np.float32)
        oa_r[:, k] = (d + radii[:, : k + 1]).max(axis=1).astype(np.float32)
    return oa_c, oa_r


def robot_trajectory_overapprox(p1, p2, rad, rows):
    """One bounding sphere per link covering its swept capsule over a monitored trajectory.

    p1, p2 [R,L,3] endpoints and rad [R,L] capsule radii for the R rows of the trajectory.
    Returns centers [L,3], radii [L]. Both endpoints are bounded, so by convexity the whole
    capsule segment (plus its radius) is covered for every interval.
    """
    pts = np.concatenate([p1[rows], p2[rows]], axis=0)       # [2R,L,3]
    center = pts.mean(axis=0)                                # [L,3]
    d = np.linalg.norm(pts - center[None], axis=2)           # [2R,L]
    radius = d.max(axis=0) + rad[rows].max(axis=0)           # [L]
    return center, radius


def overapprox_candidates(rc, rr, oa_c_k, oa_r_k):
    """Humans whose over-approx spheres may touch the robot's (broad-phase survivors).

    rc [L,3], rr [L] robot link spheres; oa_c_k [M,J,3], oa_r_k [M,J] human spheres at the
    chosen cumulative step. Returns a bool mask [M]: True where some (link, joint) sphere pair
    is within the sum of radii. Non-survivors cannot intersect any interval -> safe to skip.
    """
    cand = np.zeros(oa_c_k.shape[0], dtype=bool)
    for a in range(rc.shape[0]):
        d = np.linalg.norm(oa_c_k - rc[a], axis=2)           # [M,J]
        cand |= np.any(d <= rr[a] + oa_r_k, axis=1)
    return cand


def sample_robot_poses(num, radius, z_off, rng):
    """Sample ``num`` random robot base poses.

    (x, y) uniform in a disk of ``radius`` m (area-uniform), z uniform in [-z_off, z_off],
    yaw uniform in [-pi, pi]. Returns an array [num, 4] of (yaw, tx, ty, tz).
    """
    r = radius * np.sqrt(rng.uniform(0.0, 1.0, num))
    theta = rng.uniform(0.0, 2 * np.pi, num)
    tx, ty = r * np.cos(theta), r * np.sin(theta)
    tz = rng.uniform(-z_off, z_off, num)
    yaw = rng.uniform(-np.pi, np.pi, num)
    return np.stack([yaw, tx, ty, tz], axis=1)


def pose_rt(pose):
    """(yaw, tx, ty, tz) -> (rot_t, t) such that a point transforms as ``p @ rot_t + t``.

    rot_t = Rz(yaw).T, so ``p @ rot_t`` rotates p about the base z-axis; t is the translation.
    """
    yaw, tx, ty, tz = pose
    c, s = np.cos(yaw), np.sin(yaw)
    rot_t = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
    return rot_t, np.array([tx, ty, tz])


def transform_capsules(p1, p2, pose):
    """Apply a base pose (yaw about z, then translation) to capsule endpoints.

    p1, p2 [R,L,3] in the robot base frame. Capsule radii and link speeds are pose-invariant.
    """
    rot_t, t = pose_rt(pose)
    return p1 @ rot_t + t, p2 @ rot_t + t


# --------------------------------------------------------------------------- simulation


def run_pose_pure(st, pose):
    """Evaluate one robot base pose over all trajectories x humans. Returns a counter dict.

    ``st`` is a read-only SimpleNamespace of precomputed state (human trees, hierarchy, robot
    schedule). Self-contained so it can run in a forked worker process.
    """
    M, J, hz, rr, spd = st.M, st.J, st.horizon_times, st.rr, st.spd
    vh, vr, tp_start = st.v_human, st.v_robot, st.tp_start
    c = dict(total_pairs=0, n_verified=0, n_contact=0, n_unsafe=0, n_verified_contact=0,
             n_verified_unsafe=0, n_intervals_no_truth=0, n_pred_cand=0, n_true_cand=0,
             n_poses_skipped=0, active=M)
    rot_t, tvec = pose_rt(pose)
    p1r = st.p1 @ rot_t + tvec
    p2r = st.p2 @ rot_t + tvec

    if st.overapprox:
        r_c = st.R_c @ rot_t + tvec
        if np.linalg.norm(st.H_c - r_c) > st.H_r + st.R_r:        # level 1: skip the whole pose
            n_traj = len(st.traj_meta)
            c.update(total_pairs=M * n_traj, n_verified=M * n_traj,
                     n_intervals_no_truth=st.total_no_truth, n_poses_skipped=1, active=0)
            return c
        grp_hit = np.linalg.norm(st.hsm_c - r_c, axis=1) <= st.hsm_r + st.R_r          # level 2
        active = (np.linalg.norm(st.hm_comb_c - r_c, axis=1) <= st.hm_comb_r + st.R_r) \
            & grp_hit[st.group_id]                                                     # level 3
    else:
        active = np.ones(M, dtype=bool)
    c["active"] = int(active.sum())
    ai = np.flatnonzero(active)

    for rows, frow_rows, valid, k, max_addr, rcp_c, rcp_r, rct_c, rct_r in st.traj_meta:
        not_verified = np.zeros(M, dtype=bool)
        contact = np.zeros(M, dtype=bool)
        unsafe = np.zeros(M, dtype=bool)
        c["n_intervals_no_truth"] += int((~valid).sum())

        if not st.overapprox:
            pred_cand = np.ones(M, dtype=bool)
            true_cand = np.ones(M, dtype=bool) if valid.any() else np.zeros(M, dtype=bool)
        elif ai.size == 0:
            pred_cand = true_cand = np.zeros(M, dtype=bool)
        else:
            # ---- level 4 (per traj x active motion) then level 5 (per link x body) ----
            pred_cand = np.zeros(M, dtype=bool)
            true_cand = np.zeros(M, dtype=bool)
            rc_p = rcp_c @ rot_t + tvec
            rt_c, rt_r = bound_spheres(rc_p[None], rcp_r[None])
            dp = np.linalg.norm(st.hmi_pred_c[ai, k] - rt_c[0], axis=1)
            aip = ai[dp <= rt_r[0] + st.hmi_pred_r[ai, k]]
            if aip.size:
                keep = overapprox_candidates(rc_p, rcp_r, st.oa_pred_c[aip, k], st.oa_pred_r[aip, k])
                pred_cand[aip[keep]] = True
            if valid.any():
                rc_t = rct_c @ rot_t + tvec
                rtt_c, rtt_r = bound_spheres(rc_t[None], rct_r[None])
                dt = np.linalg.norm(st.hmi_true_c[ai, k] - rtt_c[0], axis=1)
                ait = ai[dt <= rtt_r[0] + st.hmi_true_r[ai, k]]
                if ait.size:
                    keep = overapprox_candidates(rc_t, rct_r, st.oa_true_c[ait, k], st.oa_true_r[ait, k])
                    true_cand[ait[keep]] = True
            c["n_pred_cand"] += int(pred_cand.sum())
            c["n_true_cand"] += int(true_cand.sum())

        if pred_cand.any():
            for row in rows:
                tp = tp_start[row]
                s_pred = last_step_below(hz, tp)
                addr = (tp - hz[s_pred]) * vh
                for a in range(N_LINKS):
                    eff_r = rr[row, a] + addr
                    p1, p2 = p1r[row, a], p2r[row, a]
                    cand = capsule_candidates(st.pred_trees[s_pred], p1, p2, eff_r + st.pred_rmax[s_pred])
                    if cand.size:
                        m_of = cand // J
                        cand = cand[pred_cand[m_of] & ~not_verified[m_of]]
                        if cand.size:
                            d = segment_point_distances(p1, p2, st.pred_c_flat[s_pred][cand])
                            hit = cand[d <= eff_r + st.pred_r_flat[s_pred][cand]]
                            if hit.size:
                                not_verified[np.unique(hit // J)] = True

        if true_cand.any():
            for i, row in enumerate(rows):
                frow = int(frow_rows[i])
                if frow < 0:
                    continue
                s_true = nearest_step(hz, tp_start[row])
                for a in range(N_LINKS):
                    tr = rr[frow, a]
                    p1, p2 = p1r[frow, a], p2r[frow, a]
                    fast_link = spd[frow, a] > vr
                    cand = capsule_candidates(st.true_trees[s_true], p1, p2, tr + st.true_rmax[s_true])
                    if cand.size:
                        m_of = cand // J
                        need = (~contact[m_of] if not fast_link else ~unsafe[m_of]) & true_cand[m_of]
                        cand = cand[need]
                        if cand.size:
                            d = segment_point_distances(p1, p2, st.true_c_flat[s_true][cand])
                            hit = cand[d <= tr + st.true_r_flat[s_true][cand]]
                            if hit.size:
                                m_hit = np.unique(hit // J)
                                contact[m_hit] = True
                                if fast_link:
                                    unsafe[m_hit] = True

        verified = ~not_verified
        c["total_pairs"] += M
        c["n_verified"] += int(verified.sum())
        c["n_contact"] += int(contact.sum())
        c["n_unsafe"] += int(unsafe.sum())
        c["n_verified_contact"] += int((verified & contact).sum())
        c["n_verified_unsafe"] += int((verified & unsafe).sum())
    return c


# Set in the parent before forking workers; inherited (copy-on-write) so the GB-scale human
# trees/hierarchy are shared, not pickled, to each worker.
_WORKER_ST = None


def _pose_worker(item):
    idx, pose = item
    return idx, run_pose_pure(_WORKER_ST, pose)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robot_csv", type=str,
                        default="datasets/robot_reachable_sets/ablation_conformal_prediction_sets_panda_4ms.csv")
    parser.add_argument("--results_file", type=str,
                        default="results/motion_prediction/motion_prediction_results_validation.cloudpickle")
    parser.add_argument("--config", type=str, default="h36m", choices=["h36m", "rgbd_yolo"])
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--robot_origin", type=str, default="0,0,0",
                        help="Robot base offset added to all capsule points (meters).")
    parser.add_argument("--num_robot_poses", type=int, default=1,
                        help="Number of random robot base poses to evaluate (results aggregate "
                             "over all poses). 1 = a single random placement.")
    parser.add_argument("--pose_radius", type=float, default=3.0,
                        help="Radius (m) of the disk in which (x, y) base positions are sampled.")
    parser.add_argument("--pose_z_offset", type=float, default=0.2,
                        help="Base z-offset sampled uniformly in [-pose_z_offset, pose_z_offset] (m).")
    parser.add_argument("--t_cycle", type=float, default=None,
                        help="Safety-function cycle time (s); one monitored trajectory per cycle. "
                             "Default: derived from the robot planning-grid spacing. Used to "
                             "convert the per-cycle failure rate into PFH_D.")
    # OOD filtering is off by default while the OOD score is being reworked (all current
    # samples are flagged OOD). Re-enable with --mask_ood once scores are trustworthy.
    parser.add_argument("--mask_ood", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--mask_too_fast", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--calibrate", action=argparse.BooleanOptionalAction, default=True,
                        help="Apply the project covariance calibration before forming the set.")
    parser.add_argument("--overapprox", action=argparse.BooleanOptionalAction, default=True,
                        help="Use the multi-level bounding-sphere hierarchy to cull far humans.")
    parser.add_argument("--motion_group_size", type=int, default=50,
                        help="Motions per level-2 group (consecutive samples are spatially "
                             "coherent in the sequence-ordered dataset).")
    parser.add_argument("--max_human_samples", type=int, default=None,
                        help="Cap eligible human samples (random subset) for a quick run.")
    parser.add_argument("--max_robot_timesteps", type=int, default=None,
                        help="Cap number of monitored trajectories evaluated (for a quick run).")
    parser.add_argument("--robot_stride", type=int, default=25,
                        help="Evaluate every Nth monitored trajectory (spread across the log). "
                             "25 on the 4 ms grid = trajectories 100 ms apart (decorrelated cycles).")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Processes for evaluating poses in parallel (poses are independent). "
                             "1 = serial. Uses fork so the human trees/hierarchy are shared.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.config == "rgbd_yolo":
        from conformal_human_motion_prediction.motion_prediction.rgbd_yolo_settings import (
            COV_CALIBRATION_FACTORS, COV_CALIBRATION_CT, COV_CALIBRATION_IT, SET_LIKELIHOOD,
            SARA_MEASUREMENT_UNCERTAINTY, OOD_THRESHOLD, HUMAN_RADIUS, V_HUMAN_ISO, V_ROBOT_ISO,
        )
    else:
        from conformal_human_motion_prediction.motion_prediction.h36m_settings import (
            COV_CALIBRATION_FACTORS, COV_CALIBRATION_CT, COV_CALIBRATION_IT, SET_LIKELIHOOD,
            SARA_MEASUREMENT_UNCERTAINTY, OOD_THRESHOLD, HUMAN_RADIUS, V_HUMAN_ISO, V_ROBOT_ISO,
        )

    rng = np.random.default_rng(args.seed)
    origin = [float(x) for x in args.robot_origin.split(",")]

    csv_path = os.path.join(root_dir, args.robot_csv)
    print(f"Loading robot trajectories from {csv_path} ...")
    robot = load_robot_trajectories(csv_path, origin)
    all_times_ms = np.array(sorted(robot["traj_rows"].keys()))
    # Native planning-cycle time = spacing between consecutive monitored trajectories (robust
    # median of the grid diffs, in seconds). One monitored trajectory is planned per cycle.
    t_cycle = args.t_cycle if args.t_cycle is not None else float(np.median(np.diff(all_times_ms))) / 1000.0
    print(f"Planning cycle time t_cycle = {t_cycle:g} s "
          f"({'from --t_cycle' if args.t_cycle is not None else 'derived from robot grid'})")
    times_ms = list(all_times_ms)
    if args.robot_stride > 1:
        times_ms = times_ms[:: args.robot_stride]
    if args.max_robot_timesteps is not None:
        times_ms = times_ms[: args.max_robot_timesteps]
    print(f"  {len(times_ms)} monitored trajectories, {len(robot['time'])} intervals total")

    results_file = os.path.join(root_dir, args.results_file)
    print(f"Loading human results from {results_file} ...")
    with open(results_file, "rb") as f:
        results = cloudpickle.load(f)
    horizon_times, pred_c, pred_r, true_c, true_r = build_human_arrays(
        results, args.fps, args.mask_ood, args.mask_too_fast, OOD_THRESHOLD,
        SET_LIKELIHOOD, SARA_MEASUREMENT_UNCERTAINTY, HUMAN_RADIUS,
        args.calibrate, COV_CALIBRATION_CT, COV_CALIBRATION_IT, COV_CALIBRATION_FACTORS,
        args.max_human_samples, rng,
    )
    M, S, J, _ = pred_c.shape
    if M == 0:
        raise SystemExit("No eligible human samples after filtering — relax --mask_ood/--mask_too_fast.")
    print(f"  human horizon steps (s): {np.round(horizon_times, 3).tolist()}")
    print(f"  max robot interval tp_end: {robot['tp_end'].max():.3f}s "
          f"(human horizon max {horizon_times[-1]:.3f}s)")

    # Per-step KDTrees over human joint centers (flattened (m,j) -> point).
    print("Building KDTrees over human occupancies ...")
    pred_trees, true_trees, pred_rmax, true_rmax = [], [], [], []
    for s in range(S):
        pred_trees.append(cKDTree(pred_c[:, s].reshape(M * J, 3)))
        true_trees.append(cKDTree(true_c[:, s].reshape(M * J, 3)))
        pred_rmax.append(float(pred_r[:, s].max()))
        true_rmax.append(float(true_r[:, s].max()))

    pred_r_flat = [pred_r[:, s].reshape(M * J) for s in range(S)]
    true_r_flat = [true_r[:, s].reshape(M * J) for s in range(S)]
    pred_c_flat = [pred_c[:, s].reshape(M * J, 3) for s in range(S)]
    true_c_flat = [true_c[:, s].reshape(M * J, 3) for s in range(S)]

    # Multi-level bounding-sphere hierarchy for hierarchical culling.
    if args.overapprox:
        kmax = min(int(np.searchsorted(horizon_times, robot["tp_end"].max(), side="left")), S - 1)
        print(f"Precomputing human over-approximation hierarchy (cumulative steps 0..{kmax}) ...")
        # Level 5 (finest): per-motion, per-body, per-cumulative-interval (H_MTI).
        oa_pred_c, oa_pred_r = cumulative_human_overapprox(pred_c, pred_r, kmax)
        oa_true_c, oa_true_r = cumulative_human_overapprox(true_c, true_r, kmax)
        # Level 4: per-motion, full-human, per-interval (H_MI = o over bodies).
        hmi_pred_c, hmi_pred_r = bound_spheres(oa_pred_c, oa_pred_r)   # [M,K1,3], [M,K1]
        hmi_true_c, hmi_true_r = bound_spheres(oa_true_c, oa_true_r)
        # Level 1-3 use the combined (pred U true) full-interval sphere per motion: a coarse cull
        # means neither predicted nor true occupancy intersects -> verified AND contact-free.
        hm_c = np.stack([hmi_pred_c[:, kmax], hmi_true_c[:, kmax]], axis=1)   # [M,2,3]
        hm_r = np.stack([hmi_pred_r[:, kmax], hmi_true_r[:, kmax]], axis=1)   # [M,2]
        hm_comb_c, hm_comb_r = bound_spheres(hm_c, hm_r)                      # H_M [M,3],[M]
        # Level 2: set-of-motions groups (consecutive samples are spatially coherent).
        sgrp = max(1, args.motion_group_size)
        group_id = np.arange(M) // sgrp
        n_groups = int(group_id[-1]) + 1
        hsm_c = np.empty((n_groups, 3))
        hsm_r = np.empty(n_groups)
        for g in range(n_groups):
            sel = group_id == g
            c, r = bound_spheres(hm_comb_c[sel][None], hm_comb_r[sel][None])
            hsm_c[g], hsm_r[g] = c[0], r[0]
        # Level 1: global human sphere H.
        H_c, H_r = bound_spheres(hm_comb_c[None], hm_comb_r[None])
        H_c, H_r = H_c[0], float(H_r[0])
    n_pred_cand = 0  # level-5 survivors (sum over pose,traj) for cull-rate reporting
    n_true_cand = 0

    # Aggregate counters over all (trajectory, human) pairs.
    total_pairs = 0
    n_verified = 0
    n_contact = 0
    n_unsafe = 0
    n_verified_contact = 0
    n_verified_unsafe = 0
    n_intervals_no_truth = 0  # intervals whose future robot timestep is past the log end

    # Pose-invariant per-trajectory schedule: future interval-0 rows (true robot state), the
    # cumulative over-approx step k covering the trajectory duration, and the max V_HUMAN_ISO
    # bridge on the trajectory. Only the robot capsule geometry changes between poses.
    rr, spd = robot["r"], robot["speed"]
    traj_meta = []
    glob_c, glob_r = [], []   # all base-frame robot link spheres, for the global sphere R
    for t_ms in times_ms:
        rows = np.asarray(robot["traj_rows"][t_ms], dtype=np.int64)
        fut_ms = (t_ms + np.round(robot["tp_start"][rows] * 1000)).astype(np.int64)
        frow_rows = np.array([robot["interval0_row"].get(int(f), -1) for f in fut_ms], dtype=np.int64)
        valid = frow_rows >= 0
        rcp_c = rcp_r = rct_c = rct_r = None
        if args.overapprox:
            tps = robot["tp_start"][rows]
            sp_idx = np.clip(np.searchsorted(horizon_times, tps, side="right") - 1, 0, S - 1)
            max_addr = float((tps - horizon_times[sp_idx]).max()) * V_HUMAN_ISO
            k = min(int(np.searchsorted(horizon_times, robot["tp_end"][rows].max(), side="left")), kmax)
            # Base-frame per-link spheres: predicted (this traj's capsules + bridge) and true
            # (the future interval-0 capsules). Transformed per pose; bounded once into R.
            rcp_c, rcp_r = robot_trajectory_overapprox(robot["p1"], robot["p2"], rr, rows)
            rcp_r = rcp_r + max_addr
            glob_c.append(rcp_c)
            glob_r.append(rcp_r)
            if valid.any():
                rct_c, rct_r = robot_trajectory_overapprox(robot["p1"], robot["p2"], rr, frow_rows[valid])
                glob_c.append(rct_c)
                glob_r.append(rct_r)
        else:
            max_addr, k = 0.0, 0
        traj_meta.append((rows, frow_rows, valid, k, max_addr, rcp_c, rcp_r, rct_c, rct_r))

    total_no_truth = sum(int((~m[2]).sum()) for m in traj_meta)  # for the level-1 pose skip
    if args.overapprox:
        R_c, R_r = bound_spheres(np.concatenate(glob_c)[None], np.concatenate(glob_r)[None])
        R_c, R_r = R_c[0], float(R_r[0])
    t0 = _time.time()

    # Read-only state shared with (forked) pose workers.
    st = SimpleNamespace(
        M=M, J=J, overapprox=args.overapprox, traj_meta=traj_meta, total_no_truth=total_no_truth,
        tp_start=robot["tp_start"], rr=rr, spd=spd, p1=robot["p1"], p2=robot["p2"],
        horizon_times=horizon_times, v_human=V_HUMAN_ISO, v_robot=V_ROBOT_ISO,
        pred_trees=pred_trees, true_trees=true_trees, pred_rmax=pred_rmax, true_rmax=true_rmax,
        pred_c_flat=pred_c_flat, pred_r_flat=pred_r_flat, true_c_flat=true_c_flat, true_r_flat=true_r_flat,
    )
    if args.overapprox:
        st.__dict__.update(
            oa_pred_c=oa_pred_c, oa_pred_r=oa_pred_r, oa_true_c=oa_true_c, oa_true_r=oa_true_r,
            hmi_pred_c=hmi_pred_c, hmi_pred_r=hmi_pred_r, hmi_true_c=hmi_true_c, hmi_true_r=hmi_true_r,
            hm_comb_c=hm_comb_c, hm_comb_r=hm_comb_r, hsm_c=hsm_c, hsm_r=hsm_r,
            group_id=group_id, H_c=H_c, H_r=H_r, R_c=R_c, R_r=R_r,
        )

    poses = sample_robot_poses(args.num_robot_poses, args.pose_radius, args.pose_z_offset, rng)
    n_workers = max(1, min(args.num_workers, len(poses)))
    print(f"Evaluating {len(poses)} random robot pose(s) over {n_workers} worker(s) "
          f"(xy in {args.pose_radius} m disk, z +/-{args.pose_z_offset} m, yaw +/-pi) ...")
    n_poses_skipped = 0  # whole poses culled at level 1 (robot never reaches any human)

    def accumulate(pidx, cc):
        nonlocal total_pairs, n_verified, n_contact, n_unsafe, n_verified_contact
        nonlocal n_verified_unsafe, n_intervals_no_truth, n_pred_cand, n_true_cand, n_poses_skipped
        total_pairs += cc["total_pairs"]; n_verified += cc["n_verified"]
        n_contact += cc["n_contact"]; n_unsafe += cc["n_unsafe"]
        n_verified_contact += cc["n_verified_contact"]; n_verified_unsafe += cc["n_verified_unsafe"]
        n_intervals_no_truth += cc["n_intervals_no_truth"]
        n_pred_cand += cc["n_pred_cand"]; n_true_cand += cc["n_true_cand"]
        n_poses_skipped += cc["n_poses_skipped"]
        p = poses[pidx]
        print(f"  pose {pidx + 1}/{len(poses)} "
              f"(yaw={p[0]:+.2f} xy=({p[1]:+.2f},{p[2]:+.2f}) z={p[3]:+.2f}): active={cc['active']} "
              f"cum verified&contact={n_verified_contact:,} verified&unsafe={n_verified_unsafe:,} "
              f"({_time.time() - t0:.0f}s)")

    if n_workers > 1:
        global _WORKER_ST
        _WORKER_ST = st  # set before forking so workers inherit it copy-on-write
        with mp.get_context("fork").Pool(n_workers) as pool:
            for pidx, cc in pool.imap_unordered(_pose_worker, list(enumerate(poses))):
                accumulate(pidx, cc)
    else:
        for pidx, pose in enumerate(poses):
            accumulate(pidx, run_pose_pure(st, pose))

    # ----------------------------------------------------------------- report
    def pct(num, den):
        return 100.0 * num / den if den else float("nan")

    print("\n==================== Shield simulation results ====================")
    print(f"Random robot poses     : {len(poses)}")
    print(f"Monitored trajectories : {len(times_ms)}")
    print(f"Eligible human samples : {M}")
    print(f"Total (pose, traj, human) trials : {total_pairs:,}")
    print(f"Intervals without ground-truth robot state (past log end): {n_intervals_no_truth:,}")
    if args.overapprox:
        print(f"Poses fully culled at level 1: {n_poses_skipped}/{len(poses)}")
        print(f"Level-5 survivors (detailed-checked): predicted "
              f"{pct(n_pred_cand, total_pairs):.3f}%, true {pct(n_true_cand, total_pairs):.3f}% "
              f"of trials (rest culled by the hierarchy)")
    print("-------------------------------------------------------------------")
    print(f"Verified (shield says safe) : {n_verified:,}  ({pct(n_verified, total_pairs):.3f}% of trials)")
    print(f"True contact                : {n_contact:,}  ({pct(n_contact, total_pairs):.3f}% of trials)")
    print(f"True unsafe contact         : {n_unsafe:,}  ({pct(n_unsafe, total_pairs):.3f}% of trials)")
    print("-------------------------------------------------------------------")
    print(">>> Verified BUT contact        : "
          f"{n_verified_contact:,}  "
          f"({pct(n_verified_contact, n_verified):.4f}% of verified, "
          f"{pct(n_verified_contact, total_pairs):.4f}% of trials)")
    print(">>> Verified BUT unsafe contact : "
          f"{n_verified_unsafe:,}  "
          f"({pct(n_verified_unsafe, n_verified):.4f}% of verified, "
          f"{pct(n_verified_unsafe, total_pairs):.4f}% of trials)")
    print("===================================================================")

    # ----------------------------------------------------------------- PFH_D / Performance Level
    # A "dangerous failure" is the shield declaring a trajectory verified while the ground truth
    # has an UNSAFE contact (contact at robot speed > V_ROBOT_ISO). Each (pose, traj, human) trial
    # is one safety-function cycle; we bound the per-cycle failure probability (Clopper-Pearson,
    # one-sided) and convert to PFH_D = PFC_D * 3600 / t_cycle.
    confidences = [0.99, 0.999, 0.9999]
    print("\n============== PFH_D (dangerous failure rate) per ISO 13849-1 ==============")
    print(f"Dangerous failure = verified BUT unsafe contact (speed > V_ROBOT_ISO = {V_ROBOT_ISO} m/s)")
    print(f"Test cycles N = {total_pairs:,}   dangerous failures k = {n_verified_unsafe:,}   "
          f"t_cycle = {t_cycle:g} s")
    print(f"{'confidence':>10} | {'PFC_D upper (1/cyc)':>20} | {'PFH_D upper (1/h)':>18} | PL")
    print("-" * 72)
    for C in confidences:
        pfc, pfh = pfh_d_upper_bound(total_pairs, n_verified_unsafe, t_cycle, C)
        print(f"{C:>10.4f} | {pfc:>20.3e} | {pfh:>18.3e} | {pl_from_pfh(pfh)}")
    print("=" * 76)


if __name__ == "__main__":
    main()
