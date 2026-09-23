"""Shared --prepare step for shots s11/s12/s13 (the safety-shield volume simulation).

Runs in the REPO venv (jax/scipy/cloudpickle), rebuilds *exactly* the shield state the paper run
used and dumps the little bit of geometry the three renderers need into

    video/media/s1x_shield_sim.npz

Command (from the repo root, ~3 min, mostly the 500 MB cloudpickle load)::

    XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python video/shots/s1x_shield_data.py

Everything written here is traceable to:
  * final_results/robot_shield_safety_results_risk_volume.sh    (the run's flags, method 1/5)
  * results/final/robot_shield_risk_volume/trials_ours_ood.npz  (the 434 windows + 4 dangerous trials)
  * datasets/robot_reachable_sets/ablation_conformal_prediction_sets_panda_4ms.csv (robot capsules)

The four dangerous trials are re-scored here with the repo's own TrialEvaluator (CPU/float64), so
the cache also carries the verdict triple (not_verified, contact, unsafe) per trial.
"""
from __future__ import annotations

import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
# TrialEvaluator/build_shield_state peek at argv for the backend; force the CPU path.
sys.argv = [sys.argv[0], "--backend", "cpu"]

from conformal_human_motion_prediction.examples import simulate_shield_failure_risk_volume as V  # noqa: E402
from conformal_human_motion_prediction.examples.simulate_shield_failure_risk import (  # noqa: E402
    build_failure_state,
)
from conformal_human_motion_prediction.examples.simulate_robot_shield import N_LINKS  # noqa: E402

TRIALS_NPZ = "results/final/robot_shield_risk_volume/trials_ours_ood.npz"
OUT = os.path.join(REPO, "video", "media", "s1x_shield_sim.npz")

# Method "1/5  Ours OOD filtered" of final_results/robot_shield_safety_results_risk_volume.sh.
ARGV = [
    "--results_file", "results/motion_prediction/motion_prediction_results_test.cloudpickle",
    "--conformal_calibrator",
    "models/motion_prediction/conformal_calibration/conformal_calibrator.npz",
    "--human_set", "conformal", "--mask_ood",
    "--backend", "cpu", "--gpu_dtype", "float64",
    "--robot_stride", "1", "--pose_radius", "10.0", "--pose_z_offset", "0.2",
    "--seed", "0", "--fps", "25",
]


def main():
    args = V.build_parser().parse_args(ARGV)
    ctx, _results, fail_stats = build_failure_state(args)
    st = ctx.st
    geom = V.build_trial_geometry(ctx, args)
    ev = V.TrialEvaluator(st, geom, args)

    tr = np.load(os.path.join(REPO, TRIALS_NPZ))
    ev_idx = fail_stats["evaluated"]
    assert np.array_equal(ev_idx, tr["window_idx"]), "failure set drifted from the saved run"
    assert np.allclose(geom.p_w, float(tr["p_w"])), (geom.p_w, float(tr["p_w"]))

    dang = np.asarray(tr["dangerous_trials"], dtype=np.float64)     # [4, (m, j, yaw, tx, ty, tz)]
    mi = dang[:, 0].astype(np.int64)
    ji = dang[:, 1].astype(np.int64)
    poses = dang[:, 2:6]

    nv, ct, us = ev.evaluate(mi, ji, poses)
    l4 = V.in_level4_ball(geom, mi, ji, poses)
    l5 = ev.gate5(mi, ji, poses)
    print("\n--- re-scoring the 4 saved dangerous trials with the repo kernel (CPU, float64) ---")
    for d in range(len(mi)):
        print(f"  trial {d}: m={mi[d]:4d} j={ji[d]:5d}  verified={not nv[d]}  contact={ct[d]}  "
              f"unsafe={us[d]}  level4={l4[d]}  level5={l5[d]}  "
              f"DANGEROUS={bool((~nv[d]) & ct[d])}")

    # ---- robot: the actual arm pose at every 4 ms phase = that timestep's interval-0 row -------
    robot = ctx.robot
    times_ms = np.asarray(ctx.times_ms, dtype=np.int64)
    i0 = np.array([robot["interval0_row"][int(t)] for t in times_ms], dtype=np.int64)
    r_p1 = robot["p1"][i0].astype(np.float32)          # [nT, 7, 3]
    r_p2 = robot["p2"][i0].astype(np.float32)
    r_rad = robot["r"][i0].astype(np.float32)          # [nT, 7]

    # ---- the hero trials, row-structured (one row = one monitored interval) ----------------
    # Rebuilt from st.traj_meta exactly the way build_capsule_tables does, but keeping the row
    # structure so the renderer can animate wall-clock time: interval i starts at tp_start[row_i]
    # after the trajectory phase, the PLANNED capsules (radius grown by the v_human bridge) are
    # what the shield verifies against the predicted human set, and the TRUE capsules are that
    # future instant's interval-0 occupancy, which is what can actually make contact.
    hz, vh, vr = st.horizon_times, st.v_human, st.v_robot
    from conformal_human_motion_prediction.examples.simulate_robot_shield import (
        last_step_below, nearest_step,
    )
    nd = len(mi)
    Rmax = max(len(st.traj_meta[j][0]) for j in ji)
    hero = dict(
        n_rows=np.zeros(nd, np.int64),
        pl_p1=np.zeros((nd, Rmax, N_LINKS, 3), np.float32),
        pl_p2=np.zeros((nd, Rmax, N_LINKS, 3), np.float32),
        pl_r=np.full((nd, Rmax, N_LINKS), -1.0, np.float32),
        pl_step=np.zeros((nd, Rmax), np.int64), pl_tp=np.zeros((nd, Rmax), np.float32),
        tr_p1=np.zeros((nd, Rmax, N_LINKS, 3), np.float32),
        tr_p2=np.zeros((nd, Rmax, N_LINKS, 3), np.float32),
        tr_r=np.full((nd, Rmax, N_LINKS), -1.0, np.float32),
        tr_step=np.zeros((nd, Rmax), np.int64), tr_tp=np.zeros((nd, Rmax), np.float32),
        tr_fast=np.zeros((nd, Rmax, N_LINKS), bool),
        tr_phase=np.full((nd, Rmax), -1, np.int64),   # index into robot_p1/p2 (the 4 ms grid)
    )
    for d, j in enumerate(ji):
        rows, frow_rows = st.traj_meta[j][0], st.traj_meta[j][1]
        hero["n_rows"][d] = len(rows)
        for i, row in enumerate(rows):
            row = int(row)
            tp = float(st.tp_start[row])
            s_pred = last_step_below(hz, tp)
            addr = (tp - hz[s_pred]) * vh
            hero["pl_p1"][d, i] = st.p1[row]
            hero["pl_p2"][d, i] = st.p2[row]
            hero["pl_r"][d, i] = st.rr[row] + addr
            hero["pl_step"][d, i], hero["pl_tp"][d, i] = s_pred, tp
            frow = int(frow_rows[i])
            if frow < 0:
                continue
            hero["tr_p1"][d, i] = st.p1[frow]
            hero["tr_p2"][d, i] = st.p2[frow]
            hero["tr_r"][d, i] = st.rr[frow]
            hero["tr_step"][d, i] = nearest_step(hz, tp)
            hero["tr_tp"][d, i] = tp
            hero["tr_fast"][d, i] = st.spd[frow] > vr
            hero["tr_phase"][d, i] = int(np.searchsorted(times_ms, times_ms[j] + round(tp * 1000)))

    # sanity: the row-structured true capsules must be the interval-0 rows of the 4 ms grid
    for d in range(nd):
        for i in range(int(hero["n_rows"][d])):
            ph = int(hero["tr_phase"][d, i])
            if ph >= 0 and hero["tr_r"][d, i, 0] > 0:
                assert np.allclose(hero["tr_p1"][d, i], r_p1[ph], atol=1e-6), (d, i, ph)

    # ---- per-trial diagnostics, with the repo's own segment-point geometry -------------------
    from conformal_human_motion_prediction.examples.simulate_robot_shield import (
        segment_point_distances,
    )
    clearance = np.zeros(nd)          # min gap, planned capsules vs predicted human set  (>0 = verified)
    penetration = np.zeros(nd)        # max overlap, true capsules vs true human occupancy (>0 = contact)
    contact_row = np.zeros(nd, np.int64)
    contact_link = np.zeros(nd, np.int64)
    contact_joint = np.zeros(nd, np.int64)
    contact_step = np.zeros(nd, np.int64)
    for d in range(nd):
        m, (yaw, tx, ty, tz) = int(mi[d]), poses[d]
        c, sn = np.cos(yaw), np.sin(yaw)
        rot = np.array([[c, -sn, 0.0], [sn, c, 0.0], [0.0, 0.0, 1.0]])
        tv = np.array([tx, ty, tz])
        gap, pen = np.inf, -np.inf
        for i in range(int(hero["n_rows"][d])):
            for a in range(N_LINKS):
                if hero["pl_r"][d, i, a] > 0:
                    p1 = hero["pl_p1"][d, i, a] @ rot.T + tv
                    p2 = hero["pl_p2"][d, i, a] @ rot.T + tv
                    k = int(hero["pl_step"][d, i])
                    dd = segment_point_distances(p1, p2, st.pred_c[m, k])
                    gap = min(gap, float((dd - hero["pl_r"][d, i, a] - st.pred_r[m, k]).min()))
                if hero["tr_r"][d, i, a] > 0:
                    p1 = hero["tr_p1"][d, i, a] @ rot.T + tv
                    p2 = hero["tr_p2"][d, i, a] @ rot.T + tv
                    k = int(hero["tr_step"][d, i])
                    dd = segment_point_distances(p1, p2, st.true_c[m, k])
                    ov = hero["tr_r"][d, i, a] + st.true_r[m, k] - dd
                    jbest = int(np.argmax(ov))
                    if ov[jbest] > pen:
                        pen = float(ov[jbest])
                        contact_row[d], contact_link[d] = i, a
                        contact_joint[d], contact_step[d] = jbest, k
        clearance[d], penetration[d] = gap, pen
        print(f"  trial {d}: verification clearance {gap * 1000:+.2f} mm, contact overlap "
              f"{pen * 1000:+.2f} mm at interval {contact_row[d]} link {contact_link[d]} "
              f"joint {contact_joint[d]} step {contact_step[d]}")

    out = dict(
        # --- human occupancy of the 434 failing windows (m -> the sub-population index) ---
        pred_c=st.pred_c.astype(np.float32), pred_r=st.pred_r.astype(np.float32),
        true_c=st.true_c.astype(np.float32), true_r=st.true_r.astype(np.float32),
        horizon_times=ctx.horizon_times.astype(np.float32),
        window_idx=ev_idx, results_row=ctx.keep_idx,
        escape_m=tr["escape_m"], escape_step=tr["escape_step"], escape_joint=tr["escape_joint"],
        # --- the shield's own bounding-sphere hierarchy (true side) ---
        hmi_true_c=st.hmi_true_c.astype(np.float32), hmi_true_r=st.hmi_true_r.astype(np.float32),
        oa_true_c=st.oa_true_c.astype(np.float32), oa_true_r=st.oa_true_r.astype(np.float32),
        k_of_traj=geom.k_of_traj, traj_sphere_c=geom.A.astype(np.float32),
        traj_sphere_r=geom.R.astype(np.float32),
        link_sphere_c=geom.Lc.astype(np.float32), link_sphere_r=geom.Lr.astype(np.float32),
        # --- robot ---
        robot_p1=r_p1, robot_p2=r_p2, robot_r=r_rad, traj_times_ms=times_ms,
        # --- run-level numbers ---
        v_cyl=np.float64(geom.v_cyl), p_w=np.float64(geom.p_w), v_w=np.float64(geom.v_w),
        n_trials=tr["n_trials"], n_dangerous=tr["n_dangerous"],
        dangerous_trials=dang,
        dang_m=mi, dang_j=ji, dang_pose=poses,
        dang_verified=~nv, dang_contact=ct, dang_unsafe=us,
        dang_level4=l4, dang_level5=l5,
        pose_radius=np.float64(args.pose_radius), pose_z_offset=np.float64(args.pose_z_offset),
        clearance=clearance, penetration=penetration, contact_row=contact_row,
        contact_link=contact_link, contact_joint=contact_joint, contact_step=contact_step,
        **hero,
    )
    # The summary row of the run this cache belongs to (method "1/5 Ours OOD filtered",
    # n_trials = 4e9): results/final/robot_shield_risk_volume/shield_risk_volume_results.csv.
    import csv as _csv
    with open(os.path.join(REPO, "results/final/robot_shield_risk_volume/"
                                 "shield_risk_volume_results.csv"), newline="") as fh:
        rows = [r for r in _csv.DictReader(fh)
                if r["set_kind"] == "conditional_conformal" and r["mask_ood"] == "True"
                and int(r["n_trials"]) == 4_000_000_000 and r["n_windows"] == "59154"]
    assert len(rows) == 1, f"expected one paper row, got {len(rows)}"
    out["csv_keys"] = np.array(list(rows[0].keys()))
    out["csv_vals"] = np.array([str(v) for v in rows[0].values()])

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez_compressed(OUT, **out)
    print(f"\nwrote {OUT}  ({os.path.getsize(OUT) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
