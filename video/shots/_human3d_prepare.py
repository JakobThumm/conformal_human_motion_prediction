"""Prepare step for the two 3-D human-motion shots (s05, s09).  REPO VENV ONLY.

Runs under `.venv/bin/python` (needs jax to unpickle the results, spacepy/pycdf + torch for the
H36M loader).  It writes two small, self-contained caches that the renderers -- which run under the
conda `chmp-video` interpreter -- load with plain numpy:

    video/media/s05_window.npz
    video/media/s09_window.npz

    cd /home/thumm/code/conformal_human_motion_prediction
    JAX_PLATFORMS=cpu .venv/bin/python video/shots/_human3d_prepare.py

Everything in the caches is real, in-repo data:

  * poses / predictions / targets / covariances
        results/final/conformal_prediction_sets/motion_prediction_results_test.cloudpickle
        (H36M S5 test split, produced by final_results/motion_prediction_conformal_prediction_set_results.sh)
  * the 50-frame input history
        datasets/H36M/pre_processed_motion/S5/*.npz via the repo's own
        Human36mMotionDataset3DWithInputUncertainty loader (shuffle=False -> dataset index ==
        results index; asserted below against the cloudpickle targets)
  * our conformal set radius
        models/motion_prediction/conformal_calibration/conformal_calibrator.npz applied through
        the repo's `motion_prediction.inference_helper.conformal_set_radius`
  * the ISO 13855 / SARA baseline radius
        repo's `utils.eval_utils.compute_sara_predictions`, v_h,max = V_HUMAN_ISO = 2.0 m/s,
        measurement uncertainty = the per-joint input-uncertainty radius of the last observed pose
        (exactly what examples/motion_prediction.py feeds it for the paper's table)
  * the headline median volumes
        results/final/conformal_prediction_sets/coverage_stats_sara.csv and
        coverage_stats_conformal_prediction_sets.csv
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
MEDIA = REPO / "video" / "media"
sys.path.insert(0, str(REPO / "src"))

RESULTS = REPO / "results/final/conformal_prediction_sets"
TEST_PKL = RESULTS / "motion_prediction_results_test.cloudpickle"
CALIB = REPO / "models/motion_prediction/conformal_calibration/conformal_calibrator.npz"

FPS = 25.0
K_I, K_P, J = 50, 10, 13


# --------------------------------------------------------------------------- provenance loader
def window_provenance(data_path: Path, split: str = "test"):
    """Re-walk the loader's file order to label every *unfiltered* window (action, offset, start).

    Mirrors ``Human36mMotionDataset3D.load_data_preprocessed`` exactly (same os.listdir order, same
    validity screens).  Only used to name the picked window in the notes; the pick itself is
    verified against the cloudpickle, so a drift here cannot corrupt the cache.
    """
    from conformal_human_motion_prediction.datasets.h36m_motion_prediction import SPLIT
    from conformal_human_motion_prediction.datasets.h36m import JOINT_IDX_13, JOINT_IDX_17
    from spacepy.pycdf import CDF

    prov = []
    for subject in SPLIT[split]:
        udir = data_path / "H36M" / "pre_processed_motion" / subject
        if not udir.exists():
            continue
        for filename in os.listdir(udir):   # same (arbitrary but stable) order as the loader
            if not filename.endswith(".npz"):
                continue
            action = os.path.splitext(filename)[0]
            d = np.load(udir / filename)
            pred_poses = d["poses_3d"].reshape(d["poses_3d"].shape[0], -1)
            covariances = d["covariances_3d"].reshape(d["covariances_3d"].shape[0], -1)
            valid_mask = d["valid_mask"].astype(bool)
            pose_valid = np.all(np.abs(pred_poses) <= 4000, axis=1)
            cov_estimated = np.abs(covariances).max(axis=1) > 0.0
            cov_valid = np.all(np.abs(covariances) <= 1e5, axis=1) & cov_estimated
            valid_mask = valid_mask & pose_valid & cov_valid

            gt_file = data_path / "H36M" / "extracted" / subject / "Poses_D3_Positions" / f"{action}.cdf"
            n_gt = len(pred_poses)
            if gt_file.exists():
                with CDF(str(gt_file)) as cdf:
                    n_gt = cdf["Pose"][:].reshape(-1, 32, 3).shape[0]
            m = min(len(pred_poses), n_gt, len(covariances), len(valid_mask))
            valid_mask = valid_mask[:m]
            for offset in (0, 1):
                vm = valid_mask[offset::2]
                for i in range(len(vm) - K_I - K_P + 1):
                    if vm[i:i + K_I + K_P].sum() < K_I + K_P:
                        continue
                    prov.append((subject, action, offset, i))
    return prov


# --------------------------------------------------------------------------- main
def main():
    import cloudpickle
    from conformal_human_motion_prediction.datasets import dataloader_from_string
    from conformal_human_motion_prediction.motion_prediction.h36m_settings import V_HUMAN_ISO
    from conformal_human_motion_prediction.motion_prediction.inference_helper import (
        conformal_set_radius, load_conformal_calibrator)
    from conformal_human_motion_prediction.utils.eval_utils import (
        compute_sara_predictions, convert_covariance_matrices_to_set, covariance_sigma_max)
    from conformal_human_motion_prediction.datasets.h36m import CONNECTIONS_13

    with open(TEST_PKL, "rb") as f:
        r = cloudpickle.load(f)
    pred = np.asarray(r["predictions"], dtype=np.float64)          # [N,10,13,3] mm
    tgt = np.asarray(r["targets"], dtype=np.float64)               # [N,10,13,3] mm
    cov = np.asarray(r["covariance_matrices"], dtype=np.float64)   # [N,10,13,3,3] mm^2
    li = np.asarray(r["last_input_poses"], dtype=np.float64)       # [N, 13*3 + 13*9]
    is_ood = np.asarray(r["is_oods"]).astype(bool).ravel()
    N = pred.shape[0]
    in_cov = li[:, J * 3:J * 3 + J * 9].reshape(N, J, 3, 3)
    last_pose = li[:, :J * 3].reshape(N, J, 3)
    print(f"results: N={N}  OOD flagged={int(is_ood.sum())}")

    # ---- our conformal sets: the deployed conditional calibrator ---------------------------
    calib = load_conformal_calibrator(str(CALIB))
    assert calib is not None and calib["mode"] == "conditional"
    r_ours = conformal_set_radius(cov, in_cov, calib)                       # [N,10,13] mm
    # The model's own Gaussian 99.99 % covariance ball, BEFORE conformal calibration -- this is
    # what s05 draws ("...predicts the next 400 ms, with covariances"); s06 is where the
    # calibration happens and s09 draws the calibrated set above.
    sigma_max = covariance_sigma_max(cov)                                   # [N,10,13] mm
    r_model = convert_covariance_matrices_to_set(cov, calib["level"])       # [N,10,13] mm

    # ---- ISO 13855 / SARA constant-velocity baseline ---------------------------------------
    in_set_m = convert_covariance_matrices_to_set(in_cov, calib["level"]) / 1000.0   # [N,13] m
    horizon_t = [(k + 1) / FPS for k in range(K_P)]
    iso_centres, r_iso = compute_sara_predictions(
        last_input_poses=last_pose, prediction_horizon_times=horizon_t,
        v_human=V_HUMAN_ISO, measurement_uncertainty=in_set_m)               # [N,10,13] mm
    print(f"radii (mm): ours med={np.median(r_ours):.1f}  iso med={np.median(r_iso):.1f}")

    # ---- the 50-frame input history from the repo loader -----------------------------------
    # Spy on the loader's too-fast screen so we can carry the window provenance through the same
    # filter the dataset applies (the mask is not kept on the dataset object).
    import conformal_human_motion_prediction.datasets.h36m_motion_prediction as H
    _orig_fast = H.get_too_fast_human_movement
    _masks = []

    def _spy(*a, **k):
        out = _orig_fast(*a, **k)
        _masks.append(np.asarray(out))
        return out

    H.get_too_fast_human_movement = _spy
    try:
        _, _, test_loader = dataloader_from_string(
            "Human36mMotionDataset3DWithInputUncertainty", batch_size=16, shuffle=False, seed=420,
            download=False, data_path=str(REPO / "datasets"), max_target_speed=2.0)
    finally:
        H.get_too_fast_human_movement = _orig_fast
    ds = test_loader.dataset
    poses_all = np.asarray(ds.pose_data, dtype=np.float64).reshape(-1, K_I + K_P, J, 3)  # mm
    covs_all = np.asarray(ds.covariance_data, dtype=np.float64).reshape(-1, K_I + K_P, J, 3, 3)
    # The eval DataLoader drops the last partial batch (batch_size 16), so the results file is
    # a prefix of the dataset.  Truncate to the results length.
    n_ds = poses_all.shape[0]
    assert 0 <= n_ds - N < 16, (n_ds, N)
    poses_all, covs_all = poses_all[:N], covs_all[:N]
    # index alignment proof: the loader's target block IS the cloudpickle's targets
    err = np.abs(poses_all[:, K_I:] - tgt).max()
    print(f"dataset/results alignment: max |targets diff| = {err:.3e} mm")
    assert err < 1e-2, "dataset order does not match the results file"

    # ---- selection ---------------------------------------------------------------------------
    inp = poses_all[:, :K_I]                                                  # [N,50,13,3] mm
    step = np.linalg.norm(np.diff(inp, axis=1), axis=-1)                      # [N,49,13]
    travel = np.linalg.norm(inp[:, -1] - inp[:, 0], axis=-1).mean(axis=-1)    # [N] mm
    jerk = step.max(axis=(1, 2))                                              # [N] mm, spike screen
    mpjpe = np.linalg.norm(pred - tgt, axis=-1).mean(axis=(1, 2))             # [N] mm
    wrist = np.linalg.norm(inp[:, :, [5, 6]] - inp[:, :1, [5, 6]], axis=-1).max(axis=(1, 2))
    # truth inside both set families?
    d_ours = np.linalg.norm(pred - tgt, axis=-1)                              # [N,10,13]
    d_iso = np.linalg.norm(iso_centres - tgt, axis=-1)
    cov_ours = np.all(d_ours <= r_ours, axis=(1, 2))
    cov_iso = np.all(d_iso <= r_iso, axis=(1, 2))
    # per-window mean sphere volume (m^3)
    vol_ours = (4 / 3 * np.pi * (r_ours / 1000.0) ** 3)
    vol_iso = (4 / 3 * np.pi * (r_iso / 1000.0) ** 3)
    ratio = np.median(vol_iso, axis=(1, 2)) / np.median(vol_ours, axis=(1, 2))

    prov = window_provenance(REPO / "datasets")
    keep = None
    for m in _masks:
        if m.shape[0] == len(prov):
            keep = ~np.any(m, axis=tuple(range(1, m.ndim)))
            break
    if keep is not None and int(keep.sum()) == n_ds:
        actions = np.array([p[1] for p in prov])[keep][:N]
        starts = np.array([p[3] for p in prov])[keep][:N]
        offsets = np.array([p[2] for p in prov])[keep][:N]
        print(f"provenance: {len(prov)} raw -> {int(keep.sum())} kept, matches the dataset")
    else:
        print(f"provenance MISMATCH ({len(prov)} raw, dataset {n_ds}) -- labelling disabled")
        actions = np.array(["?"] * N)
        starts = offsets = np.zeros(N, np.int64)

    def pick(mask, score, label):
        idx = np.where(mask)[0]
        if idx.size == 0:
            raise SystemExit(f"no candidate for {label}")
        best = idx[np.argmax(score[idx])]
        print(f"  {label}: idx={best} action={actions[best]} travel={travel[best]:.0f}mm "
              f"mpjpe={mpjpe[best]:.1f}mm ratio={ratio[best]:.2f} "
              f"cov_ours={cov_ours[best]} cov_iso={cov_iso[best]}")
        return int(best)

    mp_lo, mp_hi = np.percentile(mpjpe, [20, 55])
    walk = np.array([a.split()[0] in ("Walking", "WalkingDog", "WalkingTogether", "Greeting")
                     for a in actions])
    base = (~is_ood) & (jerk < 120.0) & (mpjpe > mp_lo) & (mpjpe < mp_hi) & cov_ours & cov_iso
    print(f"base candidates: {int(base.sum())}  (walking/greeting: {int((base & walk).sum())})")

    # s05: the clearest, most legible motion -- big travel AND visible arm swing.
    s05_mask = base & walk & (travel > 150) & (wrist > 200)
    i05 = pick(s05_mask, travel / 1000.0 + wrist / 1000.0, "s05")
    # s09: same screens, but the window whose ours/ISO volume ratio sits closest to the
    # dataset-wide 7.61x headline, so the picture does not oversell the result.
    target_ratio = 0.685770 / 0.090136
    s09_mask = base & walk & (travel > 120) & (wrist > 150)
    i09 = pick(s09_mask, -np.abs(ratio - target_ratio), "s09")

    # ---- dataset-wide headline volumes (from the paper's own CSVs) -------------------------
    def csv_val(path, key):
        with open(path) as f:
            for row in csv.reader(f):
                if row and row[0] == key:
                    return float(row[1])
        raise KeyError(key)

    v_iso = csv_val(RESULTS / "coverage_stats_sara.csv", "overall_volume_p50_m3")
    v_ours = csv_val(RESULTS / "coverage_stats_conformal_prediction_sets.csv", "overall_volume_p50_m3")
    print(f"headline medians: ISO {v_iso:.6f} m^3, ours {v_ours:.6f} m^3, ratio {v_iso / v_ours:.2f}x")

    MEDIA.mkdir(parents=True, exist_ok=True)
    conn = np.asarray(CONNECTIONS_13, dtype=np.int32)
    for sid, i in (("s05", i05), ("s09", i09)):
        in_r = convert_covariance_matrices_to_set(covs_all[i, :K_I], calib["level"]) / 1000.0  # [50,13] m
        np.savez_compressed(
            MEDIA / f"{sid}_window.npz",
            index=np.int64(i),
            action=str(actions[i]),
            start_frame=np.int64(starts[i]),
            phase_offset=np.int64(offsets[i]),
            connections=conn,
            input_poses=(poses_all[i, :K_I] / 1000.0).astype(np.float32),     # [50,13,3] m
            input_radius=in_r.astype(np.float32),                             # [50,13] m
            pred=(pred[i] / 1000.0).astype(np.float32),                       # [10,13,3] m
            target=(tgt[i] / 1000.0).astype(np.float32),                      # [10,13,3] m
            last_pose=(last_pose[i] / 1000.0).astype(np.float32),             # [13,3] m
            r_ours=(r_ours[i] / 1000.0).astype(np.float32),                   # [10,13] m
            r_model=(r_model[i] / 1000.0).astype(np.float32),                 # [10,13] m
            sigma_max=(sigma_max[i] / 1000.0).astype(np.float32),             # [10,13] m
            alpha_eff=(r_ours[i] / sigma_max[i]).astype(np.float32),          # [10,13] effective alpha
            alpha_gauss=np.float32(np.median(r_model[i] / sigma_max[i])),
            r_iso=(r_iso[i] / 1000.0).astype(np.float32),                     # [10,13] m
            horizon_ms=np.asarray([(k + 1) * 1000.0 / FPS for k in range(K_P)], np.float32),
            mpjpe_mm=np.float32(mpjpe[i]),
            v_iso_median_m3=np.float32(v_iso),
            v_ours_median_m3=np.float32(v_ours),
            v_ratio=np.float32(v_iso / v_ours),
            v_h_max=np.float32(V_HUMAN_ISO),
            level=np.float32(calib["level"]),
        )
        print(f"wrote {MEDIA / f'{sid}_window.npz'}  (idx {i}, {actions[i]})")


if __name__ == "__main__":
    main()
