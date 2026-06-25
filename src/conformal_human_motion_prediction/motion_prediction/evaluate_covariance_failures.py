"""Investigate the conformal-prediction coverage failures (the ~0.2% of joint-frames whose mocap
target falls OUTSIDE the predicted spherical set).

Companion to ``evaluate_covariance.py``: that script reports overall coverage; this one zooms into
the *uncovered* joint-frames and tests a specific hypothesis --

    the conformal set mainly fails when the LAST INPUT pose had high measurement uncertainty.

The model is fed the per-keypoint input covariance (the pose estimator's reported uncertainty); the
last input frame's covariance is stored in ``last_input_poses[:, J*3:]`` as [J,3,3]. We turn it into
a per-joint input-uncertainty radius (the same 3D conformal set conversion used for the output) and
ask whether coverage collapses as that input uncertainty grows.

We report, per joint and aggregated:
  * coverage as a function of input-uncertainty bins (the headline test),
  * the input-uncertainty distribution of covered vs. failed joint-frames (+ enrichment / lift),
  * per-joint coverage vs. per-joint input uncertainty (to spot joints that fail for OTHER reasons,
    e.g. under-calibrated volume rather than noisy input),
and save plots + a CSV. Poses are in mm; radii are reported in metres.

Run::

    .venv/bin/python -m conformal_human_motion_prediction.motion_prediction.evaluate_covariance_failures \
        --results_file results/motion_prediction/motion_prediction_results_validation.cloudpickle
"""
import argparse
import os
from pathlib import Path

import cloudpickle
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from conformal_human_motion_prediction.motion_prediction.inference_helper import (
    calibrate_covariance_matrices, conformal_set_radius, load_conformal_calibrator,
    DEFAULT_CONFORMAL_CALIBRATOR,
)
from conformal_human_motion_prediction.utils.eval_utils import convert_covariance_matrices_to_set
from conformal_human_motion_prediction.pose_estimation.h36m_settings import JOINT_NAMES_13

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))


def coverage_by_bin(values, covered, edges):
    """Coverage rate within each [edges[i], edges[i+1]) bin of ``values``. Returns list of rows."""
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (values >= lo) & (values < hi)
        n = int(m.sum())
        if n == 0:
            continue
        rows.append((lo, hi, n, float(covered[m].mean()), float(np.median(values[m]))))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results_file", type=str,
                        default="results/motion_prediction/motion_prediction_results_validation.cloudpickle")
    parser.add_argument("--output_dir", type=str, default="results/motion_prediction/coverage_failures")
    parser.add_argument("--config", type=str, default="h36m", choices=["h36m", "rgbd_yolo"])
    parser.add_argument("--calibrate", action=argparse.BooleanOptionalAction, default=True,
                        help="Apply the affine covariance calibration before forming the set "
                             "(fallback when --conformal_calibrator is unset/missing).")
    parser.add_argument("--conformal_calibrator", type=str, default=DEFAULT_CONFORMAL_CALIBRATOR,
                        help="Path to a conditional-conformal calibrator .npz. When present, the set "
                             "is formed with it (replacing the affine calibration) -- so this script "
                             "doubles as a check that the deployed conformal set covers uniformly. "
                             "Set to '' / 'none' (or a missing path) to evaluate the affine set.")
    args = parser.parse_args()

    if args.config == "rgbd_yolo":
        from conformal_human_motion_prediction.motion_prediction.rgbd_yolo_settings import (
            COV_CALIBRATION_FACTORS, COV_CALIBRATION_CT, COV_CALIBRATION_IT, SET_LIKELIHOOD)
    else:
        from conformal_human_motion_prediction.motion_prediction.h36m_settings import (
            COV_CALIBRATION_FACTORS, COV_CALIBRATION_CT, COV_CALIBRATION_IT, SET_LIKELIHOOD)

    output_dir = os.path.join(root_dir, args.output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    results_file = os.path.join(root_dir, args.results_file)
    print(f"Loading results from {results_file} ...")
    with open(results_file, "rb") as f:
        results = cloudpickle.load(f)

    predictions = np.asarray(results["predictions"], dtype=np.float64)        # [N,T,J,3] mm
    targets = np.asarray(results["targets"], dtype=np.float64)
    cov = np.asarray(results["covariance_matrices"], dtype=np.float64)         # [N,T,J,3,3] mm^2
    last_input = np.asarray(results["last_input_poses"], dtype=np.float64)     # [N, J*3 + J*9]
    N, T, J, _ = predictions.shape
    print(f"  N={N} samples, T={T} frames, J={J} joints")

    # ---- input uncertainty: the last input frame's per-joint covariance -> 3D set radius ---------
    if last_input.shape[1] < J * 3 + J * 9:
        raise SystemExit("last_input_poses has no covariance block -- results must come from the "
                         "input_uncertainty pipeline (the model fed per-keypoint covariances).")
    in_cov = last_input[:, J * 3: J * 3 + J * 9].reshape(N, J, 3, 3)           # [N,J,3,3] mm^2
    in_set = convert_covariance_matrices_to_set(in_cov, SET_LIKELIHOOD) / 1000.0   # [N,J] m

    # ---- output set & coverage failures (mirrors simple_coverage_stats_sara) ----------------------
    cc = args.conformal_calibrator
    if cc and cc.lower() != "none" and os.path.exists(cc):
        calibrator = load_conformal_calibrator(cc)
        radius = conformal_set_radius(cov, in_cov, calibrator)                      # [N,T,J] mm
        print(f"  set: conditional conformal ({cc}, target {calibrator['level']:.4f})")
    else:
        if args.calibrate:
            cov = calibrate_covariance_matrices(cov, COV_CALIBRATION_CT, COV_CALIBRATION_IT,
                                                COV_CALIBRATION_FACTORS)
        radius = convert_covariance_matrices_to_set(cov, likelihood=SET_LIKELIHOOD)  # [N,T,J] mm
        print(f"  set: affine calibration (conformal calibrator: {cc or 'disabled'})")
    distances = np.linalg.norm(predictions - targets, axis=-1)                     # [N,T,J] mm
    within = distances <= radius                                                  # [N,T,J] covered

    # Valid joint-frames only (drop all-zero predicted/target frames, as the coverage stats do).
    mask_pred = np.all(predictions == 0.0, axis=(2, 3))
    mask_tgt = np.all(targets == 0.0, axis=(2, 3))
    valid_TF = ~(mask_pred | mask_tgt)                                            # [N,T]
    valid = np.repeat(valid_TF[:, :, None], J, axis=2)                            # [N,T,J]

    in_set_TJ = np.repeat(in_set[:, None, :], T, axis=1)                          # [N,T,J] (const over T)

    cov_flat = within[valid]
    inunc_flat = in_set_TJ[valid]
    dist_flat = distances[valid] / 1000.0
    rad_flat = radius[valid] / 1000.0
    n_total = cov_flat.size
    n_fail = int((~cov_flat).sum())
    print(f"\nValid joint-frames: {n_total:,}   covered: {n_total - n_fail:,}   "
          f"FAILED (target outside set): {n_fail:,}  ({100 * n_fail / n_total:.3f}%)")

    failed = ~cov_flat
    print("\n=================== Hypothesis: failures concentrate at HIGH input uncertainty ===================")
    print(f"  input-uncertainty radius (last input pose) -- median over ...")
    print(f"    covered joint-frames : {np.median(inunc_flat[cov_flat]):.3f} m   "
          f"mean {inunc_flat[cov_flat].mean():.3f} m")
    print(f"    FAILED  joint-frames : {np.median(inunc_flat[failed]):.3f} m   "
          f"mean {inunc_flat[failed].mean():.3f} m")

    # Coverage as a function of input-uncertainty bins (the headline).
    edges = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.0, np.inf]
    print(f"\n  Coverage vs input-uncertainty bin (target = {SET_LIKELIHOOD:.1%}):")
    print(f"    {'input unc bin (m)':>20} {'n':>11} {'coverage':>9} {'fail rate':>10}")
    bin_rows = coverage_by_bin(inunc_flat, cov_flat, edges)
    for lo, hi, n, cvg, med in bin_rows:
        label = f"[{lo:.2f},{hi:.2f})" if np.isfinite(hi) else f">={lo:.2f}"
        print(f"    {label:>20} {n:>11,} {100 * cvg:>8.2f}% {100 * (1 - cvg):>9.3f}%")

    # Input -> output uncertainty coupling (M1): if the head propagated input uncertainty, the
    # output set radius would track the input-uncertainty radius. Baseline correlation is ~0.26.
    if inunc_flat.std() > 0 and rad_flat.std() > 0:
        corr = float(np.corrcoef(inunc_flat, rad_flat)[0, 1])
    else:
        corr = float("nan")
    print(f"\n  Input->output uncertainty correlation (Pearson, output set radius vs input "
          f"unc radius): {corr:.3f}   [higher = input uncertainty propagated; baseline ~0.26]")

    # Enrichment / lift: are failures over-represented at high input uncertainty?
    for p in (90, 95, 99):
        thr = np.percentile(inunc_flat, p)
        share_fail = float((inunc_flat[failed] > thr).mean())
        share_all = float((inunc_flat > thr).mean())
        print(f"\n  input unc > p{p} ({thr:.3f} m): {100 * share_fail:.1f}% of FAILURES vs "
              f"{100 * share_all:.1f}% of all joint-frames  -> lift x{share_fail / max(share_all, 1e-9):.1f}")

    # ---- per-joint: coverage vs input uncertainty (spot non-input-driven failures) ---------------
    print("\n  Per-joint (sorted by coverage, worst first):")
    print(f"    {'joint':>10} {'coverage':>9} {'mean in-unc':>12} {'mean dist':>10} {'mean radius':>12}")
    pj = []
    for j in range(J):
        vj = valid[:, :, j]
        cj = within[:, :, j][vj]
        ij = in_set_TJ[:, :, j][vj]
        dj = distances[:, :, j][vj] / 1000.0
        rj = radius[:, :, j][vj] / 1000.0
        pj.append((j, float(cj.mean()), float(ij.mean()), float(dj.mean()), float(rj.mean())))
    for j, cvg, iu, dd, rr in sorted(pj, key=lambda r: r[1]):
        print(f"    {JOINT_NAMES_13[j]:>10} {100 * cvg:>8.2f}% {iu:>12.3f} {dd:>10.3f} {rr:>12.3f}")
    print("=" * 96)

    # ----------------------------------------------------------------- figures
    fig, ax = plt.subplots(2, 2, figsize=(15, 11))

    # (1) coverage vs input-uncertainty bin
    a = ax[0, 0]
    labels = [f"[{lo:.2f},{hi:.2f})" if np.isfinite(hi) else f">={lo:.2f}" for lo, hi, *_ in bin_rows]
    cvgs = [100 * r[3] for r in bin_rows]
    ns = [r[2] for r in bin_rows]
    bars = a.bar(range(len(bin_rows)), cvgs, color="#4c78a8")
    a.axhline(100 * SET_LIKELIHOOD, color="k", ls="--", lw=1, label=f"target {100 * SET_LIKELIHOOD:.1f}%")
    for i, (b, n) in enumerate(zip(bars, ns)):
        a.text(b.get_x() + b.get_width() / 2, b.get_height(), f"n={n:,}", ha="center", va="bottom",
               fontsize=7, rotation=90)
    a.set_xticks(range(len(bin_rows)))
    a.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    a.set_ylim(min(cvgs) - 2, 100.5)
    a.set_title("Coverage vs last-input-pose uncertainty")
    a.set_xlabel("input uncertainty radius (m)"); a.set_ylabel("coverage (%)"); a.legend(fontsize=8)

    # (2) input-uncertainty distribution: covered vs failed
    a = ax[0, 1]
    bins = np.linspace(0, np.percentile(inunc_flat, 99.9), 60)
    a.hist(inunc_flat[cov_flat], bins=bins, density=True, alpha=0.6, color="#4c78a8", label="covered")
    a.hist(inunc_flat[failed], bins=bins, density=True, alpha=0.6, color="#e45756", label="failed")
    a.set_yscale("log")
    a.set_title("Input-uncertainty distribution (density)")
    a.set_xlabel("input uncertainty radius (m)"); a.set_ylabel("density (log)"); a.legend(fontsize=8)

    # (3) per-joint coverage vs mean input uncertainty
    a = ax[1, 0]
    jx = [r[2] for r in pj]; jy = [100 * r[1] for r in pj]
    a.scatter(jx, jy, c="#e45756")
    for j, cvg, iu, *_ in pj:
        a.annotate(JOINT_NAMES_13[j], (iu, 100 * cvg), fontsize=7,
                   xytext=(3, 3), textcoords="offset points")
    a.axhline(100 * SET_LIKELIHOOD, color="k", ls="--", lw=1, label=f"target {100 * SET_LIKELIHOOD:.1f}%")
    a.set_title("Per-joint coverage vs mean input uncertainty")
    a.set_xlabel("mean input uncertainty radius (m)"); a.set_ylabel("coverage (%)"); a.legend(fontsize=8)

    # (4) distance vs input uncertainty (subsampled), set-radius reference
    a = ax[1, 1]
    rng = np.random.default_rng(0)
    take = rng.choice(n_total, size=min(40000, n_total), replace=False)
    a.scatter(inunc_flat[take][cov_flat[take]], dist_flat[take][cov_flat[take]], s=4, alpha=0.3,
              c="#4c78a8", label="covered")
    a.scatter(inunc_flat[take][~cov_flat[take]], dist_flat[take][~cov_flat[take]], s=8, alpha=0.6,
              c="#e45756", label="failed")
    lim = np.percentile(inunc_flat, 99.5)
    a.plot([0, lim], [0, lim], "k:", lw=1, label="dist = input unc")
    a.set_xlim(0, lim); a.set_ylim(0, np.percentile(dist_flat, 99.8))
    a.set_title("Prediction error vs input uncertainty")
    a.set_xlabel("input uncertainty radius (m)"); a.set_ylabel("prediction error ||pred-true|| (m)")
    a.legend(fontsize=8)

    fig.suptitle(f"Conformal coverage failures vs input uncertainty  "
                 f"(failed {n_fail:,}/{n_total:,} = {100 * n_fail / n_total:.3f}%)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig_path = os.path.join(output_dir, "coverage_failures_vs_input_uncertainty.png")
    fig.savefig(fig_path, dpi=130)
    print(f"\nSaved figure to {fig_path}")

    # per-joint CSV
    csv_path = os.path.join(output_dir, "per_joint_failures.csv")
    with open(csv_path, "w") as f:
        f.write("joint,name,coverage_percent,mean_input_unc_m,mean_pred_err_m,mean_radius_m\n")
        for j, cvg, iu, dd, rr in pj:
            f.write(f"{j},{JOINT_NAMES_13[j]},{100 * cvg:.3f},{iu:.4f},{dd:.4f},{rr:.4f}\n")
    print(f"Saved per-joint CSV to {csv_path}")


if __name__ == "__main__":
    main()
