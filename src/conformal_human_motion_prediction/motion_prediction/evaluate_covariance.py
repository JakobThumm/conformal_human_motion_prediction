"""Script to tune the diagonal offset for covariance matrices to improve uncertainty coverage."""

import os
import numpy as np
import matplotlib.pyplot as plt
import cloudpickle
from scipy.stats import chi2
from pathlib import Path

from conformal_human_motion_prediction.motion_prediction.h36m_settings import V_HUMAN_ISO
from conformal_human_motion_prediction.motion_prediction.inference_helper import (
    calibrate_covariance_matrices, conformal_set_radius, load_conformal_calibrator,
    DEFAULT_CONFORMAL_CALIBRATOR,
)
from conformal_human_motion_prediction.utils.eval_utils import compute_sara_predictions, convert_covariance_matrices_to_set, evaluate_uncertainty_coverage_with_covariance, print_coverage_stats, print_simple_coverage_stats_sara, save_coverage_stats_sara, simple_coverage_stats_sara


root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Tune covariance matrix diagonal offset for better coverage")
    parser.add_argument(
        "--results_file",
        type=str,
        default="results/motion_prediction/motion_prediction_results_train.cloudpickle",
        help="Path to motion prediction results file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/motion_prediction/coverage_tuning",
        help="Output directory for plots"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="h36m",
        choices=["h36m", "rgbd_yolo"],
        help="Settings config to use: 'h36m' for Human3.6M, 'rgbd_yolo' for RGB-D YOLO pipeline"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=25.0,
        help="The FPS of the camera"
    )
    parser.add_argument(
        "--conformal_calibrator",
        type=str,
        default=DEFAULT_CONFORMAL_CALIBRATOR,
        help="Path to a conditional-conformal calibrator .npz (from conformal_calibration). When "
             "present, the reported spherical reachable set uses it instead of the affine "
             "calibration. Set to '' / 'none' (or a missing path) to use the affine calibration.",
    )
    parser.add_argument(
        "--calibrate", action=argparse.BooleanOptionalAction, default=True,
        help="Apply the affine covariance calibration (legacy NLL-only models). Use --no-calibrate "
             "for P2-pinball self-calibrated models (e.g. cov_p2p4), whose RAW covariance already "
             "yields a target-coverage radius -- affine would double-calibrate it.",
    )

    args = parser.parse_args()

    if args.config == "rgbd_yolo":
        from conformal_human_motion_prediction.motion_prediction.rgbd_yolo_settings import (
            COV_CALIBRATION_FACTORS,
            PREDICTION_HORIZON_LENGTH,
            COV_CALIBRATION_CT,
            COV_CALIBRATION_IT,
            SET_LIKELIHOOD,
            SARA_MEASUREMENT_UNCERTAINTY
        )
    else:
        from conformal_human_motion_prediction.motion_prediction.h36m_settings import (
            COV_CALIBRATION_FACTORS,
            PREDICTION_HORIZON_LENGTH,
            COV_CALIBRATION_CT,
            COV_CALIBRATION_IT,
            SET_LIKELIHOOD,
            SARA_MEASUREMENT_UNCERTAINTY
        )

    output_dir = os.path.join(root_dir, args.output_dir)

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load results
    results_file = os.path.join(root_dir, args.results_file)
    print(f"Loading results from {results_file}...")
    with open(results_file, 'rb') as f:
        results = cloudpickle.load(f)

    n_plot = 100000
    n_plot = min(results['predictions'].shape[0], n_plot)

    predictions = np.array(results['predictions'])[:n_plot]
    targets = np.array(results['targets'])[:n_plot]
    covariance_matrices = np.array(results['covariance_matrices'])[:n_plot]  # Shape: [N, n_t, n_j, 3, 3]
    ood_scores = np.array(results['ood_scores'])[:n_plot]
    last_input_poses = np.array(results['last_input_poses'])[:n_plot]

    N, T, J, _ = predictions.shape
    # Keep the RAW model covariance and the last-input-frame covariance for the conditional-conformal
    # set (computed from raw cov + input uncertainty), before the affine calibration overwrites cov
    # and before the cov block is stripped from last_input_poses below.
    raw_covariance_matrices = covariance_matrices.copy()
    input_covariances = (np.reshape(last_input_poses[..., J*3:J*3 + J*9], [N, J, 3, 3])
                         if last_input_poses.shape[-1] >= J*3 + J*9 else None)
    last_input_poses = np.reshape(last_input_poses[..., :J*3], [N, J, 3])

    calibrator = None
    cc = args.conformal_calibrator
    if cc and cc.lower() != "none" and os.path.exists(cc) and input_covariances is not None:
        calibrator = load_conformal_calibrator(cc)
        print(f"Using conditional-conformal calibrator {cc} (target {calibrator['level']:.4f}) "
              f"for the spherical reachable set.")
    else:
        print("Using affine calibration for the spherical reachable set "
              f"(conformal calibrator: {cc or 'disabled'}).")

    # Filter out OOD samples
    # is_ood = ood_scores > 6e5
    # predictions = predictions[~is_ood]
    # targets = targets[~is_ood]
    # covariance_matrices = covariance_matrices[~is_ood]
    # ood_scores = ood_scores[~is_ood]

    print(f"Loaded predictions shape: {predictions.shape}")
    print(f"Loaded targets shape: {targets.shape}")
    print(f"Loaded covariance matrices shape: {covariance_matrices.shape}")

    print("====================================")
    print("Coverage Stats Before Calibration")
    print("====================================")
    # Compute coverage
    coverage_stats, within_stds = evaluate_uncertainty_coverage_with_covariance(
        pred_poses=predictions, true_poses=targets, cov_matrices=covariance_matrices
    )
    # Print coverage statistics
    print_coverage_stats(coverage_stats)

    # Increase covariance for certain times and joints (affine calibration; legacy models only --
    # a P2-pinball self-calibrated model is already at target from the raw covariance).
    if args.calibrate:
        covariance_matrices = calibrate_covariance_matrices(
            covariance_matrices=covariance_matrices,
            constant_time_factor=COV_CALIBRATION_CT,
            increase_time_factor=COV_CALIBRATION_IT,
            joint_calibration_factors=COV_CALIBRATION_FACTORS
        )
    # Generate n_std range
    n_std_range = [1, 2, 3, 4]

    # Compute ideal coverage
    ideal_coverages = [68.2, 95.4, 99.7, 99.99]

    print("====================================")
    print("Coverage Stats After Calibration")
    print("====================================")
    # Compute coverage
    coverage_stats, within_stds = evaluate_uncertainty_coverage_with_covariance(
        pred_poses=predictions, true_poses=targets, cov_matrices=covariance_matrices
    )

    # Print coverage statistics
    print_coverage_stats(coverage_stats)

    if calibrator is not None:
        # Conditional conformal: r = max(r_model(raw cov) + q_hat(joint, frame, input-unc bin), 0).
        radius_predictions = conformal_set_radius(raw_covariance_matrices, input_covariances, calibrator)
    else:
        radius_predictions = convert_covariance_matrices_to_set(
            covariance_matrices,
            likelihood=SET_LIKELIHOOD
        )
    coverage_stats_predictions, _ = simple_coverage_stats_sara(
        predictions=predictions,
        radius=radius_predictions,
        targets=targets,
    )
    _set_level = calibrator["level"] if calibrator is not None else SET_LIKELIHOOD
    _set_kind = "conditional conformal" if calibrator is not None else "affine"
    print(f"Predicted spherical reachable set coverage stats ({_set_kind}) for {_set_level} likelihood:")
    print_simple_coverage_stats_sara(coverage_stats_predictions)
    save_coverage_stats_sara(coverage_stats_predictions, filename="sara_coverage_predictions", output_dir=output_dir)

    print("====================================")
    print("SARA Coverage Stats")
    print("====================================")

    dt = 1.0 / args.fps
    prediction_horizon_times = [(t + 1) * dt for t in range(PREDICTION_HORIZON_LENGTH)]

    # Evaluate SARA-style
    sara_predictions, sara_radius = compute_sara_predictions(
        last_input_poses=last_input_poses,
        prediction_horizon_times=prediction_horizon_times,
        v_human=V_HUMAN_ISO,
        measurement_uncertainty=SARA_MEASUREMENT_UNCERTAINTY
    )
    coverage_stats_sara, _ = simple_coverage_stats_sara(
        predictions=sara_predictions,
        radius=sara_radius,
        targets=targets,
    )
    print("SARA simple velocity model coverage stats:")
    print_simple_coverage_stats_sara(coverage_stats_sara)
    save_coverage_stats_sara(coverage_stats_sara, filename="sara_coverage_sara", output_dir=output_dir)

    # Plot predicted uncertainty increase over frame for each joint
    plt.figure(figsize=(12, 8))
    uncertainties = np.trace(covariance_matrices, axis1=3, axis2=4)  # [N, n_t, n_j]
    uncertainty_increase = uncertainties * (1 / uncertainties[:, 0, ...])[:, None, :]
    for i in range(uncertainty_increase.shape[2]):
        plt.plot(np.arange(T), np.mean(uncertainty_increase[:, :, i], axis=0), label=f"Joint {i}")
    plt.xlabel("Time point")
    plt.ylabel("Uncertainty increase")
    plt.legend()
    output_file = os.path.join(output_dir, 'uncertainty_increase_over_time.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved plot to {output_file}")
    plt.close

    # Plot predicted uncertainty over OOD score
    within_std_stat = np.mean(np.sum(np.array(within_stds), axis=0), axis=(1, 2))
    plt.figure(figsize=(12, 8))
    predicted_uncertainty_all = np.mean(np.trace(covariance_matrices, axis1=3, axis2=4), axis=(1, 2))
    # Reduce plotting to critical
    critical_points = within_std_stat <= 10
    ood_scores_plot = ood_scores[critical_points]
    predicted_uncertainty_all_plot = predicted_uncertainty_all[critical_points]
    c_plot = within_std_stat[critical_points]
    plt.scatter(ood_scores_plot, predicted_uncertainty_all_plot, c=c_plot, cmap='viridis')
    plt.colorbar()
    plt.xlabel("OOD score")
    plt.ylabel("Mean trace of cov. matrices")
    output_file = os.path.join(output_dir, 'uncertainty_over_ood_all.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved plot to {output_file}")
    plt.close

    # Plot predicted uncertainty over OOD score for every joint
    print("\nDone!")


if __name__ == "__main__":
    main()
