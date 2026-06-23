"""Script to tune the diagonal offset for covariance matrices to improve uncertainty coverage."""

import os
import numpy as np
import matplotlib.pyplot as plt
import cloudpickle
from scipy.stats import chi2
from pathlib import Path

from conformal_human_motion_prediction.motion_prediction.h36m_settings import MOCAP_MEASUREMENT_UNCERTAINTY, V_HUMAN_ISO
from conformal_human_motion_prediction.motion_prediction.inference_helper import calibrate_covariance_matrices, compute_human_occupancies
from conformal_human_motion_prediction.utils.eval_utils import compute_sara_predictions, convert_covariance_matrices_to_set, evaluate_uncertainty_coverage_with_covariance, get_too_fast_human_movement, print_coverage_stats, print_simple_coverage_stats_sara, save_coverage_stats_sara, simple_coverage_stats_sara


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
        "--mask_ood",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Filter out samples whose OOD score exceeds OOD_THRESHOLD"
    )
    parser.add_argument(
        "--mask_too_fast",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Filter out samples where the human moved faster than V_HUMAN_ISO"
    )

    args = parser.parse_args()

    if args.config == "rgbd_yolo":
        from conformal_human_motion_prediction.motion_prediction.rgbd_yolo_settings import (
            COV_CALIBRATION_FACTORS,
            PREDICTION_HORIZON_LENGTH,
            COV_CALIBRATION_CT,
            COV_CALIBRATION_IT,
            SET_LIKELIHOOD,
            SARA_MEASUREMENT_UNCERTAINTY,
            OOD_THRESHOLD,
            HUMAN_RADIUS
        )
    else:
        from conformal_human_motion_prediction.motion_prediction.h36m_settings import (
            COV_CALIBRATION_FACTORS,
            PREDICTION_HORIZON_LENGTH,
            COV_CALIBRATION_CT,
            COV_CALIBRATION_IT,
            SET_LIKELIHOOD,
            SARA_MEASUREMENT_UNCERTAINTY,
            OOD_THRESHOLD,
            HUMAN_RADIUS
        )

    output_dir = os.path.join(root_dir, args.output_dir)

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load results
    results_file = os.path.join(root_dir, args.results_file)
    print(f"Loading results from {results_file}...")
    with open(results_file, 'rb') as f:
        results = cloudpickle.load(f)

    n_plot = 1E20  # All predictions
    n_plot = min(results['predictions'].shape[0], n_plot)

    predictions = np.array(results['predictions'])[:n_plot]
    targets = np.array(results['targets'])[:n_plot]
    covariance_matrices = np.array(results['covariance_matrices'])[:n_plot]  # Shape: [N, n_t, n_j, 3, 3]
    ood_scores = np.array(results['ood_scores'])[:n_plot]
    last_input_poses = np.array(results['last_input_poses'])[:n_plot]

    N, T, J, _ = predictions.shape
    last_input_poses = np.reshape(last_input_poses[..., :J * 3], [N, J, 3])

    dt = 1.0 / args.fps

    def _filter(keep):
        return (
            predictions[keep],
            targets[keep],
            covariance_matrices[keep],
            ood_scores[keep],
            last_input_poses[keep],
        )

    # Filter out OOD samples
    if args.mask_ood:
        is_ood = ood_scores > OOD_THRESHOLD
        predictions, targets, covariance_matrices, ood_scores, last_input_poses = _filter(~is_ood)
        print(f"Filtered out {int(np.sum(is_ood))} OOD samples (score > {OOD_THRESHOLD}).")

    # Filter out samples, where the human moved faster than v_iso
    if args.mask_too_fast:
        too_fast = get_too_fast_human_movement(targets, V_HUMAN_ISO, dt)  # [N, n_t, n_j]
        mask = np.any(too_fast, axis=(1, 2))  # drop a sample if any joint at any time is too fast
        predictions, targets, covariance_matrices, ood_scores, last_input_poses = _filter(~mask)
        print(f"Filtered out {int(np.sum(mask))} samples with movement faster than {V_HUMAN_ISO} m/s.")


    # Compute ground truth human occupancies (positions are in mm, so convert radii m -> mm)
    human_radius = np.array(HUMAN_RADIUS) * 1000.0
    mocap_meas_uncertainty = MOCAP_MEASUREMENT_UNCERTAINTY * np.ones(targets.shape[:-1])
    true_human_occupancies = compute_human_occupancies(targets, mocap_meas_uncertainty, human_radius)
    print(f"Computed human occupancies shape: {true_human_occupancies.shape}")

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

    # Increase covariance for certain times and joints
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

    radius_predictions = convert_covariance_matrices_to_set(
        covariance_matrices,
        likelihood=SET_LIKELIHOOD
    )
    coverage_stats_predictions, _ = simple_coverage_stats_sara(
        predictions=predictions,
        radius=radius_predictions,
        targets=targets,
    )
    
    # Compute human reachable occupancies
    predicted_human_reachable_occupancies = compute_human_occupancies(predictions, radius_predictions, human_radius)
    
    print(f"Predicted spherical reachable set coverage stats for {SET_LIKELIHOOD} likelihood:")
    print_simple_coverage_stats_sara(coverage_stats_predictions)
    save_coverage_stats_sara(coverage_stats_predictions, filename="sara_coverage_predictions", output_dir=output_dir)

    print("====================================")
    print("SARA Coverage Stats")
    print("====================================")

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
