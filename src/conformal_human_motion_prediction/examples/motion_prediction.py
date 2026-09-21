"""This script evaluates a motion prediction model on the Human3.6M dataset."""

import os
from time import time
import argparse
import numpy as np
from sympy import per
import torch
import cloudpickle
from torch.utils.data import DataLoader
import jax.numpy as jnp
from tqdm import tqdm
from conformal_human_motion_prediction.pose_estimation.inference_helper import initialize_jax_models
from conformal_human_motion_prediction.motion_prediction.inference_helper import (
    calibrate_covariance_matrices, predict_poses, conformal_set_radius, load_conformal_calibrator,
)
from conformal_human_motion_prediction.utils.eval_utils import (
    compute_sara_predictions,
    convert_covariance_matrices_to_set,
    evaluate_uncertainty_coverage_with_covariance,
    print_coverage_stats,
    print_mpjpe_results,
    print_simple_coverage_stats_sara,
    save_coverage_stats,
    save_coverage_stats_sara,
    save_mpjpe_results,
    simple_coverage_stats_sara,
)
from conformal_human_motion_prediction.datasets import dataloader_from_string
from conformal_human_motion_prediction.models.dct_pose_transformer import DCTPoseTransformer
from conformal_human_motion_prediction.datasets.h36m_motion_prediction import Human36mMotionDataset3D
from conformal_human_motion_prediction.utils.visualization import visualize_motion_prediction
from conformal_human_motion_prediction.pose_estimation.h36m_settings import CONNECTIONS_13
from conformal_human_motion_prediction.motion_prediction.h36m_settings import (
    PREDICTION_HORIZON_LENGTH,
    N_JOINTS,
    OOD_THRESHOLD,
    COV_CALIBRATION_FACTORS,
    COV_CALIBRATION_CT,
    COV_CALIBRATION_IT,
    SARA_MEASUREMENT_UNCERTAINTY,
    SET_LIKELIHOOD,
    V_HUMAN_ISO,
)
from conformal_human_motion_prediction.utils.eval_utils import evaluate_pose_prediction_scores_np as evaluate_scores

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

BATCH_SIZE = 16
FPS = 25.0


def main():
    parser = argparse.ArgumentParser(description="3D Pose Estimation with OOD Detection")
    # parser.add_argument('--cache_dir', type=str, default='cache/', help='Cache directory with score functions')
    # parser.add_argument('--base_key', type=str, default=None, help='Base key for loading the OOD score functions')
    parser.add_argument("--data_path", type=str, default="datasets/", help="Path to datasets")
    parser.add_argument(
        "--model_save_path",
        type=str,
        default="models/motion_prediction/final_model/dct_pose_transformer.pickle",
        help="Path to saved models",
    )
    parser.add_argument("--split", type=str, default="validation", help="train, validation, or test")
    # parser.add_argument('--run_name', type=str, default='jax_resnet50_regressflow', help='Model run name')
    # parser.add_argument('--ood_threshold', type=float, default=OOD_THRESHOLD, help='OOD threshold')
    # parser.add_argument('--subject', type=str, default='S1', help='Subject ID (e.g., S1, S6)')
    # parser.add_argument('--action', type=str, default='WalkingDog', help='Action to visualize')
    # parser.add_argument('--camera_ids', type=str, nargs=2, default=['55011271', '60457274'], help='Camera IDs')
    # parser.add_argument('--max_frames', type=int, default=100, help='Maximum number of frames to process')
    parser.add_argument("--enable_ood", action="store_true", help="Enable OOD detection on left camera")
    parser.add_argument(
        "--motion_score_fn_path",
        type=str,
        default="models/ood_functions/dct_pose_transformer_randproj_score_fn.cloudpickle",
        help="Path to the OOD score function for the motion prediction.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="results/motion_prediction", help="Output directory for results"
    )
    parser.add_argument(
        "--dataset_name",
        default="Human36mMotionDataset3DWithInputUncertainty",
        help="Dataset name to\
        validate on. Choose from: Human36mMotionDataset3D and Human36mMotionDataset3DWithInputUncertainty (Default).",
    )
    parser.add_argument(
        "--max_target_speed",
        type=float,
        default=2.0,
        help="Too-fast target filter threshold in m/s (default 2.0 = ISO V_HUMAN_ISO). "
        "Raise it to keep faster motions; set <=0 or inf to disable the filter.",
    )
    parser.add_argument(
        "--conformal_calibrator",
        type=str,
        default="models/motion_prediction/conformal_calibration/conformal_calibrator.npz",
        help="Path to the conditional-conformal calibrator .npz. Falls back to affine "
        "calibration if the file is absent.",
    )
    parser.add_argument(
        "--conformal_calibrator_max",
        type=str,
        default=None,
        help="Optional path to a max-score (single-threshold) calibrator .npz, written by "
        "conformal_calibration --method max. When given, the ablation's coverage/volume CSVs "
        "(coverage_stats_conformal_prediction_sets_max[_ood_filtered].csv) are written alongside "
        "the conditional ones, so one evaluation run fills every row of the results table.",
    )
    parser.add_argument(
        "--conformal_calibrator_uncalibrated",
        type=str,
        default=None,
        help="Optional path to a no-calibration ablation calibrator .npz, written by "
        "conformal_calibration --method uncalibrated (r = sqrt(chi2_3(level)) * "
        "sqrt(lambda_max(cov)), i.e. trust the predicted covariance). When given, its "
        "coverage/volume CSVs (coverage_stats_conformal_prediction_sets_uncalibrated"
        "[_ood_filtered].csv) are written alongside the others.",
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 10)
    print("Evaluate Motion Prediction Model")
    print("=" * 10)
    print(f"Device: {device}")

    # Load model
    model_path = os.path.join(root_dir, args.model_save_path)
    motion_prediction_jit_fn, params, batch_stats = initialize_jax_models(checkpoint_path_jax=model_path)

    print("\nLoading OOD score functions...")
    pose_ood_score_fn = None
    motion_ood_score_fn = None
    if args.enable_ood:
        if not os.path.exists(args.motion_score_fn_path):
            raise FileNotFoundError(
                f"Motion model score functions file not found: {args.motion_score_fn_path}\n"
                f"Please run score_model.py first to generate the score functions."
            )
        with open(args.motion_score_fn_path, "rb") as f:
            motion_score_data = cloudpickle.load(f)
            motion_ood_score_fn = motion_score_data["score_fun"]

    # Load dataset
    print("\nLoading H36M dataset...")
    data_path = os.path.join(root_dir, args.data_path)  # , "H36M", "extracted")
    dataset_name = args.dataset_name
    train_loader, valid_loader, test_loader = dataloader_from_string(
        dataset_name,
        batch_size=BATCH_SIZE,
        shuffle=False,
        seed=420,
        download=False,
        data_path=data_path,
        max_target_speed=args.max_target_speed,
    )
    if args.split == "train":
        data_loader = train_loader
    elif args.split == "validation":
        data_loader = valid_loader
    elif args.split == "test":
        data_loader = test_loader
    else:
        raise NotImplementedError(f"Split {args.split} unknown.")
    # print(f"Loaded {len(dataset)} sequences.")

    # >>> Test dataset <<<
    print("\n" + "=" * 60)
    print(f"RESULTS for split {args.split}")
    print("=" * 60)
    predictions, targets, covariance_matrices, ood_scores, is_oods, last_input_poses = predict_poses(
        motion_prediction_jit_fn=motion_prediction_jit_fn,
        params=params,
        batch_stats=batch_stats,
        dataset_loader=data_loader,
        motion_ood_score_fn=motion_ood_score_fn,
        ood_threshold=OOD_THRESHOLD,
        device=device,
    )
    predictions = predictions.reshape(-1, PREDICTION_HORIZON_LENGTH, N_JOINTS, 3)
    targets = targets.reshape(-1, PREDICTION_HORIZON_LENGTH, N_JOINTS, 3)

    # Save all data
    os.makedirs(args.output_dir, exist_ok=True)
    results_cloudpickle_file = os.path.join(args.output_dir, f"motion_prediction_results_{args.split}.cloudpickle")

    motion_prediction_results = {
        "predictions": predictions,
        "targets": targets,
        "covariance_matrices": covariance_matrices,
        "ood_scores": ood_scores,
        "is_oods": is_oods,
        "last_input_poses": last_input_poses,
    }

    with open(results_cloudpickle_file, "wb") as f:
        cloudpickle.dump(motion_prediction_results, f)
        print(f"Saved results to {results_cloudpickle_file}")

    print("================================")
    print("Evaluating motion prediction.")
    print("================================")
    mpjpe, std_score, per_time_errors, per_time_stds, per_joint_errors, per_joint_std = evaluate_scores(
        predictions, targets
    )
    print("================================")
    print("Evaluating motion uncertainty prediction.")
    print("================================")
    coverage_stats, _ = evaluate_uncertainty_coverage_with_covariance(
        pred_poses=predictions, true_poses=targets, cov_matrices=covariance_matrices
    )
    print_mpjpe_results(mpjpe, per_time_errors, per_joint_errors)
    save_mpjpe_results(mpjpe, per_time_errors, per_joint_errors, split=args.split, output_dir=args.output_dir)
    print_coverage_stats(coverage_stats)
    save_coverage_stats(coverage_stats, split=args.split, output_dir=args.output_dir)

    print("================================")
    print("Evaluating conformal prediction sets.")
    print("================================")
    predictions = np.array(predictions)
    targets = np.array(targets)
    # Last-input-frame covariance (for the conditional-conformal set) before stripping the cov block.
    _li_full = np.array(last_input_poses)
    input_covariances = (_li_full[..., N_JOINTS * 3:N_JOINTS * 3 + N_JOINTS * 9].reshape(-1, N_JOINTS, 3, 3)
                         if _li_full.shape[-1] >= N_JOINTS * 3 + N_JOINTS * 9 else None)
    last_input_poses = _li_full[..., :N_JOINTS * 3].reshape(-1, N_JOINTS, 3)
    conformal_calibrator = None
    # Ablation calibrators, each keyed by the mode its .npz must declare:
    #   "max"          -> one conformal alpha_max over all joint-timesteps
    #   "uncalibrated" -> no calibration at all, alpha = sqrt(chi2_3(level))
    # Both share the apply rule alpha * sqrt(lambda_max(cov)); only the source of alpha differs.
    ablations = [
        ("max", args.conformal_calibrator_max, "max", "max-score"),
        ("uncalibrated", args.conformal_calibrator_uncalibrated, "uncalibrated", "no-calibration"),
    ]
    ablation_calibrators = {}
    if input_covariances is not None:
        cc_path = os.path.join(root_dir, args.conformal_calibrator)
        conformal_calibrator = load_conformal_calibrator(cc_path)
        if conformal_calibrator is not None:
            print(f"Using conditional-conformal calibrator {cc_path} "
                  f"(target {conformal_calibrator['level']:.4f}) for the conformal prediction sets.")
        for mode, path, _, label in ablations:
            if not path:
                continue
            abs_path = os.path.join(root_dir, path)
            calib = load_conformal_calibrator(abs_path)
            if calib is None:
                print(f"WARNING: {label} calibrator {abs_path} not found — skipping its "
                      f"ablation rows.")
                continue
            if calib.get("mode") != mode:
                raise SystemExit(
                    f"{abs_path} has mode={calib.get('mode')!r}, not {mode!r}. "
                    f"Fit it with conformal_calibration --method {mode}."
                )
            print(f"Using {label} calibrator {abs_path} (alpha={calib['alpha_max']:.4f}, target "
                  f"{calib['level']:.4f}) for the ablation rows.")
            ablation_calibrators[mode] = calib
    # Increase covariance for certain times and joints
    covariance_matrices_calibrated = calibrate_covariance_matrices(
        covariance_matrices=covariance_matrices,
        constant_time_factor=COV_CALIBRATION_CT,
        increase_time_factor=COV_CALIBRATION_IT,
        joint_calibration_factors=COV_CALIBRATION_FACTORS,
    )
    print("Coverage Stats After Calibration")
    # Compute coverage
    coverage_stats_calibrated, within_stds_calibrated = evaluate_uncertainty_coverage_with_covariance(
        pred_poses=predictions, true_poses=targets, cov_matrices=covariance_matrices_calibrated
    )
    print_coverage_stats(coverage_stats_calibrated)
    if conformal_calibrator is not None:
        # Conditional conformal set from RAW model covariance + input uncertainty (replaces affine).
        radius_conformal_prediction_sets = conformal_set_radius(
            np.array(covariance_matrices), input_covariances, conformal_calibrator
        )
    else:
        radius_conformal_prediction_sets = convert_covariance_matrices_to_set(
            np.array(covariance_matrices_calibrated), likelihood=SET_LIKELIHOOD
        )
    def report_set_coverage(radius, filename, header, keep=None):
        """Coverage/volume stats for one predicted set -> stdout + one CSV row file in output_dir."""
        pred_, rad_, tgt_ = (predictions, radius, targets) if keep is None else (
            predictions[keep], radius[keep], targets[keep])
        stats, _ = simple_coverage_stats_sara(predictions=pred_, radius=rad_, targets=tgt_)
        print(header)
        print_simple_coverage_stats_sara(stats)
        save_coverage_stats_sara(stats, filename=filename, output_dir=args.output_dir)
        return stats

    report_set_coverage(
        radius_conformal_prediction_sets,
        "coverage_stats_conformal_prediction_sets",
        f"Predicted spherical reachable set coverage stats for {SET_LIKELIHOOD} likelihood:",
    )

    # OOD-filtered conformal prediction sets ("ours with OOD filtered"): the same conformal sets,
    # evaluated only on the in-distribution samples (OOD score <= OOD_THRESHOLD). This is the third
    # row of the final results table; the two rows above use all samples ("without OOD filtered").
    is_oods_np = np.asarray(is_oods).astype(bool).ravel()
    keep_id = ~is_oods_np
    n_ood = int(is_oods_np.sum())
    print(f"OOD filtering: {n_ood}/{is_oods_np.size} samples flagged OOD "
          f"(score > OOD_THRESHOLD={OOD_THRESHOLD:g}); {int(keep_id.sum())} in-distribution kept.")
    if keep_id.any():
        report_set_coverage(
            radius_conformal_prediction_sets,
            "coverage_stats_conformal_prediction_sets_ood_filtered",
            "Predicted conformal set coverage stats (OOD-filtered, in-distribution only):",
            keep=keep_id,
        )
    else:
        print("No in-distribution samples after OOD filtering — skipping the OOD-filtered table.")

    # Ablations: the same sets formed by ONE global scalar alpha instead of the conditional q_hat
    # grid (r = alpha * sqrt(lambda_max(cov))) -- either the max-score conformal threshold or the
    # uncalibrated training-time factor sqrt(chi2_3(level)). Same two population variants as above,
    # so the table rows are directly comparable.
    for mode, _, suffix, label in ablations:
        calib = ablation_calibrators.get(mode)
        if calib is None:
            continue
        radius_ablation = conformal_set_radius(
            np.array(covariance_matrices), input_covariances, calib
        )
        report_set_coverage(
            radius_ablation,
            f"coverage_stats_conformal_prediction_sets_{suffix}",
            f"{label.capitalize()} ablation (alpha={calib['alpha_max']:.4f}) set coverage stats:",
        )
        if keep_id.any():
            report_set_coverage(
                radius_ablation,
                f"coverage_stats_conformal_prediction_sets_{suffix}_ood_filtered",
                f"{label.capitalize()} ablation set coverage stats (OOD-filtered, "
                f"in-distribution only):",
                keep=keep_id,
            )

    print("====================================")
    print("SARA Coverage Stats")
    print("====================================")

    dt = 1.0 / FPS
    prediction_horizon_times = [(t + 1) * dt for t in range(PREDICTION_HORIZON_LENGTH)]

    # Use per-joint input uncertainty from the last frame instead of fixed uncertainty
    if input_covariances is not None:
        # Convert input covariances to set radii (in m)
        input_uncertainty_m = convert_covariance_matrices_to_set(input_covariances, likelihood=SET_LIKELIHOOD) / 1000.0  # [N, J] in m
    else:
        # Fallback to fixed uncertainty if no covariance data
        input_uncertainty_m = SARA_MEASUREMENT_UNCERTAINTY

    # Evaluate SARA-style with per-joint input uncertainty
    sara_predictions, sara_radius = compute_sara_predictions(
        last_input_poses=last_input_poses,
        prediction_horizon_times=prediction_horizon_times,
        v_human=V_HUMAN_ISO,
        measurement_uncertainty=input_uncertainty_m,
    )
    coverage_stats_sara, _ = simple_coverage_stats_sara(
        predictions=sara_predictions,
        radius=sara_radius,
        targets=targets,
    )
    print("SARA simple velocity model coverage stats:")
    print_simple_coverage_stats_sara(coverage_stats_sara)
    save_coverage_stats_sara(coverage_stats_sara, filename="coverage_stats_sara", output_dir=args.output_dir)

    # Visualize a few samples
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)

    os.makedirs("eval_fixed", exist_ok=True)

    # Visualize best and worst predictions
    all_scores = np.linalg.norm(predictions - targets, axis=-1)
    per_sample_errors = np.mean(all_scores, axis=(1, 2))

    best_idx = np.argmin(per_sample_errors)
    worst_idx = np.argmax(per_sample_errors)
    median_idx = np.argsort(per_sample_errors)[len(per_sample_errors) // 2]

    for label, idx in [("best", best_idx), ("median", median_idx), ("worst", worst_idx)]:
        frame_idx = 4  # Middle frame
        pred_pose = predictions[idx, frame_idx].reshape(13, 3)
        targ_pose = targets[idx, frame_idx].reshape(13, 3)
        visualize_motion_prediction(
            pred_pose=np.array(pred_pose),
            target_pose=np.array(targ_pose),
            skeleton=CONNECTIONS_13,
            label=label,
            idx=idx,
            output_path=args.output_dir,
        )

        print(f"  Saved {label} prediction visualization")


if __name__ == "__main__":
    main()
