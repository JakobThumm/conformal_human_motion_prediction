#!/usr/bin/env python3
"""
Per-stage runtime benchmark of the full pipeline.

Replays the exact per-frame work of ``examples/eval_full_pipeline.py`` on H36M two-camera data
and times the six pipeline stages separately:

  1. pose_2d      -- 2D pose estimation (human detection + crop + RegressFlow regression), both cameras
  2. ood_pose     -- OOD scoring of the 2D pose estimator (sketched-Lanczos score fn), both cameras
  3. triangulate  -- 2D->3D linear triangulation + covariance propagation
  4. motion       -- DCT pose transformer motion prediction
  5. ood_motion   -- OOD scoring of the motion predictor
  6. set          -- conformal prediction-set radius computation

Each stage is synchronised (jax ``block_until_ready`` + ``torch.cuda.synchronize``) before its
timestamp is taken, so the numbers are device times rather than dispatch times. The first
``--warmup`` measurements of every stage are discarded (JIT compilation, cuDNN autotuning,
allocator growth).

Writes ``runtime_stages.csv`` to --output_dir; build the LaTeX table from it with
``generate_plots/generate_runtime_results.py``.
"""

import argparse
import csv
import os
from time import perf_counter

import cloudpickle
import jax
import jax.numpy as jnp
import numpy as np
import torch
from tqdm import tqdm

from conformal_human_motion_prediction.datasets.h36m import Human36mDatasetTwoCameras
from conformal_human_motion_prediction.motion_prediction.inference_helper import (
    calibrate_covariance_matrices,
    conformal_set_radius,
    load_conformal_calibrator,
)
from conformal_human_motion_prediction.ood_scoring.scores.lm_lanczos import load_score_functions_from_path
from conformal_human_motion_prediction.pose_estimation.inference_helper import (
    initialize_human_detector,
    initialize_jax_models,
)
from conformal_human_motion_prediction.pose_estimation.inference_helper_batched import (
    extract_bounding_box_images,
    fill_pose_buffer,
    joint_mapping,
    predict_pose,
)
from conformal_human_motion_prediction.pose_estimation.triangulation_helper import load_camera_parameters
from conformal_human_motion_prediction.utils.batched_transform_torch import (
    create_joint_covariance_batched,
    triangulate_points_with_covariance_batched,
)
from conformal_human_motion_prediction.utils.eval_utils import convert_covariance_matrices_to_set
from conformal_human_motion_prediction.utils.gpu_accelerated_utils import (
    transform_predictions_to_original_space_batched,
)

from conformal_human_motion_prediction.pose_estimation.h36m_settings import (
    MIRROR_13_JOINT_MODEL_MAP,
    OOD_THRESHOLD as POSE_OOD_THRESHOLD,
    YOLO_CONFIDENCE_THRESHOLD,
)
from conformal_human_motion_prediction.motion_prediction.h36m_settings import (
    COV_CALIBRATION_CT,
    COV_CALIBRATION_FACTORS,
    COV_CALIBRATION_IT,
    INPUT_HORIZON_LENGTH,
    N_CORRECT_POSES_REQUIRED,
    N_JOINTS,
    OOD_THRESHOLD as MOTION_OOD_THRESHOLD,
    PREDICTION_HORIZON_LENGTH,
    SET_LIKELIHOOD,
)

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))

# (key, latex label) in pipeline order.
STAGES = [
    ("pose_2d", r"2D pose estimation"),
    ("ood_pose", r"Pose OOD scoring"),
    ("triangulate", r"Triangulation"),
    ("motion", r"Motion prediction"),
    ("ood_motion", r"Motion OOD scoring"),
    ("set", r"Prediction-set computation"),
]


def sync():
    """Block until all queued GPU work (both jax's and torch's streams) has finished."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser(description="Per-stage runtime benchmark of the full pipeline")
    parser.add_argument('--data_path', type=str, default='datasets/')
    parser.add_argument('--pose_model_path', type=str, default='models/pose_estimation/jax_resnet50_regressflow')
    parser.add_argument('--pose_score_fn_path', type=str, default='models/ood_functions/jax_resnet18_regressflow_3joints_score_fn.cloudpickle')
    parser.add_argument('--motion_model_save_path', type=str, default='models/motion_prediction/final_model/dct_pose_transformer.pickle')
    parser.add_argument('--motion_score_fn_path', type=str, default='models/ood_functions/dct_pose_transformer_randproj_score_fn.cloudpickle')
    parser.add_argument('--conformal_calibrator', type=str, default='models/motion_prediction/conformal_calibration/conformal_calibrator.npz')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--camera_ids', type=str, nargs=2, default=['55011271', '60457274'])
    parser.add_argument('--subsample', type=int, default=2, help='Frame subsampling to match the camera frequency')
    parser.add_argument('--n_frames', type=int, default=400, help='Number of pipeline steps to time')
    parser.add_argument('--warmup', type=int, default=25, help='Measurements discarded per stage')
    parser.add_argument('--n_correct_poses_required', type=int, default=N_CORRECT_POSES_REQUIRED)
    parser.add_argument('--output_dir', type=str, default='results/final/runtime')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    device = args.device
    print("=" * 70)
    print("Full pipeline - per-stage runtime benchmark")
    print("=" * 70)

    # --- models -------------------------------------------------------------------------------
    pose_checkpoint_path = os.path.join(root_dir, args.pose_model_path)
    pose_jit_fn, pose_params, pose_batch_stats = initialize_jax_models(pose_checkpoint_path)
    motion_jit_fn, motion_params, motion_batch_stats = initialize_jax_models(
        os.path.join(root_dir, args.motion_model_save_path))
    human_detector, device_torch = initialize_human_detector(device)

    pose_ood_score_fn, _, _, _ = load_score_functions_from_path(args.pose_score_fn_path)
    with open(args.motion_score_fn_path, 'rb') as f:
        motion_ood_score_fn = cloudpickle.load(f)['score_fun']
    motion_calibrator = load_conformal_calibrator(os.path.join(root_dir, args.conformal_calibrator))
    print(f"[motion] conformal calibrator: {'loaded' if motion_calibrator is not None else 'absent -> affine'}")

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'
    print(f"[device] {gpu_name}")

    # --- data ---------------------------------------------------------------------------------
    dataset = Human36mDatasetTwoCameras(
        base_directory=os.path.join(root_dir, args.data_path, "H36M", "extracted"),
        split=args.split,
        camera_ids=args.camera_ids,
    )
    if len(dataset) == 0:
        raise RuntimeError("No data found - check the dataset path and camera IDs.")

    camera_parameters_path = os.path.join(os.path.dirname(pose_checkpoint_path), 'camera-parameters.json')

    timings = {key: [] for key, _ in STAGES}
    frame_wall = []          # end-to-end per-step wall time, as a cross-check on the stage sum
    n_done = 0

    for sample_id in range(len(dataset.data)):
        if n_done >= args.n_frames:
            break
        sample = dataset[sample_id]
        all_camera_frames = sample['all_camera_frames']
        _, _, projection_matrices = load_camera_parameters(
            camera_parameters_path, sample['subject'], args.camera_ids)
        P1 = torch.from_numpy(projection_matrices[args.camera_ids[0]]).to(device)
        P2 = torch.from_numpy(projection_matrices[args.camera_ids[1]]).to(device)

        points_3d_buffer = jnp.zeros([INPUT_HORIZON_LENGTH, N_JOINTS, 3])
        covariance_buffer = jnp.zeros([INPUT_HORIZON_LENGTH, N_JOINTS, 3, 3])
        pose_valid_buffer = jnp.zeros([INPUT_HORIZON_LENGTH])
        motion_prediction_buffer = jnp.zeros([PREDICTION_HORIZON_LENGTH, N_JOINTS, 3])
        motion_uncertainty_buffer = jnp.zeros([PREDICTION_HORIZON_LENGTH, N_JOINTS, 3, 3])

        frames_to_process = len(all_camera_frames[0]) - args.subsample * PREDICTION_HORIZON_LENGTH
        frame_counter = 0
        desc = f"Timing {sample['subject']}/{sample['action']}"
        for frame_idx in tqdm(range(0, frames_to_process, args.subsample), desc=desc):
            if n_done >= args.n_frames:
                break
            # Host-side frame fetch is dataset I/O, not pipeline work -> outside the timed region.
            left = np.asarray(all_camera_frames[0][frame_idx], dtype=np.float32)
            right = np.asarray(all_camera_frames[1][frame_idx], dtype=np.float32)
            if right.shape != left.shape:
                # The two H36M cameras differ in resolution -> crop as eval_full_pipeline does.
                right = right[:left.shape[0], :left.shape[1]]
            np_frames = np.stack([left, right], axis=0)

            step_t0 = perf_counter()

            # ---- 1) 2D pose estimation (both cameras in one batch of 2) --------------------
            t0 = perf_counter()
            frames_t = torch.from_numpy(np_frames).to(device_torch)
            if torch.mean(frames_t) > 2.0:
                frames_t = frames_t / 255.0
            bbox_struct = extract_bounding_box_images(
                full_image=frames_t,
                human_detector=human_detector,
                device_torch=device_torch,
                threshold=YOLO_CONFIDENCE_THRESHOLD,
            )
            bbox_image = bbox_struct['image']
            pred_joints_13, unc_13, cov_13 = predict_pose(
                bbox_image, pose_jit_fn, pose_params, pose_batch_stats, 17, device=device)
            pred_joints_13 = joint_mapping(pred_joints_13, MIRROR_13_JOINT_MODEL_MAP)
            unc_13 = joint_mapping(unc_13, MIRROR_13_JOINT_MODEL_MAP)
            cov_13 = joint_mapping(cov_13, MIRROR_13_JOINT_MODEL_MAP)
            scale_x, scale_y = bbox_struct['scale_factors_yolo']
            result = transform_predictions_to_original_space_batched(
                pred_joints_13, bbox_struct['trans'], scale_x, scale_y,
                uncertainties=unc_13, covariance=cov_13)
            B2, N, _ = result['keypoints'].shape
            joint_cov = torch.zeros((B2, N, 2, 2), device=device)
            joint_cov[:, :, 0, 0] = result['uncertainties'][:, :, 0] ** 2
            joint_cov[:, :, 0, 1] = result['covariance']
            joint_cov[:, :, 1, 0] = result['covariance']
            joint_cov[:, :, 1, 1] = result['uncertainties'][:, :, 1] ** 2
            sync()
            timings['pose_2d'].append(perf_counter() - t0)

            # ---- 2) pose OOD scoring -------------------------------------------------------
            t0 = perf_counter()
            jax_image = jnp.asarray(bbox_image) if isinstance(bbox_image, torch.Tensor) else bbox_image
            pose_ood_score = pose_ood_score_fn(jax_image)
            jax.block_until_ready(pose_ood_score)
            sync()
            timings['ood_pose'].append(perf_counter() - t0)
            pose_ood_score = torch.tensor(np.asarray(pose_ood_score), device=device)
            is_ood = pose_ood_score > POSE_OOD_THRESHOLD
            mask = bbox_struct['mask']

            # ---- 3) triangulation + covariance propagation ---------------------------------
            t0 = perf_counter()
            both_pose = result['keypoints'].reshape(1, 2, N_JOINTS, 2)
            both_cov = joint_cov.reshape(1, 2, N_JOINTS, 2, 2)
            both_unc = result['uncertainties'].reshape(1, 2, N_JOINTS, 2)
            C_2D = create_joint_covariance_batched(
                mapped_uncertainty_cam1=both_unc[:, 0],
                mapped_covariance_cam1=both_cov[:, 0][:, :, 0, 1],
                mapped_uncertainty_cam2=both_unc[:, 1],
                mapped_covariance_cam2=both_cov[:, 1][:, :, 0, 1],
                cross_covariance=torch.zeros((1, N_JOINTS, 2, 2), device=device),
            )
            points_3d, C_3d_all = triangulate_points_with_covariance_batched(
                both_pose[:, 0], both_pose[:, 1], P1, P2, C_2D)
            cov_valid = torch.all(torch.abs(C_3d_all) <= 1e5, dim=[1, 2, 3])
            sync()
            timings['triangulate'].append(perf_counter() - t0)

            frame_is_ood = bool(torch.any(is_ood)) or not bool(torch.all(cov_valid))
            human_detected = bool(torch.all(mask))
            points_3d_buffer, covariance_buffer, pose_valid_buffer, pose_buffer_good = fill_pose_buffer(
                points_3d_buffer=points_3d_buffer,
                covariance_buffer=covariance_buffer,
                pose_valid_buffer=pose_valid_buffer,
                points_3d=jnp.array(points_3d[0]),
                covariance=jnp.array(C_3d_all[0]),
                is_valid=(not frame_is_ood) and human_detected,
                motion_prediction_buffer=motion_prediction_buffer,
                motion_uncertainty_buffer=motion_uncertainty_buffer,
            )

            if frame_counter >= INPUT_HORIZON_LENGTH - 1:
                pose_input = points_3d_buffer.reshape([1, INPUT_HORIZON_LENGTH, N_JOINTS * 3])
                motion_input = jnp.concatenate([
                    pose_input,
                    covariance_buffer.reshape([1, INPUT_HORIZON_LENGTH, N_JOINTS * 3 * 3]),
                ], axis=-1)

                # ---- 4) motion prediction --------------------------------------------------
                t0 = perf_counter()
                if motion_batch_stats is not None:
                    motion_predicted, (motion_cov, _) = motion_jit_fn(
                        motion_params, motion_batch_stats, motion_input)
                else:
                    motion_predicted, (motion_cov, _) = motion_jit_fn(motion_params, motion_input)
                jax.block_until_ready((motion_predicted, motion_cov))
                sync()
                timings['motion'].append(perf_counter() - t0)

                # ---- 5) motion OOD scoring -------------------------------------------------
                t0 = perf_counter()
                motion_ood_score = motion_ood_score_fn(pose_input)
                jax.block_until_ready(motion_ood_score)
                sync()
                timings['ood_motion'].append(perf_counter() - t0)

                motion_predicted = motion_predicted.reshape(
                    -1, PREDICTION_HORIZON_LENGTH, N_JOINTS, 3)[0]
                motion_cov_uncalibrated = motion_cov[0]

                # ---- 6) prediction-set computation -----------------------------------------
                t0 = perf_counter()
                if motion_calibrator is not None:
                    input_cov = np.asarray(covariance_buffer).reshape(
                        INPUT_HORIZON_LENGTH, N_JOINTS, 3, 3)[-1]
                    motion_set_radius = conformal_set_radius(
                        motion_cov_uncalibrated, input_cov, motion_calibrator)
                else:
                    cov_cal = calibrate_covariance_matrices(
                        covariance_matrices=motion_cov_uncalibrated,
                        constant_time_factor=COV_CALIBRATION_CT,
                        increase_time_factor=COV_CALIBRATION_IT,
                        joint_calibration_factors=COV_CALIBRATION_FACTORS,
                    )
                    motion_set_radius = convert_covariance_matrices_to_set(
                        cov_cal, likelihood=SET_LIKELIHOOD)
                jax.block_until_ready(motion_set_radius) if isinstance(
                    motion_set_radius, jnp.ndarray) else None
                sync()
                timings['set'].append(perf_counter() - t0)

                frame_wall.append(perf_counter() - step_t0)
                n_done += 1

            frame_counter += 1

    write_results(timings, frame_wall, args, gpu_name)


def summarize(samples, warmup):
    """Mean / std / median / p95 in ms over the post-warmup samples."""
    a = np.asarray(samples[warmup:], dtype=np.float64) * 1e3
    if a.size == 0:
        raise RuntimeError(f"No measurements left after discarding {warmup} warmup samples.")
    return {
        "n": int(a.size),
        "mean_ms": float(a.mean()),
        "std_ms": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "median_ms": float(np.median(a)),
        "p95_ms": float(np.percentile(a, 95)),
        "first_ms": float(np.asarray(samples[:1], dtype=np.float64).ravel()[0] * 1e3),
    }


def write_results(timings, frame_wall, args, gpu_name):
    out_dir = os.path.join(root_dir, args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    stats = {key: summarize(timings[key], args.warmup) for key, _ in STAGES}
    total_mean = sum(s["mean_ms"] for s in stats.values())
    total_median = sum(s["median_ms"] for s in stats.values())
    wall = summarize(frame_wall, args.warmup)

    csv_path = os.path.join(out_dir, "runtime_stages.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stage", "label", "n", "mean_ms", "std_ms", "median_ms", "p95_ms",
                    "first_call_ms", "device"])
        for key, label in STAGES:
            s = stats[key]
            w.writerow([key, label, s["n"], f"{s['mean_ms']:.4f}", f"{s['std_ms']:.4f}",
                        f"{s['median_ms']:.4f}", f"{s['p95_ms']:.4f}", f"{s['first_ms']:.2f}",
                        gpu_name])
        # Sum of the stage columns -- the quantity the paper reports as the cycle time.
        w.writerow(["total", "Total", wall["n"], f"{total_mean:.4f}", "",
                    f"{total_median:.4f}", "", "", gpu_name])
        # Independent end-to-end wall clock per step; agreement with `total` validates that the
        # per-stage synchronisation is not hiding or double-counting device work.
        w.writerow(["measured_wall", "Measured end-to-end", wall["n"], f"{wall['mean_ms']:.4f}",
                    f"{wall['std_ms']:.4f}", f"{wall['median_ms']:.4f}", f"{wall['p95_ms']:.4f}",
                    f"{wall['first_ms']:.2f}", gpu_name])
    print(f"\nSaved per-stage timings to {csv_path}")

    for key, label in STAGES:
        s = stats[key]
        print(f"  {label:<28s} mean {s['mean_ms']:7.2f} ms  median {s['median_ms']:7.2f} ms  "
              f"(first call {s['first_ms']:.0f} ms)")
    print(f"  {'Total (stage sum)':<28s} mean {total_mean:7.2f} ms  median {total_median:7.2f} ms"
          f"  -> {1000.0 / total_mean:.1f} Hz")
    print(f"  {'Measured end-to-end':<28s} mean {wall['mean_ms']:7.2f} ms  "
          f"median {wall['median_ms']:7.2f} ms")
    print(f"\nBuild the LaTeX table with:\n  python -m "
          f"conformal_human_motion_prediction.generate_plots.generate_runtime_results "
          f"--results_dir {args.output_dir}")


if __name__ == "__main__":
    main()
