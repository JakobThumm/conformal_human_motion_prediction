"""Helper functions for motion prediction inference."""
from typing import Optional, Sequence, Tuple, Union
from sympy import ShapeError
from tqdm import tqdm
from time import time
import jax.numpy as jnp
import numpy as np

from conformal_human_motion_prediction.pose_estimation.inference_helper_batched import update_motion_prediction_buffer
from conformal_human_motion_prediction.utils.eval_utils import convert_covariance_matrices_to_set
from typing import List


def predict_poses(
    motion_prediction_jit_fn,
    params,
    batch_stats,
    dataset_loader,
    motion_ood_score_fn=None,
    ood_threshold=np.inf,
    max_batches=np.inf,
    device="cuda"
):
    """Evaluate the motion prediction model.

    Args:
        motion_prediction_jit_fn: JIT-compiled JAX function for motion prediction.
        params: Model parameters.
        batch_stats: Batch statistics for the model (if any).
        dataset_loader: DataLoader for the dataset.
        motion_ood_score_fn: Optional scoring function to detect OOD inputs.
        ood_threshold: Threshold for OOD detection.
        max_batches: Maximum number of batches to process.
        device: Device to run the computations on.
    Returns:
        predictions: Predicted poses. Shape: (num_samples, pred_horizon, n_joints * 3)
        targets: Ground truth poses. Shape: (num_samples, pred_horizon, n_joints * 3)
        covariance_matrices: Covariance matrices of the predictions. Shape: (num_samples, pred_horizon, n_joints * 3, n_joints * 3)
        ood_scores: OOD scores. Shape: (num_samples)
        is_oods: OOD detected. Shape: (num_samples)
        last_input_poses: Last input pose. Shape (num_samples, n_joints, 3)
    """
    predictions = []
    targets = []
    covariance_matrices = []
    ood_scores = []
    is_oods = []
    last_input_poses = []

    print("\nRunning model inference...")

    for i, batch in tqdm(enumerate(dataset_loader)):
        if i >= max_batches:
            break

        input_pose = batch[0]
        target_pose = batch[1]

        # To JAX arrays
        input_pose = jnp.array(input_pose, dtype=jnp.float32)
        target_pose = jnp.array(target_pose, dtype=jnp.float32)

        # To batch dimension
        if len(input_pose.shape) == 2:
            input_pose = jnp.expand_dims(input_pose, axis=0)
            target_pose = jnp.expand_dims(target_pose, axis=0)

        # Model inference
        t0 = time()
        if batch_stats is not None:
            pred_poses, (cov, L) = motion_prediction_jit_fn(params, batch_stats, input_pose)
        else:
            pred_poses, (cov, L) = motion_prediction_jit_fn(params, input_pose)
        t1 = time()
        # print(f"  Processed batch {i + 1} in {(t1 - t0) * 1000:.2f} ms")
        if motion_ood_score_fn is not None:
            motion_ood_score = motion_ood_score_fn(input_pose)
        else:
            motion_ood_score = jnp.zeros(input_pose.shape[0], dtype=jnp.float32)
        motion_is_ood = motion_ood_score > ood_threshold
        predictions.append(pred_poses)
        targets.append(target_pose)
        covariance_matrices.append(cov)
        ood_scores.append(motion_ood_score)
        is_oods.append(motion_is_ood)
        last_input_poses.append(input_pose[:, -1, ...])

    predictions = jnp.concatenate(predictions, axis=0)
    targets = jnp.concatenate(targets, axis=0)
    covariance_matrices = jnp.concatenate(covariance_matrices, axis=0)
    ood_scores = jnp.concatenate(ood_scores, axis=0)
    is_oods = jnp.concatenate(is_oods, axis=0)
    last_input_poses = jnp.concatenate(last_input_poses, axis=0)
    return predictions, targets, covariance_matrices, ood_scores, is_oods, last_input_poses


def compute_covariance_matrices(log_var, raw_cov):
    """Compute covariance matrices from predicted log-variances and raw covariance factors.

    Args:
        log_var: Log variances [B, T, J, 3]
        raw_cov: Raw covariance factors [B, T, J, 3]
    Returns:
        cov_matrix: Covariance matrices [B, T, J, 3, 3]
    """
    B, T, J, C = log_var.shape

    # Compute variances
    variance = jnp.exp(log_var)

    var_x, var_y, var_z = variance[..., 0], variance[..., 1], variance[..., 2]

    # Construct Cholesky factors for each frame separately
    L = jnp.zeros((B, T, J, C, C))
    eps = 0

    # Lower triangular Cholesky factor (using JAX's immutable array updates)
    L = L.at[..., 0, 0].set(jnp.sqrt(var_x + eps) * 1000)
    L = L.at[..., 1, 0].set(raw_cov[..., 0] * jnp.sqrt(var_x + eps) * 1000)
    L = L.at[..., 1, 1].set(jnp.sqrt(var_y + eps) * 1000)
    L = L.at[..., 2, 0].set(raw_cov[..., 1] * jnp.sqrt(var_x + eps) * 1000)
    L = L.at[..., 2, 1].set(raw_cov[..., 2] * jnp.sqrt(var_y + eps) * 1000)
    L = L.at[..., 2, 2].set(jnp.sqrt(var_z + eps) * 1000)

    # Compute full covariance matrix from Cholesky factors
    cov_matrix = jnp.matmul(L, jnp.matrix_transpose(L))
    return cov_matrix


def run_motion_prediction(
    points_3d_buffer: jnp.ndarray,
    covariance_buffer: jnp.ndarray,
    pose_valid_buffer: jnp.ndarray,
    motion_prediction_buffer: jnp.ndarray,
    motion_uncertainty_buffer: jnp.ndarray,
    motion_prediction_jit_fn,
    motion_prediction_params,
    motion_prediction_batch_stats,
    motion_ood_score_fn,
    n_joints: int,
    input_horizon_length: int,
    prediction_horizon_length: int,
    ood_threshold: float,
    calibration_ct: float,
    calibration_it: float,
    calibration_factors: Optional[Sequence[float]],
    n_correct_poses_required: int,
    set_likelihood: float,
    conformal_calibrator: Optional[dict] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, float, bool, bool, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Run one step of motion prediction: inference, OOD scoring, calibration, buffer update.

    Args:
        points_3d_buffer: Rolling pose buffer [T, J, 3]
        covariance_buffer: Rolling covariance buffer [T, J, 3, 3]
        pose_valid_buffer: Rolling validity buffer [T]
        motion_prediction_buffer: Current motion prediction buffer [P, J, 3]
        motion_uncertainty_buffer: Current motion uncertainty buffer [P, J, 3, 3]
        motion_prediction_jit_fn: JIT-compiled JAX motion prediction function
        motion_prediction_params: Model parameters
        motion_prediction_batch_stats: Batch statistics (or None)
        motion_ood_score_fn: OOD scoring function (or None)
        n_joints: Number of skeleton joints
        input_horizon_length: Length of the input pose buffer T
        prediction_horizon_length: Length of the prediction horizon P
        ood_threshold: Threshold for classifying motion as OOD
        calibration_ct: Constant time calibration factor for covariance
        calibration_it: Increasing time calibration factor for covariance
        calibration_factors: Per-joint calibration factors (or None)
        n_correct_poses_required: Consecutive valid poses needed before using predicted motion
        set_likelihood: Likelihood level for converting covariance to set radius

    Returns:
        - Updated motion_prediction_buffer [P, J, 3]
        - Updated motion_uncertainty_buffer [P, J, 3, 3]
        - motion_set_radius [P, J]
        - motion_ood_score: float
        - motion_is_ood: bool
        - valid_motion: bool
        - motion_predicted [P, J, 3]: Raw model position prediction (before buffer update)
        - motion_cov_calibrated [P, J, 3, 3]: Calibrated covariance (before buffer update)
        - motion_cov_uncalibrated [P, J, 3, 3]: Raw model covariance before calibration
    """
    pose_input = points_3d_buffer.reshape([1, input_horizon_length, n_joints * 3])
    motion_prediction_input = jnp.concatenate([
        pose_input,
        covariance_buffer.reshape([1, input_horizon_length, n_joints * 3 * 3])
    ], axis=-1)

    # Model inference
    if motion_prediction_batch_stats is not None:
        motion_predicted, (motion_cov_predicted, _) = motion_prediction_jit_fn(
            motion_prediction_params, motion_prediction_batch_stats, motion_prediction_input
        )
    else:
        motion_predicted, (motion_cov_predicted, _) = motion_prediction_jit_fn(
            motion_prediction_params, motion_prediction_input
        )

    # OOD score
    motion_ood_score = motion_ood_score_fn(pose_input) if motion_ood_score_fn is not None else 0.0

    motion_predicted = motion_predicted.reshape(-1, prediction_horizon_length, n_joints, 3)[0]
    motion_cov_predicted = motion_cov_predicted[0]
    motion_cov_uncalibrated = motion_cov_predicted

    # Calibrate covariance
    motion_cov_predicted = calibrate_covariance_matrices(
        covariance_matrices=motion_cov_predicted,
        constant_time_factor=calibration_ct,
        increase_time_factor=calibration_it,
        joint_calibration_factors=calibration_factors,
    )
    if isinstance(motion_cov_predicted, np.ndarray):
        motion_cov_predicted = jnp.array(motion_cov_predicted)

    motion_is_ood = bool(motion_ood_score > ood_threshold)

    motion_prediction_buffer, motion_uncertainty_buffer, valid_motion = update_motion_prediction_buffer(
        motion_prediction_buffer=motion_prediction_buffer,
        motion_uncertainty_buffer=motion_uncertainty_buffer,
        predicted_motion=motion_predicted,
        predicted_motion_uncertainty=motion_cov_predicted,
        is_ood=motion_is_ood,
        pose_valid_buffer=pose_valid_buffer,
        n_correct_poses_required=n_correct_poses_required,
    )

    if conformal_calibrator is not None:
        # Conditional-conformal set radius from RAW model cov + last-input-frame uncertainty
        # (replaces the affine-calibrated set; the affine cov is still used for the buffer/OOD path).
        input_cov = np.asarray(covariance_buffer).reshape(input_horizon_length, n_joints, 3, 3)[-1]
        motion_set_radius = conformal_set_radius(motion_cov_uncalibrated, input_cov, conformal_calibrator)
    else:
        motion_set_radius = convert_covariance_matrices_to_set(
            motion_cov_predicted, likelihood=set_likelihood
        )

    return (
        motion_prediction_buffer,
        motion_uncertainty_buffer,
        motion_set_radius,
        motion_ood_score,
        motion_is_ood,
        valid_motion,
        motion_predicted,
        motion_cov_predicted,
        motion_cov_uncalibrated,
    )


def run_motion_prediction_batched(
    points_3d_buffers: jnp.ndarray,
    covariance_buffers: jnp.ndarray,
    pose_valid_buffers: jnp.ndarray,
    motion_prediction_buffer: jnp.ndarray,
    motion_uncertainty_buffer: jnp.ndarray,
    motion_prediction_jit_fn,
    motion_prediction_params,
    motion_prediction_batch_stats,
    motion_ood_score_fn,
    n_joints: int,
    input_horizon_length: int,
    prediction_horizon_length: int,
    ood_threshold: float,
    calibration_ct: float,
    calibration_it: float,
    calibration_factors: Optional[Sequence[float]],
    n_correct_poses_required: int,
    set_likelihood: float,
    pose_buffer_good_batch: jnp.ndarray,
    frame_counter: int,
    conformal_calibrator: Optional[dict] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, List[bool], jnp.ndarray, jnp.ndarray]:
    """Run motion prediction for a batch of B intermediate pose-buffer states in one model call.

    The motion model is called once with a batched input [B, T, J*3 + J*9].
    The motion prediction buffer is then updated sequentially (one call to
    update_motion_prediction_buffer per frame), matching the causal structure of
    the pipeline.  convert_covariance_matrices_to_set is called once on the full
    batch.

    Args:
        points_3d_buffers: B intermediate pose buffers [B, T, J, 3]
        covariance_buffers: B intermediate covariance buffers [B, T, J, 3, 3]
        pose_valid_buffers: B intermediate validity buffers [B, T]
        motion_prediction_buffer: Current motion prediction buffer [P, J, 3]
        motion_uncertainty_buffer: Current motion uncertainty buffer [P, J, 3, 3]
        motion_prediction_jit_fn: JIT-compiled motion prediction function
        motion_prediction_params: Model parameters
        motion_prediction_batch_stats: Batch statistics (or None)
        motion_ood_score_fn: OOD scoring function (or None)
        n_joints: Number of skeleton joints
        input_horizon_length: Length of the pose buffer T
        prediction_horizon_length: Length of the prediction horizon P
        ood_threshold: Threshold for OOD classification
        calibration_ct: Constant-time covariance calibration factor
        calibration_it: Increasing-time covariance calibration factor
        calibration_factors: Per-joint calibration factors (or None)
        n_correct_poses_required: Consecutive valid poses needed before using predicted motion
        set_likelihood: Likelihood level for set-radius conversion
        pose_buffer_good_batch: Whether each intermediate buffer is ready for prediction [B]
        frame_counter: Global frame counter before this batch (used to determine readiness)

    Returns:
        - Updated motion_prediction_buffer [P, J, 3]
        - Updated motion_uncertainty_buffer [P, J, 3, 3]
        - motion_set_radii [B, P, J]
        - motion_ood_scores [B]
        - motion_is_oods [B]
        - valid_motions: List[bool] of length B
        - motion_predicted [B, P, J, 3]
        - motion_cov_calibrated [B, P, J, 3, 3]
    """
    B = points_3d_buffers.shape[0]
    T = input_horizon_length

    # Build batched model input [B, T, J*3 + J*3*3]
    pose_input = points_3d_buffers.reshape([B, T, n_joints * 3])
    motion_prediction_input = jnp.concatenate([
        pose_input,
        covariance_buffers.reshape([B, T, n_joints * 3 * 3]),
    ], axis=-1)

    # Single batched model call
    if motion_prediction_batch_stats is not None:
        motion_predicted, (motion_cov_predicted, _) = motion_prediction_jit_fn(
            motion_prediction_params, motion_prediction_batch_stats, motion_prediction_input
        )
    else:
        motion_predicted, (motion_cov_predicted, _) = motion_prediction_jit_fn(
            motion_prediction_params, motion_prediction_input
        )

    # OOD scores [B]
    motion_ood_scores = (
        motion_ood_score_fn(pose_input)
        if motion_ood_score_fn is not None
        else jnp.zeros(B, dtype=jnp.float32)
    )
    motion_is_oods = motion_ood_scores > ood_threshold  # [B]

    # Reshape model outputs to [B, P, J, 3] and [B, P, J, 3, 3]
    motion_predicted = motion_predicted.reshape(B, prediction_horizon_length, n_joints, 3)
    # motion_cov_predicted already [B, P, J, 3, 3]

    # Calibrate covariances for entire batch [B, P, J, 3, 3]
    motion_cov_calibrated = calibrate_covariance_matrices(
        covariance_matrices=motion_cov_predicted,
        constant_time_factor=calibration_ct,
        increase_time_factor=calibration_it,
        joint_calibration_factors=calibration_factors,
    )
    if isinstance(motion_cov_calibrated, np.ndarray):
        motion_cov_calibrated = jnp.array(motion_cov_calibrated)

    # Sequentially update motion prediction buffer (one call per frame in batch)
    valid_motions: List[bool] = []
    for b in range(B):
        if (frame_counter + b) < input_horizon_length - 1 or not bool(pose_buffer_good_batch[b]):
            valid_motions.append(False)
            continue
        motion_prediction_buffer, motion_uncertainty_buffer, valid_motion = update_motion_prediction_buffer(
            motion_prediction_buffer=motion_prediction_buffer,
            motion_uncertainty_buffer=motion_uncertainty_buffer,
            predicted_motion=motion_predicted[b],
            predicted_motion_uncertainty=motion_cov_calibrated[b],
            is_ood=bool(motion_is_oods[b]),
            pose_valid_buffer=pose_valid_buffers[b],
            n_correct_poses_required=n_correct_poses_required,
        )
        valid_motions.append(valid_motion)

    # Batch-convert covariances to set radii [B, P, J]
    if conformal_calibrator is not None:
        # Conditional-conformal set from RAW model cov + each buffer's last-input-frame uncertainty.
        input_cov = np.asarray(covariance_buffers).reshape(B, T, n_joints, 3, 3)[:, -1]  # [B,J,3,3]
        motion_set_radii = conformal_set_radius(motion_cov_predicted, input_cov, conformal_calibrator)
    else:
        motion_set_radii = convert_covariance_matrices_to_set(
            motion_cov_calibrated, likelihood=set_likelihood
        )

    return (
        motion_prediction_buffer,
        motion_uncertainty_buffer,
        motion_set_radii,
        motion_ood_scores,
        motion_is_oods,
        valid_motions,
        motion_predicted,
        motion_cov_calibrated,
    )


def calibrate_covariance_matrices(
    covariance_matrices: Union[jnp.ndarray, np.ndarray],
    constant_time_factor: float = 1.2,
    increase_time_factor: float = 0.4,
    joint_calibration_factors: Optional[Sequence[float]] = None
) -> Union[jnp.ndarray, np.ndarray]:
    if len(covariance_matrices.shape) == 5:
        T = covariance_matrices.shape[1]
        J = covariance_matrices.shape[2]
        if not joint_calibration_factors:
            scaling_factors_joints = np.ones(J)
        else:
            assert len(joint_calibration_factors) == J
            scaling_factors_joints = np.array(joint_calibration_factors)
        scaling_factors_times = (constant_time_factor + increase_time_factor * np.arange(T))[None, :, None, None, None]
        scaling_factors_joints = scaling_factors_joints[None, None, :, None, None]
    elif len(covariance_matrices.shape) == 4:
        T = covariance_matrices.shape[0]
        J = covariance_matrices.shape[1]
        if not joint_calibration_factors:
            scaling_factors_joints = np.ones(J)
        else:
            assert len(joint_calibration_factors) == J
            scaling_factors_joints = np.array(joint_calibration_factors)
        scaling_factors_times = (constant_time_factor + increase_time_factor * np.arange(T))[:, None, None, None]
        scaling_factors_joints = scaling_factors_joints[None, :, None, None]
    else:
        raise ShapeError(f"Covaraince matrices have incorrect shape: {covariance_matrices.shape}.")

    covariance_matrices = covariance_matrices * scaling_factors_times
    covariance_matrices = covariance_matrices * scaling_factors_joints
    return covariance_matrices


# --------------------------------------------------------------------------- conditional conformal
# Conditional (Mondrian/CQR) conformal calibration of the set radius, conditioned on
# (joint x horizon-frame x input-uncertainty bin). Fitted offline by
# ``motion_prediction.conformal_calibration`` and saved as an .npz; applied here as a drop-in
# replacement for the affine ``calibrate_covariance_matrices`` + ``convert_..._to_set`` pair.
# See that module for the rationale (the affine calibration cannot fix the input-uncertainty- and
# joint-conditional under-coverage). Default location written by the fitter:
DEFAULT_CONFORMAL_CALIBRATOR = "results/motion_prediction/conformal_calibration/conformal_calibrator.npz"


def load_conformal_calibrator(path):
    """Load a saved conditional-conformal calibrator .npz into the dict the apply fns expect."""
    d = np.load(path)
    return dict(bin_edges=np.asarray(d["bin_edges"], dtype=np.float64),
                q_grid=np.asarray(d["q_grid"], dtype=np.float64),
                B=int(d["B"]), level=float(d["level"]), J=int(d["J"]), T=int(d["T"]))


def conformal_set_radius(model_cov, input_cov, calibrator):
    """Conditional-conformal spherical set radius (mm) from RAW model + input covariance.

    Drop-in replacement for ``convert_covariance_matrices_to_set(calibrate_covariance_matrices(.))``.
    Conditions each (joint, frame) on the last input frame's per-joint uncertainty:
        r_cal = max(r_model + q_hat(joint, frame, input_unc_bin), 0).

    Args:
        model_cov: raw model covariance, [..., T, J, 3, 3] (mm^2); accepts batched [N,T,J,3,3] or
            a single [T,J,3,3].
        input_cov: last input frame per-joint covariance, [..., J, 3, 3] (mm^2), leading dims
            matching ``model_cov`` (or absent for the single case).
        calibrator: dict from ``load_conformal_calibrator``.
    Returns:
        radius [..., T, J] in mm (same leading shape as ``model_cov`` minus the 3x3).
    """
    mc = np.asarray(model_cov, dtype=np.float64)
    ic = np.asarray(input_cov, dtype=np.float64)
    single = (mc.ndim == 4)
    if single:
        mc, ic = mc[None], ic[None]
    N, T, J = mc.shape[:3]
    level = calibrator["level"]
    r_model = np.asarray(convert_covariance_matrices_to_set(mc, level), dtype=np.float64)   # [N,T,J] mm
    in_set = np.asarray(convert_covariance_matrices_to_set(ic, level), dtype=np.float64) / 1000.0  # [N,J] m
    in_TJ = np.repeat(in_set[:, None, :], T, axis=1)                                         # [N,T,J] m
    b = np.clip(np.searchsorted(calibrator["bin_edges"], in_TJ, side="right"), 0, calibrator["B"] - 1)
    jj = np.broadcast_to(np.arange(J)[None, None, :], (N, T, J))
    tt = np.broadcast_to(np.arange(T)[None, :, None], (N, T, J))
    r = np.maximum(r_model + calibrator["q_grid"][jj, tt, b], 0.0)                           # [N,T,J] mm
    return r[0] if single else r


def calibrated_set_radius(model_cov, input_cov, conformal_calibrator=None,
                          calibration_ct=None, calibration_it=None, calibration_factors=None,
                          set_likelihood=None):
    """Set radius (mm) via conditional conformal if a calibrator is given, else affine fallback.

    This is the single boundary every consumer should call so the calibration method is swappable.
    ``input_cov`` is only needed for the conformal path (pass None for affine).
    """
    if conformal_calibrator is not None:
        return conformal_set_radius(model_cov, input_cov, conformal_calibrator)
    cov = calibrate_covariance_matrices(model_cov, calibration_ct, calibration_it, calibration_factors)
    return convert_covariance_matrices_to_set(cov, likelihood=set_likelihood)


def compute_human_occupancies(
    human_meas: np.ndarray,
    meas_uncertainty: np.ndarray,
    human_radius: np.ndarray
) -> np.ndarray:
    """Compute the space the human body occupies at each time point.

    Args:
      human_meas: ground truth human measurements in mm, shape = [N, n_t, n_j, 3]
      meas_uncertainty: measurement uncertainty in mm, shape = [N, n_t, n_j]
      human_radius: radius of the human body parts in mm, shape = [n_j]

    Returns:
      human_occupancies: ground truth radius of spherical human occupancies in mm, shape = [N, n_t, n_j]
    """
    human_meas = np.asarray(human_meas)
    human_radius = np.asarray(human_radius)
    N, n_t, n_j, _ = human_meas.shape
    if human_radius.shape != (n_j,):
        raise ShapeError(
            f"human_radius must have shape ({n_j},), got {human_radius.shape}."
        )
    # Each joint is occupied by a sphere of its body-part radius, centered at the
    # measured joint position. The radius is constant over samples and time.
    human_occupancies = meas_uncertainty + np.broadcast_to(
        human_radius[None, None, :], (N, n_t, n_j)
    ).astype(float).copy()
    # All-zero measurements mark invalid/missing joints (codebase convention);
    # those occupy no space.
    invalid = np.all(human_meas == 0.0, axis=-1)  # [N, n_t, n_j]
    human_occupancies[invalid] = 0.0
    return human_occupancies
