"""Dataset for the motion prediction task on the Human3.6M dataset."""

import os
from typing import Optional
import torch.utils.data
from torch.utils.data import Dataset
from spacepy import pycdf
import torch
import jax.numpy as jnp
import numpy as np

from conformal_human_motion_prediction.pose_estimation.h36m_settings import JOINT_IDX_17, JOINT_IDX_13
from conformal_human_motion_prediction.motion_prediction.h36m_settings import (
    INPUT_HORIZON_LENGTH,
    PREDICTION_HORIZON_LENGTH,
    REDUCED_TIMESTEP,
    REDUCED_JOINT_INDICES,
    FAKE_INPUT_UNCERTAINTY,
    V_HUMAN_ISO,
)
from conformal_human_motion_prediction.datasets.utils import get_loader
from conformal_human_motion_prediction.utils.eval_utils import get_too_fast_human_movement

# Dataset splits matching original H36M
SPLIT = {"train": ["S1", "S6", "S7", "S8", "S9"], "validation": ["S11"], "test": ["S5"]}

# H36M is recorded at 50 Hz; load_data/load_data_preprocessed downsample by 2 (``offset::2``).
DOWNSAMPLED_FPS = 25.0


class Human36mMotionDataset3D(Dataset):
    """Dataset class for Human3.6M motion data."""

    def __init__(
        self,
        base_directory,
        split="train",
        input_frames=INPUT_HORIZON_LENGTH,
        predict_frames=PREDICTION_HORIZON_LENGTH,
        jax_format=False,
        reduce_size=False,
        reduced_timestep=REDUCED_TIMESTEP,
        reduced_joints=REDUCED_JOINT_INDICES,
        ood=False,
        input_uncertainty=False,
        directory_uncertain=None,
        seed=None,
        augment=False,
        scale_range=(0.8, 1.2),
        max_target_speed=V_HUMAN_ISO,
    ):
        self.input_frames = input_frames
        self.predict_frames = predict_frames
        self.jax_format = jax_format
        self.pose_data = []
        # GT (mocap) pose at the last input frame, per window. Only the load_data_preprocessed path
        # needs it (there the stored input frame is the noisy camera pose, not mocap); load_data
        # windows are all mocap so the boundary pose is already in the window.
        self._gt_last_input = None
        if not input_uncertainty:
            self.pose_data = self.load_data(base_directory, split)
            self.covariance_data = None
        else:
            assert directory_uncertain is not None, "directory_uncertain must be provided when input_uncertainty is True"
            self.pose_data, self.covariance_data = self.load_data_preprocessed(
                base_directory_uncertain=directory_uncertain,
                base_directory_gt=base_directory,
                split=split
            )
        # Drop windows with implausibly fast motion across the mocap prediction horizon, INCLUDING
        # the last-mocap-input -> first-mocap-prediction transition.
        # We only include data, where the human moves slower than ``max_target_speed`` (m/s;
        # defaults to the ISO limit V_HUMAN_ISO = 2.0 m/s). Pass a larger value to keep faster
        # motions, or a non-positive / infinite value to disable the filter entirely.
        self._filter_fast_target_motion(
            input_is_mocap=not input_uncertainty, threshold=max_target_speed
        )
        self.reduce_size = reduce_size
        self.reduced_timestep = reduced_timestep
        self.reduced_joints = reduced_joints
        self.ood = ood
        if seed:
            np.random.seed(seed)
        self.input_uncertainty = input_uncertainty
        self.augment = augment
        self.scale_range = scale_range

    def load_data(self, base_directory, split):
        all_data = []
        for subject in SPLIT[split]:
            directory = os.path.join(base_directory, subject, "Poses_D3_Positions")
            if not os.path.exists(directory):
                print(f"Warning: Directory {directory} not found, skipping...")
                continue

            for filename in os.listdir(directory):
                if filename.endswith(".cdf"):
                    file_path = os.path.join(directory, filename)
                    with pycdf.CDF(file_path) as cdf:
                        poses = cdf["Pose"][:]
                        poses = poses.reshape(-1, 32, 3)
                        poses_13 = poses[:, JOINT_IDX_17, :]
                        poses_13 = poses_13[:, JOINT_IDX_13, :]
                        poses_13 = poses_13.reshape(poses_13.shape[0], -1)

                        for offset in [0, 1]:
                            downsampled_poses = poses_13[offset::2]
                            for i in range(len(downsampled_poses) - self.input_frames - self.predict_frames + 1):
                                window = downsampled_poses[i : i + self.input_frames + self.predict_frames]
                                all_data.append(window)
        print(f"Loaded {len(all_data)} sequences for {split} split")
        return all_data

    def load_data_preprocessed(self, base_directory_uncertain, base_directory_gt, split):
        all_poses = []
        all_covariances = []
        all_gt_last_input = []   # GT (mocap) pose at the last input frame, for the too-fast screen
        for subject in SPLIT[split]:
            uncertain_directory = os.path.join(base_directory_uncertain, subject)
            if not os.path.exists(uncertain_directory):
                print(f"Warning: Directory {uncertain_directory} not found, skipping...")
                continue
            for filename in os.listdir(uncertain_directory):
                if not filename.endswith(".npz"):
                    continue
                action = os.path.splitext(filename)[0]
                file_path = os.path.join(uncertain_directory, filename)
                data = np.load(file_path)
                pred_poses = data['poses_3d']  # (num_frames, 13, 3)
                covariances = data['covariances_3d']  # (num_frames, 13, 3, 3)
                valid_mask = data['valid_mask']  # (num_frames,)
                pred_poses = pred_poses.reshape(pred_poses.shape[0], -1)  # (num_frames, 13*3)
                covariances = covariances.reshape(covariances.shape[0], -1)  # (num_frames, 13*3*3)
                valid_mask = valid_mask.astype(bool)
                # Mark frames with degenerate covariance or pose values as invalid.
                # Normal pose p99.9 is ~2800 mm; failed triangulations can reach 1e6 mm.
                # Normal cov p99 is ~500k mm²; failed triangulations can reach 1e17 mm².
                # These are not caught by valid_mask (which only checks human detection).
                pose_valid = np.all(np.abs(pred_poses) <= 4000, axis=1)
                cov_valid = np.all(np.abs(covariances) <= 1e5, axis=1)
                valid_mask = valid_mask & pose_valid & cov_valid

                # For DEBUG
                # if action != "Directions":
                #     continue

                gt_file = os.path.join(base_directory_gt, subject, "Poses_D3_Positions", f"{action}.cdf")
                if os.path.exists(gt_file):
                    with pycdf.CDF(gt_file) as cdf:
                        gt_poses = cdf["Pose"][:]
                        gt_poses = gt_poses.reshape(-1, 32, 3)
                        gt_poses_13 = gt_poses[:, JOINT_IDX_17, :]
                        gt_poses_13 = gt_poses_13[:, JOINT_IDX_13, :]
                        gt_poses_13 = gt_poses_13.reshape(gt_poses_13.shape[0], -1)
                else:
                    print(f"Warning: GT file {gt_file} not found, using predicted poses as GT.")
                    gt_poses_13 = pred_poses.copy()

                # Trim sequences to match lengths
                min_length = min(len(pred_poses), len(gt_poses_13), len(covariances), len(valid_mask))
                pred_poses = pred_poses[:min_length]
                gt_poses_13 = gt_poses_13[:min_length]
                covariances = covariances[:min_length]
                valid_mask = valid_mask[:min_length]

                for offset in [0, 1]:
                    downsampled_poses = pred_poses[offset::2]
                    downsampled_covariances = covariances[offset::2]
                    downsampled_valid_mask = valid_mask[offset::2]
                    downsampled_gt_poses = gt_poses_13[offset::2]
                    for i in range(len(downsampled_poses) - self.input_frames - self.predict_frames + 1):
                        poses_window = downsampled_poses[i : i + self.input_frames + self.predict_frames].copy()
                        # Replace the target frames with ground truth poses
                        poses_window[-self.predict_frames:] = downsampled_gt_poses[i + self.input_frames : i + self.input_frames + self.predict_frames]
                        covariances_window = downsampled_covariances[i : i + self.input_frames + self.predict_frames]
                        valid_mask_window = downsampled_valid_mask[i : i + self.input_frames + self.predict_frames]
                        if sum(valid_mask_window) < self.input_frames + self.predict_frames:
                            continue
                        all_poses.append(poses_window)
                        all_covariances.append(covariances_window)
                        # GT (mocap) pose at the last input frame -- the stored window's input is the
                        # noisy camera pose, so keep the mocap one for the boundary too-fast screen.
                        all_gt_last_input.append(downsampled_gt_poses[i + self.input_frames - 1])
        all_poses = np.array(all_poses)
        all_covariances = np.array(all_covariances)
        self._gt_last_input = np.array(all_gt_last_input)
        print(f"Loaded {len(all_poses)} sequences for {split} split from preprocessed data")
        return all_poses, all_covariances

    def _filter_fast_target_motion(self, input_is_mocap, threshold=V_HUMAN_ISO, fps=DOWNSAMPLED_FPS):
        """Drop windows with implausibly fast motion across the mocap prediction horizon.

        Reuses ``get_too_fast_human_movement`` on the mocap frames from the last input pose through
        the targets -- i.e. the differenced span covers the last-mocap-input -> first-mocap-target
        transition AND all in-target transitions. That function treats its first frame as speed 0,
        so the input's internal (and, in the camera case, noisy) motion is never differenced. A
        surviving speed above ``threshold`` flags a mocap measurement jump. Poses are in mm.

        The mocap source of the boundary pose differs by load path (hence ``input_is_mocap``):
          * load_data (input_is_mocap=True): the whole window is mocap, so the last input frame of
            the stored window is the boundary pose -> screen ``window[input_frames-1:]``.
          * load_data_preprocessed (input_is_mocap=False): the stored input frame is the noisy
            camera pose, so the GT boundary pose saved in ``self._gt_last_input`` is prepended to
            the (mocap) target frames.
        """
        n_before = len(self.pose_data)
        if n_before == 0:
            return
        if threshold is None or threshold <= 0 or not np.isfinite(threshold):
            print(f"Too-fast target filter: disabled (threshold={threshold}), kept all {n_before} sequences")
            self._gt_last_input = None
            return
        poses = np.asarray(self.pose_data)                       # [N, seq, J*3]
        if input_is_mocap:
            screen = poses[:, self.input_frames - 1:]            # [N, P+1, J*3] last mocap in + targets
        else:
            boundary = np.asarray(self._gt_last_input)[:, None, :]   # [N, 1, J*3] GT last input pose
            screen = np.concatenate([boundary, poses[:, self.input_frames:]], axis=1)  # [N, P+1, J*3]
        N, T, D = screen.shape
        if T < 2:
            return
        too_fast = get_too_fast_human_movement(screen.reshape(N, T, D // 3, 3), threshold, 1.0 / fps)
        keep = ~np.any(too_fast, axis=(1, 2))                    # [N]
        n_kept = int(keep.sum())
        self._gt_last_input = None                               # no longer needed after screening
        if n_kept == n_before:
            print(f"Too-fast target filter: kept all {n_before} sequences "
                  f"(no horizon joint exceeds {threshold} m/s)")
            return
        # Apply the mask, preserving the original container type of each attribute.
        self.pose_data = (self.pose_data[keep] if isinstance(self.pose_data, np.ndarray)
                          else [w for w, k in zip(self.pose_data, keep) if k])
        if self.covariance_data is not None:
            self.covariance_data = (self.covariance_data[keep]
                                    if isinstance(self.covariance_data, np.ndarray)
                                    else [w for w, k in zip(self.covariance_data, keep) if k])
        print(f"Too-fast target filter: kept {n_kept}/{n_before} sequences "
              f"(dropped {n_before - n_kept} with >{threshold} m/s mocap motion in the horizon)")

    def _augment_sequence(self, sequence, covariances=None):
        """Apply random Z-axis rotation and isotropic scaling augmentation.

        The head joint (joint 0) at the first frame is used as the center of
        rotation and scaling. All joints (and optionally covariance matrices)
        are transformed consistently across the full sequence, so both body
        proportions and global travel distance scale together.

        Args:
            sequence: np.ndarray [seq_len, N_JOINTS * 3]
            covariances: optional np.ndarray [seq_len, N_JOINTS * 9]

        Returns:
            Augmented sequence (and covariances if provided).
        """
        n_joints = sequence.shape[1] // 3
        seq = sequence.reshape(-1, n_joints, 3).copy()

        # Head position at the first frame (joint 0, columns 0:3)
        head_pos = seq[0, 0, :].copy()

        # Translate to head-centred coordinates
        seq -= head_pos

        # Random rotation around Z-axis in [-180, +180] deg
        angle = np.random.uniform(-np.pi, np.pi)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        R = np.array(
            [[cos_a, -sin_a, 0.0],
             [sin_a,  cos_a, 0.0],
             [0.0,    0.0,   1.0]],
            dtype=sequence.dtype,
        )
        # seq shape: [seq_len, n_joints, 3]  ->  apply R to last axis
        seq = np.einsum('ij,...j->...i', R, seq)

        # Random isotropic scale (body size and travel distance scale together)
        scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
        seq *= scale

        # Translate back to original head position
        seq += head_pos

        sequence_aug = seq.reshape(-1, n_joints * 3)

        if covariances is not None:
            n_cov_joints = covariances.shape[1] // 9
            covs = covariances.reshape(-1, n_cov_joints, 3, 3).copy()
            # Rotate: C' = R C R^T
            RC = np.einsum('ac,nkcb->nkab', R, covs)
            covs = np.einsum('nkac,bc->nkab', RC, R)
            # Scale: C' = scale^2 * C
            covs *= scale ** 2
            return sequence_aug, covs.reshape(-1, n_cov_joints * 9)

        return sequence_aug

    def __len__(self):
        return len(self.pose_data)

    def __getitem__(self, idx):
        sequence = self.pose_data[idx]
        covariance_sequence = (
            self.covariance_data[idx]
            if (self.input_uncertainty and self.covariance_data is not None)
            else None
        )

        if self.augment:
            if covariance_sequence is not None:
                sequence, covariance_sequence = self._augment_sequence(sequence, covariance_sequence)
            else:
                sequence = self._augment_sequence(sequence)

        input_pose = sequence[: self.input_frames]
        if self.ood:
            # Randomly shuffle the input sequence in the first dimension for OOD testing
            input_pose = input_pose.copy()
            np.random.shuffle(input_pose)
        target_pose = sequence[self.input_frames :]
        if self.reduce_size:
            # Extract only the specified timestep and joints
            target_pose_timestep = target_pose[self.reduced_timestep]  # [num_joints*3]
            target_pose_timestep = target_pose_timestep.reshape(-1, 3)  # [num_joints, 3]
            reduced_target = target_pose_timestep[self.reduced_joints, :]  # [len(reduced_joints), 3]
            target_pose = reduced_target.reshape(-1)  # [len(reduced_joints)*3]
        if covariance_sequence is not None:
            input_covariances = covariance_sequence[: self.input_frames]
            if self.ood:
                # Randomly shuffle the input covariances in the first dimension for OOD testing
                input_covariances = input_covariances.copy()
                np.random.shuffle(input_covariances)
            # Append to input_pose
            input_pose = np.concatenate([input_pose, input_covariances], axis=-1)
        if self.jax_format:
            # Convert to JAX arrays
            input_pose = jnp.array(input_pose, dtype=jnp.float32)
            target_pose = jnp.array(target_pose, dtype=jnp.float32)
        else:
            # Convert to PyTorch tensors
            input_pose = torch.FloatTensor(input_pose)
            target_pose = torch.FloatTensor(target_pose)
        return [input_pose, target_pose]


def subsample_dataset(dataset, n_samples: int, seed: Optional[int] = 0, shuffle: bool = True):
    """Subsample a dataset to a specified number of samples.

    Args:
        dataset: The original dataset (torch.utils.data.Dataset)
        n_samples: Number of samples to select
        seed: Random seed for reproducibility
        shuffle: Choose random indices for dataset. Otherwise, data will be in order.

    Returns:
        Subsampled dataset (torch.utils.data.Subset)
    """
    n_samples_train = min(n_samples, len(dataset))
    if shuffle:
        # Randomly select n_samples_train indices
        if seed is not None:
            np.random.seed(seed)
        train_indices = np.random.choice(len(dataset), n_samples_train, replace=False)
    else:
        train_indices = list(range(n_samples_train))
    subsampled_dataset = torch.utils.data.Subset(dataset, train_indices)
    return subsampled_dataset


def get_h36m_motion_dataset_function(
    base_directory: str,
    batch_size: int = 128,
    shuffle: bool = False,
    seed: Optional[int] = None,
    split_train_val_ratio: float = 1.0,
    n_samples: Optional[int] = None,
    input_uncertainty: bool = False,
    reduce_size: bool = False,
    ood: bool = False,
    directory_uncertain: Optional[str] = None,
    augment: bool = False,
    scale_range: tuple = (0.8, 1.2),
    max_target_speed: float = V_HUMAN_ISO,
):
    """
    Get data loaders for preprocessed H36M dataset

    Args:
        base_directory: Path to dataset directory
        batch_size: Batch size for data loaders
        shuffle: Whether to shuffle the data
        seed: Random seed for reproducibility
        split_train_val_ratio: Ratio for splitting train set into train/val -> Not used for this dataset!
        return_metadata: Whether to return metadata with samples
        n_samples: Number of samples to use from dataset (None = use all),
        input_uncertainty: Add artificial input uncertainty of this amount, defaults to None.
        reduced_size: Reduced output size of only head and two hand poses.
        ood: Shuffle input poses in time dimension.
        directory_uncertain: Directory containing preprocessed uncertain data.
        seed: Optional seed for ood data generation.

    Returns:
        tuple: (train_loader, valid_loader, test_loader)
    """
    # Create datasets
    train_dataset = Human36mMotionDataset3D(
        base_directory=base_directory,
        split='train',
        jax_format=False,
        input_uncertainty=input_uncertainty,
        reduce_size=reduce_size,
        ood=ood,
        directory_uncertain=directory_uncertain,
        seed=seed,
        augment=augment,
        scale_range=scale_range,
        max_target_speed=max_target_speed,
    )

    validation_dataset = Human36mMotionDataset3D(
        base_directory=base_directory,
        split='validation',
        jax_format=False,
        input_uncertainty=input_uncertainty,
        reduce_size=reduce_size,
        ood=ood,
        directory_uncertain=directory_uncertain,
        seed=seed,
        max_target_speed=max_target_speed,
    )

    test_dataset = Human36mMotionDataset3D(
        base_directory=base_directory,
        split='test',
        jax_format=False,
        input_uncertainty=input_uncertainty,
        reduce_size=reduce_size,
        ood=ood,
        directory_uncertain=directory_uncertain,
        seed=seed,
        max_target_speed=max_target_speed,
    )

    # Subsample if n_samples is specified
    if n_samples is not None:
        train_dataset = subsample_dataset(train_dataset, n_samples, seed, shuffle)
        validation_dataset = subsample_dataset(validation_dataset, n_samples, seed, shuffle)
        test_dataset = subsample_dataset(test_dataset, n_samples, seed, shuffle)

    # Split train dataset into train/val
    train_loader = get_loader(
        train_dataset,
        split_train_val_ratio=1.0,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        seed=seed
    )

    validation_loader = get_loader(
        validation_dataset,
        split_train_val_ratio=1.0,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        seed=seed
    )

    # Create test loader
    test_loader = get_loader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        seed=seed
    )

    return train_loader, validation_loader, test_loader


def get_h36m_motion_dataset(
    base_directory,
    batch_size=128,
    shuffle=False,
    seed=0,
    split_train_val_ratio=0.9,
    n_samples=None,
    augment=False,
    scale_range=(0.8, 1.2),
    max_target_speed=V_HUMAN_ISO,
):
    """
    Get data loaders for preprocessed H36M dataset

    Args:
        base_directory: Path to dataset directory
        batch_size: Batch size for data loaders
        shuffle: Whether to shuffle the data
        seed: Random seed for reproducibility
        split_train_val_ratio: Ratio for splitting train set into train/val
        return_metadata: Whether to return metadata with samples
        n_samples: Number of samples to use from dataset (None = use all)
        augment: Whether to apply Z-rotation and scale augmentation to the train split
        scale_range: (min, max) scale factor range for augmentation
        max_target_speed: Too-fast filter threshold in m/s (<=0/inf disables it)

    Returns:
        tuple: (train_loader, valid_loader, test_loader)
    """
    return get_h36m_motion_dataset_function(
        base_directory=base_directory,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        n_samples=n_samples,
        augment=augment,
        scale_range=scale_range,
        max_target_speed=max_target_speed,
    )


def get_h36m_motion_dataset_with_uncertainty(
    base_directory,
    directory_uncertain,
    batch_size=128,
    shuffle=False,
    seed=0,
    split_train_val_ratio=0.9,
    n_samples=None,
    augment=False,
    scale_range=(0.8, 1.2),
    max_target_speed=V_HUMAN_ISO,
):
    """
    Get data loaders for preprocessed H36M dataset

    Args:
        base_directory: Path to dataset directory
        directory_uncertain: Directory containing preprocessed uncertain data.
        batch_size: Batch size for data loaders
        shuffle: Whether to shuffle the data
        seed: Random seed for reproducibility
        split_train_val_ratio: Ratio for splitting train set into train/val
        return_metadata: Whether to return metadata with samples
        n_samples: Number of samples to use from dataset (None = use all)
        augment: Whether to apply Z-rotation and scale augmentation to the train split
        scale_range: (min, max) scale factor range for augmentation
        max_target_speed: Too-fast filter threshold in m/s (<=0/inf disables it)

    Returns:
        tuple: (train_loader, valid_loader, test_loader)
    """
    return get_h36m_motion_dataset_function(
        base_directory=base_directory,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        n_samples=n_samples,
        input_uncertainty=True,
        directory_uncertain=directory_uncertain,
        augment=augment,
        scale_range=scale_range,
        max_target_speed=max_target_speed,
    )


def get_h36m_motion_reduced_output_dataset(
    base_directory,
    batch_size=128,
    shuffle=False,
    seed=0,
    split_train_val_ratio=0.9,
    n_samples=None,
    augment=False,
    scale_range=(0.8, 1.2),
):
    """
    Get data loaders for preprocessed H36M dataset

    Args:
        base_directory: Path to dataset directory
        batch_size: Batch size for data loaders
        shuffle: Whether to shuffle the data
        seed: Random seed for reproducibility
        split_train_val_ratio: Ratio for splitting train set into train/val
        return_metadata: Whether to return metadata with samples
        n_samples: Number of samples to use from dataset (None = use all)
        augment: Whether to apply Z-rotation and scale augmentation to the train split
        scale_range: (min, max) scale factor range for augmentation

    Returns:
        tuple: (train_loader, valid_loader, test_loader)
    """
    return get_h36m_motion_dataset_function(
        base_directory=base_directory,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        n_samples=n_samples,
        reduce_size=True,
        augment=augment,
        scale_range=scale_range,
    )


def get_h36m_motion_ood_dataset(
    base_directory,
    batch_size=128,
    shuffle=False,
    seed=0,
    split_train_val_ratio=0.9,
    n_samples=None
):
    """Get data loaders for the H36M motion prediction OOD dataset.

    This dataset shuffles the input sequences to create out-of-distribution samples.

    Args:
        base_directory: Path to dataset directory
        batch_size: Batch size for data loaders
        shuffle: Whether to shuffle the data
        seed: Random seed for reproducibility
        split_train_val_ratio: Ratio for splitting train set into train/val
        return_metadata: Whether to return metadata with samples
        n_samples: Number of samples to use from dataset (None = use all)

    Returns:
        tuple: (train_loader, valid_loader, test_loader)
    """
    return get_h36m_motion_dataset_function(
        base_directory=base_directory,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        n_samples=n_samples,
        ood=True
    )


def get_h36m_motion_ood_dataset_with_uncertainty(
    base_directory,
    directory_uncertain,
    batch_size=128,
    shuffle=False,
    seed=0,
    split_train_val_ratio=0.9,
    n_samples=None
):
    """Get data loaders for the H36M motion prediction OOD dataset.

    This dataset shuffles the input sequences to create out-of-distribution samples.

    Args:
        base_directory: Path to dataset directory
        directory_uncertain: Directory containing preprocessed uncertain data.
        batch_size: Batch size for data loaders
        shuffle: Whether to shuffle the data
        seed: Random seed for reproducibility
        split_train_val_ratio: Ratio for splitting train set into train/val
        return_metadata: Whether to return metadata with samples
        n_samples: Number of samples to use from dataset (None = use all)

    Returns:
        tuple: (train_loader, valid_loader, test_loader)
    """
    return get_h36m_motion_dataset_function(
        base_directory=base_directory,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        n_samples=n_samples,
        ood=True,
        input_uncertainty=True,
        directory_uncertain=directory_uncertain
    )


def get_h36m_motion_reduced_output_ood_dataset(
    base_directory,
    batch_size=128,
    shuffle=False,
    seed=0,
    split_train_val_ratio=0.9,
    n_samples=None
):
    """Get data loaders for the H36M motion prediction OOD dataset.

    This dataset shuffles the input sequences to create out-of-distribution samples.
    This dataset comes with reduced output size (only specific timestep and joints).

    Args:
        base_directory: Path to dataset directory
        batch_size: Batch size for data loaders
        shuffle: Whether to shuffle the data
        seed: Random seed for reproducibility
        split_train_val_ratio: Ratio for splitting train set into train/val
        return_metadata: Whether to return metadata with samples
        n_samples: Number of samples to use from dataset (None = use all)

    Returns:
        tuple: (train_loader, valid_loader, test_loader)
    """
    return get_h36m_motion_dataset_function(
        base_directory=base_directory,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        n_samples=n_samples,
        reduce_size=True,
        ood=True
    )
