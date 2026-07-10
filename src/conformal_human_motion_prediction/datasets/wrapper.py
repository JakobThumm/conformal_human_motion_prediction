import torch
import os

from conformal_human_motion_prediction.datasets.h36m_preprocessed import get_h36m_preprocessed
from conformal_human_motion_prediction.datasets.h36m_motion_prediction import (
    get_h36m_motion_dataset,
    get_h36m_motion_reduced_output_dataset,
    get_h36m_motion_ood_dataset,
    get_h36m_motion_reduced_output_ood_dataset,
    get_h36m_motion_dataset_with_uncertainty,
)
from conformal_human_motion_prediction.datasets.tiger_pose import get_tiger_pose_preprocessed
from conformal_human_motion_prediction.datasets.human_rgbd import get_human_rgbd, get_human_rgbd_sequence


def dataloader_from_string(
    dataset_name,
    n_samples=None,
    batch_size: int = 128,
    shuffle=True,
    seed: int = 0,
    download: bool = False,
    data_path: str = "../datasets",
    max_target_speed: float = 2.0,
):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    if dataset_name == "H36M":
        train_loader, valid_loader, test_loader = get_h36m_preprocessed(
            preprocessed_dir=os.path.join(data_path, "H36M", "pre_processed"),
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            n_samples=n_samples,
        )
    elif dataset_name == "Human36mMotionDataset3D":
        train_loader, valid_loader, test_loader = get_h36m_motion_dataset(
            base_directory=os.path.join(data_path, "H36M", "extracted"),
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            n_samples=n_samples,
            max_target_speed=max_target_speed,
        )
    elif dataset_name == "Human36mMotionDataset3DWithInputUncertainty":
        train_loader, valid_loader, test_loader = get_h36m_motion_dataset_with_uncertainty(
            base_directory=os.path.join(data_path, "H36M", "extracted"),
            directory_uncertain=os.path.join(data_path, "H36M", "pre_processed_motion"),
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            n_samples=n_samples,
            max_target_speed=max_target_speed,
        )
    elif dataset_name == "Human36mMotionDataset3DAugmented":
        train_loader, valid_loader, test_loader = get_h36m_motion_dataset(
            base_directory=os.path.join(data_path, "H36M", "extracted"),
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            n_samples=n_samples,
            augment=True,
            max_target_speed=max_target_speed,
        )
    elif dataset_name == "Human36mMotionDataset3DWithInputUncertaintyAugmented":
        train_loader, valid_loader, test_loader = get_h36m_motion_dataset_with_uncertainty(
            base_directory=os.path.join(data_path, "H36M", "extracted"),
            directory_uncertain=os.path.join(data_path, "H36M", "pre_processed_motion"),
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            n_samples=n_samples,
            augment=True,
            max_target_speed=max_target_speed,
        )
    elif dataset_name == "Human36mMotionReducedOutputDataset3D":
        train_loader, valid_loader, test_loader = get_h36m_motion_reduced_output_dataset(
            base_directory=os.path.join(data_path, "H36M", "extracted"),
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            n_samples=n_samples,
        )
    elif dataset_name == "Human36mMotionReducedOutputDataset3DAugmented":
        train_loader, valid_loader, test_loader = get_h36m_motion_reduced_output_dataset(
            base_directory=os.path.join(data_path, "H36M", "extracted"),
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            n_samples=n_samples,
            augment=True,
        )
    elif dataset_name == "Human36mMotionOODDataset3D":
        train_loader, valid_loader, test_loader = get_h36m_motion_ood_dataset(
            base_directory=os.path.join(data_path, "H36M", "extracted"),
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            n_samples=n_samples,
        )
    elif dataset_name == "Human36mMotionReducedOutputOODDataset3D":
        train_loader, valid_loader, test_loader = get_h36m_motion_reduced_output_ood_dataset(
            base_directory=os.path.join(data_path, "H36M", "extracted"),
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            n_samples=n_samples,
        )
    elif dataset_name == "tiger-pose":
        train_loader, valid_loader, test_loader = get_tiger_pose_preprocessed(
            preprocessed_dir=os.path.join(data_path, "tiger-pose", "preprocessed"),
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            n_samples=n_samples,
        )
    elif dataset_name == "HumanRGBD":
        train_loader, valid_loader, test_loader = get_human_rgbd(
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            data_path=os.path.join(data_path, "rgbd_test"),
        )
    elif dataset_name == "HumanRGBDSequence":
        train_loader, valid_loader, test_loader = get_human_rgbd_sequence(
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            data_path=os.path.join(data_path, "rgbd_test"),
        )
    else:
        raise ValueError(f"Dataset {dataset_name} is not implemented")

    return train_loader, valid_loader, test_loader
