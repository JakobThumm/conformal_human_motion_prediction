# Datasets

Dataset **artifacts** live here. The actual data is git-ignored — only this README is tracked.
This document explains how to obtain and (pre)generate each dataset.

## Expected layout

```
datasets/
├── H36M/
│   ├── extracted/            # raw Human3.6M frames + 3D annotations (you provide)
│   ├── pre_processed/        # 2D pose-net inputs (generated, see below)
│   └── pre_processed_motion/ # motion-prediction inputs with pose uncertainty (generated)
├── tiger-pose/
│   └── preprocessed/         # OOD dataset for pose scoring (generated)
└── rgbd_test/                # RGB-D capture sequences for the real-hardware pipeline
```

## 1. Human3.6M (H36M)

Human3.6M is **license-restricted**. Register and download it from the official site
(<http://vision.imar.ro/human3.6m/>) and place the extracted frames + annotations under
`datasets/H36M/extracted/`. We cannot redistribute it.

### Generate the 2D pose-estimation inputs

```bash
# GPU-accelerated (recommended)
python -m conformal_human_motion_prediction.pose_estimation.preprocess_h36m_bbox_gpu \
    --dataset_dir datasets/H36M/extracted \
    --output_dir  datasets/H36M/pre_processed \
    --batch_size 128 --device cuda

# Single-frame / CPU reference implementation
python -m conformal_human_motion_prediction.pose_estimation.preprocess_h36m_bbox \
    --input_dir  datasets/H36M/extracted \
    --output_dir datasets/H36M/pre_processed \
    --splits train --num_frames 1
```

### Generate the motion-prediction inputs (with pose uncertainty)

```bash
python -m conformal_human_motion_prediction.motion_prediction.preprocess_uncertainty_input_dataset \
    --data_path datasets/ \
    --output_dir datasets/H36M/pre_processed_motion \
    --split validation --batch_size 32 --device cuda \
    --camera_ids 55011271 60457274 \
    --run_name jax_resnet50_regressflow
```

## 2. tiger-pose (OOD set for pose scoring)

Used as the out-of-distribution set when computing OOD scores. Obtain the tiger-pose images
and generate the preprocessed tensors into `datasets/tiger-pose/preprocessed/` with the GPU
preprocessing utility:

```bash
python -m conformal_human_motion_prediction.pose_estimation.preprocess_tiger_pose_gpu \
    --output_dir datasets/tiger-pose/preprocessed --device cuda
```

## 3. rgbd_test (real-hardware sequences)

RGB-D sequences captured for the on-robot pipeline, placed under `datasets/rgbd_test/`.
These are produced by the ROS2 capture stack (see `ros2_packages/`).
