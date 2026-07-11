#!/bin/bash
# Table: Motion prediction test results on H36M (MPJPE @ 80/160/320/400 ms).
# Motion prediction from ground-truth 3D pose inputs, no uncertainty, no OOD.
set -e

# Ours (stage 1): the pose-only stage-1 checkpoint.
python -m conformal_human_motion_prediction.examples.motion_prediction \
  --data_path datasets/ \
  --dataset_name Human36mMotionDataset3D \
  --split test \
  --max_target_speed 0 \
  --model_save_path models/motion_prediction/final_training_run/checkpoints/stage_1/dct_pose_transformer.pickle \
  --output_dir results/final/motion_prediction/stage_1_no_uncertainty_no_ood

# Ours (final): the final (uncertainty-aware) model evaluated on ground-truth inputs.
python -m conformal_human_motion_prediction.examples.motion_prediction \
  --data_path datasets/ \
  --dataset_name Human36mMotionDataset3D \
  --split test \
  --max_target_speed 0 \
  --model_save_path models/motion_prediction/final_model/dct_pose_transformer.pickle \
  --output_dir results/final/motion_prediction/no_uncertainty_no_ood

# Build the LaTeX table from the saved CSVs.
python -m conformal_human_motion_prediction.generate_plots.generate_motion_prediction_no_uncertainty_no_ood
