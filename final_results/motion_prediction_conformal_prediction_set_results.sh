#!/bin/bash
# Table: Conformal prediction set test results on H36M (Coverage % / Volume m^3).
# Motion prediction from PREDICTED 3D pose inputs with input uncertainty, no OOD detection.
set -e

python -m conformal_human_motion_prediction.examples.motion_prediction \
  --data_path datasets/ \
  --dataset_name Human36mMotionDataset3DWithInputUncertainty \
  --split test \
  --model_save_path models/motion_prediction/final_model/dct_pose_transformer.pickle \
  --output_dir results/final/motion_prediction/no_uncertainty_no_ood

# Build the LaTeX table (conformal prediction sets vs ISO 13855) from the saved CSVs.
python -m conformal_human_motion_prediction.generate_plots.generate_conformal_prediction_set_results
