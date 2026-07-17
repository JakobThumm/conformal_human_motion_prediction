#!/bin/bash
# Table 1/3: Conformal prediction set test results on H36M (Coverage % / Volume m^3), for the three
# methods of the final results table:
#   * ISO 13855 without OOD filtered  -> SARA constant-velocity reachable set (coverage_stats_sara.csv)
#   * Ours without OOD filtered       -> conditional-conformal set, all test samples
#   * Ours with OOD filtered          -> conditional-conformal set, in-distribution samples only
#
# All three come from a SINGLE evaluation run of examples.motion_prediction on the test set (with
# --enable_ood so the per-sample OOD score is measured); the evaluation part derives all three
# coverage/volume rows from that one run. The conditional-conformal calibrator is a prerequisite:
# it is (re)fitted here from a validation run unless it already exists.
set -e

# export XLA_PYTHON_CLIENT_PREALLOCATE=false   # share the GPU politely (jax pre-alloc off)

MODEL="${MODEL:-models/motion_prediction/final_model/dct_pose_transformer.pickle}"
SCORE_FN="${SCORE_FN:-models/ood_functions/dct_pose_transformer_score_fn.cloudpickle}"
CALIB="models/motion_prediction/conformal_calibration/conformal_calibrator.npz"
OUTPUT_DIR="results/final/conformal_prediction_sets"
LIKELIHOOD="${LIKELIHOOD:-0.9999}"

mkdir -p "$OUTPUT_DIR"

# 1) Calibrate the conditional-conformal set (validation split) unless the calibrator already exists.
if [ ! -f "$CALIB" ]; then
  echo "==================== fitting conformal calibrator ===================="
  python -m conformal_human_motion_prediction.examples.motion_prediction \
    --data_path datasets/ \
    --dataset_name Human36mMotionDataset3DWithInputUncertainty \
    --split validation \
    --model_save_path "$MODEL" \
    --output_dir "$OUTPUT_DIR"
  python -m conformal_human_motion_prediction.motion_prediction.conformal_calibration \
    --results_file "$OUTPUT_DIR/motion_prediction_results_validation.cloudpickle" \
    --calib_frac 0.5 \
    --likelihood "$LIKELIHOOD"
fi

# 2) The single evaluation run: predict the test set with OOD measurement. The eval part writes the
#    three per-method coverage CSVs into $OUTPUT_DIR:
#      coverage_stats_sara.csv
#      coverage_stats_conformal_prediction_sets.csv
#      coverage_stats_conformal_prediction_sets_ood_filtered.csv
python -m conformal_human_motion_prediction.examples.motion_prediction \
  --data_path datasets/ \
  --dataset_name Human36mMotionDataset3DWithInputUncertainty \
  --split test \
  --model_save_path "$MODEL" \
  --enable_ood \
  --motion_score_fn_path "$SCORE_FN" \
  --output_dir "$OUTPUT_DIR"

# 3) Build the standalone conformal prediction-set LaTeX table.
python -m conformal_human_motion_prediction.generate_plots.generate_conformal_prediction_set_results \
  --results_dir "$OUTPUT_DIR"
