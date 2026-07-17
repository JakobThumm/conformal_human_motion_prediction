#!/bin/bash
# Table 2/3: Robot-shield safety / certification simulation (ISO 13849-1 PFH_D / Performance Level)
# on H36M, for the three methods of the final results table. Each shield run targets the same number
# of simulated HRC test cycles (N ~= N_TEST_CYCLES); the number of random robot poses is derived per
# run so N stays fixed even as OOD/too-fast filtering changes the eligible human-sample count.
#
#   * ISO 13855 without OOD filtered : --human_set sara  --no-mask_ood  (constant-velocity, v=2 m/s)
#   * Ours without OOD filtered      : --human_set conformal --no-mask_ood
#   * Ours with OOD filtered         : --human_set conformal --mask_ood
#
# Steps: predict eval+test set -> calibrate conformal set -> three shield runs (one CSV row each) ->
# build the standalone shield LaTeX table.
#
# Prerequisites (NOT run here): a trained motion model at $MODEL and the motion OOD score function
# at $SCORE_FN (see README).
set -e

# export XLA_PYTHON_CLIENT_PREALLOCATE=false   # share the GPU politely (jax pre-alloc off)

MODEL="${MODEL:-models/motion_prediction/final_model/dct_pose_transformer.pickle}"
SCORE_FN="${SCORE_FN:-models/ood_functions/dct_pose_transformer_score_fn.cloudpickle}"
CALIB="models/motion_prediction/conformal_calibration/conformal_calibrator.npz"
CSV="results/final/robot_shield/shield_results.csv"
LIKELIHOOD="${LIKELIHOOD:-0.9999}"
N_TEST_CYCLES="${N_TEST_CYCLES:-2e13}"   # target simulated HRC test cycles per method
POSE_RADIUS="${POSE_RADIUS:-10.0}"

mkdir -p "$(dirname "$CSV")"
rm -f "$CSV"                          # fresh sweep -> one row per method below

# 2) Predict the eval and test set (RAW predictions + covariances + input uncertainty + OOD scores).
python -m conformal_human_motion_prediction.examples.motion_prediction \
  --data_path datasets/ \
  --dataset_name Human36mMotionDataset3DWithInputUncertainty \
  --split validation \
  --model_save_path "$MODEL" \
  --enable_ood \
  --motion_score_fn_path "$SCORE_FN" \
  --output_dir results/motion_prediction

python -m conformal_human_motion_prediction.examples.motion_prediction \
  --data_path datasets/ \
  --dataset_name Human36mMotionDataset3DWithInputUncertainty \
  --split test \
  --model_save_path "$MODEL" \
  --enable_ood \
  --motion_score_fn_path "$SCORE_FN" \
  --output_dir results/motion_prediction

RESULTS_EVAL="results/motion_prediction/motion_prediction_results_validation.cloudpickle"
RESULTS_TEST="results/motion_prediction/motion_prediction_results_test.cloudpickle"

# 3) Calibrate the conditional-conformal sets at the target coverage.
python -m conformal_human_motion_prediction.motion_prediction.conformal_calibration \
  --results_file "$RESULTS_EVAL" \
  --calib_frac 0.5 \
  --likelihood "$LIKELIHOOD"

# 4)+5) Three shield runs on the test set -> one CSV row per method.
#   common knobs: derive poses from N_TEST_CYCLES, GPU backend, decorrelated cycles.
COMMON="--results_file $RESULTS_TEST --conformal_calibrator $CALIB \
  --backend gpu --gpu_dtype float32 --gpu_a_chunk 512 \
  --n_test_cycles $N_TEST_CYCLES --pose_radius $POSE_RADIUS --pose_z_offset 0.2 \
  --robot_stride 25 --seed 0 --results_csv $CSV"

echo "==================== ISO 13855 without OOD filtered (SARA, v=2 m/s) ===================="
python -m conformal_human_motion_prediction.examples.simulate_robot_shield \
  $COMMON --human_set sara --no-mask_ood

echo "==================== Ours without OOD filtered ===================="
python -m conformal_human_motion_prediction.examples.simulate_robot_shield \
  $COMMON --human_set conformal --no-mask_ood

echo "==================== Ours with OOD filtered ===================="
python -m conformal_human_motion_prediction.examples.simulate_robot_shield \
  $COMMON --human_set conformal --mask_ood

# 6) Build the standalone robot-shield LaTeX table (one row per method).
python -m conformal_human_motion_prediction.generate_plots.generate_robot_shield_results \
  --csv "$CSV" \
  --output results/final/robot_shield/robot_shield_safety.tex \
  --confidence 0.9999
